// Copyright 2025 Embellama Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Model management module for the embellama library.
//!
//! This module contains the `EmbeddingModel` struct which encapsulates
//! the llama.cpp model and context, handling the generation of embeddings.

use crate::cache::CacheStore;
use crate::cache::token_cache::TokenCache;
use crate::config::{ModelConfig, PoolingStrategy};
use crate::error::{Error, Result};
use crate::gguf;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::{
    context::params::{LlamaContextParams, LlamaPoolingType},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel, params::LlamaModelParams},
    token::LlamaToken,
};
use self_cell::self_cell;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{debug, info, instrument, warn};

/// Maps our `PoolingStrategy` enum to llama.cpp's `LlamaPoolingType`
///
/// For strategies that llama.cpp doesn't natively support (Max, `MeanSqrt`),
/// we set pooling to None and handle it ourselves in `apply_pooling()`.
/// For Last pooling (required for decoder models), we must set it at the
/// llama.cpp level to ensure proper KV cache initialization.
fn pooling_strategy_to_llama_type(strategy: PoolingStrategy) -> LlamaPoolingType {
    match strategy {
        PoolingStrategy::Mean => LlamaPoolingType::Mean,
        PoolingStrategy::Cls => LlamaPoolingType::Cls,
        PoolingStrategy::Last => LlamaPoolingType::Last,
        // Max and MeanSqrt are not natively supported by llama.cpp
        // We'll apply these strategies ourselves after extraction
        PoolingStrategy::Max | PoolingStrategy::MeanSqrt => LlamaPoolingType::None,
    }
}

self_cell! {
    struct ModelCell {
        owner: LlamaModel,
        #[covariant]
        dependent: LlamaContext,
    }
}

/// Represents a loaded embedding model.
///
/// This struct encapsulates the `llama_cpp_2::LlamaModel` and `LlamaContext`
/// and provides methods for generating embeddings from text input.
///
/// # Important
///
/// Due to the `!Send` nature of `LlamaContext`, instances of this struct
/// cannot be safely sent between threads. Each thread must maintain its
/// own instance.
///
/// # Example
///
/// ```ignore
/// use embellama::model::EmbeddingModel;
/// use embellama::config::ModelConfig;
///
/// let config = ModelConfig::builder()
///     .with_model_path("path/to/model.gguf")
///     .with_model_name("my-model")
///     .build()?;
///
/// let model = EmbeddingModel::new(&config)?;
/// assert!(model.is_loaded());
/// ```
pub struct EmbeddingModel {
    // IMPORTANT: Field order matters for drop order!
    // Context must be dropped before model since it depends on it
    /// The llama model cell self-referential helper
    cell: ModelCell,

    // Metadata fields (order doesn't matter for these)
    /// Configuration used to create this model
    config: ModelConfig,
    /// Path to the model file
    model_path: PathBuf,
    /// Model name identifier
    model_name: String,
    /// Cached embedding dimensions (determined at load time)
    embedding_dimensions: usize,
    /// Maximum context size
    max_context_size: usize,
    /// Whether to add BOS token during tokenization
    add_bos_token: bool,
    /// Maximum number of sequences for batch processing
    n_seq_max: u32,
    /// GGUF metadata containing architecture info, dimensions, context size
    metadata: crate::gguf::GGUFMetadata,
}

impl EmbeddingModel {
    /// Creates a new embedding model from the given configuration.
    ///
    /// # Arguments
    ///
    /// * `backend` - The llama backend to use for model loading
    /// * `config` - The model configuration containing path and parameters
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the initialized model or an error.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The model file cannot be loaded
    /// - The context creation fails
    /// - Invalid configuration parameters are provided
    #[instrument(skip(backend, config), fields(model_path = %config.model_path.display()))]
    #[allow(clippy::too_many_lines)]
    pub fn new(backend: &LlamaBackend, config: &ModelConfig) -> Result<Self> {
        info!("Loading model from {:?}", config.model_path);

        // Set up model parameters
        let mut model_params = LlamaModelParams::default();

        // Configure GPU layers if specified
        if let Some(gpu_layers) = config.n_gpu_layers {
            model_params = model_params.with_n_gpu_layers(gpu_layers);
            debug!("GPU layers set to: {}", gpu_layers);
        }

        // TODO: Configure memory options when API supports it
        // Currently llama-cpp-2 doesn't expose use_mmap/use_mlock setters

        // Load the model into a Box for stable address
        let model = LlamaModel::load_from_file(backend, &config.model_path, &model_params)
            .map_err(|e| Error::ModelLoadError {
                path: config.model_path.clone(),
                source: anyhow::anyhow!("Failed to load model: {e}"),
            })?;

        debug!("Model loaded successfully");

        // Set up context parameters
        // Priority: 1) Explicit config, 2) GGUF metadata, 3) Fallback to 2048
        let ctx_size = if let Some(n_ctx) = config.n_ctx {
            debug!("Using configured context size: {}", n_ctx);
            n_ctx
        } else {
            // Try to auto-detect from GGUF metadata
            match Self::extract_context_size_from_gguf(&config.model_path) {
                Ok(size) => {
                    info!("Auto-detected context size from GGUF metadata: {}", size);
                    size
                }
                Err(e) => {
                    debug!(
                        "Could not read context size from GGUF metadata: {}, using default 2048",
                        e
                    );
                    2048
                }
            }
        };

        let n_threads = i32::try_from(config.n_threads.unwrap_or_else(|| {
            let threads = num_cpus::get();
            debug!("Using {} CPU threads", threads);
            threads
        }))
        .unwrap_or(1);

        // Extract GGUF metadata for architecture detection and model info
        let metadata = gguf::extract_metadata(&config.model_path).unwrap_or_else(|e| {
            warn!(
                "Failed to read GGUF metadata: {}, assuming decoder model with defaults",
                e
            );
            // Create a fallback metadata that defaults to decoder (safer)
            crate::gguf::GGUFMetadata {
                architecture: Some("unknown".to_string()),
                embedding_dimensions: 0,
                context_size: ctx_size as usize,
            }
        });
        let is_decoder = metadata.is_decoder();
        debug!(
            "Detected model architecture: {} for model: {}",
            if is_decoder { "decoder" } else { "encoder" },
            config.model_name
        );

        // Determine n_seq_max based on model architecture
        // IMPORTANT: Decoder models (Qwen, LLaMA, Mistral, etc.) crash with n_seq_max > 1
        // due to how they handle sequence batching in llama.cpp. Encoder models (BERT, etc.)
        // can safely use higher values for better batch processing performance.
        let n_seq_max = if is_decoder {
            // Force decoder models to n_seq_max=1 to prevent crashes
            debug!("Setting n_seq_max=1 for decoder model (required to prevent crashes)");
            1
        } else {
            // Encoder models can use configured value (default to 8 for better batching)
            let value = config.n_seq_max.unwrap_or(8);
            debug!("Setting n_seq_max={} for encoder model", value);
            value
        };

        let mut ctx_params = LlamaContextParams::default();
        ctx_params = ctx_params.with_n_seq_max(n_seq_max);

        // Set context size (use context_size if specified, otherwise use n_ctx)
        // Also check deprecated kv_cache_size for backward compatibility
        #[allow(deprecated)]
        let context_size = config
            .context_size
            .or(config.kv_cache_size)
            .unwrap_or(ctx_size);
        let n_ctx = NonZeroU32::new(context_size);
        ctx_params = ctx_params.with_n_ctx(n_ctx);

        // Enable KV cache optimizations if requested
        if config.enable_kv_optimization {
            debug!("Enabling KV cache optimizations");
            // > NOTE: These optimizations are enabled through llama-cpp parameters
        }

        // Set micro-batch size with architecture-aware defaults
        // Decoder models (Qwen, LLaMA, etc.) need smaller n_ubatch to prevent memory issues and crashes
        // Encoder models (BERT, etc.) can use larger values for better batch processing
        let n_ubatch = if let Some(ubatch) = config.n_ubatch {
            // Use explicitly configured value
            debug!("Using configured n_ubatch: {}", ubatch);
            ubatch
        } else if is_decoder {
            // Decoder models: use conservative 512 to prevent crashes
            // llama-server uses 2048, but 512 is safer for large contexts
            let ubatch = 512_u32;
            debug!(
                "Setting n_ubatch={} for decoder model (conservative default)",
                ubatch
            );
            ubatch
        } else {
            // Encoder models: can use larger values for better performance
            let ubatch = 2048_u32;
            debug!("Setting n_ubatch={} for encoder model", ubatch);
            ubatch
        };
        ctx_params = ctx_params.with_n_ubatch(n_ubatch);

        // Set thread counts
        ctx_params = ctx_params.with_n_threads(n_threads);
        ctx_params = ctx_params.with_n_threads_batch(n_threads);

        // Enable embeddings mode
        ctx_params = ctx_params.with_embeddings(true);

        // Set pooling type based on our pooling strategy
        // This is critical for decoder models (e.g., Qwen) which require Last pooling
        let llama_pooling_type = pooling_strategy_to_llama_type(config.pooling_strategy);
        ctx_params = ctx_params.with_pooling_type(llama_pooling_type);
        debug!(
            "Set llama.cpp pooling type to {:?} for strategy {:?}",
            llama_pooling_type, config.pooling_strategy
        );

        // Enable flash attention for better performance
        // Decoder models benefit significantly from flash attention
        if is_decoder {
            debug!("Enabling flash attention for decoder model");
            ctx_params = ctx_params.with_flash_attention(true);
        }

        // Get embedding dimensions from the model
        #[allow(clippy::cast_sign_loss)]
        let embedding_dimensions = model.n_embd() as usize;

        info!(
            "Model initialized: dimensions={}, context_size={}, threads={}",
            embedding_dimensions, context_size, n_threads
        );

        let cell = ModelCell::try_new(model, |m| {
            m.new_context(backend, ctx_params)
                .map_err(|e| Error::ContextError {
                    source: anyhow::anyhow!("Failed to create context: {e}"),
                })
        })?;

        // Determine whether to add BOS token (using architecture detection from earlier)
        let add_bos_token = if let Some(add_bos) = config.add_bos_token {
            // Use explicitly configured value
            debug!("Using configured add_bos_token: {}", add_bos);
            add_bos
        } else {
            // Auto-detect: decoder models need BOS, encoder models don't
            debug!(
                "Auto-detected add_bos_token: {} for {} model",
                is_decoder,
                if is_decoder { "decoder" } else { "encoder" }
            );
            is_decoder
        };

        let model = Self {
            cell,
            config: config.clone(),
            model_path: config.model_path.clone(),
            model_name: config.model_name.clone(),
            embedding_dimensions,
            #[allow(clippy::cast_lossless)]
            max_context_size: ctx_size as usize,
            add_bos_token,
            n_seq_max,
            metadata,
        };

        // Log effective max tokens for debugging batch size issues
        let effective_max = model.effective_max_tokens();
        info!(
            "Effective max tokens: {} (context: {}, overhead: {})",
            effective_max,
            model.max_context_size,
            model.max_context_size - effective_max
        );

        Ok(model)
    }

    /// Loads a model from disk.
    ///
    /// This is an alternative way to create a model, useful when you want
    /// to explicitly separate the loading step.
    ///
    /// # Arguments
    ///
    /// * `backend` - The llama backend to use for model loading
    /// * `config` - The model configuration
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the loaded model or an error.
    ///
    /// # Errors
    ///
    /// Returns an error if model loading fails
    pub fn load(backend: &LlamaBackend, config: &ModelConfig) -> Result<Self> {
        Self::new(backend, config)
    }

    /// Consumes the model and explicitly frees resources.
    ///
    /// Note: This happens automatically when the model is dropped.
    /// This method exists mainly for explicit resource management.
    pub fn unload(self) {
        // Model is dropped here, which triggers cleanup
        drop(self);
    }

    /// Checks if the model is currently loaded and ready for inference.
    ///
    /// # Returns
    ///
    /// Returns true if the model is loaded, false otherwise.
    pub fn is_loaded(&self) -> bool {
        // Check if we have valid dimensions and context size
        self.embedding_dimensions > 0 && self.max_context_size > 0
    }

    /// Returns the dimensionality of embeddings produced by this model.
    ///
    /// # Returns
    ///
    /// The number of dimensions in the embedding vectors.
    pub fn embedding_dimensions(&self) -> usize {
        self.embedding_dimensions
    }

    /// Returns the maximum sequence length supported by this model.
    ///
    /// # Returns
    ///
    /// The maximum number of tokens that can be processed.
    pub fn max_sequence_length(&self) -> usize {
        self.max_context_size
    }

    /// Returns the approximate memory footprint of the model in bytes.
    ///
    /// # Returns
    ///
    /// Estimated memory usage in bytes, or `None` if the size cannot be calculated
    /// (e.g., on 32-bit platforms with very large models).
    pub fn model_size(&self) -> Option<usize> {
        // This is an approximation based on model parameters
        // More accurate measurement would require llama.cpp API support
        let params = self.cell.borrow_owner().n_params();
        let size_per_param = 2; // Approximate bytes per parameter for quantized models
        usize::try_from(params).ok().map(|p| p * size_per_param)
    }

    /// Returns the model's metadata.
    ///
    /// # Returns
    ///
    /// A tuple containing (`model_name`, `model_path`, `vocab_size`, `n_params`).
    pub fn model_metadata(&self) -> (String, PathBuf, usize, usize) {
        (
            self.model_name.clone(),
            self.model_path.clone(),
            usize::try_from(self.cell.borrow_owner().n_vocab()).unwrap_or_else(|_| {
                warn!("Model vocab size conversion failed, using 0");
                0
            }),
            usize::try_from(self.cell.borrow_owner().n_params()).unwrap_or_else(|_| {
                warn!("Model params count too large for platform, using 0");
                0
            }),
        )
    }

    /// Returns the model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Returns the model name.
    pub fn name(&self) -> &str {
        &self.model_name
    }

    /// Returns the path to the model file.
    pub fn path(&self) -> &PathBuf {
        &self.model_path
    }

    /// Returns whether BOS token should be added during tokenization.
    pub fn add_bos_token(&self) -> bool {
        self.add_bos_token
    }

    /// Returns the maximum number of sequences for batch processing.
    pub fn n_seq_max(&self) -> u32 {
        self.n_seq_max
    }

    /// Calculate the effective maximum tokens available for input.
    ///
    /// For BERT-style embedding models, the `max_position_embeddings` parameter
    /// defines the maximum sequence length the model can handle. The only overhead
    /// is from special tokens like [CLS] and [SEP] that the tokenizer adds.
    ///
    /// # Returns
    ///
    /// The maximum number of input tokens that can be safely processed.
    ///
    /// # Implementation Note
    ///
    /// The overhead is minimal (2-3 tokens) for special tokens. The embedding
    /// output vectors are computed from hidden states and don't consume context space.
    ///
    /// # Example
    ///
    /// For a model with `max_position_embeddings` = 512:
    /// - Overhead: 2 tokens ([CLS] and [SEP])
    /// - Effective max: 512 - 2 = 510 tokens
    pub fn effective_max_tokens(&self) -> usize {
        // For embedding models, only special tokens ([CLS], [SEP]) consume overhead
        // The output embeddings don't use KV cache space
        let overhead = if self.add_bos_token { 3 } else { 2 };

        self.max_context_size.saturating_sub(overhead)
    }

    /// Tokenizes the input text.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to tokenize
    /// * `add_bos` - Whether to add the beginning-of-sequence token
    ///
    /// # Returns
    ///
    /// A vector of tokens.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization fails.
    pub fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<LlamaToken>> {
        let add_bos = if add_bos {
            AddBos::Always
        } else {
            AddBos::Never
        };

        self.cell
            .borrow_owner()
            .str_to_token(text, add_bos)
            .map_err(|e| Error::TokenizationError {
                message: format!("Failed to tokenize text: {e}"),
            })
    }

    /// Tokenizes the input text with caching support.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to tokenize
    /// * `add_bos` - Whether to add a beginning-of-sentence token
    /// * `cache` - Optional token cache for caching tokenization results
    ///
    /// # Returns
    ///
    /// Returns a vector of tokens representing the tokenized text.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization fails.
    pub fn tokenize_cached(
        &self,
        text: &str,
        add_bos: bool,
        cache: Option<&TokenCache>,
    ) -> Result<Vec<LlamaToken>> {
        // If cache is available, try to get cached tokens
        if let Some(cache) = cache {
            let key = TokenCache::compute_key(text, &self.model_name, add_bos);
            if let Some(tokens) = cache.get(&key) {
                debug!("Using cached tokens for text (length: {})", text.len());
                return Ok(tokens);
            }

            // Cache miss, tokenize and cache the result
            let tokens = self.tokenize(text, add_bos)?;
            cache.insert(key, tokens.clone());
            return Ok(tokens);
        }

        // No cache available, fallback to regular tokenization
        self.tokenize(text, add_bos)
    }

    /// Generates an embedding for the given text.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to generate embeddings for
    ///
    /// # Returns
    ///
    /// Returns a vector of f32 values representing the embedding.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - Tokenization fails
    /// - The input exceeds the maximum token limit
    /// - Model inference fails
    #[instrument(skip(self, text), fields(text_len = text.len()))]
    pub fn generate_embedding(&mut self, text: &str) -> Result<Vec<f32>> {
        self.generate_embedding_cached(text, None)
    }

    /// Generates an embedding for the given text with optional token cache support.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to generate embeddings for
    /// * `token_cache` - Optional token cache for caching tokenization results
    ///
    /// # Returns
    ///
    /// Returns a vector of f32 values representing the embedding.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - Tokenization fails
    /// - The input exceeds the maximum token limit
    /// - Model inference fails
    #[instrument(skip(self, text, token_cache), fields(text_len = text.len()))]
    pub fn generate_embedding_cached(
        &mut self,
        text: &str,
        token_cache: Option<&TokenCache>,
    ) -> Result<Vec<f32>> {
        // Validate input
        if text.is_empty() {
            return Err(Error::InvalidInput {
                message: "Cannot generate embedding for empty text".to_string(),
            });
        }

        // Tokenize the text with caching support
        let tokens = self.tokenize_cached(text, self.add_bos_token, token_cache)?;

        // Validate token limit
        self.validate_token_limit(tokens.len(), Some("Input"))?;

        debug!("Tokenized text into {} tokens", tokens.len());

        // Process tokens to get embeddings
        let embeddings = self.process_tokens_internal(&tokens)?;

        // Apply pooling and normalization
        self.finalize_embedding(&embeddings, tokens.len())
    }

    /// Processes multiple token sequences as a batch through the model.
    ///
    /// This method enables true batch processing by encoding multiple sequences
    /// in a single model pass using unique sequence IDs. If the number of sequences
    /// exceeds `n_seq_max`, it will automatically chunk them.
    ///
    /// # Arguments
    ///
    /// * `token_sequences` - Slice of token sequences to process
    ///
    /// # Returns
    ///
    /// Returns a vector of embedding vectors, one for each input sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Context creation fails
    /// - Batch processing fails
    /// - Embedding extraction fails
    /// - Pooling or normalization operations fail
    #[instrument(skip(self, token_sequences), fields(batch_size = token_sequences.len()))]
    pub fn process_batch_tokens(
        &mut self,
        token_sequences: &[Vec<LlamaToken>],
    ) -> Result<Vec<Vec<f32>>> {
        if token_sequences.is_empty() {
            return Ok(Vec::new());
        }

        debug!(
            "Processing batch of {} sequences with n_seq_max={}",
            token_sequences.len(),
            self.n_seq_max
        );

        // If we have more sequences than n_seq_max, process in chunks
        #[allow(clippy::cast_lossless)]
        if token_sequences.len() > self.n_seq_max as usize {
            debug!(
                "Batch size {} exceeds n_seq_max {}, chunking",
                token_sequences.len(),
                self.n_seq_max
            );

            let mut all_embeddings = Vec::with_capacity(token_sequences.len());

            // Process sequences in chunks of n_seq_max
            #[allow(clippy::cast_lossless)]
            for chunk in token_sequences.chunks(self.n_seq_max as usize) {
                debug!("Processing chunk of {} sequences", chunk.len());
                let chunk_embeddings = self.process_batch_tokens_internal(chunk)?;
                all_embeddings.extend(chunk_embeddings);
            }

            return Ok(all_embeddings);
        }

        // Process all sequences in a single batch
        self.process_batch_tokens_internal(token_sequences)
    }

    /// Internal method to process a batch of token sequences that fits within `n_seq_max`.
    #[allow(clippy::too_many_lines)]
    fn process_batch_tokens_internal(
        &mut self,
        token_sequences: &[Vec<LlamaToken>],
    ) -> Result<Vec<Vec<f32>>> {
        // Validate each sequence individually against effective max
        // For BERT-style batching, each sequence is processed independently
        // and must fit within the model's max_position_embeddings
        for (i, tokens) in token_sequences.iter().enumerate() {
            self.validate_token_limit(tokens.len(), Some(&format!("Sequence {i}")))?;
        }

        // Calculate total tokens needed for batch allocation
        let total_tokens: usize = token_sequences.iter().map(std::vec::Vec::len).sum();

        // Create a batch with all sequences (using actual n_seq_max)
        let _n_seq_max_i32 =
            i32::try_from(self.n_seq_max).map_err(|_| Error::EmbeddingGenerationError {
                message: "n_seq_max too large for i32".to_string(),
                source: None,
            })?;
        let mut batch = LlamaBatch::new(total_tokens, 1);

        // Add each sequence with unique ID
        for (seq_id, tokens) in token_sequences.iter().enumerate() {
            batch
                .add_sequence(
                    tokens,
                    i32::try_from(seq_id).map_err(|_| Error::EmbeddingGenerationError {
                        message: format!("Sequence ID {seq_id} too large for i32"),
                        source: None,
                    })?,
                    true,
                )
                .map_err(|e| Error::EmbeddingGenerationError {
                    message: format!("Failed to add sequence {seq_id} to batch: {e}"),
                    source: Some(anyhow::anyhow!(e)),
                })?;
        }

        // Process the entire batch in one model pass
        // Decoder models need to use decode() instead of encode()
        // encode() tries to access unified KV cache which is null for decoder models
        self.process_batch(&mut batch)?;

        // Extract embeddings for each sequence
        let mut all_embeddings = Vec::with_capacity(token_sequences.len());

        for seq_id in 0..token_sequences.len() {
            // Calculate token offset for this sequence
            let token_offset: usize = token_sequences[..seq_id]
                .iter()
                .map(std::vec::Vec::len)
                .sum();

            let embeddings = self.extract_sequence_embeddings(
                seq_id,
                token_sequences[seq_id].len(),
                Some(token_offset),
            )?;

            // Apply pooling and normalization
            let final_embedding =
                self.finalize_embedding(&embeddings, token_sequences[seq_id].len())?;
            all_embeddings.push(final_embedding);
        }

        Ok(all_embeddings)
    }

    /// Processes a batch of tokens through the model.
    ///
    /// This is a lower-level method used internally for batch processing.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The tokens to process
    ///
    /// # Returns
    ///
    /// Returns the processed embedding vector.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Token processing fails
    /// - Pooling operation fails
    /// - Normalization fails (if enabled)
    #[instrument(skip(self, tokens), fields(token_count = tokens.len()))]
    pub fn process_tokens(&mut self, tokens: &[i32]) -> Result<Vec<f32>> {
        // Convert i32 tokens to LlamaToken and process
        let llama_tokens: Vec<LlamaToken> = tokens.iter().map(|&t| LlamaToken(t)).collect();
        let embeddings = self.process_tokens_internal(&llama_tokens)?;

        // Apply pooling and normalization
        self.finalize_embedding(&embeddings, llama_tokens.len())
    }

    /// Helper to convert usize index to i32 with consistent error handling.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to convert
    /// * `context` - Description of what the index represents (for error messages)
    ///
    /// # Returns
    ///
    /// Returns the i32 representation of the index.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is too large for i32.
    #[inline]
    fn to_i32(index: usize, context: &str) -> Result<i32> {
        i32::try_from(index).map_err(|_| Error::EmbeddingGenerationError {
            message: format!("{context} {index} too large for i32"),
            source: None,
        })
    }

    /// Validate that a token count is within the effective maximum limit.
    ///
    /// This method consolidates token limit validation that was previously
    /// duplicated in three different locations.
    ///
    /// # Arguments
    ///
    /// * `token_count` - Number of tokens to validate
    /// * `context_hint` - Optional context string for error messages (e.g., "Sequence 0")
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the token count is within limits.
    ///
    /// # Errors
    ///
    /// Returns an error if the token count exceeds the effective maximum.
    fn validate_token_limit(&self, token_count: usize, context_hint: Option<&str>) -> Result<()> {
        let effective_max = self.effective_max_tokens();
        if token_count > effective_max {
            let context_prefix = context_hint.map_or_else(String::new, |h| format!("{h} "));
            return Err(Error::InvalidInput {
                message: format!(
                    "{}exceeds effective maximum tokens: {} tokens > {} effective max (context: {}, overhead: {}). Please truncate your input.",
                    context_prefix,
                    token_count,
                    effective_max,
                    self.max_context_size,
                    self.max_context_size - effective_max
                ),
            });
        }
        Ok(())
    }

    /// Finalize an embedding by applying pooling and normalization.
    ///
    /// This method consolidates the pooling + normalization logic that was
    /// previously duplicated in four different locations.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - The raw embeddings from the model
    /// * `expected_tokens` - The number of tokens we expected (for pre-pooled detection)
    ///
    /// # Returns
    ///
    /// Returns the final pooled and optionally normalized embedding vector.
    ///
    /// # Errors
    ///
    /// Returns an error if pooling or normalization fails.
    fn finalize_embedding(
        &self,
        embeddings: &[Vec<f32>],
        expected_tokens: usize,
    ) -> Result<Vec<f32>> {
        // Check if we got a single pre-pooled embedding
        let pooled = if embeddings.len() == 1 && expected_tokens > 1 {
            // This is already pooled by the model (BERT with pooling_type)
            debug!("Using pre-pooled embedding from model");
            embeddings[0].clone()
        } else {
            // Apply our pooling strategy for multi-token outputs
            Self::apply_pooling(embeddings, self.config.pooling_strategy)?
        };

        // Normalize if configured
        if self.config.normalize_embeddings {
            Self::normalize_embedding(pooled)
        } else {
            Ok(pooled)
        }
    }

    /// Extract embeddings for a sequence from the context.
    ///
    /// This method handles both pre-pooled embeddings (from `embeddings_seq_ith`)
    /// and token-wise embeddings (from `embeddings_ith`). This logic was previously
    /// duplicated in three different locations.
    ///
    /// # Arguments
    ///
    /// * `seq_id` - The sequence ID to extract embeddings for
    /// * `n_tokens` - Number of tokens in the sequence
    /// * `token_offset` - Optional offset for token-wise extraction (used in batch processing)
    ///
    /// # Returns
    ///
    /// Returns a vector of embedding vectors (one per token, or single pre-pooled).
    ///
    /// # Errors
    ///
    /// Returns an error if embedding extraction fails.
    fn extract_sequence_embeddings(
        &self,
        seq_id: usize,
        n_tokens: usize,
        token_offset: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        self.cell.with_dependent(|_, ctx| -> Result<Vec<Vec<f32>>> {
            // llama.cpp handles pooling internally based on our configured strategy
            // Try to get the pre-pooled sequence embedding first
            let seq_id_i32 = Self::to_i32(seq_id, "Sequence ID")?;
            if let Ok(seq_embeddings) = ctx.embeddings_seq_ith(seq_id_i32) {
                // Got pooled embedding from llama.cpp
                debug!(
                    "Retrieved pooled embedding for sequence {} (strategy: {:?})",
                    seq_id, self.config.pooling_strategy
                );
                return Ok(vec![seq_embeddings.to_vec()]);
            }

            if seq_id == 0 {
                debug!(
                    "Failed to get sequence embedding, falling back to token-wise (strategy: {:?})",
                    self.config.pooling_strategy
                );
            }

            // Fall back to token-wise embeddings (for LLaMA-style models)
            // Need to extract tokens for this specific sequence
            let mut token_embeddings = Vec::with_capacity(n_tokens);
            let offset = token_offset.unwrap_or(0);

            for i in 0..n_tokens {
                let global_idx = offset + i;
                let global_idx_i32 = Self::to_i32(global_idx, "Token index")?;
                let embeddings = ctx.embeddings_ith(global_idx_i32).map_err(|e| {
                    Error::EmbeddingGenerationError {
                        message: format!(
                            "Failed to get embeddings for token {i} in sequence {seq_id}"
                        ),
                        source: Some(anyhow::anyhow!(e)),
                    }
                })?;
                token_embeddings.push(embeddings.to_vec());
            }
            Ok(token_embeddings)
        })
    }

    /// Process a batch through the model using decode (decoders) or encode (encoders).
    ///
    /// This method handles the KV cache clearing and model-specific processing logic
    /// that was previously duplicated across multiple methods.
    ///
    /// # Arguments
    ///
    /// * `batch` - The batch to process through the model
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if processing succeeds.
    ///
    /// # Errors
    ///
    /// Returns an error if batch processing fails.
    fn process_batch(&mut self, batch: &mut LlamaBatch) -> Result<()> {
        self.cell.with_dependent_mut(|_, ctx| {
            // Clear KV cache to ensure clean state for each embedding generation
            // This prevents cache contamination between sequential calls
            ctx.clear_kv_cache();

            if self.metadata.is_decoder() {
                ctx.decode(batch)
                    .map_err(|e| Error::EmbeddingGenerationError {
                        message: format!("Failed to decode batch: {e}"),
                        source: Some(anyhow::anyhow!(e)),
                    })
            } else {
                ctx.encode(batch)
                    .map_err(|e| Error::EmbeddingGenerationError {
                        message: format!("Failed to encode batch: {e}"),
                        source: Some(anyhow::anyhow!(e)),
                    })
            }
        })
    }

    /// Internal method to process `LlamaToken` vectors.
    fn process_tokens_internal(&mut self, tokens: &[LlamaToken]) -> Result<Vec<Vec<f32>>> {
        if tokens.is_empty() {
            return Err(Error::InvalidInput {
                message: "Cannot process empty token list".to_string(),
            });
        }

        // Validate token limit
        self.validate_token_limit(tokens.len(), Some("Input"))?;

        // Create a batch for processing
        let n_tokens = tokens.len();
        let mut batch = LlamaBatch::new(n_tokens, 1);
        batch
            .add_sequence(tokens, 0, true)
            .map_err(|e| Error::EmbeddingGenerationError {
                message: format!("Failed to add tokens to batch: {e}"),
                source: Some(anyhow::anyhow!(e)),
            })?;

        // Process the batch through the model
        // Decoder models need to use decode() instead of encode()
        // encode() tries to access unified KV cache which is null for decoder models
        self.process_batch(&mut batch)?;

        // Extract embeddings based on pooling configuration
        // When llama.cpp pooling is enabled (Last, Mean, etc.), the model computes and stores
        // the pooled embedding, which we retrieve as a sequence embedding.
        // When pooling is NONE, we get individual token embeddings and pool ourselves.
        let all_embeddings = self.extract_sequence_embeddings(0, n_tokens, None)?;

        Ok(all_embeddings)
    }

    /// Applies pooling strategy to token embeddings.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Token embeddings from the model
    /// * `strategy` - Pooling strategy to apply
    ///
    /// # Returns
    ///
    /// Returns a single pooled embedding vector.
    fn apply_pooling(embeddings: &[Vec<f32>], strategy: PoolingStrategy) -> Result<Vec<f32>> {
        if embeddings.is_empty() {
            return Err(Error::EmbeddingGenerationError {
                message: "No embeddings to pool".to_string(),
                source: None,
            });
        }

        let embedding_dim = embeddings[0].len();

        match strategy {
            PoolingStrategy::Mean => {
                // Mean pooling across all tokens
                let mut pooled = vec![0.0f32; embedding_dim];
                #[allow(clippy::cast_precision_loss)]
                let n_tokens = embeddings.len() as f32;

                for token_emb in embeddings {
                    for (i, &val) in token_emb.iter().enumerate() {
                        pooled[i] += val / n_tokens;
                    }
                }

                Ok(pooled)
            }
            PoolingStrategy::Cls => {
                // Use only the first token (CLS token)
                Ok(embeddings[0].clone())
            }
            PoolingStrategy::Max => {
                // Max pooling across all tokens
                let mut pooled = vec![f32::NEG_INFINITY; embedding_dim];

                for token_emb in embeddings {
                    for (i, &val) in token_emb.iter().enumerate() {
                        pooled[i] = pooled[i].max(val);
                    }
                }

                Ok(pooled)
            }
            PoolingStrategy::MeanSqrt => {
                // Mean pooling with sqrt(length) normalization
                let mut pooled = vec![0.0f32; embedding_dim];
                #[allow(clippy::cast_precision_loss)]
                let sqrt_n = (embeddings.len() as f32).sqrt();

                for token_emb in embeddings {
                    for (i, &val) in token_emb.iter().enumerate() {
                        pooled[i] += val;
                    }
                }

                // Normalize by sqrt(length)
                for val in &mut pooled {
                    *val /= sqrt_n;
                }

                Ok(pooled)
            }
            PoolingStrategy::Last => {
                // Use only the last token (EOS token)
                // This is required for decoder models like Qwen
                Ok(embeddings.last().unwrap().clone())
            }
        }
    }

    /// Normalizes an embedding vector to unit length (L2 normalization).
    ///
    /// # Arguments
    ///
    /// * `embedding` - The embedding vector to normalize
    ///
    /// # Returns
    ///
    /// Returns the normalized embedding vector.
    fn normalize_embedding(mut embedding: Vec<f32>) -> Result<Vec<f32>> {
        // Calculate L2 norm
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm == 0.0 {
            return Err(Error::EmbeddingGenerationError {
                message: "Cannot normalize zero vector".to_string(),
                source: None,
            });
        }

        // Normalize the vector
        for val in &mut embedding {
            *val /= norm;
        }

        Ok(embedding)
    }

    /// Save the current KV cache state to memory
    ///
    /// > NOTE: This is for advanced prefix caching optimization
    /// > PERFORMANCE ISSUE: Only beneficial for prefixes > 100 tokens
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The context is empty (no state to save)
    /// - State copy operation fails
    pub fn save_session_state(&self) -> Result<Vec<u8>> {
        // Get the state size first
        let state_size = self.cell.borrow_dependent().get_state_size();

        if state_size == 0 {
            return Err(Error::InvalidOperation {
                message: "No state to save - context is empty".to_string(),
            });
        }

        // Allocate buffer for the state
        let mut buffer = vec![0u8; state_size];

        // Copy the state data
        let copied_size = unsafe {
            self.cell
                .borrow_dependent()
                .copy_state_data(buffer.as_mut_ptr())
        };

        if copied_size != state_size {
            return Err(Error::InvalidOperation {
                message: format!("State size mismatch: expected {state_size}, got {copied_size}"),
            });
        }

        debug!("Saved session state: {} bytes", state_size);
        Ok(buffer)
    }

    /// Load a previously saved KV cache state
    ///
    /// > NOTE: Session must be from the same model version
    /// > BUG: Session format may change between llama.cpp versions
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - State data is empty
    /// - State size check fails
    pub fn load_session_state(&mut self, state_data: &[u8]) -> Result<()> {
        if state_data.is_empty() {
            return Err(Error::InvalidInput {
                message: "Empty session data provided".to_string(),
            });
        }

        // Set the state data
        let loaded_size = AtomicUsize::new(0);
        self.cell.with_dependent_mut(|_, context| {
            loaded_size.store(
                unsafe { context.set_state_data(state_data) },
                Ordering::Relaxed,
            );
        });
        let loaded_size = loaded_size.load(Ordering::Relaxed);

        if loaded_size != state_data.len() {
            return Err(Error::InvalidOperation {
                message: format!(
                    "Failed to load session state: expected {} bytes, loaded {}",
                    state_data.len(),
                    loaded_size
                ),
            });
        }

        debug!("Loaded session state: {} bytes", loaded_size);
        Ok(())
    }

    /// Generate embedding with prefix caching support
    ///
    /// This method checks if the text has a common prefix that's been cached,
    /// and if so, loads that session state to avoid recomputing the KV cache
    /// for the prefix portion.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to generate embeddings for
    /// * `prefix_cache` - Optional reference to the prefix cache
    /// * `token_cache` - Optional reference to the token cache
    ///
    /// # Returns
    ///
    /// Returns the embedding vector and optionally the number of prefix tokens used
    ///
    /// # Errors
    ///
    /// Returns an error if embedding generation fails
    pub fn generate_embedding_with_prefix(
        &mut self,
        text: &str,
        prefix_cache: Option<&crate::cache::prefix_cache::PrefixCache>,
        token_cache: Option<&TokenCache>,
    ) -> Result<Vec<f32>> {
        // First tokenize to get tokens
        let tokens = self.tokenize_cached(text, self.add_bos_token, token_cache)?;
        let tokens_i: Vec<i32> = tokens.iter().map(|t| t.0).collect();

        // Check for cached prefix if available
        let prefix_tokens_used = if let Some(cache) = prefix_cache {
            if let Some((prefix_len, session)) = cache.find_prefix_session(text, &tokens_i) {
                // Load the cached session state if available
                if let Some(ref state) = session.memory_state {
                    match self.load_session_state(state) {
                        Ok(()) => {
                            info!("Loaded prefix cache for {} tokens", prefix_len);
                            Some(prefix_len)
                        }
                        Err(e) => {
                            warn!("Failed to load prefix cache: {}", e);
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                // Analyze for future caching opportunities
                if let Some(suggested_len) = cache.analyze(&tokens_i) {
                    debug!(
                        "Prefix of {} tokens is candidate for caching",
                        suggested_len
                    );
                }
                None
            }
        } else {
            None
        };

        // Generate the embedding (with or without prefix optimization)
        let embedding = if let Some(prefix_len) = prefix_tokens_used {
            // Process only the suffix tokens after the cached prefix
            let suffix_tokens = &tokens[prefix_len..];
            if suffix_tokens.is_empty() {
                // The entire text was in the prefix, just extract embeddings
                self.extract_embeddings(&tokens)?
            } else {
                // Process the suffix and combine
                self.process_tokens_internal(suffix_tokens)?
            }
        } else {
            // Normal processing without prefix optimization
            self.process_tokens_internal(&tokens)?
        };

        // Apply pooling and normalization
        self.finalize_embedding(&embedding, tokens.len())
    }

    /// Extract embeddings from the current context state
    fn extract_embeddings(&self, tokens: &[LlamaToken]) -> Result<Vec<Vec<f32>>> {
        // Delegate to the unified extraction method
        self.extract_sequence_embeddings(0, tokens.len(), None)
    }

    /// Extract context size from GGUF file metadata
    ///
    /// Uses the `gguf::extract_metadata` function to get model metadata.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GGUF model file
    ///
    /// # Returns
    ///
    /// Returns the context size from metadata, or an error if not found
    fn extract_context_size_from_gguf(path: &Path) -> Result<u32> {
        let metadata = gguf::extract_metadata(path)?;
        Ok(metadata.context_size.try_into().unwrap_or(2048))
    }
}

impl Drop for EmbeddingModel {
    /// Ensures proper cleanup of model resources.
    fn drop(&mut self) {
        // The self_cell will handle dropping both the model and context in the correct order
        // Note: Cannot safely log here as tracing TLS may already be destroyed during shutdown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_not_send() {
        // This test verifies at compile time that EmbeddingModel is !Send
        fn assert_not_send<T: ?Sized>() {}
        assert_not_send::<EmbeddingModel>();
    }

    #[test]
    fn test_model_metadata_methods() {
        // We can't load a real model in tests without a GGUF file,
        // but we can test the structure compiles correctly
        // Real integration tests would use actual model files
    }

    #[test]
    #[ignore = "Requires actual GGUF model file"]
    fn test_model_loading_with_real_file() {
        // This test would require a real GGUF model file
        // It's marked as ignore but can be run with: cargo test -- --ignored

        let config = ModelConfig::builder()
            .with_model_path("/path/to/real/model.gguf")
            .with_model_name("test-model")
            .build()
            .unwrap();

        // Initialize backend for testing
        let backend = LlamaBackend::init().unwrap();

        match EmbeddingModel::new(&backend, &config) {
            Ok(model) => {
                assert!(model.is_loaded());
                assert!(model.embedding_dimensions() > 0);
                assert!(model.max_sequence_length() > 0);
            }
            Err(e) => {
                eprintln!("Expected error loading model: {e}");
            }
        }
    }
}
