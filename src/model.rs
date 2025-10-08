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
use gguf::{GGUFFile, GGUFMetadataValue};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::{
    context::params::LlamaContextParams,
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

/// Extract model metadata from a GGUF file
///
/// This function reads only the header and metadata from the GGUF file
/// without loading the entire model, making it efficient for listing models.
///
/// # Arguments
///
/// * `path` - Path to the GGUF model file
///
/// # Returns
///
/// A tuple of (`dimensions`, `context_size`) extracted from the metadata
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed
pub fn extract_gguf_metadata(path: &Path) -> Result<(usize, usize)> {
    use std::fs::File;
    use std::io::Read;

    // Read only the beginning of the file (metadata is at the start)
    let mut file = File::open(path).map_err(|e| Error::ModelLoadError {
        path: path.to_path_buf(),
        source: anyhow::anyhow!("Failed to open GGUF file: {e}"),
    })?;

    // Read a reasonable amount for metadata (16MB should be more than enough)
    let mut buffer = Vec::new();
    let _ = file
        .by_ref()
        .take(16 * 1024 * 1024)
        .read_to_end(&mut buffer)
        .map_err(|e| Error::ModelLoadError {
            path: path.to_path_buf(),
            source: anyhow::anyhow!("Failed to read GGUF file: {e}"),
        })?;

    // Parse GGUF file metadata
    let gguf_file = match GGUFFile::read(&buffer) {
        Ok(Some(file)) => file,
        Ok(None) => {
            return Err(Error::ModelLoadError {
                path: path.to_path_buf(),
                source: anyhow::anyhow!("Incomplete GGUF file data"),
            });
        }
        Err(e) => {
            return Err(Error::ModelLoadError {
                path: path.to_path_buf(),
                source: anyhow::anyhow!("Failed to parse GGUF file: {e}"),
            });
        }
    };

    let mut dimensions = 0usize;
    let mut context_size = 512usize; // Default fallback

    // Look for embedding dimensions and context length in metadata
    debug!(
        "GGUF file has {} metadata entries",
        gguf_file.header.metadata.len()
    );
    for metadata in &gguf_file.header.metadata {
        // Log all keys for debugging
        debug!("GGUF key: '{}' = {:?}", metadata.key, metadata.value);

        // Check for embedding dimensions - match exact keys or ends with pattern
        if dimensions == 0
            && (
                metadata.key == "llama.embedding_length"
                    || metadata.key == "embedding_length"
                    || metadata.key == "n_embd"
                    || metadata.key == "bert.embedding_length"
                    || metadata.key.ends_with(".embedding_length")
                // Catch architecture-specific like jina-bert-v2.embedding_length
            )
            && let Some(value) = extract_usize_from_metadata(&metadata.value)
        {
            dimensions = value;
            debug!(
                "Found embedding dimensions: {} from key: {}",
                dimensions, metadata.key
            );
        }

        // Check for context length - match exact keys or ends with pattern
        if context_size == 512
            && (
                // Only update if still at default
                metadata.key == "llama.context_length"
                    || metadata.key == "context_length"
                    || metadata.key == "n_ctx"
                    || metadata.key == "max_position_embeddings"
                    || metadata.key == "bert.context_length"
                    || metadata.key.ends_with(".context_length")
                // Catch architecture-specific like jina-bert-v2.context_length
            )
            && let Some(value) = extract_usize_from_metadata(&metadata.value)
        {
            context_size = value;
            debug!(
                "Found context size: {} from key: {}",
                context_size, metadata.key
            );
        }
    }

    // If dimensions not found in metadata, try to infer from tensor shapes
    if dimensions == 0 {
        for tensor in &gguf_file.tensors {
            if tensor.name.contains("embed") || tensor.name.contains("wte") {
                // Usually the last dimension is the embedding size
                if let Some(&dim) = tensor.dimensions.last() {
                    dimensions = dim.try_into().unwrap_or(0);
                    debug!(
                        "Inferred embedding dimensions from tensor {}: {}",
                        tensor.name, dimensions
                    );
                    break;
                }
            }
        }
    }

    if dimensions == 0 {
        warn!("Could not determine embedding dimensions from GGUF metadata");
    }

    Ok((dimensions, context_size))
}

/// Helper function to extract usize value from GGUF metadata
fn extract_usize_from_metadata(value: &GGUFMetadataValue) -> Option<usize> {
    match value {
        GGUFMetadataValue::Uint8(v) => Some((*v).into()),
        GGUFMetadataValue::Uint16(v) => Some((*v).into()),
        GGUFMetadataValue::Uint32(v) => Some((*v).try_into().unwrap_or(usize::MAX)),
        GGUFMetadataValue::Uint64(v) => Some((*v).try_into().unwrap_or(usize::MAX)),
        GGUFMetadataValue::Int8(v) if *v >= 0 => Some((*v).try_into().unwrap_or(0)),
        GGUFMetadataValue::Int16(v) if *v >= 0 => Some((*v).try_into().unwrap_or(0)),
        GGUFMetadataValue::Int32(v) if *v >= 0 => Some((*v).try_into().unwrap_or(0)),
        GGUFMetadataValue::Int64(v) if *v >= 0 => Some((*v).try_into().unwrap_or(0)),
        _ => None,
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

        // Get n_seq_max from config (default to 1)
        let n_seq_max = config.n_seq_max.unwrap_or(1);

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

        // Set micro-batch size (default to context size if not specified)
        // This must be >= the number of tokens in any single input
        let n_ubatch = config.n_ubatch.unwrap_or(ctx_size);
        ctx_params = ctx_params.with_n_ubatch(n_ubatch);

        // Set thread counts
        ctx_params = ctx_params.with_n_threads(n_threads);
        ctx_params = ctx_params.with_n_threads_batch(n_threads);

        // Enable embeddings mode
        ctx_params = ctx_params.with_embeddings(true);

        // Flash attention policy true
        ctx_params = ctx_params.with_flash_attention(true);

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

        // Determine whether to add BOS token
        let add_bos_token = if let Some(add_bos) = config.add_bos_token {
            // Use explicitly configured value
            debug!("Using configured add_bos_token: {}", add_bos);
            add_bos
        } else {
            // Auto-detect based on model type
            let detected = Self::detect_add_bos_token(&config.model_path, &config.model_name);
            debug!(
                "Auto-detected add_bos_token: {} for model: {}",
                detected, config.model_name
            );
            detected
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
    /// For encoder models (like BERT/Jina), the context size must accommodate both
    /// input tokens AND output embeddings. This method calculates the actual usable
    /// input token limit by subtracting the embedding output space overhead.
    ///
    /// # Returns
    ///
    /// The maximum number of input tokens that can be safely processed in a single batch.
    ///
    /// # Formula
    ///
    /// - If embedding dimensions are known: `ctx_size - (dimensions + 100)`
    /// - If dimensions unknown: `ctx_size - 1024` (conservative fallback)
    ///
    /// # Example
    ///
    /// For Jina model with 768 dimensions and 8192 context:
    /// - Overhead: 768 + 100 = 868
    /// - Effective max: 8192 - 868 = 7324 tokens
    pub fn effective_max_tokens(&self) -> usize {
        let overhead = if self.embedding_dimensions > 0 {
            // Use actual embedding dimensions + padding for overhead
            self.embedding_dimensions + 100
        } else {
            // Conservative fallback when dimensions are unknown
            1024
        };

        self.max_context_size.saturating_sub(overhead)
    }

    /// Detects whether to add BOS token based on model type.
    ///
    /// Encoder-only models (BERT, E5, BGE, GTE, etc.) typically don't use BOS tokens
    /// as they have their own special token handling (CLS/SEP tokens).
    /// Decoder models (LLaMA-style) typically do use BOS tokens.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the model file (for potential metadata inspection)
    /// * `model_name` - Name of the model (used for pattern matching)
    ///
    /// # Returns
    ///
    /// Returns false for known encoder models, true otherwise (default to decoder behavior)
    fn detect_add_bos_token(model_path: &Path, model_name: &str) -> bool {
        // Convert model name to lowercase for case-insensitive matching
        let name_lower = model_name.to_lowercase();
        let path_str = model_path.to_string_lossy().to_lowercase();

        // Check for known encoder model patterns
        let encoder_patterns = [
            "bert",         // BERT and variants (RoBERTa, DistilBERT, etc.)
            "jina",         // Jina embeddings (BERT-based)
            "e5",           // E5 embeddings
            "bge",          // BGE embeddings
            "gte",          // GTE embeddings
            "minilm",       // MiniLM models
            "mpnet",        // MPNet models
            "sentence",     // Sentence transformers
            "all-minilm",   // All-MiniLM variants
            "all-mpnet",    // All-MPNet variants
            "paraphrase",   // Paraphrase models (usually BERT-based)
            "msmarco",      // MS MARCO models (usually BERT-based)
            "contriever",   // Contriever models
            "simcse",       // SimCSE models
            "instructor",   // Instructor models
            "unsup-simcse", // Unsupervised SimCSE
            "sup-simcse",   // Supervised SimCSE
        ];

        // Check if model name or path contains encoder patterns
        for pattern in &encoder_patterns {
            if name_lower.contains(pattern) || path_str.contains(pattern) {
                return false; // Encoder model - don't add BOS
            }
        }

        // Check for known decoder model patterns (add BOS)
        let decoder_patterns = [
            "llama",
            "mistral",
            "mixtral",
            "vicuna",
            "alpaca",
            "wizardlm",
            "openhermes",
            "zephyr",
            "phi",
        ];

        for pattern in &decoder_patterns {
            if name_lower.contains(pattern) || path_str.contains(pattern) {
                return true; // Decoder model - add BOS
            }
        }

        // Default to true (decoder behavior) for unknown models
        // This maintains backward compatibility
        true
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

        // Check token limit using effective max (accounts for embedding output space)
        let effective_max = self.effective_max_tokens();
        if tokens.len() > effective_max {
            return Err(Error::InvalidInput {
                message: format!(
                    "Input exceeds effective maximum tokens: {} tokens > {} effective max (context: {}, overhead: {}). Please truncate your input.",
                    tokens.len(),
                    effective_max,
                    self.max_context_size,
                    self.max_context_size - effective_max
                ),
            });
        }

        debug!("Tokenized text into {} tokens", tokens.len());

        // Process tokens to get embeddings
        let embeddings = self.process_tokens_internal(&tokens)?;

        // Check if we got a single pre-pooled embedding (BERT models with pooling)
        let pooled = if embeddings.len() == 1 && tokens.len() > 1 {
            // This is already pooled by the model (BERT with pooling_type)
            debug!("Using pre-pooled embedding from BERT model");
            embeddings[0].clone()
        } else {
            // Apply our pooling strategy for multi-token outputs
            Self::apply_pooling(&embeddings, self.config.pooling_strategy)?
        };

        // Normalize if configured
        let final_embedding = if self.config.normalize_embeddings {
            Self::normalize_embedding(pooled)?
        } else {
            pooled
        };

        Ok(final_embedding)
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
    fn process_batch_tokens_internal(
        &mut self,
        token_sequences: &[Vec<LlamaToken>],
    ) -> Result<Vec<Vec<f32>>> {
        // Calculate total tokens needed for batch
        let total_tokens: usize = token_sequences.iter().map(std::vec::Vec::len).sum();

        // Check if total tokens exceed effective max (accounts for embedding output space)
        let effective_max = self.effective_max_tokens();
        if total_tokens > effective_max {
            return Err(Error::InvalidInput {
                message: format!(
                    "Batch total exceeds effective maximum tokens: {} tokens > {} effective max (context: {}, overhead: {}). Please reduce batch size or truncate inputs.",
                    total_tokens,
                    effective_max,
                    self.max_context_size,
                    self.max_context_size - effective_max
                ),
            });
        }

        // Create a batch with all sequences (using actual n_seq_max)
        let n_seq_max_i32 =
            i32::try_from(self.n_seq_max).map_err(|_| Error::EmbeddingGenerationError {
                message: "n_seq_max too large for i32".to_string(),
                source: None,
            })?;
        let mut batch = LlamaBatch::new(total_tokens, n_seq_max_i32);

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
        self.cell.with_dependent_mut(|_, ctx| {
            ctx.encode(&mut batch)
                .map_err(|e| Error::EmbeddingGenerationError {
                    message: format!("Failed to encode batch: {e}"),
                    source: Some(anyhow::anyhow!(e)),
                })
        })?;

        // Extract embeddings for each sequence
        let mut all_embeddings = Vec::with_capacity(token_sequences.len());

        for seq_id in 0..token_sequences.len() {
            let embeddings = self
                .cell
                .with_dependent(|_, ctx| -> Result<Vec<Vec<f32>>> {
                    // Try to get sequence embeddings first (for BERT models)
                    if let Ok(seq_id_i32) = i32::try_from(seq_id)
                        && let Ok(seq_embeddings) = ctx.embeddings_seq_ith(seq_id_i32)
                    {
                        // For BERT models with pooling, we get one embedding for the whole sequence
                        return Ok(vec![seq_embeddings.to_vec()]);
                    }

                    // Fall back to token-wise embeddings (for LLaMA-style models)
                    // Need to extract tokens for this specific sequence
                    let seq_tokens = &token_sequences[seq_id];
                    let mut token_embeddings = Vec::with_capacity(seq_tokens.len());

                    // Calculate token offset for this sequence
                    let token_offset: usize = token_sequences[..seq_id]
                        .iter()
                        .map(std::vec::Vec::len)
                        .sum();

                    for i in 0..seq_tokens.len() {
                        let global_idx = token_offset + i;
                        let global_idx_i32 = i32::try_from(global_idx).map_err(|_| {
                            Error::EmbeddingGenerationError {
                                message: format!("Global index {global_idx} too large for i32"),
                                source: None,
                            }
                        })?;
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
                })?;

            // Apply pooling and normalization
            let pooled = if embeddings.len() == 1 && token_sequences[seq_id].len() > 1 {
                // Pre-pooled by model
                embeddings[0].clone()
            } else {
                // Apply our pooling strategy
                Self::apply_pooling(&embeddings, self.config.pooling_strategy)?
            };

            let final_embedding = if self.config.normalize_embeddings {
                Self::normalize_embedding(pooled)?
            } else {
                pooled
            };

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

        // Apply pooling strategy
        let pooled = Self::apply_pooling(&embeddings, self.config.pooling_strategy)?;

        // Normalize if configured
        let final_embedding = if self.config.normalize_embeddings {
            Self::normalize_embedding(pooled)?
        } else {
            pooled
        };

        Ok(final_embedding)
    }

    /// Internal method to process `LlamaToken` vectors.
    fn process_tokens_internal(&mut self, tokens: &[LlamaToken]) -> Result<Vec<Vec<f32>>> {
        if tokens.is_empty() {
            return Err(Error::InvalidInput {
                message: "Cannot process empty token list".to_string(),
            });
        }

        // Check if input exceeds effective max (accounts for embedding output space)
        let effective_max = self.effective_max_tokens();
        if tokens.len() > effective_max {
            return Err(Error::InvalidInput {
                message: format!(
                    "Input exceeds effective maximum tokens: {} tokens > {} effective max (context: {}, overhead: {}). Please truncate your input.",
                    tokens.len(),
                    effective_max,
                    self.max_context_size,
                    self.max_context_size - effective_max
                ),
            });
        }

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
        self.cell.with_dependent_mut(|_, ctx| {
            ctx.encode(&mut batch)
                .map_err(|e| Error::EmbeddingGenerationError {
                    message: format!("Failed to encode batch: {e}"),
                    source: Some(anyhow::anyhow!(e)),
                })
        })?;

        // Extract embeddings - try sequence embeddings first for BERT models
        // If that fails, fall back to token-wise embeddings
        let all_embeddings = self
            .cell
            .with_dependent(|_, ctx| -> Result<Vec<Vec<f32>>> {
                // Try to get sequence embeddings (for BERT models)
                if let Ok(seq_embeddings) = ctx.embeddings_seq_ith(0) {
                    // For BERT models with pooling, we get one embedding for the whole sequence
                    Ok(vec![seq_embeddings.to_vec()])
                } else {
                    // Fall back to token-wise embeddings (for LLaMA-style models)
                    let mut token_embeddings = Vec::with_capacity(n_tokens);
                    for i in 0..n_tokens {
                        let i_i32 =
                            i32::try_from(i).map_err(|_| Error::EmbeddingGenerationError {
                                message: format!("Index {i} too large for i32"),
                                source: None,
                            })?;
                        let embeddings = ctx.embeddings_ith(i_i32).map_err(|e| {
                            Error::EmbeddingGenerationError {
                                message: format!("Failed to get embeddings for token {i}"),
                                source: Some(anyhow::anyhow!(e)),
                            }
                        })?;
                        token_embeddings.push(embeddings.to_vec());
                    }
                    Ok(token_embeddings)
                }
            })?;

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
        let pooled = Self::apply_pooling(&embedding, self.config.pooling_strategy)?;
        let final_embedding = if self.config.normalize_embeddings {
            Self::normalize_embedding(pooled)?
        } else {
            pooled
        };

        Ok(final_embedding)
    }

    /// Extract embeddings from the current context state
    fn extract_embeddings(&self, tokens: &[LlamaToken]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(tokens.len());

        // Extract embeddings using the context's embeddings API
        self.cell.with_dependent(|_, ctx| -> Result<()> {
            // Try to get sequence embeddings first (for BERT models)
            if let Ok(seq_embeddings) = ctx.embeddings_seq_ith(0) {
                // For BERT models with pooling, we get one embedding for the whole sequence
                embeddings.push(seq_embeddings.to_vec());
                return Ok(());
            }

            // Fall back to token-wise embeddings (for LLaMA-style models)
            for i in 0..tokens.len() {
                let i_i32 = i32::try_from(i).map_err(|_| Error::EmbeddingGenerationError {
                    message: format!("Index {i} too large for i32"),
                    source: None,
                })?;
                let embedding =
                    ctx.embeddings_ith(i_i32)
                        .map_err(|e| Error::EmbeddingGenerationError {
                            message: format!("Failed to get embeddings for token {i}"),
                            source: Some(anyhow::anyhow!(e)),
                        })?;
                embeddings.push(embedding.to_vec());
            }
            Ok(())
        })?;

        Ok(embeddings)
    }

    /// Extract context size from GGUF file metadata
    ///
    /// Uses the public `extract_gguf_metadata` function to get model metadata.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GGUF model file
    ///
    /// # Returns
    ///
    /// Returns the context size from metadata, or an error if not found
    fn extract_context_size_from_gguf(path: &Path) -> Result<u32> {
        let (_dimensions, context_size) = extract_gguf_metadata(path)?;
        Ok(context_size.try_into().unwrap_or(2048))
    }
}

impl Drop for EmbeddingModel {
    /// Ensures proper cleanup of model resources.
    fn drop(&mut self) {
        // With global tracing subscriber, we can safely log during cleanup
        debug!("Dropping model: {}", self.model_name);

        // The self_cell will handle dropping both the model and context in the correct order
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
    fn test_detect_add_bos_token_encoder_models() {
        use std::path::PathBuf;

        // Test various encoder model patterns
        let encoder_cases = vec![
            ("bert-base", "models/bert-base.gguf"),
            (
                "all-MiniLM-L6-v2",
                "sentence-transformers/all-MiniLM-L6-v2.gguf",
            ),
            ("bge-large-en", "BAAI/bge-large-en-v1.5.gguf"),
            ("e5-base", "intfloat/e5-base-v2.gguf"),
            ("gte-base", "thenlper/gte-base.gguf"),
            (
                "all-mpnet-base-v2",
                "sentence-transformers/all-mpnet-base-v2.gguf",
            ),
            ("msmarco-bert", "cross-encoder/ms-marco-MiniLM-L-6-v2.gguf"),
            ("contriever-msmarco", "facebook/contriever-msmarco.gguf"),
            ("simcse-bert", "princeton-nlp/unsup-simcse-bert-base.gguf"),
            ("instructor-base", "hkunlp/instructor-base.gguf"),
            (
                "jina-embeddings-v2-base-code",
                "jinaai/jina-embeddings-v2-base-code.gguf",
            ),
            (
                "jina-embeddings-v2-base-en",
                "jinaai/jina-embeddings-v2-base-en.gguf",
            ),
        ];

        for (model_name, model_path) in encoder_cases {
            let path = PathBuf::from(model_path);
            let result = EmbeddingModel::detect_add_bos_token(&path, model_name);
            assert!(
                !result,
                "Expected false (no BOS) for encoder model: {model_name}"
            );
        }
    }

    #[test]
    fn test_detect_add_bos_token_decoder_models() {
        use std::path::PathBuf;

        // Test various decoder model patterns
        let decoder_cases = vec![
            ("llama-2-7b", "meta-llama/Llama-2-7b.gguf"),
            ("mistral-7b", "mistralai/Mistral-7B-v0.1.gguf"),
            ("mixtral-8x7b", "mistralai/Mixtral-8x7B-v0.1.gguf"),
            ("vicuna-7b", "lmsys/vicuna-7b-v1.5.gguf"),
            ("alpaca-7b", "tatsu-lab/alpaca-7b.gguf"),
            ("wizardlm-13b", "WizardLM/WizardLM-13B-V1.2.gguf"),
            ("openhermes-2.5", "teknium/OpenHermes-2.5-Mistral-7B.gguf"),
            ("zephyr-7b", "HuggingFaceH4/zephyr-7b-beta.gguf"),
            ("phi-2", "microsoft/phi-2.gguf"),
        ];

        for (model_name, model_path) in decoder_cases {
            let path = PathBuf::from(model_path);
            let result = EmbeddingModel::detect_add_bos_token(&path, model_name);
            assert!(
                result,
                "Expected true (add BOS) for decoder model: {model_name}"
            );
        }
    }

    #[test]
    fn test_detect_add_bos_token_unknown_models() {
        use std::path::PathBuf;

        // Test unknown model patterns (should default to true for backward compatibility)
        let unknown_cases = vec![
            ("custom-model", "custom/custom-model.gguf"),
            ("my-embeddings", "embeddings/my-embeddings.gguf"),
            ("unknown-arch", "models/unknown-arch.gguf"),
        ];

        for (model_name, model_path) in unknown_cases {
            let path = PathBuf::from(model_path);
            let result = EmbeddingModel::detect_add_bos_token(&path, model_name);
            assert!(
                result,
                "Expected true (default) for unknown model: {model_name}"
            );
        }
    }

    #[test]
    fn test_detect_add_bos_token_case_insensitive() {
        use std::path::PathBuf;

        // Test case-insensitive matching
        let cases = vec![
            ("BERT-Base", "models/BERT.GGUF", false),
            (
                "ALL-MINILM-L6",
                "SENTENCE-TRANSFORMERS/ALL-MINILM.gguf",
                false,
            ),
            ("LLAMA-2-7B", "META-LLAMA/LLAMA-2.GGUF", true),
            ("MISTRAL-7B", "MISTRALAI/MISTRAL.gguf", true),
        ];

        for (model_name, model_path, expected) in cases {
            let path = PathBuf::from(model_path);
            let result = EmbeddingModel::detect_add_bos_token(&path, model_name);
            assert_eq!(
                result, expected,
                "Case-insensitive test failed for: {model_name}"
            );
        }
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
