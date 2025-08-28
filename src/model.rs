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

use crate::config::{ModelConfig, PoolingStrategy};
use crate::error::{Error, Result};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::LogOptions;
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    token::LlamaToken,
};
use self_cell::self_cell;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::num::NonZeroU32;
use tracing::{debug, info, instrument};

/// Initialize the llama backend once per process
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

/// Lock to protect backend initialization
static BACKEND_INIT_LOCK: Mutex<()> = Mutex::new(());

/// Initialize the llama backend
fn init_backend() -> Result<&'static LlamaBackend> {
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }

    let _guard = BACKEND_INIT_LOCK.lock().unwrap();

    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    llama_cpp_2::send_logs_to_tracing(LogOptions::default().with_logs_enabled(true));

    let backend = LlamaBackend::init().map_err(|e| Error::ModelInitError {
        message: "Failed to initialize llama backend".to_string(),
        source: Some(anyhow::anyhow!("{}", e)),
    })?;

    info!("Initialized llama backend");
    let _ = BACKEND.set(backend);
    Ok(BACKEND.get().unwrap())
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
}

impl EmbeddingModel {
    /// Creates a new embedding model from the given configuration.
    ///
    /// # Arguments
    ///
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
    #[instrument(skip(config), fields(model_path = %config.model_path.display()))]
    pub fn new(config: &ModelConfig) -> Result<Self> {
        // Initialize the backend if not already done
        let backend = init_backend()?;

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
        let model = 
            LlamaModel::load_from_file(backend, &config.model_path, &model_params)
                .map_err(|e| {
                    Error::ModelLoadError {
                        path: config.model_path.clone(),
                        source: anyhow::anyhow!("Failed to load model: {}", e),
                    }
                })?;

        debug!("Model loaded successfully");

        // Set up context parameters
        let ctx_size = config.n_ctx.unwrap_or(2048);
        let n_threads = config.n_threads.unwrap_or_else(|| {
            let threads = num_cpus::get();
            debug!("Using {} CPU threads", threads);
            threads
        }) as i32;

        let mut ctx_params = LlamaContextParams::default();
        
        // Set context size
        let n_ctx = NonZeroU32::new(ctx_size);
        ctx_params = ctx_params.with_n_ctx(n_ctx);
        
        // Set thread counts
        ctx_params = ctx_params.with_n_threads(n_threads);
        ctx_params = ctx_params.with_n_threads_batch(n_threads);
        
        // Enable embeddings mode
        ctx_params = ctx_params.with_embeddings(true);
        
        // Enable flash attention if available
        ctx_params = ctx_params.with_flash_attention(true);

        // Get embedding dimensions from the model
        let embedding_dimensions = model.n_embd() as usize;
        
        info!(
            "Model initialized: dimensions={}, context_size={}, threads={}",
            embedding_dimensions, ctx_size, n_threads
        );

        let cell = ModelCell::try_new(model, |m| m.new_context(backend, ctx_params)
            .map_err(|e| Error::ContextError {
                source: anyhow::anyhow!("Failed to create context: {}", e),
            }))?;

        // Determine whether to add BOS token
        let add_bos_token = if let Some(add_bos) = config.add_bos_token {
            // Use explicitly configured value
            debug!("Using configured add_bos_token: {}", add_bos);
            add_bos
        } else {
            // Auto-detect based on model type
            let detected = Self::detect_add_bos_token(&config.model_path, &config.model_name);
            debug!("Auto-detected add_bos_token: {} for model: {}", detected, config.model_name);
            detected
        };

        Ok(Self {
            cell,
            config: config.clone(),
            model_path: config.model_path.clone(),
            model_name: config.model_name.clone(),
            embedding_dimensions,
            max_context_size: ctx_size as usize,
            add_bos_token,
        })
    }

    /// Loads a model from disk.
    ///
    /// This is an alternative way to create a model, useful when you want
    /// to explicitly separate the loading step.
    ///
    /// # Arguments
    ///
    /// * `config` - The model configuration
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the loaded model or an error.
    pub fn load(config: &ModelConfig) -> Result<Self> {
        Self::new(config)
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
    /// Estimated memory usage in bytes.
    pub fn model_size(&self) -> usize {
        // This is an approximation based on model parameters
        // More accurate measurement would require llama.cpp API support
        let params = self.cell.borrow_owner().n_params();
        let size_per_param = 2; // Approximate bytes per parameter for quantized models
        params as usize * size_per_param
    }

    /// Returns the model's metadata.
    ///
    /// # Returns
    ///
    /// A tuple containing (model_name, model_path, vocab_size, n_params).
    pub fn model_metadata(&self) -> (String, PathBuf, usize, usize) {
        (
            self.model_name.clone(),
            self.model_path.clone(),
            self.cell.borrow_owner().n_vocab() as usize,
            self.cell.borrow_owner().n_params() as usize,
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
    fn detect_add_bos_token(model_path: &PathBuf, model_name: &str) -> bool {
        // Convert model name to lowercase for case-insensitive matching
        let name_lower = model_name.to_lowercase();
        let path_str = model_path.to_string_lossy().to_lowercase();
        
        // Check for known encoder model patterns
        let encoder_patterns = [
            "bert",      // BERT and variants (RoBERTa, DistilBERT, etc.)
            "e5",        // E5 embeddings
            "bge",       // BGE embeddings  
            "gte",       // GTE embeddings
            "minilm",    // MiniLM models
            "mpnet",     // MPNet models
            "sentence",  // Sentence transformers
            "all-minilm", // All-MiniLM variants
            "all-mpnet", // All-MPNet variants
            "paraphrase", // Paraphrase models (usually BERT-based)
            "msmarco",   // MS MARCO models (usually BERT-based)
            "contriever", // Contriever models
            "simcse",    // SimCSE models
            "instructor", // Instructor models
            "unsup-simcse", // Unsupervised SimCSE
            "sup-simcse", // Supervised SimCSE
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

        self.cell.borrow_owner()
            .str_to_token(text, add_bos)
            .map_err(|e| Error::TokenizationError {
                message: format!("Failed to tokenize text: {}", e),
            })
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
        // Validate input
        if text.is_empty() {
            return Err(Error::InvalidInput {
                message: "Cannot generate embedding for empty text".to_string(),
            });
        }

        // Tokenize the text (using configured/detected add_bos_token setting)
        let tokens = self.tokenize(text, self.add_bos_token)?;
        
        // Check token limit
        if tokens.len() > self.max_context_size {
            return Err(Error::InvalidInput {
                message: format!(
                    "Text exceeds maximum token limit: {} > {}",
                    tokens.len(),
                    self.max_context_size
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
            self.apply_pooling(&embeddings, &self.config.pooling_strategy)?
        };
        
        // Normalize if configured
        let final_embedding = if self.config.normalize_embeddings {
            self.normalize_embedding(pooled)?
        } else {
            pooled
        };

        Ok(final_embedding)
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
    #[instrument(skip(self, tokens), fields(token_count = tokens.len()))]
    pub fn process_tokens(&mut self, tokens: &[i32]) -> Result<Vec<f32>> {
        // Convert i32 tokens to LlamaToken and process
        let llama_tokens: Vec<LlamaToken> = tokens.iter().map(|&t| LlamaToken(t)).collect();
        let embeddings = self.process_tokens_internal(&llama_tokens)?;
        
        // Apply pooling strategy
        let pooled = self.apply_pooling(&embeddings, &self.config.pooling_strategy)?;
        
        // Normalize if configured
        let final_embedding = if self.config.normalize_embeddings {
            self.normalize_embedding(pooled)?
        } else {
            pooled
        };
        
        Ok(final_embedding)
    }

    /// Internal method to process LlamaToken vectors.
    fn process_tokens_internal(&mut self, tokens: &[LlamaToken]) -> Result<Vec<Vec<f32>>> {
        if tokens.is_empty() {
            return Err(Error::InvalidInput {
                message: "Cannot process empty token list".to_string(),
            });
        }

        // Create a batch for processing
        let n_tokens = tokens.len();
        let mut batch = LlamaBatch::new(n_tokens, 1);
        batch.add_sequence(tokens, 0, true).map_err(|e| {
                Error::EmbeddingGenerationError {
                    message: format!("Failed to add tokens to batch: {}", e),
                    source: Some(anyhow::anyhow!(e)),
                }})?;

        // Process the batch through the model
        self.cell.with_dependent_mut(|_, ctx| {
            ctx.encode(&mut batch).map_err(|e| {
                Error::EmbeddingGenerationError {
                    message: format!("Failed to encode batch: {}", e),
                    source: Some(anyhow::anyhow!(e)),
                }
            })
        })?;

        // Extract embeddings - try sequence embeddings first for BERT models
        // If that fails, fall back to token-wise embeddings
        let all_embeddings = self.cell.with_dependent(|_, ctx| -> Result<Vec<Vec<f32>>> {
            // Try to get sequence embeddings (for BERT models)
            if let Ok(seq_embeddings) = ctx.embeddings_seq_ith(0) {
                // For BERT models with pooling, we get one embedding for the whole sequence
                Ok(vec![seq_embeddings.to_vec()])
            } else {
                // Fall back to token-wise embeddings (for LLaMA-style models)
                let mut token_embeddings = Vec::with_capacity(n_tokens);
                for i in 0..n_tokens {
                    let embeddings = ctx.embeddings_ith(i as i32)
                        .map_err(|e| Error::EmbeddingGenerationError {
                            message: format!("Failed to get embeddings for token {}", i),
                            source: Some(anyhow::anyhow!(e)),
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
    fn apply_pooling(
        &self,
        embeddings: &[Vec<f32>],
        strategy: &PoolingStrategy,
    ) -> Result<Vec<f32>> {
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
    fn normalize_embedding(&self, mut embedding: Vec<f32>) -> Result<Vec<f32>> {
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
}

impl Drop for EmbeddingModel {
    /// Ensures proper cleanup of model resources.
    fn drop(&mut self) {
        debug!("Dropping model: {}", self.model_name);
        // The self_cell will handle dropping both the model and context in the correct order
    }
}

/// Helper function to create a test model for unit tests.
#[cfg(test)]
pub(crate) fn create_test_config() -> ModelConfig {
    use std::fs;
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let model_path = dir.path().join("test_model.gguf");
    
    // Create a dummy file for testing
    fs::write(&model_path, b"dummy model file").unwrap();
    
    ModelConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .with_n_ctx(512)
        .with_n_threads(1)
        .build()
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_initialization() {
        // Test that backend initialization works
        let _backend = init_backend();
        // Calling it again should be safe (Once ensures single initialization)
        let _backend2 = init_backend();
    }

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
            ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2.gguf"),
            ("bge-large-en", "BAAI/bge-large-en-v1.5.gguf"),
            ("e5-base", "intfloat/e5-base-v2.gguf"),
            ("gte-base", "thenlper/gte-base.gguf"),
            ("all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2.gguf"),
            ("msmarco-bert", "cross-encoder/ms-marco-MiniLM-L-6-v2.gguf"),
            ("contriever-msmarco", "facebook/contriever-msmarco.gguf"),
            ("simcse-bert", "princeton-nlp/unsup-simcse-bert-base.gguf"),
            ("instructor-base", "hkunlp/instructor-base.gguf"),
        ];
        
        for (model_name, model_path) in encoder_cases {
            let path = PathBuf::from(model_path);
            let result = EmbeddingModel::detect_add_bos_token(&path, model_name);
            assert!(!result, "Expected false (no BOS) for encoder model: {}", model_name);
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
            assert!(result, "Expected true (add BOS) for decoder model: {}", model_name);
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
            assert!(result, "Expected true (default) for unknown model: {}", model_name);
        }
    }
    
    #[test]
    fn test_detect_add_bos_token_case_insensitive() {
        use std::path::PathBuf;
        
        // Test case-insensitive matching
        let cases = vec![
            ("BERT-Base", "models/BERT.GGUF", false),
            ("ALL-MINILM-L6", "SENTENCE-TRANSFORMERS/ALL-MINILM.gguf", false),
            ("LLAMA-2-7B", "META-LLAMA/LLAMA-2.GGUF", true),
            ("MISTRAL-7B", "MISTRALAI/MISTRAL.gguf", true),
        ];
        
        for (model_name, model_path, expected) in cases {
            let path = PathBuf::from(model_path);
            let result = EmbeddingModel::detect_add_bos_token(&path, model_name);
            assert_eq!(result, expected, "Case-insensitive test failed for: {}", model_name);
        }
    }

    #[test]
    #[ignore] // Requires actual GGUF model file
    fn test_model_loading_with_real_file() {
        // This test would require a real GGUF model file
        // It's marked as ignore but can be run with: cargo test -- --ignored
        
        let config = ModelConfig::builder()
            .with_model_path("/path/to/real/model.gguf")
            .with_model_name("test-model")
            .build()
            .unwrap();

        match EmbeddingModel::new(&config) {
            Ok(model) => {
                assert!(model.is_loaded());
                assert!(model.embedding_dimensions() > 0);
                assert!(model.max_sequence_length() > 0);
            }
            Err(e) => {
                eprintln!("Expected error loading model: {}", e);
            }
        }
    }

}
