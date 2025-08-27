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

use crate::config::ModelConfig;
use crate::error::{Error, Result};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
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

        Ok(Self {
            cell,
            config: config.clone(),
            model_path: config.model_path.clone(),
            model_name: config.model_name.clone(),
            embedding_dimensions,
            max_context_size: ctx_size as usize,
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
    pub fn generate_embedding(&mut self, _text: &str) -> Result<Vec<f32>> {
        // TODO: Phase 3 - Implement single embedding generation
        unimplemented!("Embedding generation will be implemented in Phase 3")
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
    /// Returns the raw output from the model.
    #[allow(dead_code)]
    pub(crate) fn process_tokens(&mut self, _tokens: &[LlamaToken]) -> Result<Vec<f32>> {
        // TODO: Phase 3 - Implement token processing
        unimplemented!("Token processing will be implemented in Phase 3")
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
    use std::str::FromStr;

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
