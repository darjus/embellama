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

//! Embedding engine module for the embellama library.
//!
//! This module provides the main `EmbeddingEngine` struct which serves as
//! the primary interface for the library, managing model lifecycle and
//! providing high-level embedding generation APIs.

use crate::batch::{BatchProcessor, BatchProcessorBuilder};
use crate::config::EngineConfig;
use crate::error::{Error, Result};
use crate::model::EmbeddingModel;
use parking_lot::RwLock;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, instrument};

/// Thread-local storage for models due to !Send constraint of LlamaContext
thread_local! {
    static THREAD_MODELS: RefCell<HashMap<String, EmbeddingModel>> = RefCell::new(HashMap::new());
}

/// The main entry point for the embellama library.
///
/// `EmbeddingEngine` manages the lifecycle of embedding models and provides
/// a high-level API for generating embeddings. It supports loading multiple
/// models and switching between them.
///
/// # Important
///
/// Due to the `!Send` constraint of `LlamaContext`, each thread maintains
/// its own copy of loaded models. This means:
/// - Models are loaded per-thread when first accessed
/// - Memory usage scales with number of threads × number of models
/// - Model loading may happen multiple times across threads
///
/// # Example
///
/// ```ignore
/// use embellama::{EmbeddingEngine, EngineConfig};
///
/// let config = EngineConfig::builder()
///     .with_model_path("path/to/model.gguf")
///     .with_model_name("my-model")
///     .build()?;
///
/// let engine = EmbeddingEngine::new(config)?;
/// let embedding = engine.embed("my-model", "Hello, world!")?;
/// ```
pub struct EmbeddingEngine {
    /// Registry of model configurations
    model_configs: Arc<RwLock<HashMap<String, EngineConfig>>>,
    /// Default model name if none specified
    default_model: Option<String>,
}

impl EmbeddingEngine {
    /// Creates a new embedding engine with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The engine configuration
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the initialized engine or an error.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The configuration is invalid
    /// - Model loading fails
    #[instrument(skip(config), fields(model_name = %config.model_name))]
    pub fn new(config: EngineConfig) -> Result<Self> {
        // Validate configuration
        config.validate()?;
        
        let model_name = config.model_name.clone();
        info!("Initializing embedding engine with model: {}", model_name);
        
        // Create the engine with the initial model config
        let mut model_configs = HashMap::new();
        model_configs.insert(model_name.clone(), config);
        
        let engine = Self {
            model_configs: Arc::new(RwLock::new(model_configs)),
            default_model: Some(model_name.clone()),
        };
        
        // Load the model in the current thread
        engine.ensure_model_loaded(&model_name)?;
        
        info!("Embedding engine initialized successfully");
        Ok(engine)
    }

    /// Loads a model with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The model configuration
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - A model with the same name is already loaded
    /// - Model loading fails
    #[instrument(skip(self, config), fields(model_name = %config.model_name))]
    pub fn load_model(&mut self, config: EngineConfig) -> Result<()> {
        // Validate configuration
        config.validate()?;
        
        let model_name = config.model_name.clone();
        
        // Check if model already exists
        {
            let configs = self.model_configs.read();
            if configs.contains_key(&model_name) {
                return Err(Error::ConfigurationError {
                    message: format!("Model '{}' is already loaded", model_name),
                });
            }
        }
        
        // Add configuration to registry
        {
            let mut configs = self.model_configs.write();
            configs.insert(model_name.clone(), config);
        }
        
        // Set as default if it's the first model
        if self.default_model.is_none() {
            self.default_model = Some(model_name.clone());
        }
        
        info!("Model '{}' configuration added to registry", model_name);
        Ok(())
    }

    /// Unloads a model by name.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to unload
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The model is not found
    #[instrument(skip(self))]
    pub fn unload_model(&mut self, model_name: &str) -> Result<()> {
        // Remove from config registry
        {
            let mut configs = self.model_configs.write();
            if !configs.contains_key(model_name) {
                return Err(Error::ModelNotFound {
                    name: model_name.to_string(),
                });
            }
            configs.remove(model_name);
        }
        
        // Remove from thread-local storage
        THREAD_MODELS.with(|models| {
            let mut models = models.borrow_mut();
            models.remove(model_name);
        });
        
        // Update default model if needed
        if self.default_model.as_ref() == Some(&model_name.to_string()) {
            let configs = self.model_configs.read();
            self.default_model = configs.keys().next().cloned();
        }
        
        info!("Model '{}' unloaded", model_name);
        Ok(())
    }

    /// Ensures a model is loaded in the current thread.
    ///
    /// This is an internal method that handles thread-local model loading.
    fn ensure_model_loaded(&self, model_name: &str) -> Result<()> {
        THREAD_MODELS.with(|models| {
            let mut models = models.borrow_mut();
            
            // Check if model is already loaded in this thread
            if models.contains_key(model_name) {
                debug!("Model '{}' already loaded in current thread", model_name);
                return Ok(());
            }
            
            // Get configuration from registry
            let config = {
                let configs = self.model_configs.read();
                configs
                    .get(model_name)
                    .ok_or_else(|| Error::ModelNotFound {
                        name: model_name.to_string(),
                    })?
                    .clone()
            };
            
            info!("Loading model '{}' in current thread", model_name);
            
            // Convert EngineConfig to ModelConfig and create model
            let model_config = config.to_model_config();
            let model = EmbeddingModel::new(&model_config)?;
            
            // Store in thread-local map
            models.insert(model_name.to_string(), model);
            
            info!("Model '{}' loaded successfully in current thread", model_name);
            Ok(())
        })
    }

    /// Generates an embedding for a single text using the specified model.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to use (or None for default)
    /// * `text` - The text to generate embeddings for
    ///
    /// # Returns
    ///
    /// Returns a vector of f32 values representing the embedding.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The model is not found
    /// - Embedding generation fails
    #[instrument(skip(self, text), fields(text_len = text.len()))]
    pub fn embed(&self, model_name: Option<&str>, text: &str) -> Result<Vec<f32>> {
        // Determine which model to use
        let model_name = model_name
            .map(|s| s.to_string())
            .or_else(|| self.default_model.clone())
            .ok_or_else(|| Error::ConfigurationError {
                message: "No model specified and no default model set".to_string(),
            })?;
        
        // Ensure model is loaded in current thread
        self.ensure_model_loaded(&model_name)?;
        
        // Generate embedding using thread-local model
        THREAD_MODELS.with(|models| {
            let mut models = models.borrow_mut();
            let model = models.get_mut(&model_name).ok_or_else(|| {
                Error::ModelNotFound {
                    name: model_name.clone(),
                }
            })?;
            
            model.generate_embedding(text)
        })
    }

    /// Generates embeddings for a batch of texts using the specified model.
    ///
    /// This method processes multiple texts efficiently using parallel processing
    /// for tokenization and post-processing while respecting the single-threaded
    /// constraint of model inference.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to use (or None for default)
    /// * `texts` - A vector of texts to generate embeddings for
    ///
    /// # Returns
    ///
    /// Returns a vector of embedding vectors.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The model is not found
    /// - Any embedding generation fails
    #[instrument(skip(self, texts), fields(batch_size = texts.len()))]
    pub fn embed_batch(&self, model_name: Option<&str>, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        // Determine which model to use
        let model_name = model_name
            .map(|s| s.to_string())
            .or_else(|| self.default_model.clone())
            .ok_or_else(|| Error::ConfigurationError {
                message: "No model specified and no default model set".to_string(),
            })?;
        
        // Ensure model is loaded in current thread
        self.ensure_model_loaded(&model_name)?;
        
        // Get model configuration for batch processing
        let config = self.model_configs.read()
            .get(&model_name)
            .ok_or_else(|| Error::ModelNotFound { name: model_name.clone() })?
            .clone();
        
        // Create batch processor with model configuration
        let batch_processor = BatchProcessorBuilder::default()
            .with_max_batch_size(64)  // Default batch size
            .with_normalization(config.normalize_embeddings)
            .with_pooling_strategy(config.pooling_strategy.clone())
            .build();
        
        // Process batch using the BatchProcessor
        THREAD_MODELS.with(|models| {
            let mut models = models.borrow_mut();
            let model = models.get_mut(&model_name).ok_or_else(|| {
                Error::ModelNotFound {
                    name: model_name.clone(),
                }
            })?;
            
            batch_processor.process_batch(model, texts)
        })
    }

    /// Lists all currently loaded models.
    ///
    /// Note: This returns models registered in the engine, not necessarily
    /// loaded in the current thread.
    ///
    /// # Returns
    ///
    /// Returns a vector of model names.
    pub fn list_models(&self) -> Vec<String> {
        let configs = self.model_configs.read();
        configs.keys().cloned().collect()
    }

    /// Checks if a model is loaded.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to check
    ///
    /// # Returns
    ///
    /// Returns true if the model is loaded, false otherwise.
    pub fn is_model_loaded(&self, model_name: &str) -> bool {
        let configs = self.model_configs.read();
        configs.contains_key(model_name)
    }

    /// Gets the default model name.
    ///
    /// # Returns
    ///
    /// Returns the default model name if set.
    pub fn default_model(&self) -> Option<String> {
        self.default_model.clone()
    }

    /// Sets the default model.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to set as default
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not loaded.
    pub fn set_default_model(&mut self, model_name: &str) -> Result<()> {
        if !self.is_model_loaded(model_name) {
            return Err(Error::ModelNotFound {
                name: model_name.to_string(),
            });
        }
        self.default_model = Some(model_name.to_string());
        Ok(())
    }

    /// Gets information about a loaded model.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model
    ///
    /// # Returns
    ///
    /// Returns model information if the model is loaded.
    pub fn model_info(&self, model_name: &str) -> Result<ModelInfo> {
        // Ensure model is loaded in current thread
        self.ensure_model_loaded(model_name)?;
        
        THREAD_MODELS.with(|models| {
            let models = models.borrow();
            let model = models.get(model_name).ok_or_else(|| {
                Error::ModelNotFound {
                    name: model_name.to_string(),
                }
            })?;
            
            Ok(ModelInfo {
                name: model_name.to_string(),
                dimensions: model.embedding_dimensions(),
                max_tokens: model.max_sequence_length(),
                model_size: model.model_size(),
            })
        })
    }

    /// Warms up a model by generating a test embedding.
    ///
    /// This can be useful to ensure the model is fully loaded and ready
    /// before processing actual requests.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to warm up (or None for default)
    ///
    /// # Errors
    ///
    /// Returns an error if warmup fails.
    pub fn warmup_model(&self, model_name: Option<&str>) -> Result<()> {
        let test_text = "This is a warmup text for model initialization.";
        let _ = self.embed(model_name, test_text)?;
        debug!("Model warmed up successfully");
        Ok(())
    }
}

/// Information about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model name
    pub name: String,
    /// Embedding dimensions
    pub dimensions: usize,
    /// Maximum token count
    pub max_tokens: usize,
    /// Approximate model size in bytes
    pub model_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn create_test_config() -> EngineConfig {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("test_model.gguf");
        fs::write(&model_path, b"dummy model file").unwrap();
        
        EngineConfig::builder()
            .with_model_path(model_path)
            .with_model_name("test-model")
            .build()
            .unwrap()
    }

    #[test]
    fn test_engine_creation() {
        // This test would require a real GGUF model file
        // For now, we just test that the structure compiles correctly
    }

    #[test]
    fn test_model_listing() {
        // Test would require real model files
    }

    #[test]
    #[ignore] // Requires actual GGUF model file
    fn test_embedding_generation() {
        let config = create_test_config();
        let engine = EmbeddingEngine::new(config).unwrap();
        
        let text = "Hello, world!";
        let embedding = engine.embed(None, text).unwrap();
        
        assert!(!embedding.is_empty());
    }
}