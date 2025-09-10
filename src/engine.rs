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

use crate::batch::BatchProcessorBuilder;
use crate::config::EngineConfig;
use crate::error::{Error, Result};
use crate::model::EmbeddingModel;
use llama_cpp_2::llama_backend::LlamaBackend;
use parking_lot::RwLock;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use tracing::{debug, info, instrument};

// Global singleton instance of the engine
static INSTANCE: RwLock<Option<Arc<Mutex<EmbeddingEngine>>>> = RwLock::new(None);

// Lock to protect singleton initialization
static INIT_LOCK: Mutex<()> = Mutex::new(());

// Global singleton backend instance
static BACKEND: OnceLock<Arc<Mutex<LlamaBackend>>> = OnceLock::new();

// Thread-local storage for models due to !Send constraint of LlamaContext
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
    /// Shared reference to the llama backend instance
    backend: Arc<Mutex<LlamaBackend>>,
    /// Registry of model configurations
    model_configs: Arc<RwLock<HashMap<String, EngineConfig>>>,
    /// Default model name if none specified
    default_model: Option<String>,
}

impl EmbeddingEngine {
    /// Gets or creates the singleton `LlamaBackend` instance.
    ///
    /// This ensures only one backend is created per process, avoiding the
    /// `BackendAlreadyInitialized` error when creating multiple engines.
    fn get_or_create_backend() -> Result<Arc<Mutex<LlamaBackend>>> {
        if let Some(backend) = BACKEND.get() {
            return Ok(Arc::clone(backend));
        }

        // Initialize the backend for the first time
        let mut backend = LlamaBackend::init().map_err(|e| {
            let error_str = format!("{e}");
            if error_str.contains("BackendAlreadyInitialized") {
                Error::ConfigurationError {
                    message: "LlamaBackend already initialized. This is an internal error."
                        .to_string(),
                }
            } else {
                Error::ModelInitError {
                    message: "Failed to initialize llama backend".to_string(),
                    source: Some(anyhow::anyhow!("{}", e)),
                }
            }
        })?;
        backend.void_logs();

        let backend_arc = Arc::new(Mutex::new(backend));
        // Try to set it, but if another thread beat us to it, use theirs
        match BACKEND.set(Arc::clone(&backend_arc)) {
            Ok(()) => Ok(backend_arc),
            Err(_) => Ok(Arc::clone(BACKEND.get().unwrap())),
        }
    }

    /// Gets or initializes the singleton embedding engine with the given configuration.
    ///
    /// If the engine is already initialized, returns the existing instance.
    /// The configuration is only used for the first initialization.
    ///
    /// # Arguments
    ///
    /// * `config` - The engine configuration (used only on first call)
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the engine instance or an error.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The configuration is invalid (on first initialization)
    /// - Model loading fails (on first initialization)
    ///
    /// # Panics
    ///
    /// Panics if the initialization lock cannot be acquired.
    #[instrument(skip(config), fields(model_name = %config.model_name))]
    pub fn get_or_init(config: EngineConfig) -> Result<Arc<Mutex<Self>>> {
        // Fast path: check if already initialized
        {
            let instance_guard = INSTANCE.read();
            if let Some(ref instance) = *instance_guard {
                debug!("Returning existing engine instance");
                return Ok(Arc::clone(instance));
            }
        }

        // Slow path: initialize the singleton
        let _lock = INIT_LOCK.lock().unwrap();

        // Double-check after acquiring lock
        {
            let instance_guard = INSTANCE.read();
            if let Some(ref instance) = *instance_guard {
                debug!("Returning existing engine instance (after lock)");
                return Ok(Arc::clone(instance));
            }
        }

        // Create new instance
        info!("Initializing singleton embedding engine");
        let engine = Self::new_internal(config)?;
        let arc_engine = Arc::new(Mutex::new(engine));

        // Store the instance
        {
            let mut instance_guard = INSTANCE.write();
            *instance_guard = Some(Arc::clone(&arc_engine));
        }

        Ok(arc_engine)
    }

    /// Gets the existing engine instance if it has been initialized.
    ///
    /// # Returns
    ///
    /// Returns `Some(engine)` if initialized, `None` otherwise.
    pub fn instance() -> Option<Arc<Mutex<Self>>> {
        let instance_guard = INSTANCE.read();
        instance_guard.as_ref().map(Arc::clone)
    }

    /// Resets the singleton instance (test-only).
    ///
    /// This method is only available in test builds and should be called
    /// at the start of tests that need a fresh engine state.
    ///
    /// # Safety
    ///
    /// This method should only be called when no other code is using the engine.
    /// Tests using this must be marked with `#[serial]` to prevent parallel execution.
    ///
    /// # Panics
    ///
    /// Panics if the mutex lock cannot be acquired
    #[cfg(test)]
    pub fn reset() {
        let _lock = INIT_LOCK.lock().unwrap();

        // Clear thread-local models first
        THREAD_MODELS.with(|models| {
            models.borrow_mut().clear();
        });

        // Take and drop the instance to ensure backend is dropped
        let mut instance_guard = INSTANCE.write();
        if let Some(instance) = instance_guard.take() {
            // Check if we're the only reference
            if Arc::strong_count(&instance) > 1 {
                // Other references exist - this is likely a test error
                // Put it back and panic
                *instance_guard = Some(instance);
                panic!(
                    "Cannot reset engine: other references exist. Ensure tests are marked with #[serial]"
                );
            }
            // Explicitly drop the instance (and its backend)
            drop(instance);
            debug!("Dropped engine instance and backend");
        }

        // instance_guard is now None
        info!("Engine singleton reset - backend dropped");
    }

    /// Convenience method for tests to get a fresh instance.
    ///
    /// Resets the singleton and initializes with the given config.
    ///
    /// # Errors
    ///
    /// Returns an error if engine creation fails
    #[cfg(test)]
    pub fn fresh_instance(config: EngineConfig) -> Result<Arc<Mutex<Self>>> {
        Self::reset();
        Self::get_or_init(config)
    }

    /// Internal method to create a new engine instance.
    ///
    /// This is the actual implementation, separated from the singleton logic.
    fn new_internal(config: EngineConfig) -> Result<Self> {
        // Validate configuration
        config.validate()?;

        let model_name = config.model_name.clone();
        info!("Initializing embedding engine with model: {}", model_name);

        // Get or create the shared backend
        let backend = Self::get_or_create_backend()?;
        info!("Llama backend ready");

        // Create the engine with the initial model config
        let mut model_configs = HashMap::new();
        model_configs.insert(model_name.clone(), config);

        let engine = Self {
            backend,
            model_configs: Arc::new(RwLock::new(model_configs)),
            default_model: Some(model_name.clone()),
        };

        // Load the model in the current thread
        engine.ensure_model_loaded(&model_name)?;

        info!("Embedding engine initialized successfully");
        Ok(engine)
    }

    /// Creates a new embedding engine with the given configuration.
    ///
    /// **Note**: This now uses the singleton pattern internally. Use `get_or_init()`
    /// for explicit singleton access.
    ///
    /// # Arguments
    ///
    /// * `config` - The engine configuration
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the engine or an error.
    ///
    /// # Errors
    ///
    /// Returns an error if model loading fails
    pub fn new(config: EngineConfig) -> Result<Self> {
        // Use the internal method directly for backward compatibility
        // This allows tests to create instances without singleton
        Self::new_internal(config)
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
                    message: format!("Model '{model_name}' is already loaded"),
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

    /// Unregisters a model from the registry, preventing future loads.
    ///
    /// This removes the model configuration from the registry but does not
    /// affect already-loaded model instances in threads.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to unregister
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not found in the registry.
    #[instrument(skip(self))]
    pub fn unregister_model(&mut self, model_name: &str) -> Result<()> {
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

        // Update default model if needed
        if self.default_model.as_ref() == Some(&model_name.to_string()) {
            let configs = self.model_configs.read();
            self.default_model = configs.keys().next().cloned();
        }

        info!("Model '{}' unregistered from config registry", model_name);
        Ok(())
    }

    /// Drops a model from the current thread's cache.
    ///
    /// This removes the model instance from the current thread but keeps
    /// its configuration in the registry, allowing it to be reloaded later.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to drop from thread
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not registered.
    #[instrument(skip(self))]
    pub fn drop_model_from_thread(&self, model_name: &str) -> Result<()> {
        // First check if model is registered
        {
            let configs = self.model_configs.read();
            if !configs.contains_key(model_name) {
                return Err(Error::ModelNotFound {
                    name: model_name.to_string(),
                });
            }
        }

        // Remove from thread-local storage
        THREAD_MODELS.with(|models| {
            let mut models = models.borrow_mut();
            if models.remove(model_name).is_some() {
                info!("Model '{}' dropped from current thread", model_name);
            } else {
                debug!("Model '{}' was not loaded in current thread", model_name);
            }
        });

        Ok(())
    }

    /// Unloads a model completely (unregisters and drops from thread).
    ///
    /// This is a convenience method that combines `unregister_model` and
    /// `drop_model_from_thread`. It maintains backward compatibility with
    /// the original `unload_model` behavior.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to unload
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not found.
    #[instrument(skip(self))]
    pub fn unload_model(&mut self, model_name: &str) -> Result<()> {
        // Drop from current thread first (while config still exists)
        self.drop_model_from_thread(model_name)?;

        // Then unregister from config
        self.unregister_model(model_name)?;

        info!("Model '{}' fully unloaded", model_name);
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
            let backend_guard = self.backend.lock().unwrap();
            let model = EmbeddingModel::new(&backend_guard, &model_config)?;
            drop(backend_guard); // Release lock as soon as we're done

            // Store in thread-local map
            models.insert(model_name.to_string(), model);

            info!(
                "Model '{}' loaded successfully in current thread",
                model_name
            );
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
            .map(std::string::ToString::to_string)
            .or_else(|| self.default_model.clone())
            .ok_or_else(|| Error::ConfigurationError {
                message: "No model specified and no default model set".to_string(),
            })?;

        // Ensure model is loaded in current thread
        self.ensure_model_loaded(&model_name)?;

        // Generate embedding using thread-local model
        THREAD_MODELS.with(|models| {
            let mut models = models.borrow_mut();
            let model = models
                .get_mut(&model_name)
                .ok_or_else(|| Error::ModelNotFound {
                    name: model_name.clone(),
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
    pub fn embed_batch(&self, model_name: Option<&str>, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Determine which model to use
        let model_name = model_name
            .map(std::string::ToString::to_string)
            .or_else(|| self.default_model.clone())
            .ok_or_else(|| Error::ConfigurationError {
                message: "No model specified and no default model set".to_string(),
            })?;

        // Ensure model is loaded in current thread
        self.ensure_model_loaded(&model_name)?;

        // Get model configuration for batch processing
        let config = self
            .model_configs
            .read()
            .get(&model_name)
            .ok_or_else(|| Error::ModelNotFound {
                name: model_name.clone(),
            })?
            .clone();

        // Create batch processor with model configuration
        let batch_processor = BatchProcessorBuilder::default()
            .with_max_batch_size(64) // Default batch size
            .with_normalization(config.normalize_embeddings)
            .with_pooling_strategy(config.pooling_strategy)
            .build();

        // Process batch using the BatchProcessor
        THREAD_MODELS.with(|models| {
            let mut models = models.borrow_mut();
            let model = models
                .get_mut(&model_name)
                .ok_or_else(|| Error::ModelNotFound {
                    name: model_name.clone(),
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

    /// Checks if a model is registered in the config registry.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to check
    ///
    /// # Returns
    ///
    /// Returns true if the model is registered, false otherwise.
    pub fn is_model_registered(&self, model_name: &str) -> bool {
        let configs = self.model_configs.read();
        configs.contains_key(model_name)
    }

    /// Checks if a model is loaded in the current thread.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to check
    ///
    /// # Returns
    ///
    /// Returns true if the model is loaded in the current thread, false otherwise.
    pub fn is_model_loaded_in_thread(&self, model_name: &str) -> bool {
        THREAD_MODELS.with(|models| {
            let models = models.borrow();
            models.contains_key(model_name)
        })
    }

    /// Checks if a model is loaded (deprecated, use `is_model_registered`).
    ///
    /// # Deprecated
    ///
    /// This method has been deprecated in favor of `is_model_registered` for clarity.
    /// The original name was misleading as it only checked registration, not actual loading.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to check
    ///
    /// # Returns
    ///
    /// Returns true if the model is registered, false otherwise.
    #[deprecated(since = "0.2.0", note = "Use is_model_registered instead for clarity")]
    pub fn is_model_loaded(&self, model_name: &str) -> bool {
        self.is_model_registered(model_name)
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
        if !self.is_model_registered(model_name) {
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
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not found
    pub fn model_info(&self, model_name: &str) -> Result<ModelInfo> {
        // Ensure model is loaded in current thread
        self.ensure_model_loaded(model_name)?;

        THREAD_MODELS.with(|models| {
            let models = models.borrow();
            let model = models.get(model_name).ok_or_else(|| Error::ModelNotFound {
                name: model_name.to_string(),
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

    /// Performs explicit cleanup of all models in the current thread.
    ///
    /// With global tracing subscriber, this method is now optional but can
    /// still be useful for explicit resource management in tests.
    pub fn cleanup_thread_models(&self) {
        THREAD_MODELS.with(|models| {
            let mut models = models.borrow_mut();

            // Clear all models from the thread
            let count = models.len();
            models.clear();

            if count > 0 {
                info!("Cleared {} thread-local models", count);
            }
        });
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
    /// Approximate model size in bytes (None if unable to calculate)
    pub model_size: Option<usize>,
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
    #[ignore = "Requires actual GGUF model file"]
    fn test_embedding_generation() {
        let config = create_test_config();
        let engine = EmbeddingEngine::new(config).unwrap();

        let text = "Hello, world!";
        let embedding = engine.embed(None, text).unwrap();

        assert!(!embedding.is_empty());
    }
}
