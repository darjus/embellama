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

//! Application state for the HTTP server
//!
//! This module defines the shared state that is passed to HTTP handlers.
//! The state must be Send + Sync + Clone for use with Axum.

use crate::server::dispatcher::Dispatcher;
use crate::{
    EmbeddingEngine, EngineConfig, ModelConfig, NormalizationMode, PoolingStrategy,
    extract_gguf_metadata,
};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Model name for API responses
    pub model_name: String,
    /// Path to the GGUF model file
    pub model_path: String,
    /// Number of worker threads
    pub worker_count: usize,
    /// Maximum pending requests per worker
    pub queue_size: usize,
    /// Server host address
    pub host: String,
    /// Server port
    pub port: u16,
    /// Request timeout duration
    pub request_timeout: std::time::Duration,
    /// Maximum number of sequences to process in parallel (`n_seq_max`)
    pub n_seq_max: u32,
    /// Pooling strategy for embeddings (None = use default)
    pub pooling_strategy: Option<PoolingStrategy>,
    /// Normalization mode for embeddings (None = use default L2)
    pub normalization_mode: Option<NormalizationMode>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            model_name: "default".to_string(),
            model_path: String::new(),
            worker_count: num_cpus::get(),
            queue_size: 100,
            host: "127.0.0.1".to_string(),
            port: 8080,
            request_timeout: std::time::Duration::from_secs(60),
            n_seq_max: 8,
            pooling_strategy: None,
            normalization_mode: None,
        }
    }
}

impl ServerConfig {
    /// Create a new builder for server configuration
    pub fn builder() -> ServerConfigBuilder {
        ServerConfigBuilder::default()
    }
}

/// Builder for `ServerConfig`
#[derive(Debug, Default)]
pub struct ServerConfigBuilder {
    model_name: Option<String>,
    model_path: Option<String>,
    worker_count: Option<usize>,
    queue_size: Option<usize>,
    host: Option<String>,
    port: Option<u16>,
    request_timeout: Option<std::time::Duration>,
    n_seq_max: Option<u32>,
    pooling_strategy: Option<PoolingStrategy>,
    normalization_mode: Option<NormalizationMode>,
}

impl ServerConfigBuilder {
    /// Set the model name
    #[must_use]
    pub fn model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = Some(name.into());
        self
    }

    /// Set the model path
    #[must_use]
    pub fn model_path(mut self, path: impl Into<String>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Set the number of worker threads
    #[must_use]
    pub fn worker_count(mut self, count: usize) -> Self {
        self.worker_count = Some(count);
        self
    }

    /// Set the queue size per worker
    #[must_use]
    pub fn queue_size(mut self, size: usize) -> Self {
        self.queue_size = Some(size);
        self
    }

    /// Set the server host address
    #[must_use]
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = Some(host.into());
        self
    }

    /// Set the server port
    #[must_use]
    pub fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    /// Set the request timeout duration
    #[must_use]
    pub fn request_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.request_timeout = Some(timeout);
        self
    }

    /// Set the maximum number of sequences to process in parallel (`n_seq_max`)
    #[must_use]
    pub fn n_seq_max(mut self, n_seq_max: u32) -> Self {
        self.n_seq_max = Some(n_seq_max);
        self
    }

    /// Set the pooling strategy for embeddings
    #[must_use]
    pub fn pooling_strategy(mut self, strategy: PoolingStrategy) -> Self {
        self.pooling_strategy = Some(strategy);
        self
    }

    /// Set the normalization mode for embeddings
    #[must_use]
    pub fn normalization_mode(mut self, mode: NormalizationMode) -> Self {
        self.normalization_mode = Some(mode);
        self
    }

    /// Build the `ServerConfig`
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The `model_path` is not provided
    /// - The model file does not exist
    /// - Worker count is 0 or greater than 128
    /// - Queue size is 0 or greater than 10000
    pub fn build(self) -> crate::Result<ServerConfig> {
        let default = ServerConfig::default();

        let model_path = self
            .model_path
            .ok_or_else(|| crate::Error::ConfigurationError {
                message: "Model path is required".to_string(),
            })?;

        // Validate model file exists
        if !std::path::Path::new(&model_path).exists() {
            return Err(crate::Error::ConfigurationError {
                message: format!("Model file not found: {model_path}"),
            });
        }

        let worker_count = self.worker_count.unwrap_or(default.worker_count);
        let queue_size = self.queue_size.unwrap_or(default.queue_size);

        // Validate worker count
        if worker_count == 0 {
            return Err(crate::Error::ConfigurationError {
                message: "Worker count must be at least 1".to_string(),
            });
        }
        if worker_count > 128 {
            return Err(crate::Error::ConfigurationError {
                message: "Worker count cannot exceed 128".to_string(),
            });
        }

        // Validate queue size
        if queue_size == 0 {
            return Err(crate::Error::ConfigurationError {
                message: "Queue size must be at least 1".to_string(),
            });
        }
        if queue_size > 10000 {
            return Err(crate::Error::ConfigurationError {
                message: "Queue size cannot exceed 10000".to_string(),
            });
        }

        Ok(ServerConfig {
            model_name: self.model_name.unwrap_or(default.model_name),
            model_path,
            worker_count,
            queue_size,
            host: self.host.unwrap_or(default.host),
            port: self.port.unwrap_or(default.port),
            request_timeout: self.request_timeout.unwrap_or(default.request_timeout),
            n_seq_max: self.n_seq_max.unwrap_or(default.n_seq_max),
            pooling_strategy: self.pooling_strategy,
            normalization_mode: self.normalization_mode,
        })
    }
}

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    /// Request dispatcher
    pub dispatcher: Arc<Dispatcher>,
    /// Server configuration
    pub config: Arc<ServerConfig>,
    /// Embedding engine instance
    pub engine: Arc<Mutex<EmbeddingEngine>>,
}

impl AppState {
    /// Create a new application state
    ///
    /// # Arguments
    /// * `config` - Server configuration
    ///
    /// # Returns
    /// A new `AppState` instance or an error
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Engine configuration validation fails
    /// - Engine initialization fails
    pub fn new(config: ServerConfig) -> crate::Result<Self> {
        // Extract context_size from GGUF metadata before engine initialization
        let context_size = match extract_gguf_metadata(Path::new(&config.model_path)) {
            Ok(metadata) => {
                info!(
                    "Auto-detected context size from GGUF: {}",
                    metadata.context_size
                );
                // Convert usize to u32, with fallback
                u32::try_from(metadata.context_size).ok()
            }
            Err(e) => {
                warn!("Could not extract GGUF metadata: {}", e);
                None
            }
        };

        // Build the model configuration first
        let mut model_config_builder = ModelConfig::builder()
            .with_model_path(&config.model_path)
            .with_model_name(&config.model_name)
            .with_n_seq_max(config.n_seq_max);

        // Add context_size if we extracted it
        if let Some(size) = context_size {
            model_config_builder = model_config_builder.with_context_size(size);
        }

        // Add pooling_strategy if specified
        if let Some(strategy) = config.pooling_strategy {
            model_config_builder = model_config_builder.with_pooling_strategy(strategy);
        }

        // Add normalization_mode if specified, otherwise use default L2
        let normalization_mode = config.normalization_mode.unwrap_or(NormalizationMode::L2);
        model_config_builder = model_config_builder.with_normalization_mode(normalization_mode);

        let model_config = model_config_builder.build()?;

        // Build the engine configuration using the model configuration
        let engine_config = EngineConfig::builder()
            .with_model_config(model_config)
            .build()?;

        let engine = EmbeddingEngine::get_or_init(engine_config)?;

        // Create the dispatcher with worker pool
        let dispatcher = Dispatcher::new(config.worker_count, config.queue_size);

        Ok(Self {
            dispatcher: Arc::new(dispatcher),
            config: Arc::new(config),
            engine,
        })
    }

    /// Get the model name
    pub fn model_name(&self) -> &str {
        &self.config.model_name
    }

    /// Check if the server is ready
    pub fn is_ready(&self) -> bool {
        self.dispatcher.is_ready()
    }
}
