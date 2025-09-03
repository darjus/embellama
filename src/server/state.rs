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
use std::sync::Arc;

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
        }
    }
}

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    /// Request dispatcher
    pub dispatcher: Arc<Dispatcher>,
    /// Server configuration
    pub config: Arc<ServerConfig>,
}

impl AppState {
    /// Create a new application state
    ///
    /// # Arguments
    /// * `config` - Server configuration
    ///
    /// # Returns
    /// A new `AppState` instance
    pub fn new(config: ServerConfig) -> Self {
        let dispatcher = Dispatcher::new(
            config.worker_count,
            config.queue_size,
            config.model_path.clone(),
        );
        
        Self {
            dispatcher: Arc::new(dispatcher),
            config: Arc::new(config),
        }
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