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

//! Server module for Embellama HTTP API
//!
//! This module provides an OpenAI-compatible REST API for the embedding engine.
//! It uses a worker pool architecture to handle the `!Send` constraint of LlamaContext.
//!
//! # Library Usage
//!
//! The server can be embedded into other applications as a library feature:
//!
//! ```no_run
//! use embellama::server::{AppState, ServerConfig, create_router, run_server};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Option 1: Use the convenient run_server function
//!     let config = ServerConfig::builder()
//!         .model_path("/path/to/model.gguf")
//!         .model_name("my-model")
//!         .port(8080)
//!         .build()?;
//!
//!     run_server(config.clone()).await?;
//!
//!     // Option 2: Create router for custom integration
//!     let state = AppState::new(config)?;
//!     let router = create_router(state);
//!     // Add your own routes or middleware...
//!
//!     Ok(())
//! }
//! ```

pub mod api_types;
pub mod cache_handlers;
pub mod channel;
pub mod dispatcher;
pub mod handlers;
pub mod middleware;
pub mod state;
pub mod worker;

// Phase 4 modules
#[cfg(feature = "server")]
pub mod backpressure;
#[cfg(feature = "server")]
pub mod metrics;
#[cfg(feature = "server")]
pub mod rate_limiter;

// Re-exports for convenience
pub use middleware::{
    API_KEY_HEADER, ApiKeyConfig, MAX_REQUEST_SIZE, REQUEST_ID_HEADER, authenticate_api_key,
    extract_request_id, inject_request_id, limit_request_size,
};
pub use state::{AppState, ServerConfig, ServerConfigBuilder};

use axum::{
    Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::get,
};
use gguf::{GGUFFile, GGUFMetadataValue};
use serde_json::json;
use std::net::SocketAddr;
use std::path::PathBuf;
use tokio::signal;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Model provider trait for custom model loading strategies
///
/// This trait allows users to implement custom model loading logic,
/// such as downloading models from cloud storage or managing multiple models.
#[async_trait::async_trait]
pub trait ModelProvider: Send + Sync {
    /// Get the path to a model file by name
    ///
    /// # Arguments
    /// * `model_name` - Name of the model to load
    ///
    /// # Returns
    /// Path to the model file or an error
    async fn get_model_path(&self, model_name: &str) -> crate::Result<PathBuf>;

    /// List available models
    ///
    /// # Returns
    /// List of available model information with metadata extracted from GGUF files,
    /// including actual embedding dimensions, max tokens, and file size
    async fn list_models(&self) -> crate::Result<Vec<crate::ModelInfo>>;
}

/// Default file-based model provider
///
/// This provider serves a single GGUF model file and extracts metadata
/// such as embedding dimensions and context length directly from the file.
pub struct FileModelProvider {
    model_path: PathBuf,
    model_name: String,
}

impl FileModelProvider {
    /// Create a new file-based model provider
    pub fn new(model_path: impl Into<PathBuf>, model_name: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            model_name: model_name.into(),
        }
    }

    /// Extract model metadata from a GGUF file
    ///
    /// This function reads only the header and metadata from the GGUF file
    /// without loading the entire model, making it efficient for listing models.
    ///
    /// # Returns
    /// A tuple of (dimensions, `max_tokens`) extracted from the metadata
    fn extract_gguf_metadata(path: &PathBuf) -> crate::Result<(usize, usize)> {
        use std::fs::File;
        use std::io::Read;

        // Read only the beginning of the file (usually metadata is at the start)
        let mut file = File::open(path).map_err(|e| crate::Error::ModelLoadError {
            path: path.clone(),
            source: anyhow::anyhow!("Failed to open GGUF file: {}", e),
        })?;

        // Read a reasonable amount for metadata (16MB should be more than enough)
        let mut buffer = Vec::new();
        let _ = file
            .by_ref()
            .take(16 * 1024 * 1024)
            .read_to_end(&mut buffer)
            .map_err(|e| crate::Error::ModelLoadError {
                path: path.clone(),
                source: anyhow::anyhow!("Failed to read GGUF file: {}", e),
            })?;

        // Parse GGUF file metadata
        let gguf_file = match GGUFFile::read(&buffer) {
            Ok(Some(file)) => file,
            Ok(None) => {
                return Err(crate::Error::ModelLoadError {
                    path: path.clone(),
                    source: anyhow::anyhow!("Incomplete GGUF file data"),
                });
            }
            Err(e) => {
                return Err(crate::Error::ModelLoadError {
                    path: path.clone(),
                    source: anyhow::anyhow!("Failed to parse GGUF file: {}", e),
                });
            }
        };

        let mut dimensions = 0usize;
        let mut max_tokens = 512usize; // Default fallback

        // Look for embedding dimensions and context length in metadata
        debug!("GGUF metadata count: {}", gguf_file.header.metadata.len());

        for metadata in &gguf_file.header.metadata {
            // Log all keys for debugging
            debug!(
                "GGUF metadata key: '{}' = {:?}",
                metadata.key, metadata.value
            );

            match metadata.key.as_str() {
                // Common keys for embedding dimensions
                "llama.embedding_length"
                | "embedding_length"
                | "n_embd"
                | "bert.embedding_length" => {
                    if let Some(value) = extract_usize_from_metadata(&metadata.value) {
                        dimensions = value;
                        debug!(
                            "Found embedding dimensions: {} from key: {}",
                            dimensions, metadata.key
                        );
                    }
                }
                // Common keys for context length
                "llama.context_length"
                | "context_length"
                | "n_ctx"
                | "max_position_embeddings"
                | "bert.context_length" => {
                    if let Some(value) = extract_usize_from_metadata(&metadata.value) {
                        max_tokens = value;
                        debug!(
                            "Found max tokens: {} from key: {}",
                            max_tokens, metadata.key
                        );
                    }
                }
                _ => {}
            }
        }

        // If dimensions not found in metadata, try to infer from tensor shapes
        if dimensions == 0 {
            // Look for embedding tensor dimensions
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

        Ok((dimensions, max_tokens))
    }
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

#[async_trait::async_trait]
impl ModelProvider for FileModelProvider {
    async fn get_model_path(&self, model_name: &str) -> crate::Result<PathBuf> {
        if model_name == self.model_name {
            Ok(self.model_path.clone())
        } else {
            Err(crate::Error::ModelNotFound {
                name: model_name.to_string(),
            })
        }
    }

    async fn list_models(&self) -> crate::Result<Vec<crate::ModelInfo>> {
        // Get file size
        let model_size = match std::fs::metadata(&self.model_path) {
            Ok(metadata) => Some(metadata.len()),
            Err(e) => {
                warn!(
                    "Failed to get file size for {}: {}",
                    self.model_path.display(),
                    e
                );
                None
            }
        };

        // Extract model metadata from GGUF file
        let (dimensions, max_tokens) = match Self::extract_gguf_metadata(&self.model_path) {
            Ok((dims, tokens)) => {
                debug!(
                    "Successfully extracted metadata: dimensions={}, max_tokens={}",
                    dims, tokens
                );
                (dims, tokens)
            }
            Err(e) => {
                warn!(
                    "Failed to extract GGUF metadata from {}: {}",
                    self.model_path.display(),
                    e
                );
                // Return default values if metadata extraction fails
                (0, 512)
            }
        };

        Ok(vec![crate::ModelInfo {
            name: self.model_name.clone(),
            dimensions,
            max_tokens,
            model_size: model_size.and_then(|s| s.try_into().ok()),
        }])
    }
}

/// Create a router with the default configuration
///
/// This function creates an Axum router with all the standard routes and middleware
/// configured. Users can add additional routes or middleware before starting the server.
///
/// # Arguments
/// * `state` - Application state containing the embedding engine and configuration
///
/// # Returns
/// Configured Axum router ready to be served
///
/// # Example
/// ```no_run
/// # use embellama::server::{AppState, ServerConfig, create_router};
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ServerConfig::builder()
///     .model_path("/path/to/model.gguf")
///     .build()?;
///
/// let state = AppState::new(config)?;
/// let router = create_router(state);
///
/// // Add custom routes
/// let router = router.route("/custom", axum::routing::get(|| async { "Custom route" }));
///
/// // Start server...
/// # Ok(())
/// # }
/// ```
pub fn create_router(state: AppState) -> Router<()> {
    // TODO: Consider adding commonly-needed middleware by default:
    // - inject_request_id: Adds request ID to all requests for tracing
    // - limit_request_size: Prevents excessive payload sizes
    // Users can still add these manually if needed
    // Health check handler
    async fn health_handler(State(state): State<AppState>) -> impl IntoResponse {
        if state.is_ready() {
            (
                StatusCode::OK,
                Json(json!({
                    "status": "healthy",
                    "model": state.model_name(),
                    "version": env!("CARGO_PKG_VERSION"),
                })),
            )
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "status": "unhealthy",
                    "error": "Service not ready",
                })),
            )
        }
    }

    // Build router with routes and middleware
    Router::new()
        .route("/health", get(health_handler))
        // OpenAI-compatible API routes
        .route(
            "/v1/embeddings",
            axum::routing::post(handlers::embeddings_handler),
        )
        .route("/v1/models", get(handlers::list_models_handler))
        // Cache management endpoints
        .route("/cache/stats", get(cache_handlers::cache_stats_handler))
        .route(
            "/cache/clear",
            axum::routing::post(cache_handlers::cache_clear_handler),
        )
        .route(
            "/cache/warm",
            axum::routing::post(cache_handlers::cache_warm_handler),
        )
        // Prefix cache endpoints
        .route(
            "/v1/embeddings/prefix",
            axum::routing::post(cache_handlers::prefix_register_handler),
        )
        .route(
            "/v1/embeddings/prefix",
            get(cache_handlers::prefix_list_handler),
        )
        .route(
            "/v1/embeddings/prefix",
            axum::routing::delete(cache_handlers::prefix_clear_handler),
        )
        .route(
            "/v1/embeddings/prefix/stats",
            get(cache_handlers::prefix_stats_handler),
        )
        .layer(
            tower::ServiceBuilder::new()
                // Add tracing/logging middleware
                .layer(TraceLayer::new_for_http().make_span_with(
                    |request: &axum::http::Request<_>| {
                        let request_id = Uuid::new_v4();
                        tracing::info_span!(
                            "http_request",
                            request_id = %request_id,
                            method = %request.method(),
                            uri = %request.uri(),
                        )
                    },
                ))
                // Add CORS middleware
                .layer(CorsLayer::permissive()),
        )
        .with_state(state)
}

/// Run the server with the provided configuration
///
/// This is a convenience function that creates the application state,
/// builds the router, and starts the server with graceful shutdown handling.
///
/// # Arguments
/// * `config` - Server configuration
///
/// # Returns
/// Result indicating success or failure
///
/// # Errors
///
/// Returns an error if:
/// - Application state creation fails
/// - Server binding fails
/// - Server startup fails
///
/// # Example
/// ```no_run
/// # use embellama::server::{ServerConfig, run_server};
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ServerConfig::builder()
///     .model_path("/path/to/model.gguf")
///     .model_name("my-model")
///     .host("0.0.0.0")
///     .port(8080)
///     .build()?;
///
/// run_server(config).await?;
/// # Ok(())
/// # }
/// ```
pub async fn run_server(config: ServerConfig) -> crate::Result<()> {
    info!("Starting Embellama server v{}", env!("CARGO_PKG_VERSION"));
    info!("Model: {} ({})", config.model_path, config.model_name);
    info!(
        "Workers: {}, Queue size: {}",
        config.worker_count, config.queue_size
    );

    // Create application state
    let state = AppState::new(config.clone())?;

    // Build the router
    let app = create_router(state);

    // Create socket address
    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .map_err(|e| crate::Error::Other(anyhow::anyhow!("Invalid address: {e}")))?;

    info!("Server listening on http://{}", addr);

    // Create TCP listener
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| crate::Error::Other(anyhow::anyhow!("Failed to bind to address: {}", e)))?;

    // Run server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| crate::Error::Other(anyhow::anyhow!("Server error: {}", e)))?;

    info!("Server shutdown complete");
    Ok(())
}

/// Signal handler for graceful shutdown
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {
            info!("Received Ctrl+C, shutting down");
        }
        () = terminate => {
            info!("Received terminate signal, shutting down");
        }
    }
}
