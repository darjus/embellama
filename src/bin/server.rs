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

use clap::Parser;
use embellama::server::{state::{AppState, ServerConfig}};
use std::net::SocketAddr;
use std::path::PathBuf;
use tokio::signal;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::{info, warn, error};
use uuid::Uuid;

/// Embellama server - OpenAI-compatible embedding API
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to GGUF model file
    #[arg(long, env = "EMBELLAMA_MODEL_PATH")]
    model_path: PathBuf,
    
    /// Model identifier for API responses
    #[arg(long, env = "EMBELLAMA_MODEL_NAME", default_value = "default")]
    model_name: String,
    
    /// Bind address
    #[arg(long, env = "EMBELLAMA_HOST", default_value = "127.0.0.1")]
    host: String,
    
    /// Server port
    #[arg(short, long, env = "EMBELLAMA_PORT", default_value_t = 8080)]
    port: u16,
    
    /// Number of worker threads
    #[arg(long, env = "EMBELLAMA_WORKERS", default_value_t = num_cpus::get())]
    workers: usize,
    
    /// Maximum pending requests per worker
    #[arg(long, env = "EMBELLAMA_QUEUE_SIZE", default_value_t = 100)]
    queue_size: usize,
    
    /// Log level (trace, debug, info, warn, error)
    #[arg(long, env = "EMBELLAMA_LOG_LEVEL", default_value = "info")]
    log_level: String,
}

impl Args {
    fn validate(&self) -> Result<(), String> {
        // Check if model path exists
        if !self.model_path.exists() {
            return Err(format!("Model file not found: {:?}", self.model_path));
        }
        
        // Validate workers count
        if self.workers == 0 {
            return Err("Worker count must be at least 1".to_string());
        }
        
        if self.workers > 128 {
            return Err("Worker count cannot exceed 128".to_string());
        }
        
        // Validate queue size
        if self.queue_size == 0 {
            return Err("Queue size must be at least 1".to_string());
        }
        
        if self.queue_size > 10000 {
            return Err("Queue size cannot exceed 10000".to_string());
        }
        
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args = Args::parse();
    
    // Initialize logging
    init_logging(&args.log_level)?;
    
    // Validate arguments
    args.validate()?;
    
    info!("Starting Embellama server v{}", env!("CARGO_PKG_VERSION"));
    info!("Model: {:?} ({})", args.model_path, args.model_name);
    info!("Workers: {}, Queue size: {}", args.workers, args.queue_size);
    
    // Create server configuration
    let config = ServerConfig {
        model_name: args.model_name,
        model_path: args.model_path.to_string_lossy().to_string(),
        worker_count: args.workers,
        queue_size: args.queue_size,
        host: args.host.clone(),
        port: args.port,
    };
    
    // Create application state
    let state = AppState::new(config.clone());
    
    // Build the router
    let app = build_router(state);
    
    // Create socket address
    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    info!("Server listening on http://{}", addr);
    
    // Create TCP listener
    let listener = tokio::net::TcpListener::bind(addr).await?;
    
    // Run server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    
    info!("Server shutdown complete");
    Ok(())
}

/// Build the Axum router with all routes and middleware
fn build_router(state: AppState) -> axum::Router {
    use axum::{
        extract::State,
        http::StatusCode,
        response::{IntoResponse, Json},
        routing::get,
        Router,
    };
    use serde_json::json;
    
    // Health check handler
    async fn health_handler(State(state): State<AppState>) -> impl IntoResponse {
        if state.is_ready() {
            (StatusCode::OK, Json(json!({
                "status": "healthy",
                "model": state.model_name(),
                "version": env!("CARGO_PKG_VERSION"),
            })))
        } else {
            (StatusCode::SERVICE_UNAVAILABLE, Json(json!({
                "status": "unhealthy",
                "error": "Service not ready",
            })))
        }
    }
    
    // Build router with routes and middleware
    Router::new()
        .route("/health", get(health_handler))
        // Phase 3: Add /v1/embeddings and other routes here
        .layer(
            tower::ServiceBuilder::new()
                // Add tracing/logging middleware
                .layer(
                    TraceLayer::new_for_http()
                        .make_span_with(|request: &axum::http::Request<_>| {
                            let request_id = Uuid::new_v4();
                            tracing::info_span!(
                                "http_request",
                                request_id = %request_id,
                                method = %request.method(),
                                uri = %request.uri(),
                            )
                        })
                )
                // Add CORS middleware
                .layer(CorsLayer::permissive())
        )
        .with_state(state)
}

/// Initialize logging with the specified level
fn init_logging(level: &str) -> Result<(), Box<dyn std::error::Error>> {
    use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};
    
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(level));
    
    tracing_subscriber::registry()
        .with(fmt::layer()
            .with_target(true)
            .with_thread_ids(true)
            .with_thread_names(true)
            .with_file(true)
            .with_line_number(true))
        .with(env_filter)
        .init();
    
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
        _ = ctrl_c => {
            info!("Received Ctrl+C, shutting down");
        }
        _ = terminate => {
            info!("Received terminate signal, shutting down");
        }
    }
}