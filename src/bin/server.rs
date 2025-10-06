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
use embellama::server::ServerConfig;
use std::path::PathBuf;
use tracing::info;

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

    /// Request timeout in seconds
    #[arg(long, env = "EMBELLAMA_REQUEST_TIMEOUT", default_value_t = 60)]
    request_timeout: u64,

    /// Maximum number of sequences to process in parallel (`n_seq_max`)
    #[arg(long, env = "EMBELLAMA_N_SEQ_MAX", default_value_t = 8)]
    n_seq_max: u32,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, env = "EMBELLAMA_LOG_LEVEL", default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args = Args::parse();

    // Initialize logging
    init_logging(&args.log_level);

    info!("Starting Embellama server v{}", env!("CARGO_PKG_VERSION"));
    info!("Model: {:?} ({})", args.model_path, args.model_name);
    info!("Workers: {}, Queue size: {}", args.workers, args.queue_size);
    info!(
        "Request timeout: {}s, n_seq_max: {}",
        args.request_timeout, args.n_seq_max
    );

    // Create server configuration using the library API
    let config = ServerConfig::builder()
        .model_name(args.model_name)
        .model_path(args.model_path.to_string_lossy().to_string())
        .worker_count(args.workers)
        .queue_size(args.queue_size)
        .host(args.host)
        .port(args.port)
        .request_timeout(std::time::Duration::from_secs(args.request_timeout))
        .n_seq_max(args.n_seq_max)
        .build()?;

    // Use the library's run_server function
    embellama::server::run_server(config).await?;

    Ok(())
}

/// Initialize logging with the specified level
fn init_logging(level: &str) {
    use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));

    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .with_target(true)
                .with_thread_ids(true)
                .with_thread_names(true)
                .with_file(true)
                .with_line_number(true),
        )
        .with(env_filter)
        .init();
}
