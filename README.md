# Embellama

[![Crates.io](https://img.shields.io/crates/v/embellama.svg)](https://crates.io/crates/embellama)
[![Documentation](https://docs.rs/embellama/badge.svg)](https://docs.rs/embellama)
[![License](https://img.shields.io/crates/l/embellama.svg)](https://github.com/darjus/embellama/blob/master/LICENSE)
[![CI](https://github.com/darjus/embellama/actions/workflows/ci.yml/badge.svg)](https://github.com/darjus/embellama/actions/workflows/ci.yml)

High-performance Rust library for generating text embeddings using llama-cpp.

## Features

- **High Performance**: Optimized for speed with parallel pre/post-processing
- **Thread Safety**: Compile-time guarantees for safe concurrent usage
- **Multiple Models**: Support for managing multiple embedding models
- **Batch Processing**: Efficient batch embedding generation
- **Flexible Configuration**: Extensive configuration options for model tuning
- **Multiple Pooling Strategies**: Mean, CLS, Max, and MeanSqrt pooling
- **Hardware Acceleration**: Support for Metal (macOS), CUDA (NVIDIA), Vulkan, and optimized CPU backends

## Quick Start

```rust
use embellama::{EmbeddingEngine, EngineConfig};

// Create configuration
let config = EngineConfig::builder()
    .with_model_path("/path/to/model.gguf")
    .with_model_name("my-model")
    .with_normalize_embeddings(true)
    .build()?;

// Create engine (uses singleton pattern internally)
let engine = EmbeddingEngine::new(config)?;

// Generate single embedding
let text = "Hello, world!";
let embedding = engine.embed(None, text)?;

// Generate batch embeddings
let texts = vec!["Text 1", "Text 2", "Text 3"];
let embeddings = engine.embed_batch(None, texts)?;
```

### Singleton Pattern (Advanced)

The engine can optionally use a singleton pattern for shared access across your application. The singleton methods return `Arc<Mutex<EmbeddingEngine>>` for thread-safe access:

```rust
// Get or initialize singleton instance (returns Arc<Mutex<EmbeddingEngine>>)
let engine = EmbeddingEngine::get_or_init(config)?;

// Access the singleton from anywhere in your application
let engine_clone = EmbeddingEngine::instance()
    .expect("Engine not initialized");

// Use the engine (requires locking the mutex)
let embedding = {
    let engine_guard = engine.lock().unwrap();
    engine_guard.embed(None, "text")?
};
```

## Tested Models

The library has been tested with the following GGUF models:
- **MiniLM-L6-v2** (Q4_K_M): ~15MB, 384-dimensional embeddings - used for integration tests
- **Jina Embeddings v2 Base Code** (Q4_K_M): ~110MB, 768-dimensional embeddings - used for benchmarks

Both BERT-style and LLaMA-style embedding models are supported.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
embellama = "0.4.0"
```

### Backend Features

The library supports multiple backends for hardware acceleration. By default, it uses OpenMP for CPU parallelization. You can enable specific backends based on your hardware:

```toml
# Default - OpenMP CPU parallelization
embellama = "0.4.0"

# macOS Metal GPU acceleration
embellama = { version = "0.4.0", features = ["metal"] }

# NVIDIA CUDA GPU acceleration
embellama = { version = "0.4.0", features = ["cuda"] }

# Vulkan GPU acceleration (cross-platform)
embellama = { version = "0.4.0", features = ["vulkan"] }

# Native CPU optimizations
embellama = { version = "0.4.0", features = ["native"] }

# CPU-optimized build (native + OpenMP)
embellama = { version = "0.4.0", features = ["cpu-optimized"] }
```

**Note**: GPU backends (Metal, CUDA, Vulkan) are mutually exclusive. Use only one at a time for best results.

## Configuration

### Basic Configuration

```rust
let config = EngineConfig::builder()
    .with_model_path("/path/to/model.gguf")
    .with_model_name("my-model")
    .build()?;
```

### Advanced Configuration

```rust
let config = EngineConfig::builder()
    .with_model_path("/path/to/model.gguf")
    .with_model_name("my-model")
    .with_context_size(2048)           // Model context window (usize)
    .with_n_threads(8)                 // CPU threads (usize)
    .with_use_gpu(true)                // Enable GPU acceleration
    .with_n_gpu_layers(32)             // Layers to offload to GPU (u32)
    .with_batch_size(64)               // Batch processing size (usize)
    .with_normalize_embeddings(true)   // L2 normalize embeddings
    .with_pooling_strategy(PoolingStrategy::Mean)  // Pooling method
    .with_add_bos_token(Some(false))   // Disable BOS for encoder models (Option<bool>)
    .build()?;
```

### Backend Auto-Detection

The library can automatically detect and use the best available backend:

```rust
use embellama::{EngineConfig, detect_best_backend, BackendInfo};

// Automatic backend detection
let config = EngineConfig::with_backend_detection()
    .with_model_path("/path/to/model.gguf")
    .with_model_name("my-model")
    .build()?;

// Check which backend was selected
let backend_info = BackendInfo::new();
println!("Using backend: {}", backend_info.backend);
println!("Available features: {:?}", backend_info.available_features);
```

## Pooling Strategies

- **Mean**: Average pooling across all tokens (default)
- **CLS**: Use the CLS token embedding
- **Max**: Maximum pooling across dimensions
- **MeanSqrt**: Mean pooling with square root of sequence length normalization

## Model-Specific Configuration

### BOS Token Handling

The library automatically detects model types and applies appropriate BOS token handling:

**Encoder Models** (BERT, E5, BGE, GTE, MiniLM, etc.):
- BOS token is **not** added (these models use CLS/SEP tokens)
- Auto-detected by model name patterns

**Decoder Models** (LLaMA, Mistral, Vicuna, etc.):
- BOS token **is** added (standard for autoregressive models)
- Default behavior for unknown models

**Manual Override**:
```rust
// Force disable BOS for a specific model
let config = EngineConfig::builder()
    .with_model_path("/path/to/model.gguf")
    .with_model_name("custom-encoder")
    .with_add_bos_token(Some(false))  // Explicitly disable BOS
    .build()?;

// Force enable BOS
let config = EngineConfig::builder()
    .with_model_path("/path/to/model.gguf")
    .with_model_name("custom-decoder")
    .with_add_bos_token(Some(true))   // Explicitly enable BOS
    .build()?;

// Auto-detect (default)
let config = EngineConfig::builder()
    .with_model_path("/path/to/model.gguf")
    .with_model_name("some-model")
    .with_add_bos_token(None)         // Let the library decide
    .build()?;
```

## Thread Safety

**⚠️ IMPORTANT**: The `LlamaContext` from llama-cpp is `!Send` and `!Sync`, which means:

- Models cannot be moved between threads
- Models cannot be shared using `Arc` alone
- Each thread must own its model instance
- All concurrency must use message passing

The library is designed with these constraints in mind:

- Models are `!Send` due to llama-cpp constraints
- Use thread-local storage for model instances
- Batch processing uses parallel pre/post-processing with sequential inference

Example of thread-safe usage with regular (non-singleton) engine:

```rust
use std::thread;

// Each thread needs its own engine instance due to llama-cpp constraints
let handles: Vec<_> = (0..4)
    .map(|i| {
        let config = config.clone(); // Clone config for each thread
        thread::spawn(move || {
            // Create engine instance in each thread
            let engine = EmbeddingEngine::new(config)?;
            let text = format!("Thread {} text", i);
            let embedding = engine.embed(None, &text)?;
            Ok::<_, embellama::Error>(embedding)
        })
    })
    .collect();

for handle in handles {
    let embedding = handle.join().unwrap()?;
    // Process embedding
}
```

Or using the singleton pattern for shared access:

```rust
use std::thread;

// Initialize singleton once
let engine = EmbeddingEngine::get_or_init(config)?;

let handles: Vec<_> = (0..4)
    .map(|i| {
        let engine = engine.clone(); // Clone Arc<Mutex<>>
        thread::spawn(move || {
            let text = format!("Thread {} text", i);
            let embedding = {
                let engine_guard = engine.lock().unwrap();
                engine_guard.embed(None, &text)?
            };
            Ok::<_, embellama::Error>(embedding)
        })
    })
    .collect();

for handle in handles {
    let embedding = handle.join().unwrap()?;
    // Process embedding
}
```

## API Reference

### Model Management

The library provides granular control over model lifecycle:

#### Registration vs Loading

- **Registration**: Model configuration stored in registry
- **Loading**: Model actually loaded in thread-local memory

```rust
// Check if model is registered (has configuration)
if engine.is_model_registered("my-model") {
    println!("Model configuration exists");
}

// Check if model is loaded in current thread
if engine.is_model_loaded_in_thread("my-model") {
    println!("Model is ready to use in this thread");
}

// Deprecated - use is_model_registered() for clarity
#[deprecated]
engine.is_model_loaded("my-model");  // Same as is_model_registered()
```

#### Granular Unload Operations

```rust
// Remove only from current thread (keeps registration)
engine.drop_model_from_thread("my-model")?;
// Model can be reloaded on next use

// Remove only from registry (prevents future loads)
engine.unregister_model("my-model")?;
// Existing thread-local instances continue working

// Full unload - removes from both registry and thread
engine.unload_model("my-model")?;
// Completely removes the model
```

### Model Loading Behavior

- **Initial model** (via `EmbeddingEngine::new()`): Loaded immediately in current thread
- **Additional models** (via `load_model()`): Lazy-loaded on first use

```rust
// First model - loaded immediately
let engine = EmbeddingEngine::new(config)?;
assert!(engine.is_model_loaded_in_thread("model1"));

// Additional model - lazy loaded
engine.load_model(config2)?;
assert!(engine.is_model_registered("model2"));
assert!(!engine.is_model_loaded_in_thread("model2"));  // Not yet loaded

// Triggers actual loading in thread
engine.embed(Some("model2"), "text")?;
assert!(engine.is_model_loaded_in_thread("model2"));  // Now loaded
```

## Performance

The library is optimized for high performance:

- Parallel tokenization for batch processing
- Efficient memory management
- Configurable thread counts
- GPU acceleration support

### Benchmarks

Run benchmarks with:

```bash
EMBELLAMA_BENCH_MODEL=/path/to/model.gguf cargo bench
```

### Performance Tips

1. **Batch Processing**: Use `embed_batch()` for multiple texts
2. **Thread Configuration**: Set `n_threads` based on CPU cores
3. **GPU Acceleration**: Enable GPU for larger models
4. **Warmup**: Call `warmup_model()` before processing

## Development

For development setup, testing, and contributing guidelines, please see [DEVELOPMENT.md](DEVELOPMENT.md).

## Examples

See the `examples/` directory for more examples:

- `simple.rs` - Basic embedding generation
- `batch.rs` - Batch processing example
- `multi_model.rs` - Using multiple models
- `config.rs` - Configuration examples
- `error_handling.rs` - Error handling patterns

Run examples with:

```bash
cargo run --example simple
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [DEVELOPMENT.md](DEVELOPMENT.md) for development setup and contribution guidelines.

## Support

For issues and questions, please use the GitHub issue tracker.
