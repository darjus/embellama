# Embellama

High-performance Rust library for generating text embeddings using llama-cpp.

## Features

- **High Performance**: Optimized for speed with parallel pre/post-processing
- **Thread Safety**: Compile-time guarantees for safe concurrent usage
- **Multiple Models**: Support for managing multiple embedding models
- **Batch Processing**: Efficient batch embedding generation
- **Flexible Configuration**: Extensive configuration options for model tuning
- **Multiple Pooling Strategies**: Mean, CLS, Max, and MeanSqrt pooling

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

The engine can optionally use a singleton pattern for shared access:

```rust
use std::sync::{Arc, Mutex};

// Get or initialize singleton instance
let engine = EmbeddingEngine::get_or_init(config)?;

// Access from multiple places
let engine_clone = EmbeddingEngine::instance()
    .expect("Engine not initialized");
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
embellama = "0.1.0"
```

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

Example of thread-safe usage:

```rust
use std::sync::Arc;
use std::thread;

let engine = Arc::new(EmbeddingEngine::new(config)?);

let handles: Vec<_> = (0..4)
    .map(|i| {
        let engine = engine.clone();
        thread::spawn(move || {
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

This project uses [just](https://github.com/casey/just) for task automation. 

### Available Commands

```bash
just               # Show all available commands
just test          # Run all tests (unit + integration + concurrency)
just test-unit     # Run unit tests only
just test-integration # Run integration tests with real model
just test-concurrency # Run concurrency tests
just bench         # Run full benchmarks
just bench-quick   # Run quick benchmark subset
just dev           # Run fix, fmt, clippy, and unit tests
just pre-commit    # Run all checks before committing
just clean-all     # Clean build artifacts and models
```

### Testing

The project includes comprehensive test suites:

#### Unit Tests
```bash
just test-unit
```

#### Integration Tests
Tests with real GGUF models (downloads MiniLM automatically):
```bash
just test-integration
```

#### Concurrency Tests
Tests thread safety and parallel processing:
```bash
just test-concurrency
```

#### Testing Considerations

**Important**: Integration tests use the `serial_test` crate to ensure tests run sequentially. This is necessary because:
- The `LlamaBackend` can only be initialized once per process
- Each `EmbeddingEngine` owns its backend instance
- Tests must run serially to avoid backend initialization conflicts

When writing tests that create multiple engines, use a single engine with `load_model()` for different configurations:

```rust
#[test]
#[serial]  // Required for all integration tests
fn test_multiple_configurations() {
    let mut engine = EmbeddingEngine::new(initial_config)?;
    
    // Load additional models instead of creating new engines
    engine.load_model(config2)?;
    engine.load_model(config3)?;
}
```

### Model Management

Test models are automatically downloaded and cached:
- **Test model** (MiniLM-L6-v2): ~15MB, for integration tests
- **Benchmark model** (Jina Embeddings v2): ~110MB, for performance testing

```bash
just download-test-model   # Download test model
just download-bench-model  # Download benchmark model
just models-status        # Check cached models
```

### Environment Variables

- `EMBELLAMA_TEST_MODEL`: Path to test model (auto-set by justfile)
- `EMBELLAMA_BENCH_MODEL`: Path to benchmark model (auto-set by justfile)
- `EMBELLAMA_MODEL`: Path to model for examples

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

## Architecture

### Backend and Engine Management

The library manages the LlamaBackend lifecycle:

- Each `EmbeddingEngine` owns its `LlamaBackend` instance
- Backend is initialized when the engine is created
- Backend is dropped when the engine is dropped
- Singleton pattern available for shared engine access

### Model Management

The library uses a thread-local architecture due to llama-cpp's `!Send` constraint:

- Each thread maintains its own model instance
- Models cannot be shared between threads
- Use message passing for concurrent operations

### Batch Processing Pipeline

1. **Parallel Pre-processing**: Tokenization in parallel using Rayon
2. **Sequential Inference**: Model inference on single thread
3. **Parallel Post-processing**: Normalization and formatting in parallel

## Error Handling

The library provides comprehensive error handling:

```rust
use embellama::Error;

match engine.embed(None, text) {
    Ok(embedding) => process_embedding(embedding),
    Err(Error::ModelNotFound { name }) => {
        println!("Model {} not found", name);
    }
    Err(Error::InvalidInput { message }) => {
        println!("Invalid input: {}", message);
    }
    Err(e) if e.is_retryable() => {
        // Retry logic for transient errors
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## Support

For issues and questions, please use the GitHub issue tracker.