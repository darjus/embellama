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

// Create engine
let engine = EmbeddingEngine::new(config)?;

// Generate single embedding
let text = "Hello, world!";
let embedding = engine.embed(None, text)?;

// Generate batch embeddings
let texts = vec!["Text 1", "Text 2", "Text 3"];
let embeddings = engine.embed_batch(None, texts)?;
```

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
    .build()?;
```

## Pooling Strategies

- **Mean**: Average pooling across all tokens (default)
- **CLS**: Use the CLS token embedding
- **Max**: Maximum pooling across dimensions
- **MeanSqrt**: Mean pooling with square root of sequence length normalization

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

## Testing

### Unit Tests

```bash
cargo test
```

### Integration Tests

Set the test model environment variable:

```bash
EMBELLAMA_TEST_MODEL=/path/to/model.gguf cargo test --test integration_tests
```

### Environment Variables

- `EMBELLAMA_TEST_MODEL`: Path to a GGUF model file for integration tests
- `EMBELLAMA_BENCH_MODEL`: Path to a GGUF model file for benchmarks
- `EMBELLAMA_MODEL`: Path to a GGUF model file for examples

### Concurrency Tests

```bash
cargo test --test concurrency_tests
```

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