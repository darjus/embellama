# Embellama Library Implementation Plan

> **STATUS**: Phase 1 (Project Setup & Core Infrastructure) - ✅ COMPLETED

## Overview
Create a high-performance Rust library for generating text embeddings using llama-cpp-2, providing a robust and ergonomic API for Rust applications.

## Phase 1: Project Setup & Core Infrastructure

### 1.1 Initialize Project Structure
- [x] Set up `.gitignore` with Rust-specific patterns
- [x] Configure `Cargo.toml` with initial dependencies:
  - `llama-cpp-2 = "0.1.117"`
  - `thiserror` for error handling
  - `anyhow` for convenient error propagation
- [x] Set up feature flags in `Cargo.toml`:
  - `server` feature flag (for optional server binary)
- [x] Create initial module structure:
  - `src/lib.rs` - Main library entry point
  - `src/error.rs` - Custom error types
  - `src/config.rs` - Configuration structures

### 1.2 Development Environment Setup
- [x] Set up CI/CD pipeline (GitHub Actions)
- [x] Configure rustfmt and clippy for code quality
- [x] Add pre-commit hooks for formatting and linting
- [x] Set up logging with `tracing` crate
- [x] Create development documentation in DEVELOPMENT.md

> **NOTE**: Phase 1 completed successfully. All foundational infrastructure is in place:
> - Comprehensive error handling system with custom error types
> - Configuration module with builder pattern and validation
> - Full dependency setup including optional server dependencies
> - CI/CD pipeline with multi-OS testing, security audits, and coverage
> - Development documentation and contribution guidelines
> - Ready to proceed with Phase 2: Core Library Implementation

## Phase 2: Core Library Implementation

### 2.1 Error Handling System
- [ ] Define `Error` enum with variants:
  - `ModelLoadError`
  - `ModelNotFound`
  - `EmbeddingGenerationError`
  - `ConfigurationError`
  - `InvalidInput`
- [ ] Implement `std::error::Error` trait
- [ ] Add conversions from `llama-cpp-2` errors
- [ ] Create `Result<T>` type alias

### 2.2 Configuration Module
- [ ] Create `EngineConfig` struct with fields:
  - `model_path: PathBuf`
  - `model_name: String`
  - `context_size: Option<usize>`
  - `n_threads: Option<usize>`
  - `use_gpu: bool`
- [ ] Implement builder pattern for `EngineConfig`
- [ ] Add validation logic for configuration
- [ ] Support environment variable overrides

### 2.3 Model Management
- [ ] Create `EmbeddingModel` struct wrapping llama-cpp-2 types
- [ ] Implement model loading with proper error handling
- [ ] Add model metadata extraction (dimensions, max tokens)
- [ ] Implement model unloading and cleanup
- [ ] Add model warmup functionality

### 2.4 Embedding Engine
- [ ] Create `EmbeddingEngine` struct with:
  - Model registry using `HashMap<String, Arc<EmbeddingModel>>`
  - Thread-safe access with `RwLock`
- [ ] Implement single embedding generation:
  - Text tokenization
  - Context creation
  - Embedding extraction
  - Normalization options
- [ ] Add model lifecycle management:
  - `load_model()`
  - `unload_model()`
  - `list_models()`
  - `get_model_info()`

### 2.5 Batch Processing
- [ ] Create `BatchProcessor` for efficient batch operations
- [ ] Implement chunking strategy for large batches
- [ ] Add parallel processing with `rayon` crate for:
  - **Pre-processing** (parallel): Text tokenization, validation, normalization
  - **Embedding generation** (sequential): Single model instance, llama.cpp manages internal threading
  - **Post-processing** (parallel): Embedding normalization, response formatting
- [ ] Optimize memory usage for batch operations
- [ ] Implement progress tracking for large batches
- [ ] Design pipeline approach for overlapping CPU and model operations

## Phase 3: Testing & Quality Assurance

### 3.1 Unit Tests
- [ ] Test error handling in all modules
- [ ] Test configuration builder and validation
- [ ] Test embedding engine methods
- [ ] Test batch processing logic

### 3.2 Integration Tests
- [ ] Download test models:
  - `sentence-transformers/all-MiniLM-L6-v2` (GGUF)
  - `jinaai/jina-embeddings-v2-base-code` (GGUF)
- [ ] Test full embedding generation flow
- [ ] Test batch processing with various sizes
- [ ] Test model loading/unloading
- [ ] Test concurrent embedding generation

### 3.3 Performance Testing
- [ ] Benchmark single embedding generation
- [ ] Benchmark batch processing throughput
- [ ] Memory usage profiling
- [ ] GPU utilization testing
- [ ] Latency distribution analysis

## Phase 4: Documentation & Examples

### 4.1 Library Documentation
- [ ] Generate rustdoc documentation
- [ ] Write usage examples for library
- [ ] Document configuration options
- [ ] Add troubleshooting guide

### 4.2 Examples & Tutorials
- [ ] Basic library usage example
- [ ] Batch processing example
- [ ] Performance tuning guide

## Phase 5: Optimization & Production Readiness

### 5.1 Performance Optimization
- [ ] Profile and optimize hot paths
- [ ] Implement caching for repeated inputs
- [ ] Optimize tokenization process
- [ ] Add memory pooling for allocations
- [ ] Tune batch size recommendations

### 5.2 Library Hardening
- [ ] Add input validation
- [ ] Implement memory limits
- [ ] Security audit with `cargo audit`
- [ ] Fuzz testing for edge cases
- [ ] Thread safety verification

## Phase 6: Release & Maintenance

### 6.1 Release Preparation
- [ ] Version tagging strategy (SemVer)
- [ ] Create CHANGELOG.md
- [ ] Prepare crates.io publication
- [ ] Create GitHub releases
- [ ] Binary distribution setup

### 6.2 Community & Support
- [ ] Set up issue templates
- [ ] Create contribution guidelines
- [ ] Add code of conduct
- [ ] Set up discussions/forum
- [ ] Create roadmap document

## Key Dependencies Summary

```toml
[dependencies]
llama-cpp-2 = "0.1.117"
thiserror = "1.0"
anyhow = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
rayon = "1.8"  # For parallel batch processing

[dev-dependencies]
criterion = "0.5"  # For benchmarking
serial_test = "3.0"  # For sequential test execution
tempfile = "3.8"  # For test file handling
```

## Risk Mitigation

1. **llama-cpp-2 API stability**: Pin version, monitor updates
2. **Memory management**: Implement proper cleanup, add monitoring
3. **Performance bottlenecks**: Profile early, benchmark regularly
4. **GGUF format changes**: Abstract model loading interface
5. **GPU compatibility**: Test on various hardware, provide CPU fallback

## Success Metrics

- [ ] < 50ms latency for single embeddings (CPU)
- [ ] > 1000 embeddings/sec batch throughput
- [ ] < 500MB memory baseline
- [ ] Comprehensive test coverage (> 80%)

## Implementation Notes

### Technology Insights from Research

#### llama-cpp-2 Crate (v0.1.117)
- Actively maintained with semi-regular updates syncing with llama.cpp
- Provides safe wrappers around nearly direct bindings
- Not fully idiomatic Rust API to maintain compatibility with upstream
- Supports embedding generation through llama.cpp backend

#### Library Design Principles
- Zero-copy where possible for performance
- Builder pattern for configuration
- Thread-safe by default with Arc/RwLock
- Graceful error handling with Result types
- Minimal dependencies for lighter footprint

#### GGUF Format & Testing
- GGUF is the required format for llama.cpp models
- Supports multiple quantization levels (Q4_K_M recommended for balance)
- Test with common models like all-MiniLM-L6-v2 and jina-embeddings-v2
- Use llama-bench for performance benchmarking
- Docker containers available for model conversion

### Parallel Processing Architecture

#### What Gets Parallelized (using Rayon)
- **Pre-processing stage**: Tokenization, input validation, text cleaning/normalization
- **Post-processing stage**: Embedding normalization, response formatting, result aggregation

#### What Remains Sequential  
- **Model inference**: Single model instance in memory, llama.cpp handles its own internal threading via `n_threads`
- **GPU operations**: Already optimized internally by llama.cpp

#### Implementation Strategy
```rust
// Conceptual batch processing flow
BatchProcessor {
    // Step 1: Parallel pre-processing with rayon
    tokenized_inputs = texts.par_iter().map(tokenize).collect()
    
    // Step 2: Sequential model inference (llama.cpp internal threading)
    embeddings = tokenized_inputs.iter().map(|tokens| 
        model.generate_embedding(tokens)  // Uses n_threads internally
    ).collect()
    
    // Step 3: Parallel post-processing with rayon  
    normalized = embeddings.par_iter().map(normalize).collect()
}
```

This approach maximizes CPU utilization for I/O-bound operations while respecting the single model instance constraint and leveraging llama.cpp's optimized inference engine.

### Development Workflow

1. **Start with Phase 1** - Set up the project foundation
2. **Iterate on Phase 2** - Build core library functionality
3. **Test continuously** - Write tests alongside implementation
4. **Document as you go** - Keep documentation current
5. **Optimize last** - Profile and optimize after functionality is complete

This plan provides a systematic approach to building a production-ready embedding library with clear phases and checkpoints.

## Library API Example

```rust
use embellama::{EmbeddingEngine, EngineConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure and build the engine
    let config = EngineConfig::builder()
        .with_model_path("models/all-MiniLM-L6-v2.gguf")
        .with_model_name("minilm")
        .with_context_size(512)
        .with_n_threads(4)
        .build()?;

    let engine = EmbeddingEngine::new(config)?;

    // Generate a single embedding
    let embedding = engine.embed("minilm", "Hello, world!")?;
    println!("Embedding dimension: {}", embedding.len());

    // Generate batch embeddings
    let texts = vec![
        "First document",
        "Second document",
        "Third document",
    ];
    let embeddings = engine.embed_batch("minilm", texts)?;
    println!("Generated {} embeddings", embeddings.len());

    Ok(())
}
```