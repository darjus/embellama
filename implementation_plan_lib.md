# Implementation Plan: Embellama Core Library

This document outlines the phased implementation plan for the `embellama` core library component. Each phase is designed to be independently implementable with clear deliverables and success criteria.

## Critical Design Constraint

**⚠️ IMPORTANT**: The `LlamaContext` from `llama-cpp-2` is `!Send` and `!Sync`. This means:
- Context cannot be moved between threads
- Context cannot be shared using `Arc`
- Each thread must own its model instance
- All concurrency must use message passing

## Phase 1: Project Setup & Core Infrastructure
**Priority: CRITICAL** | **Estimated Time: 2-4 hours**

### Objectives
Establish the foundational project structure, dependencies, and core error handling infrastructure.

### Tasks
- [x] Initialize Cargo project with appropriate metadata
  - [x] Set package name, version, authors, description
  - [x] Configure license (Apache-2.0)
  - [x] Add repository and documentation links
  
- [x] Configure dependencies in `Cargo.toml`
  - [x] Add `llama-cpp-2 = "0.1.117"` for model backend
  - [x] Add `thiserror = "1.0"` for error handling
  - [x] Add `anyhow = "1.0"` for application errors
  - [x] Add `tracing = "0.1"` for structured logging
  - [x] Add `serde = { version = "1.0", features = ["derive"] }` for serialization
  - [x] Add `rayon = "1.8"` for parallel batch processing
  
- [x] Create module structure
  - [x] Create `src/lib.rs` with module declarations
  - [x] Create `src/error.rs` for error types
  - [x] Create `src/config.rs` for configuration types
  - [x] Create placeholder files for other modules
  
- [x] Implement error handling foundation (`src/error.rs`)
  - [x] Define `EmbellamaError` enum with `thiserror`
  - [x] Add variants for common error cases:
    - [x] `ModelNotFound(String)`
    - [x] `ModelLoadFailed { path: String, source: llama_cpp_2::Error }`
    - [x] `EmbeddingFailed`
    - [x] `InvalidInput(String)`
    - [x] `ConfigurationError(String)`
  - [x] Implement `From` conversions for underlying errors
  
- [x] Set up logging infrastructure
  - [x] Configure `tracing` subscriber in examples/tests
  - [x] Define logging macros/utilities if needed
  - [x] Add instrumentation attributes to key functions
  
- [x] Add license headers to all source files
  - [x] Create script or template for Apache 2.0 headers
  - [x] Apply to all `.rs` files

### Success Criteria
- [x] Project compiles with `cargo build`
- [x] Basic error types are usable in tests
- [x] Logging produces output in examples
- [x] All files have proper license headers

### Dependencies
- None (first phase)

---

## Phase 2: Basic Model Management
**Priority: HIGH** | **Estimated Time: 4-6 hours**

### Objectives
Implement the core model loading and management functionality, respecting the `!Send` constraint.

### Tasks
- [x] Implement `EmbeddingModel` struct (`src/model.rs`)
  - [x] Define struct using `self_cell` crate to handle self-referential fields
  - [x] Use `ModelCell` to safely store both `LlamaModel` and `LlamaContext`
  - [x] Ensure struct is marked as `!Send` (automatic due to `LlamaContext`)
  - [x] Add model metadata fields (name, path, dimensions)
  
- [x] Implement model initialization
  - [x] Create `new()` method accepting model path
  - [x] Initialize `LlamaBackend` (once per process)
  - [x] Load model with `LlamaModelParams`
  - [x] Create context with `LlamaContextParams`
  - [x] Configure thread count settings
  
- [x] Implement model configuration (`src/config.rs`)
  - [x] Define `ModelConfig` struct with:
    - [x] `model_path: PathBuf`
    - [x] `model_name: String`
    - [x] `n_ctx: Option<u32>` (context size)
    - [x] `n_threads: Option<usize>` (CPU threads)
    - [x] `n_gpu_layers: Option<u32>` (GPU offload)
    - [x] ~~`seed: Option<u32>`~~ (removed - not applicable for embeddings)
  - [x] Implement builder pattern for `ModelConfig`
  
- [x] Add model lifecycle methods
  - [x] `load()` - Load model from disk
  - [x] `unload()` - Clean up resources
  - [x] `is_loaded()` - Check model state
  - [x] Implement `Drop` trait for cleanup
  
- [x] Add basic model information methods
  - [x] `embedding_dimensions()` - Get output dimension size
  - [x] `max_sequence_length()` - Get max input length
  - [x] `model_size()` - Get model memory footprint
  
- [x] Write unit tests
  - [x] Test model loading with valid path
  - [x] Test error handling for invalid paths
  - [x] Test configuration builder
  - [x] Verify `!Send` constraint at compile time

### Success Criteria
- [x] Can load a GGUF model file successfully (requires actual model file)
- [x] Proper error messages for load failures
- [x] Model cleanup on drop
- [x] Configuration builder works as expected

### Dependencies
- Phase 1 (Core Infrastructure)

---

## Phase 3: Single Embedding Generation
**Priority: HIGH** | **Estimated Time: 6-8 hours**

### Objectives
Implement single-text embedding generation with proper tokenization and processing.

### Tasks
- [x] Implement tokenization
  - [x] Add tokenization method to `EmbeddingModel`
  - [x] Handle text encoding properly (UTF-8)
  - [x] Implement token limit checking
  - [x] Add special tokens handling (BOS/EOS)
  
- [x] Implement embedding generation
  - [x] Create `generate_embedding(&self, text: &str) -> Result<Vec<f32>>`
  - [x] Tokenize input text
  - [x] Create `LlamaBatch` for tokens
  - [x] Perform forward pass through model
  - [x] Extract embedding vector from output
  - [x] Normalize embeddings if required
  
- [x] Implement `EmbeddingEngine` (`src/engine.rs`)
  - [x] Define public interface struct
  - [x] Store model instances (thread-local)
  - [x] Implement model registry/lookup
  - [x] Add `embed()` public method
  
- [x] Add embedding post-processing
  - [x] L2 normalization option
  - [x] Pooling strategies (Mean, CLS, Max, MeanSqrt)
  - [x] Output format configuration
  
- [x] Implement builder pattern for `EmbeddingEngine`
  - [x] `EngineConfig` struct with defaults (already existed)
  - [x] Builder methods for configuration
  - [x] Validation of configuration
  
- [x] Error handling improvements
  - [x] Add specific error types for embedding failures
  - [x] Provide helpful error messages
  - [x] Include context in errors (model name, text length)
  
- [x] Write integration tests
  - [x] Test with small test model (requires actual model)
  - [x] Verify embedding dimensions
  - [x] Test error cases (empty text, too long text)
  - [x] Benchmark single embedding performance

### Success Criteria
- [x] Can generate embeddings for simple text
- [x] Embeddings have correct dimensions
- [x] Performance meets target (<100ms for small text)
- [x] Proper error handling for edge cases

### Dependencies
- Phase 2 (Model Management)

---

## Phase 4: Batch Processing
**Priority: MEDIUM** | **Estimated Time: 8-10 hours**

### Objectives
Implement efficient batch processing with parallel pre/post-processing while respecting thread constraints.

### Tasks
- [ ] Implement batch processing logic (`src/batch.rs`)
  - [ ] Define `BatchProcessor` struct
  - [ ] Implement text collection and validation
  - [ ] Handle variable-length inputs
  - [ ] Optimize memory allocation
  
- [ ] Implement parallel tokenization
  - [ ] Use `rayon` for parallel text processing
  - [ ] Tokenize multiple texts concurrently
  - [ ] Collect tokens into batches
  - [ ] Handle tokenization errors gracefully
  
- [ ] Implement sequential model inference
  - [ ] Process token batches through model (single-threaded)
  - [ ] Handle batch size limits
  - [ ] Implement progress tracking
  - [ ] Manage memory efficiently
  
- [ ] Implement parallel post-processing
  - [ ] Normalize embeddings in parallel
  - [ ] Format output concurrently
  - [ ] Aggregate results efficiently
  
- [ ] Add batch API to `EmbeddingEngine`
  - [ ] `embed_batch()` method
  - [ ] Configure batch size limits
  - [ ] Add progress callback option
  - [ ] Return results in input order
  
- [ ] Optimize batch performance
  - [ ] Profile with `flamegraph`
  - [ ] Minimize allocations
  - [ ] Optimize memory layout
  - [ ] Tune batch sizes
  
- [ ] Handle edge cases
  - [ ] Empty batch
  - [ ] Single item batch
  - [ ] Very large batches
  - [ ] Mixed text lengths
  
- [ ] Write comprehensive tests
  - [ ] Test various batch sizes
  - [ ] Verify order preservation
  - [ ] Test concurrent batch requests
  - [ ] Benchmark throughput

### Success Criteria
- [ ] Batch processing faster than sequential
- [ ] Throughput >1000 embeddings/second (target)
- [ ] Memory usage scales linearly
- [ ] Results match single-embedding quality

### Dependencies
- Phase 3 (Single Embedding Generation)

---

## Phase 5: Testing & Documentation
**Priority: MEDIUM** | **Estimated Time: 6-8 hours**

### Objectives
Ensure code quality with comprehensive testing and provide excellent documentation.

### Tasks
- [ ] Set up test infrastructure
  - [ ] Download test models (MiniLM, jina-embeddings)
  - [ ] Create test fixtures
  - [ ] Set up CI test environment
  - [ ] Configure code coverage
  
- [ ] Write unit tests (per module)
  - [ ] Test `error.rs` - Error conversions and display
  - [ ] Test `config.rs` - Builder patterns and validation
  - [ ] Test `model.rs` - Model lifecycle
  - [ ] Test `batch.rs` - Batch processing logic
  - [ ] Test `engine.rs` - Public API
  
- [ ] Write integration tests
  - [ ] End-to-end embedding generation
  - [ ] Model loading and unloading cycles
  - [ ] Batch processing with real models
  - [ ] Error recovery scenarios
  
- [ ] Write concurrency tests
  - [ ] Verify thread-local model isolation
  - [ ] Test parallel batch operations
  - [ ] Ensure no data races
  - [ ] Test resource cleanup
  
- [ ] Write performance benchmarks
  - [ ] Single embedding latency
  - [ ] Batch throughput
  - [ ] Memory usage patterns
  - [ ] Scaling characteristics
  
- [ ] Create documentation
  - [ ] Write rustdoc for all public APIs
  - [ ] Create usage examples
  - [ ] Write README with quickstart
  - [ ] Document configuration options
  - [ ] Add architecture diagrams
  
- [ ] Create example applications
  - [ ] Simple embedding generation
  - [ ] Batch processing example
  - [ ] Configuration examples
  - [ ] Error handling patterns
  
- [ ] Set up quality checks
  - [ ] Configure `clippy` lints
  - [ ] Set up `rustfmt` configuration
  - [ ] Add pre-commit hooks
  - [ ] Configure CI/CD pipeline

### Success Criteria
- [ ] Test coverage >80%
- [ ] All public APIs documented
- [ ] Examples run successfully
- [ ] CI/CD pipeline green

### Dependencies
- Phase 4 (Batch Processing)

---

## Phase 6: Performance Optimization
**Priority: LOW** | **Estimated Time: 4-6 hours**

### Objectives
Optimize performance based on profiling results and real-world usage patterns.

### Tasks
- [ ] Profile current implementation
  - [ ] Use `cargo flamegraph` for CPU profiling
  - [ ] Measure memory allocations with `valgrind`
  - [ ] Identify bottlenecks
  - [ ] Create baseline benchmarks
  
- [ ] Optimize hot paths
  - [ ] Minimize allocations in embedding loop
  - [ ] Use `SmallVec` for small collections
  - [ ] Optimize vector operations
  - [ ] Consider SIMD for normalization
  
- [ ] Improve memory efficiency
  - [ ] Implement token buffer pooling
  - [ ] Reuse embedding buffers
  - [ ] Optimize batch memory layout
  - [ ] Reduce temporary allocations
  
- [ ] Enhance caching strategies
  - [ ] Cache tokenization results
  - [ ] Implement LRU cache for frequent inputs
  - [ ] Cache model metadata
  - [ ] Consider memory-mapped models
  
- [ ] Optimize configuration
  - [ ] Auto-tune thread counts
  - [ ] Optimize batch sizes
  - [ ] Profile-guided optimization
  - [ ] Platform-specific optimizations
  
- [ ] Consider alternative implementations
  - [ ] Evaluate `parking_lot` mutexes
  - [ ] Test different channel implementations
  - [ ] Compare allocator options
  - [ ] Investigate GPU acceleration
  
- [ ] Validate optimizations
  - [ ] Ensure correctness maintained
  - [ ] Verify performance improvements
  - [ ] Check memory usage reduction
  - [ ] Test on different hardware

### Success Criteria
- [ ] 20% performance improvement
- [ ] Reduced memory footprint
- [ ] No regression in accuracy
- [ ] Maintained code clarity

### Dependencies
- Phase 5 (Testing & Documentation)

---

## Implementation Notes

### Self-Referential Struct Solution

The `LlamaContext` holds a reference to `LlamaModel`, creating a self-referential struct challenge. This was solved using the `self_cell` crate:

1. **The Problem**: `LlamaContext<'a>` borrows from `LlamaModel` with lifetime `'a`
2. **The Solution**: Use `self_cell::self_cell!` macro to safely create a self-referential struct
3. **Implementation**:
   ```rust
   self_cell! {
       struct ModelCell {
           owner: LlamaModel,
           #[covariant]
           dependent: LlamaContext,
       }
   }
   ```
4. **Access Pattern**: 
   - Use `cell.borrow_owner()` to access the model
   - Use `cell.borrow_dependent()` to access the context
   - The macro ensures proper lifetime management and drop order

### Thread Safety Considerations

Due to the `!Send` constraint of `LlamaContext`:

1. **Never attempt to share models between threads**
   ```rust
   // ❌ Won't compile
   let model = Arc::new(EmbeddingModel::new(...)?);
   
   // ✅ Correct approach
   thread_local! {
       static MODEL: RefCell<Option<EmbeddingModel>> = RefCell::new(None);
   }
   ```

2. **Use message passing for concurrent operations**
   ```rust
   // For the library, batch processing uses:
   // - Parallel pre-processing (rayon)
   // - Sequential model inference (single thread)
   // - Parallel post-processing (rayon)
   ```

3. **Each worker thread needs its own model instance**
   - This increases memory usage but ensures thread safety
   - Models cannot be pooled or shared
   - Consider model lazy loading to manage memory

### Testing Strategy

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test with real GGUF models
3. **Property Tests**: Use `proptest` for edge cases
4. **Benchmarks**: Track performance regressions
5. **Thread Safety Tests**: Verify compile-time guarantees

### Performance Targets

- Single embedding: <50ms (CPU), <10ms (GPU)
- Batch throughput: >1000 embeddings/second
- Memory overhead: <2x model size
- Startup time: <5 seconds for 1GB model

### Error Handling Philosophy

- Use `Result<T, E>` everywhere
- Provide context in errors
- Make errors actionable
- Never panic in library code
- Log errors appropriately

## Success Metrics

### Phase Completion Criteria
- [ ] All tasks completed
- [ ] Tests passing
- [ ] Documentation complete
- [ ] Code reviewed
- [ ] Performance targets met

### Overall Project Success
- [ ] Library is usable standalone
- [ ] API is ergonomic and Rust-idiomatic
- [ ] Performance meets or exceeds targets
- [ ] Thread safety guaranteed at compile time
- [ ] Comprehensive documentation and examples