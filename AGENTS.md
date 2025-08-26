# Agent Guidelines for Embellama



## Project Overview

Embellama is a high-performance Rust crate and server for generating text embeddings using `llama-cpp-2`. The project provides:

1. **Core Library**: A robust and ergonomic Rust API for interacting with `llama.cpp` to generate embeddings
2. **API Server**: An optional OpenAI-compatible REST API server (available via `server` feature flag)

### Primary Goals
- Simple and intuitive Rust API for embedding generation
- Support for model loading/unloading and batch processing
- High performance for both low-latency single requests and high-throughput batch operations
- OpenAI API compatibility (`/v1/embeddings` endpoint)
- Clean separation between library and server concerns

## Rust Development Standards

### Code Quality Requirements

**MANDATORY**: Before ANY commit, you MUST:

1. **Format Code**: Run `cargo fmt` to ensure consistent formatting
2. **Lint Code**: Run `cargo clippy` and address ALL warnings
3. **Code Review**: Use the rust code review agent to validate changes
4. **Test**: Run `cargo test` to ensure all tests pass
5. **Update Changelog**: Use `clog` CLI to update the changelog

### Error Handling

#### Use `thiserror` for Error Types
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbellamaError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Failed to load model from {path}")]
    ModelLoadFailed { path: String, #[source] source: llama_cpp_2::Error },
    
    #[error("Embedding generation failed")]
    EmbeddingFailed(#[from] llama_cpp_2::Error),
}
```

#### Return `Result` Types
- All fallible operations MUST return `Result<T, E>`
- Use specific error types from `thiserror`
- Ensure compatibility with `anyhow` for application-level error handling

#### Example Usage
```rust
// Library code with specific errors
pub fn load_model(path: &str) -> Result<Model, EmbellamaError> {
    // Implementation
}

// Application code can use anyhow
use anyhow::Result;
fn main() -> Result<()> {
    let model = embellama::load_model("model.gguf")?;
    Ok(())
}
```

### Logging with `tracing`

#### Setup
```rust
use tracing::{debug, error, info, warn, trace};
```

#### Logging Guidelines
- **TRACE**: Very detailed information, hot path details
- **DEBUG**: Useful debugging information (model loading steps, batch sizes)
- **INFO**: Important state changes (model loaded, server started)
- **WARN**: Recoverable issues (fallback behavior, deprecated usage)
- **ERROR**: Unrecoverable errors with context

#### Example
```rust
#[tracing::instrument(skip(model_data))]
pub fn load_model(path: &str, model_data: &[u8]) -> Result<Model, EmbellamaError> {
    info!(path = %path, "Loading embedding model");
    debug!(size = model_data.len(), "Model data size");
    
    match internal_load(model_data) {
        Ok(model) => {
            info!("Model loaded successfully");
            Ok(model)
        }
        Err(e) => {
            error!(error = %e, "Failed to load model");
            Err(EmbellamaError::ModelLoadFailed { path: path.to_string(), source: e })
        }
    }
}
```

### Visibility and Encapsulation

#### Keep Implementation Details Private
```rust
// Use module-level privacy
mod internal {
    pub(crate) fn helper_function() { }
}

// Or crate-level visibility
pub(crate) struct InternalState { }

// Only expose necessary public interface
pub struct EmbeddingEngine {
    inner: Arc<EngineInner>, // Private implementation
}
```

#### Public API Design
- Minimize public surface area
- Use builder patterns for complex configurations
- Document all public items with rustdoc
- Implement `Debug` for ALL public types

### Debug Trait Implementation

**MANDATORY**: All public types MUST implement `Debug`

```rust
#[derive(Debug)]
pub struct EmbeddingEngine {
    // fields
}

// For types with sensitive data
impl fmt::Debug for ModelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelConfig")
            .field("path", &self.path)
            .field("name", &self.name)
            .finish_non_exhaustive() // Hide internal details
    }
}
```

## License Compliance Requirements

**MANDATORY**: All source files MUST include the Apache 2.0 license header:

```rust
// Copyright 2024 Embellama Contributors
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
```

### When to Add License Headers

1. **New Files**: ALWAYS add the license header when creating any new `.rs` source file
2. **Modified Files**: Ensure existing files have the header before making changes
3. **Binary Files**: Do not add headers to binary files or generated content
4. **Test Files**: Include headers in test files as well

## Commit and Changelog Management

### Commit Messages

**CRITICAL RULES**:
1. **NEVER** mention AI tools, assistants, or automation in commit messages
2. Focus on WHAT changed and WHY, not HOW you arrived at the solution
3. Use conventional commit format:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation only
   - `style:` Formatting, missing semicolons, etc.
   - `refactor:` Code change that neither fixes a bug nor adds a feature
   - `perf:` Performance improvement
   - `test:` Adding missing tests
   - `chore:` Maintenance tasks

### Using `clog` CLI

Before committing, update the changelog:

```bash
# Install clog-cli if not present
cargo install clog-cli

# Generate/update changelog
clog -r https://github.com/yourusername/embellama -o CHANGELOG.md

# For specific version
clog --setversion v0.1.0
```

### Pre-Commit Checklist

1. ✅ Run `cargo fmt`
2. ✅ Run `cargo clippy` and fix ALL warnings
3. ✅ Run `cargo test`
4. ✅ Use rust code review agent
5. ✅ Update changelog with `clog`
6. ✅ Ensure no secrets or API keys
7. ✅ Verify commit message follows guidelines

## Problem-Solving Tools and MCPs

### When to Use Each Tool

#### Web Search (Fetch and Brave MCPs)
Use for:
- Looking up Rust documentation and crate usage
- Finding solutions to specific error messages
- Researching llama.cpp integration details
- Checking best practices and patterns

Example scenarios:
- "How to use llama-cpp-2 crate effectively"
- "Rust async performance optimization techniques"
- "OpenAI embedding API specification"

#### zen:thinkdeeper
Use when:
- Designing new architectural components
- Evaluating complex trade-offs
- Solving intricate algorithmic problems
- Making critical design decisions

Example scenarios:
- "Should we use channels or shared state for batch processing?"
- "How to optimize memory usage for large embedding batches?"
- "Architecture for dynamic model loading/unloading"

#### zen:debug
Use when:
- Standard debugging hasn't revealed root cause
- Dealing with complex async/concurrent issues
- Investigating memory leaks or performance problems
- Encountering mysterious segfaults or panics

Example scenarios:
- "Why does the model crash only under high concurrency?"
- "Memory usage grows unbounded during batch processing"
- "Deadlock occurring in specific request patterns"

### Decision Framework

```
Start with standard debugging
    ↓
If unclear → Use web search for similar issues
    ↓
If complex design question → Use zen:thinkdeeper
    ↓
If persistent bug → Use zen:debug
```

## Project Structure

```
embellama/
├── Cargo.toml          # Dependencies and features
├── CHANGELOG.md        # Maintained with clog
├── AGENTS.md          # This file
├── ARCHITECTURE.md    # Design documentation
└── src/
    ├── lib.rs         # Public API, feature flags
    ├── engine.rs      # EmbeddingEngine (public interface)
    ├── model.rs       # EmbeddingModel (internal)
    ├── batch.rs       # Batch processing (internal)
    ├── config.rs      # Configuration types (public)
    ├── error.rs       # Error types with thiserror
    └── bin/
        └── server.rs  # Server binary (server feature)
```

## Testing Requirements

### Unit Tests
- Test each module in isolation
- Mock external dependencies
- Test error conditions thoroughly
- Use `#[tracing_test::traced_test]` for tests with logging

### Integration Tests
- Test against real GGUF models
- Verify batch processing correctness
- Test model loading/unloading cycles
- Validate OpenAI API compatibility

### Performance Tests
- Benchmark single embedding generation
- Benchmark batch processing
- Memory usage under load
- Concurrent request handling

## Security Considerations

- **Never** commit secrets, API keys, or credentials
- Validate all inputs in the server component
- Use `secrecy` crate for sensitive data if needed
- Run in trusted environments only (no built-in auth)
- Sanitize error messages to avoid information leakage

## Performance Goals

- Single embedding: < 50ms latency (model-dependent)
- Batch processing: > 1000 embeddings/second
- Memory efficiency: < 2x model size overhead
- Concurrent requests: Scale linearly with cores

### Optimization Checklist
- ✅ Profile with `cargo flamegraph`
- ✅ Minimize allocations in hot paths
- ✅ Use `Arc` for shared immutable data
- ✅ Prefer borrowing over cloning
- ✅ Use `SmallVec` for small collections
- ✅ Consider `parking_lot` for better mutex performance

## Documentation Standards

### Rustdoc Requirements
```rust
/// Generates embeddings for the given text.
/// 
/// # Arguments
/// * `model_name` - The name of the model to use
/// * `text` - The text to generate embeddings for
/// 
/// # Returns
/// A vector of floating-point embeddings
/// 
/// # Errors
/// Returns `EmbellamaError::ModelNotFound` if the model doesn't exist
/// 
/// # Example
/// ```
/// let embeddings = engine.embed("my-model", "Hello, world!")?;
/// ```
pub fn embed(&self, model_name: &str, text: &str) -> Result<Vec<f32>, EmbellamaError> {
    // Implementation
}
```

### Documentation Priorities
1. All public APIs must have rustdoc comments
2. Include usage examples for complex APIs
3. Document error conditions clearly
4. Keep internal documentation focused and technical
5. Update ARCHITECTURE.md for significant changes

## Development Workflow

1. **Planning**: Use zen:thinkdeeper for design decisions
2. **Research**: Use Fetch/Brave for documentation lookup
3. **Implementation**: Follow Rust best practices
4. **Debugging**: Use zen:debug for complex issues
5. **Testing**: Write comprehensive tests
6. **Review**: Run rust code review agent
7. **Commit**: Update changelog with clog, write clear message
8. **Document**: Update rustdoc and ARCHITECTURE.md if needed