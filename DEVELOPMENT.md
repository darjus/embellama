# Development Guide

Welcome to the Embellama development guide! This document provides guidelines and best practices for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Benchmarking](#benchmarking)
- [Documentation](#documentation)
- [Debugging](#debugging)
- [Contributing](#contributing)
- [Release Process](#release-process)

## Getting Started

### Prerequisites

- Rust 1.70.0 or later (MSRV)
- CMake (for building llama.cpp)
- A C++ compiler (gcc, clang, or MSVC)
- Git

### Initial Setup

1. Clone the repository:
```bash
git clone https://github.com/embellama/embellama.git
cd embellama
```

2. Install Rust (if not already installed):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

3. Add required components:
```bash
rustup component add rustfmt clippy
```

4. Build the project:
```bash
cargo build
```

5. Run tests:
```bash
cargo test
```

## Development Setup

### IDE Configuration

#### VS Code

Install the following extensions:
- rust-analyzer
- CodeLLDB (for debugging)
- Even Better TOML

Recommended settings (`.vscode/settings.json`):
```json
{
    "rust-analyzer.cargo.features": "all",
    "rust-analyzer.checkOnSave.command": "clippy"
}
```

#### IntelliJ IDEA / CLion

Install the Rust plugin and configure it to use `cargo clippy` for on-save checks.

### Pre-commit Hooks

Install pre-commit hooks to ensure code quality:

```bash
# Create the hooks directory
mkdir -p .git/hooks

# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/sh
set -e

# Format code
cargo fmt -- --check

# Run clippy
cargo clippy --all-features -- -D warnings

# Run tests
cargo test --quiet

echo "Pre-commit checks passed!"
EOF

# Make it executable
chmod +x .git/hooks/pre-commit
```

## Code Style

We use `rustfmt` for automatic code formatting and `clippy` for linting.

### Formatting

Always format your code before committing:
```bash
cargo fmt
```

Check formatting without modifying files:
```bash
cargo fmt -- --check
```

### Linting

Run clippy with all features enabled:
```bash
cargo clippy --all-features -- -D warnings
```

### Best Practices

1. **Error Handling**: Use the custom `Error` type from `error.rs` for all error handling
2. **Documentation**: All public APIs must have documentation comments
3. **Tests**: Write unit tests for all new functionality
4. **Safety**: Minimize use of `unsafe` code; document safety invariants when necessary
5. **Performance**: Profile before optimizing; use benchmarks to validate improvements

### Naming Conventions

- **Modules**: snake_case (e.g., `embedding_engine`)
- **Types**: PascalCase (e.g., `EmbeddingEngine`)
- **Functions/Methods**: snake_case (e.g., `generate_embedding`)
- **Constants**: SCREAMING_SNAKE_CASE (e.g., `MAX_BATCH_SIZE`)
- **Type Parameters**: Single capital letter or PascalCase (e.g., `T` or `ModelType`)

## Testing

### Running Tests

Run all tests:
```bash
cargo test
```

Run tests with output:
```bash
cargo test -- --nocapture
```

Run specific test:
```bash
cargo test test_name
```

Run tests in release mode:
```bash
cargo test --release
```

### Test Organization

- Unit tests: In the same file as the code being tested, in a `#[cfg(test)]` module
- Integration tests: In the `tests/` directory
- Doc tests: In documentation comments using ` ```rust ` blocks

### Test Models

For integration tests, you'll need GGUF model files. See `tests/README.md` for instructions on obtaining test models.

### Writing Tests

Example unit test:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_something() {
        // Arrange
        let input = "test";
        
        // Act
        let result = function_under_test(input);
        
        // Assert
        assert_eq!(result, expected);
    }
}
```

## Benchmarking

### Running Benchmarks

Run all benchmarks:
```bash
cargo bench
```

Run specific benchmark:
```bash
cargo bench -- benchmark_name
```

### Writing Benchmarks

Benchmarks are located in `benches/` and use the `criterion` crate:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_embedding(c: &mut Criterion) {
    c.bench_function("single_embedding", |b| {
        b.iter(|| {
            // Code to benchmark
            generate_embedding(black_box("test input"))
        });
    });
}

criterion_group!(benches, benchmark_embedding);
criterion_main!(benches);
```

### Performance Profiling

Use `cargo-flamegraph` for performance profiling:

```bash
cargo install flamegraph
cargo flamegraph --bench embeddings
```

## Documentation

### Building Documentation

Build and open documentation:
```bash
cargo doc --open
```

Build documentation for all dependencies:
```bash
cargo doc --open --all-features
```

### Writing Documentation

All public items must have documentation:

```rust
/// Brief description of what this does.
///
/// # Arguments
///
/// * `input` - Description of the input parameter
///
/// # Returns
///
/// Description of the return value
///
/// # Examples
///
/// ```rust
/// let result = function(input);
/// assert_eq!(result, expected);
/// ```
///
/// # Errors
///
/// Returns `Error::InvalidInput` if the input is invalid
pub fn function(input: &str) -> Result<String> {
    // Implementation
}
```

## Debugging

### Logging

Use the `tracing` crate for logging:

```rust
use tracing::{debug, info, warn, error};

info!("Loading model from {}", path.display());
debug!("Model metadata: {:?}", metadata);
warn!("Using fallback configuration");
error!("Failed to load model: {}", err);
```

Enable debug logging:
```bash
RUST_LOG=debug cargo run
```

Enable trace-level logging for specific modules:
```bash
RUST_LOG=embellama::engine=trace cargo run
```

### Using the Debugger

#### VS Code with CodeLLDB

1. Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "lldb",
            "request": "launch",
            "name": "Debug unit tests",
            "cargo": {
                "args": ["test", "--no-run"],
                "filter": {
                    "name": "embellama",
                    "kind": "lib"
                }
            },
            "args": [],
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

2. Set breakpoints and press F5 to debug

#### Command Line with rust-gdb

```bash
cargo build
rust-gdb target/debug/embellama-server
```

## Contributing

### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `cargo test`
6. Format your code: `cargo fmt`
7. Run clippy: `cargo clippy -- -D warnings`
8. Update documentation as needed
9. Commit with a descriptive message
10. Push to your fork
11. Create a pull request

### Commit Message Format

Follow the conventional commits specification:

```
type(scope): brief description

Longer description if needed.

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or fixes
- `build`: Build system changes
- `ci`: CI configuration changes
- `chore`: Other changes

### Code Review Guidelines

- Be constructive and respectful
- Explain the "why" behind suggestions
- Consider performance implications
- Verify test coverage
- Check for breaking changes

## Release Process

### Version Bumping

Update version in `Cargo.toml`:
```toml
[package]
version = "0.2.0"
```

### Creating a Release

1. Update `CHANGELOG.md`
2. Bump version in `Cargo.toml`
3. Create a git tag: `git tag -a v0.2.0 -m "Release v0.2.0"`
4. Push tags: `git push origin --tags`
5. Create GitHub release
6. Publish to crates.io: `cargo publish`

### Checking before Publishing

Dry run to verify:
```bash
cargo publish --dry-run
```

Check package contents:
```bash
cargo package --list
```

## Troubleshooting

### Common Issues

#### llama-cpp-2 build failures

Ensure you have CMake and a C++ compiler installed:
```bash
# Ubuntu/Debian
sudo apt-get install cmake build-essential

# macOS
brew install cmake

# Windows
# Install Visual Studio with C++ development tools
```

#### Out of memory during model loading

Reduce the context size or use a smaller model:
```rust
let config = EngineConfig::builder()
    .with_context_size(512)  // Smaller context
    .with_use_mmap(true)     // Enable memory mapping
    .build()?;
```

#### Slow inference on CPU

Enable multi-threading and optimize thread count:
```rust
let config = EngineConfig::builder()
    .with_n_threads(num_cpus::get())
    .build()?;
```

### Getting Help

- Check existing [GitHub Issues](https://github.com/embellama/embellama/issues)
- Join our [Discord server](https://discord.gg/embellama)
- Read the [API documentation](https://docs.rs/embellama)

## Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

## License

By contributing to Embellama, you agree that your contributions will be licensed under the Apache License 2.0.