# Embellama Build & Test Automation
# Run `just` to see available commands

# Default recipe - show help
default:
    @just --list

# Model cache directory
models_dir := "models"
test_model_dir := models_dir + "/test"
bench_model_dir := models_dir + "/bench"

# Model URLs
test_model_url := "https://huggingface.co/Jarbas/all-MiniLM-L6-v2-Q4_K_M-GGUF/resolve/main/all-minilm-l6-v2-q4_k_m.gguf"
bench_model_url := "https://huggingface.co/gaianet/jina-embeddings-v2-base-code-GGUF/resolve/main/jina-embeddings-v2-base-code-Q4_K_M.gguf"

# Model filenames
test_model_file := test_model_dir + "/all-minilm-l6-v2-q4_k_m.gguf"
bench_model_file := bench_model_dir + "/jina-embeddings-v2-base-code-Q4_K_M.gguf"

# Download small test model (MiniLM ~15MB) for integration tests
download-test-model:
    @echo "Setting up test model..."
    @mkdir -p {{test_model_dir}}
    @if [ ! -f {{test_model_file}} ]; then \
        echo "Downloading MiniLM-L6-v2 model (~15MB)..."; \
        curl -L --progress-bar -o {{test_model_file}} {{test_model_url}}; \
        echo "✓ Test model (MiniLM) downloaded successfully"; \
    else \
        echo "✓ Test model (MiniLM) already cached"; \
    fi

# Download benchmark model (Jina ~110MB) for performance testing
download-bench-model:
    @echo "Setting up benchmark model..."
    @mkdir -p {{bench_model_dir}}
    @if [ ! -f {{bench_model_file}} ]; then \
        echo "Downloading Jina Embeddings v2 Base Code model (~110MB)..."; \
        curl -L --progress-bar -o {{bench_model_file}} {{bench_model_url}}; \
        echo "✓ Benchmark model (Jina) downloaded successfully"; \
    else \
        echo "✓ Benchmark model (Jina) already cached"; \
    fi

# Download all models
download-all: download-test-model download-bench-model
    @echo "✓ All models ready"

# Run unit tests (no models required)
test-unit:
    @echo "Running unit tests..."
    cargo test --lib

# Run integration tests with real model
test-integration: download-test-model
    @echo "Running integration tests with real model..."
    EMBELLAMA_TEST_MODEL={{test_model_file}} \
    cargo test --test integration_tests -- --nocapture

# Run concurrency tests
test-concurrency: download-test-model
    @echo "Running concurrency tests..."
    EMBELLAMA_TEST_MODEL={{test_model_file}} \
    cargo test --test concurrency_tests -- --nocapture

# Run all tests
test: test-unit test-integration test-concurrency
    @echo "✓ All tests completed"

# Run benchmarks with real model
bench: download-bench-model
    @echo "Running benchmarks with real model..."
    EMBELLAMA_BENCH_MODEL={{bench_model_file}} \
    cargo bench

# Quick benchmark (subset of benchmarks for faster testing)
bench-quick: download-bench-model
    @echo "Running quick benchmarks (subset only)..."
    EMBELLAMA_BENCH_MODEL={{bench_model_file}} \
    cargo bench -- "single_embedding/text_length/11$|batch_embeddings/batch_size/1$|thread_scaling/threads/1$"

# Run example with model
example NAME="simple": download-test-model
    @echo "Running example: {{NAME}}..."
    EMBELLAMA_MODEL={{test_model_file}} \
    cargo run --example {{NAME}}

# Check for compilation warnings
check:
    @echo "Checking for warnings..."
    cargo check --all-targets --all-features

# Fix common issues
fix:
    @echo "Running cargo fix..."
    cargo fix --all-targets --allow-dirty --allow-staged
    cargo clippy --fix --all-targets --allow-dirty --allow-staged

# Format code
fmt:
    @echo "Formatting code..."
    cargo fmt

# Run clippy
clippy:
    @echo "Running clippy..."
    cargo clippy --all-targets --all-features -- -D warnings

# Clean build artifacts
clean:
    @echo "Cleaning build artifacts..."
    cargo clean

# Clean downloaded models
clean-models:
    @echo "Removing cached models..."
    rm -rf {{models_dir}}

# Clean everything
clean-all: clean clean-models
    @echo "✓ All cleaned"

# Show model cache status
models-status:
    @echo "Model cache status:"
    @echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    @if [ -f {{test_model_file}} ]; then \
        size=`du -h {{test_model_file}} | cut -f1`; \
        echo "✓ Test model: {{test_model_file}} ($$size)"; \
    else \
        echo "✗ Test model: Not downloaded"; \
    fi
    @if [ -f {{bench_model_file}} ]; then \
        size=`du -h {{bench_model_file}} | cut -f1`; \
        echo "✓ Bench model: {{bench_model_file}} ($$size)"; \
    else \
        echo "✗ Bench model: Not downloaded"; \
    fi
    @echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    @if [ -d {{models_dir}} ]; then \
        total=`du -sh {{models_dir}} | cut -f1`; \
        echo "Total cache size: $$total"; \
    fi

# Development workflow - fix issues and test
dev: fix fmt clippy test-unit
    @echo "✓ Ready for integration testing"

# Pre-commit checks
pre-commit: fmt clippy test
    @echo "✓ All pre-commit checks passed"

# Full CI simulation
ci: clean check clippy test bench
    @echo "✓ CI checks completed successfully"