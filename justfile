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
    RUST_BACKTRACE=1 EMBELLAMA_TEST_MODEL={{test_model_file}} \
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
    @echo "Running clippy on library and binaries (strict)..."
    cargo clippy --lib --bins --all-features -- -D warnings -D clippy::pedantic
    @echo "Running clippy on tests, examples, and benches (lenient)..."
    cargo clippy --tests --examples --benches --all-features -- -W warnings -W clippy::pedantic

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

# Build the server binary
build-server:
    @echo "Building server binary..."
    cargo build --features server --bin embellama-server
    @echo "✓ Server binary built"

# Run the server with test model
run-server: download-test-model build-server
    @echo "Starting server with test model..."
    cargo run --features server --bin embellama-server -- \
        --model-path {{test_model_file}} \
        --model-name test-model \
        --workers 2 \
        --log-level info

# Run server in background for testing (returns immediately)
start-server: download-test-model build-server
    @echo "Starting server in background..."
    @cargo run --features server --bin embellama-server -- \
        --model-path {{test_model_file}} \
        --model-name test-model \
        --workers 2 \
        --log-level info > server.log 2>&1 & \
        echo $$! > server.pid
    @sleep 2
    @echo "✓ Server started (PID: `cat server.pid`)"

# Stop the background server
stop-server:
    @if [ -f server.pid ]; then \
        echo "Stopping server (PID: `cat server.pid`)..."; \
        kill `cat server.pid` 2>/dev/null || true; \
        rm -f server.pid; \
        echo "✓ Server stopped"; \
    else \
        echo "No server running"; \
    fi

# Test server API endpoints
test-server-api:
    @echo "Testing server API endpoints..."
    @echo "================================"
    @echo
    @echo "1. Testing /health endpoint:"
    @curl -s "http://localhost:8080/health" | jq . || echo "Failed - is server running?"
    @echo
    @echo "2. Testing /v1/models endpoint:"
    @curl -s "http://localhost:8080/v1/models" | jq . || echo "Failed - is server running?"
    @echo
    @echo "3. Testing /v1/embeddings with single text:"
    @curl -s -X POST "http://localhost:8080/v1/embeddings" \
        -H "Content-Type: application/json" \
        -d '{"model": "test-model", "input": "Hello, world!"}' \
        | jq '.object, .model, .usage' || echo "Failed - is server running?"
    @echo
    @echo "4. Testing /v1/embeddings with batch:"
    @curl -s -X POST "http://localhost:8080/v1/embeddings" \
        -H "Content-Type: application/json" \
        -d '{"model": "test-model", "input": ["Hello", "World"]}' \
        | jq '.object, .model, (.data | length)' || echo "Failed - is server running?"
    @echo
    @echo "================================"
    @echo "✓ API tests complete"

# Run server integration tests
test-server-integration: download-test-model
    @echo "Running server integration tests..."
    cargo test --features server --test server_api_tests -- --nocapture

# Run OpenAI compatibility tests
test-server-compat: download-test-model
    @echo "Running OpenAI compatibility tests..."
    cargo test --features server --test openai_compat_tests -- --nocapture

# Run server load tests (excluding slow tests)
test-server-load: download-test-model
    @echo "Running server load tests..."
    cargo test --features server --test server_load_tests -- --nocapture

# Run ALL server load tests (including slow/ignored tests)
test-server-load-all: download-test-model
    @echo "Running all server load tests (including slow tests)..."
    cargo test --features server --test server_load_tests -- --nocapture --ignored

# Run all server tests
test-server-all: test-server-integration test-server-compat test-server-load
    @echo "✓ All server tests completed"

# Test with Python OpenAI SDK
test-server-python: start-server
    @echo "Testing with Python OpenAI SDK..."
    @python3 scripts/test-openai-python.py || echo "Python SDK test failed - is openai package installed?"
    @just stop-server

# Test with JavaScript OpenAI SDK
test-server-js: start-server
    @echo "Testing with JavaScript OpenAI SDK..."
    @node scripts/test-openai-js.mjs || echo "JS SDK test failed - is openai package installed?"
    @just stop-server

# Full server test workflow (old compatibility)
test-server: start-server test-server-api stop-server
    @echo "✓ Server tests completed"

# Check server compilation
check-server:
    @echo "Checking server compilation..."
    cargo check --features server --bin embellama-server
    cargo check --features server --tests

# Clean server artifacts
clean-server: stop-server
    @echo "Cleaning server artifacts..."
    @rm -f server.log server.pid
    @echo "✓ Server artifacts cleaned"
