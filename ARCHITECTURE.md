# Architecture Proposal for Embellama

This document outlines the architectural design for `embellama`, a Rust crate and server for generating text embeddings using `llama-cpp-2`.

## 1. Overview

Embellama will consist of two main components:

1.  A **core library** that provides a robust and ergonomic Rust API for interacting with `llama.cpp` to generate embeddings.
2.  An **API server**, available as a feature flag, that exposes the library's functionality through an OpenAI-compatible REST API.

The primary goal is to create a high-performance, easy-to-use tool for developers who need to integrate local embedding models into their Rust applications.

## 2. Goals and Non-Goals

### Goals

*   Provide a simple and intuitive Rust API for embedding generation.
*   Support for loading/unloading models, and both single-text and batch embedding generation.
*   Offer an optional, feature-flagged `axum`-based server with an OpenAI-compatible API (`/v1/embeddings`).
*   Prioritize both low-latency single requests and high-throughput batch processing.
*   Enable configuration of the library via a programmatic builder pattern and the server via CLI arguments.

### Non-Goals

*   The library will **not** handle the downloading of models. Users are responsible for providing their own GGUF-formatted model files.
*   The initial version will only support the `llama.cpp` backend via the `llama-cpp-2` crate. Other backends are out of scope for now.
*   The server will not handle authentication or authorization. It is expected to run in a trusted environment.

## 3. Core Concepts

### `EmbeddingModel`

A struct that represents a loaded `llama.cpp` model. It will encapsulate the `llama_cpp_2::LlamaModel` and `llama_cpp_2::LlamaContext` and handle the logic for generating embeddings.

### `EmbeddingEngine`

The main entry point for the library. It will manage the lifecycle of `EmbeddingModel` instances (loading, unloading) and provide the public-facing API for generating embeddings. It will be configurable using a builder pattern.

### `AppState` (for Server)

An `axum` state struct that holds an instance of the `EmbeddingEngine`. This will be shared across all API handlers to provide access to the loaded model.

## 4. Library (`embellama`) Design

### Module Structure

```
src/
├── lib.rs         # Main library file, feature flags
├── engine.rs      # EmbeddingEngine implementation and builder
├── model.rs       # EmbeddingModel implementation
├── batch.rs       # Batch processing logic
├── config.rs      # Configuration structs for the engine
└── error.rs       # Custom error types
```

### Public API & Usage

The library will be configured using a builder pattern for `EmbeddingEngine`.

**Example Usage:**

```rust
use embellama::{EmbeddingEngine, EngineConfig};

// 1. Configure and build the engine
let config = EngineConfig::builder()
    .with_model_path("path/to/your/model.gguf")
    .with_model_name("my-embedding-model")
    .build()?;

let engine = EmbeddingEngine::new(config)?;

// 2. Generate a single embedding
let embedding = engine.embed("my-embedding-model", "Hello, world!")?;

// 3. Generate embeddings in a batch
let texts = vec!["First text", "Second text"];
let embeddings = engine.embed_batch("my-embedding-model", texts)?;

// 4. Unload the model when no longer needed
engine.unload_model("my-embedding-model")?;
```

### Error Handling

A custom `Error` enum will be defined in `src/error.rs` to handle all possible failures, from model loading to embedding generation. It will implement `std::error::Error` and provide conversions from underlying errors like those from `llama-cpp-2`.

## 5. Server (`server` feature) Design

The server will be enabled with a `server` feature flag.

### Dependencies

*   `axum`: For the web framework.
*   `tokio`: For the async runtime.
*   `clap`: For parsing CLI arguments.
*   `serde`: For JSON serialization/deserialization.
*   `tracing`: For logging.

### CLI Arguments

The server will be configured via CLI arguments using `clap`.

```bash
embellama-server --model-path /path/to/model.gguf --model-name my-model --port 8080
```

### `main.rs` (under `src/bin/server.rs` or similar)

The server's entry point will:
1.  Parse CLI arguments using `clap`.
2.  Instantiate and configure the `EmbeddingEngine` from the library.
3.  Create an `axum` `Router` with the engine wrapped in an `Arc` for state management.
4.  Define the `/v1/embeddings` and `/health` endpoints.
5.  Start the server.

### OpenAI-Compatible API

**Endpoint:** `POST /v1/embeddings`

**Request Body (`EmbeddingsRequest`):**

```json
{
  "model": "my-model",
  "input": "A single string"
}
```

or

```json
{
  "model": "my-model",
  "input": ["An array", "of strings"]
}
```

**Response Body (`EmbeddingsResponse`):**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.1, 0.2, ...]
    }
  ],
  "model": "my-model",
  "usage": {
    "prompt_tokens": 12,
    "total_tokens": 12
  }
}
```

## 6. Project Structure

```
embellama/
├── .gitignore
├── Cargo.toml
├── ARCHITECTURE.md
└── src/
    ├── lib.rs
    ├── engine.rs
    ├── model.rs
    ├── batch.rs
    ├── config.rs
    └── error.rs
    └── bin/
        └── server.rs  # Compiled only when "server" feature is enabled
```

In `Cargo.toml`:

```toml
[package]
name = "embellama"
version = "0.1.0"
edition = "2021"

[dependencies]
llama-cpp-2 = "..."
# other library dependencies

# Server-only dependencies
[features]
server = ["dep:axum", "dep:tokio", "dep:clap", "dep:serde", "dep:tracing"]

[[bin]]
name = "embellama-server"
required-features = ["server"]
path = "src/bin/server.rs"
```

## 7. Testing Strategy

*   **Unit Tests:** Each module in the library will have unit tests to verify its logic in isolation.
*   **Integration Tests:** An `integration` test module will be created. These tests will require embedding models to be present for testing the full flow. We will specifically test against GGUF-converted versions of `sentence-transformers/all-MiniLM-L6-v2` and `jinaai/jina-embeddings-v2-base-code`. A build script or a helper script can be provided to download these models for testing purposes.
*   **Server E2E Tests:** A separate test suite will make HTTP requests to a running instance of the server to verify API compliance and behavior, using the same test models.
