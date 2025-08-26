# Embellama Server Implementation Plan

## Overview
Build an OpenAI-compatible REST API server for the Embellama library using Axum, providing high-performance embedding generation endpoints.

## Phase 1: Server Implementation

### 1.1 CLI Interface
- [ ] Set up `clap` for argument parsing with:
  - `--model-path` (required)
  - `--model-name` (default: model filename)
  - `--host` (default: 127.0.0.1)
  - `--port` (default: 8080)
  - `--context-size` (optional)
  - `--threads` (optional)
  - `--gpu` (flag)
- [ ] Add configuration file support (TOML/YAML)
- [ ] Implement environment variable fallbacks

### 1.2 API Data Models
- [ ] Create request/response DTOs:
  ```rust
  struct EmbeddingsRequest {
      model: String,
      input: StringOrVec,
      encoding_format: Option<EncodingFormat>,
  }
  
  struct EmbeddingsResponse {
      object: String,
      data: Vec<EmbeddingData>,
      model: String,
      usage: Usage,
  }
  ```
- [ ] Implement `serde` serialization/deserialization
- [ ] Add input validation middleware

### 1.3 Axum Server Setup
- [ ] Create `AppState` with `Arc<EmbeddingEngine>`
- [ ] Set up router with routes:
  - `POST /v1/embeddings` - Main embedding endpoint
  - `GET /health` - Health check
  - `GET /models` - List available models
  - `GET /metrics` - Prometheus metrics (optional)
- [ ] Implement middleware stack:
  - CORS handling
  - Request logging with tracing
  - Timeout middleware
  - Compression (gzip/brotli)
  - Rate limiting

### 1.4 API Endpoints
- [ ] Implement `/v1/embeddings` handler:
  - Parse request body
  - Handle single string and array inputs
  - Generate embeddings using engine
  - Format OpenAI-compatible response
  - Track token usage
- [ ] Add `/health` endpoint with model status
- [ ] Implement `/models` to list loaded models
- [ ] Add request ID tracking for debugging

### 1.5 Server Infrastructure
- [ ] Implement graceful shutdown handling
- [ ] Add connection pooling if needed
- [ ] Set up metrics collection (optional)
- [ ] Implement request/response logging
- [ ] Add OpenTelemetry support (optional)

## Phase 2: Testing & Quality Assurance

### 2.1 Server E2E Tests
- [ ] Test OpenAI API compatibility
- [ ] Test error responses and status codes
- [ ] Test rate limiting and timeouts
- [ ] Test graceful shutdown
- [ ] Load testing with `vegeta` or `k6`

### 2.2 Performance Testing
- [ ] Benchmark API latency under load
- [ ] Test concurrent request handling
- [ ] Memory usage under sustained load
- [ ] Connection pool efficiency
- [ ] Rate limiting effectiveness

## Phase 3: Documentation

### 3.1 API Documentation
- [ ] Create OpenAPI/Swagger spec
- [ ] Document all endpoints with examples
- [ ] Document error codes and responses
- [ ] Add authentication documentation (if implemented)

### 3.2 Deployment Documentation
- [ ] Create Docker/Containerfile
- [ ] Write Kubernetes deployment manifests
- [ ] Add docker-compose example
- [ ] Document systemd service setup
- [ ] Create production deployment guide

### 3.3 Client Examples
- [ ] Python client example
- [ ] JavaScript/TypeScript client example
- [ ] cURL command examples
- [ ] Postman collection

## Phase 4: Production Readiness

### 4.1 Production Features
- [ ] Add circuit breaker pattern
- [ ] Implement retry logic with backoff
- [ ] Add distributed tracing
- [ ] Implement health check probes
- [ ] Add admin endpoints for management

### 4.2 Security Hardening
- [ ] Add input sanitization
- [ ] Implement request size limits
- [ ] Add API key authentication (optional)
- [ ] Set up TLS/HTTPS support
- [ ] Security audit with `cargo audit`

### 4.3 Monitoring & Observability
- [ ] Prometheus metrics integration
- [ ] Structured logging with tracing
- [ ] Request/response logging
- [ ] Performance metrics dashboard
- [ ] Alert configuration

## Server-Specific Dependencies

```toml
[[bin]]
name = "embellama-server"
required-features = ["server"]
path = "src/bin/server.rs"

[features]
server = ["dep:axum", "dep:tokio", "dep:clap", "dep:tower", "dep:tower-http"]

[dependencies.tokio]
version = "1.35"
features = ["full"]
optional = true

[dependencies.axum]
version = "0.7"
optional = true

[dependencies.clap]
version = "4.4"
features = ["derive", "env"]
optional = true

[dependencies.tower]
version = "0.4"
optional = true

[dependencies.tower-http]
version = "0.5"
features = ["cors", "compression-gzip", "compression-br", "timeout", "trace"]
optional = true

[dependencies.prometheus]
version = "0.13"
optional = true

[dev-dependencies]
reqwest = { version = "0.11", features = ["json"] }
```

## Architecture Notes

### Axum Best Practices (2024)
- Use `Arc<AppState>` for shared state management
- Leverage Tower middleware ecosystem for cross-cutting concerns
- Implement extractors for clean request handling
- Use tokio runtime for async operations
- Modular router organization with `merge()` for scalability

### OpenAI API Compatibility
- Endpoint: `POST /v1/embeddings`
- Support both single string and array inputs
- Return format includes `object`, `data`, `model`, and `usage` fields
- Maximum input: 8192 tokens for latest models
- Batch limit: 2048 inputs per request

### State Management Pattern
```rust
#[derive(Clone)]
struct AppState {
    engine: Arc<EmbeddingEngine>,
    metrics: Arc<Mutex<Metrics>>,
    config: Arc<ServerConfig>,
}

// Usage in handlers
async fn embeddings_handler(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, ApiError> {
    // Handler implementation
}
```

### Middleware Stack Order
1. Request ID injection
2. Logging/Tracing
3. CORS
4. Rate limiting
5. Timeout
6. Compression
7. Route handlers

## Risk Mitigation

1. **API compatibility**: Regularly test against OpenAI client libraries
2. **Load handling**: Implement backpressure and queue management
3. **Memory leaks**: Use connection pooling with proper cleanup
4. **Security**: Regular dependency updates and security audits
5. **Observability**: Comprehensive metrics and logging from day one

## Success Metrics

- [ ] < 10ms API overhead (excluding model inference)
- [ ] Handle 1000+ concurrent connections
- [ ] 99.99% API availability
- [ ] Full OpenAI client compatibility
- [ ] < 100MB server memory overhead

## Development Workflow

1. **Build on stable library** - Server depends on completed library
2. **Start with core endpoints** - `/v1/embeddings` first
3. **Add middleware incrementally** - Test each layer
4. **Load test early** - Identify bottlenecks before production
5. **Document as you build** - Keep OpenAPI spec current

This plan focuses on building a production-ready REST API server that complements the Embellama library.