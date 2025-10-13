# Batching Strategy Refactoring Plan

## Executive Summary

Refactor embellama's batching implementation to properly distinguish between:
- **`n_batch`**: Total batch packing capacity (tokens that can be packed together)
- **`n_ubatch`**: Physical processing chunk size (internal llama.cpp micro-batching)
- **`n_seq_max`**: Parallel sequence slots (KV cache management)

This aligns with llama.cpp server behavior and enables better throughput through improved batch packing.

## Current Architecture Issues

### Problem 1: Missing n_batch Parameter
- Currently uses `n_seq_max` (max parallel sequences) to control batch packing
- Should use `n_batch` (total token capacity) instead
- Results in premature chunking and reduced throughput

**Current code** (src/batch.rs:126):
```rust
if total_tokens <= effective_max && token_sequences.len() <= n_seq_max {
    // Process all in single batch
}
```

**llama.cpp server approach**:
```cpp
while (slot.n_past < slot.n_prompt_tokens && batch.n_tokens < n_batch) {
    common_batch_add(batch, cur_tok, slot.n_past, { slot.id }, need_embd);
}
```

### Problem 2: Inefficient Batch Allocation
- Allocates batches to exact size: `LlamaBatch::new(total_tokens, 1)`
- No pre-allocation or reuse
- Causes unnecessary memory allocations and copies

**Current code** (src/model.rs:663, 955):
```rust
let mut batch = LlamaBatch::new(total_tokens, 1);
```

**Should be**:
```rust
let mut batch = LlamaBatch::new(n_batch, 1);  // Pre-allocate to capacity
```

### Problem 3: Confused Semantics
- `n_seq_max` used for both packing decisions AND parallel processing
- These are separate concerns that should be decoupled

## Constraints for Embeddings

**Embedding models are finicky and cannot split sequences across batches.**

Unlike generation models that can process incrementally, embedding models require atomic sequence processing. From llama.cpp server code:

```cpp
if (!slot.can_split()) {
    if (slot.n_prompt_tokens > n_ubatch) {
        send_error(slot, "input is too large to process. increase the physical batch size");
    }
}
```

**For embellama (embeddings-only)**:
- ✅ **Individual sequences**: Each must be ≤ `n_ubatch` (atomic processing required)
- ✅ **Batch packing**: Pack multiple sequences up to `n_batch` total tokens
- ✅ **Largest sequence constraint**: Must be < `n_batch`
- ✅ **Recommendation**: Set `n_ubatch = n_batch` for simplicity in embedding workloads
- ✅ **Active batches**: Total active tokens should fit within `context_size`
- ✅ **No n_seq_max packing constraint**: llama.cpp handles sequential GPU processing internally
- ✅ **Processing**: Prefilling with one copy in/out is faster even if GPU processes non-parallel

## Batch Scheduler Architecture

To enable intelligent batch packing and concurrent batch processing, the scheduler needs to track:

### State Management

```rust
struct BatchScheduler {
    /// Atomic counter of total tokens waiting in queue
    pending_tokens: AtomicUsize,

    /// Track in-flight batches and their sizes
    active_batches: Mutex<Vec<ActiveBatch>>,

    /// Reference to the embedding model
    model: Arc<Mutex<EmbeddingModel>>,

    /// Batch pool (thread-local or model-aware global)
    batch_pool: /* implementation choice in Phase 4 */,
}

struct ActiveBatch {
    id: Uuid,
    token_count: usize,
    created_at: Instant,
}
```

### Intelligent Batch Sizing Decision

When creating a batch, the scheduler must:

1. **Read pending work**: Check atomic pending token counter
2. **Check active capacity**: Sum tokens in all active batches
3. **Make sizing decision**:
   - If `pending_tokens < n_batch`: Create batch sized to pending work (avoid waste)
   - If `pending_tokens >= n_batch` AND room in context: Create full n_batch size
   - If at context capacity: Wait or create smaller batch to fit

```rust
let pending = self.pending_tokens.load(Ordering::Relaxed);
let active_info = self.active_batches.lock().unwrap();
let total_active_tokens: usize = active_info.iter().map(|b| b.token_count).sum();

let batch_size = if pending < n_batch {
    pending  // Light load: exact size
} else if total_active_tokens + n_batch <= context_size {
    n_batch  // Heavy load + capacity: full size
} else {
    std::cmp::min(pending, context_size - total_active_tokens)  // At capacity
};
```

### Concurrent Batch Creation

When queue has significant pending work AND context capacity available:

```rust
let max_batches = context_size / n_batch;
let pending = self.pending_tokens.load(Ordering::Relaxed);
let active_count = self.active_batches.lock().unwrap().len();

if pending >= n_batch * 2 && active_count < max_batches {
    // Create multiple batches to maximize throughput
    let num_batches = std::cmp::min(
        pending / n_batch,
        max_batches - active_count
    );

    for _ in 0..num_batches {
        spawn_batch_processing_task(n_batch);
    }
}
```

### Key Constraints

- **Active batch limit**: `total_active_tokens <= context_size`
- **Individual sequence limit**: Each sequence `<= n_ubatch` (embeddings can't split)
- **Batch capacity**: Pack multiple sequences up to `n_batch` total tokens
- **No n_seq_max packing constraint**: Let llama.cpp handle sequential processing internally

---

## Leveraging Existing Crates

The batch scheduler can be implemented using crates already in our dependency tree (via `server` feature):

### Already Available

- **`tokio::sync::mpsc`** - Unbounded/bounded MPSC channels for request queue
- **`tokio::sync::oneshot`** - One-shot channels for request/response coordination
- **`tokio::spawn`** - Spawn concurrent batch processing tasks
- **`parking_lot::Mutex`** - Faster mutex for active batch tracking (vs `std::sync::Mutex`)
- **`uuid::Uuid`** - Unique IDs for batch tracking (v4 feature enabled)
- **`prometheus`** - Metrics collection (already optional with server feature)

### Add Only One Crate

- **`metrics = "0.24"`** - Metrics facade for cleaner observability abstraction

### Producer-Consumer Architecture Pattern

```rust
use tokio::sync::{mpsc, oneshot};
use parking_lot::Mutex;
use uuid::Uuid;

struct BatchScheduler {
    /// Atomic counter of total tokens waiting in queue
    pending_tokens: AtomicUsize,

    /// Track in-flight batches
    active_batches: Mutex<Vec<ActiveBatch>>,

    /// Work queue for incoming requests
    request_rx: Mutex<mpsc::UnboundedReceiver<BatchRequest>>,
    request_tx: mpsc::UnboundedSender<BatchRequest>,

    /// Reference to the embedding model
    model: Arc<Mutex<EmbeddingModel>>,
}

struct BatchRequest {
    texts: Vec<String>,
    response_tx: oneshot::Sender<Result<Vec<Vec<f32>>>>,
    token_count: usize,
}

impl BatchScheduler {
    fn new(model: EmbeddingModel) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();

        Self {
            pending_tokens: AtomicUsize::new(0),
            active_batches: Mutex::new(Vec::new()),
            request_rx: Mutex::new(rx),
            request_tx: tx,
            model: Arc::new(Mutex::new(model)),
        }
    }

    /// Get a cloneable sender for Axum handlers
    fn get_sender(&self) -> mpsc::UnboundedSender<BatchRequest> {
        self.request_tx.clone()
    }
}
```

### Axum Handler Pattern

```rust
async fn embed_handler(
    State(scheduler): State<Arc<BatchScheduler>>,
    Json(request): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, Error> {
    let (tx, rx) = oneshot::channel();
    let token_count = estimate_tokens(&request.texts);

    // Send to work queue
    scheduler.get_sender().send(BatchRequest {
        texts: request.texts,
        response_tx: tx,
        token_count,
    }).map_err(|_| Error::SchedulerShutdown)?;

    // Update pending counter
    scheduler.pending_tokens.fetch_add(token_count, Ordering::Relaxed);

    // Wait for response
    let embeddings = rx.await.map_err(|_| Error::ResponseDropped)??;

    // Decrement pending counter
    scheduler.pending_tokens.fetch_sub(token_count, Ordering::Relaxed);

    Ok(Json(EmbedResponse { embeddings }))
}
```

### Model Worker Loop

```rust
async fn model_worker(scheduler: Arc<BatchScheduler>) {
    let mut batch_requests = Vec::new();
    let n_batch = scheduler.model.lock().n_batch() as usize;

    loop {
        // Collect requests up to n_batch capacity
        let mut rx = scheduler.request_rx.lock();

        while let Ok(req) = rx.try_recv() {
            batch_requests.push(req);

            let total_tokens: usize = batch_requests.iter()
                .map(|r| r.token_count)
                .sum();

            if total_tokens >= n_batch {
                break;
            }
        }
        drop(rx); // Release lock

        if !batch_requests.is_empty() {
            let batch_id = Uuid::new_v4();

            // Track active batch
            let token_count: usize = batch_requests.iter()
                .map(|r| r.token_count)
                .sum();

            scheduler.active_batches.lock().push(ActiveBatch {
                id: batch_id,
                token_count,
                created_at: Instant::now(),
            });

            // Process batch
            let result = process_batch(&scheduler.model, &batch_requests).await;

            // Remove from active tracking
            scheduler.active_batches.lock()
                .retain(|b| b.id != batch_id);

            // Send responses
            for (req, embedding) in batch_requests.drain(..).zip(result.iter()) {
                let _ = req.response_tx.send(Ok(embedding.clone()));
            }
        } else {
            // No work available, yield
            tokio::time::sleep(Duration::from_micros(100)).await;
        }
    }
}
```

### Benefits of This Approach

1. **Minimal new dependencies** - Only `metrics` crate added
2. **Single model ownership** - Model stays on one thread, no Arc<Mutex> overhead during processing
3. **Natural backpressure** - tokio channels provide flow control
4. **Simple coordination** - Channels handle multi-producer complexity
5. **Works with !Send models** - Model never crosses thread boundaries

---

## Implementation Plan

---

## Phase 1: Configuration Infrastructure

### Goals
- Add `n_batch` parameter to configuration
- Set appropriate defaults
- Validate relationships between parameters

### Tasks

#### 1.1 Add n_batch to ModelConfig
- [x] Add `pub n_batch: Option<u32>` field to `ModelConfig` struct (src/config.rs:23)
- [x] Add builder method `with_n_batch(mut self, n_batch: u32) -> Self`
- [x] Add validation in `ModelConfig::validate()`:
  - [x] If set, `n_batch > 0`
  - [x] If both set, `n_batch >= n_ubatch` (batch capacity ≥ processing chunk)
  - [x] **Recommendation**: For embedding workloads, set `n_ubatch = n_batch` to ensure atomic sequence processing
  - [x] Document: "Embedding models cannot split sequences; largest sequence must be < n_batch"

#### 1.2 Set Architecture-Aware Defaults
- [x] Update `ModelConfig::default()` with `n_batch: None`
- [x] In `EmbeddingModel::new()`, set defaults:
  > NOTE: Used 2048 default for all models (both decoder and encoder) as requested
  - Default: 2048 (matches llama-server default)
  - If `n_ubatch` is not set, default it to `n_batch` for atomic sequence processing

#### 1.3 Documentation
- [x] Add doc comments explaining the three parameters with proper semantics
  - n_batch: Batch packing capacity (total tokens)
  - n_ubatch: Physical processing chunk (defaults to n_batch if not set)
  - n_seq_max: Parallel KV cache sequence slots

#### 1.4 Wire Through Configuration Layers
- [x] Add `n_batch` to `EngineConfigBuilder` convenience methods
- [x] Add `n_batch` to `ServerConfig` (src/server/state.rs:46)
- [x] Add `n_batch` to `ServerConfigBuilder`
- [x] Wire through in `AppState::new()` to model configuration

#### 1.5 Add CLI Support
- [x] Add `--n-batch` / `-b` flag to server binary (src/bin/server.rs)
- [x] Add environment variable support: `EMBELLAMA_N_BATCH`
- [x] Update startup logging to show n_batch value

**Acceptance Criteria**:
- [x] Configuration builds with all three parameters
- [x] Validation enforces `n_batch >= n_ubatch`
- [x] Default is 2048 for all models
- [x] All configuration paths support n_batch
- [x] When n_ubatch is not set, it defaults to n_batch
- [x] All tests pass

---

## Phase 2: Model Layer Updates

### Goals
- Store and expose `n_batch` in EmbeddingModel
- Pre-allocate batches to capacity
- Prepare for batch reuse

### Tasks

#### 2.1 Add n_batch to EmbeddingModel
- [x] Add `n_batch: u32` field to `EmbeddingModel` struct (src/model.rs:89)
- [x] Extract from config in `EmbeddingModel::new()` constructor
- [x] Add accessor method:
  ```rust
  /// Returns the batch packing capacity (total tokens)
  pub fn n_batch(&self) -> u32 {
      self.n_batch
  }
  ```

#### 2.2 Update Batch Allocation Strategy
- [ ] Update `process_batch_tokens_internal()` (src/model.rs:663):
  ```rust
  // OLD: let mut batch = LlamaBatch::new(total_tokens, 1);
  // NEW:
  let batch_capacity = usize::try_from(self.n_batch)?;
  let mut batch = LlamaBatch::new(batch_capacity, 1);
  ```
- [ ] Update `process_tokens_internal()` (src/model.rs:955):
  ```rust
  // OLD: let mut batch = LlamaBatch::new(n_tokens, 1);
  // NEW:
  let batch_capacity = usize::try_from(self.n_batch)?;
  let mut batch = LlamaBatch::new(batch_capacity, 1);
  ```

#### 2.3 Add Validation and Logging
- [x] Log effective batch configuration on model load:
  ```rust
  info!(
      "Batch configuration: n_batch={}, n_ubatch={}, n_seq_max={}",
      n_batch, n_ubatch, n_seq_max
  );
  ```
- [ ] Add debug assertion that `total_tokens <= n_batch` when adding to batch
- [ ] Update existing log messages to distinguish batch capacity from sequence count

**Acceptance Criteria**:
- [x] EmbeddingModel stores and exposes n_batch
- [ ] Batches pre-allocated to n_batch capacity (pending Phase 2.2)
- [x] Clear logging of batch parameters
- [x] No functional changes to processing logic yet

#### 2.4 Add Batch Scheduler State (NEW)
- [ ] Use `tokio::sync::mpsc::unbounded_channel` for request queue
- [ ] Use `tokio::sync::oneshot` for request/response coordination
- [ ] Use `parking_lot::Mutex` for active batch tracking (faster than std)
- [ ] Implementation:
  ```rust
  use tokio::sync::{mpsc, oneshot};
  use parking_lot::Mutex;
  use uuid::Uuid;

  struct BatchScheduler {
      pending_tokens: AtomicUsize,
      active_batches: Mutex<Vec<ActiveBatch>>,
      request_rx: Mutex<mpsc::UnboundedReceiver<BatchRequest>>,
      request_tx: mpsc::UnboundedSender<BatchRequest>,
      model: Arc<Mutex<EmbeddingModel>>,
  }

  struct BatchRequest {
      texts: Vec<String>,
      response_tx: oneshot::Sender<Result<Vec<Vec<f32>>>>,
      token_count: usize,
  }
  ```
- [ ] Update pending_tokens when requests arrive/complete to maintain accurate counts
- [ ] Log queue depth and active batch count for monitoring

#### 2.5 Intelligent Batch Sizing (NEW)
- [ ] Implement smart batch allocation based on queue state:
  ```rust
  fn determine_batch_size(&self) -> usize {
      let pending = self.pending_tokens.load(Ordering::Relaxed);

      // Use parking_lot::Mutex (no .unwrap() needed, never panics)
      let active_info = self.active_batches.lock();
      let total_active: usize = active_info.iter().map(|b| b.token_count).sum();
      let n_batch = self.n_batch as usize;

      if pending < n_batch {
          // Light load: allocate exactly what we need
          pending
      } else if total_active + n_batch <= self.context_size {
          // Heavy load + capacity: allocate full batch
          n_batch
      } else {
          // At capacity: fit what we can
          std::cmp::min(pending, self.context_size - total_active)
      }
  }
  ```
- [ ] Use queue-aware sizing instead of always allocating to n_batch
- [ ] Use `parking_lot::Mutex` for faster locking (no poisoning, no .unwrap())
- [ ] Add debug logging showing sizing decision rationale

#### 2.6 GPU OOM Retry Logic (NEW)
- [ ] Implement pragmatic retry strategy with exponential backoff:
  ```rust
  fn process_with_retry(&mut self, initial_size: usize) -> Result<Vec<Vec<f32>>> {
      let min_size = 64; // Minimum viable batch
      let mut batch_size = initial_size;

      loop {
          match self.process_batch(batch_size) {
              Ok(result) => return Ok(result),
              Err(Error::OutOfMemory) if batch_size > min_size => {
                  warn!("GPU OOM at batch_size={}, retrying with half", batch_size);
                  batch_size = batch_size / 2;
              }
              Err(e) => return Err(e),
          }
      }
  }
  ```
- [ ] Wrap batch processing calls with retry logic
- [ ] Log OOM events and retry attempts for monitoring
- [ ] Set sensible minimum batch size (e.g., 64 tokens)

---

## Phase 3: Batch Processor Refactoring

### Goals
- Update chunking logic to use `n_batch` instead of `n_seq_max`
- Improve batch packing efficiency
- Maintain correctness for individual sequence limits

### Tasks

#### 3.1 Update Chunking Logic in BatchProcessor
- [ ] Refactor `process_batch()` chunking condition (src/batch.rs:126-200)
- [ ] **OLD logic**:
  ```rust
  if total_tokens <= effective_max && token_sequences.len() <= n_seq_max {
      // Process all in single batch
  }
  ```
- [ ] **NEW logic**:
  ```rust
  let n_batch = model.n_batch() as usize;

  if total_tokens <= n_batch {
      // Process all in single batch - n_batch is the packing constraint
      // Individual sequences are already validated against effective_max
  }
  ```

#### 3.2 Update Chunk Building Logic
- [ ] Refactor chunk building loop (src/batch.rs:157-184)
- [ ] **OLD constraint**: `current_tokens + seq_len > effective_max || current_batch.len() >= n_seq_max`
- [ ] **NEW constraint**: `current_tokens + seq_len > n_batch` (ONLY token-based, NO n_seq_max check)
- [ ] **Remove `n_seq_max` from chunking decisions entirely**
  - llama.cpp handles sequential GPU processing internally
  - Prefilling with one copy in/out is faster even if GPU processes non-parallel
  - Let llama.cpp manage parallelism based on hardware capabilities
- [ ] Keep individual sequence validation against `n_ubatch` (embedding constraint)

#### 3.3 Optimize for Common Cases
- [ ] Add fast path for single-sequence batches:
  ```rust
  if token_sequences.len() == 1 {
      // Fast path: single sequence, no chunking needed
      return model.process_batch_tokens(token_sequences);
  }
  ```
- [ ] Add metrics/logging for batch packing efficiency:
  ```rust
  debug!(
      "Packed {} sequences ({} total tokens) into {} batches (n_batch={})",
      token_sequences.len(),
      total_tokens,
      num_batches,
      n_batch
  );
  ```

#### 3.4 Update Comments and Documentation
- [ ] Update comments to clarify:
  - `n_batch`: Controls when to split into multiple batches
  - `effective_max`: Per-sequence token limit (embedding constraint)
  - `n_seq_max`: Internal llama.cpp parallelism (not a packing constraint)
- [ ] Update module-level documentation in `batch.rs`

**Acceptance Criteria**:
- Chunking based on n_batch capacity, not n_seq_max
- Individual sequences still validated against n_ubatch (embedding constraint)
- More efficient packing (fewer chunks for same input)
- Clear documentation of constraints

#### 3.5 Concurrent Batch Processing (NEW)
- [ ] Enable creation of multiple batches when queue has significant pending work:
  ```rust
  use tokio::task::JoinHandle;

  fn spawn_concurrent_batches(&self) -> Vec<JoinHandle<Result<()>>> {
      let n_batch = self.n_batch as usize;
      let max_batches = self.context_size / n_batch;
      let pending = self.pending_tokens.load(Ordering::Relaxed);

      // parking_lot::Mutex doesn't need .unwrap()
      let active_count = self.active_batches.lock().len();

      if pending >= n_batch * 2 && active_count < max_batches {
          // Create multiple batches to maximize throughput
          let num_batches = std::cmp::min(
              pending / n_batch,
              max_batches - active_count
          );

          (0..num_batches)
              .map(|_| {
                  let scheduler = Arc::clone(&self);
                  tokio::spawn(async move {
                      model_worker_single_batch(scheduler).await
                  })
              })
              .collect()
      } else {
          vec![]
      }
  }
  ```
- [ ] Use `tokio::spawn` to spawn batch processing tasks concurrently
- [ ] Track each active batch in `active_batches` with ID and token count
- [ ] Remove from `active_batches` when batch completes
- [ ] Ensure `total_active_tokens <= context_size` at all times

---

## Phase 4: Performance Optimization (Batch Reuse)

### Goals
- Reuse batch allocations across requests
- Reduce allocation overhead
- Measure performance improvements

### Tasks

#### 4.1 Choose Pooling Strategy (Optional Enhancement)

**Architecture Context**:
- Single model instance (or small number) shared across Axum request handlers
- Multiple Axum "threads" bound to same model instance
- Need coordinated access to model and batch pool

**Option A: Thread-Local Pool** (Simpler, Recommended)
- [ ] Evaluate if batch reuse provides measurable benefit
- [ ] Each Axum worker thread maintains its own batch pool
- [ ] No cross-thread synchronization needed for pool access
- [ ] Requires shared state tracking (AtomicUsize) for pending tokens
- [ ] Implementation:
  ```rust
  thread_local! {
      static BATCH_POOL: RefCell<Vec<LlamaBatch>> = RefCell::new(Vec::new());
  }

  fn get_or_create_batch(capacity: usize) -> LlamaBatch {
      BATCH_POOL.with(|pool| {
          pool.borrow_mut()
              .pop()
              .unwrap_or_else(|| LlamaBatch::new(capacity, 1))
      })
  }

  fn return_batch(batch: LlamaBatch) {
      BATCH_POOL.with(|pool| {
          let mut pool = pool.borrow_mut();
          if pool.len() < MAX_POOLED_BATCHES {
              batch.clear();  // Reset for reuse
              pool.push(batch);
          }
      });
  }
  ```

**Option B: Model-Aware Global Pool**
- [ ] If thread-local doesn't provide sufficient reuse
- [ ] Single global pool using `parking_lot::Mutex<Vec<LlamaBatch>>`
- [ ] Implementation:
  ```rust
  use parking_lot::Mutex;
  use once_cell::sync::Lazy;  // Already have this

  static BATCH_POOL: Lazy<Mutex<Vec<LlamaBatch>>> = Lazy::new(|| {
      Mutex::new(Vec::new())
  });

  const MAX_POOLED_BATCHES: usize = 32;

  fn get_or_create_batch(capacity: usize) -> LlamaBatch {
      BATCH_POOL.lock()
          .pop()
          .unwrap_or_else(|| LlamaBatch::new(capacity, 1))
  }

  fn return_batch(mut batch: LlamaBatch) {
      batch.clear();  // Reset for reuse
      let mut pool = BATCH_POOL.lock();
      if pool.len() < MAX_POOLED_BATCHES {
          pool.push(batch);
      }
  }
  ```

**Recommendation**: Start with **Option A (thread-local)** - simpler and avoids lock contention. Use Option B only if profiling shows it's beneficial.

#### 4.2 Measure Performance Impact
- [ ] Add benchmarks for batch allocation overhead
- [ ] Compare before/after metrics:
  - Allocations per request
  - Throughput (requests/sec)
  - Latency (p50, p95, p99)
- [ ] Document performance improvements in CHANGELOG

#### 4.3 Memory Management
- [ ] Ensure batches are properly cleared between uses
- [ ] Add mechanism to drain pool when idle (prevent memory leaks)
- [ ] Monitor memory usage under load

**Acceptance Criteria**:
- Measurable reduction in allocation overhead (if pool implemented)
- No memory leaks from batch reuse
- Performance benchmarks documented

---

## Phase 5: Testing and Validation

### Goals
- Ensure correctness across all scenarios
- Add regression tests for edge cases
- Validate against llama.cpp server behavior

### Tasks

#### 5.1 Unit Tests for Configuration
- [ ] Test `n_batch` validation:
  - [ ] Rejects `n_batch = 0`
  - [ ] Rejects `n_batch < n_ubatch`
  - [ ] Accepts valid configurations
- [ ] Test default selection:
  - [ ] Decoder models get conservative defaults
  - [ ] Encoder models get larger defaults
- [ ] Test configuration override via builder

#### 5.2 Integration Tests for Batch Processing
- [ ] Test single sequence (fast path)
- [ ] Test multiple sequences within n_batch (single batch)
- [ ] Test multiple sequences exceeding n_batch (chunking)
- [ ] Test edge case: exactly n_batch tokens
- [ ] Test edge case: sequences with varying lengths
- [ ] Test individual sequence limit (effective_max) enforcement

#### 5.3 Batch Overflow Tests
- [ ] Update `jina_model_batch_overflow_test.rs` for new packing logic
- [ ] Test sequences that should pack together now pack correctly
- [ ] Ensure individual sequence limits still enforced

#### 5.4 Property-Based Tests
- [ ] Add proptest for batch packing:
  - Generate random sequence sets
  - Verify chunks respect n_batch limit
  - Verify no sequence exceeds effective_max
  - Verify all sequences processed exactly once
- [ ] Test invariants:
  - `sum(chunk_tokens) == total_tokens`
  - `all(seq_tokens <= effective_max)`
  - `all(chunk_tokens <= n_batch)`

#### 5.5 Performance Regression Tests
- [ ] Benchmark suite for batch processing:
  - Small batches (1-5 sequences)
  - Medium batches (10-50 sequences)
  - Large batches (100+ sequences)
- [ ] Compare against baseline (current implementation)
- [ ] Verify no throughput regression
- [ ] Document expected improvements

#### 5.6 Cross-Reference with llama.cpp Server
- [ ] Test equivalent scenarios in both systems
- [ ] Verify packing behavior matches
- [ ] Document any intentional divergence

#### 5.7 Add Scheduler Metrics (NEW)
- [ ] Add `metrics = "0.24"` to dependencies in `Cargo.toml`
- [ ] Instrument batch scheduler with metrics:
  ```rust
  use metrics::{counter, histogram, gauge};

  // In scheduler - batch creation
  counter!("embellama.batches.created").increment(1);
  histogram!("embellama.batch.token_count").record(token_count as f64);
  histogram!("embellama.batch.utilization_pct").record(utilization * 100.0);

  // Queue metrics
  gauge!("embellama.queue.pending_tokens").set(pending_tokens as f64);
  histogram!("embellama.batch.queue_wait_ms").record(wait_time.as_millis() as f64);

  // Active batch tracking
  gauge!("embellama.batches.active_count").set(active_count as f64);
  gauge!("embellama.context.utilization_pct").record(context_util * 100.0);

  // OOM retries
  counter!("embellama.batch.oom_retries").increment(1);
  histogram!("embellama.batch.oom_retry_size").record(retry_size as f64);
  ```
- [ ] Export to prometheus using existing `prometheus` crate (already in server feature)
- [ ] Add `/metrics` endpoint to Axum router
- [ ] Document available metrics in API docs

**Acceptance Criteria**:
- All tests pass
- No regressions in correctness
- Measurable improvement in batch packing efficiency
- Documentation updated with test results
- Metrics instrumentation working and documented

---

## Phase 6: Documentation and Release

### Goals
- Update user-facing documentation
- Provide migration guide
- Document performance characteristics

### Tasks

#### 6.1 Update Configuration Documentation
- [ ] Add `n_batch` to configuration examples
- [ ] Provide tuning guidelines:
  - When to increase n_batch (high throughput scenarios)
  - When to decrease n_batch (memory-constrained environments)
  - Relationship to n_ubatch and n_seq_max
- [ ] Add troubleshooting section for batch-related issues

#### 6.2 Update API Documentation
- [ ] Update docstrings for new parameters
- [ ] Add examples of batch configuration
- [ ] Document performance characteristics

#### 6.3 Migration Guide
- [ ] Create MIGRATION.md if breaking changes
- [ ] Document behavioral changes:
  - Better batch packing (more sequences per batch)
  - New configuration parameter (n_batch)
  - Updated defaults
- [ ] Provide before/after configuration examples

#### 6.4 CHANGELOG Entry
- [ ] Add entry under "Performance" section:
  ```markdown
  ### Performance
  - **Improved batch packing efficiency** by introducing `n_batch` parameter
    - Decoupled batch packing capacity from parallel sequence processing
    - Pre-allocate batches to reduce allocation overhead
    - Better alignment with llama.cpp server behavior
    - Typical improvement: X% higher throughput for multi-sequence batches
  ```

#### 6.5 Blog Post / Release Notes (Optional)
- [ ] Write detailed explanation of batching architecture
- [ ] Include performance benchmarks
- [ ] Provide tuning recommendations
- [ ] Compare to llama.cpp server approach

**Acceptance Criteria**:
- All documentation updated
- Migration path clear for existing users
- Performance improvements quantified and documented

---

## Rollout Strategy

### Phase Dependencies
```
Phase 1 (Config) → Phase 2 (Model) → Phase 3 (Batch Logic) → Phase 4 (Optimization)
                                                              ↘
                                                    Phase 5 (Testing) → Phase 6 (Docs)
```

### Recommended Order
1. **Phase 1**: Configuration infrastructure (required foundation)
2. **Phase 2**: Model layer updates (enables new behavior)
3. **Phase 5.1-5.3**: Basic testing (validate correctness before optimization)
4. **Phase 3**: Batch processor refactoring (core logic changes)
5. **Phase 5.4-5.6**: Comprehensive testing (validate refactored logic)
6. **Phase 4**: Performance optimization (optional, can be deferred)
7. **Phase 6**: Documentation and release

### Risk Mitigation
- Each phase is independently testable
- Can feature-flag batch reuse (Phase 4) if needed
- Extensive testing before release
- Backward compatibility via defaults

---

## Success Metrics

### Performance
- [ ] **Batch packing efficiency**: Reduce number of batches by 30-50% for typical workloads
- [ ] **Allocation overhead**: Reduce batch allocations by 40-60% with pre-allocation/pooling
- [ ] **Throughput**: Improve multi-sequence throughput by 20-40% (target known bottleneck)
- [ ] **Memory usage**: Dynamic allocation prevents waste under light load
- [ ] **Baseline**: Single sequence and short sequences already fast (no regression expected)

### Scheduler Metrics (NEW)
- [ ] **Batch utilization**: Average percentage of n_batch capacity used per batch
- [ ] **Active batch count**: Average and peak number of concurrent batches
- [ ] **Queue wait time**: Time from request arrival to batch assignment (p50, p95, p99)
- [ ] **Context utilization**: Percentage of context_size used by active batches (average and peak)
- [ ] **Pending token count**: Distribution of pending tokens over time
- [ ] **Concurrent batch efficiency**: Throughput improvement from concurrent batch processing
- [ ] **GPU OOM retry rate**: Frequency of OOM events and successful retries

### Correctness
- [ ] All existing tests pass
- [ ] No regressions in embedding quality
- [ ] Individual sequence limits still enforced
- [ ] Numerical results match (within floating point tolerance)

### Code Quality
- [ ] Clear separation of concerns (packing vs. processing vs. parallelism)
- [ ] Well-documented parameter meanings
- [ ] Maintainable test suite
- [ ] Alignment with llama.cpp conventions

---

## Future Enhancements (Out of Scope)

### Dynamic Batch Sizing
- Automatically adjust n_batch based on memory pressure
- Retry with smaller batches on OOM (like llama.cpp server)

### Batch Splitting for Very Large Inputs
- Allow splitting of individual sequences across multiple batches
- Requires architectural changes (stateful processing)
- Only relevant if supporting generation in future

### Advanced Scheduling
- Priority queues for batch scheduling
- Batching delay tolerance for better packing
- Load balancing across workers

### Metrics and Observability
- Prometheus metrics for batch packing efficiency
- Tracing for batch lifecycle
- Alerts for suboptimal packing

---

## References

### llama.cpp Server Code
- `examples/server/server.cpp`: Batch packing and processing loop
- `common/common.cpp`: `common_batch_add()` implementation
- Context size handling and retry logic

### embellama Current Implementation
- `src/model.rs`: Batch allocation and processing
- `src/batch.rs`: Batch packing and chunking logic
- `src/config.rs`: Configuration parameters

### Related Issues
- Memory allocation overhead in high-throughput scenarios
- Premature chunking reducing batch efficiency
- Confusion between n_seq_max and batch capacity

---

## Dependency Changes

### Add to Cargo.toml

Only **one new crate** needs to be added:

```toml
[dependencies]
# ... existing dependencies ...

# NEW: Add metrics facade for observability
metrics = "0.24"

# OPTIONAL: If using prometheus exporter for metrics
metrics-exporter-prometheus = { version = "0.16", optional = true }
```

### Already Available (via server feature)

No need to add these - they're already present:
- ✅ `tokio` - async runtime with mpsc, oneshot channels
- ✅ `parking_lot` - faster mutexes
- ✅ `uuid` - batch IDs
- ✅ `prometheus` - metrics collection
- ✅ `once_cell` - lazy statics

### Update server feature (optional)

If adding prometheus exporter for metrics:
```toml
server = [
    "dep:axum", "dep:tokio", "dep:clap", "dep:async-trait",
    "dep:tower", "dep:tower-http", "dep:uuid", "dep:base64",
    "dep:governor", "dep:prometheus", "dep:rand", "dep:subtle",
    "dep:metrics-exporter-prometheus"  # Add this line
]
```

---

## Appendix: Parameter Quick Reference

| Parameter | Purpose | Constraint | Default (Decoder) | Default (Encoder) |
|-----------|---------|------------|-------------------|-------------------|
| `n_batch` | Batch packing capacity (total tokens) | `n_batch >= n_ubatch` | 2048 | 8192 |
| `n_ubatch` | Physical processing chunk size | Individual sequences ≤ n_ubatch (embedding constraint) | 512 | 2048 |
| `n_seq_max` | Parallel KV cache sequence slots | Internal llama.cpp limit ≤ 64; NOT used for packing | 2 | 8 |
| `effective_max` | Per-sequence token limit | `context_size - overhead` | Auto-detected | Auto-detected |
| `context_size` | Total model context window | Active batches must fit within this | Auto-detected | Auto-detected |

**Recommendation for Embeddings**: Set `n_ubatch = n_batch` to ensure atomic sequence processing and simplify configuration.

### Visual Example

```
Request with 3 sequences: [100 tokens, 200 tokens, 150 tokens] = 450 total tokens

Current behavior (n_seq_max=2):
  Batch 1: [100, 200] = 300 tokens (chunked by n_seq_max)
  Batch 2: [150] = 150 tokens

New behavior (n_batch=2048):
  Batch 1: [100, 200, 150] = 450 tokens (single batch, better packing!)

Processing within each batch:
  - llama.cpp internally chunks if total > n_ubatch
  - Each sequence validated against effective_max
  - n_seq_max validated internally by llama.cpp
```
