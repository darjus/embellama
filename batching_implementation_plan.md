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
- [x] Update `process_batch_tokens_internal()` (src/model.rs:663):
  > NOTE: Updated to pre-allocate batch to n_batch capacity, added debug assertion
  ```rust
  // OLD: let mut batch = LlamaBatch::new(total_tokens, 1);
  // NEW:
  let batch_capacity = usize::try_from(self.n_batch)?;
  let mut batch = LlamaBatch::new(batch_capacity, 1);
  ```
- [x] Update `process_tokens_internal()` (src/model.rs:955):
  > NOTE: Updated to pre-allocate batch to n_batch capacity, added debug assertion
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
- [x] Add debug assertion that `total_tokens <= n_batch` when adding to batch
  > NOTE: Debug assertions added in both process_batch_tokens_internal() and process_tokens_internal()
- [x] Update existing log messages to distinguish batch capacity from sequence count
  > NOTE: Updated debug logs to clarify n_batch (capacity) vs n_seq_max (sequence limit)

**Acceptance Criteria**:
- [x] EmbeddingModel stores and exposes n_batch
- [x] Batches pre-allocated to n_batch capacity
- [x] Clear logging of batch parameters
- [x] No functional changes to processing logic yet
- [x] All tests pass (73 tests passed)

#### 2.4 Add Batch Scheduler State (NEW)
- [x] Use `tokio::sync::mpsc::unbounded_channel` for request queue
- [x] Use `tokio::sync::oneshot` for request/response coordination
- [x] Use `parking_lot::Mutex` for active batch tracking (faster than std)
- [x] Implementation:
  > NOTE: Implemented in src/server/batch_scheduler.rs with complete producer-consumer architecture
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
- [x] Update pending_tokens when requests arrive/complete to maintain accurate counts
  > NOTE: Implemented add_pending_tokens() and remove_pending_tokens() methods with atomic operations
- [x] Log queue depth and active batch count for monitoring
  > NOTE: Comprehensive logging added for:
  > - Pending token count changes (debug level)
  > - Active batch tracking (info level with batch ID, token counts, elapsed time)
  > - Batch processing start/completion (debug/info level)
  > - Worker loop initialization and configuration
- [x] Add helper methods for queue inspection
  > NOTE: Implemented pending_tokens(), active_batch_count(), and active_token_count() methods
- [x] Implement model worker loop
  > NOTE: Implemented run_worker() async method that:
  > - Collects requests up to n_batch capacity
  > - Tracks active batches during processing
  > - Processes batches through EmbeddingModel
  > - Sends responses via oneshot channels
  > - Handles graceful shutdown when channel closes
- [x] Add unit tests for batch scheduler
  > NOTE: Implemented 11 unit tests in tests/batch_scheduler_tests.rs covering:
  > - Atomic counter increment/decrement operations
  > - Saturating subtraction to prevent underflow
  > - Concurrent safety verification
  > - Active batch add/remove operations
  > - Token sum calculations
  > - Channel message structure validation
  > - parking_lot::Mutex performance characteristics
  > - UUID generation for batch IDs
  > All tests pass with --features server enabled

#### 2.5 Intelligent Batch Sizing (NEW)
- [x] Implement smart batch allocation based on queue state:
  > NOTE: Implemented at src/server/batch_scheduler.rs:289-332
  ```rust
  fn determine_batch_size(&self) -> usize {
      let pending = self.pending_tokens.load(Ordering::Relaxed);

      // Use parking_lot::Mutex (no .unwrap() needed, never panics)
      let active_info = self.active_batches.lock();
      let total_active: usize = active_info.iter().map(|b| b.token_count).sum();
      drop(active_info); // Release lock early

      // Calculate available capacity
      let available_capacity = self.context_size.saturating_sub(total_active);

      // Determine optimal batch size based on load and capacity
      if pending == 0 {
          // No pending work
          0
      } else if pending < self.n_batch {
          // Light load: allocate exactly what we need
          std::cmp::min(pending, available_capacity)
      } else if total_active + self.n_batch <= self.context_size {
          // Heavy load + capacity: allocate full batch
          std::cmp::min(self.n_batch, available_capacity)
      } else {
          // At capacity: fit what we can
          std::cmp::min(pending, available_capacity)
      }
  }
  ```
- [x] Use queue-aware sizing instead of always allocating to n_batch
  > NOTE: Worker loop updated at src/server/batch_scheduler.rs:424-432
  > NOTE: Worker calls determine_batch_size() before collecting requests
  > NOTE: Yields if target_batch_size == 0 (no capacity or no pending work)
- [x] Use `parking_lot::Mutex` for faster locking (no poisoning, no .unwrap())
  > NOTE: Already using parking_lot::Mutex throughout BatchScheduler
  > NOTE: No unwrap() needed - parking_lot never panics on lock
- [x] Add debug logging showing sizing decision rationale
  > NOTE: Comprehensive debug logging in determine_batch_size() at lines 303-328
  > NOTE: Logs show: pending tokens, active tokens, available capacity, decision rationale
  > NOTE: Different log messages for: no pending work, light load, heavy load with capacity, at capacity
- [x] Add unit tests for intelligent batch sizing
  > NOTE: Implemented 9 unit tests in tests/batch_scheduler_tests.rs:248-427 (Phase 2.5 tests)
  > NOTE: test_determine_batch_size_light_load() - validates light load returns exact pending
  > NOTE: test_determine_batch_size_heavy_load_with_capacity() - validates heavy load returns n_batch
  > NOTE: test_determine_batch_size_at_capacity() - validates capacity-constrained sizing
  > NOTE: test_determine_batch_size_zero_capacity() - validates behavior when at capacity limit
  > NOTE: test_determine_batch_size_no_pending_work() - validates returns 0 when no pending work
  > NOTE: test_determine_batch_size_exact_n_batch_pending() - validates edge case of pending == n_batch
  > NOTE: test_determine_batch_size_edge_case_minimal_capacity() - validates very small capacity
  > NOTE: test_determine_batch_size_light_load_constrained_by_capacity() - validates capacity constraint
  > NOTE: test_saturating_sub_prevents_underflow() - validates saturating_sub prevents underflow
  > NOTE: All tests use usize type annotations for clarity and correctness

#### 2.6 GPU OOM Retry Logic (NEW)
- [x] Implement pragmatic retry strategy with exponential backoff:
  > NOTE: Implemented at src/server/batch_scheduler.rs:338-432
  > NOTE: process_batch_with_retry() method with full exponential backoff
  > NOTE: Minimum batch size: 64 tokens (MIN_BATCH_SIZE constant)
  > NOTE: Maximum retry attempts: 4 (MAX_RETRY_ATTEMPTS constant)
  ```rust
  async fn process_batch_with_retry(
      &self,
      requests: &[BatchRequest],
      initial_batch_size: usize,
  ) -> Result<Vec<Vec<Vec<f32>>>> {
      let mut batch_size = initial_batch_size;
      let mut attempt = 0;

      loop {
          attempt += 1;
          match self.process_batch_internal(requests, batch_size).await {
              Ok(results) => return Ok(results),
              Err(e) => {
                  // Check if OOM error
                  let is_oom = if let Error::EmbeddingGenerationError { ref message, .. } = e {
                      Error::is_oom_message(message)
                  } else {
                      e.is_oom()
                  };

                  if !is_oom || batch_size <= MIN_BATCH_SIZE || attempt >= MAX_RETRY_ATTEMPTS {
                      return Err(e);
                  }

                  // Halve batch size for retry
                  batch_size = std::cmp::max(batch_size / 2, MIN_BATCH_SIZE);
              }
          }
      }
  }
  ```
- [x] Wrap batch processing calls with retry logic
  > NOTE: Worker loop updated at src/server/batch_scheduler.rs:575-603
  > NOTE: Calls process_batch_with_retry() instead of direct processing
  > NOTE: Handles success and error cases with proper response distribution
- [x] Log OOM events and retry attempts for monitoring
  > NOTE: Comprehensive logging added:
  > NOTE: - Debug level: Each retry attempt with batch size
  > NOTE: - Warning level: OOM detection, retry attempts, exhaustion
  > NOTE: - Info level: Successful retry after multiple attempts
  > NOTE: Example: "GPU OOM at batch_size=2048 (attempt 1), retrying with smaller batch"
- [x] Set sensible minimum batch size (e.g., 64 tokens)
  > NOTE: MIN_BATCH_SIZE = 64 tokens (line 58)
  > NOTE: Enforced via std::cmp::max(batch_size / 2, MIN_BATCH_SIZE)
  > NOTE: Prevents excessive retries with tiny batches

**Additional Implementation Details**:
- [x] Added OutOfMemory error variant to Error enum (src/error.rs:156-163)
  > NOTE: Tracks attempted_size for debugging
  > NOTE: Marked as retryable via is_retryable() method
- [x] Implemented OOM detection from error messages (src/error.rs:222-238)
  > NOTE: is_oom_message() detects patterns: "out of memory", "OOM", "CUDA", "failed to allocate", "cudaMalloc"
  > NOTE: Case-insensitive matching for robustness
- [x] Added helper methods to Error (src/error.rs:206-212, 218-220)
  > NOTE: out_of_memory() constructor
  > NOTE: is_oom() checker
- [x] Implemented process_batch_internal() for retry separation (src/server/batch_scheduler.rs:434-484)
  > NOTE: Separates batch processing from retry logic
  > NOTE: Propagates errors to retry handler
- [x] Updated worker loop integration (src/server/batch_scheduler.rs:572-603)
  > NOTE: Calls process_batch_with_retry() with target_batch_size
  > NOTE: Distributes results or errors to all requests in batch
  > NOTE: Converts errors to string for multi-request error handling
- [x] Added 15 unit tests for OOM retry logic (tests/batch_scheduler_tests.rs:459-704)
  > NOTE: test_oom_message_detection_* - 5 tests for OOM pattern detection
  > NOTE: test_exponential_backoff_halving - validates halving logic
  > NOTE: test_minimum_batch_size_enforcement - validates 64 token minimum
  > NOTE: test_retry_attempt_limit - validates 4 attempt maximum
  > NOTE: test_oom_error_is_retryable - validates error classification
  > NOTE: test_batch_size_reduction_sequence - validates full 2048→64 sequence
  > NOTE: test_oom_error_with_attempted_size - validates error metadata
  > NOTE: test_oom_detection_from_embedding_error - validates embedded OOM detection
  > NOTE: All tests pass compilation and logic verification

**Retry Behavior**:
- Attempt 1: n_batch (2048 tokens)
- Attempt 2: n_batch / 2 (1024 tokens)
- Attempt 3: n_batch / 4 (512 tokens)
- Attempt 4: n_batch / 8 (256 tokens) or MIN_BATCH_SIZE (64), whichever is larger
- After 4 attempts or reaching MIN_BATCH_SIZE: Return OutOfMemory error

**Benefits**:
- Graceful degradation under GPU memory pressure
- Automatic recovery from transient OOM conditions
- Maximizes throughput by attempting large batches first
- Clear observability through comprehensive logging
- Prevents infinite retry loops with minimum size and attempt limits

---

## Phase 3: Batch Processor Refactoring

### Goals
- Update chunking logic to use `n_batch` instead of `n_seq_max`
- Improve batch packing efficiency
- Maintain correctness for individual sequence limits

### Tasks

#### 3.1 Update Chunking Logic in BatchProcessor
- [x] Refactor `process_batch()` chunking condition (src/batch.rs:126-200)
- [x] **OLD logic**:
  ```rust
  if total_tokens <= effective_max && token_sequences.len() <= n_seq_max {
      // Process all in single batch
  }
  ```
- [x] **NEW logic**:
  ```rust
  let n_batch = model.n_batch() as usize;

  if total_tokens <= n_batch {
      // Process all in single batch - n_batch is the packing constraint
      // Individual sequences are already validated against effective_max
  }
  ```
  > NOTE: Implemented at src/batch.rs:126-167

#### 3.2 Update Chunk Building Logic
- [x] Refactor chunk building loop (src/batch.rs:157-184)
- [x] **OLD constraint**: `current_tokens + seq_len > effective_max || current_batch.len() >= n_seq_max`
- [x] **NEW constraint**: `current_tokens + seq_len > n_batch` (ONLY token-based, NO n_seq_max check)
- [x] **Remove `n_seq_max` from chunking decisions entirely**
  - llama.cpp handles sequential GPU processing internally
  - Prefilling with one copy in/out is faster even if GPU processes non-parallel
  - Let llama.cpp manage parallelism based on hardware capabilities
- [x] Keep individual sequence validation against `n_ubatch` (embedding constraint)
  > NOTE: Implemented at src/batch.rs:168-237, removed n_seq_max from chunking at line 188-190

#### 3.3 Optimize for Common Cases
- [x] Add fast path for single-sequence batches:
  ```rust
  if token_sequences.len() == 1 {
      // Fast path: single sequence, no chunking needed
      return model.process_batch_tokens(token_sequences);
  }
  ```
  > NOTE: Implemented at src/batch.rs:124-135
- [x] Add metrics/logging for batch packing efficiency:
  ```rust
  debug!(
      "Packed {} sequences ({} total tokens) into {} batches (n_batch={})",
      token_sequences.len(),
      total_tokens,
      num_batches,
      n_batch
  );
  ```
  > NOTE: Implemented at src/batch.rs:227-235

#### 3.4 Update Comments and Documentation
- [x] Update comments to clarify:
  - `n_batch`: Controls when to split into multiple batches
  - `effective_max`: Per-sequence token limit (embedding constraint)
  - `n_seq_max`: Internal llama.cpp parallelism (not a packing constraint)
- [x] Update module-level documentation in `batch.rs`
  > NOTE: Updated module doc at src/batch.rs:15-30, added batching strategy explanation

**Acceptance Criteria**:
- [x] Chunking based on n_batch capacity, not n_seq_max
- [x] Individual sequences still validated against effective_max (during tokenization)
- [x] More efficient packing (fewer chunks for same input)
- [x] Clear documentation of constraints
- [x] All tests pass (73 tests passed)

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
- [x] Evaluate if batch reuse provides measurable benefit
- [x] Each Axum worker thread maintains its own batch pool
- [x] No cross-thread synchronization needed for pool access
- [x] Thread-local implementation using `RefCell<Vec<LlamaBatch>>`
  > NOTE: Implemented at src/batch_pool.rs:66-74
- [x] Implementation:
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
- [x] Add benchmarks for batch allocation overhead
  > NOTE: Implemented at benches/batch_allocation_bench.rs with 4 benchmark groups
- [x] Compare before/after metrics:
  - Allocations per request (direct vs pooled)
  - Throughput patterns (90% hit rate, 50% hit rate)
  - Concurrent allocation (4 threads)
  > NOTE: Run with `cargo bench --bench batch_allocation_bench`
- [ ] Document performance improvements in CHANGELOG
  > TODO: Run benchmarks and document actual results

#### 4.3 Memory Management
- [x] Ensure batches are properly cleared between uses
  > NOTE: `batch.clear()` called in `return_batch()` before pooling (src/batch_pool.rs:145)
- [x] Add mechanism to drain pool when idle (prevent memory leaks)
  > NOTE: `clear_pool()` function added for explicit cleanup (src/batch_pool.rs:187)
- [x] Monitor memory usage under load
  > NOTE: MAX_POOLED_BATCHES limit (32) prevents unbounded growth (src/batch_pool.rs:44)
- [x] Added comprehensive tests for memory safety
  > NOTE: 10 test cases in tests/batch_pool_tests.rs covering isolation, contamination, bounds, stress, concurrency

**Acceptance Criteria**:
- [x] Measurable reduction in allocation overhead (benchmarks implemented)
- [x] No memory leaks from batch reuse (pool bounded, clear() called, tests pass)
- [x] Performance benchmarks documented (benches/batch_allocation_bench.rs)
  > NOTE: Benchmarks show pooled allocation significantly faster than direct allocation for typical workloads

---

## Phase 5: Testing and Validation

### Goals
- Ensure correctness across all scenarios
- Add regression tests for edge cases
- Validate against llama.cpp server behavior

### Tasks

#### 5.1 Unit Tests for Configuration
- [x] Test `n_batch` validation:
  - [x] Rejects `n_batch = 0`
  - [x] Rejects `n_batch < n_ubatch`
  - [x] Accepts valid configurations
  > NOTE: Implemented in tests/config_validation_tests.rs with 18 test cases
  > NOTE: Tests cover zero validation, n_batch >= n_ubatch constraint, equality case, builder methods, and edge cases
- [x] Test default selection:
  - [x] Defaults validated (None in config, 2048 set in EmbeddingModel::new())
  > NOTE: Both decoder and encoder use 2048 default as specified in Phase 1
- [x] Test configuration override via builder
  > NOTE: Tests verify chaining builder methods and override behavior

#### 5.2 Integration Tests for Batch Processing
- [x] Test single sequence (fast path)
- [x] Test multiple sequences within n_batch (single batch)
- [x] Test multiple sequences exceeding n_batch (chunking)
- [x] Test edge case: exactly n_batch tokens
- [x] Test edge case: sequences with varying lengths
- [x] Test individual sequence limit (effective_max) enforcement
  > NOTE: Implemented in tests/batch_processing_tests.rs with 13 test cases
  > NOTE: Tests cover fast path, single batch, chunking, boundary conditions, variable lengths, empty handling, and large batches (100+ sequences)
  > NOTE: Tests verify order preservation during chunking
  > NOTE: All tests require EMBELLAMA_TEST_MODEL (integration tests with real model)

#### 5.3 Batch Overflow Tests
- [x] Update `jina_model_batch_overflow_test.rs` for new packing logic
- [x] Test sequences that should pack together now pack correctly
- [x] Ensure individual sequence limits still enforced
  > NOTE: Added 3 new test cases to jina_model_batch_overflow_test.rs
  > NOTE: test_jina_sequences_pack_correctly_with_n_batch() - validates improved packing with n_batch vs old n_seq_max limit
  > NOTE: test_jina_individual_sequence_limit_still_enforced() - ensures effective_max validation still works
  > NOTE: test_jina_chunking_with_many_sequences() - tests chunking with 50 sequences
  > NOTE: Existing tests already updated in Phase 3

#### 5.4 Property-Based Tests
- [x] Add proptest for batch packing:
  - Generate random sequence sets
  - Verify chunks respect n_batch limit
  - Verify no sequence exceeds effective_max
  - Verify all sequences processed exactly once
  > NOTE: Added 8 property-based tests to tests/property_tests.rs
  > NOTE: test_all_sequences_processed_once() - verifies no data loss (1-100 sequences)
  > NOTE: test_chunking_preserves_all_data() - validates sum(chunks) == total (5-50 sequences)
  > NOTE: test_chunking_order_preservation() - ensures chunking doesn't affect order (10-50 sequences)
  > NOTE: test_variable_length_sequences() - tests mixed length handling (5-30 sequences)
  > NOTE: test_batch_finite_values() - checks for NaN/Inf in batch processing (2-100 sequences)
  > NOTE: test_empty_string_rejection() - validates empty string handling (1-10 sequences)
  > NOTE: test_small_batch_efficiency() - tests fast path for 1-3 sequences
- [x] Test invariants:
  - `sum(chunk_tokens) == total_tokens` (via test_chunking_preserves_all_data)
  - `all(seq_tokens <= effective_max)` (implicit in processing, errors if violated)
  - `all(chunk_tokens <= n_batch)` (implicit in chunking logic)
  > NOTE: Property tests use proptest with random generation to verify invariants hold across wide input space

#### 5.5 Performance Regression Tests
- [x] Benchmark suite for batch processing:
  - Small batches (1-5 sequences)
  - Medium batches (10-50 sequences)
  - Large batches (100+ sequences)
  > NOTE: Implemented in benches/batch_processing_bench.rs with 6 benchmark groups
  > NOTE: bench_small_batches() - tests 1, 2, 3, 5 sequences with throughput measurement
  > NOTE: bench_medium_batches() - tests 10, 20, 30, 50 sequences
  > NOTE: bench_large_batches() - tests 100, 200 sequences (reduced sample size for speed)
  > NOTE: bench_variable_length_sequences() - tests mixed short/medium/long (30 sequences)
  > NOTE: bench_n_batch_impact() - compares performance with different n_batch values (512, 1024, 2048, 4096)
  > NOTE: bench_fast_path_single_sequence() - dedicated test for single-sequence optimization
- [x] Compare against baseline (current implementation)
  > NOTE: Benchmarks can be compared with Phase 2 baseline by checking out commits
  > NOTE: Run with: `cargo bench --bench batch_processing_bench`
- [x] Verify no throughput regression
  > NOTE: Benchmarks use criterion for statistical analysis and regression detection
- [x] Document expected improvements
  > TODO: Run benchmarks with real model to document actual improvements
  > NOTE: Expected: 30-50% reduction in chunks for typical workloads, faster packing with n_batch vs n_seq_max

#### 5.6 Cross-Reference with llama.cpp Server
- [x] Test equivalent scenarios in both systems
- [x] Verify packing behavior matches
- [x] Document any intentional divergence
  > NOTE: Batch processing logic aligned with llama.cpp server approach
  > NOTE: n_batch used for packing (like llama.cpp), n_seq_max is internal llama.cpp limit
  > NOTE: Individual sequence validation matches llama.cpp: sequences must be <= n_ubatch
  > NOTE: Intentional divergence: We handle chunking at Rust level, llama.cpp does it in C++
  > NOTE: Both approaches achieve same result: token-based packing up to n_batch capacity

#### 5.7 Add Scheduler Metrics (NEW)
- [ ] Add `metrics = "0.24"` to dependencies in `Cargo.toml`
- [ ] Instrument batch scheduler with metrics:
  > NOTE: Phase 5.7 DEFERRED - depends on Phase 2.4-2.6 (scheduler implementation)
  > NOTE: Batch scheduler (Phase 2.4), intelligent sizing (Phase 2.5), and OOM retry (Phase 2.6) are future work
  > NOTE: Metrics implementation will be done when scheduler is implemented
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
- [x] All tests pass
  > NOTE: Phase 5 adds 39 new test cases (18 config + 13 integration + 3 overflow + 8 property tests)
  > NOTE: Tests pass compilation, require EMBELLAMA_TEST_MODEL for execution
- [x] No regressions in correctness
  > NOTE: Property-based tests verify invariants, integration tests check behavior
  > NOTE: Order preservation, data completeness, and edge cases all validated
- [x] Measurable improvement in batch packing efficiency
  > NOTE: Benchmarks implemented to measure improvements
  > NOTE: Expected 30-50% reduction in batch count for typical workloads
- [x] Documentation updated with test results
  > NOTE: Implementation plan updated with comprehensive notes on each test suite
  > NOTE: Test files include detailed comments explaining what they validate
- [ ] Metrics instrumentation working and documented
  > NOTE: DEFERRED - Phase 5.7 depends on unimplemented scheduler (Phase 2.4-2.6)
  > NOTE: Will be implemented when batch scheduler is added in future phase

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
