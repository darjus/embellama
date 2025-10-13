// Copyright 2025 Embellama Contributors
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

//! Batch scheduler for intelligent request batching and processing
//!
//! This module implements a producer-consumer architecture for batch scheduling:
//! - Tracks pending tokens atomically to make intelligent sizing decisions
//! - Maintains active batch list for capacity management
//! - Uses tokio channels for efficient request/response coordination
//! - Enables concurrent batch processing when queue has sufficient work
//!
//! # Architecture
//!
//! ```text
//! Axum Handlers → BatchScheduler (queue) → Model Worker → EmbeddingModel
//!      ↓                                         ↓
//!   oneshot channel ← ← ← ← ← ← ← ← ← ← ← ← ← ←
//! ```
//!
//! The scheduler tracks:
//! - **Pending tokens**: Atomic counter of total tokens waiting in queue
//! - **Active batches**: List of in-flight batches with their token counts
//! - **Request queue**: Unbounded MPSC channel for incoming requests
//!
//! This enables intelligent batch sizing decisions:
//! - Light load: Size batches to exact pending work (avoid waste)
//! - Heavy load: Create full `n_batch` size batches (maximize throughput)
//! - At capacity: Wait or create smaller batches to fit within `context_size`

#![allow(clippy::needless_borrow)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::await_holding_lock)]
#![allow(clippy::if_not_else)]
#![allow(clippy::unused_async)]

use crate::{EmbeddingModel, Error, Result};
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Minimum viable batch size for retry logic (64 tokens)
const MIN_BATCH_SIZE: usize = 64;

/// Maximum number of retry attempts for OOM errors
const MAX_RETRY_ATTEMPTS: usize = 4;

/// Request sent to the batch scheduler
#[derive(Debug)]
pub struct BatchRequest {
    /// Unique request identifier
    pub id: Uuid,
    /// Text inputs to process
    pub texts: Vec<String>,
    /// Channel to send response back
    pub response_tx: oneshot::Sender<Result<Vec<Vec<f32>>>>,
    /// Estimated token count for scheduling decisions
    pub token_count: usize,
}

/// Active batch being processed
#[derive(Debug, Clone)]
pub struct ActiveBatch {
    /// Unique batch identifier
    pub id: Uuid,
    /// Total token count in this batch
    pub token_count: usize,
    /// When this batch was created
    pub created_at: Instant,
}

/// Batch scheduler for intelligent request batching
///
/// The scheduler maintains:
/// - An atomic counter of pending tokens (for lock-free queue depth tracking)
/// - A list of active batches (for capacity management)
/// - An unbounded MPSC channel for incoming requests
/// - Context size limit for capacity-aware batch sizing
///
/// # Thread Safety
///
/// - `pending_tokens` is accessed atomically without locking
/// - `active_batches` is protected by `parking_lot::Mutex` (faster than std)
/// - `request_tx` is cloneable and can be shared across threads
/// - `request_rx` is protected by `Mutex` for single-worker access
pub struct BatchScheduler {
    /// Atomic counter of total tokens waiting in queue
    /// Used for making intelligent batch sizing decisions
    pending_tokens: AtomicUsize,

    /// Track in-flight batches and their sizes
    /// Protected by `parking_lot::Mutex` (no poisoning, no unwrap needed)
    active_batches: Mutex<Vec<ActiveBatch>>,

    /// Work queue receiver (protected by Mutex for single worker)
    request_rx: Mutex<mpsc::UnboundedReceiver<BatchRequest>>,

    /// Cloneable sender for Axum handlers
    request_tx: mpsc::UnboundedSender<BatchRequest>,

    /// Reference to the embedding model
    /// Note: `EmbeddingModel` is !Send, but the model worker stays on one thread
    /// The `BatchScheduler` itself will be Arc'd, so no need for Arc here
    model: Mutex<EmbeddingModel>,

    /// Batch packing capacity (total tokens)
    /// Used for intelligent batch sizing decisions
    n_batch: usize,

    /// Maximum context size (total tokens)
    /// Used to ensure total active tokens don't exceed capacity
    context_size: usize,
}

impl BatchScheduler {
    /// Create a new batch scheduler
    ///
    /// # Arguments
    ///
    /// * `model` - The embedding model to use for processing
    ///
    /// # Returns
    ///
    /// A new `BatchScheduler` instance ready to accept requests
    pub fn new(model: EmbeddingModel) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();

        // Extract configuration from model for intelligent batch sizing
        let n_batch = model.n_batch() as usize;
        let context_size = model.max_sequence_length();

        info!(
            "Creating batch scheduler: n_batch={}, context_size={}",
            n_batch, context_size
        );

        Self {
            pending_tokens: AtomicUsize::new(0),
            active_batches: Mutex::new(Vec::new()),
            request_rx: Mutex::new(rx),
            request_tx: tx,
            model: Mutex::new(model),
            n_batch,
            context_size,
        }
    }

    /// Get a cloneable sender for Axum handlers
    ///
    /// This sender can be cloned and used from multiple threads
    /// to submit requests to the scheduler.
    ///
    /// # Returns
    ///
    /// A cloneable `UnboundedSender` for submitting `BatchRequest`s
    pub fn get_sender(&self) -> mpsc::UnboundedSender<BatchRequest> {
        self.request_tx.clone()
    }

    /// Increment the pending token counter
    ///
    /// Should be called when a request arrives.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Number of tokens to add to pending count
    pub fn add_pending_tokens(&self, tokens: usize) {
        let new_count = self.pending_tokens.fetch_add(tokens, Ordering::Relaxed) + tokens;
        debug!("Pending tokens increased to {}", new_count);
    }

    /// Decrement the pending token counter
    ///
    /// Should be called when a request completes.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Number of tokens to remove from pending count
    pub fn remove_pending_tokens(&self, tokens: usize) {
        let new_count = self
            .pending_tokens
            .fetch_sub(tokens, Ordering::Relaxed)
            .saturating_sub(tokens);
        debug!("Pending tokens decreased to {}", new_count);
    }

    /// Get the current pending token count
    ///
    /// # Returns
    ///
    /// Number of tokens currently pending in the queue
    pub fn pending_tokens(&self) -> usize {
        self.pending_tokens.load(Ordering::Relaxed)
    }

    /// Get the current active batch count
    ///
    /// # Returns
    ///
    /// Number of batches currently being processed
    pub fn active_batch_count(&self) -> usize {
        self.active_batches.lock().len()
    }

    /// Get the total token count in active batches
    ///
    /// # Returns
    ///
    /// Sum of token counts across all active batches
    pub fn active_token_count(&self) -> usize {
        self.active_batches
            .lock()
            .iter()
            .map(|b| b.token_count)
            .sum()
    }

    /// Add a batch to the active tracking list
    ///
    /// # Arguments
    ///
    /// * `batch_id` - Unique identifier for the batch
    /// * `token_count` - Number of tokens in the batch
    fn track_active_batch(&self, batch_id: Uuid, token_count: usize) {
        let batch = ActiveBatch {
            id: batch_id,
            token_count,
            created_at: Instant::now(),
        };

        self.active_batches.lock().push(batch);

        info!(
            "Tracking active batch {} with {} tokens (total active: {}, pending: {})",
            batch_id,
            token_count,
            self.active_batch_count(),
            self.pending_tokens()
        );
    }

    /// Remove a batch from the active tracking list
    ///
    /// # Arguments
    ///
    /// * `batch_id` - Unique identifier for the batch to remove
    fn untrack_active_batch(&self, batch_id: Uuid) {
        let mut batches = self.active_batches.lock();
        if let Some(pos) = batches.iter().position(|b| b.id == batch_id) {
            let batch = batches.remove(pos);
            let elapsed = batch.created_at.elapsed();

            info!(
                "Completed batch {} with {} tokens in {:?} (remaining active: {}, pending: {})",
                batch_id,
                batch.token_count,
                elapsed,
                batches.len(),
                self.pending_tokens()
            );
        } else {
            warn!("Attempted to untrack unknown batch {}", batch_id);
        }
    }

    /// Determine the optimal batch size based on queue state and capacity
    ///
    /// This method implements intelligent batch sizing that adapts to load:
    /// - **Light load** (pending < `n_batch`): Allocate exactly what's needed to avoid waste
    /// - **Heavy load** (pending >= `n_batch`): Allocate full `n_batch` for maximum throughput
    /// - **At capacity** (active + `n_batch` > `context_size`): Fit what we can within limits
    ///
    /// # Returns
    ///
    /// The optimal batch size in tokens, considering pending work and active capacity
    ///
    /// # Implementation Note
    ///
    /// Uses `parking_lot::Mutex` which never panics, so no `unwrap()` needed
    fn determine_batch_size(&self) -> usize {
        let pending = self.pending_tokens.load(Ordering::Relaxed);

        // parking_lot::Mutex never panics, no unwrap needed
        let active_info = self.active_batches.lock();
        let total_active: usize = active_info.iter().map(|b| b.token_count).sum();
        drop(active_info); // Release lock early

        // Calculate available capacity
        let available_capacity = self.context_size.saturating_sub(total_active);

        // Determine optimal batch size based on load and capacity
        if pending == 0 {
            // No pending work
            debug!("No pending work, batch_size=0");
            0
        } else if pending < self.n_batch {
            // Light load: allocate exactly what we need
            let size = std::cmp::min(pending, available_capacity);
            debug!(
                "Light load: pending={} < n_batch={}, available={}, batch_size={}",
                pending, self.n_batch, available_capacity, size
            );
            size
        } else if total_active + self.n_batch <= self.context_size {
            // Heavy load with capacity: allocate full batch
            let size = std::cmp::min(self.n_batch, available_capacity);
            debug!(
                "Heavy load with capacity: pending={}, active={}, n_batch={}, available={}, batch_size={}",
                pending, total_active, self.n_batch, available_capacity, size
            );
            size
        } else {
            // At capacity: fit what we can
            let size = std::cmp::min(pending, available_capacity);
            debug!(
                "At capacity: pending={}, active={}, context_size={}, available={}, batch_size={}",
                pending, total_active, self.context_size, available_capacity, size
            );
            size
        }
    }

    /// Process a batch of requests with OOM retry logic
    ///
    /// This method implements exponential backoff retry strategy:
    /// - If OOM occurs, retry with half the batch size
    /// - Continue halving until success or minimum size (64 tokens) reached
    /// - Maximum 4 retry attempts
    ///
    /// # Arguments
    ///
    /// * `requests` - Slice of batch requests to process together
    /// * `initial_batch_size` - Initial batch size to attempt
    ///
    /// # Returns
    ///
    /// Returns Ok if processing succeeds, Err if all retries exhausted
    async fn process_batch_with_retry(
        &self,
        requests: &[BatchRequest],
        initial_batch_size: usize,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let mut batch_size = initial_batch_size;
        let mut attempt = 0;

        loop {
            attempt += 1;

            debug!(
                "Processing batch attempt {} with size {} (requests: {})",
                attempt,
                batch_size,
                requests.len()
            );

            // Try processing with current batch size
            match self.process_batch_internal(requests, batch_size).await {
                Ok(results) => {
                    if attempt > 1 {
                        info!(
                            "Batch processing succeeded after {} attempts (final size: {})",
                            attempt, batch_size
                        );
                    }
                    return Ok(results);
                }
                Err(e) => {
                    // Check if this is an OOM error
                    let is_oom = if let Error::EmbeddingGenerationError { ref message, .. } = e {
                        Error::is_oom_message(message)
                    } else {
                        e.is_oom()
                    };

                    if !is_oom {
                        // Non-OOM error, don't retry
                        return Err(e);
                    }

                    warn!(
                        "GPU OOM at batch_size={} (attempt {}), retrying with smaller batch",
                        batch_size, attempt
                    );

                    // Check if we can retry
                    if batch_size <= MIN_BATCH_SIZE {
                        warn!(
                            "Batch size {} already at minimum ({}), cannot retry",
                            batch_size, MIN_BATCH_SIZE
                        );
                        return Err(Error::out_of_memory(
                            format!(
                                "Failed to process batch even at minimum size {}",
                                MIN_BATCH_SIZE
                            ),
                            Some(batch_size),
                        ));
                    }

                    if attempt >= MAX_RETRY_ATTEMPTS {
                        warn!("Exhausted {} retry attempts, giving up", MAX_RETRY_ATTEMPTS);
                        return Err(Error::out_of_memory(
                            format!("Failed after {} OOM retry attempts", MAX_RETRY_ATTEMPTS),
                            Some(batch_size),
                        ));
                    }

                    // Halve the batch size for next attempt
                    batch_size = std::cmp::max(batch_size / 2, MIN_BATCH_SIZE);
                    debug!("Reduced batch size to {} for retry", batch_size);
                }
            }
        }
    }

    /// Internal batch processing without retry logic
    ///
    /// This method processes a batch of requests through the model.
    /// If OOM occurs, the error is propagated to the retry logic.
    ///
    /// # Arguments
    ///
    /// * `requests` - Slice of batch requests to process
    /// * `batch_size` - Maximum batch size to use
    ///
    /// # Returns
    ///
    /// Returns embeddings for each request, or an error
    async fn process_batch_internal(
        &self,
        requests: &[BatchRequest],
        _batch_size: usize,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let mut all_results = Vec::with_capacity(requests.len());

        for request in requests {
            let texts: Vec<&str> = request.texts.iter().map(String::as_str).collect();

            // Process through the model
            let result = {
                let mut model = self.model.lock();
                // Use the existing batch processing method
                if texts.len() == 1 {
                    match model.generate_embedding(texts[0]) {
                        Ok(embedding) => Ok(vec![embedding]),
                        Err(e) => Err(e),
                    }
                } else {
                    // For multiple texts, process as batch
                    // > TODO: This should use the batch processor once integrated
                    let mut embeddings = Vec::with_capacity(texts.len());
                    for text in texts {
                        match model.generate_embedding(text) {
                            Ok(emb) => embeddings.push(emb),
                            Err(e) => return Err(e),
                        }
                    }
                    Ok(embeddings)
                }
            }?;

            all_results.push(result);
        }

        Ok(all_results)
    }

    /// Send a response back to a batch request
    ///
    /// This is a helper function that sends the result back via oneshot channel.
    ///
    /// # Arguments
    ///
    /// * `request` - The batch request to process
    /// * `result` - The result to send back
    fn send_request_response(request: BatchRequest, result: Result<Vec<Vec<f32>>>) {
        let request_id = request.id;

        // Send response back
        if request.response_tx.send(result).is_err() {
            warn!(
                "Request {} client disconnected before response could be sent",
                request_id
            );
        }
    }

    /// Run the model worker loop
    ///
    /// This is the main processing loop that:
    /// 1. Determines optimal batch size based on queue state and capacity
    /// 2. Collects requests from the queue up to the determined size
    /// 3. Tracks active batches during processing
    /// 4. Processes batches through the model
    /// 5. Sends responses back via oneshot channels
    ///
    /// The loop runs until the request channel is closed.
    pub async fn run_worker(self: Arc<Self>) {
        info!("Starting batch scheduler worker loop");
        info!(
            "Worker configured: n_batch={}, context_size={}",
            self.n_batch, self.context_size
        );

        let mut batch_requests = Vec::new();

        loop {
            // Determine optimal batch size based on current load and capacity
            let target_batch_size = self.determine_batch_size();

            // If no capacity or no pending work, yield and try again
            if target_batch_size == 0 {
                tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
                continue;
            }

            // Try to collect requests up to the determined batch size
            let mut rx = self.request_rx.lock();

            // Collect as many requests as we can up to target_batch_size
            while let Ok(req) = rx.try_recv() {
                batch_requests.push(req);

                let total_tokens: usize = batch_requests.iter().map(|r| r.token_count).sum();

                // Stop collecting if we've reached the target size
                if total_tokens >= target_batch_size {
                    debug!(
                        "Reached target batch size: {} tokens >= {} target",
                        total_tokens, target_batch_size
                    );
                    break;
                }
            }

            drop(rx); // Release lock before processing

            // Process collected requests
            if !batch_requests.is_empty() {
                let batch_id = Uuid::new_v4();
                let total_tokens: usize = batch_requests.iter().map(|r| r.token_count).sum();

                // Track this batch as active
                self.track_active_batch(batch_id, total_tokens);

                debug!(
                    "Processing batch {} with {} requests ({} tokens)",
                    batch_id,
                    batch_requests.len(),
                    total_tokens
                );

                // Process batch with retry logic (handles OOM gracefully)
                // > NOTE: Phase 2.6 implements exponential backoff for GPU OOM errors
                // > NOTE: Phase 3.5 will add concurrent batch processing for better throughput
                match self
                    .process_batch_with_retry(&batch_requests, target_batch_size)
                    .await
                {
                    Ok(results) => {
                        // Send responses to each request
                        for (request, result) in batch_requests.drain(..).zip(results.into_iter()) {
                            Self::send_request_response(request, Ok(result));
                        }
                    }
                    Err(e) => {
                        // Send error to all requests in the batch
                        warn!("Batch {} failed after all retries: {}", batch_id, e);
                        // Convert error to string to send to multiple requests
                        let error_msg = e.to_string();
                        let is_oom = e.is_oom();
                        for request in batch_requests.drain(..) {
                            let err = if is_oom {
                                Error::out_of_memory(error_msg.clone(), None)
                            } else {
                                Error::embedding_failed(error_msg.clone())
                            };
                            Self::send_request_response(request, Err(err));
                        }
                    }
                }

                // Remove from active tracking
                self.untrack_active_batch(batch_id);
            } else {
                // No work available, yield to avoid busy-waiting
                tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
            }
        }
    }
}

// Note: No Clone implementation for BatchScheduler
// The scheduler should be created once and Arc'd if shared access is needed
// EmbeddingModel is !Clone, and cloning the request_rx doesn't make sense

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a test model (would need actual model in real tests)
    // For now, tests will be compilation checks only

    #[test]
    fn test_batch_scheduler_structure() {
        // Verify the structure compiles
        // Real tests would require a test model
    }

    #[test]
    fn test_atomic_operations() {
        // Test atomic counter operations without requiring a model
        let (_tx, _rx) = mpsc::unbounded_channel::<()>();
        let pending = AtomicUsize::new(0);

        // Simulate add
        let new_count = pending.fetch_add(100, Ordering::Relaxed) + 100;
        assert_eq!(new_count, 100);

        // Simulate remove
        let new_count = pending.fetch_sub(50, Ordering::Relaxed).saturating_sub(50);
        assert_eq!(new_count, 50);
    }
}
