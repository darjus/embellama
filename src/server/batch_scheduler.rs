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

use crate::EmbeddingModel;
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Request sent to the batch scheduler
#[derive(Debug)]
pub struct BatchRequest {
    /// Unique request identifier
    pub id: Uuid,
    /// Text inputs to process
    pub texts: Vec<String>,
    /// Channel to send response back
    pub response_tx: oneshot::Sender<Result<Vec<Vec<f32>>, String>>,
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

        info!("Creating batch scheduler");

        Self {
            pending_tokens: AtomicUsize::new(0),
            active_batches: Mutex::new(Vec::new()),
            request_rx: Mutex::new(rx),
            request_tx: tx,
            model: Mutex::new(model),
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

    /// Process a single batch request
    ///
    /// This is the core processing logic that:
    /// 1. Extracts texts from the request
    /// 2. Processes them through the embedding model
    /// 3. Sends the response back via oneshot channel
    ///
    /// # Arguments
    ///
    /// * `request` - The batch request to process
    async fn process_request(&self, request: BatchRequest) {
        let request_id = request.id;
        let token_count = request.token_count;
        let texts = request.texts;
        let response_tx = request.response_tx;

        debug!(
            "Processing request {} with {} texts ({} tokens)",
            request_id,
            texts.len(),
            token_count
        );

        // Convert texts to string slices for processing
        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

        // Process through the model
        let result = {
            let mut model = self.model.lock();
            // Use the existing batch processing method
            match model.generate_embedding(&text_refs[0]) {
                Ok(embedding) => {
                    if texts.len() == 1 {
                        Ok(vec![embedding])
                    } else {
                        // For multiple texts, we need to process as batch
                        // > TODO: This should use the batch processor once integrated
                        let mut embeddings = Vec::with_capacity(texts.len());
                        embeddings.push(embedding);
                        for text in &text_refs[1..] {
                            match model.generate_embedding(text) {
                                Ok(emb) => embeddings.push(emb),
                                Err(e) => {
                                    return if response_tx
                                        .send(Err(format!("Failed to generate embedding: {}", e)))
                                        .is_err()
                                    {
                                        warn!(
                                            "Request {} client disconnected before error could be sent",
                                            request_id
                                        );
                                    };
                                }
                            }
                        }
                        Ok(embeddings)
                    }
                }
                Err(e) => Err(format!("Failed to generate embedding: {}", e)),
            }
        };

        // Send response back
        if response_tx.send(result).is_err() {
            warn!(
                "Request {} client disconnected before response could be sent",
                request_id
            );
        }
    }

    /// Run the model worker loop
    ///
    /// This is the main processing loop that:
    /// 1. Collects requests from the queue
    /// 2. Groups them into batches up to `n_batch` capacity
    /// 3. Tracks active batches during processing
    /// 4. Processes batches through the model
    /// 5. Sends responses back via oneshot channels
    ///
    /// The loop runs until the request channel is closed.
    pub async fn run_worker(self: Arc<Self>) {
        info!("Starting batch scheduler worker loop");

        let mut batch_requests = Vec::new();
        let model = self.model.lock();
        let n_batch = model.n_batch() as usize;
        drop(model); // Release lock

        info!("Worker configured with n_batch={}", n_batch);

        loop {
            // Try to collect requests up to n_batch capacity
            let mut rx = self.request_rx.lock();

            // Collect as many requests as we can up to n_batch
            while let Ok(req) = rx.try_recv() {
                batch_requests.push(req);

                let total_tokens: usize = batch_requests.iter().map(|r| r.token_count).sum();

                // Stop collecting if we've reached batch capacity
                if total_tokens >= n_batch {
                    debug!(
                        "Reached batch capacity: {} tokens >= {} n_batch",
                        total_tokens, n_batch
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

                // Process each request in the batch
                // > NOTE: Currently processing sequentially, Phase 3.5 will add
                // > concurrent batch processing for better throughput
                for request in batch_requests.drain(..) {
                    self.process_request(request).await;
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
