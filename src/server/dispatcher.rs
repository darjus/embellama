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

//! Request dispatcher for routing to worker threads
//!
//! This module handles the distribution of embedding requests to available
//! worker threads using a round-robin routing strategy.

use crate::EmbeddingEngine;
use crate::server::channel::WorkerRequest;
use crate::server::worker::Worker;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// Dispatcher for routing requests to workers
pub struct Dispatcher {
    /// Worker sender channels for round-robin distribution
    workers: Arc<Vec<mpsc::Sender<WorkerRequest>>>,
    /// Current worker index for round-robin
    current_worker: Arc<AtomicUsize>,
    /// Worker thread handles (not cloneable, so wrapped in Option)
    #[allow(dead_code)]
    handles: Option<Vec<thread::JoinHandle<()>>>,
}

impl Clone for Dispatcher {
    fn clone(&self) -> Self {
        Self {
            workers: Arc::clone(&self.workers),
            current_worker: Arc::clone(&self.current_worker),
            handles: None, // Don't clone thread handles
        }
    }
}

impl Dispatcher {
    /// Create a new dispatcher
    ///
    /// # Arguments
    /// * `worker_count` - Number of worker threads to spawn
    /// * `queue_size` - Maximum pending requests per worker
    ///
    /// # Returns
    /// A new `Dispatcher` instance
    pub fn new(worker_count: usize, queue_size: usize) -> Self {
        info!("Creating dispatcher with {} workers", worker_count);

        // Get the engine instance (should already be initialized)
        let engine = EmbeddingEngine::instance()
            .expect("EmbeddingEngine should be initialized before creating Dispatcher");

        let mut workers = Vec::with_capacity(worker_count);
        let mut handles = Vec::with_capacity(worker_count);

        // Spawn worker threads
        for id in 0..worker_count {
            // Create channel for this worker
            let (tx, rx) = mpsc::channel::<WorkerRequest>(queue_size);

            // Spawn the worker thread
            let handle = Worker::spawn(id, Arc::clone(&engine), rx);

            workers.push(tx);
            handles.push(handle);

            debug!("Spawned worker {}", id);
        }

        info!("Dispatcher created with {} workers", worker_count);

        Self {
            workers: Arc::new(workers),
            current_worker: Arc::new(AtomicUsize::new(0)),
            handles: Some(handles),
        }
    }

    /// Send a request to the dispatcher
    ///
    /// Routes the request to the next worker using round-robin distribution.
    ///
    /// # Arguments
    /// * `request` - The worker request to process
    ///
    /// # Returns
    /// Result indicating success or failure
    pub async fn send(&self, request: WorkerRequest) -> Result<(), String> {
        // Select next worker using round-robin
        let worker_count = self.workers.len();
        let worker_index = self.current_worker.fetch_add(1, Ordering::Relaxed) % worker_count;

        debug!(
            "Routing request {:?} to worker {}",
            request.id, worker_index
        );

        // Send to selected worker
        self.workers[worker_index].send(request).await.map_err(|e| {
            warn!("Worker {} channel full or closed: {}", worker_index, e);
            format!("Failed to send request to worker {}: {}", worker_index, e)
        })
    }

    /// Check if the dispatcher is ready to accept requests
    pub fn is_ready(&self) -> bool {
        // Check if at least one worker is available
        self.workers.iter().any(|tx| !tx.is_closed())
    }

    /// Get the number of active workers
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Shutdown all workers gracefully
    ///
    /// This drops all sender channels, causing workers to exit their loops
    pub async fn shutdown(self) {
        info!("Shutting down dispatcher and workers");

        // Note: Dropping the Arc will signal workers to stop when last reference is dropped
        // We can't directly manipulate the Arc<Vec>, but dropping self will work
        drop(self);

        info!("Workers signaled to shutdown");
    }
}
