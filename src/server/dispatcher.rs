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
//! worker threads using a round-robin or load-based routing strategy.

use crate::server::channel::WorkerRequest;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, info};

/// Dispatcher for routing requests to workers
#[derive(Clone)]
pub struct Dispatcher {
    /// Sender channel to dispatcher thread
    sender: mpsc::Sender<WorkerRequest>,
}

impl Dispatcher {
    /// Create a new dispatcher
    ///
    /// # Arguments
    /// * `worker_count` - Number of worker threads to spawn
    /// * `queue_size` - Maximum pending requests per worker
    /// * `model_path` - Path to the GGUF model file
    ///
    /// # Returns
    /// A new `Dispatcher` instance
    pub fn new(worker_count: usize, queue_size: usize, model_path: String) -> Self {
        info!("Creating dispatcher with {} workers", worker_count);
        
        // Create channel for receiving requests
        let (tx, mut rx) = mpsc::channel::<WorkerRequest>(queue_size * worker_count);
        
        // Spawn dispatcher task (stub for Phase 1)
        tokio::spawn(async move {
            debug!("Dispatcher task started");
            while let Some(request) = rx.recv().await {
                debug!("Received request {:?}, but worker pool not yet implemented", request.id);
                // Phase 2: Route to workers
                // For now, just log and drop
            }
            info!("Dispatcher task shutting down");
        });
        
        Self { sender: tx }
    }
    
    /// Send a request to the dispatcher
    ///
    /// # Arguments
    /// * `request` - The worker request to process
    ///
    /// # Returns
    /// Result indicating success or failure
    pub async fn send(&self, request: WorkerRequest) -> Result<(), String> {
        self.sender
            .send(request)
            .await
            .map_err(|e| format!("Failed to send request: {}", e))
    }
    
    /// Check if the dispatcher is ready to accept requests
    pub fn is_ready(&self) -> bool {
        !self.sender.is_closed()
    }
}