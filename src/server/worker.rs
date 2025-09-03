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

//! Worker thread implementation for model inference
//!
//! This module implements worker threads that each own a LlamaContext instance.
//! Workers process embedding requests received via channels.

use crate::server::channel::{WorkerRequest, WorkerResponse};
use std::thread;
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::{debug, error, info};

/// Worker thread that owns a model instance
pub struct Worker {
    /// Worker identifier
    id: usize,
    /// Path to the model file
    model_path: String,
    /// Request receiver channel
    receiver: mpsc::Receiver<WorkerRequest>,
}

impl Worker {
    /// Create a new worker
    ///
    /// # Arguments
    /// * `id` - Worker identifier
    /// * `model_path` - Path to the GGUF model file
    /// * `receiver` - Channel to receive requests
    ///
    /// # Returns
    /// A new `Worker` instance
    pub fn new(id: usize, model_path: String, receiver: mpsc::Receiver<WorkerRequest>) -> Self {
        Self {
            id,
            model_path,
            receiver,
        }
    }
    
    /// Run the worker main loop
    ///
    /// This method runs in a dedicated thread and processes requests
    /// until the channel is closed.
    pub fn run(mut self) {
        info!("Worker {} starting", self.id);
        
        // Phase 2: Load model here
        // For now, just process requests without actual model
        
        // Use blocking recv since we're in a dedicated thread
        while let Some(request) = self.receiver.blocking_recv() {
            debug!("Worker {} processing request {:?}", self.id, request.id);
            let start = Instant::now();
            
            // Phase 2: Process with actual model
            // For now, send dummy response
            let response = WorkerResponse {
                embeddings: vec![vec![0.0; 384]], // Dummy embedding
                token_count: 10,
                processing_time_ms: start.elapsed().as_millis() as u64,
            };
            
            // Send response back
            if let Err(_) = request.response_tx.send(response) {
                error!("Worker {} failed to send response for request {:?}", self.id, request.id);
            }
        }
        
        info!("Worker {} shutting down", self.id);
    }
    
    /// Spawn a worker in a new thread
    ///
    /// # Arguments
    /// * `id` - Worker identifier  
    /// * `model_path` - Path to the GGUF model file
    /// * `receiver` - Channel to receive requests
    ///
    /// # Returns
    /// Handle to the spawned thread
    pub fn spawn(
        id: usize,
        model_path: String,
        receiver: mpsc::Receiver<WorkerRequest>,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let worker = Worker::new(id, model_path, receiver);
            worker.run();
        })
    }
}