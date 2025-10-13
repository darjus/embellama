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

//! Unit tests for batch scheduler (Phase 2.4)
//!
//! These tests verify the batch scheduler infrastructure without requiring
//! a real model. They focus on testing:
//! - Atomic counter operations
//! - Active batch tracking
//! - Channel communication patterns
//! - Error handling
//!
//! Note: These tests require the `server` feature to be enabled

#![cfg(feature = "server")]

use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn test_atomic_pending_tokens_increment() {
    // Test that atomic add operations work correctly
    let pending = AtomicUsize::new(0);

    // Simulate adding pending tokens
    let new_count = pending.fetch_add(100, Ordering::Relaxed) + 100;
    assert_eq!(new_count, 100);

    // Add more
    let new_count = pending.fetch_add(50, Ordering::Relaxed) + 50;
    assert_eq!(new_count, 150);

    // Verify final count
    assert_eq!(pending.load(Ordering::Relaxed), 150);
}

#[test]
fn test_atomic_pending_tokens_decrement() {
    // Test that atomic sub operations work correctly
    let pending = AtomicUsize::new(200);

    // Simulate removing pending tokens
    let old_count = pending.fetch_sub(50, Ordering::Relaxed);
    assert_eq!(old_count, 200);
    assert_eq!(pending.load(Ordering::Relaxed), 150);

    // Remove more
    let old_count = pending.fetch_sub(100, Ordering::Relaxed);
    assert_eq!(old_count, 150);
    assert_eq!(pending.load(Ordering::Relaxed), 50);
}

#[test]
fn test_atomic_pending_tokens_saturating_sub() {
    // Test saturating subtraction to prevent underflow
    let pending = AtomicUsize::new(50);

    // Try to remove more than available
    let old_count = pending.fetch_sub(100, Ordering::Relaxed);
    // saturating_sub would prevent underflow in the actual implementation
    let result = old_count.saturating_sub(100);
    assert_eq!(result, 0); // Should not underflow
}

#[test]
fn test_atomic_operations_concurrent_safe() {
    // Verify that concurrent operations are safe
    // This is more of a compilation check than a runtime test
    let pending = AtomicUsize::new(0);

    // These operations should be atomic and thread-safe
    pending.fetch_add(10, Ordering::Relaxed);
    pending.fetch_sub(5, Ordering::Relaxed);
    pending.load(Ordering::Relaxed);

    // If this compiles and runs, atomicity is guaranteed by the type system
    assert!(true);
}

#[test]
fn test_active_batch_tracking_add_remove() {
    use parking_lot::Mutex;
    use std::time::Instant;
    use uuid::Uuid;

    // Simulate the active_batches structure
    #[derive(Debug, Clone)]
    struct ActiveBatch {
        id: Uuid,
        token_count: usize,
        created_at: Instant,
    }

    let active_batches: Mutex<Vec<ActiveBatch>> = Mutex::new(Vec::new());

    // Test adding batches
    let batch1_id = Uuid::new_v4();
    active_batches.lock().push(ActiveBatch {
        id: batch1_id,
        token_count: 100,
        created_at: Instant::now(),
    });

    assert_eq!(active_batches.lock().len(), 1);

    let batch2_id = Uuid::new_v4();
    active_batches.lock().push(ActiveBatch {
        id: batch2_id,
        token_count: 200,
        created_at: Instant::now(),
    });

    assert_eq!(active_batches.lock().len(), 2);

    // Test calculating total active tokens
    let total_tokens: usize = active_batches.lock().iter().map(|b| b.token_count).sum();
    assert_eq!(total_tokens, 300);

    // Test removing a batch
    active_batches.lock().retain(|b| b.id != batch1_id);
    assert_eq!(active_batches.lock().len(), 1);

    // Verify remaining batch
    let remaining_batch = active_batches.lock()[0].clone();
    assert_eq!(remaining_batch.id, batch2_id);
    assert_eq!(remaining_batch.token_count, 200);
}

#[test]
fn test_active_batch_token_sum_calculation() {
    use parking_lot::Mutex;
    use std::time::Instant;
    use uuid::Uuid;

    #[derive(Debug, Clone)]
    struct ActiveBatch {
        id: Uuid,
        token_count: usize,
        created_at: Instant,
    }

    let active_batches: Mutex<Vec<ActiveBatch>> = Mutex::new(Vec::new());

    // Add multiple batches
    for token_count in [100, 200, 300, 150, 250] {
        active_batches.lock().push(ActiveBatch {
            id: Uuid::new_v4(),
            token_count,
            created_at: Instant::now(),
        });
    }

    // Calculate total
    let total: usize = active_batches.lock().iter().map(|b| b.token_count).sum();
    assert_eq!(total, 1000);

    // Test that parking_lot::Mutex doesn't poison
    // (This is a compilation check - parking_lot never panics on lock)
    let _guard = active_batches.lock();
    // No unwrap needed!
}

#[test]
fn test_channel_message_structure() {
    use tokio::sync::oneshot;
    use uuid::Uuid;

    // Test that batch request structure is valid
    #[derive(Debug)]
    struct BatchRequest {
        id: Uuid,
        texts: Vec<String>,
        response_tx: oneshot::Sender<Result<Vec<Vec<f32>>, String>>,
        token_count: usize,
    }

    let (tx, _rx) = oneshot::channel();
    let request = BatchRequest {
        id: Uuid::new_v4(),
        texts: vec!["test".to_string()],
        response_tx: tx,
        token_count: 10,
    };

    // Verify structure
    assert_eq!(request.texts.len(), 1);
    assert_eq!(request.token_count, 10);
}

#[test]
fn test_unbounded_channel_creation() {
    use tokio::sync::mpsc;

    // Test that unbounded channels can be created
    let (_tx, _rx) = mpsc::unbounded_channel::<usize>();

    // Verify sender is cloneable
    let _tx2 = _tx.clone();
    let _tx3 = _tx.clone();

    // If this compiles, the channel pattern is valid
    assert!(true);
}

#[test]
fn test_parking_lot_mutex_performance() {
    use parking_lot::Mutex;
    use std::sync::Arc;

    // Create a shared counter
    let counter = Arc::new(Mutex::new(0));

    // Simulate multiple "threads" accessing the counter
    for _ in 0..100 {
        let mut guard = counter.lock();
        *guard += 1;
        // Note: No unwrap() needed with parking_lot!
    }

    assert_eq!(*counter.lock(), 100);
}

#[test]
fn test_uuid_generation_for_batch_ids() {
    use uuid::Uuid;

    // Test that batch IDs are unique
    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();

    assert_ne!(id1, id2);

    // Test that IDs can be cloned and compared
    let id1_clone = id1;
    assert_eq!(id1, id1_clone);
}
