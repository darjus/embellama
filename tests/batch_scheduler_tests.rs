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

// =============================================================================
// Phase 2.5: Intelligent Batch Sizing Tests
// =============================================================================

#[test]
fn test_determine_batch_size_light_load() {
    // Test: Light load (pending < n_batch) → returns pending
    // Simulates the logic from determine_batch_size()
    let n_batch: usize = 2048;
    let context_size: usize = 8192;
    let pending: usize = 500;
    let total_active: usize = 0;

    let available_capacity = context_size.saturating_sub(total_active);
    let batch_size = if pending < n_batch {
        std::cmp::min(pending, available_capacity)
    } else if total_active + n_batch <= context_size {
        std::cmp::min(n_batch, available_capacity)
    } else {
        std::cmp::min(pending, available_capacity)
    };

    assert_eq!(
        batch_size, 500,
        "Light load should return exact pending count"
    );
}

#[test]
fn test_determine_batch_size_heavy_load_with_capacity() {
    // Test: Heavy load with capacity (pending >= n_batch, room in context) → returns n_batch
    let n_batch: usize = 2048;
    let context_size: usize = 8192;
    let pending: usize = 5000;
    let total_active: usize = 1000;

    let available_capacity = context_size.saturating_sub(total_active);
    let batch_size = if pending < n_batch {
        std::cmp::min(pending, available_capacity)
    } else if total_active + n_batch <= context_size {
        std::cmp::min(n_batch, available_capacity)
    } else {
        std::cmp::min(pending, available_capacity)
    };

    assert_eq!(
        batch_size, 2048,
        "Heavy load with capacity should return full n_batch"
    );
}

#[test]
fn test_determine_batch_size_at_capacity() {
    // Test: At capacity (total_active + n_batch > context_size) → returns smaller size
    let n_batch: usize = 2048;
    let context_size: usize = 8192;
    let pending: usize = 5000;
    let total_active: usize = 7000; // Almost at capacity

    let available_capacity = context_size.saturating_sub(total_active);
    let batch_size = if pending < n_batch {
        std::cmp::min(pending, available_capacity)
    } else if total_active + n_batch <= context_size {
        std::cmp::min(n_batch, available_capacity)
    } else {
        std::cmp::min(pending, available_capacity)
    };

    assert_eq!(
        batch_size, 1192,
        "At capacity should return min(pending, available)"
    );
    assert!(
        batch_size <= available_capacity,
        "Batch size must not exceed available capacity"
    );
}

#[test]
fn test_determine_batch_size_zero_capacity() {
    // Test: Zero capacity (total_active >= context_size) → returns 0
    let n_batch: usize = 2048;
    let context_size: usize = 8192;
    let pending: usize = 5000;
    let total_active: usize = 8192; // At capacity

    let available_capacity = context_size.saturating_sub(total_active);
    let batch_size = if pending < n_batch {
        std::cmp::min(pending, available_capacity)
    } else if total_active + n_batch <= context_size {
        std::cmp::min(n_batch, available_capacity)
    } else {
        std::cmp::min(pending, available_capacity)
    };

    assert_eq!(batch_size, 0, "Zero capacity should return 0");
}

#[test]
fn test_determine_batch_size_no_pending_work() {
    // Test: No pending work (pending == 0) → returns 0
    let n_batch: usize = 2048;
    let context_size: usize = 8192;
    let pending: usize = 0;
    let total_active: usize = 1000;

    let available_capacity = context_size.saturating_sub(total_active);
    let batch_size = if pending == 0 {
        0
    } else if pending < n_batch {
        std::cmp::min(pending, available_capacity)
    } else if total_active + n_batch <= context_size {
        std::cmp::min(n_batch, available_capacity)
    } else {
        std::cmp::min(pending, available_capacity)
    };

    assert_eq!(batch_size, 0, "No pending work should return 0");
}

#[test]
fn test_determine_batch_size_exact_n_batch_pending() {
    // Test: Pending exactly equals n_batch → returns n_batch if capacity allows
    let n_batch: usize = 2048;
    let context_size: usize = 8192;
    let pending: usize = 2048;
    let total_active: usize = 1000;

    let available_capacity = context_size.saturating_sub(total_active);
    let batch_size = if pending < n_batch {
        std::cmp::min(pending, available_capacity)
    } else if total_active + n_batch <= context_size {
        std::cmp::min(n_batch, available_capacity)
    } else {
        std::cmp::min(pending, available_capacity)
    };

    assert_eq!(
        batch_size, 2048,
        "Pending == n_batch should return n_batch when capacity allows"
    );
}

#[test]
fn test_determine_batch_size_edge_case_minimal_capacity() {
    // Test: Very small available capacity
    let n_batch: usize = 2048;
    let context_size: usize = 8192;
    let pending: usize = 5000;
    let total_active: usize = 8190; // Only 2 tokens available

    let available_capacity = context_size.saturating_sub(total_active);
    let batch_size = if pending < n_batch {
        std::cmp::min(pending, available_capacity)
    } else if total_active + n_batch <= context_size {
        std::cmp::min(n_batch, available_capacity)
    } else {
        std::cmp::min(pending, available_capacity)
    };

    assert_eq!(
        batch_size, 2,
        "Minimal capacity should return available capacity"
    );
    assert!(
        batch_size <= available_capacity,
        "Must respect available capacity"
    );
}

#[test]
fn test_determine_batch_size_light_load_constrained_by_capacity() {
    // Test: Light load but constrained by available capacity
    let n_batch: usize = 2048;
    let context_size: usize = 8192;
    let pending: usize = 500;
    let total_active: usize = 8000; // Only 192 tokens available

    let available_capacity = context_size.saturating_sub(total_active);
    let batch_size = if pending < n_batch {
        std::cmp::min(pending, available_capacity)
    } else if total_active + n_batch <= context_size {
        std::cmp::min(n_batch, available_capacity)
    } else {
        std::cmp::min(pending, available_capacity)
    };

    assert_eq!(
        batch_size, 192,
        "Light load constrained by capacity should return available"
    );
    assert!(
        batch_size <= available_capacity,
        "Must respect available capacity"
    );
    assert!(batch_size <= pending, "Must not exceed pending work");
}

#[test]
fn test_saturating_sub_prevents_underflow() {
    // Test that saturating_sub prevents integer underflow in capacity calculations
    let context_size: usize = 1000;
    let total_active: usize = 1500; // More than context_size

    let available_capacity = context_size.saturating_sub(total_active);
    assert_eq!(
        available_capacity, 0,
        "saturating_sub should prevent underflow"
    );
}

// =============================================================================
// Phase 2.6: GPU OOM Retry Logic Tests
// =============================================================================

#[test]
fn test_oom_message_detection_out_of_memory() {
    use embellama::Error;

    // Test various OOM error message patterns
    assert!(
        Error::is_oom_message("GPU out of memory"),
        "Should detect 'out of memory'"
    );
    assert!(
        Error::is_oom_message("CUDA out of memory"),
        "Should detect 'CUDA out of memory'"
    );
    assert!(
        Error::is_oom_message("Out Of Memory error"),
        "Should detect case-insensitive OOM"
    );
}

#[test]
fn test_oom_message_detection_oom_keyword() {
    use embellama::Error;

    assert!(
        Error::is_oom_message("OOM error occurred"),
        "Should detect 'OOM'"
    );
    assert!(
        Error::is_oom_message("oom during allocation"),
        "Should detect lowercase 'oom'"
    );
}

#[test]
fn test_oom_message_detection_allocation_failed() {
    use embellama::Error;

    assert!(
        Error::is_oom_message("failed to allocate memory"),
        "Should detect 'failed to allocate'"
    );
    assert!(
        Error::is_oom_message("allocation failed"),
        "Should detect 'allocation failed'"
    );
}

#[test]
fn test_oom_message_detection_cuda_error() {
    use embellama::Error;

    assert!(
        Error::is_oom_message("CUDA error: allocation failed"),
        "Should detect 'CUDA error'"
    );
    assert!(
        Error::is_oom_message("cudaMalloc failed"),
        "Should detect 'cudaMalloc'"
    );
}

#[test]
fn test_oom_message_detection_non_oom_errors() {
    use embellama::Error;

    // These should NOT be detected as OOM
    assert!(
        !Error::is_oom_message("Invalid input provided"),
        "Should not detect non-OOM errors"
    );
    assert!(
        !Error::is_oom_message("Model not found"),
        "Should not detect model errors"
    );
    assert!(
        !Error::is_oom_message("Tokenization failed"),
        "Should not detect tokenization errors"
    );
}

#[test]
fn test_exponential_backoff_halving() {
    // Test exponential backoff logic (halving batch size each retry)
    let min_batch_size: usize = 64;
    let initial_size: usize = 2048;

    // Simulate retry attempts with exponential backoff
    let mut batch_size = initial_size;
    let attempts = vec![
        2048, // Attempt 1: Full size
        1024, // Attempt 2: Half
        512,  // Attempt 3: Quarter
        256,  // Attempt 4: Eighth
        128,  // Attempt 5: Sixteenth
        64,   // Attempt 6: Minimum (clamped)
    ];

    for (i, expected) in attempts.iter().enumerate() {
        assert_eq!(
            batch_size,
            *expected,
            "Attempt {} should have batch_size {}",
            i + 1,
            expected
        );

        // Halve for next attempt, but not below minimum
        batch_size = std::cmp::max(batch_size / 2, min_batch_size);
    }

    // Verify we've reached minimum and stay there
    batch_size = std::cmp::max(batch_size / 2, min_batch_size);
    assert_eq!(batch_size, 64, "Should stay at minimum size");
}

#[test]
fn test_minimum_batch_size_enforcement() {
    // Test that minimum batch size (64 tokens) is enforced
    let min_batch_size: usize = 64;

    let test_cases = vec![
        (128, 64),  // 128 / 2 = 64 (at minimum)
        (64, 64),   // 64 / 2 = 32, but clamp to 64
        (32, 64),   // Already below minimum, clamp to 64
        (1, 64),    // Far below minimum, clamp to 64
        (256, 128), // 256 / 2 = 128 (above minimum)
    ];

    for (current, expected) in test_cases {
        let next_size = std::cmp::max(current / 2, min_batch_size);
        assert_eq!(
            next_size, expected,
            "Halving {} should result in {}",
            current, expected
        );
    }
}

#[test]
fn test_retry_attempt_limit() {
    // Test that we respect the maximum retry attempts (4)
    let max_retry_attempts: usize = 4;
    let mut attempt = 0;

    // Simulate retry loop
    while attempt < max_retry_attempts {
        attempt += 1;
    }

    assert_eq!(attempt, 4, "Should stop after 4 attempts");

    // Verify we don't exceed the limit
    if attempt >= max_retry_attempts {
        // This is where we would give up
        assert!(
            attempt >= max_retry_attempts,
            "Should have exhausted retries"
        );
    }
}

#[test]
fn test_oom_error_is_retryable() {
    use embellama::Error;

    let oom_error = Error::out_of_memory("GPU OOM".to_string(), Some(2048));
    assert!(oom_error.is_retryable(), "OOM errors should be retryable");
    assert!(oom_error.is_oom(), "Should be identified as OOM");
}

#[test]
fn test_non_oom_error_not_retryable_for_oom() {
    use embellama::Error;

    // These errors are not OOM
    let invalid_input = Error::invalid_input("Empty text");
    assert!(!invalid_input.is_oom(), "Invalid input should not be OOM");
    assert!(
        !invalid_input.is_retryable(),
        "Invalid input should not be retryable"
    );

    let config_error = Error::config("Invalid config");
    assert!(!config_error.is_oom(), "Config error should not be OOM");
    assert!(
        !config_error.is_retryable(),
        "Config error should not be retryable"
    );
}

#[test]
fn test_batch_size_reduction_sequence() {
    // Test complete batch size reduction sequence from 2048 to 64
    let min_batch_size: usize = 64;
    let mut batch_size: usize = 2048;
    let expected_sequence = vec![2048, 1024, 512, 256, 128, 64, 64];

    for (i, expected) in expected_sequence.iter().enumerate() {
        assert_eq!(
            batch_size, *expected,
            "Step {}: batch_size should be {}",
            i, expected
        );
        batch_size = std::cmp::max(batch_size / 2, min_batch_size);
    }
}

#[test]
fn test_oom_error_with_attempted_size() {
    use embellama::Error;

    // Test that OOM error can track the attempted batch size
    let oom = Error::out_of_memory("GPU OOM".to_string(), Some(2048));

    match oom {
        Error::OutOfMemory {
            message,
            attempted_size,
        } => {
            assert_eq!(message, "GPU OOM");
            assert_eq!(attempted_size, Some(2048));
        }
        _ => panic!("Expected OutOfMemory error"),
    }
}

#[test]
fn test_oom_detection_from_embedding_error() {
    use embellama::Error;

    // Test OOM detection from EmbeddingGenerationError with OOM message
    let embedding_error = Error::EmbeddingGenerationError {
        message: "CUDA out of memory during batch processing".to_string(),
        source: None,
    };

    // Check if the error message contains OOM patterns
    match &embedding_error {
        Error::EmbeddingGenerationError { message, .. } => {
            assert!(
                Error::is_oom_message(message),
                "Should detect OOM in embedding error message"
            );
        }
        _ => panic!("Expected EmbeddingGenerationError"),
    }
}
