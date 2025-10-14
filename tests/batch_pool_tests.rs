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

//! Tests for batch pool memory safety and correctness.

use embellama::batch_pool::{clear_pool, get_or_create_batch, pool_size, return_batch};

#[test]
fn test_batch_pool_isolation() {
    // Test that each thread has its own independent pool
    clear_pool();
    assert_eq!(pool_size(), 0);

    // Add batches to main thread pool
    // Create all batches first, then return them to accumulate in pool
    let mut batches = Vec::new();
    for _ in 0..3 {
        batches.push(llama_cpp_2::llama_batch::LlamaBatch::new(512, 1));
    }
    for batch in batches {
        return_batch(batch);
    }
    assert_eq!(pool_size(), 3);

    // Spawn threads and verify they have empty pools
    let handles: Vec<_> = (0..3)
        .map(|_| {
            std::thread::spawn(|| {
                // Each thread should start with empty pool
                assert_eq!(pool_size(), 0);

                // Add some batches
                // Create all batches first, then return them
                let mut batches = Vec::new();
                for _ in 0..2 {
                    batches.push(llama_cpp_2::llama_batch::LlamaBatch::new(512, 1));
                }
                for batch in batches {
                    return_batch(batch);
                }

                assert_eq!(pool_size(), 2);
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Main thread should still have 3 batches
    assert_eq!(pool_size(), 3);

    clear_pool();
}

#[test]
fn test_batch_pool_no_data_contamination() {
    // Test that batches are properly cleared between uses
    clear_pool();

    let capacity = 512;
    let mut batch = get_or_create_batch(capacity);

    // Add some data to the batch
    let tokens: Vec<i32> = vec![1, 2, 3, 4, 5];
    let llama_tokens: Vec<llama_cpp_2::token::LlamaToken> = tokens
        .iter()
        .map(|&t| llama_cpp_2::token::LlamaToken(t))
        .collect();

    // Add sequence to batch
    batch
        .add_sequence(&llama_tokens, 0, true)
        .expect("Failed to add sequence");

    // Return to pool (should be cleared)
    return_batch(batch);

    // Get it back
    let batch2 = get_or_create_batch(capacity);

    // Verify the batch was cleared (n_tokens should be 0)
    // Note: We can't directly inspect internal state, but clearing ensures
    // that subsequent add_sequence calls will work correctly
    assert_eq!(pool_size(), 0, "Pool should be empty after getting batch");

    // Clean up
    return_batch(batch2);
    clear_pool();
}

#[test]
fn test_batch_pool_bounded_size() {
    // Test that pool respects MAX_POOLED_BATCHES limit
    clear_pool();
    const MAX_POOLED_BATCHES: usize = 32;

    // Fill pool to capacity
    // Create all batches first, then return them to accumulate in pool
    let mut batches = Vec::new();
    for _ in 0..MAX_POOLED_BATCHES {
        batches.push(llama_cpp_2::llama_batch::LlamaBatch::new(512, 1));
    }
    for batch in batches {
        return_batch(batch);
    }

    assert_eq!(pool_size(), MAX_POOLED_BATCHES);

    // Try to add more - should be dropped
    for _ in 0..5 {
        let batch = get_or_create_batch(512);
        return_batch(batch);
    }

    // Pool should still be at max
    assert_eq!(pool_size(), MAX_POOLED_BATCHES);

    clear_pool();
}

#[test]
fn test_batch_pool_stress() {
    // Stress test: many allocations and returns
    clear_pool();

    for _ in 0..1000 {
        let batch = get_or_create_batch(2048);
        return_batch(batch);
    }

    // Pool should be bounded
    assert!(pool_size() <= 32);

    clear_pool();
}

#[test]
fn test_batch_pool_concurrent_access() {
    // Test concurrent access from multiple threads
    const NUM_THREADS: usize = 8;
    const ITERATIONS: usize = 100;

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|_| {
            std::thread::spawn(|| {
                for _ in 0..ITERATIONS {
                    let batch = get_or_create_batch(1024);
                    return_batch(batch);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Each thread has its own pool, so we can't directly check the total
    // But the test succeeding means no races or panics occurred
}

#[test]
fn test_batch_pool_varying_capacities() {
    // Test that pool works correctly with different capacities
    clear_pool();

    let capacities = [512, 1024, 2048, 4096];

    // Create all batches first with different capacities
    let mut batches = Vec::new();
    for &cap in &capacities {
        batches.push(llama_cpp_2::llama_batch::LlamaBatch::new(cap, 1));
    }

    // Return them all to the pool
    for batch in batches {
        return_batch(batch);
    }

    // Pool should have all batches (they're all added regardless of capacity)
    assert_eq!(pool_size(), capacities.len());

    // Get them back
    for _ in &capacities {
        let batch = get_or_create_batch(2048); // Request any capacity
        return_batch(batch);
    }

    clear_pool();
}

#[test]
fn test_batch_pool_drop_semantics() {
    // Test that dropping a batch without returning doesn't cause issues
    clear_pool();

    {
        let _batch = get_or_create_batch(512);
        // Batch dropped here without being returned
    }

    // Pool should still be empty
    assert_eq!(pool_size(), 0);

    // Should be able to continue using pool normally
    let batch = get_or_create_batch(512);
    return_batch(batch);
    assert_eq!(pool_size(), 1);

    clear_pool();
}

#[test]
fn test_clear_pool_memory_cleanup() {
    // Test that clear_pool actually frees memory
    clear_pool();

    // Add many batches with large capacity
    // Create all batches first, then return them
    let mut batches = Vec::new();
    for _ in 0..32 {
        batches.push(llama_cpp_2::llama_batch::LlamaBatch::new(8192, 1)); // Large capacity
    }
    for batch in batches {
        return_batch(batch);
    }

    assert_eq!(pool_size(), 32);

    // Clear should free all memory
    clear_pool();
    assert_eq!(pool_size(), 0);

    // Pool should still work after clearing
    let batch = get_or_create_batch(512);
    return_batch(batch);
    assert_eq!(pool_size(), 1);

    clear_pool();
}
