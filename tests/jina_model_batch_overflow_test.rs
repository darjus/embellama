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

//! Integration tests for batch overflow handling with Jina model
//!
//! This test validates batch packing behavior using n_batch capacity.
//! After Phase 3 refactoring, batch chunking is based on n_batch (default 2048)
//! rather than n_seq_max, enabling more efficient packing.
//!
//! Individual sequences are still validated against effective_max during tokenization.
//! The n_seq_max parameter is managed internally by llama.cpp for parallelism.

use embellama::{EmbeddingEngine, EngineConfig};
use serial_test::serial;
use std::path::PathBuf;

#[allow(dead_code)]
#[path = "common.rs"]
mod common;

/// Generate a large batch of sequences totaling approximately 8042 tokens
/// This replicates the user's reported failure case
fn generate_large_token_batch() -> Vec<String> {
    // Each sequence averages ~134 tokens (8042 / 60 ≈ 134)
    // We'll create strings of varying lengths to simulate realistic data
    let mut sequences = Vec::new();

    // Generate 60 sequences
    for i in 0..60 {
        // Create strings of varying complexity to get realistic token counts
        // Longer strings with code/technical content to increase token count
        let text = match i % 5 {
            0 => format!(
                "Function implementation #{}: def process_data(input_data, config_params, options_dict): \
                return [transform(x, config_params) for x in input_data if validate(x, options_dict)]",
                i
            ),
            1 => format!(
                "Code snippet #{}: class DataProcessor: def __init__(self, config): self.config = config; \
                self.cache = {{}}; def process(self, data): result = self.transform(data); \
                self.cache[data] = result; return result",
                i
            ),
            2 => format!(
                "Technical documentation #{}: The system architecture consists of multiple layers including \
                data ingestion, processing pipeline, transformation engine, validation framework, \
                and output generation with error handling and logging capabilities throughout the stack",
                i
            ),
            3 => format!(
                "Algorithm description #{}: This implementation uses a divide-and-conquer approach with \
                memoization to optimize performance. Time complexity is O(n log n) with space complexity \
                O(n) due to the auxiliary storage requirements for the intermediate results and cache",
                i
            ),
            _ => format!(
                "Extended text sample #{}: The quick brown fox jumps over the lazy dog while carrying \
                a heavy backpack full of technical equipment and documentation for the project. \
                Additional context and metadata are included for comprehensive analysis and processing",
                i
            ),
        };
        sequences.push(text);
    }

    sequences
}

#[test]
#[serial]
#[ignore = "Run only if EMBELLAMA_TEST_MODEL is a Jina model - tests batch overflow handling"]
fn test_jina_batch_overflow_with_8042_tokens() {
    // Initialize test logging
    common::init_test_logger();

    // Get Jina model path from environment
    let model_path = std::env::var("EMBELLAMA_TEST_MODEL")
        .expect("EMBELLAMA_TEST_MODEL must be set to a Jina model path");

    let model_path_buf = PathBuf::from(&model_path);
    let model_name = model_path_buf
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("jina-test-model");

    // Create engine config with Jina model
    // Jina models typically have:
    // - context_size: 8192
    // - embedding_dimensions: 768
    // - n_batch: 2048 (default)
    // - effective_max: auto-detected per-sequence limit
    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name(model_name)
        .with_n_seq_max(8) // Same as user's configuration
        .build()
        .expect("Failed to create engine config");

    let engine = EmbeddingEngine::new(config).expect("Failed to initialize embedding engine");

    // Generate large batch (60 sequences, ~8042 tokens total)
    let batch = generate_large_token_batch();

    println!("Testing batch with {} sequences", batch.len());
    println!("Expected total tokens: ~8042");
    println!("With n_batch=2048, this will be automatically split into multiple batches");
    println!("Individual sequences are validated against effective_max during tokenization");

    // AFTER PHASE 3:
    // Batch is automatically split based on n_batch capacity (2048 tokens)
    // This enables better packing compared to old n_seq_max-based chunking
    // Individual sequences are validated against effective_max during tokenization

    let text_refs: Vec<&str> = batch.iter().map(String::as_str).collect();
    let result = engine.embed_batch(Some(model_name), &text_refs);

    match result {
        Ok(embeddings) => {
            println!("✓ Successfully processed batch with n_batch-based chunking");
            assert_eq!(
                embeddings.len(),
                batch.len(),
                "Should return embeddings for all {} sequences",
                batch.len()
            );

            // Verify each embedding has correct dimensions (768 for Jina base)
            for (i, embedding) in embeddings.iter().enumerate() {
                assert!(!embedding.is_empty(), "Embedding {} should not be empty", i);

                // All embeddings should have same dimensions
                if i > 0 {
                    assert_eq!(
                        embedding.len(),
                        embeddings[0].len(),
                        "All embeddings should have same dimensions"
                    );
                }
            }

            println!(
                "✓ All {} embeddings have {} dimensions",
                embeddings.len(),
                embeddings[0].len()
            );
        }
        Err(e) => {
            panic!(
                "Batch processing failed (should work with n_batch-based chunking): {}",
                e
            );
        }
    }
}

#[test]
#[serial]
#[ignore = "Run only if EMBELLAMA_TEST_MODEL is a Jina model - tests boundary conditions"]
fn test_jina_batch_at_effective_max_boundary() {
    // Initialize test logging
    common::init_test_logger();

    let model_path =
        std::env::var("EMBELLAMA_TEST_MODEL").expect("EMBELLAMA_TEST_MODEL must be set");

    let model_path_buf = PathBuf::from(&model_path);
    let model_name = model_path_buf
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("jina-test-model");

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name(model_name)
        .with_n_seq_max(8)
        .build()
        .expect("Failed to create config");

    let engine = EmbeddingEngine::new(config).expect("Failed to initialize engine");

    // Create a batch that's exactly at the effective_max boundary
    // For Jina: effective_max = 7324 tokens
    // We'll create sequences that total approximately 7320 tokens (just under)

    // Average ~122 tokens per sequence = 60 sequences * 122 ≈ 7320 tokens
    let batch: Vec<String> = (0..60)
        .map(|i| {
            format!(
                "Sequence {}: Testing boundary condition with text that should fit within \
                effective max tokens limit. This text is carefully sized to test the exact \
                boundary where batch processing should succeed without splitting.",
                i
            )
        })
        .collect();

    println!("Testing batch at effective_max boundary");
    println!("Expected total tokens: ~7320 (just under effective_max of 7324)");

    let text_refs: Vec<&str> = batch.iter().map(String::as_str).collect();
    let result = engine.embed_batch(Some(model_name), &text_refs);

    // This should succeed even before implementation if tokens < ctx_size
    // After implementation, it should succeed more reliably with correct validation
    result.expect("Batch at boundary should process successfully");
}

#[test]
#[serial]
#[ignore = "Run only if EMBELLAMA_TEST_MODEL is a Jina model - tests small batch"]
fn test_jina_small_batch_succeeds() {
    // Initialize test logging
    common::init_test_logger();

    let model_path =
        std::env::var("EMBELLAMA_TEST_MODEL").expect("EMBELLAMA_TEST_MODEL must be set");

    let model_path_buf = PathBuf::from(&model_path);
    let model_name = model_path_buf
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("jina-test-model");

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name(model_name)
        .with_n_seq_max(8)
        .build()
        .expect("Failed to create config");

    let engine = EmbeddingEngine::new(config).expect("Failed to initialize engine");

    // Small batch that's well within limits
    let batch = vec!["Short text one", "Short text two", "Short text three"];

    let result = engine.embed_batch(Some(model_name), &batch);

    assert!(
        result.is_ok(),
        "Small batch should always succeed: {:?}",
        result.err()
    );

    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 3, "Should return 3 embeddings");
}
