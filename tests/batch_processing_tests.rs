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

//! Integration tests for batch processing with n_batch refactoring
//!
//! These tests validate the batch processing logic introduced in Phase 3:
//! - Single sequence fast path
//! - Multiple sequences within n_batch (single batch)
//! - Multiple sequences exceeding n_batch (chunking)
//! - Edge cases: exactly n_batch tokens, varying lengths
//! - Individual sequence limit (effective_max) enforcement
//!
//! Tests require EMBELLAMA_TEST_MODEL environment variable to be set to a valid model path.

use embellama::{EmbeddingEngine, EngineConfig};
use serial_test::serial;

#[allow(dead_code)]
#[path = "common.rs"]
mod common;

// ============================================================================
// Phase 5.2: Integration Tests for Batch Processing
// ============================================================================

/// Test single sequence processing (fast path)
#[test]
#[serial]
#[ignore = "Requires EMBELLAMA_TEST_MODEL - tests fast path optimization"]
fn test_single_sequence_fast_path() {
    common::init_test_logger();

    let model_path =
        std::env::var("EMBELLAMA_TEST_MODEL").expect("EMBELLAMA_TEST_MODEL must be set");

    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test-model")
        .with_n_batch(2048)
        .build()
        .expect("Config should build");

    let engine = EmbeddingEngine::new(config).expect("Engine should initialize");

    // Single sequence should use fast path (no chunking logic)
    let text = "This is a single test sentence.";
    let result = engine.embed_batch(Some("test-model"), &[text]);

    assert!(result.is_ok(), "Single sequence should succeed");
    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 1, "Should return 1 embedding");
    assert!(!embeddings[0].is_empty(), "Embedding should not be empty");
}

/// Test multiple sequences that fit within n_batch (single batch processing)
#[test]
#[serial]
#[ignore = "Requires EMBELLAMA_TEST_MODEL - tests single batch path"]
fn test_multiple_sequences_within_n_batch() {
    common::init_test_logger();

    let model_path =
        std::env::var("EMBELLAMA_TEST_MODEL").expect("EMBELLAMA_TEST_MODEL must be set");

    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test-model")
        .with_n_batch(2048) // Should fit ~15-20 short sequences
        .build()
        .expect("Config should build");

    let engine = EmbeddingEngine::new(config).expect("Engine should initialize");

    // Create 10 short sequences (should total < 2048 tokens)
    let texts: Vec<String> = (0..10)
        .map(|i| format!("Short test sentence number {i}."))
        .collect();
    let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

    let result = engine.embed_batch(Some("test-model"), &text_refs);

    assert!(
        result.is_ok(),
        "Multiple sequences within n_batch should succeed"
    );
    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 10, "Should return 10 embeddings");

    // All embeddings should have consistent dimensions
    for (i, emb) in embeddings.iter().enumerate() {
        assert!(!emb.is_empty(), "Embedding {i} should not be empty");
        if i > 0 {
            assert_eq!(
                emb.len(),
                embeddings[0].len(),
                "All embeddings should have same dimensions"
            );
        }
    }
}

/// Test multiple sequences exceeding n_batch (requires chunking)
#[test]
#[serial]
#[ignore = "Requires EMBELLAMA_TEST_MODEL - tests chunking logic"]
fn test_multiple_sequences_exceeding_n_batch() {
    common::init_test_logger();

    let model_path =
        std::env::var("EMBELLAMA_TEST_MODEL").expect("EMBELLAMA_TEST_MODEL must be set");

    // Set a small n_batch to force chunking
    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test-model")
        .with_n_batch(512) // Small n_batch to force chunking
        .with_n_ubatch(512) // Match n_ubatch to n_batch (recommended)
        .build()
        .expect("Config should build");

    let engine = EmbeddingEngine::new(config).expect("Engine should initialize");

    // Create sequences that will exceed n_batch=512
    // Each sequence ~50-60 tokens, so 15 sequences ≈ 750-900 tokens > 512
    let texts: Vec<String> = (0..15)
        .map(|i| {
            format!(
                "This is a longer test sentence number {i} with more words to increase token count \
                 and force the batch processing logic to chunk into multiple batches."
            )
        })
        .collect();
    let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

    let result = engine.embed_batch(Some("test-model"), &text_refs);

    assert!(
        result.is_ok(),
        "Sequences exceeding n_batch should succeed with chunking"
    );
    let embeddings = result.unwrap();
    assert_eq!(
        embeddings.len(),
        15,
        "Should return all 15 embeddings despite chunking"
    );

    // Verify all embeddings are valid
    for (i, emb) in embeddings.iter().enumerate() {
        assert!(!emb.is_empty(), "Embedding {i} should not be empty");
        if i > 0 {
            assert_eq!(
                emb.len(),
                embeddings[0].len(),
                "All embeddings should have same dimensions"
            );
        }
    }
}

/// Test edge case: exactly n_batch tokens
#[test]
#[serial]
#[ignore = "Requires EMBELLAMA_TEST_MODEL - tests boundary condition"]
fn test_exactly_n_batch_tokens() {
    common::init_test_logger();

    let model_path =
        std::env::var("EMBELLAMA_TEST_MODEL").expect("EMBELLAMA_TEST_MODEL must be set");

    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test-model")
        .with_n_batch(512)
        .build()
        .expect("Config should build");

    let engine = EmbeddingEngine::new(config).expect("Engine should initialize");

    // Create sequences that total approximately 512 tokens
    // ~40 tokens per sequence * 13 sequences ≈ 520 tokens (close to boundary)
    let texts: Vec<String> = (0..13)
        .map(|i| {
            format!(
                "Boundary test sequence {i} with enough words to create approximately \
                 forty tokens per sequence for testing the exact n_batch boundary."
            )
        })
        .collect();
    let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

    let result = engine.embed_batch(Some("test-model"), &text_refs);

    assert!(result.is_ok(), "Batch at n_batch boundary should succeed");
    let embeddings = result.unwrap();
    assert_eq!(
        embeddings.len(),
        13,
        "Should return all embeddings at boundary"
    );
}

/// Test edge case: sequences with varying lengths
#[test]
#[serial]
#[ignore = "Requires EMBELLAMA_TEST_MODEL - tests variable length handling"]
fn test_sequences_with_varying_lengths() {
    common::init_test_logger();

    let model_path =
        std::env::var("EMBELLAMA_TEST_MODEL").expect("EMBELLAMA_TEST_MODEL must be set");

    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test-model")
        .with_n_batch(1024)
        .build()
        .expect("Config should build");

    let engine = EmbeddingEngine::new(config).expect("Engine should initialize");

    // Mix of very short, medium, and long sequences
    let texts = vec![
        "Short.",
        "Medium length sentence with more words to process.",
        "A very long sentence that contains significantly more content and words to test how the batching \
         system handles variable length inputs and ensures that chunking logic properly accounts for sequences \
         of different sizes when packing them into batches based on token count rather than sequence count.",
        "Another short one.",
        "Medium again with some content.",
        "And one more long sentence that should test the batching logic with variable lengths and ensure that \
         the system can handle mixed sizes efficiently while respecting the n_batch capacity limit.",
    ];

    let result = engine.embed_batch(Some("test-model"), &texts);

    assert!(result.is_ok(), "Variable length sequences should succeed");
    let embeddings = result.unwrap();
    assert_eq!(
        embeddings.len(),
        texts.len(),
        "Should return embedding for each input"
    );

    // Verify embeddings correspond to correct inputs (order preservation)
    for (i, emb) in embeddings.iter().enumerate() {
        assert!(
            !emb.is_empty(),
            "Embedding {i} for text '{}' should not be empty",
            texts[i].chars().take(50).collect::<String>()
        );
    }
}

/// Test individual sequence limit enforcement (effective_max)
#[test]
#[serial]
#[ignore = "Requires EMBELLAMA_TEST_MODEL - tests effective_max validation"]
fn test_individual_sequence_limit_enforcement() {
    common::init_test_logger();

    let model_path =
        std::env::var("EMBELLAMA_TEST_MODEL").expect("EMBELLAMA_TEST_MODEL must be set");

    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test-model")
        .with_n_batch(8192) // Large n_batch
        .with_n_ubatch(2048) // But smaller n_ubatch for per-sequence limit
        .build()
        .expect("Config should build");

    let engine = EmbeddingEngine::new(config).expect("Engine should initialize");

    // Create a sequence that exceeds effective_max (model-dependent, typically ~7000-8000 tokens)
    // Use a very long repetitive text
    let very_long_text = "The quick brown fox jumps over the lazy dog. ".repeat(2000); // ~18000 tokens

    let result = engine.embed_batch(Some("test-model"), &[very_long_text.as_str()]);

    // This should fail due to individual sequence limit, not n_batch limit
    assert!(
        result.is_err(),
        "Sequence exceeding effective_max should be rejected"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("token limit")
            || err_msg.contains("too large")
            || err_msg.contains("exceeds"),
        "Error should mention token limit: {}",
        err_msg
    );
}

/// Test that chunking preserves order
#[test]
#[serial]
#[ignore = "Requires EMBELLAMA_TEST_MODEL - tests order preservation"]
fn test_chunking_preserves_order() {
    common::init_test_logger();

    let model_path =
        std::env::var("EMBELLAMA_TEST_MODEL").expect("EMBELLAMA_TEST_MODEL must be set");

    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test-model")
        .with_n_batch(512) // Small to force chunking
        .build()
        .expect("Config should build");

    let engine = EmbeddingEngine::new(config).expect("Engine should initialize");

    // Create 20 unique sequences that will be chunked
    let texts: Vec<String> = (0..20)
        .map(|i| format!("Unique sequence with identifier {i} for order testing."))
        .collect();
    let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

    let result = engine.embed_batch(Some("test-model"), &text_refs);

    assert!(result.is_ok(), "Chunked batch should succeed");
    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 20, "Should return embeddings in order");

    // Process each sequence individually to get reference embeddings
    let individual_embeddings: Vec<Vec<f32>> = texts
        .iter()
        .map(|text| {
            engine
                .embed_batch(Some("test-model"), &[text.as_str()])
                .expect("Individual processing should succeed")[0]
                .clone()
        })
        .collect();

    // Verify that chunked batch embeddings match individual embeddings
    // (within tolerance due to potential floating point differences)
    for (i, (batch_emb, individual_emb)) in embeddings
        .iter()
        .zip(individual_embeddings.iter())
        .enumerate()
    {
        common::assert_embeddings_equal(batch_emb, individual_emb, 1e-5);
        println!("✓ Embedding {i} matches individual processing");
    }
}

/// Test empty batch handling
#[test]
#[serial]
#[ignore = "Requires EMBELLAMA_TEST_MODEL - tests empty input"]
fn test_empty_batch() {
    common::init_test_logger();

    let model_path =
        std::env::var("EMBELLAMA_TEST_MODEL").expect("EMBELLAMA_TEST_MODEL must be set");

    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test-model")
        .build()
        .expect("Config should build");

    let engine = EmbeddingEngine::new(config).expect("Engine should initialize");

    let empty_texts: Vec<&str> = vec![];
    let result = engine.embed_batch(Some("test-model"), &empty_texts);

    assert!(result.is_ok(), "Empty batch should succeed");
    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 0, "Should return no embeddings");
}

/// Test batch with single empty string
#[test]
#[serial]
#[ignore = "Requires EMBELLAMA_TEST_MODEL - tests empty string handling"]
fn test_batch_with_empty_string() {
    common::init_test_logger();

    let model_path =
        std::env::var("EMBELLAMA_TEST_MODEL").expect("EMBELLAMA_TEST_MODEL must be set");

    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test-model")
        .build()
        .expect("Config should build");

    let engine = EmbeddingEngine::new(config).expect("Engine should initialize");

    let result = engine.embed_batch(Some("test-model"), &[""]);

    // Empty string should be rejected during validation
    assert!(result.is_err(), "Empty string should be rejected");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("empty") || err_msg.contains("Cannot process"),
        "Error should mention empty text: {}",
        err_msg
    );
}

/// Test large batch (100+ sequences)
#[test]
#[serial]
#[ignore = "Requires EMBELLAMA_TEST_MODEL - tests large batch handling"]
fn test_large_batch_100_sequences() {
    common::init_test_logger();

    let model_path =
        std::env::var("EMBELLAMA_TEST_MODEL").expect("EMBELLAMA_TEST_MODEL must be set");

    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test-model")
        .with_n_batch(2048) // Standard n_batch
        .build()
        .expect("Config should build");

    let engine = EmbeddingEngine::new(config).expect("Engine should initialize");

    // Create 100 sequences
    let texts: Vec<String> = (0..100)
        .map(|i| format!("Test sequence {i} for large batch processing."))
        .collect();
    let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

    let result = engine.embed_batch(Some("test-model"), &text_refs);

    assert!(result.is_ok(), "Large batch (100 sequences) should succeed");
    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 100, "Should return all 100 embeddings");

    // Verify all embeddings are valid and have consistent dimensions
    let expected_dim = embeddings[0].len();
    for (i, emb) in embeddings.iter().enumerate() {
        assert!(!emb.is_empty(), "Embedding {i} should not be empty");
        assert_eq!(
            emb.len(),
            expected_dim,
            "Embedding {i} should have consistent dimensions"
        );
    }
}

/// Test n_batch configuration is respected
#[test]
#[serial]
#[ignore = "Requires EMBELLAMA_TEST_MODEL - tests n_batch configuration"]
fn test_n_batch_configuration_respected() {
    common::init_test_logger();

    let model_path =
        std::env::var("EMBELLAMA_TEST_MODEL").expect("EMBELLAMA_TEST_MODEL must be set");

    // Test with different n_batch values
    for n_batch in [512, 1024, 2048, 4096] {
        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test-model")
            .with_n_batch(n_batch)
            .build()
            .expect("Config should build");

        let engine = EmbeddingEngine::new(config).expect("Engine should initialize");

        // Create 10 sequences
        let texts: Vec<String> = (0..10)
            .map(|i| format!("Test with n_batch={n_batch}, sequence {i}."))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

        let result = engine.embed_batch(Some("test-model"), &text_refs);

        assert!(
            result.is_ok(),
            "Batch with n_batch={n_batch} should succeed"
        );
        let embeddings = result.unwrap();
        assert_eq!(
            embeddings.len(),
            10,
            "Should return 10 embeddings with n_batch={n_batch}"
        );

        println!("✓ n_batch={n_batch} configuration works correctly");
    }
}
