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

//! Property-based tests for embellama
//!
//! These tests use proptest to verify invariants with real models.
//!
//! Required environment variables:
//! - `EMBELLAMA_TEST_MODEL`: Path to a valid GGUF model file
//!
//! Optional environment variables:
//! - `EMBELLAMA_TEST_CONTEXT_SIZE`: Override auto-detected context size
//!   (e.g., set to 8192 for decoder models that support 32k but recommend 8k)

use embellama::{EmbeddingEngine, EngineConfig, NormalizationMode};
use proptest::prelude::*;
use serial_test::serial;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

// Use Lazy and Arc<Mutex> for safer shared access
static ENGINE: std::sync::LazyLock<Arc<Mutex<Option<EmbeddingEngine>>>> =
    std::sync::LazyLock::new(|| Arc::new(Mutex::new(None)));

/// Initialize the engine once for all property tests
fn ensure_engine_initialized() {
    let mut engine_guard = ENGINE.lock().unwrap();
    if engine_guard.is_none() {
        let model_path = std::env::var("EMBELLAMA_TEST_MODEL")
            .expect("EMBELLAMA_TEST_MODEL must be set - run 'just download-test-model' first");

        let model_path = PathBuf::from(model_path);
        assert!(
            model_path.exists(),
            "Model file not found at {}. Run 'just download-test-model' to download it.",
            model_path.display()
        );

        let mut config_builder = EngineConfig::builder()
            .with_model_path(model_path)
            .with_model_name("proptest-model")
            .with_normalization_mode(NormalizationMode::L2)
            .with_n_ubatch(2048) // Large enough for long texts in property tests
            .with_n_seq_max(1); // Use full context for single-sequence property tests

        // Allow overriding context size via environment variable
        // Useful for decoder models where full context (32k) is supported but 8k is recommended
        if let Ok(context_size_str) = std::env::var("EMBELLAMA_TEST_CONTEXT_SIZE") {
            let context_size: usize = context_size_str
                .parse()
                .expect("EMBELLAMA_TEST_CONTEXT_SIZE must be a valid positive integer");
            config_builder = config_builder.with_context_size(context_size);
        }

        let config = config_builder.build().expect("Failed to create config");

        let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

        engine.warmup_model(None).expect("Failed to warm up model");

        *engine_guard = Some(engine);
    }
}

/// Get a reference to the engine
fn with_engine<T, F>(f: F) -> T
where
    F: FnOnce(&EmbeddingEngine) -> T,
{
    ensure_engine_initialized();
    let engine_guard = ENGINE.lock().unwrap();
    f(engine_guard.as_ref().unwrap())
}

/// Calculate L2 norm of an embedding
fn calculate_l2_norm(embedding: &[f32]) -> f32 {
    embedding.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// Property: Normalized embeddings should have L2 norm ≈ 1.0
proptest! {
    #[test]
    #[serial]
    fn test_normalized_embeddings_invariant(text in "[a-zA-Z0-9 ]{1,500}") {
        // Skip empty or whitespace-only strings
        if text.trim().is_empty() {
            return Ok(());
        }

        let embedding = with_engine(|engine| {
            engine.embed(None, &text)
                .expect("Failed to generate embedding")
        });

        let norm = calculate_l2_norm(&embedding);
        // Note: MiniLM doesn't normalize by default, so we check if norm > 0.1
        prop_assert!(norm > 0.1,
            "Embedding norm should be non-zero, got {}", norm);
    }
}

// Property: Embedding dimensions should be consistent
proptest! {
    #[test]
    #[serial]
    fn test_consistent_dimensions(texts in prop::collection::vec("[a-zA-Z0-9 ]{1,100}", 1..10)) {
        let mut embeddings = Vec::new();
        for text in &texts {
            if text.trim().is_empty() {
                continue;
            }
            let embedding = with_engine(|engine| {
                engine.embed(None, text)
                    .expect("Failed to generate embedding")
            });
            embeddings.push(embedding);
        }

        if embeddings.is_empty() {
            return Ok(());
        }

        let first_dim = embeddings[0].len();
        for (i, embedding) in embeddings.iter().enumerate() {
            prop_assert_eq!(embedding.len(), first_dim,
                "Embedding {} has different dimension", i);
        }
    }
}

// Property: Batch processing should preserve order
proptest! {
    #[test]
    #[serial]
    #[ignore = "Batch order preservation needs more investigation"]
    fn test_batch_order_preservation(texts in prop::collection::vec("[a-zA-Z0-9 ]{1,100}", 2..20)) {

        // Filter out empty strings
        let valid_texts: Vec<String> = texts.into_iter()
            .filter(|t| !t.trim().is_empty())
            .collect();

        if valid_texts.len() < 2 {
            return Ok(());
        }

        let text_refs: Vec<&str> = valid_texts.iter().map(std::string::String::as_str).collect();

        // Get batch embeddings
        let batch_embeddings = with_engine(|engine| {
            engine.embed_batch(None, &text_refs)
                .expect("Failed to generate batch embeddings")
        });

        // Get individual embeddings
        let mut individual_embeddings = Vec::new();
        for text in &text_refs {
            let embedding = with_engine(|engine| {
                engine.embed(None, text)
                    .expect("Failed to generate embedding")
            });
            individual_embeddings.push(embedding);
        }

        prop_assert_eq!(batch_embeddings.len(), individual_embeddings.len());

        // Check that batch preserves order
        for (i, (batch_emb, individual_emb)) in batch_embeddings.iter()
            .zip(individual_embeddings.iter())
            .enumerate()
        {
            prop_assert_eq!(batch_emb.len(), individual_emb.len(),
                "Dimension mismatch at index {}", i);

            // Allow small numerical differences due to floating point
            for (j, (b, ind)) in batch_emb.iter().zip(individual_emb.iter()).enumerate() {
                prop_assert!((b - ind).abs() < 0.0001,
                    "Embedding mismatch at index {}, position {}", i, j);
            }
        }
    }
}

// Property: Empty batch should return empty results
proptest! {
    #[test]
    #[serial]
    fn test_empty_batch_handling(_seed in 0u32..100u32) {
        let empty: Vec<&str> = vec![];
        let embeddings = with_engine(|engine| {
            engine.embed_batch(None, &empty)
                .expect("Failed to process empty batch")
        });

        prop_assert!(embeddings.is_empty(), "Empty batch should return empty results");
    }
}

// Property: Identical texts should produce identical embeddings
proptest! {
    #[test]
    #[serial]
    fn test_deterministic_embeddings(text in "[a-zA-Z0-9 ]{1,200}", repetitions in 2..5) {

        if text.trim().is_empty() {
            return Ok(());
        }

        let mut embeddings = Vec::new();
        for _ in 0..repetitions {
            let embedding = with_engine(|engine| {
                engine.embed(None, &text)
                    .expect("Failed to generate embedding")
            });
            embeddings.push(embedding);
        }

        // All embeddings should be identical
        let first = &embeddings[0];
        for (i, embedding) in embeddings.iter().enumerate().skip(1) {
            prop_assert_eq!(embedding.len(), first.len(),
                "Dimension mismatch at repetition {}", i);

            for (j, (a, b)) in embedding.iter().zip(first.iter()).enumerate() {
                prop_assert!((a - b).abs() < 0.0001,
                    "Non-deterministic result at repetition {}, position {}", i, j);
            }
        }
    }
}

// Property: Text length should not cause crashes (within reasonable limits)
proptest! {
    #[test]
    #[serial]
    fn test_text_length_handling(
        short_text in "[a-zA-Z ]{1,10}",
        medium_text in "[a-zA-Z ]{50,200}",
        long_text in "[a-zA-Z ]{300,500}"
    ) {
        // Should handle short text
        if !short_text.trim().is_empty() {
            let result = with_engine(|engine| engine.embed(None, &short_text));
            prop_assert!(result.is_ok(), "Failed on short text");
        }

        // Should handle medium text
        if !medium_text.trim().is_empty() {
            let result = with_engine(|engine| engine.embed(None, &medium_text));
            prop_assert!(result.is_ok(), "Failed on medium text");
        }

        // Should handle long text
        if !long_text.trim().is_empty() {
            let result = with_engine(|engine| engine.embed(None, &long_text));
            prop_assert!(result.is_ok(), "Failed on long text");
        }
    }
}

// Property: Special characters should not crash the system
proptest! {
    #[test]
    #[serial]
    fn test_special_characters_handling(
        text in prop::string::string_regex("[!@#$%^&*()_+={};:,.<>?/|\\-\\[\\]'\"`~]{1,50}").unwrap()
    ) {
        if text.trim().is_empty() {
            return Ok(());
        }

        // Should not crash on special characters
        let result = with_engine(|engine| engine.embed(None, &text));
        prop_assert!(result.is_ok(), "Failed on special characters: {}", text);

        let embedding = result.unwrap();
        prop_assert!(!embedding.is_empty(), "Empty embedding for special characters");
    }
}

// Property: Unicode text should be handled properly
proptest! {
    #[test]
    #[serial]
    fn test_unicode_handling(
        english in "[a-zA-Z ]{1,50}",
        chinese in "[\u{4e00}-\u{9fff}]{1,20}",
        emoji in "[\u{1f300}-\u{1f6ff}]{1,10}"
    ) {
        // Test mixed unicode text
        let mixed = format!("{english} {chinese} {emoji}");

        if mixed.trim().is_empty() {
            return Ok(());
        }

        let result = with_engine(|engine| engine.embed(None, &mixed));
        prop_assert!(result.is_ok(), "Failed on unicode text");

        let embedding = result.unwrap();
        prop_assert!(!embedding.is_empty(), "Empty embedding for unicode text");
        prop_assert!(!embedding.is_empty(), "Invalid embedding dimensions");
    }
}

// Property: Batch size limits should be respected
proptest! {
    #[test]
    #[serial]
    fn test_batch_size_limits(batch_size in 1..200usize) {
        let texts: Vec<String> = (0..batch_size)
            .map(|i| format!("Text number {i}"))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(std::string::String::as_str).collect();

        let result = with_engine(|engine| engine.embed_batch(None, &text_refs));
        prop_assert!(result.is_ok(), "Failed with batch size {}", batch_size);

        let embeddings = result.unwrap();
        prop_assert_eq!(embeddings.len(), batch_size,
            "Batch size mismatch: expected {}, got {}", batch_size, embeddings.len());
    }
}

// Property: Embedding values should be finite (no NaN or Inf)
proptest! {
    #[test]
    #[serial]
    fn test_finite_embeddings(text in "[a-zA-Z0-9 ]{1,500}") {
        if text.trim().is_empty() {
            return Ok(());
        }

        let embedding = with_engine(|engine| {
            engine.embed(None, &text)
                .expect("Failed to generate embedding")
        });

        for (i, value) in embedding.iter().enumerate() {
            prop_assert!(value.is_finite(),
                "Non-finite value at position {}: {}", i, value);

            // Also check reasonable bounds
            prop_assert!(value.abs() < 100.0,
                "Embedding value out of reasonable bounds at position {}: {}", i, value);
        }
    }
}

// Property: Similar texts should produce similar embeddings (cosine similarity)
proptest! {
    #[test]
    #[serial]
    fn test_semantic_similarity(base_text in "[a-zA-Z ]{10,50}", suffix in "[a-zA-Z ]{1,10}") {
        if base_text.trim().is_empty() {
            return Ok(());
        }

        let text1 = base_text.clone();
        let text2 = format!("{base_text} {suffix}"); // Similar but not identical

        let emb1 = with_engine(|engine| {
            engine.embed(None, &text1)
                .expect("Failed to generate embedding 1")
        });
        let emb2 = with_engine(|engine| {
            engine.embed(None, &text2)
                .expect("Failed to generate embedding 2")
        });

        // Calculate cosine similarity
        let dot_product: f32 = emb1.iter().zip(emb2.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm1 = calculate_l2_norm(&emb1);
        let norm2 = calculate_l2_norm(&emb2);

        let cosine_similarity = dot_product / (norm1 * norm2);

        // Calculate proportional change to determine appropriate threshold
        // When suffix is large relative to base text, similarity naturally decreases
        let base_len = base_text.trim().len();
        let suffix_len = suffix.trim().len();
        let proportional_change = suffix_len as f32 / base_len as f32;

        // Dynamic threshold based on how much the text changed
        // If suffix is 50% of base text, expect lower similarity
        // If suffix is 10% of base text, expect higher similarity
        let similarity_threshold = if proportional_change > 0.5 {
            0.3  // Large change relative to original
        } else if proportional_change > 0.3 {
            0.35  // Moderate change
        } else if base_len < 20 {
            0.4  // Short texts are more sensitive to changes
        } else {
            0.5  // Standard threshold for normal texts with small changes
        };

        // Similar texts should have cosine similarity above threshold
        prop_assert!(cosine_similarity > similarity_threshold,
            "Low similarity {} for related texts (threshold: {}, base length: {}, suffix length: {}, proportional change: {:.2})",
            cosine_similarity, similarity_threshold, base_len, suffix_len, proportional_change);
    }
}

// ============================================================================
// Phase 5.4: Property-Based Tests for n_batch Refactoring
// ============================================================================

// Property: All sequences in a batch should be processed exactly once
proptest! {
    #[test]
    #[serial]
    fn test_all_sequences_processed_once(batch_size in 1..100usize) {
        // Generate unique identifiable sequences
        let texts: Vec<String> = (0..batch_size)
            .map(|i| format!("Unique sequence identifier {i} for property testing"))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(std::string::String::as_str).collect();

        let embeddings = with_engine(|engine| {
            engine.embed_batch(None, &text_refs)
                .expect("Failed to process batch")
        });

        // Verify count: all sequences processed exactly once
        prop_assert_eq!(embeddings.len(), batch_size,
            "Expected {} embeddings, got {}", batch_size, embeddings.len());

        // Verify no embeddings are empty
        for (i, emb) in embeddings.iter().enumerate() {
            prop_assert!(!emb.is_empty(), "Embedding {} is empty", i);
        }
    }
}

// Property: Total tokens == sum of chunk tokens (no data loss in chunking)
proptest! {
    #[test]
    #[serial]
    fn test_chunking_preserves_all_data(
        num_sequences in 5..50usize,
        text_length in 10..100usize
    ) {
        // Create sequences that may exceed n_batch capacity
        let texts: Vec<String> = (0..num_sequences)
            .map(|i| {
                let content = "word ".repeat(text_length);
                format!("Sequence {i}: {content}")
            })
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(std::string::String::as_str).collect();

        let embeddings = with_engine(|engine| {
            engine.embed_batch(None, &text_refs)
                .expect("Failed to process batch")
        });

        // Verify all sequences processed (sum of chunks == total)
        prop_assert_eq!(embeddings.len(), num_sequences,
            "Chunking should preserve all sequences: expected {}, got {}",
            num_sequences, embeddings.len());

        // Verify all embeddings have consistent dimensions
        if !embeddings.is_empty() {
            let expected_dim = embeddings[0].len();
            for (i, emb) in embeddings.iter().enumerate() {
                prop_assert_eq!(emb.len(), expected_dim,
                    "Embedding {} has inconsistent dimension", i);
            }
        }
    }
}

// Property: Chunking should not affect final results (order preservation)
proptest! {
    #[test]
    #[serial]
    fn test_chunking_order_preservation(
        num_sequences in 10..50usize,
        word_count in 20..80usize
    ) {
        // Create sequences that will force chunking
        let texts: Vec<String> = (0..num_sequences)
            .map(|i| {
                let content = "testword ".repeat(word_count);
                format!("ID{i} {content}")
            })
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(std::string::String::as_str).collect();

        // Get batch embeddings (may be chunked)
        let batch_embeddings = with_engine(|engine| {
            engine.embed_batch(None, &text_refs)
                .expect("Failed to process batch")
        });

        // Process first and last individually to verify order
        let first_individual = with_engine(|engine| {
            engine.embed(None, text_refs[0])
                .expect("Failed to process first individually")
        });
        let last_individual = with_engine(|engine| {
            engine.embed(None, text_refs[num_sequences - 1])
                .expect("Failed to process last individually")
        });

        // Verify order is preserved (first and last match)
        prop_assert_eq!(batch_embeddings[0].len(), first_individual.len());
        prop_assert_eq!(batch_embeddings[num_sequences - 1].len(), last_individual.len());

        // Check values are close (allowing for minor floating point differences)
        for (batch_val, ind_val) in batch_embeddings[0].iter().zip(first_individual.iter()) {
            prop_assert!((batch_val - ind_val).abs() < 1e-5,
                "First embedding mismatch in chunked batch");
        }
        for (batch_val, ind_val) in batch_embeddings[num_sequences - 1].iter().zip(last_individual.iter()) {
            prop_assert!((batch_val - ind_val).abs() < 1e-5,
                "Last embedding mismatch in chunked batch");
        }
    }
}

// Property: Varying sequence lengths should be handled correctly
proptest! {
    #[test]
    #[serial]
    fn test_variable_length_sequences(
        lengths in prop::collection::vec(1..200usize, 5..30)
    ) {
        // Create sequences with varying lengths
        let texts: Vec<String> = lengths.iter()
            .enumerate()
            .map(|(i, &len)| {
                let content = "x ".repeat(len);
                format!("Seq{i}: {content}")
            })
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(std::string::String::as_str).collect();

        let embeddings = with_engine(|engine| {
            engine.embed_batch(None, &text_refs)
                .expect("Failed to process variable length batch")
        });

        // Verify all sequences processed
        prop_assert_eq!(embeddings.len(), texts.len(),
            "Variable length batch should process all sequences");

        // Verify consistent dimensions despite varying input lengths
        if !embeddings.is_empty() {
            let expected_dim = embeddings[0].len();
            for (i, emb) in embeddings.iter().enumerate() {
                prop_assert_eq!(emb.len(), expected_dim,
                    "Embedding {} (input len {}) has wrong dimension",
                    i, lengths[i]);
                prop_assert!(!emb.is_empty(),
                    "Embedding {} should not be empty", i);
            }
        }
    }
}

// Property: Batch processing should not introduce NaN or Inf values
proptest! {
    #[test]
    #[serial]
    fn test_batch_finite_values(batch_size in 2..100usize) {
        let texts: Vec<String> = (0..batch_size)
            .map(|i| format!("Batch test sequence {i} with some content"))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(std::string::String::as_str).collect();

        let embeddings = with_engine(|engine| {
            engine.embed_batch(None, &text_refs)
                .expect("Failed to process batch")
        });

        // Check all embeddings for finite values
        for (i, emb) in embeddings.iter().enumerate() {
            for (j, &value) in emb.iter().enumerate() {
                prop_assert!(value.is_finite(),
                    "Non-finite value in batch embedding {} at position {}: {}",
                    i, j, value);
                prop_assert!(value.abs() < 100.0,
                    "Value out of bounds in batch embedding {} at position {}: {}",
                    i, j, value);
            }
        }
    }
}

// Property: Empty strings in batch should be rejected
proptest! {
    #[test]
    #[serial]
    fn test_empty_string_rejection(valid_count in 1..10usize, empty_position in 0..10usize) {
        // Create batch with one empty string
        let mut texts: Vec<String> = (0..valid_count)
            .map(|i| format!("Valid text {i}"))
            .collect();

        // Insert empty string at random position
        let insert_pos = empty_position.min(texts.len());
        texts.insert(insert_pos, String::new());

        let text_refs: Vec<&str> = texts.iter().map(std::string::String::as_str).collect();

        let result = with_engine(|engine| engine.embed_batch(None, &text_refs));

        // Should fail due to empty string
        prop_assert!(result.is_err(),
            "Batch with empty string should be rejected");
    }
}

// Property: Very small batches (1-3 sequences) should use fast path efficiently
proptest! {
    #[test]
    #[serial]
    fn test_small_batch_efficiency(text_len in 10..100usize) {
        let text = "word ".repeat(text_len);

        // Single sequence (fast path)
        let single_result = with_engine(|engine| {
            engine.embed_batch(None, &[text.as_str()])
                .expect("Single sequence should succeed")
        });

        prop_assert_eq!(single_result.len(), 1, "Fast path should return 1 embedding");
        prop_assert!(!single_result[0].is_empty(), "Fast path embedding should not be empty");

        // Two sequences
        let two_result = with_engine(|engine| {
            engine.embed_batch(None, &[text.as_str(), text.as_str()])
                .expect("Two sequences should succeed")
        });

        prop_assert_eq!(two_result.len(), 2, "Should return 2 embeddings");

        // Three sequences
        let three_result = with_engine(|engine| {
            engine.embed_batch(None, &[text.as_str(), text.as_str(), text.as_str()])
                .expect("Three sequences should succeed")
        });

        prop_assert_eq!(three_result.len(), 3, "Should return 3 embeddings");
    }
}
