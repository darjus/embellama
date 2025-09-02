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

//! Common test utilities and fixtures

use std::path::PathBuf;
use std::fs;
use tempfile::TempDir;
use embellama::{EngineConfig, PoolingStrategy};

/// Creates a dummy model file for testing purposes
pub fn create_dummy_model() -> (TempDir, PathBuf) {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let model_path = dir.path().join("test_model.gguf");
    
    // Create a minimal GGUF file structure (simplified for testing)
    // > NOTE: This is not a valid GGUF file but sufficient for path validation tests
    fs::write(&model_path, b"GGUF\x00\x00\x00\x04dummy_model_content")
        .expect("Failed to write dummy model");
    
    (dir, model_path)
}

/// Creates a test configuration with sensible defaults
pub fn create_test_config(model_path: PathBuf) -> EngineConfig {
    EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .with_context_size(512)
        .with_n_threads(2)
        .with_normalize_embeddings(true)
        .with_pooling_strategy(PoolingStrategy::Mean)
        .build()
        .expect("Failed to create test config")
}

/// Gets the path to a real test model if available
/// Returns None if EMBELLAMA_TEST_MODEL environment variable is not set
pub fn get_test_model_path() -> Option<PathBuf> {
    std::env::var("EMBELLAMA_TEST_MODEL")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

/// Checks if real model tests should be run
pub fn should_run_model_tests() -> bool {
    get_test_model_path().is_some()
}

/// Initialize test logging
/// Uses global tracing subscriber to prevent TLS issues
pub fn init_test_logger() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    
    INIT.call_once(|| {
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::from_default_env()
                    .add_directive("embellama=debug".parse().unwrap())
            )
            .with_test_writer()
            .init();
    });
}

/// Generate sample texts for batch testing
pub fn generate_sample_texts(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| format!("Sample text number {} for testing embeddings", i))
        .collect()
}

/// Assert that two embedding vectors are approximately equal
pub fn assert_embeddings_equal(emb1: &[f32], emb2: &[f32], tolerance: f32) {
    assert_eq!(emb1.len(), emb2.len(), "Embeddings have different dimensions");
    
    for (i, (a, b)) in emb1.iter().zip(emb2.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < tolerance,
            "Embedding values differ at index {}: {} vs {} (diff: {})",
            i, a, b, diff
        );
    }
}

/// Calculate L2 norm of an embedding vector
pub fn calculate_l2_norm(embedding: &[f32]) -> f32 {
    embedding.iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
}

/// Assert that an embedding is normalized (L2 norm ≈ 1.0)
pub fn assert_normalized(embedding: &[f32], tolerance: f32) {
    let norm = calculate_l2_norm(embedding);
    assert!(
        (norm - 1.0).abs() < tolerance,
        "Embedding is not normalized. L2 norm: {} (expected: ~1.0)",
        norm
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_dummy_model() {
        let (_dir, model_path) = create_dummy_model();
        assert!(model_path.exists());
        assert!(model_path.to_str().unwrap().ends_with(".gguf"));
    }

    #[test]
    fn test_generate_sample_texts() {
        let texts = generate_sample_texts(5);
        assert_eq!(texts.len(), 5);
        assert!(texts[0].contains("Sample text number 0"));
        assert!(texts[4].contains("Sample text number 4"));
    }

    #[test]
    fn test_l2_norm_calculation() {
        let embedding = vec![0.6, 0.8]; // 3-4-5 triangle
        let norm = calculate_l2_norm(&embedding);
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_assert_embeddings_equal() {
        let emb1 = vec![0.1, 0.2, 0.3];
        let emb2 = vec![0.1001, 0.2001, 0.3001];
        assert_embeddings_equal(&emb1, &emb2, 0.001);
    }
}