// Copyright 2025 Embellama Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Configuration validation tests for n_batch parameter
//!
//! This module tests the validation logic introduced in Phase 1 of the
//! batching strategy refactoring. It ensures that:
//! - n_batch must be > 0
//! - n_batch must be >= n_ubatch
//! - Default values are correct
//! - Configuration builder methods work correctly

use embellama::{EngineConfig, ModelConfig};
use std::fs;
use tempfile::tempdir;

#[allow(dead_code)]
#[path = "common.rs"]
mod common;

// ============================================================================
// Phase 5.1: Unit Tests for Configuration
// ============================================================================

/// Test that n_batch=0 is rejected
#[test]
fn test_n_batch_validation_rejects_zero() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("model.gguf");
    fs::write(&model_path, b"dummy").unwrap();

    let result = ModelConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test")
        .with_n_batch(0)
        .build();

    assert!(result.is_err(), "n_batch=0 should be rejected");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Batch size (n_batch) must be greater than 0"),
        "Error message should mention n_batch validation: {}",
        err_msg
    );
}

/// Test that n_batch < n_ubatch is rejected
#[test]
fn test_n_batch_validation_rejects_less_than_ubatch() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("model.gguf");
    fs::write(&model_path, b"dummy").unwrap();

    // n_batch (1024) < n_ubatch (2048) should fail
    let result = ModelConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test")
        .with_n_batch(1024)
        .with_n_ubatch(2048)
        .build();

    assert!(result.is_err(), "n_batch < n_ubatch should be rejected");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("must be >= micro-batch size"),
        "Error message should mention n_batch >= n_ubatch constraint: {}",
        err_msg
    );
}

/// Test that n_batch == n_ubatch is accepted (recommended for embeddings)
#[test]
fn test_n_batch_validation_accepts_equal_to_ubatch() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("model.gguf");
    fs::write(&model_path, b"dummy").unwrap();

    // n_batch == n_ubatch should pass (recommended for embeddings)
    let config = ModelConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test")
        .with_n_batch(2048)
        .with_n_ubatch(2048)
        .build()
        .expect("n_batch == n_ubatch should be valid");

    assert_eq!(config.n_batch, Some(2048));
    assert_eq!(config.n_ubatch, Some(2048));
}

/// Test that n_batch > n_ubatch is accepted
#[test]
fn test_n_batch_validation_accepts_greater_than_ubatch() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("model.gguf");
    fs::write(&model_path, b"dummy").unwrap();

    // n_batch (4096) > n_ubatch (2048) should pass
    let config = ModelConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test")
        .with_n_batch(4096)
        .with_n_ubatch(2048)
        .build()
        .expect("n_batch > n_ubatch should be valid");

    assert_eq!(config.n_batch, Some(4096));
    assert_eq!(config.n_ubatch, Some(2048));
}

/// Test that n_batch without n_ubatch is accepted
#[test]
fn test_n_batch_validation_without_ubatch() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("model.gguf");
    fs::write(&model_path, b"dummy").unwrap();

    // n_batch set without n_ubatch should pass
    let config = ModelConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test")
        .with_n_batch(2048)
        .build()
        .expect("n_batch without n_ubatch should be valid");

    assert_eq!(config.n_batch, Some(2048));
    assert!(config.n_ubatch.is_none());
}

/// Test that n_ubatch without n_batch is accepted
#[test]
fn test_n_batch_validation_ubatch_without_batch() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("model.gguf");
    fs::write(&model_path, b"dummy").unwrap();

    // n_ubatch set without n_batch should pass
    let config = ModelConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test")
        .with_n_ubatch(512)
        .build()
        .expect("n_ubatch without n_batch should be valid");

    assert!(config.n_batch.is_none());
    assert_eq!(config.n_ubatch, Some(512));
}

/// Test default configuration (n_batch should be None by default)
#[test]
fn test_n_batch_default_is_none() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("model.gguf");
    fs::write(&model_path, b"dummy").unwrap();

    let config = ModelConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test")
        .build()
        .expect("Default config should be valid");

    // Default should be None, actual default (2048) is set in EmbeddingModel::new()
    assert!(config.n_batch.is_none(), "Default n_batch should be None");
}

/// Test ModelConfig builder with n_batch
#[test]
fn test_model_config_builder_with_n_batch() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("model.gguf");
    fs::write(&model_path, b"dummy").unwrap();

    let config = ModelConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test")
        .with_n_batch(4096)
        .build()
        .expect("Builder should work");

    assert_eq!(config.n_batch, Some(4096));
}

/// Test EngineConfig builder with n_batch (convenience method)
#[test]
fn test_engine_config_builder_with_n_batch() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("model.gguf");
    fs::write(&model_path, b"dummy").unwrap();

    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test")
        .with_n_batch(8192)
        .build()
        .expect("Builder should work");

    assert_eq!(config.model_config.n_batch, Some(8192));
}

/// Test configuration override via builder (chaining methods)
#[test]
fn test_configuration_override_via_builder() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("model.gguf");
    fs::write(&model_path, b"dummy").unwrap();

    // Build config with one value
    let config1 = ModelConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test")
        .with_n_batch(2048)
        .build()
        .expect("First config should build");

    assert_eq!(config1.n_batch, Some(2048));

    // Build config with different value (override)
    let config2 = ModelConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test")
        .with_n_batch(4096)
        .build()
        .expect("Second config should build");

    assert_eq!(config2.n_batch, Some(4096));
}

/// Test full configuration with all batch-related parameters
#[test]
fn test_full_batch_configuration() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("model.gguf");
    fs::write(&model_path, b"dummy").unwrap();

    let config = ModelConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test")
        .with_n_ctx(8192)
        .with_n_batch(2048)
        .with_n_ubatch(2048)
        .with_n_seq_max(8)
        .with_context_size(8192)
        .build()
        .expect("Full config should build");

    assert_eq!(config.n_ctx, Some(8192));
    assert_eq!(config.n_batch, Some(2048));
    assert_eq!(config.n_ubatch, Some(2048));
    assert_eq!(config.n_seq_max, Some(8));
    assert_eq!(config.context_size, Some(8192));
}

/// Test large n_batch values are accepted
#[test]
fn test_large_n_batch_values() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("model.gguf");
    fs::write(&model_path, b"dummy").unwrap();

    // Test with very large n_batch (128K tokens)
    let config = ModelConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test")
        .with_n_batch(131072)
        .build()
        .expect("Large n_batch should be valid");

    assert_eq!(config.n_batch, Some(131072));
}

/// Test edge case: n_batch=1 (minimum valid value)
#[test]
fn test_n_batch_minimum_valid_value() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("model.gguf");
    fs::write(&model_path, b"dummy").unwrap();

    let config = ModelConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test")
        .with_n_batch(1)
        .build()
        .expect("n_batch=1 should be valid");

    assert_eq!(config.n_batch, Some(1));
}

/// Test that validation message suggests n_ubatch = n_batch for embeddings
#[test]
fn test_validation_message_suggests_equal_values() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("model.gguf");
    fs::write(&model_path, b"dummy").unwrap();

    let result = ModelConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test")
        .with_n_batch(512)
        .with_n_ubatch(1024)
        .build();

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("consider setting n_ubatch = n_batch"),
        "Error message should suggest n_ubatch = n_batch: {}",
        err_msg
    );
}
