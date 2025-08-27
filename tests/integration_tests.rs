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

//! Integration tests for the embellama library

use embellama::{EmbeddingEngine, EngineConfig, ModelConfig, PoolingStrategy};
use serial_test::serial;
use std::fs;
use std::path::PathBuf;
use tempfile::tempdir;

/// Creates a dummy model file for testing
fn create_test_model_file() -> PathBuf {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("test_model.gguf");
    fs::write(&model_path, b"dummy model file").unwrap();
    model_path
}

/// Creates a test configuration
fn create_test_config(model_path: PathBuf, name: &str) -> EngineConfig {
    EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name(name)
        .with_context_size(512)
        .with_n_threads(1)
        .with_normalize_embeddings(true)
        .with_pooling_strategy(PoolingStrategy::Mean)
        .build()
        .unwrap()
}

#[test]
#[ignore] // Requires actual GGUF model file
#[serial]
fn test_engine_creation_and_embedding() {
    // This test would require a real GGUF model file
    // Set EMBELLAMA_TEST_MODEL environment variable to point to a real model
    let model_path = std::env::var("EMBELLAMA_TEST_MODEL")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            eprintln!("Skipping test: EMBELLAMA_TEST_MODEL not set");
            return PathBuf::from("test.gguf");
        });
    
    if !model_path.exists() {
        eprintln!("Skipping test: Model file does not exist at {:?}", model_path);
        return;
    }

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .with_context_size(512)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");
    
    // Test single embedding
    let text = "Hello, world!";
    let embedding = engine.embed(None, text).expect("Failed to generate embedding");
    
    assert!(!embedding.is_empty());
    assert!(embedding.len() > 0);
    
    // Check that embeddings are normalized (L2 norm should be close to 1.0)
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01, "Embedding should be normalized, got norm: {}", norm);
}

#[test]
#[ignore] // Requires actual GGUF model file
#[serial]
fn test_batch_embeddings() {
    let model_path = std::env::var("EMBELLAMA_TEST_MODEL")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            eprintln!("Skipping test: EMBELLAMA_TEST_MODEL not set");
            return PathBuf::from("test.gguf");
        });
    
    if !model_path.exists() {
        eprintln!("Skipping test: Model file does not exist at {:?}", model_path);
        return;
    }

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");
    
    let texts = vec![
        "First document about technology",
        "Second document about science",
        "Third document about mathematics",
    ];
    
    let embeddings = engine.embed_batch(None, texts.clone())
        .expect("Failed to generate batch embeddings");
    
    assert_eq!(embeddings.len(), texts.len());
    
    // Check that all embeddings have the same dimension
    let dim = embeddings[0].len();
    for (i, emb) in embeddings.iter().enumerate() {
        assert_eq!(emb.len(), dim, "Embedding {} has different dimension", i);
    }
}

#[test]
#[ignore] // Requires actual GGUF model file
#[serial]
fn test_multiple_models() {
    let model_path = std::env::var("EMBELLAMA_TEST_MODEL")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            eprintln!("Skipping test: EMBELLAMA_TEST_MODEL not set");
            return PathBuf::from("test.gguf");
        });
    
    if !model_path.exists() {
        eprintln!("Skipping test: Model file does not exist at {:?}", model_path);
        return;
    }

    // Create engine with first model
    let config1 = EngineConfig::builder()
        .with_model_path(model_path.clone())
        .with_model_name("model1")
        .build()
        .unwrap();

    let mut engine = EmbeddingEngine::new(config1).expect("Failed to create engine");
    
    // Load second model with different config
    let config2 = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("model2")
        .with_normalize_embeddings(false) // Different config
        .build()
        .unwrap();
    
    engine.load_model(config2).expect("Failed to load second model");
    
    // Test that both models are listed
    let models = engine.list_models();
    assert_eq!(models.len(), 2);
    assert!(models.contains(&"model1".to_string()));
    assert!(models.contains(&"model2".to_string()));
    
    // Generate embeddings with both models
    let text = "Test text";
    let emb1 = engine.embed(Some("model1"), text).expect("Failed with model1");
    let emb2 = engine.embed(Some("model2"), text).expect("Failed with model2");
    
    assert!(!emb1.is_empty());
    assert!(!emb2.is_empty());
}

#[test]
#[ignore] // Requires actual GGUF model file  
#[serial]
fn test_error_handling() {
    let model_path = std::env::var("EMBELLAMA_TEST_MODEL")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            eprintln!("Skipping test: EMBELLAMA_TEST_MODEL not set");
            return PathBuf::from("test.gguf");
        });
    
    if !model_path.exists() {
        eprintln!("Skipping test: Model file does not exist at {:?}", model_path);
        return;
    }

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");
    
    // Test empty text
    let result = engine.embed(None, "");
    assert!(result.is_err());
    
    // Test non-existent model
    let result = engine.embed(Some("non-existent"), "test");
    assert!(result.is_err());
}

#[test]
#[ignore] // Requires actual GGUF model file
#[serial]
fn test_pooling_strategies() {
    let model_path = std::env::var("EMBELLAMA_TEST_MODEL")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            eprintln!("Skipping test: EMBELLAMA_TEST_MODEL not set");
            return PathBuf::from("test.gguf");
        });
    
    if !model_path.exists() {
        eprintln!("Skipping test: Model file does not exist at {:?}", model_path);
        return;
    }

    // Test different pooling strategies
    let strategies = vec![
        PoolingStrategy::Mean,
        PoolingStrategy::Cls,
        PoolingStrategy::Max,
        PoolingStrategy::MeanSqrt,
    ];
    
    for strategy in strategies {
        let config = EngineConfig::builder()
            .with_model_path(model_path.clone())
            .with_model_name(format!("model-{:?}", strategy))
            .with_pooling_strategy(strategy)
            .build()
            .unwrap();
            
        let engine = EmbeddingEngine::new(config).expect("Failed to create engine");
        
        let text = "Testing pooling strategy";
        let embedding = engine.embed(None, text).expect("Failed to generate embedding");
        
        assert!(!embedding.is_empty(), "Embedding empty for strategy {:?}", strategy);
    }
}

#[test]
#[ignore] // Requires actual GGUF model file
#[serial]
fn test_model_warmup() {
    let model_path = std::env::var("EMBELLAMA_TEST_MODEL")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            eprintln!("Skipping test: EMBELLAMA_TEST_MODEL not set");
            return PathBuf::from("test.gguf");
        });
    
    if !model_path.exists() {
        eprintln!("Skipping test: Model file does not exist at {:?}", model_path);
        return;
    }

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");
    
    // Warmup should not fail
    engine.warmup_model(None).expect("Warmup failed");
    
    // After warmup, embeddings should work
    let embedding = engine.embed(None, "test").expect("Failed after warmup");
    assert!(!embedding.is_empty());
}

#[test]
#[ignore] // Requires actual GGUF model file
#[serial]
fn test_model_info() {
    let model_path = std::env::var("EMBELLAMA_TEST_MODEL")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            eprintln!("Skipping test: EMBELLAMA_TEST_MODEL not set");
            return PathBuf::from("test.gguf");
        });
    
    if !model_path.exists() {
        eprintln!("Skipping test: Model file does not exist at {:?}", model_path);
        return;
    }

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");
    
    let info = engine.model_info("test-model").expect("Failed to get model info");
    
    assert_eq!(info.name, "test-model");
    assert!(info.dimensions > 0);
    assert!(info.max_tokens > 0);
    assert!(info.model_size > 0);
}

#[test]
fn test_configuration_validation() {
    // Test invalid model path
    let result = EngineConfig::builder()
        .with_model_path("/non/existent/path.gguf")
        .with_model_name("test")
        .build();
    assert!(result.is_err());
    
    // Test empty model name
    let model_path = create_test_model_file();
    let result = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("")
        .build();
    assert!(result.is_err());
}

#[test]
#[ignore] // Requires actual GGUF model file
#[serial]
fn test_thread_safety() {
    use std::thread;
    use std::sync::Arc;
    
    let model_path = std::env::var("EMBELLAMA_TEST_MODEL")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            eprintln!("Skipping test: EMBELLAMA_TEST_MODEL not set");
            return PathBuf::from("test.gguf");
        });
    
    if !model_path.exists() {
        eprintln!("Skipping test: Model file does not exist at {:?}", model_path);
        return;
    }

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .build()
        .unwrap();

    let engine = Arc::new(EmbeddingEngine::new(config).expect("Failed to create engine"));
    
    // Spawn multiple threads that use the same engine
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let engine = engine.clone();
            thread::spawn(move || {
                let text = format!("Thread {} test text", i);
                let embedding = engine.embed(None, &text).expect("Failed in thread");
                assert!(!embedding.is_empty());
            })
        })
        .collect();
    
    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
}

/// Performance benchmark for single embedding
#[test]
#[ignore] // Requires actual GGUF model file
#[serial]
fn bench_single_embedding() {
    let model_path = std::env::var("EMBELLAMA_TEST_MODEL")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            eprintln!("Skipping test: EMBELLAMA_TEST_MODEL not set");
            return PathBuf::from("test.gguf");
        });
    
    if !model_path.exists() {
        eprintln!("Skipping test: Model file does not exist at {:?}", model_path);
        return;
    }

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");
    
    // Warmup
    engine.warmup_model(None).expect("Warmup failed");
    
    let text = "This is a sample text for benchmarking embedding generation performance.";
    
    let start = std::time::Instant::now();
    let iterations = 10;
    
    for _ in 0..iterations {
        let _ = engine.embed(None, text).expect("Failed to generate embedding");
    }
    
    let duration = start.elapsed();
    let avg_time = duration / iterations;
    
    println!("Average time per embedding: {:?}", avg_time);
    
    // Assert performance target (adjust based on hardware)
    assert!(avg_time.as_millis() < 1000, "Embedding generation too slow: {:?}", avg_time);
}