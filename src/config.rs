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

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::path::{Path, PathBuf};

/// Configuration for a single model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to the GGUF model file
    pub model_path: PathBuf,

    /// Name identifier for the model
    pub model_name: String,

    /// Context size (number of tokens)
    pub n_ctx: Option<u32>,

    /// Number of threads for CPU inference
    pub n_threads: Option<usize>,

    /// Number of GPU layers to offload (0 = CPU only)
    pub n_gpu_layers: Option<u32>,

    /// Use memory mapping for model loading
    /// NOTE: This setting is not yet supported by llama-cpp-2 API
    pub use_mmap: bool,

    /// Use memory locking to prevent swapping
    /// NOTE: This setting is not yet supported by llama-cpp-2 API
    pub use_mlock: bool,

    /// Enable embedding normalization
    pub normalize_embeddings: bool,

    /// Pooling strategy for embeddings
    pub pooling_strategy: PoolingStrategy,

    /// Whether to add BOS (beginning-of-sequence) token during tokenization
    /// None = auto-detect based on model type (encoder models like BERT/E5/BGE/GTE = false, decoder models = true)
    pub add_bos_token: Option<bool>,

    /// Maximum number of sequences for batch processing
    /// Default: 1, max: 64 (llama.cpp limit)
    pub n_seq_max: Option<u32>,
}

impl ModelConfig {
    /// Create a new configuration builder
    pub fn builder() -> ModelConfigBuilder {
        ModelConfigBuilder::new()
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.model_path.as_os_str().is_empty() {
            return Err(Error::config("Model path cannot be empty"));
        }

        if !self.model_path.exists() {
            return Err(Error::config(format!(
                "Model file does not exist: {}",
                self.model_path.display()
            )));
        }

        if self.model_name.trim().is_empty() {
            return Err(Error::config("Model name cannot be empty"));
        }

        if let Some(n_ctx) = self.n_ctx {
            if n_ctx == 0 {
                return Err(Error::config("Context size must be greater than 0"));
            }
        }

        if let Some(n_threads) = self.n_threads {
            if n_threads == 0 {
                return Err(Error::config("Number of threads must be greater than 0"));
            }
        }

        if let Some(n_seq) = self.n_seq_max {
            if n_seq == 0 {
                return Err(Error::config("n_seq_max must be greater than 0"));
            }
            if n_seq > 64 {
                return Err(Error::config(
                    "n_seq_max cannot exceed 64 (llama.cpp limit)",
                ));
            }
        }

        Ok(())
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            model_name: String::new(),
            n_ctx: None,
            n_threads: None,
            n_gpu_layers: None,
            use_mmap: true,
            use_mlock: false,
            normalize_embeddings: true,
            pooling_strategy: PoolingStrategy::default(),
            add_bos_token: None,
            n_seq_max: None,
        }
    }
}

/// Builder for creating ModelConfig instances
pub struct ModelConfigBuilder {
    config: ModelConfig,
}

impl ModelConfigBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: ModelConfig::default(),
        }
    }

    /// Set the model path
    pub fn with_model_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.config.model_path = path.as_ref().to_path_buf();
        self
    }

    /// Set the model name
    pub fn with_model_name<S: Into<String>>(mut self, name: S) -> Self {
        self.config.model_name = name.into();
        self
    }

    /// Set the context size
    pub fn with_n_ctx(mut self, ctx: u32) -> Self {
        self.config.n_ctx = Some(ctx);
        self
    }

    /// Set the number of threads
    pub fn with_n_threads(mut self, threads: usize) -> Self {
        self.config.n_threads = Some(threads);
        self
    }

    /// Set the number of GPU layers
    pub fn with_n_gpu_layers(mut self, layers: u32) -> Self {
        self.config.n_gpu_layers = Some(layers);
        self
    }

    /// Set whether to use memory mapping
    pub fn with_use_mmap(mut self, use_mmap: bool) -> Self {
        self.config.use_mmap = use_mmap;
        self
    }

    /// Set whether to use memory locking
    pub fn with_use_mlock(mut self, use_mlock: bool) -> Self {
        self.config.use_mlock = use_mlock;
        self
    }

    /// Set whether to normalize embeddings
    pub fn with_normalize_embeddings(mut self, normalize: bool) -> Self {
        self.config.normalize_embeddings = normalize;
        self
    }

    /// Set the pooling strategy
    pub fn with_pooling_strategy(mut self, strategy: PoolingStrategy) -> Self {
        self.config.pooling_strategy = strategy;
        self
    }

    /// Set whether to add BOS token during tokenization
    /// None = auto-detect based on model type
    pub fn with_add_bos_token(mut self, add_bos: Option<bool>) -> Self {
        self.config.add_bos_token = add_bos;
        self
    }

    /// Set the maximum number of sequences for batch processing
    /// Default: 1, max: 64 (llama.cpp limit)
    pub fn with_n_seq_max(mut self, n_seq_max: u32) -> Self {
        self.config.n_seq_max = Some(n_seq_max);
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<ModelConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for ModelConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the embedding engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Path to the GGUF model file
    pub model_path: PathBuf,

    /// Name identifier for the model
    pub model_name: String,

    /// Context size for the model (defaults to model's default if None)
    pub context_size: Option<usize>,

    /// Number of threads to use for CPU inference (defaults to number of CPU cores)
    pub n_threads: Option<usize>,

    /// Whether to use GPU acceleration if available
    pub use_gpu: bool,

    /// Number of GPU layers to offload (0 = CPU only)
    pub n_gpu_layers: Option<u32>,

    /// Batch size for processing
    pub batch_size: Option<usize>,

    /// Enable embedding normalization
    pub normalize_embeddings: bool,

    /// Pooling strategy for embeddings
    pub pooling_strategy: PoolingStrategy,

    /// Maximum number of tokens per input
    pub max_tokens: Option<usize>,

    /// Memory limit in MB (None for unlimited)
    pub memory_limit_mb: Option<usize>,

    /// Enable verbose logging
    pub verbose: bool,

    /// Seed for reproducibility (None for random)
    pub seed: Option<u32>,

    /// Temperature for sampling (not typically used for embeddings)
    pub temperature: Option<f32>,

    /// Use memory mapping for model loading
    /// NOTE: This setting is not yet supported by llama-cpp-2 API
    pub use_mmap: bool,

    /// Use memory locking to prevent swapping
    /// NOTE: This setting is not yet supported by llama-cpp-2 API
    pub use_mlock: bool,

    /// Whether to add BOS (beginning-of-sequence) token during tokenization
    /// None = auto-detect based on model type
    pub add_bos_token: Option<bool>,

    /// Maximum number of sequences for batch processing
    /// Default: 1, max: 64 (llama.cpp limit)
    pub n_seq_max: Option<u32>,
}

/// Pooling strategy for combining token embeddings
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Mean pooling across all tokens
    Mean,
    /// Use only the [CLS] token embedding
    Cls,
    /// Max pooling across all tokens
    Max,
    /// Mean pooling with sqrt(length) normalization
    MeanSqrt,
}

impl Default for PoolingStrategy {
    fn default() -> Self {
        Self::Mean
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            model_name: String::new(),
            context_size: None,
            n_threads: None,
            use_gpu: false,
            n_gpu_layers: None,
            batch_size: None,
            normalize_embeddings: false,
            pooling_strategy: PoolingStrategy::default(),
            max_tokens: None,
            memory_limit_mb: None,
            verbose: false,
            seed: None,
            temperature: None,
            use_mmap: true,
            use_mlock: false,
            add_bos_token: None,
            n_seq_max: None,
        }
    }
}

impl EngineConfig {
    /// Create a new configuration builder
    pub fn builder() -> EngineConfigBuilder {
        EngineConfigBuilder::new()
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.model_path.as_os_str().is_empty() {
            return Err(Error::config("Model path cannot be empty"));
        }

        if !self.model_path.exists() {
            return Err(Error::config(format!(
                "Model file does not exist: {}",
                self.model_path.display()
            )));
        }

        if self.model_name.trim().is_empty() {
            return Err(Error::config("Model name cannot be empty"));
        }

        if let Some(context_size) = self.context_size {
            if context_size == 0 {
                return Err(Error::config("Context size must be greater than 0"));
            }
        }

        if let Some(n_threads) = self.n_threads {
            if n_threads == 0 {
                return Err(Error::config("Number of threads must be greater than 0"));
            }
        }

        if let Some(batch_size) = self.batch_size {
            if batch_size == 0 {
                return Err(Error::config("Batch size must be greater than 0"));
            }
        }

        if let Some(max_tokens) = self.max_tokens {
            if max_tokens == 0 {
                return Err(Error::config("Max tokens must be greater than 0"));
            }
        }

        if let Some(n_seq) = self.n_seq_max {
            if n_seq == 0 {
                return Err(Error::config("n_seq_max must be greater than 0"));
            }
            if n_seq > 64 {
                return Err(Error::config(
                    "n_seq_max cannot exceed 64 (llama.cpp limit)",
                ));
            }
        }

        Ok(())
    }

    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let mut builder = EngineConfigBuilder::new();

        if let Ok(path) = env::var("EMBELLAMA_MODEL_PATH") {
            builder = builder.with_model_path(path);
        }

        if let Ok(name) = env::var("EMBELLAMA_MODEL_NAME") {
            builder = builder.with_model_name(name);
        }

        if let Ok(size) = env::var("EMBELLAMA_CONTEXT_SIZE") {
            let size = size
                .parse()
                .map_err(|_| Error::config("Invalid EMBELLAMA_CONTEXT_SIZE value"))?;
            builder = builder.with_context_size(size);
        }

        if let Ok(threads) = env::var("EMBELLAMA_N_THREADS") {
            let threads = threads
                .parse()
                .map_err(|_| Error::config("Invalid EMBELLAMA_N_THREADS value"))?;
            builder = builder.with_n_threads(threads);
        }

        if let Ok(use_gpu) = env::var("EMBELLAMA_USE_GPU") {
            let use_gpu = use_gpu
                .parse()
                .map_err(|_| Error::config("Invalid EMBELLAMA_USE_GPU value"))?;
            builder = builder.with_use_gpu(use_gpu);
        }

        builder.build()
    }

    /// Convert EngineConfig to ModelConfig
    pub fn to_model_config(&self) -> ModelConfig {
        ModelConfig {
            model_path: self.model_path.clone(),
            model_name: self.model_name.clone(),
            n_ctx: self.context_size.map(|s| s as u32),
            n_threads: self.n_threads,
            n_gpu_layers: self.n_gpu_layers,
            use_mmap: self.use_mmap,
            use_mlock: self.use_mlock,
            normalize_embeddings: self.normalize_embeddings,
            pooling_strategy: self.pooling_strategy,
            add_bos_token: self.add_bos_token,
            n_seq_max: self.n_seq_max,
        }
    }
}

/// Builder for creating EngineConfig instances
pub struct EngineConfigBuilder {
    config: EngineConfig,
}

impl EngineConfigBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: EngineConfig::default(),
        }
    }

    /// Set the model path
    pub fn with_model_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.config.model_path = path.as_ref().to_path_buf();
        self
    }

    /// Set the model name
    pub fn with_model_name<S: Into<String>>(mut self, name: S) -> Self {
        self.config.model_name = name.into();
        self
    }

    /// Set the context size
    pub fn with_context_size(mut self, size: usize) -> Self {
        self.config.context_size = Some(size);
        self
    }

    /// Set the number of threads
    pub fn with_n_threads(mut self, threads: usize) -> Self {
        self.config.n_threads = Some(threads);
        self
    }

    /// Set whether to use GPU
    pub fn with_use_gpu(mut self, use_gpu: bool) -> Self {
        self.config.use_gpu = use_gpu;
        self
    }

    /// Set the number of GPU layers
    pub fn with_n_gpu_layers(mut self, layers: u32) -> Self {
        self.config.n_gpu_layers = Some(layers);
        self
    }

    /// Set the batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = Some(size);
        self
    }

    /// Set whether to normalize embeddings
    pub fn with_normalize_embeddings(mut self, normalize: bool) -> Self {
        self.config.normalize_embeddings = normalize;
        self
    }

    /// Set the pooling strategy
    pub fn with_pooling_strategy(mut self, strategy: PoolingStrategy) -> Self {
        self.config.pooling_strategy = strategy;
        self
    }

    /// Set the maximum tokens
    pub fn with_max_tokens(mut self, tokens: usize) -> Self {
        self.config.max_tokens = Some(tokens);
        self
    }

    /// Set the memory limit in MB
    pub fn with_memory_limit_mb(mut self, limit_mb: usize) -> Self {
        self.config.memory_limit_mb = Some(limit_mb);
        self
    }

    /// Set verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Set the seed for reproducibility
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Set the temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = Some(temperature);
        self
    }

    /// Set whether to use memory mapping
    pub fn with_use_mmap(mut self, use_mmap: bool) -> Self {
        self.config.use_mmap = use_mmap;
        self
    }

    /// Set whether to use memory locking
    pub fn with_use_mlock(mut self, use_mlock: bool) -> Self {
        self.config.use_mlock = use_mlock;
        self
    }

    /// Set whether to add BOS token during tokenization
    /// None = auto-detect based on model type
    pub fn with_add_bos_token(mut self, add_bos: Option<bool>) -> Self {
        self.config.add_bos_token = add_bos;
        self
    }

    /// Set the maximum number of sequences for batch processing
    /// Default: 1, max: 64 (llama.cpp limit)
    pub fn with_n_seq_max(mut self, n_seq_max: u32) -> Self {
        self.config.n_seq_max = Some(n_seq_max);
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<EngineConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for EngineConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_model_config_builder() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let config = ModelConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test-model")
            .with_n_ctx(512)
            .with_n_threads(4)
            .with_n_gpu_layers(0)
            .build()
            .unwrap();

        assert_eq!(config.model_path, model_path);
        assert_eq!(config.model_name, "test-model");
        assert_eq!(config.n_ctx, Some(512));
        assert_eq!(config.n_threads, Some(4));
        assert_eq!(config.n_gpu_layers, Some(0));
    }

    #[test]
    fn test_model_config_validation() {
        let result = ModelConfig::builder().with_model_name("test").build();

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            Error::ConfigurationError { .. }
        ));

        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let result = ModelConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_n_ctx(0)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_engine_to_model_config() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let engine_config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test-model")
            .with_context_size(1024)
            .with_n_threads(8)
            .build()
            .unwrap();

        let model_config = engine_config.to_model_config();
        assert_eq!(model_config.model_path, model_path);
        assert_eq!(model_config.model_name, "test-model");
        assert_eq!(model_config.n_ctx, Some(1024));
        assert_eq!(model_config.n_threads, Some(8));
    }

    #[test]
    fn test_config_builder() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test-model")
            .with_context_size(512)
            .with_n_threads(4)
            .with_use_gpu(true)
            .build()
            .unwrap();

        assert_eq!(config.model_path, model_path);
        assert_eq!(config.model_name, "test-model");
        assert_eq!(config.context_size, Some(512));
        assert_eq!(config.n_threads, Some(4));
        assert!(config.use_gpu);
    }

    #[test]
    fn test_config_validation_empty_path() {
        let result = EngineConfig::builder().with_model_name("test").build();

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            Error::ConfigurationError { .. }
        ));
    }

    #[test]
    fn test_config_validation_nonexistent_file() {
        let result = EngineConfig::builder()
            .with_model_path("/nonexistent/path/model.gguf")
            .with_model_name("test")
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_config_validation_empty_name() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let result = EngineConfig::builder().with_model_path(model_path).build();

        assert!(result.is_err());
    }

    #[test]
    fn test_config_validation_invalid_values() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let result = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_context_size(0)
            .build();
        assert!(result.is_err());

        let result = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_n_threads(0)
            .build();
        assert!(result.is_err());

        let result = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_batch_size(0)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_pooling_strategy_default() {
        assert_eq!(PoolingStrategy::default(), PoolingStrategy::Mean);
    }

    #[test]
    fn test_engine_config_full_builder() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("full-test")
            .with_context_size(2048)
            .with_n_threads(16)
            .with_use_gpu(true)
            .with_n_gpu_layers(32)
            .with_normalize_embeddings(true)
            .with_pooling_strategy(PoolingStrategy::Cls)
            .with_batch_size(128)
            .build()
            .unwrap();

        assert_eq!(config.context_size, Some(2048));
        assert_eq!(config.n_threads, Some(16));
        assert!(config.use_gpu);
        assert_eq!(config.n_gpu_layers, Some(32));
        assert!(config.normalize_embeddings);
        assert_eq!(config.pooling_strategy, PoolingStrategy::Cls);
        assert_eq!(config.batch_size, Some(128));
    }

    #[test]
    fn test_model_config_defaults() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let config = ModelConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .build()
            .unwrap();

        // Check that defaults are None
        assert!(config.n_ctx.is_none());
        assert!(config.n_threads.is_none());
        assert!(config.n_gpu_layers.is_none());
    }

    #[test]
    fn test_engine_config_defaults() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .build()
            .unwrap();

        // Check defaults
        assert!(config.context_size.is_none());
        assert!(config.n_threads.is_none());
        assert!(!config.use_gpu);
        assert!(config.n_gpu_layers.is_none());
        assert!(!config.normalize_embeddings);
        assert_eq!(config.pooling_strategy, PoolingStrategy::Mean);
        assert!(config.batch_size.is_none());
    }

    #[test]
    fn test_all_pooling_strategies() {
        let strategies = vec![
            PoolingStrategy::Mean,
            PoolingStrategy::Cls,
            PoolingStrategy::Max,
            PoolingStrategy::MeanSqrt,
        ];

        for strategy in strategies {
            let dir = tempdir().unwrap();
            let model_path = dir.path().join("model.gguf");
            fs::write(&model_path, b"dummy").unwrap();

            let config = EngineConfig::builder()
                .with_model_path(&model_path)
                .with_model_name(format!("test-{:?}", strategy))
                .with_pooling_strategy(strategy.clone())
                .build()
                .unwrap();

            assert_eq!(config.pooling_strategy, strategy);
        }
    }

    #[test]
    fn test_model_config_path_types() {
        let dir = tempdir().unwrap();

        // Test with PathBuf
        let model_path_buf = dir.path().join("model1.gguf");
        fs::write(&model_path_buf, b"dummy").unwrap();

        let config = ModelConfig::builder()
            .with_model_path(&model_path_buf)
            .with_model_name("test1")
            .build()
            .unwrap();
        assert_eq!(config.model_path, model_path_buf);

        // Test with &Path
        let model_path = dir.path().join("model2.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let config = ModelConfig::builder()
            .with_model_path(model_path.as_path())
            .with_model_name("test2")
            .build()
            .unwrap();
        assert_eq!(config.model_path, model_path);

        // Test with String
        let model_path_str = dir.path().join("model3.gguf");
        fs::write(&model_path_str, b"dummy").unwrap();

        let config = ModelConfig::builder()
            .with_model_path(model_path_str.to_str().unwrap())
            .with_model_name("test3")
            .build()
            .unwrap();
        assert_eq!(config.model_path, model_path_str);
    }

    #[test]
    fn test_config_validation_large_values() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        // Test with very large context size
        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_context_size(1_000_000)
            .build()
            .unwrap();
        assert_eq!(config.context_size, Some(1_000_000));

        // Test with large thread count
        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_n_threads(256)
            .build()
            .unwrap();
        assert_eq!(config.n_threads, Some(256));

        // Test with large batch size
        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_batch_size(10000)
            .build()
            .unwrap();
        assert_eq!(config.batch_size, Some(10000));
    }

    #[test]
    fn test_config_clone() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let original = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_context_size(512)
            .build()
            .unwrap();

        let cloned = original.clone();
        assert_eq!(cloned.model_path, original.model_path);
        assert_eq!(cloned.model_name, original.model_name);
        assert_eq!(cloned.context_size, original.context_size);
    }

    #[test]
    fn test_model_config_debug_format() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let config = ModelConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("debug-test")
            .build()
            .unwrap();

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("ModelConfig"));
        assert!(debug_str.contains("debug-test"));
    }

    #[test]
    fn test_special_characters_in_name() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        // Test with special characters in name
        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test-model_v2.0")
            .build()
            .unwrap();
        assert_eq!(config.model_name, "test-model_v2.0");

        // Test with Unicode characters
        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("模型-测试")
            .build()
            .unwrap();
        assert_eq!(config.model_name, "模型-测试");
    }

    #[test]
    fn test_whitespace_in_model_name() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        // Empty name should fail
        let result = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("")
            .build();
        assert!(result.is_err());

        // Whitespace-only name should fail
        let result = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("   ")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_gpu_config_consistency() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        // GPU layers without GPU flag should still be valid
        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_use_gpu(false)
            .with_n_gpu_layers(10)
            .build()
            .unwrap();

        assert!(!config.use_gpu);
        assert_eq!(config.n_gpu_layers, Some(10));
    }
}
