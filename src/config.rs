use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::path::{Path, PathBuf};

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
    
    /// Number of GPU layers to offload (0 = CPU only, -1 = all layers)
    pub n_gpu_layers: Option<i32>,
    
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
    pub use_mmap: bool,

    /// Use memory locking to prevent swapping
    pub use_mlock: bool,
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
            use_gpu: true,
            n_gpu_layers: Some(-1),
            batch_size: Some(32),
            normalize_embeddings: true,
            pooling_strategy: PoolingStrategy::default(),
            max_tokens: Some(512),
            memory_limit_mb: None,
            verbose: false,
            seed: None,
            temperature: None,
            use_mmap: true,
            use_mlock: false,
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

        if self.model_name.is_empty() {
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
            let size = size.parse().map_err(|_| {
                Error::config("Invalid EMBELLAMA_CONTEXT_SIZE value")
            })?;
            builder = builder.with_context_size(size);
        }

        if let Ok(threads) = env::var("EMBELLAMA_N_THREADS") {
            let threads = threads.parse().map_err(|_| {
                Error::config("Invalid EMBELLAMA_N_THREADS value")
            })?;
            builder = builder.with_n_threads(threads);
        }

        if let Ok(use_gpu) = env::var("EMBELLAMA_USE_GPU") {
            let use_gpu = use_gpu.parse().map_err(|_| {
                Error::config("Invalid EMBELLAMA_USE_GPU value")
            })?;
            builder = builder.with_use_gpu(use_gpu);
        }

        builder.build()
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
    pub fn with_n_gpu_layers(mut self, layers: i32) -> Self {
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
        let result = EngineConfig::builder()
            .with_model_name("test")
            .build();

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::ConfigurationError { .. }));
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

        let result = EngineConfig::builder()
            .with_model_path(model_path)
            .build();

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
}