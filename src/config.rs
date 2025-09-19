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

    /// Micro-batch size for prompt processing
    /// This must be >= the number of tokens in any single input
    pub n_ubatch: Option<u32>,

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

    /// Context size override (defaults to n_ctx if not specified)
    /// This controls the KV cache/attention cache size for better performance
    /// Note: Previously named `kv_cache_size` - that field is now deprecated
    pub context_size: Option<u32>,

    /// Deprecated: Use `context_size` instead
    /// This field is maintained for backward compatibility and will be removed in v0.4.0
    #[deprecated(since = "0.3.1", note = "Use `context_size` instead")]
    pub kv_cache_size: Option<u32>,

    /// Enable KV cache optimization for batch processing
    /// This includes batch reordering and similar-length grouping
    pub enable_kv_optimization: bool,
}

impl ModelConfig {
    /// Create a new configuration builder
    pub fn builder() -> ModelConfigBuilder {
        ModelConfigBuilder::new()
    }

    /// Create configuration with backend auto-detection
    pub fn with_backend_detection() -> ModelConfigBuilder {
        let backend = crate::backend::detect_best_backend();
        let mut builder = ModelConfigBuilder::new();

        // Set GPU layers based on backend
        if let Some(gpu_layers) = backend.recommended_gpu_layers() {
            builder = builder.with_n_gpu_layers(gpu_layers);
        }

        builder
    }

    /// Validate the configuration
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The model path is empty
    /// - The model file does not exist
    /// - The model file extension is not `.gguf`
    /// - Invalid thread count (0)
    /// - Invalid batch size (0)
    /// - Invalid `n_seq_max` (0 or > 64)
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

        // Validate GGUF extension
        if self.model_path.extension().and_then(|e| e.to_str()) != Some("gguf") {
            return Err(Error::config(format!(
                "Model file must have .gguf extension: {}",
                self.model_path.display()
            )));
        }

        if self.model_name.trim().is_empty() {
            return Err(Error::config("Model name cannot be empty"));
        }

        if let Some(n_ctx) = self.n_ctx
            && n_ctx == 0
        {
            return Err(Error::config("Context size must be greater than 0"));
        }

        if let Some(n_ubatch) = self.n_ubatch
            && n_ubatch == 0
        {
            return Err(Error::config("Micro-batch size must be greater than 0"));
        }

        if let Some(n_threads) = self.n_threads
            && n_threads == 0
        {
            return Err(Error::config("Number of threads must be greater than 0"));
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
            n_ubatch: None,
            n_threads: None,
            n_gpu_layers: None,
            use_mmap: true,
            use_mlock: false,
            normalize_embeddings: true,
            pooling_strategy: PoolingStrategy::default(),
            add_bos_token: None,
            n_seq_max: None,
            context_size: None,
            #[allow(deprecated)]
            kv_cache_size: None,
            enable_kv_optimization: true,
        }
    }
}

/// Builder for creating `ModelConfig` instances
pub struct ModelConfigBuilder {
    config: ModelConfig,
}

impl ModelConfigBuilder {
    /// Create a new builder with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ModelConfig::default(),
        }
    }

    /// Set the model path
    #[must_use]
    pub fn with_model_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.config.model_path = path.as_ref().to_path_buf();
        self
    }

    /// Set the model name
    #[must_use]
    pub fn with_model_name<S: Into<String>>(mut self, name: S) -> Self {
        self.config.model_name = name.into();
        self
    }

    /// Set the context size
    #[must_use]
    pub fn with_n_ctx(mut self, ctx: u32) -> Self {
        self.config.n_ctx = Some(ctx);
        self
    }

    /// Set the micro-batch size for prompt processing
    #[must_use]
    pub fn with_n_ubatch(mut self, ubatch: u32) -> Self {
        self.config.n_ubatch = Some(ubatch);
        self
    }

    /// Set the number of threads
    #[must_use]
    pub fn with_n_threads(mut self, threads: usize) -> Self {
        self.config.n_threads = Some(threads);
        self
    }

    /// Set the number of GPU layers
    #[must_use]
    pub fn with_n_gpu_layers(mut self, layers: u32) -> Self {
        self.config.n_gpu_layers = Some(layers);
        self
    }

    /// Set whether to use memory mapping
    #[must_use]
    pub fn with_use_mmap(mut self, use_mmap: bool) -> Self {
        self.config.use_mmap = use_mmap;
        self
    }

    /// Set whether to use memory locking
    #[must_use]
    pub fn with_use_mlock(mut self, use_mlock: bool) -> Self {
        self.config.use_mlock = use_mlock;
        self
    }

    /// Set whether to normalize embeddings
    #[must_use]
    pub fn with_normalize_embeddings(mut self, normalize: bool) -> Self {
        self.config.normalize_embeddings = normalize;
        self
    }

    /// Set the pooling strategy
    #[must_use]
    pub fn with_pooling_strategy(mut self, strategy: PoolingStrategy) -> Self {
        self.config.pooling_strategy = strategy;
        self
    }

    /// Set whether to add BOS token during tokenization
    /// None = auto-detect based on model type
    #[must_use]
    pub fn with_add_bos_token(mut self, add_bos: Option<bool>) -> Self {
        self.config.add_bos_token = add_bos;
        self
    }

    /// Set the maximum number of sequences for batch processing
    /// Default: 1, max: 64 (llama.cpp limit)
    #[must_use]
    pub fn with_n_seq_max(mut self, n_seq_max: u32) -> Self {
        self.config.n_seq_max = Some(n_seq_max);
        self
    }

    /// Set the KV cache size
    #[must_use]
    /// Set the context size (KV cache size)
    pub fn with_context_size(mut self, context_size: u32) -> Self {
        self.config.context_size = Some(context_size);
        #[allow(deprecated)]
        {
            self.config.kv_cache_size = Some(context_size); // Maintain backward compat
        }
        self
    }

    /// Deprecated: Use `with_context_size` instead
    #[deprecated(since = "0.3.1", note = "Use `with_context_size` instead")]
    pub fn with_kv_cache_size(mut self, kv_cache_size: u32) -> Self {
        self.config.context_size = Some(kv_cache_size);
        #[allow(deprecated)]
        {
            self.config.kv_cache_size = Some(kv_cache_size);
        }
        self
    }

    /// Enable or disable KV cache optimization
    #[must_use]
    pub fn with_kv_optimization(mut self, enable: bool) -> Self {
        self.config.enable_kv_optimization = enable;
        self
    }

    /// Build the configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration validation fails
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
#[allow(clippy::struct_excessive_bools)]
pub struct EngineConfig {
    /// Path to the GGUF model file
    pub model_path: PathBuf,

    /// Name identifier for the model
    pub model_name: String,

    /// Context size for the model (defaults to model's default if None)
    pub context_size: Option<usize>,

    /// Micro-batch size for prompt processing
    /// This must be >= the number of tokens in any single input
    pub n_ubatch: Option<u32>,

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

    /// Cache configuration
    pub cache: Option<CacheConfig>,
}

/// Pooling strategy for combining token embeddings
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Mean pooling across all tokens
    Mean,
    /// Use only the \[CLS\] token embedding
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

/// Configuration for caching system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Whether caching is enabled
    pub enabled: bool,
    /// Maximum number of entries in token cache
    pub token_cache_size: usize,
    /// Maximum number of entries in embedding cache
    pub embedding_cache_size: usize,
    /// Maximum memory usage in megabytes
    pub max_memory_mb: usize,
    /// Time-to-live for cache entries in seconds
    pub ttl_seconds: u64,
    /// Whether to enable metrics collection
    pub enable_metrics: bool,
    /// Whether prefix caching is enabled
    pub prefix_cache_enabled: bool,
    /// Maximum number of cached prefix sessions
    pub prefix_cache_size: usize,
    /// Minimum prefix length in tokens to consider for caching
    pub min_prefix_length: usize,
    /// Frequency threshold for automatic prefix caching
    pub prefix_frequency_threshold: usize,
    /// TTL for prefix cache sessions in seconds
    pub prefix_ttl_seconds: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            token_cache_size: 10_000,
            embedding_cache_size: 10_000,
            max_memory_mb: 1024,
            ttl_seconds: 3600,
            enable_metrics: true,
            prefix_cache_enabled: false, // Disabled by default for gradual rollout
            prefix_cache_size: 100,
            min_prefix_length: 100,
            prefix_frequency_threshold: 5,
            prefix_ttl_seconds: 7200, // 2 hours
        }
    }
}

impl CacheConfig {
    /// Create a new cache configuration builder
    pub fn builder() -> CacheConfigBuilder {
        CacheConfigBuilder::new()
    }

    /// Validate the cache configuration
    pub fn validate(&self) -> Result<()> {
        if self.token_cache_size == 0 {
            return Err(Error::config("Token cache size must be greater than 0"));
        }
        if self.embedding_cache_size == 0 {
            return Err(Error::config("Embedding cache size must be greater than 0"));
        }
        if self.max_memory_mb == 0 {
            return Err(Error::config("Max memory must be greater than 0"));
        }
        if self.ttl_seconds == 0 {
            return Err(Error::config("TTL must be greater than 0"));
        }
        if self.prefix_cache_enabled {
            if self.prefix_cache_size == 0 {
                return Err(Error::config("Prefix cache size must be greater than 0"));
            }
            if self.min_prefix_length < 50 {
                return Err(Error::config(
                    "Minimum prefix length must be at least 50 tokens",
                ));
            }
            if self.prefix_frequency_threshold == 0 {
                return Err(Error::config(
                    "Prefix frequency threshold must be greater than 0",
                ));
            }
            if self.prefix_ttl_seconds == 0 {
                return Err(Error::config("Prefix TTL must be greater than 0"));
            }
        }
        Ok(())
    }
}

/// Builder for `CacheConfig`
pub struct CacheConfigBuilder {
    config: CacheConfig,
}

impl Default for CacheConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheConfigBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: CacheConfig::default(),
        }
    }

    /// Set whether caching is enabled
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Set token cache size
    pub fn with_token_cache_size(mut self, size: usize) -> Self {
        self.config.token_cache_size = size;
        self
    }

    /// Set embedding cache size
    pub fn with_embedding_cache_size(mut self, size: usize) -> Self {
        self.config.embedding_cache_size = size;
        self
    }

    /// Set maximum memory usage in MB
    pub fn with_max_memory_mb(mut self, mb: usize) -> Self {
        self.config.max_memory_mb = mb;
        self
    }

    /// Set TTL in seconds
    pub fn with_ttl_seconds(mut self, seconds: u64) -> Self {
        self.config.ttl_seconds = seconds;
        self
    }

    /// Set metrics collection enabled
    pub fn with_enable_metrics(mut self, enabled: bool) -> Self {
        self.config.enable_metrics = enabled;
        self
    }

    /// Enable prefix caching
    pub fn with_prefix_cache_enabled(mut self, enabled: bool) -> Self {
        self.config.prefix_cache_enabled = enabled;
        self
    }

    /// Set prefix cache size
    pub fn with_prefix_cache_size(mut self, size: usize) -> Self {
        self.config.prefix_cache_size = size;
        self
    }

    /// Set minimum prefix length
    pub fn with_min_prefix_length(mut self, length: usize) -> Self {
        self.config.min_prefix_length = length;
        self
    }

    /// Set prefix frequency threshold
    pub fn with_prefix_frequency_threshold(mut self, threshold: usize) -> Self {
        self.config.prefix_frequency_threshold = threshold;
        self
    }

    /// Set prefix TTL in seconds
    pub fn with_prefix_ttl_seconds(mut self, seconds: u64) -> Self {
        self.config.prefix_ttl_seconds = seconds;
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<CacheConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            model_name: String::new(),
            context_size: None,
            n_ubatch: None,
            n_threads: None,
            use_gpu: false,
            n_gpu_layers: None,
            batch_size: None,
            normalize_embeddings: true,
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
            cache: None,
        }
    }
}

impl EngineConfig {
    /// Create a new configuration builder
    pub fn builder() -> EngineConfigBuilder {
        EngineConfigBuilder::new()
    }

    /// Create configuration with backend auto-detection
    pub fn with_backend_detection() -> EngineConfigBuilder {
        let backend = crate::backend::detect_best_backend();
        let mut builder = EngineConfigBuilder::new();

        // Set GPU configuration based on backend
        if backend.is_gpu_accelerated() {
            builder = builder.with_use_gpu(true);
            if let Some(gpu_layers) = backend.recommended_gpu_layers() {
                builder = builder.with_n_gpu_layers(gpu_layers);
            }
        }

        builder
    }

    /// Validate the configuration
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The model path is empty
    /// - The model file does not exist
    /// - The model file extension is not `.gguf`
    /// - Invalid thread count (0)
    /// - Invalid batch size (0)
    /// - Invalid `n_seq_max` (0 or > 64)
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

        // Validate GGUF extension
        if self.model_path.extension().and_then(|e| e.to_str()) != Some("gguf") {
            return Err(Error::config(format!(
                "Model file must have .gguf extension: {}",
                self.model_path.display()
            )));
        }

        if self.model_name.trim().is_empty() {
            return Err(Error::config("Model name cannot be empty"));
        }

        if let Some(context_size) = self.context_size
            && context_size == 0
        {
            return Err(Error::config("Context size must be greater than 0"));
        }

        if let Some(n_ubatch) = self.n_ubatch
            && n_ubatch == 0
        {
            return Err(Error::config("Micro-batch size must be greater than 0"));
        }

        if let Some(n_threads) = self.n_threads
            && n_threads == 0
        {
            return Err(Error::config("Number of threads must be greater than 0"));
        }

        if let Some(batch_size) = self.batch_size
            && batch_size == 0
        {
            return Err(Error::config("Batch size must be greater than 0"));
        }

        if let Some(max_tokens) = self.max_tokens
            && max_tokens == 0
        {
            return Err(Error::config("Max tokens must be greater than 0"));
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

        // Validate cache configuration if present
        if let Some(ref cache) = self.cache {
            cache.validate()?;
        }

        Ok(())
    }

    /// Load configuration from environment variables
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration validation fails
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

    /// Convert `EngineConfig` to `ModelConfig`
    pub fn to_model_config(&self) -> ModelConfig {
        ModelConfig {
            model_path: self.model_path.clone(),
            model_name: self.model_name.clone(),
            n_ctx: self.context_size.and_then(|s| u32::try_from(s).ok()),
            n_ubatch: self.n_ubatch,
            n_threads: self.n_threads,
            n_gpu_layers: self.n_gpu_layers,
            use_mmap: self.use_mmap,
            use_mlock: self.use_mlock,
            normalize_embeddings: self.normalize_embeddings,
            pooling_strategy: self.pooling_strategy,
            add_bos_token: self.add_bos_token,
            n_seq_max: self.n_seq_max,
            context_size: self.context_size.and_then(|s| u32::try_from(s).ok()),
            #[allow(deprecated)]
            kv_cache_size: None, // Deprecated field for backward compatibility
            enable_kv_optimization: true, // Enable by default
        }
    }
}

/// Builder for creating `EngineConfig` instances
pub struct EngineConfigBuilder {
    config: EngineConfig,
}

impl EngineConfigBuilder {
    /// Create a new builder with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: EngineConfig::default(),
        }
    }

    /// Set the model path
    #[must_use]
    pub fn with_model_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.config.model_path = path.as_ref().to_path_buf();
        self
    }

    /// Set the model name
    #[must_use]
    pub fn with_model_name<S: Into<String>>(mut self, name: S) -> Self {
        self.config.model_name = name.into();
        self
    }

    /// Set the context size
    #[must_use]
    pub fn with_context_size(mut self, size: usize) -> Self {
        self.config.context_size = Some(size);
        self
    }

    /// Set the micro-batch size for prompt processing
    #[must_use]
    pub fn with_n_ubatch(mut self, ubatch: u32) -> Self {
        self.config.n_ubatch = Some(ubatch);
        self
    }

    /// Set the number of threads
    #[must_use]
    pub fn with_n_threads(mut self, threads: usize) -> Self {
        self.config.n_threads = Some(threads);
        self
    }

    /// Set whether to use GPU
    #[must_use]
    pub fn with_use_gpu(mut self, use_gpu: bool) -> Self {
        self.config.use_gpu = use_gpu;
        self
    }

    /// Set the number of GPU layers
    #[must_use]
    pub fn with_n_gpu_layers(mut self, layers: u32) -> Self {
        self.config.n_gpu_layers = Some(layers);
        self
    }

    /// Set the batch size
    #[must_use]
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = Some(size);
        self
    }

    /// Set whether to normalize embeddings
    #[must_use]
    pub fn with_normalize_embeddings(mut self, normalize: bool) -> Self {
        self.config.normalize_embeddings = normalize;
        self
    }

    /// Set the pooling strategy
    #[must_use]
    pub fn with_pooling_strategy(mut self, strategy: PoolingStrategy) -> Self {
        self.config.pooling_strategy = strategy;
        self
    }

    /// Set the maximum tokens
    #[must_use]
    pub fn with_max_tokens(mut self, tokens: usize) -> Self {
        self.config.max_tokens = Some(tokens);
        self
    }

    /// Set the memory limit in MB
    #[must_use]
    pub fn with_memory_limit_mb(mut self, limit_mb: usize) -> Self {
        self.config.memory_limit_mb = Some(limit_mb);
        self
    }

    /// Set verbose logging
    #[must_use]
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Set the seed for reproducibility
    #[must_use]
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Set the temperature
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = Some(temperature);
        self
    }

    /// Set whether to use memory mapping
    #[must_use]
    pub fn with_use_mmap(mut self, use_mmap: bool) -> Self {
        self.config.use_mmap = use_mmap;
        self
    }

    /// Set whether to use memory locking
    #[must_use]
    pub fn with_use_mlock(mut self, use_mlock: bool) -> Self {
        self.config.use_mlock = use_mlock;
        self
    }

    /// Set whether to add BOS token during tokenization
    /// None = auto-detect based on model type
    #[must_use]
    pub fn with_add_bos_token(mut self, add_bos: Option<bool>) -> Self {
        self.config.add_bos_token = add_bos;
        self
    }

    /// Set the maximum number of sequences for batch processing
    /// Default: 1, max: 64 (llama.cpp limit)
    #[must_use]
    pub fn with_n_seq_max(mut self, n_seq_max: u32) -> Self {
        self.config.n_seq_max = Some(n_seq_max);
        self
    }

    /// Set cache configuration
    #[must_use]
    pub fn with_cache_config(mut self, cache: CacheConfig) -> Self {
        self.config.cache = Some(cache);
        self
    }

    /// Enable caching with default configuration
    #[must_use]
    pub fn with_cache_enabled(mut self) -> Self {
        self.config.cache = Some(CacheConfig::default());
        self
    }

    /// Disable caching
    #[must_use]
    pub fn with_cache_disabled(mut self) -> Self {
        self.config.cache = None;
        self
    }

    /// Build the configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration validation fails
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
        assert!(config.normalize_embeddings);
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
                .with_model_name(format!("test-{strategy:?}"))
                .with_pooling_strategy(strategy)
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

        let debug_str = format!("{config:?}");
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
