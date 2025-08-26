use std::path::PathBuf;
use thiserror::Error;

/// Custom error type for the embellama library
#[derive(Error, Debug)]
pub enum Error {
    #[error("Failed to load model from path: {path}")]
    ModelLoadError { 
        path: PathBuf,
        #[source]
        source: anyhow::Error,
    },

    #[error("Model not found: {name}")]
    ModelNotFound { name: String },

    #[error("Failed to generate embedding: {message}")]
    EmbeddingGenerationError { 
        message: String,
        #[source]
        source: Option<anyhow::Error>,
    },

    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },

    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    #[error("Model initialization failed: {message}")]
    ModelInitError { 
        message: String,
        #[source]
        source: Option<anyhow::Error>,
    },

    #[error("Context creation failed")]
    ContextError {
        #[source]
        source: anyhow::Error,
    },

    #[error("Tokenization failed: {message}")]
    TokenizationError { message: String },

    #[error("Batch processing error: {message}")]
    BatchError { 
        message: String,
        failed_indices: Vec<usize>,
    },

    #[error("Thread pool error")]
    ThreadPoolError {
        #[source]
        source: anyhow::Error,
    },

    #[error("Model registry lock poisoned")]
    LockPoisoned,

    #[error("IO error: {message}")]
    IoError { 
        message: String,
        #[source]
        source: std::io::Error,
    },

    #[error("Model warmup failed")]
    WarmupError {
        #[source]
        source: anyhow::Error,
    },

    #[error("Invalid model dimensions: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Resource limit exceeded: {message}")]
    ResourceLimitExceeded { message: String },

    #[error("Operation timeout: {message}")]
    Timeout { message: String },

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl Error {
    /// Create a configuration error with a custom message
    pub fn config(message: impl Into<String>) -> Self {
        Self::ConfigurationError {
            message: message.into(),
        }
    }

    /// Create an invalid input error with a custom message
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }

    /// Create a model load error
    pub fn model_load(path: PathBuf, source: anyhow::Error) -> Self {
        Self::ModelLoadError { path, source }
    }

    /// Create an embedding generation error
    pub fn embedding_failed(message: impl Into<String>) -> Self {
        Self::EmbeddingGenerationError {
            message: message.into(),
            source: None,
        }
    }

    /// Create an embedding generation error with source
    pub fn embedding_failed_with_source(message: impl Into<String>, source: anyhow::Error) -> Self {
        Self::EmbeddingGenerationError {
            message: message.into(),
            source: Some(source),
        }
    }

    /// Check if this is a retryable error
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Timeout { .. } 
            | Self::ThreadPoolError { .. }
            | Self::LockPoisoned
        )
    }

    /// Check if this is a configuration-related error
    pub fn is_configuration_error(&self) -> bool {
        matches!(
            self,
            Self::ConfigurationError { .. }
            | Self::InvalidInput { .. }
            | Self::ModelNotFound { .. }
        )
    }
}

/// Type alias for Results in this crate
pub type Result<T> = std::result::Result<T, Error>;

/// Convert std::io::Error to our Error type
impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Self::IoError {
            message: err.to_string(),
            source: err,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::ModelNotFound {
            name: "test-model".to_string(),
        };
        assert_eq!(err.to_string(), "Model not found: test-model");
    }

    #[test]
    fn test_error_helpers() {
        let err = Error::config("Invalid setting");
        assert!(err.is_configuration_error());
        assert!(!err.is_retryable());

        let err = Error::Timeout {
            message: "Operation timed out".to_string(),
        };
        assert!(err.is_retryable());
        assert!(!err.is_configuration_error());
    }

    #[test]
    fn test_invalid_input() {
        let err = Error::invalid_input("Empty text provided");
        assert!(matches!(err, Error::InvalidInput { .. }));
        assert!(err.is_configuration_error());
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::IoError { .. }));
    }
}