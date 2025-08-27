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

use std::path::PathBuf;
use thiserror::Error;

/// Custom error type for the embellama library
#[derive(Error, Debug)]
pub enum Error {
    /// Error when model loading fails
    #[error("Failed to load model from path: {path}")]
    ModelLoadError { 
        /// Path to the model that failed to load
        path: PathBuf,
        #[source]
        /// Underlying error from llama-cpp-2
        source: anyhow::Error,
    },

    /// Error when a requested model is not found
    #[error("Model not found: {name}")]
    ModelNotFound {
        /// Name of the model that was not found
        name: String
    },

    /// Error during embedding generation
    #[error("Failed to generate embedding: {message}")]
    EmbeddingGenerationError { 
        /// Description of what went wrong
        message: String,
        #[source]
        /// Optional underlying error
        source: Option<anyhow::Error>,
    },

    /// Error in configuration
    #[error("Configuration error: {message}")]
    ConfigurationError {
        /// Description of the configuration error
        message: String
    },

    /// Error for invalid input
    #[error("Invalid input: {message}")]
    InvalidInput {
        /// Description of what makes the input invalid
        message: String
    },

    /// Error during model initialization
    #[error("Model initialization failed: {message}")]
    ModelInitError { 
        /// Description of initialization failure
        message: String,
        #[source]
        /// Optional underlying error for init
        source: Option<anyhow::Error>,
    },

    /// Error when creating llama context
    #[error("Context creation failed")]
    ContextError {
        #[source]
        /// Underlying error from context creation
        source: anyhow::Error,
    },

    /// Error during text tokenization
    #[error("Tokenization failed: {message}")]
    TokenizationError {
        /// Description of tokenization failure
        message: String
    },

    /// Error during batch processing
    #[error("Batch processing error: {message}")]
    BatchError { 
        /// Description of batch processing error
        message: String,
        /// Indices of texts that failed processing
        failed_indices: Vec<usize>,
    },

    /// Error from thread pool operations
    #[error("Thread pool error")]
    ThreadPoolError {
        #[source]
        /// Underlying thread pool error
        source: anyhow::Error,
    },

    /// Error when a lock is poisoned
    #[error("Model registry lock poisoned")]
    LockPoisoned,

    /// I/O operation error
    #[error("IO error: {message}")]
    IoError { 
        /// Description of I/O error
        message: String,
        #[source]
        /// The underlying I/O error
        source: std::io::Error,
    },

    /// Error during model warmup
    #[error("Model warmup failed")]
    WarmupError {
        #[source]
        /// Underlying warmup error
        source: anyhow::Error,
    },

    /// Error when embedding dimensions don't match expectations
    #[error("Invalid model dimensions: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected number of dimensions
        expected: usize,
        /// Actual number of dimensions
        actual: usize
    },

    /// Error when resource limits are exceeded
    #[error("Resource limit exceeded: {message}")]
    ResourceLimitExceeded {
        /// Description of which limit was exceeded
        message: String
    },

    /// Error when an operation times out
    #[error("Operation timeout: {message}")]
    Timeout {
        /// Description of what timed out
        message: String
    },

    /// Catch-all for other errors
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