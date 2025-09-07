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

//! OpenAI-compatible API types for the embeddings endpoint
//!
//! This module defines the request and response structures that match
//! the OpenAI API format for maximum compatibility.

use serde::{Deserialize, Serialize};

/// Input type for embeddings - can be a single string or array of strings
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum InputType {
    /// Single text input
    Single(String),
    /// Batch of text inputs
    Batch(Vec<String>),
}

impl InputType {
    /// Convert to the internal TextInput format
    pub fn into_text_input(self) -> crate::server::channel::TextInput {
        match self {
            Self::Single(text) => crate::server::channel::TextInput::Single(text),
            Self::Batch(texts) => crate::server::channel::TextInput::Batch(texts),
        }
    }
}

/// OpenAI-compatible embeddings request
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingsRequest {
    /// Model identifier to use for embeddings
    pub model: String,
    /// Input text(s) to generate embeddings for
    pub input: InputType,
    /// Encoding format for the embeddings ("float" or "base64")
    #[serde(default = "default_encoding_format")]
    pub encoding_format: String,
    /// Optional dimensions for the embedding (for dimension reduction)
    pub dimensions: Option<usize>,
    /// Optional user identifier for tracking
    pub user: Option<String>,
}

fn default_encoding_format() -> String {
    "float".to_string()
}

/// OpenAI-compatible embeddings response
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingsResponse {
    /// Object type (always "list")
    pub object: String,
    /// Array of embedding data
    pub data: Vec<EmbeddingData>,
    /// Model used for generation
    pub model: String,
    /// Token usage statistics
    pub usage: Usage,
}

impl EmbeddingsResponse {
    /// Create a new embeddings response
    pub fn new(model: String, embeddings: Vec<Vec<f32>>, token_count: usize) -> Self {
        let data = embeddings
            .into_iter()
            .enumerate()
            .map(|(index, embedding)| EmbeddingData {
                index,
                object: "embedding".to_string(),
                embedding: EmbeddingValue::Float(embedding),
            })
            .collect();

        Self {
            object: "list".to_string(),
            data,
            model,
            usage: Usage {
                prompt_tokens: token_count,
                total_tokens: token_count,
            },
        }
    }

    /// Create a response with base64-encoded embeddings
    pub fn new_base64(model: String, embeddings: Vec<Vec<f32>>, token_count: usize) -> Self {
        let data = embeddings
            .into_iter()
            .enumerate()
            .map(|(index, embedding)| {
                let bytes: Vec<u8> = embedding
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect();
                let base64 = STANDARD.encode(&bytes);
                
                EmbeddingData {
                    index,
                    object: "embedding".to_string(),
                    embedding: EmbeddingValue::Base64(base64),
                }
            })
            .collect();

        Self {
            object: "list".to_string(),
            data,
            model,
            usage: Usage {
                prompt_tokens: token_count,
                total_tokens: token_count,
            },
        }
    }
}

/// Individual embedding data
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingData {
    /// Index of this embedding in the batch
    pub index: usize,
    /// Object type (always "embedding")
    pub object: String,
    /// The embedding vector
    pub embedding: EmbeddingValue,
}

/// Embedding value - either float array or base64 string
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum EmbeddingValue {
    /// Float array representation
    Float(Vec<f32>),
    /// Base64-encoded representation
    Base64(String),
}

/// Token usage statistics
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    /// Number of tokens in the prompt
    pub prompt_tokens: usize,
    /// Total tokens processed
    pub total_tokens: usize,
}

/// Error response format matching OpenAI API
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    /// Error details
    pub error: ErrorDetail,
}

/// Detailed error information
#[derive(Debug, Clone, Serialize)]
pub struct ErrorDetail {
    /// Error message
    pub message: String,
    /// Error type (e.g., "invalid_request_error")
    #[serde(rename = "type")]
    pub error_type: String,
    /// Optional error code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl ErrorResponse {
    /// Create an invalid request error
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                error_type: "invalid_request_error".to_string(),
                code: None,
            },
        }
    }

    /// Create a model not found error
    pub fn model_not_found(model: &str) -> Self {
        Self {
            error: ErrorDetail {
                message: format!("Model '{}' not found", model),
                error_type: "model_not_found_error".to_string(),
                code: Some("model_not_found".to_string()),
            },
        }
    }

    /// Create a rate limit error
    pub fn rate_limit() -> Self {
        Self {
            error: ErrorDetail {
                message: "Rate limit exceeded. Please try again later.".to_string(),
                error_type: "rate_limit_error".to_string(),
                code: Some("rate_limit_exceeded".to_string()),
            },
        }
    }

    /// Create an internal server error
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                error_type: "internal_error".to_string(),
                code: Some("internal_server_error".to_string()),
            },
        }
    }
}

/// List models response
#[derive(Debug, Clone, Serialize)]
pub struct ListModelsResponse {
    /// Object type (always "list")
    pub object: String,
    /// Array of available models
    pub data: Vec<ModelData>,
}

/// Individual model information
#[derive(Debug, Clone, Serialize)]
pub struct ModelData {
    /// Model identifier
    pub id: String,
    /// Object type (always "model")
    pub object: String,
    /// Unix timestamp of when the model was created
    pub created: i64,
    /// Owner of the model
    pub owned_by: String,
}

impl ModelData {
    /// Create new model data
    pub fn new(id: String) -> Self {
        Self {
            id,
            object: "model".to_string(),
            created: 1700000000, // Fixed timestamp for consistency
            owned_by: "embellama".to_string(),
        }
    }
}

/// Add base64 encoding support
use base64::{engine::general_purpose::STANDARD, Engine as _};