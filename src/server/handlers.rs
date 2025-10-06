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

//! HTTP request handlers for the OpenAI-compatible API endpoints
//!
//! This module implements the handler functions for the embeddings
//! and models endpoints.

use crate::extract_gguf_metadata;
use crate::server::api_types::{
    EmbeddingsRequest, EmbeddingsResponse, ErrorResponse, ListModelsResponse, ModelData,
};
use crate::server::channel::WorkerRequest;
use crate::server::state::AppState;
use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use std::path::Path;
use std::time::Duration;
use tokio::sync::oneshot;
use tokio::time::timeout;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Timeout for embedding generation requests
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Handler for POST /v1/embeddings
///
/// Generates embeddings for the provided input text(s) using the specified model.
pub async fn embeddings_handler(
    State(state): State<AppState>,
    Json(request): Json<EmbeddingsRequest>,
) -> Response {
    let request_id = Uuid::new_v4();
    debug!(
        "Processing embeddings request {} for model '{}'",
        request_id, request.model
    );

    // Validate encoding format
    if request.encoding_format != "float" && request.encoding_format != "base64" {
        warn!(
            "Invalid encoding format '{}' in request {}",
            request.encoding_format, request_id
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::invalid_request(format!(
                "Invalid encoding_format '{}'. Must be 'float' or 'base64'",
                request.encoding_format
            ))),
        )
            .into_response();
    }

    // Validate input is not empty
    let text_input = request.input.into_text_input();
    if text_input.is_empty() {
        warn!("Empty input in request {}", request_id);
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::invalid_request("Input cannot be empty")),
        )
            .into_response();
    }

    // Check if model exists
    // > TODO: Validate against actual loaded models once engine exposes model list

    // Create oneshot channel for response
    let (tx, rx) = oneshot::channel();

    // Create worker request
    let worker_request = WorkerRequest {
        id: request_id,
        model: request.model.clone(),
        input: text_input,
        response_tx: tx,
    };

    // Send to dispatcher
    if let Err(e) = state.dispatcher.send(worker_request).await {
        error!("Failed to send request {} to dispatcher: {}", request_id, e);
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse::rate_limit()),
        )
            .into_response();
    }

    // Wait for response with timeout
    match timeout(REQUEST_TIMEOUT, rx).await {
        Ok(Ok(response)) => {
            // Check if embeddings are empty (indicates error in current implementation)
            if response.embeddings.is_empty() {
                error!(
                    "Worker returned empty embeddings for request {}",
                    request_id
                );
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse::internal_error(
                        "Failed to generate embeddings",
                    )),
                )
                    .into_response();
            }

            info!(
                "Request {} completed in {}ms with {} embeddings",
                request_id,
                response.processing_time_ms,
                response.embeddings.len()
            );

            // Create appropriate response based on encoding format
            let embeddings_response = if request.encoding_format == "base64" {
                EmbeddingsResponse::new_base64(
                    request.model,
                    response.embeddings,
                    response.token_count,
                )
            } else {
                EmbeddingsResponse::new(request.model, response.embeddings, response.token_count)
            };

            (StatusCode::OK, Json(embeddings_response)).into_response()
        }
        Ok(Err(_)) => {
            error!("Worker dropped response channel for request {}", request_id);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::internal_error(
                    "Internal communication error",
                )),
            )
                .into_response()
        }
        Err(_) => {
            error!(
                "Request {} timed out after {:?}",
                request_id, REQUEST_TIMEOUT
            );
            (
                StatusCode::REQUEST_TIMEOUT,
                Json(ErrorResponse::internal_error("Request timed out")),
            )
                .into_response()
        }
    }
}

/// Handler for GET /v1/models
///
/// Lists all available models in OpenAI-compatible format.
///
/// # Panics
///
/// Panics if the engine mutex is poisoned
#[allow(clippy::unused_async)] // Required by axum even though we don't await
pub async fn list_models_handler(State(state): State<AppState>) -> Response {
    debug!("Listing available models");

    // Get model list with details from engine
    let engine = state.engine.lock().unwrap();
    let model_details = engine.get_model_details();

    // Convert to ModelData with context size information
    let models: Vec<ModelData> = model_details
        .into_iter()
        .map(|(name, context_size)| ModelData::new_with_context(name, context_size))
        .collect();

    // If no models are loaded, return at least the default configured model
    let models = if models.is_empty() {
        // Extract context size for fallback model
        let context_size = extract_gguf_metadata(Path::new(&state.config.model_path))
            .ok()
            .and_then(|(_, ctx)| u32::try_from(ctx).ok());
        vec![ModelData::new_with_context(
            state.model_name().to_string(),
            context_size,
        )]
    } else {
        models
    };

    let response = ListModelsResponse {
        object: "list".to_string(),
        data: models,
    };

    info!("Returning {} available models", response.data.len());
    (StatusCode::OK, Json(response)).into_response()
}
