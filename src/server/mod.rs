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

//! Server module for Embellama HTTP API
//!
//! This module provides an OpenAI-compatible REST API for the embedding engine.
//! It uses a worker pool architecture to handle the `!Send` constraint of LlamaContext.

pub mod api_types;
pub mod channel;
pub mod dispatcher;
pub mod handlers;
pub mod state;
pub mod worker;

// Re-exports for convenience
pub use state::AppState;