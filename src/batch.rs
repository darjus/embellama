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

//! Batch processing module for the embellama library.
//!
//! This module provides efficient batch processing capabilities for generating
//! embeddings for multiple texts simultaneously. It leverages parallel processing
//! for pre/post-processing while respecting the single-threaded constraint of
//! model inference.

use crate::error::{Error, Result};
use crate::model::EmbeddingModel;
use crate::config::PoolingStrategy;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{debug, instrument};

/// Represents a batch of texts to be processed.
pub struct BatchProcessor {
    /// Maximum number of texts to process in a single batch
    max_batch_size: usize,
    /// Optional progress callback for tracking batch processing
    progress_callback: Option<Arc<dyn Fn(usize, usize) + Send + Sync>>,
    /// Whether to normalize embeddings
    normalize: bool,
    /// Pooling strategy to apply
    pooling_strategy: PoolingStrategy,
}

impl BatchProcessor {
    /// Creates a new batch processor with the specified maximum batch size.
    ///
    /// # Arguments
    ///
    /// * `max_batch_size` - Maximum number of texts to process in a single batch
    ///
    /// # Returns
    ///
    /// Returns a new `BatchProcessor` instance.
    pub fn new(max_batch_size: usize) -> Self {
        BatchProcessor {
            max_batch_size,
            progress_callback: None,
            normalize: true,
            pooling_strategy: PoolingStrategy::Mean,
        }
    }
    
    /// Creates a batch processor with custom configuration.
    pub fn builder() -> BatchProcessorBuilder {
        BatchProcessorBuilder::default()
    }

    /// Processes a batch of texts to generate embeddings.
    ///
    /// This function implements the following pipeline:
    /// 1. Parallel tokenization of all texts (using rayon)
    /// 2. Sequential model inference (respecting !Send constraint)
    /// 3. Parallel post-processing and normalization (using rayon)
    ///
    /// # Arguments
    ///
    /// * `model` - The embedding model to use
    /// * `texts` - Vector of texts to process
    ///
    /// # Returns
    ///
    /// Returns a vector of embedding vectors, maintaining the input order.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - Any text fails tokenization
    /// - Model inference fails
    /// - Memory allocation fails
    #[instrument(skip(self, model, texts), fields(batch_size = texts.len()))]
    pub fn process_batch(
        &self,
        model: &mut EmbeddingModel,
        texts: Vec<&str>,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        
        debug!("Processing batch of {} texts", texts.len());
        
        // Progress tracking
        let progress_counter = Arc::new(AtomicUsize::new(0));
        let total = texts.len();
        
        // Step 1: Parallel validation (not tokenization in Phase 4)
        // Real tokenization happens in model.generate_embedding()
        self.parallel_validate(&texts)?;
        
        // Step 2: Sequential model inference (respecting !Send constraint)
        // This includes tokenization, inference, pooling, and normalization
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            // Process through model (handles everything internally)
            let embedding = model.generate_embedding(text)?;
            embeddings.push(embedding);
            
            // Update progress
            let current = progress_counter.fetch_add(1, Ordering::Relaxed) + 1;
            if let Some(ref callback) = self.progress_callback {
                callback(current, total);
            }
            
            debug!("Processed text {}/{}", current, total);
        }
        
        // Step 3: Return embeddings directly (already normalized by model)
        // > NOTE: In Phase 4, post-processing is handled by model
        // Future phases may add additional parallel post-processing
        
        Ok(embeddings)
    }

    /// Sets a progress callback for batch processing.
    ///
    /// The callback will be called with (current_index, total_count) during processing.
    ///
    /// # Arguments
    ///
    /// * `callback` - A function that receives progress updates
    pub fn set_progress_callback<F>(&mut self, callback: F)
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Arc::new(callback));
    }

    /// Validates texts in parallel.
    ///
    /// # Arguments
    ///
    /// * `texts` - Vector of texts to validate
    ///
    /// # Returns
    ///
    /// Returns Ok if all texts are valid, Err otherwise.
    #[instrument(skip(self, texts), fields(count = texts.len()))]
    fn parallel_validate(&self, texts: &[&str]) -> Result<()> {
        debug!("Validating {} texts in parallel", texts.len());
        
        // Parallel validation with error handling
        texts
            .par_iter()
            .try_for_each(|text| {
                if text.is_empty() {
                    return Err(Error::InvalidInput {
                        message: "Cannot process empty text".to_string(),
                    });
                }
                // Could add more validation here (e.g., max length check)
                Ok(())
            })?;
        
        Ok(())
    }
    
    /// Tokenizes texts in parallel.
    ///
    /// # Arguments
    ///
    /// * `texts` - Vector of texts to tokenize
    ///
    /// # Returns
    ///
    /// Returns a vector of tokenized sequences.
    #[instrument(skip(self, texts), fields(count = texts.len()))]
    fn parallel_tokenize(&self, texts: &[&str]) -> Result<Vec<Vec<i32>>> {
        debug!("Starting parallel tokenization of {} texts", texts.len());
        
        // Process texts in chunks to avoid overwhelming the system
        let chunk_size = self.max_batch_size.min(texts.len());
        
        // Parallel tokenization with error handling
        let results: Result<Vec<_>> = texts
            .par_chunks(chunk_size)
            .flat_map(|chunk| {
                chunk.par_iter().map(|text| {
                    // Validate text
                    if text.is_empty() {
                        return Err(Error::InvalidInput {
                            message: "Cannot tokenize empty text".to_string(),
                        });
                    }
                    
                    // Simulate tokenization (in real implementation, this would use the model's tokenizer)
                    // For now, we'll create placeholder tokens
                    // > NOTE: Real tokenization requires access to model's tokenizer
                    // This is a simplified version for Phase 4 implementation
                    let tokens: Vec<i32> = text.bytes().map(|b| b as i32).collect();
                    
                    if tokens.len() > 512 {  // Example max token limit
                        return Err(Error::InvalidInput {
                            message: format!("Text exceeds maximum token limit: {} > 512", tokens.len()),
                        });
                    }
                    
                    Ok(tokens)
                })
            })
            .collect();
        
        results
    }

    /// Post-processes embeddings in parallel.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Raw embeddings from model
    ///
    /// # Returns
    ///
    /// Returns processed embeddings with normalization applied if configured.
    #[instrument(skip(self, embeddings), fields(count = embeddings.len()))]
    fn parallel_postprocess(
        &self,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<Vec<Vec<f32>>> {
        debug!("Starting parallel post-processing of {} embeddings", embeddings.len());
        
        // Parallel normalization if configured
        let processed: Vec<Vec<f32>> = if self.normalize {
            embeddings
                .into_par_iter()
                .map(|mut embedding| {
                    // L2 normalization
                    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                    
                    if norm > 0.0 {
                        for val in &mut embedding {
                            *val /= norm;
                        }
                    }
                    
                    embedding
                })
                .collect()
        } else {
            embeddings
        };
        
        debug!("Completed post-processing");
        Ok(processed)
    }
}

/// Builder for creating configured BatchProcessor instances.
#[derive(Default)]
pub struct BatchProcessorBuilder {
    max_batch_size: Option<usize>,
    normalize: bool,
    pooling_strategy: PoolingStrategy,
    progress_callback: Option<Arc<dyn Fn(usize, usize) + Send + Sync>>,
}

impl BatchProcessorBuilder {
    /// Sets the maximum batch size.
    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = Some(size);
        self
    }
    
    /// Sets whether to normalize embeddings.
    pub fn with_normalization(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
    
    /// Sets the pooling strategy.
    pub fn with_pooling_strategy(mut self, strategy: PoolingStrategy) -> Self {
        self.pooling_strategy = strategy;
        self
    }
    
    /// Sets the progress callback.
    pub fn with_progress_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Arc::new(callback));
        self
    }
    
    /// Builds the BatchProcessor.
    pub fn build(self) -> BatchProcessor {
        BatchProcessor {
            max_batch_size: self.max_batch_size.unwrap_or(32),
            progress_callback: self.progress_callback,
            normalize: self.normalize,
            pooling_strategy: self.pooling_strategy,
        }
    }
}

/// Utilities for batch processing optimization.
pub mod utils {
    // TODO: Phase 4 - Uncomment when implementing
    // use super::*;

    /// Calculates optimal batch size based on available memory and model size.
    ///
    /// # Arguments
    ///
    /// * `model_size_mb` - Size of the model in megabytes
    /// * `embedding_dim` - Dimension of embeddings
    /// * `available_memory_mb` - Available memory in megabytes
    ///
    /// # Returns
    ///
    /// Returns the recommended batch size.
    pub fn calculate_optimal_batch_size(
        _model_size_mb: usize,
        _embedding_dim: usize,
        _available_memory_mb: usize,
    ) -> usize {
        // TODO: Phase 4 - Implement batch size calculation
        32 // Default placeholder
    }

    /// Chunks a large vector of texts into smaller batches.
    ///
    /// # Arguments
    ///
    /// * `texts` - Vector of texts to chunk
    /// * `chunk_size` - Size of each chunk
    ///
    /// # Returns
    ///
    /// Returns an iterator over chunks.
    pub fn chunk_texts<'a>(
        texts: &'a [&'a str],
        chunk_size: usize,
    ) -> impl Iterator<Item = &'a [&'a str]> {
        texts.chunks(chunk_size)
    }
}

#[cfg(test)]
mod tests {
    use super::utils;

    #[test]
    fn test_chunk_texts() {
        let texts = vec!["a", "b", "c", "d", "e"];
        let chunks: Vec<_> = utils::chunk_texts(&texts, 2).collect();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], &["a", "b"]);
        assert_eq!(chunks[1], &["c", "d"]);
        assert_eq!(chunks[2], &["e"]);
    }

    #[test]
    #[ignore] // Will be enabled in Phase 4
    fn test_batch_processing() {
        // TODO: Phase 4 - Add actual batch processing tests
    }
}