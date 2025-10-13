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

//! Performance benchmarks for batch processing
//!
//! These benchmarks measure the performance improvements from Phase 3 refactoring:
//! - Small batches (1-5 sequences)
//! - Medium batches (10-50 sequences)
//! - Large batches (100+ sequences)
//!
//! Run with: `cargo bench --bench batch_processing_bench`
//!
//! Requires EMBELLAMA_TEST_MODEL environment variable.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use embellama::{EmbeddingEngine, EngineConfig};
use std::path::PathBuf;

fn setup_engine() -> EmbeddingEngine {
    let model_path =
        std::env::var("EMBELLAMA_TEST_MODEL").expect("EMBELLAMA_TEST_MODEL must be set");

    let config = EngineConfig::builder()
        .with_model_path(PathBuf::from(model_path))
        .with_model_name("bench-model")
        .with_n_batch(2048) // Standard n_batch
        .build()
        .expect("Failed to create config");

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");
    engine.warmup_model(None).expect("Failed to warm up");
    engine
}

fn generate_texts(count: usize, text_length: usize) -> Vec<String> {
    (0..count)
        .map(|i| {
            let content = "The quick brown fox jumps over the lazy dog. ".repeat(text_length);
            format!("Sequence {i}: {content}")
        })
        .collect()
}

/// Benchmark small batches (1-5 sequences)
fn bench_small_batches(c: &mut Criterion) {
    let engine = setup_engine();

    let mut group = c.benchmark_group("small_batches");
    for size in [1, 2, 3, 5] {
        let texts = generate_texts(size, 2); // ~30-40 tokens per sequence
        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &text_refs, |b, texts| {
            b.iter(|| {
                let embeddings = engine
                    .embed_batch(Some("bench-model"), black_box(texts))
                    .expect("Batch processing failed");
                black_box(embeddings);
            });
        });
    }
    group.finish();
}

/// Benchmark medium batches (10-50 sequences)
fn bench_medium_batches(c: &mut Criterion) {
    let engine = setup_engine();

    let mut group = c.benchmark_group("medium_batches");
    for size in [10, 20, 30, 50] {
        let texts = generate_texts(size, 3); // ~50-60 tokens per sequence
        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &text_refs, |b, texts| {
            b.iter(|| {
                let embeddings = engine
                    .embed_batch(Some("bench-model"), black_box(texts))
                    .expect("Batch processing failed");
                black_box(embeddings);
            });
        });
    }
    group.finish();
}

/// Benchmark large batches (100+ sequences)
fn bench_large_batches(c: &mut Criterion) {
    let engine = setup_engine();

    let mut group = c.benchmark_group("large_batches");
    group.sample_size(10); // Fewer samples for large batches

    for size in [100, 200] {
        let texts = generate_texts(size, 2); // ~30-40 tokens per sequence
        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &text_refs, |b, texts| {
            b.iter(|| {
                let embeddings = engine
                    .embed_batch(Some("bench-model"), black_box(texts))
                    .expect("Batch processing failed");
                black_box(embeddings);
            });
        });
    }
    group.finish();
}

/// Benchmark varying sequence lengths
fn bench_variable_length_sequences(c: &mut Criterion) {
    let engine = setup_engine();

    let mut group = c.benchmark_group("variable_length");

    // Mix of short, medium, and long sequences
    let short_texts = generate_texts(10, 1); // ~15-20 tokens
    let medium_texts = generate_texts(10, 5); // ~80-100 tokens
    let long_texts = generate_texts(10, 10); // ~150-200 tokens

    let mut all_texts = Vec::new();
    all_texts.extend(short_texts);
    all_texts.extend(medium_texts);
    all_texts.extend(long_texts);

    let text_refs: Vec<&str> = all_texts.iter().map(String::as_str).collect();

    group.throughput(Throughput::Elements(30));
    group.bench_function("mixed_30_sequences", |b| {
        b.iter(|| {
            let embeddings = engine
                .embed_batch(Some("bench-model"), black_box(&text_refs))
                .expect("Batch processing failed");
            black_box(embeddings);
        });
    });

    group.finish();
}

/// Benchmark effect of n_batch on chunking
fn bench_n_batch_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("n_batch_impact");

    // Create a batch that will require chunking with small n_batch but not with large
    let texts = generate_texts(20, 5); // ~80-100 tokens each = 1600-2000 total
    let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

    // Test with different n_batch values
    for n_batch in [512, 1024, 2048, 4096] {
        let config = EngineConfig::builder()
            .with_model_path(PathBuf::from(
                std::env::var("EMBELLAMA_TEST_MODEL").unwrap(),
            ))
            .with_model_name("bench-model")
            .with_n_batch(n_batch)
            .build()
            .expect("Failed to create config");

        let engine = EmbeddingEngine::new(config).expect("Failed to create engine");
        engine.warmup_model(None).expect("Failed to warm up");

        group.throughput(Throughput::Elements(20));
        group.bench_with_input(
            BenchmarkId::new("n_batch", n_batch),
            &text_refs,
            |b, texts| {
                b.iter(|| {
                    let embeddings = engine
                        .embed_batch(Some("bench-model"), black_box(texts))
                        .expect("Batch processing failed");
                    black_box(embeddings);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark fast path (single sequence)
fn bench_fast_path_single_sequence(c: &mut Criterion) {
    let engine = setup_engine();

    let mut group = c.benchmark_group("fast_path");

    let text = "This is a single test sentence for fast path benchmarking.";

    group.throughput(Throughput::Elements(1));
    group.bench_function("single_sequence", |b| {
        b.iter(|| {
            let embeddings = engine
                .embed_batch(Some("bench-model"), black_box(&[text]))
                .expect("Fast path failed");
            black_box(embeddings);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_small_batches,
    bench_medium_batches,
    bench_large_batches,
    bench_variable_length_sequences,
    bench_n_batch_impact,
    bench_fast_path_single_sequence,
);
criterion_main!(benches);
