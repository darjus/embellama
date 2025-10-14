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

//! Benchmark for batch allocation overhead with and without pooling.
//!
//! This benchmark measures the performance improvement from batch pooling
//! by comparing direct allocation vs pooled allocation patterns.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use llama_cpp_2::llama_batch::LlamaBatch;

/// Benchmark direct batch allocation without pooling
fn bench_direct_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_allocation");

    for capacity in [512, 1024, 2048, 4096, 8192] {
        group.bench_with_input(
            BenchmarkId::new("direct", capacity),
            &capacity,
            |b, &cap| {
                b.iter(|| {
                    // Allocate and immediately drop (simulating request lifecycle)
                    let batch = LlamaBatch::new(cap, 1);
                    black_box(batch);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark pooled batch allocation
fn bench_pooled_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_allocation");

    for capacity in [512, 1024, 2048, 4096, 8192] {
        group.bench_with_input(
            BenchmarkId::new("pooled", capacity),
            &capacity,
            |b, &cap| {
                // Pre-warm the pool
                embellama::batch_pool::clear_pool();
                let batch = embellama::batch_pool::get_or_create_batch(cap);
                embellama::batch_pool::return_batch(batch);

                b.iter(|| {
                    // Get from pool and return (simulating request lifecycle)
                    let batch = embellama::batch_pool::get_or_create_batch(cap);
                    black_box(&batch);
                    embellama::batch_pool::return_batch(batch);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark allocation patterns with varying pool hit rates
fn bench_mixed_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_patterns");

    // Simulate 90% pool hit rate (typical for steady-state server)
    group.bench_function("90_percent_hits", |b| {
        embellama::batch_pool::clear_pool();
        const CAPACITY: usize = 2048;

        b.iter(|| {
            for i in 0..100 {
                let batch = if i % 10 == 0 {
                    // 10% miss rate - allocate new
                    LlamaBatch::new(CAPACITY, 1)
                } else {
                    // 90% hit rate - use pool
                    embellama::batch_pool::get_or_create_batch(CAPACITY)
                };

                black_box(&batch);
                embellama::batch_pool::return_batch(batch);
            }
        });
    });

    // Simulate 50% pool hit rate (mixed workload)
    group.bench_function("50_percent_hits", |b| {
        embellama::batch_pool::clear_pool();
        const CAPACITY: usize = 2048;

        b.iter(|| {
            for i in 0..100 {
                let batch = if i % 2 == 0 {
                    // 50% miss rate
                    LlamaBatch::new(CAPACITY, 1)
                } else {
                    // 50% hit rate
                    embellama::batch_pool::get_or_create_batch(CAPACITY)
                };

                black_box(&batch);
                embellama::batch_pool::return_batch(batch);
            }
        });
    });

    group.finish();
}

/// Benchmark concurrent allocation from multiple threads
fn bench_concurrent_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent");

    group.bench_function("pooled_4_threads", |b| {
        const CAPACITY: usize = 2048;
        const ITERATIONS_PER_THREAD: usize = 25;

        b.iter(|| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    std::thread::spawn(|| {
                        for _ in 0..ITERATIONS_PER_THREAD {
                            let batch = embellama::batch_pool::get_or_create_batch(CAPACITY);
                            black_box(&batch);
                            embellama::batch_pool::return_batch(batch);
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_direct_allocation,
    bench_pooled_allocation,
    bench_mixed_allocation_patterns,
    bench_concurrent_allocation
);
criterion_main!(benches);
