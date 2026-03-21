---
title: "Custom Memory Allocator"
excerpt: "Thread-safe allocator with multiple allocation strategies and benchmark coverage against production-grade allocators."
collection: portfolio
permalink: /projects/custom-memory-allocator/
tech: "C++, pthreads, AVX2, performance profiling"
project_url: "https://github.com/Herr-Professor/memory_allocator"
---

Custom Memory Allocator explores how allocator strategy affects throughput, overhead, and scaling under realistic workloads. It is a low-level systems project built to compare tradeoffs rather than chase a single synthetic benchmark.

## What it includes

- Four allocation strategies: best-fit, fixed-size, pool, and segregated
- Thread-safe access paths and AVX2-aware alignment work
- Reproducible benchmarks against system `malloc`, `jemalloc`, and `tcmalloc`
- CSV outputs and automated plots for comparing throughput, scaling, and overhead ratio

## Why it matters

Allocator behavior becomes visible only when workloads are measured carefully. This project turns allocator design into something inspectable and repeatable, especially for inference-style and RL-adjacent workloads where memory patterns matter.
