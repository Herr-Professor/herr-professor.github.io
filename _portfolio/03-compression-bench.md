---
title: "Compression Bench"
excerpt: "Compression benchmark suite implementing classic algorithms from scratch with SIMD acceleration and reproducible reporting."
collection: portfolio
permalink: /projects/compression-bench/
tech: "C++20, SIMD, multithreading, CMake, benchmarking"
project_url: "https://github.com/Herr-Professor/compression-bench"
---

Compression Bench is a systems project focused on implementation quality and measurement discipline. It benchmarks multiple lossless compression algorithms under a consistent harness and publishes results in terminal, CSV, and HTML formats.

## What it includes

- Implementations of Huffman, LZ77, DEFLATE, RLE, and LZW from scratch
- Streaming APIs and a self-describing container format with CRC32 verification
- Multi-threaded DEFLATE with AVX2-accelerated byte matching
- Benchmark routines with warmup iterations and median-of-N reporting across a fixed corpus

## Engineering focus

The project emphasizes repeatable performance measurement, explicit correctness checks, and enough tooling around the algorithms to make comparisons meaningful instead of anecdotal.
