---
title: "Compression Benchmark Notes"
collection: talks
permalink: /experiments/compression-benchmark-notes/
focus: "Repeatable measurement, SIMD-aware implementation work, and fair algorithm comparisons"
excerpt: "A closer look at the benchmarking discipline behind Compression Bench."
project_url: "https://github.com/Herr-Professor/compression-bench"
---

Compression Bench is as much about measurement discipline as it is about implementing algorithms.

## Benchmarking approach

- Use warmup iterations before collecting timing data
- Compare algorithms over the same corpus and reporting format
- Keep correctness checks close to the performance path with CRC validation and repeatable harness logic

## Why it matters

Performance claims are only useful when measurement is stable. This experiment focuses on making algorithm comparisons reproducible instead of impressionistic.
