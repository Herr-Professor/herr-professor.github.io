---
layout: archive
title: "Resume"
permalink: /resume/
author_profile: true
redirect_from:
  - /cv/
  - /cv
---

{% include base_path %}

[Download PDF resume](/files/Farouq_Oguntoye_Resume.pdf)
{: .notice--primary}

## Summary

ML infrastructure and evaluation engineer with about four years of experience building reproducible evaluation harnesses, verifiable data pipelines, and performance-critical systems. I focus on deterministic testing, benchmark design, observability, and systems-level engineering for AI and data-heavy products.

## Skills

- Languages: C++20, Python, JavaScript/TypeScript, SQL
- ML infrastructure: Gymnasium environments, deterministic verifiers, sandboxed execution, pass@k evaluation with bootstrap confidence intervals, offline RL, experiment tracking, multi-model evaluation registries
- Data systems: Parquet/Arrow, versioned dataset artifacts, data contracts, leakage audits, drift gating, provenance tracking, PostgreSQL, SQLite, async pipelines
- Systems: custom allocators, compression algorithms, SIMD (AVX2), memory management, profiling, observability, thread safety
- Backend and infra: Flask, FastAPI, REST APIs, Docker, Linux, Git, GitHub Actions, CMake

## Selected Projects

- [TinyCodeTest](/projects/tinycodetest/): deterministic code-evaluation environment with sandboxed verifiers, difficulty-bucketed datasets, pass@k scoring, and multi-model benchmarking
- [Market Data Pipeline](/projects/market-data-pipeline/): data-quality framework producing versioned Parquet datasets with checksums, provenance, and automated leakage audits
- [Compression Bench](/projects/compression-bench/): compression benchmark suite implementing classic algorithms from scratch with streaming APIs and reproducible reporting
- [Custom Memory Allocator](/projects/custom-memory-allocator/): thread-safe allocator with benchmark coverage against system and alternative allocators on inference-style workloads
- [F1 Strategy RL Environment](/projects/f1-strategy-rl-environment/): deterministic race-strategy environment with verifiers, stress tests, and RL baselines
- [Heath](/projects/heath/): telemetry simulator and replay tooling for F1 mini-season strategy analysis

## Experience

### Tonfans
Lead Developer, Sep 2024 - Nov 2025

- Owned end-to-end delivery of a production Telegram Mini App integrated with the TON blockchain for community payments and access control
- Implemented purchase and verification flows for NFTs and Jettons with automated ownership gating
- Built secure Python backend workflows and Telegram Bot API integrations for high-volume user interactions

### Lifestores Pharmacy
Systems Engineer, Sep 2022 - Aug 2024

- Built a real-time C++ order-processing system sustaining 500+ orders per hour with sub-second latency
- Developed a thread-safe C++ memory-management library with pooling and free lists, improving throughput by roughly 25%
- Added structured logging and telemetry for critical services, reducing time-to-detect and time-to-resolve incidents

### Ultra Cloud Technologies
Backend Developer, Aug 2021 - Jun 2022

- Developed REST APIs in Python and Flask for authentication and customer-management services
- Optimized PostgreSQL queries and containerized services with Docker to improve performance and deployment reliability

## Education

- B.Sc. Applied Geophysics, Obafemi Awolowo University, Jul 2025
- National Diploma, Mechanical Engineering, Federal Polytechnic Ado Ekiti, Jun 2019
