---
title: "Market Data Pipeline"
excerpt: "Versioned Parquet data pipeline with provenance tracking, deterministic leakage audits, and drift gating."
collection: portfolio
permalink: /projects/market-data-pipeline/
tech: "Python, Parquet, Arrow, SQL, data validation"
project_url: "https://github.com/Herr-Professor/market_data_pipeline"
---

Market Data Pipeline is a reproducible data-quality framework for building versioned datasets with auditability built in from the start. The goal is to make every output easy to trace back to its inputs, configuration, and execution context.

## What it does

- Produces versioned Parquet datasets with manifests, checksums, and run provenance
- Tracks git commit, config hash, and dependency versions for each generated artifact
- Adds deterministic leakage audits for forward-looking joins, survivorship bias, and label misalignment
- Gates releases on drift and validation checks instead of pushing questionable data downstream

## Evaluation layer

The pipeline also includes a baseline evaluation harness that models cost and latency assumptions, then measures how those assumptions affect the apparent value of a signal. That makes it easier to spot inflated results caused by hidden lookahead.
