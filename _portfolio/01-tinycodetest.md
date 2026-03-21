---
title: "TinyCodeTest"
excerpt: "Deterministic code-evaluation environment with sandboxed verifiers, pass@k scoring, and a browser-based eval runner."
collection: portfolio
permalink: /projects/tinycodetest/
tech: "Python, RL evaluation, sandboxed execution, benchmarking, Vercel"
project_url: "https://github.com/Herr-Professor/TinyCode"
---

TinyCodeTest is a deterministic evaluation environment for code-generation models. It combines verifier-driven task execution with reproducible scoring so model outputs can be compared under the same constraints every time.

## What it does

- Runs generated code in sandboxed subprocesses with per-attempt timeouts and memory caps
- Organizes tasks into difficulty buckets with adversarial dataset splits
- Computes pass@k metrics with bootstrap confidence intervals for more stable comparisons
- Supports multiple model providers and prompt strategies from one evaluation registry

## Why it matters

The project is built around trustable evaluation. Instead of treating model performance as a one-number benchmark, it exposes the execution conditions, verification trace, and scoring pipeline so results can be reproduced and inspected.

## Delivery

- Multi-model registry for OpenAI, Anthropic, and Gemini APIs
- Export paths for JSON, Markdown, and HTML leaderboard output
- Vercel-based UI for dataset upload, model selection, API key management, and live evaluation runs
