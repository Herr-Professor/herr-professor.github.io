---
title: "TinyCodeTest Evaluation Stack"
collection: talks
permalink: /experiments/tinycodetest-evaluation-stack/
focus: "Deterministic verifiers, difficulty buckets, and multi-model pass@k benchmarking"
excerpt: "Notes on how the TinyCodeTest evaluation stack turns model runs into reproducible, inspectable benchmark results."
project_url: "https://github.com/Herr-Professor/TinyCode"
---

This writeup captures the evaluation design decisions behind TinyCodeTest.

## Core experiment

The main question was how to benchmark code-generation systems without hiding the execution conditions. The environment therefore treats every attempt as a constrained run with explicit verifier logic, timeout boundaries, and scoring rules.

## Design choices

- Deterministic subprocess execution instead of ad hoc notebook-style testing
- Difficulty buckets and adversarial splits to reduce misleading aggregate scores
- Bootstrap confidence intervals so pass@k is interpreted with uncertainty, not as a single magic number

## Outcome

The result is an evaluation stack that is easier to trust, compare, and debug when a model improves in one slice and regresses in another.
