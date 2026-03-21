---
title: "Heath"
excerpt: "F1 telemetry simulator and replay tooling for mini-season strategy analysis with offline RL baselines."
collection: portfolio
permalink: /projects/heath/
tech: "Python, Flask, telemetry simulation, offline RL"
project_url: "https://github.com/Herr-Professor/heath"
---

Heath is a telemetry-heavy simulation project built around race reconstruction and decision analysis. It combines ingest, validation, and replay tooling so strategy experiments can be reviewed end to end instead of as disconnected scripts.

## What it includes

- F1 mini-season telemetry simulator and race reconstruction pipeline
- Track reference alignment plus safety-car and virtual-safety-car inference
- Validation steps for rebuilding race context before policy evaluation
- Offline RL baseline comparisons alongside a Flask replay viewer

## Project goal

The main goal is to bridge data processing and analysis. By reconstructing the race state cleanly, the project makes downstream experimentation more credible and easier to debug.
