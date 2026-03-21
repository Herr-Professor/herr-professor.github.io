---
title: "F1 Strategy Lab Notes"
collection: talks
permalink: /experiments/f1-strategy-lab-notes/
focus: "Scenario design, deterministic grading, ablations, and stress tests for strategic reasoning"
excerpt: "Experiment notes from building and evaluating an RL-style environment for race-strategy reasoning."
project_url: "https://github.com/Herr-Professor/f1_rl"
---

The F1 strategy environment was built to test whether reasoning-heavy decisions can be evaluated with explicit rules instead of informal judgment.

## Experiment design

- Build scenario tasks from race state rather than free-form prompts
- Attach deterministic verifiers and tool-use rubrics to each scenario
- Compare outputs against baseline agents with stress tests and ablations

## Engineering lesson

Domain evaluation becomes much more useful when the environment makes failure legible. The important part is not just the reward signal, but the ability to see why a strategy was accepted or rejected.
