---
layout: single
title: "Neon, So Far: Building a Small Solver-Plus-ML Research Platform"
excerpt: "A progress report on Neon: a small scalar Helmholtz solver and ML research platform for validation, surrogate modeling, uncertainty, and benchmark-scale numerical experiments in simplified nanophotonics."
permalink: /blog/neon-so-far/
date: 2026-03-22 22:30:00 +0100
author_profile: true
show_date: false
read_time: true
share: false
related: false
comments: false
tags:
  - neon
  - computational photonics
  - surrogate modeling
  - numerical methods
---

<style>
  .neon-post-figure {
    margin: 2rem 0;
  }

  .neon-post-figure img {
    width: 100%;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
  }

  .neon-post-figure figcaption {
    margin-top: 0.75rem;
    color: #4b5563;
    font-size: 0.95rem;
    line-height: 1.6;
  }
</style>

I started Neon because I wanted a research codebase I could understand end to end.

There are plenty of powerful simulation environments in photonics, but for the questions I cared about most, I wanted something smaller and more inspectable: a numerical solver I could validate directly, extend carefully, and then use as a base for machine-learning experiments without losing track of what the model was actually learning. Neon started there, as a minimal scalar Helmholtz code. It has gradually turned into something broader: a private solver-plus-ML platform for careful, benchmark-scale numerical modeling experiments in simplified nanophotonics settings.

This post is now a completed-work summary. The codebase is stable, the experiments are done, and a paper is in preparation. The direct PDE solver is still the reference path. The learned models are still narrow slab-family surrogates. Some milestones have worked well, some have produced mixed results, and some of the most useful results so far are the ones that forced the project to get more honest.

Because the repository is still private, this write-up focuses on the benchmark design, validation milestones, and what the current results actually justify.

## Why I Started Neon

I wanted to build a serious numerical modeling toolchain from the ground up, but I also wanted it to stay small enough that every piece remained legible. That meant starting with a reduced model instead of pretending to build a full production-grade electromagnetic platform on day one.

The motivation was partly numerical and partly scientific-ML-driven. On the numerical side, I wanted a solver I could validate with simple cases, inspect with tests, and extend in a controlled way. On the ML side, I wanted to look at questions that are easy to overstate in photonics-flavored surrogate modeling: when a surrogate helps, when it does not, whether physics-informed losses really matter, what uncertainty signals are worth trusting, and whether better held-out metrics actually improve downstream screening or inverse-design workflows.

That combination is what Neon has become: a small research platform built around a scalar benchmark where the direct solver remains the oracle.

## What Neon Is Today

Today, Neon is a C++20 scalar frequency-domain Helmholtz code with a Python ML layer built on top of it.

At the solver level, the current codebase supports:

- a 2D scalar frequency-domain Helmholtz solve on a structured Cartesian grid
- heterogeneous complex relative permittivity
- two source models: `plane_wave_line` and `point`
- two absorber modes: a legacy sponge and a scalar stretched-coordinate PML
- Eigen `SparseLU` as the default linear solve path, with optional `BiCGSTAB` and fallback
- CSV output plus optional VTK export

On top of that, Neon now includes:

- slab-family dataset generation driven by the direct solver
- a baseline MLP surrogate
- a reduced hybrid surrogate that predicts scalar targets plus a cropped centerline field
- an enhanced hybrid variant with residual, boundary-aware, and source-aware penalties
- benchmark-facing slab normalization for the current scalar normal-incidence slab family
- deep-ensemble uncertainty
- solver-as-oracle active-learning pilots
- dense screening inverse-design workflows with mandatory direct solver reevaluation

The important qualifier is that all of this remains narrow in scope. Neon is not a full-vector Maxwell platform, and I do not think it should be described that way.

## The Solver Core

The core numerical model is the scalar out-of-plane Helmholtz equation on a regular grid. That is a deliberately reduced model, but it is still enough to make the code scientifically useful as a controlled environment for validation, surrogate modeling, and workflow benchmarking.

The current repository still keeps the direct solver at the center of the project. Every learned model is trained from direct Neon outputs. Every inverse-design-style result still requires direct reevaluation. That design choice matters because it keeps the project honest: the learned model is never allowed to quietly become the ground truth.

<figure class="neon-post-figure">
  <img src="/images/blog/neon/direct_solver_field_plots.png" alt="Field intensity and real field plots from Neon’s homogeneous benchmark case.">
  <figcaption>Direct solver output from Neon’s homogeneous benchmark case.</figcaption>
</figure>

The codebase is also in a decent engineering state now. The current test suite passes `12/12`, and the solver/validation stack is backed by C++ tests plus Python smoke tests for the ML workflows.

## What Changed With the PML / Validation Milestone

One of the first meaningful upgrades was boundary treatment. Neon kept its original sponge absorber as a legacy path, but added a scalar stretched-coordinate PML for the current Helmholtz formulation.

On the shipped homogeneous benchmark, that change made a real difference. The left reflection ratio drops from `0.1772` with the sponge path to `0.0109` with the scalar PML, and the right-side backscatter ratio drops from `0.2765` to `0.0266`. In rough terms, that is about a `16.3x` reduction in the left reflection metric and a `10.4x` reduction in the right backscatter metric. The homogeneous thickness sweep is also well behaved: increasing the PML from `10` to `18` cells reduces the left reflection ratio from `0.0570` to `0.0069` and the right backscatter ratio from `0.1113` to `0.0159`.

<figure class="neon-post-figure">
  <img src="/images/blog/neon/pml_vs_sponge_validation.png" alt="Validation plots comparing scalar PML and sponge absorbers, plus PML thickness sweeps.">
  <figcaption>Scalar PML validation on the current benchmark family.</figcaption>
</figure>

That does not prove broad absorber quality for every geometry, but it does show that the current scalar PML path is materially cleaner than the legacy sponge on the benchmark it was built for.

The validation story also became more interesting once I started comparing the slab benchmark against external references. The transfer-matrix-method check was the most useful example. The first version failed badly, but the failure turned out to be more about what Neon was outputting than about the direct solve itself.

The legacy slab transmission and reflection quantities are monitor-based proxies. On the current slab sweep, comparing those directly to analytic TMM gives mean absolute errors of `0.2279` for transmission and `0.0662` for reflection, with a strongly positive signed bias of `+0.2279` and `+0.0594`. That looks much more like a normalization or proxy offset than random solver scatter. After adding a benchmark-facing scalar normalization for the present lossless normal-incidence slab family, the same comparison improves to mean absolute errors of `0.0489` and `0.0489`, with max absolute errors of about `0.1000` for both terms. That benchmark-facing comparison now passes the current threshold for the shipped slab sweep.

<figure class="neon-post-figure">
  <img src="/images/blog/neon/tmm_comparison.png" alt="Scatter plots comparing Neon slab coefficients to analytic transfer-matrix-method values.">
  <figcaption>Legacy proxy outputs versus benchmark-facing slab coefficients in the TMM check.</figcaption>
</figure>

That is a good result, but it is also a narrow one. I would describe it as improved benchmark-facing scalar slab normalization, not as a general rigorous power-normalization framework.

Not every external check came out well. The corrected `Ez`-to-`Ez` Ceviche comparison still fails its threshold: the current magnitude correlation is `0.8737`, and the normalized field MAE is `1.1199`. I keep that result in the repo because it is useful caution, not because it supports a strong claim.

## The First Surrogate Modeling Milestone

Once the solver and output pipeline were solid enough, I added a solver-driven slab dataset and a small baseline surrogate.

The shipped slab dataset currently contains `18` designs and `108` samples, with deterministic `design_id` splits of `12/3/3` designs or `72/18/18` samples across train, validation, and test. The feature space is simple: slab thickness, slab relative permittivity real part, and wavelength.

The first baseline model was a PyTorch MLP trained on the legacy slab targets:

- `normalized_transmission`
- `normalized_reflection`
- `normalized_peak_intensity`

On that legacy target set, the current baseline held-out mean MAE is `0.1548`, and the current OOD probe mean MAE is `0.2627`. The separate OOD degradation check labels that behavior as mild degradation rather than catastrophic collapse, but it is not a robustness story: the mean OOD absolute errors in that check are `0.2749` for transmission, `0.1608` for reflection, and `0.1610` for peak intensity.

That baseline milestone was important because it established the first solver-trained surrogate workflow end to end. But it also made the limitations visible quickly: the model is narrow, the targets started out as proxy quantities, and OOD behavior remains constrained.

## The Hybrid AI + PDE Milestone

The next question was whether a purely tabular surrogate was leaving useful structure on the table. That led to Neon’s reduced hybrid workflow.

The current hybrid model predicts the same scalar slab targets as the baseline, but it also predicts a cropped complex centerline field. During training, Neon can then penalize not just scalar errors but also a reduced discrete Helmholtz residual along that centerline.

That gives three model families in the current comparison ladder:

- **Model A:** baseline data-driven MLP
- **Model B:** scalar targets + centerline field + reduced residual
- **Model C:** Model B plus a boundary-aware loss and a source-aware loss

The extra physics terms in Model C are still benchmark-specific and reduced. The boundary-aware term penalizes backward-wave content in a transmitted-side background window near the absorber entrance. The source-aware term penalizes mismatch to the expected forward-wave structure in the source-side background region. This is not a full-domain PINN, not a neural operator, and not a Maxwell-level physics model. But it is physics-derived in a real, limited sense.

<figure class="neon-post-figure">
  <img src="/images/blog/neon/hybrid_centerline_example.png" alt="Held-out centerline comparison between reference and predicted real, imaginary, and intensity values.">
  <figcaption>One held-out centerline field example from the reduced hybrid workflow.</figcaption>
</figure>

## What the Current Results Actually Say

This is where the project stopped being just a feature stack and started becoming a benchmark platform.

On the original legacy target set, the full-data A/B/C comparison is mixed but informative. The current test mean MAEs are:

- Model A: `0.1548`
- Model B: `0.1616`
- Model C: `0.1522`

On the shipped OOD probe, the corresponding mean MAEs are:

- Model A: `0.2627`
- Model B: `0.2802`
- Model C: `0.2653`

So Model C improves clearly over Model B in both held-out and OOD error on this benchmark, but it still does not beat the simple baseline on OOD.

The ablation result is even more useful than the headline comparison. Removing the source-aware term degrades Model C from `0.1522` to `0.1625` on held-out test MAE and from `0.2653` to `0.2704` on the OOD probe. Removing the boundary-aware term barely changes those numbers: `0.1522` to `0.1521` on test and `0.2653` to `0.2651` on OOD. On this benchmark, the source-aware term is doing real work; the boundary-aware term is much weaker.

The reduced-data study tells a similarly cautious story. Model C beats Model B on held-out test error in `4/6` train-size settings and on OOD error in `5/6`, and it beats Model A on held-out test error in `5/6`. But it beats Model A on OOD in `0/6`. That is the kind of mixed result I want the repo to preserve rather than smooth over.

<figure class="neon-post-figure">
  <img src="/images/blog/neon/model_abc_test_ood.png" alt="Bar charts comparing Models A, B, and C on held-out and out-of-domain mean absolute error.">
  <figcaption>A/B/C comparison on the shipped slab benchmark, including the boundary and source ablations.</figcaption>
</figure>

Later work in the repository moved to the benchmark-facing slab targets for uncertainty and active-learning studies, so those numbers are not directly comparable to the earlier legacy-target results. On that newer target set, the best current single benchmark-facing Model C run reaches a held-out mean MAE of `0.1145`. The shipped 5-member ensemble results are:

- Model A ensemble: `0.1123` test mean MAE and `0.1172` OOD mean MAE
- Model C ensemble: `0.1155` test mean MAE and `0.1293` OOD mean MAE

Those ensembles are useful because they expose predictive spread, but the uncertainty signal is still mixed. For Model A, the overall mean predictive standard deviation rises from `0.0086` on test to `0.0212` on the OOD probe. For Model C, it rises from `0.0129` to `0.0258`. That is directionally sensible. But the error-correlation signal is not strong enough to call the uncertainty calibrated: the overall uncertainty-error correlation is `-0.0087` on test and `-0.0920` on the OOD probe for Model A, and `-0.2913` on test versus `0.2458` on the OOD probe for Model C.

## What Is Still Not Solved

The biggest unresolved issues are exactly the ones I would expect in a project at this stage.

First, the scalar benchmark itself is still narrow. The benchmark-facing slab normalization fixed an important comparison problem, but it only fixes it for the current lossless normal-incidence scalar slab family. It is not a general flux-normalization framework for arbitrary geometries.

Second, the external numerical comparison is still incomplete. The TMM benchmark is now more defensible, but the Ceviche cross-check remains below threshold, so it stays in the project as a warning sign rather than a success claim.

Third, the active-learning story is still a pilot, not a full paper-scale experiment. The shipped passive-vs-active comparison uses one selection seed, `4` initial labeled train designs, one acquisition round to `6` labeled designs, and `3`-member ensembles inside the loop. At that reduced budget, uncertainty-guided acquisition improves held-out test MAE for all three model families, but it hurts the shipped OOD probe for all three:

- Model A: `0.1273` active vs `0.1330` passive on test, but `0.1320` active vs `0.1249` passive on OOD
- Model B: `0.1300` active vs `0.1374` passive on test, but `0.1440` active vs `0.1293` passive on OOD
- Model C: `0.1308` active vs `0.1388` passive on test, but `0.1362` active vs `0.1300` passive on OOD

The current physics-interaction summary labels the active-learning effect as neutral: Model C’s test-gain advantage over Model B is only `+0.0003`.

<figure class="neon-post-figure">
  <img src="/images/blog/neon/passive_vs_active_test_ood.png" alt="Line plots comparing passive and uncertainty-guided acquisition on held-out and out-of-domain error.">
  <figcaption>Reduced-budget passive versus uncertainty-guided acquisition on the current slab benchmark.</figcaption>
</figure>

Fourth, the inverse-design path is still screening, not mature optimization. In the current A/B/C comparison, the best directly reevaluated candidate actually comes from Model B, not Model C: the direct objective values are `0.4732` for Model A, `0.1653` for Model B, and `0.4238` for Model C. The uncertainty-aware ranking workflow helps a little inside the benchmark-facing Model C branch, nudging the best directly reevaluated candidate from `0.5462` for the single-model ranking to `0.5364` for the conservative ensemble ranking, but that is still a modest improvement inside a workflow where direct solver reevaluation remains mandatory.

## Why I Think This Direction Matters

What keeps this project interesting to me is not that every new layer wins cleanly. It is that the codebase now has enough structure to show where the wins are real, where the losses are real, and where the conclusions need to stay narrow.

The PML milestone improved the direct solver in a way that shows up clearly in validation. The benchmark-facing normalization improved a comparison that had been misleading. The enhanced hybrid model does contain a genuinely useful source-aware term on this slab benchmark. The uncertainty layer is useful enough to support experiments, but not trustworthy enough to overclaim. The active-learning workflow works as code and as a reduced pilot, but it does not yet answer the broader data-efficiency question. I find that much more valuable than a project that only reports the parts that happened to look good.

## What Comes Next: The Paper

A paper is now in preparation:

*Toward Trustworthy Surrogate Models for Electromagnetic Simulation: A Systematic Evaluation of Physics-Informed Training, Uncertainty, and Active Learning on a Controlled Benchmark*

The project is organized around one question: does this training pipeline produce a model you can actually trust? The paper is the formal answer to that question across the experiments described in this post. It reports the results honestly, including the places where the answer is only partial or mixed.

## Neon’s Model

The codebase is now live at [https://github.com/Herr-Professor/Neon](https://github.com/Herr-Professor/Neon), and the trained model is live on HuggingFace at [https://huggingface.co/Herrprofessor/Neon](https://huggingface.co/Herrprofessor/Neon).

It is a fast neural network predictor for dielectric slab optical response. Give it slab thickness, relative permittivity real part, and wavelength, and it returns transmission, reflection, and peak intensity in milliseconds.

This is not a general photonics model. It is trained on FDFD simulation data from Neon’s scalar slab benchmark. Researchers who work with this class of problem can use it as a fast screener, a baseline comparison, or a reproducibility tool for the paper results.

```python
from neon import Neon
model = Neon.from_pretrained()
result = model.predict(thickness=0.30, epsilon_real=2.25, wavelength=0.80)
print(result)
```
