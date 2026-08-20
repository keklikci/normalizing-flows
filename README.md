# Normalizing flows

Beta-VAE experiments with normalizing flows for variational inference and
generative modeling.

## Project layout

- `src/normalizing_flows` contains reusable statistical utilities
- `scripts` contains the canonical Python experiment scripts
- `notebooks` contains archived notebook versions of the experiments
- `tests` contains fast numerical and repository structure checks
- `requirements-legacy.txt` preserves the original TensorFlow 2.5 environment

Archived notebooks are retained for provenance. Run the scripts when a
repeatable Python entry point is preferred.

## Setup

The supported package manager is [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

The default environment installs the reusable metrics package and development
tools. The legacy TensorFlow experiments require the pinned environment in
`requirements-legacy.txt` and may require an x86 Python environment because
TensorFlow 2.5 does not provide Apple Silicon wheels.

## Experiments

The scripts cover noisy moons, thoracic surgery preprocessing, and Beta-VAE
experiments using NICE, RealNVP, MAF, and IAF flows. Dataset paths are kept in
the experiment scripts and must be updated for a local dataset checkout.

```bash
uv run python scripts/beta-vae-iaf-noisy-moons.py
```

Experiments can be slow and may require TensorFlow, TensorFlow Probability,
datasets, and a graphical backend. They are not part of the default test run.

## Quality checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

Tests cover the reusable two-dimensional KS and energy statistics plus script
syntax and notebook-to-script parity. Full model training and notebook
execution remain manual validation steps.

## References

- [Variational inference with normalizing flows](https://arxiv.org/abs/1505.05770)
- [Normalizing flows an introduction and review](https://arxiv.org/abs/1908.09257)
- [TensorFlow Probability](https://www.tensorflow.org/probability)
- [Noisy moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)
- [Thoracic surgery dataset](https://www.kaggle.com/sid321axn/thoraric-surgery)
