# NNTK — Newton Neural Tangent Kernel

Code accompanying the NeurIPS 2026 paper **"Convergence Analysis of Newton's Method for Neural
Networks in the Overparameterized Limit"** by Konstantin Riedl, Konstantinos Spiliopoulos, and
Justin Sirignano.

The paper analyzes the regularized Newton method for training overparameterized shallow neural
networks and shows that, unlike gradient descent, its convergence rate is uniform across the
frequency spectrum of the target function. This repository contains the numerical experiments
(Figure 3 and Figure 5 of the paper) that illustrate this by training a shallow `tanh` network on
a synthetic target with both low- and high-frequency components, comparing gradient descent,
Adam, the regularized Gauss–Newton method, and the regularized Newton method.

## What it does

- Trains a single-hidden-layer network `f(x) = N^{-β} Σᵢ cᵢ σ(wᵢ·x + ηᵢ)` on a 1D regression
  target `y = 4x + sin(2πx) + 0.8(sin(8πx) + sin(10πx) + sin(16πx))`.
- Implements the Newton and Gauss–Newton updates with exact (hand-derived, not autograd-generic)
  Hessian-vector and Gauss–Newton-vector products, solved via conjugate gradient with adaptive,
  Levenberg–Marquardt-style damping.
- Tracks, per iteration and per wall-clock second: the training loss, the error in each known
  target-frequency coefficient, and the error in each empirical-NTK eigenmode coefficient.
- Produces the loss-vs-iteration, loss-vs-time, per-frequency convergence, and NTK-mode
  convergence figures, plus optional training-evolution animations.

## Repository layout

| File | Contents |
|---|---|
| `config.py` | Experiment configuration: network width, target function, optimizer hyperparameters, run count |
| `data.py` | Synthetic target construction and seed control |
| `model.py` | The network, loss, and parameter flatten/unflatten helpers |
| `linear_algebra.py` | Exact HVP/GNVP, conjugate gradient, preconditioners |
| `ntk.py` | Empirical NTK Gram matrix, eigensystem, and mode projections |
| `optimizers.py` | Gradient descent, Adam, and damped Newton/Gauss–Newton optimizers |
| `training.py` | Training loop, learning-rate tuning, multi-seed aggregation |
| `plotting.py` | Figure generation (loss curves, per-component/NTK-mode error, animations) |
| `run_experiment.py` | End-to-end driver script producing all figures |
| `Notebook.ipynb` | Interactive, cell-by-cell walkthrough of the same pipeline |

## Setup

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python run_experiment.py
```

This tunes the gradient descent and Adam learning rates, runs all four optimizers across
`cfg.n_runs` random seeds, and writes the resulting figures to `Figures/NNTK/`.

Alternatively, open `Notebook.ipynb` for an interactive, annotated version of the same pipeline
(update the `PROJECT_DIR` path in the first cell to point at this repository before running).

All experiment parameters (network width, number of neurons, target-function frequencies,
number of runs, etc.) live in `config.py`'s `ExperimentConfig`. By default `n_runs = 10` for
fast iteration; the results reported in the paper are averaged over 100 seeds — set
`n_runs = 100` to reproduce those exactly (at correspondingly higher runtime).

## License

Released under the [MIT License](LICENSE).
