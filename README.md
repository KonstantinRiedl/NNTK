# NNTK — Newton Neural Tangent Kernel

Code accompanying the NeurIPS 2026 paper **"Convergence Analysis of Newton's Method for Neural
Networks in the Overparameterized Limit"** by Konstantin Riedl, Konstantinos Spiliopoulos, and
Justin Sirignano.

The paper analyzes the regularized Newton method for training overparameterized shallow neural
networks in the NTK regime and shows that, unlike gradient descent, its convergence rate is
uniform across the frequency spectrum of the target function. This repository contains the
numerical experiments behind all of the paper's figures:

- **Figures 1 and 2** — the NTK/NNTK eigenvalue spectrum and finite-width-to-infinite-width
  convergence sweep illustrating Remarks 5–6, Lemma 9, and Theorem 7 (`Notebook_NNTK.ipynb`).
- **Figures 3 and 5** — training gradient descent, Adam, the regularized Gauss–Newton method,
  and the regularized Newton method on a synthetic target with both low- and high-frequency
  components, comparing loss decay and per-frequency convergence (`run_experiment.py`,
  `Notebook_TrainingComparison.ipynb`).

## What it does

- Trains a single-hidden-layer network `f(x) = N^{-β} Σᵢ cᵢ σ(wᵢ·x + ηᵢ)`.
- Computes the empirical NTK and NNTK Gram matrices directly from Eq. (14)–(15) and sweeps
  network width and regularizer strength to show the NNTK's flatter eigenvalue spectrum,
  one-step convergence as the regularizer vanishes, and convergence of the finite-width NNTK and
  Newton updates to their infinite-width limits (Figures 1 and 2).
- Implements the Newton and Gauss–Newton updates with exact (hand-derived, not autograd-generic)
  Hessian-vector and Gauss–Newton-vector products, solved via conjugate gradient with adaptive,
  Levenberg–Marquardt-style damping (Figures 3 and 5).
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
| `Notebook_NNTK.ipynb` | NTK/NNTK eigenvalue spectrum and finite-width convergence sweep producing Figures 1 and 2 |
| `run_experiment.py` | End-to-end driver script producing Figures 3 and 5 |
| `Notebook_TrainingComparison.ipynb` | Interactive, cell-by-cell walkthrough of the same pipeline |

## Setup

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

### Figures 1 and 2 — NTK/NNTK spectrum and finite-width convergence

Open `Notebook_NNTK.ipynb`. Set `FIGURE = 1` (target `y = 2x + 0.4 sin(5πx)`) or `FIGURE = 2`
(target `y = 2x + 0.4 sin(20πx)`) in the "Training data" cell to select which figure's target
function to use, then run all cells. It sweeps network width `N` (up to 10⁶) and regularizer
`γ` (10⁻¹ to 10⁻¹⁰) with 100 trials per combination by default — matching the paper exactly,
but correspondingly slow to run end-to-end; reduce `Ns`, `gammas`, or `n_trials` in the
"Experimental setting" cell for a quicker, smaller-scale run. Figures are written to
`Figures/NNTK/`.

### Figures 3 and 5 — training comparison

```bash
python run_experiment.py
```

This tunes the gradient descent and Adam learning rates, runs all four optimizers across
`cfg.n_runs` random seeds, and writes the resulting figures to `Figures/NNTK/`.

Alternatively, open `Notebook_TrainingComparison.ipynb` for an interactive, annotated version of
the same pipeline.

All experiment parameters (network width, number of neurons, target-function frequencies,
number of runs, etc.) live in `config.py`'s `ExperimentConfig`. By default `n_runs = 10` for
fast iteration; the results reported in the paper are averaged over 100 seeds — set
`n_runs = 100` to reproduce those exactly (at correspondingly higher runtime).

## License

Released under the [MIT License](LICENSE).
