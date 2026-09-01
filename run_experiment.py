import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import ExperimentConfig
from data import make_problem, set_run_seed
from model import NN
from ntk import ntk_eigensystem, project_onto_ntk_modes
from plotting import (
    plot_component_errors,
    plot_loss_vs_iteration,
    plot_loss_vs_time,
    plot_ntk_errors,
    set_plot_style,
)
from training import aggregate_runs, make_lr_grid, run_one_seed, tune_learning_rate


def main():
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = ExperimentConfig()
    run_seeds = list(range(cfg.n_runs))

    set_plot_style()
    set_run_seed(0)

    problem = make_problem(cfg, device, dtype)
    x = problem["x"]
    y = problem["y"]
    basis = problem["basis"]
    target_component_coef = problem["target_component_coef"]

    base_model = NN(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
        beta=cfg.beta,
        freeze_first_layer=cfg.freeze_first_layer,
    ).to(device)

    ntk_evals, ntk_evecs = ntk_eigensystem(base_model, x)
    target_ntk_coef = project_onto_ntk_modes(y, ntk_evecs).detach().cpu().numpy()

    gd_grid = make_lr_grid(cfg.gd_lr, orders=3, n_points=9)
    adam_grid = make_lr_grid(cfg.adam_lr, orders=3, n_points=9)

    best_gd_lr, gd_sweep_losses = tune_learning_rate(
        "GD",
        gd_grid,
        cfg,
        x,
        y,
        basis,
        target_component_coef,
        ntk_evecs,
        target_ntk_coef,
        seed=0,
        pilot_epochs=min(300, cfg.gd_epochs),
    )

    best_adam_lr, adam_sweep_losses = tune_learning_rate(
        "Adam",
        adam_grid,
        cfg,
        x,
        y,
        basis,
        target_component_coef,
        ntk_evecs,
        target_ntk_coef,
        seed=0,
        pilot_epochs=min(300, cfg.adam_epochs),
    )

    print(f"Best GD lr:   {best_gd_lr:.3e}")
    print(f"Best Adam lr: {best_adam_lr:.3e}")

    cfg.gd_lr = best_gd_lr
    cfg.adam_lr = best_adam_lr

    os.makedirs("Figures/NNTK", exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.semilogx(gd_grid, gd_sweep_losses, marker="o", label="GD")
    plt.semilogx(adam_grid, adam_sweep_losses, marker="o", label="Adam")
    plt.xlabel("Learning rate")
    plt.ylabel("Pilot final loss")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Figures/NNTK/lr_sweep.pdf", bbox_inches="tight")
    plt.show()

    run_histories = [
        run_one_seed(
            cfg,
            x,
            y,
            basis,
            target_component_coef,
            ntk_evecs,
            target_ntk_coef,
            seed,
        )
        for seed in run_seeds
    ]

    histories = aggregate_runs(run_histories)

    method_order = ["GD", "Adam", "GN", "Newton"]
    method_labels = {
        "GD": "gradient descent",
        "Adam": "Adam",
        "GN": "regularized Gauss-Newton method",
        "Newton": "regularized Newton method",
    }
    method_colors = {
        "GD": "C3",
        "Adam": "crimson",
        "Newton": "C0",
        "GN": "C2",
    }

    plot_loss_vs_iteration(
        run_histories,
        method_order,
        method_colors,
        method_labels=method_labels,
        save_path="Figures/NNTK/XX_Training_NewtonVSAdam_Iterations.pdf",
    )
    plt.show()

    plot_loss_vs_time(
        run_histories,
        method_order,
        method_colors,
        method_labels=method_labels,
        save_path="Figures/NNTK/XX_Training_NewtonVSAdam.pdf",
    )
    plt.show()

    plot_component_errors(
        histories,
        cfg,
        method_order,
        method_labels,
        method_colors,
        save_path="Figures/NNTK/XX_Componentwise_NewtonVSAdam_Frequency.pdf",
    )
    plt.show()

    plot_ntk_errors(
        histories,
        ntk_evals,
        target_ntk_coef,
        method_order,
        method_labels,
        method_colors,
        save_path="Figures/NNTK/XX_NTKMode_NewtonVSAdam.pdf",
    )
    plt.show()


if __name__ == "__main__":
    main()
