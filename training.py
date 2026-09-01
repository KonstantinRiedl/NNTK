import copy
import time

import numpy as np
import torch

from data import set_run_seed
from model import compute_loss, flatten
from ntk import project_onto_basis, project_onto_ntk_modes
from optimizers import AdamOptimizer, GradientDescentOptimizer, make_optimizer


def make_lr_grid(base_lr, orders=3, n_points=9):
    """Geometric learning-rate grid centered (in log space) on ``base_lr``.

    Spans ``base_lr * 10**-orders`` to ``base_lr * 10**orders`` with
    ``n_points`` points, endpoints included.
    """
    base_lr = float(base_lr)
    if base_lr <= 0:
        raise ValueError("base_lr must be positive")
    if n_points < 1:
        raise ValueError("n_points must be >= 1")
    exponents = np.linspace(-orders, orders, n_points)
    return base_lr * 10.0 ** exponents


def train_model(
    model,
    optimizer,
    epochs,
    x,
    y,
    component_basis,
    target_component_coef,
    ntk_evecs,
    target_ntk_coef,
    iterate_interval=None,
    iterate_time_interval=None,
):
    if iterate_interval is not None and iterate_time_interval is not None:
        raise ValueError("choose either iterate_interval or iterate_time_interval, not both")
    if iterate_interval is not None and iterate_interval <= 0:
        raise ValueError("iterate_interval must be positive")
    if iterate_time_interval is not None and iterate_time_interval <= 0:
        raise ValueError("iterate_time_interval must be positive")

    losses = []
    component_coefs = []
    component_rel_errs = []
    component_abs_errs = []
    ntk_coefs = []
    ntk_rel_errs = []
    ntk_abs_errs = []
    elapsed = []
    iterate_steps = []
    iterate_times = []
    iterates = []
    last_iterate_time = 0.0
    compute_time = 0.0

    if x.device.type == "cuda":
        torch.cuda.synchronize()

    if iterate_interval is not None or iterate_time_interval is not None:
        iterates.append(flatten(model.trainable_parameters()).detach().cpu().numpy())
        iterate_steps.append(0)
        iterate_times.append(0.0)

    for iteration in range(1, epochs + 1):
        def closure():
            optimizer.zero_grad()
            return compute_loss(model(x), y)

        if x.device.type == "cuda":
            torch.cuda.synchronize()
        step_start = time.perf_counter()

        loss, step = optimizer.step(closure, x=x, y=y, model=model)

        if isinstance(optimizer, (AdamOptimizer, GradientDescentOptimizer)):
            optimizer.apply_step(step)

        if x.device.type == "cuda":
            torch.cuda.synchronize()
        compute_time += time.perf_counter() - step_start

        with torch.no_grad():
            fx = model(x)
            current_loss = compute_loss(fx, y).item()
            comp_coef = project_onto_basis(fx, component_basis).detach().cpu().numpy()
            ntk_coef = project_onto_ntk_modes(fx, ntk_evecs).detach().cpu().numpy()

        comp_rel_err = np.abs(comp_coef - target_component_coef) / (np.abs(target_component_coef) + 1e-12)
        ntk_rel_err = np.abs(ntk_coef - target_ntk_coef) / (np.abs(target_ntk_coef) + 1e-12)
        comp_abs_err = np.abs(comp_coef - target_component_coef)
        ntk_abs_err = np.abs(ntk_coef - target_ntk_coef)

        losses.append(current_loss)
        component_coefs.append(comp_coef)
        component_rel_errs.append(comp_rel_err)
        component_abs_errs.append(comp_abs_err)
        ntk_coefs.append(ntk_coef)
        ntk_rel_errs.append(ntk_rel_err)
        ntk_abs_errs.append(ntk_abs_err)
        elapsed_time = compute_time
        elapsed.append(elapsed_time)

        save_by_iteration = iterate_interval is not None and iteration % iterate_interval == 0
        save_by_time = (
            iterate_time_interval is not None
            and elapsed_time - last_iterate_time >= iterate_time_interval
        )
        if save_by_iteration or save_by_time:
            iterates.append(flatten(model.trainable_parameters()).detach().cpu().numpy())
            iterate_steps.append(iteration)
            iterate_times.append(elapsed_time)
            last_iterate_time = elapsed_time

    if (
        (iterate_interval is not None or iterate_time_interval is not None)
        and epochs
        and (not iterate_steps or iterate_steps[-1] != epochs)
    ):
        iterates.append(flatten(model.trainable_parameters()).detach().cpu().numpy())
        iterate_steps.append(epochs)
        iterate_times.append(elapsed[-1])

    return {
        "loss": np.asarray(losses),
        "component_coef": np.asarray(component_coefs),
        "component_rel_err": np.asarray(component_rel_errs),
        "component_abs_err": np.asarray(component_abs_errs),
        "ntk_coef": np.asarray(ntk_coefs),
        "ntk_rel_err": np.asarray(ntk_rel_errs),
        "ntk_abs_err": np.asarray(ntk_abs_errs),
        "time": np.asarray(elapsed),
        "iterate_steps": np.asarray(iterate_steps, dtype=int),
        "iterate_times": np.asarray(iterate_times),
        "iterates": (
            np.stack(iterates)
            if iterates
            else np.empty((0, sum(p.numel() for p in model.trainable_parameters())))
        ),
    }


def run_one_seed(cfg, x, y, component_basis, target_component_coef, ntk_evecs, target_ntk_coef, seed):
    set_run_seed(seed)

    from model import NN

    base_model = NN(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
        beta=cfg.beta,
        freeze_first_layer=cfg.freeze_first_layer,
    ).to(x.device)

    models = {
        "GD": copy.deepcopy(base_model),
        "Adam": copy.deepcopy(base_model),
        "Newton": copy.deepcopy(base_model),
        "GN": copy.deepcopy(base_model),
    }

    opts = {name: make_optimizer(name, models[name], cfg) for name in models}

    histories = {}
    histories["GD"] = train_model(
        models["GD"], opts["GD"], cfg.gd_epochs, x, y, component_basis, target_component_coef, ntk_evecs, target_ntk_coef,
        iterate_interval=cfg.iterate_interval,
        iterate_time_interval=cfg.iterate_time_interval,
    )
    histories["Adam"] = train_model(
        models["Adam"], opts["Adam"], cfg.adam_epochs, x, y, component_basis, target_component_coef, ntk_evecs, target_ntk_coef,
        iterate_interval=cfg.iterate_interval,
        iterate_time_interval=cfg.iterate_time_interval,
    )
    histories["Newton"] = train_model(
        models["Newton"], opts["Newton"], cfg.epochs, x, y, component_basis, target_component_coef, ntk_evecs, target_ntk_coef,
        iterate_interval=cfg.iterate_interval,
        iterate_time_interval=cfg.iterate_time_interval,
    )
    histories["GN"] = train_model(
        models["GN"], opts["GN"], cfg.epochs, x, y, component_basis, target_component_coef, ntk_evecs, target_ntk_coef,
        iterate_interval=cfg.iterate_interval,
        iterate_time_interval=cfg.iterate_time_interval,
    )

    return histories


def aggregate_runs(run_histories, lower_q=5, upper_q=95):
    aggregated = {}

    for method in ["GD", "Adam", "Newton", "GN"]:
        loss_runs = np.stack([r[method]["loss"] for r in run_histories], axis=0)
        comp_coef_runs = np.stack([r[method]["component_coef"] for r in run_histories], axis=0)
        comp_err_runs = np.stack([r[method]["component_rel_err"] for r in run_histories], axis=0)
        comp_abs_err_runs = np.stack([r[method]["component_abs_err"] for r in run_histories], axis=0)
        ntk_coef_runs = np.stack([r[method]["ntk_coef"] for r in run_histories], axis=0)
        ntk_err_runs = np.stack([r[method]["ntk_rel_err"] for r in run_histories], axis=0)
        ntk_abs_err_runs = np.stack([r[method]["ntk_abs_err"] for r in run_histories], axis=0)
        time_runs = np.stack([r[method]["time"] for r in run_histories], axis=0)

        aggregated[method] = {
            "time_mean": time_runs.mean(axis=0),
            "loss_mean": loss_runs.mean(axis=0),
            "loss_low": np.percentile(loss_runs, lower_q, axis=0),
            "loss_high": np.percentile(loss_runs, upper_q, axis=0),
            "component_coef_mean": comp_coef_runs.mean(axis=0),
            "component_coef_low": np.percentile(comp_coef_runs, lower_q, axis=0),
            "component_coef_high": np.percentile(comp_coef_runs, upper_q, axis=0),
            "component_rel_err_mean": comp_err_runs.mean(axis=0),
            "component_rel_err_low": np.percentile(comp_err_runs, lower_q, axis=0),
            "component_rel_err_high": np.percentile(comp_err_runs, upper_q, axis=0),
            "component_abs_err_mean": comp_abs_err_runs.mean(axis=0),
            "component_abs_err_low": np.percentile(comp_abs_err_runs, lower_q, axis=0),
            "component_abs_err_high": np.percentile(comp_abs_err_runs, upper_q, axis=0),
            "ntk_coef_mean": ntk_coef_runs.mean(axis=0),
            "ntk_coef_low": np.percentile(ntk_coef_runs, lower_q, axis=0),
            "ntk_coef_high": np.percentile(ntk_coef_runs, upper_q, axis=0),
            "ntk_rel_err_mean": ntk_err_runs.mean(axis=0),
            "ntk_rel_err_low": np.percentile(ntk_err_runs, lower_q, axis=0),
            "ntk_rel_err_high": np.percentile(ntk_err_runs, upper_q, axis=0),
            "ntk_abs_err_mean": ntk_abs_err_runs.mean(axis=0),
            "ntk_abs_err_low": np.percentile(ntk_abs_err_runs, lower_q, axis=0),
            "ntk_abs_err_high": np.percentile(ntk_abs_err_runs, upper_q, axis=0),
        }

    return aggregated


def tune_learning_rate(
    method,
    lr_grid,
    cfg,
    x,
    y,
    component_basis,
    target_component_coef,
    ntk_evecs,
    target_ntk_coef,
    seed=0,
    pilot_epochs=300,
):
    """
    Tune a learning rate using short pilot runs averaged across ``cfg.n_runs``
    consecutive seeds, beginning with ``seed``. Non-finite losses are treated
    as divergence and penalized.
    """
    from model import NN

    scores = []

    for lr in lr_grid:
        trial_cfg = copy.deepcopy(cfg)

        if method == "GD":
            trial_cfg.gd_lr = float(lr)
        elif method == "Adam":
            trial_cfg.adam_lr = float(lr)
        else:
            raise ValueError("tuning only implemented for GD and Adam")

        seed_scores = []
        for run_seed in range(seed, seed + cfg.n_runs):
            set_run_seed(run_seed)
            model = NN(
                input_dim=cfg.input_dim,
                hidden_dim=cfg.hidden_dim,
                output_dim=cfg.output_dim,
                beta=cfg.beta,
                freeze_first_layer=cfg.freeze_first_layer,
            ).to(x.device)
            opt = make_optimizer(method, model, trial_cfg)

            hist = train_model(
                model,
                opt,
                pilot_epochs,
                x,
                y,
                component_basis,
                target_component_coef,
                ntk_evecs,
                target_ntk_coef,
            )

            traj = np.asarray(hist["loss"], dtype=float)
            finite = traj[np.isfinite(traj)]
            seed_scores.append(np.inf if finite.size == 0 else np.min(finite))

        scores.append(np.mean(seed_scores))

    scores = np.asarray(scores, dtype=float)
    best_idx = int(np.argmin(scores))
    return float(lr_grid[best_idx]), scores
