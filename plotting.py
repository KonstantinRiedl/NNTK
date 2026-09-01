import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def set_plot_style():
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
        }
    )

    sns.set_theme(
        style="ticks",
        rc={
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        },
    )


def smooth_positive_curve(y, window=1000):
    y = np.asarray(y, dtype=float)
    finite = y[np.isfinite(y)]
    fill = finite.max() if finite.size else 1.0
    y = np.where(np.isfinite(y), y, fill)
    y = np.clip(y, 1e-16, None)

    if window is None or window < 3:
        return y

    window = int(window)
    if window % 2 == 0:
        window += 1

    pad = window // 2
    kernel = np.ones(window, dtype=float) / window

    logy = np.log10(y)
    logy_pad = np.pad(logy, (pad, pad), mode="edge")
    logy_smooth = np.convolve(logy_pad, kernel, mode="valid")

    return 10 ** logy_smooth


def smoothing_window_from_time(time, fraction):
    """Convert a fraction of an elapsed-time trajectory to a sample window."""
    time = np.asarray(time, dtype=float)
    if len(time) < 3:
        return None

    time_steps = np.diff(time)
    positive_steps = time_steps[time_steps > 0]
    if positive_steps.size == 0:
        return None

    duration = time[-1] - time[0]
    return max(3, int(round(fraction * duration / np.median(positive_steps))))


def best_seen_loss(loss):
    """Return the lowest finite loss attained at or before each step."""
    loss = np.asarray(loss, dtype=float)
    return np.minimum.accumulate(np.where(np.isfinite(loss), loss, np.inf))


def build_loss_dataframe_iterations(
    run_histories,
    method_order,
    clip_min=1e-16,
    best_so_far=True,
    time_horizon=None,
):
    rows = []
    for method in method_order:
        if time_horizon is None:
            max_iterations = None
        else:
            max_iterations = min(
                np.count_nonzero(np.asarray(run[method]["time"], dtype=float) <= time_horizon)
                for run in run_histories
            )

        for run_id, run in enumerate(run_histories):
            loss = np.asarray(run[method]["loss"], dtype=float)
            if best_so_far:
                loss = best_seen_loss(loss)
            if max_iterations is not None:
                loss = loss[:max_iterations]
            for iteration, li in enumerate(loss):
                rows.append({
                    "iteration": iteration,
                    "loss": max(li, clip_min),
                    "method": method,
                    "run": run_id,
                })
    return pd.DataFrame(rows)


def build_loss_dataframe_time(
    run_histories,
    method_order,
    n_grid=120,
    clip_min=1e-16,
    best_so_far=True,
    time_horizon=None,
):
    rows = []
    for method in method_order:
        min_time = max(np.min(np.asarray(run[method]["time"], dtype=float)) for run in run_histories)
        max_time = min(np.max(np.asarray(run[method]["time"], dtype=float)) for run in run_histories)
        if time_horizon is not None:
            max_time = min(max_time, time_horizon)
        if min_time > max_time:
            continue
        grid = np.linspace(min_time, max_time, n_grid)

        for run_id, run in enumerate(run_histories):
            t = np.asarray(run[method]["time"], dtype=float)
            loss = np.asarray(run[method]["loss"], dtype=float)
            order = np.argsort(t)
            t = t[order]
            loss = loss[order]
            if best_so_far:
                loss = best_seen_loss(loss)
            loss_interp = np.interp(grid, t, loss)

            for ti, li in zip(grid, loss_interp):
                rows.append({
                    "time": ti,
                    "loss": max(li, clip_min),
                    "method": method,
                    "run": run_id,
                })
    return pd.DataFrame(rows)


def plot_loss_vs_iteration(
    run_histories,
    method_order,
    method_colors,
    method_labels=None,
    title=None,
    save_path=None,
    production=0,
    time_horizon=20.0,
):
    """Plot loss trajectories, optionally truncated per run at ``time_horizon``."""
    df_loss = build_loss_dataframe_iterations(
        run_histories,
        method_order,
        best_so_far=True,
        time_horizon=time_horizon if production else None,
    )

    plt.figure(figsize=(8, 4))
    ax = sns.lineplot(
        data=df_loss,
        x="iteration",
        y="loss",
        hue="method",
        hue_order=method_order,
        palette=method_colors,
        estimator="mean",
        errorbar=("pi", 80),
        err_style="band",
        err_kws={"alpha": 0.15},
        linestyle="-",
        marker=None,
    )

    ax.set_yscale("log")
    ax.set_xlabel("iteration", fontsize=12)
    ax.set_ylabel(r"empirical training loss $L^{N}(\theta)$", fontsize=12)
    if title:
        ax.set_title(title)

    ax.grid(True, which="major", alpha=0.4, linestyle="-")
    ax.grid(True, which="minor", alpha=0.2, linestyle=":")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    ax.tick_params(axis="both", which="major", direction="out", length=5, width=0.8, colors="black", top=False, right=False)
    ax.tick_params(axis="both", which="minor", direction="out", length=3, width=0.6, colors="black", top=False, right=False)
    ax.set_axisbelow(True)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=method_colors[name],
            linestyle="-",
            linewidth=2.5,
            label=method_labels.get(name, name) if method_labels else name,
        )
        for name in method_order
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.9, fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return ax


def plot_loss_vs_time(
    run_histories,
    method_order,
    method_colors,
    method_labels=None,
    title=None,
    save_path=None,
    production=0,
    time_horizon=20.0,
):
    """Plot loss against time, optionally restricted to ``[0, time_horizon]``."""
    df_loss = build_loss_dataframe_time(
        run_histories,
        method_order,
        n_grid=120,
        best_so_far=True,
        time_horizon=time_horizon if production else None,
    )

    plt.figure(figsize=(8, 4))
    ax = sns.lineplot(
        data=df_loss,
        x="time",
        y="loss",
        hue="method",
        hue_order=method_order,
        palette=method_colors,
        estimator="mean",
        errorbar=("pi", 80),
        err_style="band",
        err_kws={"alpha": 0.15},
        linestyle="-",
        marker=None,
        sort=False,
    )

    ax.set_yscale("log")
    ax.set_xlabel("compute time (seconds)", fontsize=12)
    ax.set_ylabel(r"empirical training loss $L^{N}(\theta)$", fontsize=12)
    if production:
        x_padding = 0.05 * time_horizon
        ax.set_xlim(-x_padding, time_horizon + x_padding)
    if title:
        ax.set_title(title)

    ax.grid(True, which="major", alpha=0.4, linestyle="-")
    ax.grid(True, which="minor", alpha=0.2, linestyle=":")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    ax.tick_params(axis="both", which="major", direction="out", length=5, width=0.8, colors="black", top=False, right=False)
    ax.tick_params(axis="both", which="minor", direction="out", length=3, width=0.6, colors="black", top=False, right=False)
    ax.set_axisbelow(True)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=method_colors[name],
            linestyle="-",
            linewidth=2.5,
            label=method_labels.get(name, name) if method_labels else name,
        )
        for name in method_order
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.9, fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return ax


def plot_component_errors(
    histories,
    cfg,
    method_order,
    method_labels,
    method_colors,
    save_path=None,
    smooth_window=None,
    smooth_time_fraction=0.05,
    error_type="relative",
    production=0,
    time_horizon=20.0,
):
    if error_type not in {"relative", "absolute"}:
        raise ValueError("error_type must be 'relative' or 'absolute'")

    component_info = [
        (0, r"$x$", "-", None),
        (1, rf"$\sin({cfg.freq0}\pi x)$", (0, (6, 3)), cfg.amp0),
        (2, rf"$\sin({cfg.freq1}\pi x)$", (0, (5, 1, 1, 1)), cfg.amp1),
        (3, rf"$\sin({cfg.freq2}\pi x)$", (0, (3, 1, 1, 1, 1, 1)), cfg.amp2),
        (4, rf"$\sin({cfg.freq3}\pi x)$", ":", cfg.amp3),
    ]
    active_component_info = [
        (idx, name, style)
        for idx, name, style, amp in component_info
        if (idx == 0) or (abs(amp) > 1e-14)
    ]

    plt.figure(figsize=(8, 4))
    ax = plt.gca()

    for name in method_order:
        t = histories[name]["time_mean"]
        error_prefix = "component_rel_err" if error_type == "relative" else "component_abs_err"
        coef_err_mean = histories[name][f"{error_prefix}_mean"]
        coef_err_low = histories[name][f"{error_prefix}_low"]
        coef_err_high = histories[name][f"{error_prefix}_high"]

        if production:
            within_horizon = t <= time_horizon
            t = t[within_horizon]
            coef_err_mean = coef_err_mean[within_horizon]
            coef_err_low = coef_err_low[within_horizon]
            coef_err_high = coef_err_high[within_horizon]

        if t.size == 0:
            continue

        method_smooth_window = (
            smooth_window
            if smooth_window is not None
            else smoothing_window_from_time(t, smooth_time_fraction)
        )

        for idx, _, style in active_component_info:
            mean_s = smooth_positive_curve(best_seen_loss(coef_err_mean[:, idx]), window=method_smooth_window)
            low_s = smooth_positive_curve(best_seen_loss(coef_err_low[:, idx]), window=method_smooth_window)
            high_s = smooth_positive_curve(best_seen_loss(coef_err_high[:, idx]), window=method_smooth_window)

            ax.fill_between(t, low_s, high_s, color=method_colors[name], alpha=0.06, linewidth=0)
            ax.plot(t, mean_s, color=method_colors[name], linestyle=style, alpha=0.95)

    ax.set_yscale("log")
    ax.set_xlabel("compute time (seconds)", fontsize=12)
    ax.set_ylabel(f"best {error_type} coefficient error", fontsize=12)
    if production:
        x_padding = 0.05 * time_horizon
        ax.set_xlim(-x_padding, time_horizon + x_padding)
    ax.grid(True, which="major", alpha=0.4, linestyle="-")
    ax.grid(True, which="minor", alpha=0.2, linestyle=":")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    ax.tick_params(axis="both", which="major", direction="out", length=5, width=0.8, colors="black", top=False, right=False)
    ax.tick_params(axis="both", which="minor", direction="out", length=3, width=0.6, colors="black", top=False, right=False)
    ax.set_axisbelow(True)

    method_handles = [
        plt.Line2D([0], [0], color=method_colors[name], linestyle="-", linewidth=2.5, label=method_labels[name])
        for name in method_order
    ]
    component_handles = [
        plt.Line2D([0], [0], color="black", linestyle=style, linewidth=2.5, label=name)
        for _, name, style in active_component_info
    ]

    leg1 = ax.legend(handles=method_handles, loc="upper right", frameon=True, framealpha=0.9, fontsize=10)
    leg1.get_frame().set_edgecolor("black")
    leg1.get_frame().set_linewidth(0.8)
    ax.add_artist(leg1)

    leg2 = ax.legend(handles=component_handles, loc="lower right", frameon=True, framealpha=0.9, fontsize=10)
    leg2.get_frame().set_edgecolor("black")
    leg2.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(output_path, bbox_inches="tight")
    return ax


def plot_ntk_errors(
    histories,
    ntk_evals,
    target_ntk_coef,
    method_order,
    method_labels,
    method_colors,
    save_path=None,
    smooth_window=None,
    smooth_time_fraction=0.05,
    error_type="relative",
    production=0,
    time_horizon=20.0,
):
    if error_type not in {"relative", "absolute"}:
        raise ValueError("error_type must be 'relative' or 'absolute'")

    mode_styles = ["-", (0, (6, 3)), (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), ":"]
    mode_order = np.argsort(np.abs(target_ntk_coef))[::-1]
    top_k = min(len(mode_styles), len(mode_order))
    active_modes = sorted(
        mode_order[:top_k],
        key=lambda mode: ntk_evals[mode].item(),
        reverse=True,
    )

    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    for name in method_order:
        t = histories[name]["time_mean"]
        error_prefix = "ntk_rel_err" if error_type == "relative" else "ntk_abs_err"
        err_mean = histories[name][f"{error_prefix}_mean"]
        err_low = histories[name][f"{error_prefix}_low"]
        err_high = histories[name][f"{error_prefix}_high"]

        if production:
            within_horizon = t <= time_horizon
            t = t[within_horizon]
            err_mean = err_mean[within_horizon]
            err_low = err_low[within_horizon]
            err_high = err_high[within_horizon]

        if t.size == 0:
            continue

        method_smooth_window = (
            smooth_window
            if smooth_window is not None
            else smoothing_window_from_time(t, smooth_time_fraction)
        )

        for linestyle, m in zip(mode_styles, active_modes):
            mean_s = smooth_positive_curve(best_seen_loss(err_mean[:, m]), window=method_smooth_window)
            low_s = smooth_positive_curve(best_seen_loss(err_low[:, m]), window=method_smooth_window)
            high_s = smooth_positive_curve(best_seen_loss(err_high[:, m]), window=method_smooth_window)

            ax.fill_between(t, low_s, high_s, color=method_colors[name], alpha=0.06, linewidth=0)
            ax.plot(t, mean_s, color=method_colors[name], linestyle=linestyle, alpha=0.95)

    ax.set_yscale("log")
    ax.set_xlabel("compute time (seconds)", fontsize=12)
    ax.set_ylabel(f"best {error_type} NTK-mode error", fontsize=12)
    if production:
        x_padding = 0.05 * time_horizon
        ax.set_xlim(-x_padding, time_horizon + x_padding)
    ax.grid(True, which="major", alpha=0.4, linestyle="-")
    ax.grid(True, which="minor", alpha=0.2, linestyle=":")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    ax.tick_params(axis="both", which="major", direction="out", length=5, width=0.8, colors="black", top=False, right=False)
    ax.tick_params(axis="both", which="minor", direction="out", length=3, width=0.6, colors="black", top=False, right=False)
    ax.set_axisbelow(True)

    method_handles = [
        plt.Line2D([0], [0], color=method_colors[name], linestyle="-", linewidth=2.5, label=method_labels[name])
        for name in method_order
    ]
    mode_handles = [
        plt.Line2D([0], [0], color="black", linestyle=style, linewidth=2.5, label=rf"NTK mode {m}, $\lambda={ntk_evals[m].item():.2e}$")
        for style, m in zip(mode_styles, active_modes)
    ]

    leg1 = ax.legend(handles=method_handles, loc="upper right", frameon=True, framealpha=0.9, fontsize=10)
    leg1.get_frame().set_edgecolor("black")
    leg1.get_frame().set_linewidth(0.8)
    ax.add_artist(leg1)

    leg2 = ax.legend(handles=mode_handles, loc="lower right", frameon=True, framealpha=0.9, fontsize=10)
    leg2.get_frame().set_edgecolor("black")
    leg2.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(output_path, bbox_inches="tight")
    return ax


def save_training_animation(
    single_seed_history,
    cfg,
    method,
    method_labels,
    method_colors,
    x_train,
    y_train,
    x_plot,
    y_target,
    output_path=None,
    device=None,
    dtype=None,
    max_frames=120,
    fps=10,
):
    """Save a GIF showing one method's predictions across stored iterates.

    The function works with either iteration- or time-sampled iterates. The
    snapshot policy is determined when ``run_one_seed`` creates the history.
    """
    from pathlib import Path

    import matplotlib.animation as animation
    import torch

    from model import NN

    if method not in single_seed_history:
        raise ValueError(f"unknown method: {method}")

    history = single_seed_history[method]
    iterates = history["iterates"]
    if len(iterates) == 0:
        raise ValueError("no iterates were stored; enable an iterate snapshot interval before training")

    device = x_train.device if device is None else device
    dtype = x_train.dtype if dtype is None else dtype
    method_label = method_labels.get(method, method)
    method_color = method_colors[method]

    iterate_steps = np.asarray(history.get("iterate_steps", np.arange(len(iterates))), dtype=int)
    iterate_times = np.asarray(history.get("iterate_times", np.full(len(iterates), np.nan)), dtype=float)
    iterates = torch.as_tensor(iterates, dtype=dtype, device=device)
    if iterates.ndim == 1:
        iterates = iterates.unsqueeze(0)

    x_plot = torch.as_tensor(x_plot, dtype=dtype, device=device)
    x_plot_np = x_plot.detach().cpu().numpy().reshape(-1)
    y_target_np = torch.as_tensor(y_target).detach().cpu().numpy().reshape(-1)
    x_train_np = x_train.detach().cpu().numpy().reshape(-1)
    y_train_np = y_train.detach().cpu().numpy().reshape(-1)

    anim_model = NN(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
        beta=cfg.beta,
        freeze_first_layer=cfg.freeze_first_layer,
    ).to(device=device, dtype=dtype)

    def load_iterate(vector):
        with torch.no_grad():
            # Training histories contain only trainable parameters.  The
            # random-feature model stores fc1 as frozen, so loading into all
            # model parameters would exhaust the shorter snapshot vector.
            idx = 0
            for parameter in anim_model.trainable_parameters():
                n = parameter.numel()
                parameter.copy_(vector.flatten()[idx:idx + n].view_as(parameter))
                idx += n

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_plot_np, y_target_np, color="black", linewidth=2, label="target")
    model_line, = ax.plot([], [], color=method_color, linewidth=2, label=f"{method_label} prediction")
    ax.scatter(x_train_np, y_train_np, color="gray", s=25, alpha=0.6, label="training points")
    step_text = ax.text(
        0.5,
        0.02,
        "",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{method_label} training evolution")
    ax.legend(loc="upper right")
    ax.set_xlim(float(x_plot_np.min()), float(x_plot_np.max()))
    y_min = min(y_target_np.min(), y_train_np.min()) - 0.2
    y_max = max(y_target_np.max(), y_train_np.max()) + 0.2
    ax.set_ylim(y_min, y_max)

    frame_indices = np.linspace(0, len(iterates) - 1, min(max_frames, len(iterates)), dtype=int)

    def init():
        model_line.set_data([], [])
        step_text.set_text("")
        return model_line, step_text

    def update(frame_index):
        load_iterate(iterates[frame_index])
        y_pred = anim_model(x_plot).detach().cpu().numpy().reshape(-1)
        model_line.set_data(x_plot_np, y_pred)

        text = f"{method_label} step: {iterate_steps[frame_index]}"
        if np.isfinite(iterate_times[frame_index]):
            text += f"\ncompute time: {iterate_times[frame_index]:.2f} s"
        step_text.set_text(text)
        return model_line, step_text
    
    animation_object = animation.FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        init_func=init,
        blit=True,
        interval=1000 / fps,
    )

    if output_path is None:
        output_path = f"Figures/NNTK/Comparison_GD_Newton_training_evolution_{method}.gif"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation_object.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    return output_path
