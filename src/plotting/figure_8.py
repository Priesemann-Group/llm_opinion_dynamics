#!/usr/bin/env python3

import os
import argparse
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
import json
from scipy.stats import linregress

LEFT_DATA_PATH_DEFAULT = (
    "../../data/all_discussions_s=25_l=5.parquet"
)

RIGHT_DATA_PATH_DEFAULT = (
    "../../data/inference_ready_data_all_discussions_s=25_l=5.parquet"
)

OUTPUT_DIR_DEFAULT = "../../figures"

TOPICS = [
    "Climate_Change",
    "Vaccination",
    "Shape_of_the_Earth",
    "Global_Wealth_Distribution",
    "Abortion",
    "Social_Media",
    "Artificial_Intelligence",
    "Morality_and_Religion",
    "Free_Will",
    "Musical_Preference",
    "Food_Preference",
    "Art_Style_Preference",
]

MODELS = [
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct",
    "dolphin-2.7-mixtral-8x7b-AWQ",
    "Mixtral-8x7B-Instruct-v0.1-AWQ-INT4",
    "gpt-4o-mini",
    "grok-4-1-fast-non-reasoning",
]

MODEL_LABELS = {
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "dolphin-2.7-mixtral-8x7b-AWQ": "DolphinMixtral-8x7B",
    "Mixtral-8x7B-Instruct-v0.1-AWQ-INT4": "Mixtral-8x7B",
    "gpt-4o-mini": "GPT-4o-mini",
    "grok-4-1-fast-non-reasoning": "Grok-4.1-fast",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a combined 3x4 figure. "
            "Left block (3x2): opinion uncertainty curve vs opinion, averaged across topics per LLM. "
            "Right block (3x2): variance in opinion shift vs opinion uncertainty, averaged across topics per LLM."
        )
    )
    parser.add_argument(
        "--left-data-path",
        default=LEFT_DATA_PATH_DEFAULT,
        help="Path to all_discussions parquet used for the left block.",
    )
    parser.add_argument(
        "--right-data-path",
        default=RIGHT_DATA_PATH_DEFAULT,
        help="Path to inference_ready_data parquet used for the right block.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR_DEFAULT,
        help="Directory where the figure is saved.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="DPI used when saving the figure.",
    )
    parser.add_argument(
        "--framing-mode",
        choices=["all", "normal", "reverse"],
        default="all",
        help="Which framing subset to use when pooling across topics.",
    )
    parser.add_argument(
        "--n-bins-left",
        type=int,
        default=41,
        help="Number of opinion bins for the left-block uncertainty curves.",
    )
    parser.add_argument(
        "--min-count-left-bin",
        type=int,
        default=50,
        help="Minimum count in an opinion bin to plot the left-block mean uncertainty curve.",
    )
    parser.add_argument(
        "--n-bins-right",
        type=int,
        default=10,
        help="Number of uncertainty bins for the right-block variance curves.",
    )
    parser.add_argument(
        "--min-count-right-bin",
        type=int,
        default=20,
        help="Minimum count per uncertainty bin for the right block.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap samples for right-block confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for bootstrap sampling.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively.",
    )
    return parser.parse_args()


def coerce_reverse_framing(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer, float, np.floating)):
        if value == 1 or value == 1.0:
            return False
        if value == -1 or value == -1.0:
            return True
        raise ValueError(f"Unexpected numeric framing value: {value}")

    if isinstance(value, str):
        value_norm = value.strip().lower()
        if value_norm in {"normal", "base", "original", "forward", "1", "false"}:
            return False
        if value_norm in {"reverse", "reversed", "rev", "-1", "true"}:
            return True

    raise ValueError(f"Unexpected framing value: {value}")


def load_parquet_data(data_path: str, stride2: bool) -> pd.DataFrame:
    df = pd.read_parquet(data_path)
    if stride2:
        df = df.iloc[::2].copy()
    else:
        df = df.copy()
    df["framing_bool"] = df["framing"].apply(coerce_reverse_framing)
    return df


def apply_framing_mode(df: pd.DataFrame, framing_mode: str) -> pd.DataFrame:
    if framing_mode == "all":
        return df.copy()
    if framing_mode == "normal":
        return df.loc[~df["framing_bool"]].copy()
    if framing_mode == "reverse":
        return df.loc[df["framing_bool"]].copy()
    raise ValueError(f"Unknown framing_mode={framing_mode}")


def to_jsonable_float(x: float):
    return None if not np.isfinite(x) else float(x)


# ----------------------------------------------------------
# Entropy bounds
# ----------------------------------------------------------

def shannon_entropy(p, eps: float = 1e-15) -> float:
    p = np.array(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log2(p))


def minimal_entropy(m: float) -> float:
    if not np.isfinite(m):
        return np.nan
    if m < -2.0 - 1e-12 or m > 2.0 + 1e-12:
        return np.nan

    m = float(np.clip(m, -2.0, 2.0))

    if np.isclose(m, -2): return 0.0
    if np.isclose(m, -1): return 0.0
    if np.isclose(m,  0): return 0.0
    if np.isclose(m,  1): return 0.0
    if np.isclose(m,  2): return 0.0

    p = np.zeros(5)

    if -2 <= m < -1:
        x = -(m + 1)
        p[0] = x
        p[1] = 1.0 - x
    elif -1 <= m < 0:
        x = -m
        p[1] = x
        p[2] = 1.0 - x
    elif 0 <= m < 1:
        x = 1.0 - m
        p[2] = x
        p[3] = 1.0 - x
    elif 1 <= m < 2:
        x = 2.0 - m
        p[3] = x
        p[4] = 1.0 - x

    p = np.maximum(p, 0.0)
    s = np.sum(p)
    if s <= 0.0:
        return np.nan
    p /= s
    return shannon_entropy(p)


def mean_of_beta(beta: float, values: np.ndarray) -> float:
    w = np.exp(beta * values)
    return np.sum(values * w) / np.sum(w)


def maxent_entropy(m: float) -> float:
    values = np.array([-2, -1, 0, 1, 2], dtype=float)

    if np.isclose(m, -2) or np.isclose(m, 2):
        return 0.0

    def f(beta):
        return mean_of_beta(beta, values) - m

    beta_star = brentq(f, -100.0, 100.0)
    w = np.exp(beta_star * values)
    p = w / np.sum(w)
    return shannon_entropy(p)


def compute_entropy_bounds(n_points: int = 201) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m_values = np.linspace(-2, 2, n_points)
    h_min = np.array([minimal_entropy(m) for m in m_values])
    h_max = np.array([maxent_entropy(m) for m in m_values])
    return m_values, h_min, h_max


# ----------------------------------------------------------
# Left-block preparation
# ----------------------------------------------------------

def build_left_model_dataframe(
    df: pd.DataFrame,
    model: str,
    framing_mode: str,
) -> pd.DataFrame:
    model_df = df.loc[df["model"] == model].copy()
    model_df = apply_framing_mode(model_df, framing_mode)

    if model_df.empty:
        raise ValueError(f"No left-block rows found for model={model}, framing_mode={framing_mode}.")

    common_cols = [c for c in ["discussion_id", "t", "model", "topic", "framing", "framing_bool"] if c in model_df.columns]

    i_df = model_df[common_cols + ["x_i", "H_i"]].copy()
    i_df = i_df.rename(columns={"x_i": "x", "H_i": "H"})
    i_df["agent"] = "i"

    j_df = model_df[common_cols + ["x_j", "H_j"]].copy()
    j_df = j_df.rename(columns={"x_j": "x", "H_j": "H"})
    j_df["agent"] = "j"

    long_df = pd.concat([i_df, j_df], ignore_index=True)
    long_df["t_plot"] = long_df["t"].astype(int)
    long_df["x_plot"] = long_df["x"].astype(float)
    long_df["h_plot"] = long_df["H"].astype(float) / np.log(2.0)

    h_min = np.array([minimal_entropy(m) for m in long_df["x_plot"].to_numpy()])
    tol = 1e-6

    valid = (
        np.isfinite(long_df["x_plot"].to_numpy())
        & np.isfinite(long_df["h_plot"].to_numpy())
        & np.isfinite(long_df["t_plot"].to_numpy())
        & np.isfinite(h_min)
        & (long_df["x_plot"].to_numpy() >= -2.0 - tol)
        & (long_df["x_plot"].to_numpy() <=  2.0 + tol)
        & (long_df["h_plot"].to_numpy() >= h_min - tol)
    )

    long_df = long_df.loc[valid].copy()

    if long_df.empty:
        raise ValueError(f"No valid left-block rows remain for model={model}.")

    return long_df


def compute_mean_uncertainty_curve(
    model_df: pd.DataFrame,
    n_bins_left: int,
    min_count_left_bin: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = model_df["x_plot"].to_numpy(dtype=float)
    h = model_df["h_plot"].to_numpy(dtype=float)

    x_edges = np.linspace(-2.0, 2.0, n_bins_left + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

    h_mean = np.full(n_bins_left, np.nan, dtype=float)
    counts = np.zeros(n_bins_left, dtype=int)

    for i in range(n_bins_left):
        left = x_edges[i]
        right = x_edges[i + 1]

        if i == n_bins_left - 1:
            mask = (x >= left) & (x <= right)
        else:
            mask = (x >= left) & (x < right)

        vals = h[mask]
        vals = vals[np.isfinite(vals)]
        counts[i] = vals.size

        if vals.size >= min_count_left_bin:
            h_mean[i] = np.mean(vals)

    return x_centers, h_mean, counts


# ----------------------------------------------------------
# Right-block preparation
# ----------------------------------------------------------

def build_right_model_dataframe(
    df: pd.DataFrame,
    model: str,
    framing_mode: str,
) -> pd.DataFrame:
    model_df = df.loc[df["model"] == model].copy()
    model_df = apply_framing_mode(model_df, framing_mode)

    if model_df.empty:
        raise ValueError(f"No right-block rows found for model={model}, framing_mode={framing_mode}.")

    required_cols = ["t", "H_i", "delta_x_i"]
    missing = [c for c in required_cols if c not in model_df.columns]
    if missing:
        raise ValueError(f"Right-block dataframe missing required columns for model={model}: {missing}")

    model_df["h_plot"] = model_df["H_i"].astype(float) / np.log(2.0)
    model_df["dx_plot"] = model_df["delta_x_i"].astype(float)

    valid = (
        np.isfinite(model_df["h_plot"].to_numpy())
        & np.isfinite(model_df["dx_plot"].to_numpy())
    )

    if "x_i" in model_df.columns:
        x = model_df["x_i"].to_numpy(dtype=float)
        h_min = np.array([minimal_entropy(m) for m in x])
        tol = 1e-6
        valid &= (
            np.isfinite(x)
            & np.isfinite(h_min)
            & (x >= -2.0 - tol)
            & (x <=  2.0 + tol)
            & (model_df["h_plot"].to_numpy() >= h_min - tol)
        )

    model_df = model_df.loc[valid].copy()

    if model_df.empty:
        raise ValueError(f"No valid right-block rows remain for model={model}.")

    return model_df


def bootstrap_variance_ci(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    ci: float = 95.0,
    chunk_size: int = 200,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size < 2:
        return np.nan, np.nan

    alpha = (100.0 - ci) / 2.0
    boot_vars = np.empty(n_bootstrap, dtype=float)

    n = values.size
    done = 0
    while done < n_bootstrap:
        m = min(chunk_size, n_bootstrap - done)
        idx = rng.integers(0, n, size=(m, n))
        samples = values[idx]
        boot_vars[done:done + m] = np.var(samples, axis=1, ddof=1)
        done += m

    low = np.percentile(boot_vars, alpha)
    high = np.percentile(boot_vars, 100.0 - alpha)
    return low, high


def compute_binned_variance_curve(
    model_df: pd.DataFrame,
    n_bins_right: int,
    min_count_right_bin: int,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, object]:
    h = model_df["h_plot"].to_numpy(dtype=float)
    dx = model_df["dx_plot"].to_numpy(dtype=float)

    h_edges = np.linspace(0.0, np.log2(5.0), n_bins_right + 1)
    h_centers = 0.5 * (h_edges[:-1] + h_edges[1:])

    variance = np.full(n_bins_right, np.nan, dtype=float)
    ci_low = np.full(n_bins_right, np.nan, dtype=float)
    ci_high = np.full(n_bins_right, np.nan, dtype=float)
    counts = np.zeros(n_bins_right, dtype=int)

    rng = np.random.default_rng(seed)

    for i in range(n_bins_right):
        left = h_edges[i]
        right = h_edges[i + 1]

        if i == n_bins_right - 1:
            mask = (h >= left) & (h <= right)
        else:
            mask = (h >= left) & (h < right)

        vals = dx[mask]
        vals = vals[np.isfinite(vals)]
        counts[i] = vals.size

        if vals.size < max(2, min_count_right_bin):
            continue

        variance[i] = np.var(vals, ddof=1)
        low, high = bootstrap_variance_ci(
            values=vals,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        ci_low[i] = low
        ci_high[i] = high

    valid = np.isfinite(variance)

    if valid.sum() >= 2:
        x_fit = h_centers[valid]
        y_fit = variance[valid]

        reg = linregress(x_fit, y_fit)
        slope = float(reg.slope)
        intercept = float(reg.intercept)
        r_value = float(reg.rvalue)
        p_value = float(reg.pvalue)

        y_pred = slope * h_centers + intercept
    else:
        y_pred = np.full_like(h_centers, np.nan, dtype=float)
        r_value = np.nan
        p_value = np.nan
        slope = np.nan
        intercept = np.nan

    return {
        "h_centers": h_centers,
        "variance": variance,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "counts": counts,
        "y_pred": y_pred,
        "r_value": r_value,
        "p_value": p_value,
        "slope": slope,
        "intercept": intercept,
    }


# ----------------------------------------------------------
# Plotting
# ----------------------------------------------------------

def plot_combined_figure(
    left_raw: pd.DataFrame,
    right_raw: pd.DataFrame,
    framing_mode: str,
    output_dir: str,
    dpi: int,
    n_bins_left: int,
    min_count_left_bin: int,
    n_bins_right: int,
    min_count_right_bin: int,
    n_bootstrap: int,
    seed: int,
    show: bool,
) -> Tuple[str, str]:
    m_values, h_min_values, h_max_values = compute_entropy_bounds()

    left_for_cbar = apply_framing_mode(left_raw, framing_mode)
    t_min = int(left_for_cbar["t"].min())
    t_max = int(left_for_cbar["t"].max())
    bounds = np.arange(t_min - 0.5, t_max + 1.5, 1)
    cmap_steps = plt.get_cmap("viridis_r", t_max - t_min + 1)
    norm_steps = BoundaryNorm(bounds, ncolors=cmap_steps.N)

    fig = plt.figure(figsize=(8.57, 3.5))
    gs = fig.add_gridspec(
        3,
        5,
        width_ratios=[1.0, 1.0, 0.15, 1.0, 1.0],
        hspace=0.4,
        wspace=0.35,
    )

    axes_left = np.empty((3, 2), dtype=object)
    axes_right = np.empty((3, 2), dtype=object)

    for idx, model in enumerate(MODELS):
        r = idx // 2
        c = idx % 2

        share_left = axes_left[0, 0] if idx > 0 else None
        share_right = axes_right[0, 0] if idx > 0 else None

        axes_left[r, c] = fig.add_subplot(gs[r, c], sharex=share_left, sharey=share_left)
        axes_right[r, c] = fig.add_subplot(gs[r, c + 3], sharex=share_right, sharey=share_right)

    model_colors = {
        "Llama-3.1-8B-Instruct": "tab:blue",
        "Qwen2.5-7B-Instruct": "tab:purple",
        "dolphin-2.7-mixtral-8x7b-AWQ": "tab:green",
        "Mixtral-8x7B-Instruct-v0.1-AWQ-INT4": "tab:orange",
        "gpt-4o-mini": "tab:red",
        "grok-4-1-fast-non-reasoning": "tab:pink",
    }
    model_colors = {model: model_colors.get(model, "tab:gray") for model in MODELS}

    # ------------------------
    # Left block
    # ------------------------
    for idx, model in enumerate(MODELS):
        r = idx // 2
        c = idx % 2
        ax = axes_left[r, c]

        model_df = build_left_model_dataframe(
            df=left_raw,
            model=model,
            framing_mode=framing_mode,
        )

        x_left = model_df["x_plot"].to_numpy(dtype=float)
        h_left = model_df["h_plot"].to_numpy(dtype=float)
        t_left = model_df["t_plot"].to_numpy(dtype=int)

        rng = np.random.default_rng(seed + idx)
        perm = rng.permutation(x_left.size)

        x_left = x_left[perm]
        h_left = h_left[perm]
        t_left = t_left[perm]

        ax.scatter(
            x_left,
            h_left,
            c=t_left,
            cmap=cmap_steps,
            norm=norm_steps,
            alpha=1.0,
            s=10,
            marker=".",
            edgecolors="none",
            rasterized=True,
        )

        ax.plot(
            m_values,
            h_max_values,
            color="dimgrey",
            lw=2.5,
            label="Maximal uncertainty",
        )

        ax.plot(
            m_values,
            h_min_values,
            color="dimgrey",
            lw=1.6,
            ls=(0, (1, 1)),
            label="Minimal uncertainty" if idx == 0 else None,
        )

        ax.set_title(MODEL_LABELS.get(model, model), fontsize=11)
        ax.set_xlim(-2.05, 2.05)
        ax.set_ylim(-0.03, np.log2(5.0) * 1.03)
        ax.set_xticks(np.arange(-2, 3, 1))
        ax.set_yticks([0, 1, 2])
        ax.tick_params(axis="both", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if r < 2:
            ax.tick_params(labelbottom=False)
        if c > 0:
            ax.tick_params(labelleft=False)

    axes_left[0, 0].legend(
        frameon=False,
        fontsize=9,
        loc="lower left",
        bbox_to_anchor=(1.1, 1.45),
        borderaxespad=0.0,
        ncol=1,
    )

    pos = axes_left[0, 0].get_position()
    cbar_ax = fig.add_axes([
        pos.x0 + 0.01,
        pos.y1 + 0.2,
        0.11,
        0.015,
    ])
    cbar_left = fig.colorbar(
        ScalarMappable(norm=norm_steps, cmap=cmap_steps),
        cax=cbar_ax,
        orientation="horizontal",
        ticks=np.arange(t_min, t_max + 1),
    )
    cbar_left.set_label("Discussion step", fontsize=11, labelpad=2)
    cbar_left.ax.tick_params(labelsize=9)
    cbar_left.ax.tick_params(which="minor", length=0)

    # ------------------------
    # Right block
    # ------------------------
    panel_b_stats = {}
    for idx, model in enumerate(MODELS):
        r = idx // 2
        c = idx % 2
        ax = axes_right[r, c]
        color = model_colors[model]

        model_df = build_right_model_dataframe(
            df=right_raw,
            model=model,
            framing_mode=framing_mode,
        )

        stats = compute_binned_variance_curve(
            model_df=model_df,
            n_bins_right=n_bins_right,
            min_count_right_bin=min_count_right_bin,
            n_bootstrap=n_bootstrap,
            seed=seed + idx,
        )

        valid = np.isfinite(stats["variance"])
        panel_b_stats[model] = {
            "model_label": MODEL_LABELS.get(model, model),
            "n_valid_bins": int(valid.sum()),
            "r_value": to_jsonable_float(stats["r_value"]),
            "p_value": to_jsonable_float(stats["p_value"]),
            "slope": to_jsonable_float(stats["slope"]),
            "intercept": to_jsonable_float(stats["intercept"]),
        }

        if valid.any():
            lower = np.maximum(stats["variance"] - stats["ci_low"], 0.0)
            upper = np.maximum(stats["ci_high"] - stats["variance"], 0.0)
            yerr = np.array([lower, upper])

            ax.errorbar(
                stats["h_centers"][valid],
                stats["variance"][valid],
                yerr=yerr[:, valid],
                fmt="s",
                ms=4.5,
                mfc=color,
                mec=color,
                ecolor=color,
                elinewidth=1.2,
                capsize=2.5,
                alpha=0.95,
                zorder=2,
                label="Binned variance (95% CI)" if idx == 0 else None,
            )

        valid_fit = np.isfinite(stats["y_pred"]) & valid
        if valid_fit.sum() >= 2:
            ax.plot(
                stats["h_centers"][valid_fit],
                stats["y_pred"][valid_fit],
                linestyle="--",
                linewidth=2.2,
                color="black",
                zorder=1,
                label="Linear fit" if idx == 0 else None,
            )

        ax.set_title(MODEL_LABELS.get(model, model), fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))

        if r < 2:
            ax.tick_params(labelbottom=False)
        if c > 0:
            ax.tick_params(labelleft=False)

        if np.isfinite(stats["r_value"]):
            ax.text(
                0.03,
                0.93,
                rf"$r={stats['r_value']:.2f}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )

    axes_right[0, 0].legend(
        frameon=False,
        fontsize=9,
        loc="lower left",
        bbox_to_anchor=(-0.05, 1.6),
        borderaxespad=0.0,
        ncol=2,
    )

    # Block labels
    axes_left[0, 0].text(
        -0.36,
        1.35,
        "a",
        transform=axes_left[0, 0].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
    )
    axes_right[0, 0].text(
        -0.38,
        1.35,
        "b",
        transform=axes_right[0, 0].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # Shared labels
    fig.text(0.29, -0.02, "Opinion", ha="center", fontsize=14)
    fig.text(0.74, -0.02, "Opinion uncertainty", ha="center", fontsize=14)
    fig.text(0.07, 0.5, "Opinion uncertainty", va="center", rotation="vertical", fontsize=14)
    fig.text(0.51, 0.5, "Variance in opinion shift", va="center", rotation="vertical", fontsize=14)

    model_tag = "all_models"
    framing_tag = framing_mode

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"figure_9.pdf",
    )

    json_path = os.path.join(
        output_dir,
        f"figure_9_panel_b_regression_stats.json",
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(panel_b_stats, f, indent=2)

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path, json_path


def main() -> None:
    args = parse_args()

    left_raw = load_parquet_data(args.left_data_path, stride2=False)
    right_raw = load_parquet_data(args.right_data_path, stride2=False)

    output_path, json_path = plot_combined_figure(
        left_raw=left_raw,
        right_raw=right_raw,
        framing_mode=args.framing_mode,
        output_dir=args.output_dir,
        dpi=args.dpi,
        n_bins_left=args.n_bins_left,
        min_count_left_bin=args.min_count_left_bin,
        n_bins_right=args.n_bins_right,
        min_count_right_bin=args.min_count_right_bin,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        show=args.show,
    )

    print(f"Saved figure to: {output_path}")
    print(f"Saved regression stats to: {json_path}")


if __name__ == "__main__":
    main()