from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# =============================================================================
# Config
# =============================================================================

DATA_PATH = Path(
    "../../data/inference_ready_data_all_discussions_s=25_l=5.parquet"
)

OUTDIR = Path("../../figures")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTBASE = OUTDIR / "figure_3"

models = [
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct",
    "dolphin-2.7-mixtral-8x7b-AWQ",
    "Mixtral-8x7B-Instruct-v0.1-AWQ-INT4",
    "gpt-4o-mini",
    "grok-4-1-fast-non-reasoning",
]

model_labels = {
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "dolphin-2.7-mixtral-8x7b-AWQ": "DolphinMixtral-8x7B",
    "Mixtral-8x7B-Instruct-v0.1-AWQ-INT4": "Mixtral-8x7B",
    "gpt-4o-mini": "GPT-4o-mini",
    "grok-4-1-fast-non-reasoning": "Grok-4.1-fast",
}

model_colors = {
    "Llama-3.1-8B-Instruct": "tab:blue",
    "Qwen2.5-7B-Instruct": "tab:purple",
    "dolphin-2.7-mixtral-8x7b-AWQ": "tab:green",
    "Mixtral-8x7B-Instruct-v0.1-AWQ-INT4": "tab:orange",
    "gpt-4o-mini": "tab:red",
    "grok-4-1-fast-non-reasoning": "tab:pink",
}

topics = [
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

DPI = 600
SUMMARY_Y_PAD_FRAC = 0.08

N_BOOT = 5000
CI_LEVEL = 0.95
RNG_SEED = 42


# =============================================================================
# Style
# =============================================================================

def set_publication_style():
    mpl.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


# =============================================================================
# Helpers
# =============================================================================

def summary_limits(values, pad_frac=SUMMARY_Y_PAD_FRAC, lower_clip=None, upper_clip=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        lo, hi = 0.0, 1.0
    else:
        ymin = float(np.nanmin(values))
        ymax = float(np.nanmax(values))
        span = ymax - ymin

        if span <= 0:
            pad = pad_frac * (abs(ymax) if ymax != 0 else 1.0)
        else:
            pad = pad_frac * span

        lo = ymin - pad
        hi = ymax + pad

        if hi <= lo:
            hi = lo + 1.0

    if lower_clip is not None:
        lo = max(lower_clip, lo)
    if upper_clip is not None:
        hi = min(upper_clip, hi)

    if hi <= lo:
        hi = lo + 1e-6

    return lo, hi


def bootstrap_mean_ci(x, n_boot=N_BOOT, ci_level=CI_LEVEL, seed=RNG_SEED):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.nan, np.nan
    if x.size == 1:
        return x[0], x[0]

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot_means = x[idx].mean(axis=1)

    alpha = 1.0 - ci_level
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2))
    return lo, hi


def one_row_figsize_with_original_panel_aspect():
    """
    Preserve the physical panel aspect ratio from the original 2 x 1 figure.

    Original figure:
        figsize=(2, 3.5)
        left=0.18, right=0.98
        bottom=0.12, top=0.82
        hspace=0.18

    For two vertically stacked axes, total inner height is:
        2 * ax_height + hspace * ax_height = (2 + hspace) * ax_height
    """
    old_fig_w, old_fig_h = 2.0, 3.5
    old_left, old_right = 0.18, 0.98
    old_bottom, old_top = 0.12, 0.82
    old_hspace = 0.18

    old_ax_w = old_fig_w * (old_right - old_left)
    old_ax_h = old_fig_h * (old_top - old_bottom) / (2.0 + old_hspace)

    new_left, new_right = 0.11, 0.985
    new_bottom, new_top = 0.22, 0.73
    new_wspace = 0.775

    new_fig_w = old_ax_w * (2.0 + new_wspace) / (new_right - new_left)
    new_fig_h = old_ax_h / (new_top - new_bottom)

    return (new_fig_w, new_fig_h), {
        "left": new_left,
        "right": new_right,
        "bottom": new_bottom,
        "top": new_top,
        "wspace": new_wspace,
    }


# =============================================================================
# Data prep
# =============================================================================

def prepare_relative_distance_summary(df: pd.DataFrame):
    d = (
        df.loc[
            df["model"].isin(models) & df["topic"].isin(topics),
            ["model", "topic", "framing", "run", "t", "x_i", "x_j", "init_x_i", "init_x_j"],
        ]
        .copy()
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["framing", "run", "t", "x_i", "x_j", "init_x_i", "init_x_j"])
    )

    for col in ["framing", "t", "x_i", "x_j", "init_x_i", "init_x_j"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["framing", "t", "x_i", "x_j", "init_x_i", "init_x_j"])

    if np.allclose(d["t"], np.round(d["t"])):
        d["t"] = d["t"].astype(int)

    d["pair_distance"] = (d["x_i"] - d["x_j"]).abs()
    d["d0"] = (d["init_x_i"] - d["init_x_j"]).abs()

    pair = (
        d.groupby(["model", "topic", "framing", "run", "t"], observed=True)
        .agg(
            pair_distance=("pair_distance", "mean"),
            d0=("d0", "first"),
        )
        .reset_index()
        .sort_values(["model", "topic", "framing", "run", "t"])
    )

    pair = pair.loc[pair["d0"] > 0].copy()
    pair["relative_distance"] = pair["pair_distance"] / pair["d0"]

    pair["model"] = pd.Categorical(pair["model"], categories=models, ordered=True)
    pair["topic"] = pd.Categorical(pair["topic"], categories=topics, ordered=True)

    topic_framing_level = (
        pair.groupby(["model", "topic", "framing", "t"], observed=True)["relative_distance"]
        .mean()
        .rename("mean_relative_distance")
        .reset_index()
        .sort_values(["model", "topic", "framing", "t"])
    )

    rows = []
    for (model, t), x in topic_framing_level.groupby(["model", "t"], observed=True)["mean_relative_distance"]:
        x = x.to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        ci_low, ci_high = bootstrap_mean_ci(x)
        rows.append({
            "model": model,
            "t": t,
            "center": float(np.mean(x)) if x.size else np.nan,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })

    summary = pd.DataFrame(rows).sort_values(["model", "t"]).reset_index(drop=True)
    return summary


def prepare_shift_summary(df: pd.DataFrame):
    d = (
        df.loc[
            df["model"].isin(models) & df["topic"].isin(topics),
            ["model", "topic", "framing", "t", "delta_x_i"],
        ]
        .copy()
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["framing", "t", "delta_x_i"])
    )

    for col in ["framing", "t", "delta_x_i"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["framing", "t", "delta_x_i"])

    if np.allclose(d["t"], np.round(d["t"])):
        d["t"] = d["t"].astype(int)

    d["abs_shift"] = d["delta_x_i"].abs()
    d["model"] = pd.Categorical(d["model"], categories=models, ordered=True)
    d["topic"] = pd.Categorical(d["topic"], categories=topics, ordered=True)

    topic_framing_level = (
        d.groupby(["model", "topic", "framing", "t"], observed=True)["abs_shift"]
        .mean()
        .rename("mean_abs_shift")
        .reset_index()
        .sort_values(["model", "topic", "framing", "t"])
    )

    rows = []
    for (model, t), x in topic_framing_level.groupby(["model", "t"], observed=True)["mean_abs_shift"]:
        x = x.to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        ci_low, ci_high = bootstrap_mean_ci(x)
        rows.append({
            "model": model,
            "t": t,
            "center": float(np.mean(x)) if x.size else np.nan,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })

    summary = pd.DataFrame(rows).sort_values(["model", "t"]).reset_index(drop=True)
    return summary


# =============================================================================
# Plot
# =============================================================================

def make_composite_figure(relative_distance_summary, shift_summary):
    all_t = np.array(sorted(relative_distance_summary["t"].unique()))

    figsize, adjust = one_row_figsize_with_original_panel_aspect()

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2,
        figsize=(4, 2.5),
        sharex=True,
    )

    # -------------------------------------------------------------------------
    # Panel A: relative opinion distance d_t / d_0
    # -------------------------------------------------------------------------
    for model in models:
        s = relative_distance_summary.loc[
            relative_distance_summary["model"] == model
        ].sort_values("t")
        if s.empty:
            continue

        ax_left.fill_between(
            s["t"].to_numpy(),
            s["ci_low"].to_numpy(),
            s["ci_high"].to_numpy(),
            color=model_colors[model],
            alpha=0.14,
            linewidth=0,
        )
        ax_left.plot(
            s["t"].to_numpy(),
            s["center"].to_numpy(),
            color=model_colors[model],
            lw=1.8,
            label=model_labels[model],
        )

    ymin, ymax = summary_limits(
        np.r_[
            relative_distance_summary["ci_low"].to_numpy(),
            relative_distance_summary["ci_high"].to_numpy(),
        ],
        lower_clip=0.0,
    )
    ax_left.set_ylim(ymin, ymax)
    ax_left.set_xlim(all_t.min(), all_t.max())
    ax_left.set_ylabel("Opinion distance", fontsize=12)
    # increase tick label fontsize for both axes
    ax_left.tick_params(axis="both", which="major", labelsize=10)

    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.text(
        -0.7, 1.025, "a",
        transform=ax_left.transAxes,
        fontweight="bold",
        fontsize=12,
        va="bottom",
    )

    # -------------------------------------------------------------------------
    # Panel B: mean absolute opinion shift
    # -------------------------------------------------------------------------
    for model in models:
        s = shift_summary.loc[shift_summary["model"] == model].sort_values("t")
        if s.empty:
            continue

        ax_right.fill_between(
            s["t"].to_numpy(),
            s["ci_low"].to_numpy(),
            s["ci_high"].to_numpy(),
            color=model_colors[model],
            alpha=0.14,
            linewidth=0,
        )
        ax_right.plot(
            s["t"].to_numpy(),
            s["center"].to_numpy(),
            color=model_colors[model],
            lw=1.8,
            label=model_labels[model],
        )

    ymin, ymax = summary_limits(
        np.r_[
            shift_summary["ci_low"].to_numpy(),
            shift_summary["ci_high"].to_numpy(),
        ],
        lower_clip=0.0,
    )
    ax_right.set_ylim(ymin, ymax)
    ax_right.set_ylabel("Mean absolute\nopinion shift", fontsize=12)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.tick_params(axis="both", which="major", labelsize=10)
    ax_right.text(
        -0.725, 1.025, "b",
        transform=ax_right.transAxes,
        fontweight="bold",
        fontsize=12,
        va="bottom",
    )

    ax_left.set_xticks(all_t)
    ax_right.set_xticks(all_t)

    fig.supxlabel("Discussion round", fontsize=12)

    handles, labels = ax_left.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.45, 1.025),
        ncol=3,
        frameon=False,
        columnspacing=1.0,
        handlelength=2.0,
        fontsize=9
    )

    fig.subplots_adjust(**adjust)

    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    set_publication_style()

    df = pd.read_parquet(DATA_PATH)

    relative_distance_summary = prepare_relative_distance_summary(df)
    shift_summary = prepare_shift_summary(df)

    fig = make_composite_figure(relative_distance_summary, shift_summary)

    fig.savefig(OUTBASE.with_suffix(".pdf"), bbox_inches="tight")
    #fig.savefig(OUTBASE.with_suffix(".png"), bbox_inches="tight")
    plt.show()