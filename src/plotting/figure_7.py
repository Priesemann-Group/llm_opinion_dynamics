#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Fixed model list and plotting order
# ---------------------------------------------------------------------

LLMS = [
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Mixtral-8x7B-Instruct-v0.1-AWQ-INT4",
    "dolphin-2.7-mixtral-8x7b-AWQ",
    "gpt-4o-mini",
    "grok-4-1-fast-non-reasoning",
]

llm_colors = {
    "Llama-3.1-8B-Instruct": "tab:blue",
    "Qwen2.5-7B-Instruct": "tab:purple",
    "dolphin-2.7-mixtral-8x7b-AWQ": "tab:green",
    "Mixtral-8x7B-Instruct-v0.1-AWQ-INT4": "tab:orange",
    "gpt-4o-mini": "tab:red",
    "grok-4-1-fast-non-reasoning": "tab:pink",
}

DISPLAY_NAMES = {
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Qwen2.5-7B-Instruct": "Qwen-2.5-7B",
    "Mixtral-8x7B-Instruct-v0.1-AWQ-INT4": "Mixtral-8x7B",
    "dolphin-2.7-mixtral-8x7b-AWQ": "DolphinMixtral-8x7B",
    "gpt-4o-mini": "GPT-4o-mini",
    "grok-4-1-fast-non-reasoning": "Grok-4-1-fast",
}

# Greedy model-building sequence
GREEDY_SPECS = [
    ("base", "none", "none", "none", "none", False),
    ("topic_static", "none", "static", "none", "none", True),
    ("topic_decay", "none", "decay", "none", "none", True),
    ("interaction_static", "static", "decay", "none", "none", True),
    ("interaction_decay", "decay", "decay", "none", "none", True),
    ("agree_static", "decay", "decay", "static", "none", True),
    ("agree_decay", "decay", "decay", "decay", "none", True),
    ("anchor_static", "decay", "decay", "decay", "static", True),
    ("full", "decay", "decay", "decay", "decay", True),
]

X_LABELS = [
    "Zero mean\nbaseline",
    "+Topic\n(static)",
    "+Topic\n(decay)",
    "+Interaction\n(static)",
    "+Interaction\n(decay)",
    "+Agree\n(static)",
    "+Agree\n(decay)",
    "+Anchor\n(static)",
    "+Anchor\n(decay)",
]

NOISE_CEILING_BINS = {
    "Llama-3.1-8B-Instruct": 5,
    "Qwen2.5-7B-Instruct": 5,
    "Mixtral-8x7B-Instruct-v0.1-AWQ-INT4": 8,
    "dolphin-2.7-mixtral-8x7b-AWQ": 5,
    "gpt-4o-mini": 8,
    "grok-4-1-fast-non-reasoning": 6,
}

def fmt_value(x: float, ndigits: int = 3) -> str:
    s = f"{x:.{ndigits}f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s
# ---------------------------------------------------------------------
# Same model-tag logic as in fitting script
# ---------------------------------------------------------------------

def build_model_tag(interaction, topic_bias, agree_bias, anchor_bias, epsilon):
    eps_tag = "eps" if epsilon else "noeps"
    return (
        f"I-{interaction}"
        f"__T-{topic_bias}"
        f"__A-{agree_bias}"
        f"__C-{anchor_bias}"
        f"__E-{eps_tag}"
    )


def build_json_path(
    values_dir: Path,
    model_tag: str,
    llm: str,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
    max_treedepth: int,
) -> Path:
    return (
        values_dir
        / f"loo_r2_{model_tag}_{llm}_dr{draws}_tu{tune}_"
          f"ch{chains}_co{cores}_ar{target_accept}_td{max_treedepth}.json"
    )


# ---------------------------------------------------------------------
# Robust extraction of loo_r2 from saved json
# ---------------------------------------------------------------------

def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def extract_loo_r2(payload: dict) -> float:
    preferred_keys = ["loo_r2", "r2", "estimate", "mean", "value"]
    for key in preferred_keys:
        if key in payload:
            val = to_float(payload[key])
            if val is not None:
                return val

    numeric_vals = [to_float(v) for v in payload.values()]
    numeric_vals = [v for v in numeric_vals if v is not None]
    if numeric_vals:
        return numeric_vals[0]

    raise ValueError(f"Could not extract loo_r2 from keys: {list(payload.keys())}")


def extract_loo_r2_interval(payload: dict) -> tuple[float, float, float]:
    mean = extract_loo_r2(payload)
    eti_lb = to_float(payload.get("eti_lb"))
    eti_ub = to_float(payload.get("eti_ub"))

    if eti_lb is None or eti_ub is None:
        eti_lb = mean
        eti_ub = mean

    return mean, eti_lb, eti_ub

def build_noise_ceiling_source_path(noise_ceiling_dir: Path, llm: str) -> Path:
    k = NOISE_CEILING_BINS[llm]
    return noise_ceiling_dir / f"bins_{k}_loo_r2_{llm}_full.json"


def build_combined_noise_ceilings(noise_ceiling_dir: Path) -> dict:
    combined = {}

    for llm in LLMS:
        src = build_noise_ceiling_source_path(noise_ceiling_dir, llm)

        with open(src, "r") as f:
            payload = json.load(f)

        combined[llm] = {
            "k": NOISE_CEILING_BINS[llm],
            "noise_ceiling_lower": float(payload["noise_ceiling_lower"]),
            "noise_ceiling_lower_ci_lb": float(payload["noise_ceiling_lower_ci_lb"]),
            "noise_ceiling_lower_ci_ub": float(payload["noise_ceiling_lower_ci_ub"]),
            "noise_ceiling_upper": float(payload["noise_ceiling_upper"]),
        }

    return combined

def extract_noise_ceiling_interval(payload) -> tuple[float, float, float]:
    if isinstance(payload, (int, float, str)):
        nc = float(payload)
        return nc, nc, nc

    nc = to_float(payload.get("noise_ceiling_lower"))
    nc_lb = to_float(payload.get("noise_ceiling_lower_ci_lb"))
    nc_ub = to_float(payload.get("noise_ceiling_lower_ci_ub"))

    if nc is None:
        raise ValueError(f"Could not extract noise ceiling from keys: {list(payload.keys())}")

    if nc_lb is None or nc_ub is None:
        nc_lb = nc
        nc_ub = nc

    return nc, nc_lb, nc_ub

# ---------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------

def set_style():
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--values_dir",
        type=Path,
        default=Path("../../data/variance_explained/loo_r2_values"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("../../figures"),
    )
    parser.add_argument(
        "--noise_ceiling_path",
        type=Path,
        default=Path("../../data/variance_explained/nc_values"),
    )

    # Same hyperparameter args as fitting script
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target_accept", type=float, default=0.9)
    parser.add_argument("--max_treedepth", type=int, default=10)

    args = parser.parse_args()

    set_style()
    args.outdir.mkdir(parents=True, exist_ok=True)

    noise_ceiling_dir = args.noise_ceiling_path
    noise_ceilings = build_combined_noise_ceilings(noise_ceiling_dir)

    data = {}
    data_lb = {}
    data_ub = {}
    old_y_vals = []

    for llm in LLMS:
        y_vals = []
        lb_vals = []
        ub_vals = []
        for _, interaction, topic_bias, agree_bias, anchor_bias, epsilon in GREEDY_SPECS:
            model_tag = build_model_tag(
                interaction=interaction,
                topic_bias=topic_bias,
                agree_bias=agree_bias,
                anchor_bias=anchor_bias,
                epsilon=epsilon,
            )

            json_path = build_json_path(
                values_dir=args.values_dir,
                model_tag=model_tag,
                llm=llm,
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                cores=args.cores,
                target_accept=args.target_accept,
                max_treedepth=args.max_treedepth,
            )

            with open(json_path, "r") as f:
                payload = json.load(f)

            loo_r2, loo_r2_lb, loo_r2_ub = extract_loo_r2_interval(payload)
            y_vals.append(loo_r2)
            lb_vals.append(loo_r2_lb)
            ub_vals.append(loo_r2_ub)

        data[llm] = np.array(y_vals, dtype=float)
        data_lb[llm] = np.array(lb_vals, dtype=float)
        data_ub[llm] = np.array(ub_vals, dtype=float)


    x = np.arange(len(GREEDY_SPECS))

    fig, axes = plt.subplots(
        nrows=len(LLMS),
        ncols=1,
        figsize=(4.5, 1.1 * len(LLMS) + 0.8),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )

    # adjust wspace
    fig.subplots_adjust(wspace=0.05)

    if len(LLMS) == 1:
        axes = [axes]

    for ax, llm in zip(axes, LLMS):
        y = data[llm]
        y_lb = data_lb[llm]
        y_ub = data_ub[llm]
        color = llm_colors[llm]
        nc, nc_lb, nc_ub = extract_noise_ceiling_interval(noise_ceilings[llm])
        local_min = min(np.min(y_lb), nc_lb)
        local_max = max(np.max(y_ub), nc_ub)
        span = local_max - local_min
        pad = max(0.015, 0.14 * span if span > 0 else 0.05)

        ax.plot(x, y, linewidth=1.6, color=color, zorder=2)

        ax.fill_between(x, y_lb, y_ub, color=color, alpha=0.5, zorder=1)

        ax.scatter(x, y, s=28, color=color, edgecolor="white", linewidth=0.6, zorder=3)

       
        ax.axhspan(
        nc_lb,
        nc_ub,
        facecolor=color,
        alpha=0.25,
        edgecolor=color,
        linewidth=0.8,
        zorder=0,
    )

        ax.axhline(
            nc,
            color=color,
            linestyle="--",
            linewidth=1,
            alpha=1,
            zorder=1,
        )


        ax.text(
            0.01,
            1.025,
            DISPLAY_NAMES.get(llm, llm),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
        )

        ax.text(
            0.99,
            1.01,
            f"Explainable variance (noise ceiling) = {nc:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            fontweight="bold",
            color=color,
        )

        for xi, yi in zip(x, y):
            if yi<0.01:
                yi=0.0
                ax.annotate(
                f"{yi:.1f}",
                (xi, yi),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=6.5,
                color=color,
            )
            else:           
                ax.annotate(
                    fmt_value(yi),
                    (xi, yi),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    fontweight="bold",
                    color=color,
                )

        ax.grid(axis="y", linewidth=0.6, alpha=0.35)
        ax.set_ylim(local_min - pad, local_max + pad)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3, prune="both"))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(X_LABELS, rotation=55, ha="right")
    axes[-1].set_xlim(-0.35, len(x) - 0.2)
    # set fontsize of x/y tick labels
    for ax in axes:
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=10)

    fig.supylabel("Explained variance (LOO-R²)", fontsize=14)

    stem = (
        f"figure_7"
    )

    pdf_path = args.outdir / f"{stem}.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()