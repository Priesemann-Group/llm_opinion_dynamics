import os
import warnings
import argparse

import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--draws", type=int, default=2000)
parser.add_argument("--tune", type=int, default=1000)
parser.add_argument("--chains", type=int, default=4)
parser.add_argument("--cores", type=int, default=4)
parser.add_argument("--target_accept", type=float, default=0.9)
parser.add_argument("--max_treedepth", type=int, default=10)

parser.add_argument(
    "--trace_dir",
    type=str,
    default="../../data/traces/idv_model",
)
parser.add_argument(
    "--parquet_path",
    type=str,
    default=(
        "../../data/inference_ready_data_all_discussions_s=25_l=5.parquet"
    ),
)
parser.add_argument(
    "--mixtral_csv_path",
    type=str,
    default="../../data/mixtral_tuned_clim_data.csv",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="../../figures",
)
parser.add_argument(
    "--save_stem",
    type=str,
    default="figure_8",
)
parser.add_argument("--max_es_draws", type=int, default=800)
parser.add_argument("--hdi_prob", type=float, default=0.95)

args = parser.parse_args()


# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 450,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ----------------------------------------------------------------------
# Constants / model metadata
# ----------------------------------------------------------------------
CLIMATE_TOPIC = "Climate_Change"
A_TARGET = 1.0
INIT_LEVELS = np.array([-2, -1, 0, 1, 2], dtype=np.float64)

models = [
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct",
    "dolphin-2.7-mixtral-8x7b-AWQ",
    "Mixtral-8x7B-Instruct-v0.1-AWQ-INT4",
    "gpt-4o-mini",
    "mixtral-finetuned",
]

model_labels = {
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "dolphin-2.7-mixtral-8x7b-AWQ": "DolphinMixtral-8x7B",
    "Mixtral-8x7B-Instruct-v0.1-AWQ-INT4": "Mixtral-8x7B",
    "gpt-4o-mini": "GPT-4o-mini",
    "mixtral-finetuned": "Custom fine-tuned Mixtral",
}

colors = {
    "Llama-3.1-8B-Instruct": "tab:blue",
    "Qwen2.5-7B-Instruct": "tab:purple",
    "dolphin-2.7-mixtral-8x7b-AWQ": "tab:green",
    "Mixtral-8x7B-Instruct-v0.1-AWQ-INT4": "tab:orange",
    "gpt-4o-mini": "tab:red",
    "mixtral-finetuned": "tab:brown",
}

markers = {
    "Llama-3.1-8B-Instruct": "o",
    "Qwen2.5-7B-Instruct": "s",
    "dolphin-2.7-mixtral-8x7b-AWQ": "^",
    "Mixtral-8x7B-Instruct-v0.1-AWQ-INT4": "D",
    "gpt-4o-mini": "P",
    "mixtral-finetuned": "v",
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def build_model_tag():
    return "climate_idv_attractors_full_decay_eps"


def build_trace_path(llm_name: str) -> str:
    suffix = (
        f"_dr{args.draws}_tu{args.tune}_ch{args.chains}_co{args.cores}"
        f"_ar{args.target_accept}_td{args.max_treedepth}.nc"
    )
    fname = f"{build_model_tag()}_{llm_name}{suffix}"
    return os.path.join(args.trace_dir, fname)


def thin_indices(n_samples: int, max_draws: int) -> np.ndarray:
    if max_draws is None or max_draws >= n_samples:
        return np.arange(n_samples, dtype=np.int64)
    return np.linspace(0, n_samples - 1, max_draws).astype(np.int64)


def summarize_draws(draws: np.ndarray, hdi_prob: float = 0.95):
    draws = np.asarray(draws, dtype=np.float64)
    med = float(np.median(draws))
    lo, hi = az.hdi(draws, hdi_prob=hdi_prob)
    return med, float(lo), float(hi)


def save_figure(fig, stem: str):
    os.makedirs(args.save_dir, exist_ok=True)
    pdf_path = os.path.join(args.save_dir, f"{stem}.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")


def reconstruct_x0_mixtral(df: pd.DataFrame):
    """
    Reconstruct x_0 exactly in the spirit of inference_idv.py:
    one constant anchor per discussion, equal to the first opinion of the round,
    copied into every row of that discussion.

    Assumes:
    - df['t'] starts at 0 in the old CSV and is shifted by +1 for consistency.
    - rows of a discussion are consecutive.
    """
    t_shifted = df["t"].to_numpy(dtype=np.float64) + 1.0
    x_j = df["x_j"].to_numpy(dtype=np.float64)

    x_0 = np.empty(len(df), dtype=np.float64)
    current_x0 = np.nan

    for i in range(len(df)):
        if t_shifted[i] == 1 and df["is_initiator"].iloc[i] == 1:
            current_x0 = x_i[i]   # first opinion of the discussion
        x_0[i] = current_x0

    if np.isnan(x_0).any():
        raise ValueError("Failed to reconstruct x_0 for mixtral-finetuned.")

    return x_0, t_shifted


def load_model_df(llm_name: str) -> pd.DataFrame:
    if llm_name == "mixtral-finetuned":
        df = pd.read_csv(args.mixtral_csv_path).copy()
        if df.empty:
            raise ValueError(f"No rows found in {args.mixtral_csv_path}")

        x_0, t_shifted = reconstruct_x0_mixtral(df)
        df["x_0"] = x_0
        df["t_shifted"] = t_shifted
        df["is_responder"] = (df["is_initiator"].to_numpy(dtype=np.float64) != 1).astype(np.float64)
        return df

    df = pd.read_parquet(
        args.parquet_path,
        columns=[
            "model", "topic", "delta_x_i", "x_i", "x_j", "H_i", "t", "framing",
            "x_0", "is_responder", "init_x_i"
        ],
    )
    df = df[(df["model"] == llm_name) & (df["topic"] == CLIMATE_TOPIC)].copy()
    if df.empty:
        raise ValueError(f"No rows found for model={llm_name} and topic={CLIMATE_TOPIC}.")
    df["t_shifted"] = df["t"].to_numpy(dtype=np.float64)  # already starts at 1
    return df


def prep_arrays_for_idv_model(df: pd.DataFrame, llm_name: str):
    """
    Exact scaling used in inference_idv.py.
    """
    if llm_name == "mixtral-finetuned":
        delta_x = df["dx"].to_numpy(dtype=np.float64) / 4.0
        x_i = df["x_i"].to_numpy(dtype=np.float64) / 2.0
        x_j = df["x_j"].to_numpy(dtype=np.float64) / 2.0
        x_0 = df["x_0"].to_numpy(dtype=np.float64) / 2.0
        t_scaled = (df["t_shifted"].to_numpy(dtype=np.float64) - 1.0) / 4.0
        d_data = df["delta"].to_numpy(dtype=np.float64)
        is_responder = df["is_responder"].to_numpy(dtype=np.float64)
        init_x_i = df["init_x_i"].to_numpy(dtype=np.float64)
    else:
        delta_x = df["delta_x_i"].to_numpy(dtype=np.float64) / 4.0
        x_i = df["x_i"].to_numpy(dtype=np.float64) / 2.0
        x_j = df["x_j"].to_numpy(dtype=np.float64) / 2.0
        x_0 = df["x_0"].to_numpy(dtype=np.float64) / 2.0
        t_scaled = (df["t_shifted"].to_numpy(dtype=np.float64) - 1.0) / 4.0
        d_data = df["framing"].to_numpy(dtype=np.float64)
        is_responder = df["is_responder"].to_numpy(dtype=np.float64)
        init_x_i = df["init_x_i"].to_numpy(dtype=np.float64)

    init_idx = np.searchsorted(INIT_LEVELS, init_x_i)
    valid = (
        (init_idx >= 0)
        & (init_idx < len(INIT_LEVELS))
        & np.isclose(INIT_LEVELS[init_idx], init_x_i)
    )
    if not np.all(valid):
        bad = np.unique(init_x_i[~valid])
        raise ValueError(f"Invalid init_x_i values for {llm_name}: {bad}")

    return delta_x, x_i, x_j, x_0, t_scaled, d_data, is_responder, init_idx


# ----------------------------------------------------------------------
# Collect posterior summaries and global ES draws
# ----------------------------------------------------------------------
attractor_summary = {}
global_effect_sizes = {
    "interaction": {},
    "topic_bias": {},
    "agree_bias": {},
    "anchor_bias": {},
}
available_models = []

for m in models:
    trace_path = build_trace_path(m)
    if not os.path.exists(trace_path):
        warnings.warn(f"Skipping missing trace: {trace_path}")
        continue

    try:
        df_m = load_model_df(m)
    except Exception as e:
        warnings.warn(f"Skipping {m}: {e}")
        continue

    delta_x, x_i, x_j, x_0, t_scaled, d_data, is_responder, init_idx = prep_arrays_for_idv_model(df_m, m)

    sd_delta = float(np.std(delta_x))
    if sd_delta == 0.0:
        warnings.warn(f"Skipping {m}: SD(delta_x)=0.")
        continue

    idata = az.from_netcdf(trace_path)
    post = idata.posterior

    # full b posterior for left panel
    b_full = post["b"].values.reshape(-1, 5)
    b_stats = np.zeros((5, 3), dtype=np.float64)  # median, low, high
    for k in range(5):
        med, lo, hi = summarize_draws(b_full[:, k], hdi_prob=args.hdi_prob)
        b_stats[k, 0] = med
        b_stats[k, 1] = lo
        b_stats[k, 2] = hi
    attractor_summary[m] = b_stats

    # thinned draws for ES
    n_samples = post.sizes["chain"] * post.sizes["draw"]
    idx = thin_indices(n_samples, args.max_es_draws)

    alpha = post["alpha"].values.reshape(-1)[idx]
    beta_t = post["beta_t"].values.reshape(-1)[idx]
    beta_a = post["beta_a"].values.reshape(-1)[idx]
    beta_c = post["beta_c"].values.reshape(-1)[idx]

    lambda_interact = post["lambda_interact"].values.reshape(-1)[idx]
    lambda_b_topic = post["lambda_b_topic"].values.reshape(-1)[idx]
    lambda_b_agree = post["lambda_b_agree"].values.reshape(-1)[idx]
    lambda_b_anchor = post["lambda_b_anchor"].values.reshape(-1)[idx]
    b_draws = post["b"].values.reshape(-1, 5)[idx, :]

    n_used = len(idx)

    base_int = x_j - x_i
    base_agr = A_TARGET - x_i
    base_anc = (x_0 - x_i) * is_responder

    sd_pred = np.empty(n_used, dtype=np.float64)
    for r in range(n_used):
        sd_pred[r] = np.std(np.exp(-lambda_interact[r] * t_scaled) * base_int)
    global_effect_sizes["interaction"][m] = alpha * (sd_pred / sd_delta)

    sd_pred = np.empty(n_used, dtype=np.float64)
    for r in range(n_used):
        b_per_obs = b_draws[r, init_idx]
        base_top = d_data * b_per_obs - x_i
        sd_pred[r] = np.std(np.exp(-lambda_b_topic[r] * t_scaled) * base_top)
    global_effect_sizes["topic_bias"][m] = beta_t * (sd_pred / sd_delta)

    sd_pred = np.empty(n_used, dtype=np.float64)
    for r in range(n_used):
        sd_pred[r] = np.std(np.exp(-lambda_b_agree[r] * t_scaled) * base_agr)
    global_effect_sizes["agree_bias"][m] = beta_a * (sd_pred / sd_delta)

    sd_pred = np.empty(n_used, dtype=np.float64)
    for r in range(n_used):
        sd_pred[r] = np.std(np.exp(-lambda_b_anchor[r] * t_scaled) * base_anc)
    global_effect_sizes["anchor_bias"][m] = beta_c * (sd_pred / sd_delta)

    available_models.append(m)

if len(available_models) == 0:
    raise RuntimeError("No models available for plotting.")


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(8, 2.5),
    gridspec_kw={"width_ratios": [1.05, 1.2]},
)
fig.subplots_adjust(wspace=0.4, top=0.78)

# ===================================================
# 1) Individual attractors
# ===================================================
init_ops = INIT_LEVELS.copy()
offsets = np.linspace(-0.30, 0.30, len(available_models))

for i, m in enumerate(available_models):
    stats = attractor_summary[m]
    med = stats[:, 0]
    lo = stats[:, 1]
    hi = stats[:, 2]

    yerr = np.vstack([med - lo, hi - med])

    ax1.errorbar(
        init_ops + offsets[i],
        med,
        yerr=yerr,
        fmt=markers[m],
        ecolor=colors[m],
        mec=colors[m],
        mfc=colors[m],
        color=colors[m],
        capsize=3,
        elinewidth=1.0,
        capthick=1.0,
        markersize=5.0,
        alpha=0.95,
        label=model_labels[m],
        zorder= 20 if m in ["Mixtral-8x7B-Instruct-v0.1-AWQ-INT4", "mixtral-finetuned"] else 10 - i,
    )

    if m in ["Mixtral-8x7B-Instruct-v0.1-AWQ-INT4", "mixtral-finetuned"]:
        ax1.plot(
            init_ops + offsets[i],
            med,
            linestyle=":",
            linewidth=1.5,
            color=colors[m],
            alpha=0.9,
            zorder=20,
        )


for i in range(len(init_ops)):
    rect = Rectangle(
        (init_ops[i] - 0.45, -1.15),
        0.85,
        2.3,
        facecolor="gainsboro" if i % 2 == 0 else "whitesmoke",
        edgecolor="none",
        zorder=-100,
        alpha=0.55,
    )
    ax1.add_patch(rect)

ax1.set_xlabel("Initial opinion")
ax1.set_ylabel("Individual\ntopic attractor\n$b_{\mathrm{climate}}$", multialignment="center")
ax1.set_xticks(init_ops)
ax1.set_xlim(-2.55, 2.55)
ax1.set_ylim(-0.55, 1.05)
ax1.set_yticks([-0.5, 0, 0.5, 1])
ax1.set_yticklabels([-1, 0, 1, 2])
ax1.axhline(0.0, color="black", lw=1.0, linestyle="--", zorder=-10)
ax1.tick_params(axis="both", length=4)

# ===================================================
# 2) Global effect sizes
# ===================================================
params = ["interaction", "topic_bias", "agree_bias", "anchor_bias"]
param_labels = [
    r"$\alpha_{\mathrm{interact}}$",
    r"$\beta_{\mathrm{topic}}$",
    r"$\beta_{\mathrm{agree}}$",
    r"$\beta_{\mathrm{anchor}}$",
]

base_y = np.arange(len(params))[::-1] * 1.25
offset_vals = np.linspace(0.36, -0.36, len(available_models))
offset_map = dict(zip(available_models, offset_vals))

all_draws_for_xlim = []

for p_idx, (param, param_label) in enumerate(zip(params, param_labels)):
    rect = Rectangle(
        (-10, base_y[p_idx] - 0.56),
        20,
        1.12,
        facecolor="gainsboro" if (p_idx % 2 == 0) else "whitesmoke",
        edgecolor="none",
        zorder=-100,
        alpha=0.55,
    )
    ax2.add_patch(rect)

    for m in available_models:
        draws = np.asarray(global_effect_sizes[param][m], dtype=np.float64)
        all_draws_for_xlim.append(draws)

        y_pos = base_y[p_idx] + offset_map[m]

        parts = ax2.violinplot(
            draws,
            positions=[y_pos],
            vert=False,
            widths=0.3,
            showmeans=False,
            showextrema=False,
            showmedians=False,
        )
        body = parts["bodies"][0]
        body.set_facecolor(colors[m])
        body.set_edgecolor("none")
        body.set_alpha(0.32)

        hdi_low, hdi_high = az.hdi(draws, hdi_prob=args.hdi_prob)
        med = float(np.median(draws))

        ax2.plot([hdi_low, hdi_high], [y_pos, y_pos], color=colors[m], lw=2.0)
        ax2.plot(med, y_pos, marker="o", color=colors[m], markersize=3.5)

all_draws_for_xlim = np.concatenate(all_draws_for_xlim)
q_lo, q_hi = np.nanpercentile(all_draws_for_xlim, [1.0, 99.0])
pad = 0.12 * (q_hi - q_lo + 1e-12)
x_min = min(q_lo - pad, 0.0)
x_max = max(q_hi + pad, 0.0)

ax2.axvline(0.0, linestyle="--", color="black", lw=1.0, zorder=-2)
ax2.set_xlim(x_min, x_max)
ax2.set_yticks(base_y)
ax2.tick_params(axis="y", length=0)
ax2.set_yticklabels(param_labels)
ax2.set_xlabel("Standardized effect size (global)")
ax2.tick_params(axis="y", length=0)
ax2.tick_params(axis="x", length=4)

# ---------------------------------------------------
# Shared legend
# ---------------------------------------------------
handles = [
    Patch(facecolor=colors[m], edgecolor="none", alpha=0.5, label=model_labels[m])
    for m in available_models
]
fig.legend(
    handles=handles,
    frameon=False,
    ncols=3,
    loc="upper center",
    bbox_to_anchor=(0.45, 1.08),
)

# panel letters
ax1.text(-0.45, 1., "a", transform=ax1.transAxes, fontsize=15, fontweight="bold")
ax2.text(-0.27, 1, "b", transform=ax2.transAxes, fontsize=15, fontweight="bold")

# ---------------------------------------------------
# Save / show
# ---------------------------------------------------
stem = (
    "figure_8"
)
save_figure(fig, stem)
plt.show()