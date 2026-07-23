import os
import warnings
import argparse

import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()

# match inference.py / posterior_plot.py naming
parser.add_argument("--interaction", choices=["none", "static", "decay"], default="decay")
parser.add_argument("--topic_bias", choices=["none", "static", "decay"], default="decay")
parser.add_argument("--agree_bias", choices=["none", "static", "decay"], default="decay")
parser.add_argument("--anchor_bias", choices=["none", "static", "decay"], default="decay")

parser.add_argument("--epsilon", dest="epsilon", action="store_true")
parser.add_argument("--no_epsilon", dest="epsilon", action="store_false")
parser.set_defaults(epsilon=True)

parser.add_argument("--draws", type=int, default=2000)
parser.add_argument("--tune", type=int, default=1000)
parser.add_argument("--chains", type=int, default=4)
parser.add_argument("--cores", type=int, default=4)
parser.add_argument("--target_accept", type=float, default=0.9)
parser.add_argument("--max_treedepth", type=int, default=10)

parser.add_argument(
    "--base_dir",
    type=str,
    default="../../data/traces/full_model",
)
parser.add_argument(
    "--data_path",
    type=str,
    default=(
        "../../data/inference_ready_data_all_discussions_s=25_l=5.parquet"
    ),
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="../../figures",
)
parser.add_argument("--save_stem", type=str, default="effect_sizes")
parser.add_argument("--max_es_draws", type=int, default=800)
parser.add_argument("--hdi_prob", type=float, default=0.95)

# trajectory denominator:
# False = pooled SD(delta_x) within model across all times/topics
# True  = SD(delta_x) within each time step
parser.add_argument("--denom_per_time", type=bool, default=False)

args = parser.parse_args()


# ----------------------------------------------------------------------
# Plot style
# ----------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 450,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ----------------------------------------------------------------------
# Models / labels / colors
# ----------------------------------------------------------------------
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

violin_colors = {
    "Llama-3.1-8B-Instruct": "tab:blue",
    "Qwen2.5-7B-Instruct": "tab:purple",
    "dolphin-2.7-mixtral-8x7b-AWQ": "tab:green",
    "Mixtral-8x7B-Instruct-v0.1-AWQ-INT4": "tab:orange",
    "gpt-4o-mini": "tab:red",
    "grok-4-1-fast-non-reasoning": "tab:pink",
}
hdi_colors = violin_colors


# ----------------------------------------------------------------------
# Topic order: must match inference.py exactly
# ----------------------------------------------------------------------
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
topic_to_idx = {t: i for i, t in enumerate(topics)}
n_topics = len(topics)


# ----------------------------------------------------------------------
# Effect definitions
# ----------------------------------------------------------------------
effect_order = []
effect_titles = {}
effect_labels = {}

if args.interaction != "none":
    effect_order.append("interaction")
    effect_titles["interaction"] = "Interaction"
    effect_labels["interaction"] = r"$\bar{\alpha}_{\mathrm{int}}$"

if args.topic_bias != "none":
    effect_order.append("topic_bias")
    effect_titles["topic_bias"] = "Topic bias"
    effect_labels["topic_bias"] = r"$\bar{\beta}_{\mathrm{top}}$"

if args.agree_bias != "none":
    effect_order.append("agree_bias")
    effect_titles["agree_bias"] = "Agreement bias"
    effect_labels["agree_bias"] = r"$\bar{\beta}_{\mathrm{agr}}$"

if args.anchor_bias != "none":
    effect_order.append("anchor_bias")
    effect_titles["anchor_bias"] = "Anchoring bias"
    effect_labels["anchor_bias"] = r"$\bar{\beta}_{\mathrm{anc}}$"

if len(effect_order) == 0:
    raise ValueError("At least one of interaction/topic_bias/agree_bias/anchor_bias must be active.")


# ----------------------------------------------------------------------
# File naming
# ----------------------------------------------------------------------
def build_model_tag(cli_args):
    eps_tag = "eps" if cli_args.epsilon else "noeps"
    return (
        f"I-{cli_args.interaction}"
        f"__T-{cli_args.topic_bias}"
        f"__A-{cli_args.agree_bias}"
        f"__C-{cli_args.anchor_bias}"
        f"__E-{eps_tag}"
    )


model_tag = build_model_tag(args)
suffix = (
    f"_dr{args.draws}_tu{args.tune}_ch{args.chains}_co{args.cores}"
    f"_ar{args.target_accept}_td{args.max_treedepth}.nc"
)
trace_paths = {
    m: os.path.join(args.base_dir, f"{model_tag}_{m}{suffix}")
    for m in models
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
A_TARGET = 1.0


def thin_indices(n_samples, max_draws):
    if max_draws is None or max_draws >= n_samples:
        return np.arange(n_samples, dtype=np.int64)
    return np.linspace(0, n_samples - 1, max_draws).astype(np.int64)


def prep_arrays(df_sub):
    """
    Exact scaling from inference.py.

    Important:
    Effect sizes are computed in this scaled space. Because ES is standardized,
    no additional back-transformation of coefficients is required.
    """
    delta_x = df_sub["delta_x_i"].to_numpy(dtype=np.float64) / 4.0
    x_i = df_sub["x_i"].to_numpy(dtype=np.float64) / 2.0
    x_j = df_sub["x_j"].to_numpy(dtype=np.float64) / 2.0
    x_0 = df_sub["x_0"].to_numpy(dtype=np.float64) / 2.0
    t_scaled = (df_sub["t"].to_numpy(dtype=np.float64) - 1.0) / 4.0
    d_data = df_sub["framing"].to_numpy(dtype=np.float64)
    is_resp = df_sub["is_responder"].to_numpy(dtype=np.float64)
    return delta_x, x_i, x_j, x_0, t_scaled, d_data, is_resp


def summarize_draws(draws, hdi_prob=0.95):
    draws = np.asarray(draws, dtype=np.float64)
    med = float(np.median(draws))
    lo, hi = az.hdi(draws, hdi_prob=hdi_prob)
    return med, float(lo), float(hi)


def get_lambda_draws(post, mode, var_name, idx):
    if mode == "none":
        return None
    if mode == "static":
        return np.zeros(idx.size, dtype=np.float64)
    return post[var_name].values.reshape(-1)[idx]


def save_figure(fig, stem):
    os.makedirs(args.save_dir, exist_ok=True)
    pdf_path = os.path.join(args.save_dir, f"{stem}.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")


# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
df = pd.read_parquet(
    args.data_path,
    columns=["model", "topic", "delta_x_i", "x_i", "x_j", "t", "framing", "x_0", "is_responder"],
)
df["t"] = df["t"].astype(int)


# ----------------------------------------------------------------------
# Read posterior draws needed for ES computation
# ----------------------------------------------------------------------
posterior = {}
available_models = []

for m in models:
    path = trace_paths[m]
    if not os.path.exists(path):
        warnings.warn(f"Skipping missing trace: {path}")
        continue

    df_m = df[df["model"] == m].copy()
    if df_m.empty:
        warnings.warn(f"Skipping model with no data rows: {m}")
        continue

    idata = az.from_netcdf(path)
    post = idata.posterior

    n_samples = post.sizes["chain"] * post.sizes["draw"]
    idx = thin_indices(n_samples, args.max_es_draws)

    pdata = {
        "n_used": idx.size,
        "interaction_coef": None,
        "topic_coef": None,
        "agree_coef": None,
        "anchor_coef": None,
        "lambda_interact": get_lambda_draws(post, args.interaction, "lambda_interact", idx),
        "lambda_b_topic": get_lambda_draws(post, args.topic_bias, "lambda_b_topic", idx),
        "lambda_b_agree": get_lambda_draws(post, args.agree_bias, "lambda_b_agree", idx),
        "lambda_b_anchor": get_lambda_draws(post, args.anchor_bias, "lambda_b_anchor", idx),
        "b": None,
    }

    if args.interaction != "none":
        pdata["interaction_coef"] = post["alpha_mu"].values.reshape(-1)[idx]

    if args.topic_bias != "none":
        # beta_t_mu is already the correctly transformed deterministic parameter
        pdata["topic_coef"] = post["beta_t_mu"].values.reshape(-1)[idx]
        pdata["b"] = post["b"].values.reshape(-1, n_topics)[idx, :]

    if args.agree_bias != "none":
        pdata["agree_coef"] = post["beta_a_mu"].values.reshape(-1)[idx]

    if args.anchor_bias != "none":
        pdata["anchor_coef"] = post["beta_c_mu"].values.reshape(-1)[idx]

    posterior[m] = pdata
    available_models.append(m)

if len(available_models) == 0:
    raise RuntimeError("No trace files were found for the requested configuration.")


# ----------------------------------------------------------------------
# Compute ES trajectories and global ES
# ----------------------------------------------------------------------
trajectory_rows = []
global_effect_sizes = {eff: {} for eff in effect_order}

for m in available_models:
    df_m = df[df["model"] == m].copy()
    p = posterior[m]
    n_used = p["n_used"]

    delta_x_all, x_i_all, x_j_all, x_0_all, t_all, d_all, is_resp_all = prep_arrays(df_m)
    sd_delta_global = float(np.std(delta_x_all))
    if sd_delta_global == 0.0:
        raise ValueError(f"Global SD(delta_x)=0 for model={m}")

    base_int_all = (x_j_all - x_i_all)
    base_agr_all = (A_TARGET - x_i_all)
    base_anc_all = (x_0_all - x_i_all) * is_resp_all

    topic_idx_all = df_m["topic"].map(topic_to_idx).to_numpy(dtype=np.int64)
    if np.any(topic_idx_all < 0) or np.any(topic_idx_all >= n_topics):
        bad = df_m.loc[(topic_idx_all < 0) | (topic_idx_all >= n_topics), "topic"].unique()
        raise ValueError(f"Unmapped topics for model={m}: {bad}")

    # -------------------- global ES pooled across all times/topics
    if "interaction" in effect_order:
        sd_pred = np.empty(n_used, dtype=np.float64)
        lam = p["lambda_interact"]
        for r in range(n_used):
            w = np.exp(-lam[r] * t_all)
            sd_pred[r] = np.std(w * base_int_all)
        global_effect_sizes["interaction"][m] = p["interaction_coef"] * (sd_pred / sd_delta_global)

    if "topic_bias" in effect_order:
        sd_pred = np.empty(n_used, dtype=np.float64)
        lam = p["lambda_b_topic"]
        b_draw = p["b"]
        for r in range(n_used):
            w = np.exp(-lam[r] * t_all)
            b_per_obs = b_draw[r, topic_idx_all]
            base_top = d_all * b_per_obs - x_i_all
            sd_pred[r] = np.std(w * base_top)
        global_effect_sizes["topic_bias"][m] = p["topic_coef"] * (sd_pred / sd_delta_global)

    if "agree_bias" in effect_order:
        sd_pred = np.empty(n_used, dtype=np.float64)
        lam = p["lambda_b_agree"]
        for r in range(n_used):
            w = np.exp(-lam[r] * t_all)
            sd_pred[r] = np.std(w * base_agr_all)
        global_effect_sizes["agree_bias"][m] = p["agree_coef"] * (sd_pred / sd_delta_global)

    if "anchor_bias" in effect_order:
        sd_pred = np.empty(n_used, dtype=np.float64)
        lam = p["lambda_b_anchor"]
        for r in range(n_used):
            w = np.exp(-lam[r] * t_all)
            sd_pred[r] = np.std(w * base_anc_all)
        global_effect_sizes["anchor_bias"][m] = p["anchor_coef"] * (sd_pred / sd_delta_global)

    # -------------------- ES trajectories by discrete time step
    t_steps = np.sort(df_m["t"].unique())

    for t_step in t_steps:
        df_mt = df_m[df_m["t"] == int(t_step)].copy()
        if df_mt.empty:
            continue

        delta_x, x_i, x_j, x_0, t_scaled, d_data, is_resp = prep_arrays(df_mt)

        sd_den = float(np.std(delta_x)) if args.denom_per_time else sd_delta_global
        if sd_den == 0.0:
            continue

        t0 = float(t_scaled[0])

        base_int = (x_j - x_i)
        base_agr = (A_TARGET - x_i)
        base_anc = (x_0 - x_i) * is_resp

        if "interaction" in effect_order:
            sd_pred = np.exp(-p["lambda_interact"] * t0) * np.std(base_int)
            draws = p["interaction_coef"] * (sd_pred / sd_den)
            med, lo, hi = summarize_draws(draws, hdi_prob=args.hdi_prob)
            trajectory_rows.append({
                "model": m,
                "model_label": model_labels[m],
                "effect": "interaction",
                "t": int(t_step),
                "t_scaled": t0,
                "median": med,
                "hdi_low": lo,
                "hdi_high": hi,
            })

        if "topic_bias" in effect_order:
            topic_idx = df_mt["topic"].map(topic_to_idx).to_numpy(dtype=np.int64)
            if np.any(topic_idx < 0) or np.any(topic_idx >= n_topics):
                bad = df_mt.loc[(topic_idx < 0) | (topic_idx >= n_topics), "topic"].unique()
                raise ValueError(f"Unmapped topics at t={t_step} for model={m}: {bad}")

            sd_pred = np.empty(n_used, dtype=np.float64)
            exp_fac = np.exp(-p["lambda_b_topic"] * t0)
            b_draw = p["b"]
            for r in range(n_used):
                b_per_obs = b_draw[r, topic_idx]
                base_top = d_data * b_per_obs - x_i
                sd_pred[r] = exp_fac[r] * np.std(base_top)

            draws = p["topic_coef"] * (sd_pred / sd_den)
            med, lo, hi = summarize_draws(draws, hdi_prob=args.hdi_prob)
            trajectory_rows.append({
                "model": m,
                "model_label": model_labels[m],
                "effect": "topic_bias",
                "t": int(t_step),
                "t_scaled": t0,
                "median": med,
                "hdi_low": lo,
                "hdi_high": hi,
            })

        if "agree_bias" in effect_order:
            sd_pred = np.exp(-p["lambda_b_agree"] * t0) * np.std(base_agr)
            draws = p["agree_coef"] * (sd_pred / sd_den)
            med, lo, hi = summarize_draws(draws, hdi_prob=args.hdi_prob)
            trajectory_rows.append({
                "model": m,
                "model_label": model_labels[m],
                "effect": "agree_bias",
                "t": int(t_step),
                "t_scaled": t0,
                "median": med,
                "hdi_low": lo,
                "hdi_high": hi,
            })

        if "anchor_bias" in effect_order:
            sd_pred = np.exp(-p["lambda_b_anchor"] * t0) * np.std(base_anc)
            draws = p["anchor_coef"] * (sd_pred / sd_den)
            med, lo, hi = summarize_draws(draws, hdi_prob=args.hdi_prob)
            trajectory_rows.append({
                "model": m,
                "model_label": model_labels[m],
                "effect": "anchor_bias",
                "t": int(t_step),
                "t_scaled": t0,
                "median": med,
                "hdi_low": lo,
                "hdi_high": hi,
            })

traj = pd.DataFrame(trajectory_rows)
if traj.empty:
    raise RuntimeError("No trajectory data computed.")


# ----------------------------------------------------------------------
# Combined plot: one row per effect
# left = global ES, right = ES trajectories
# ----------------------------------------------------------------------
panel_order = ["interaction", "topic_bias", "agree_bias", "anchor_bias"]
params = [eff for eff in panel_order if eff in effect_order]
n_rows = len(params)

fig, axes = plt.subplots(
    n_rows,
    2,
    figsize=(5, 4),
    sharex="col",
    gridspec_kw={"width_ratios": [2, 1.75]},
)

# increase space between columns

axes = np.atleast_2d(axes)

x_vals = np.sort(traj["t"].unique())

offset_values = np.linspace(0.32, -0.32, len(available_models))
offsets = dict(zip(available_models, offset_values))

# global x-limits
all_global_draws = [
    np.asarray(global_effect_sizes[eff][m], dtype=np.float64)
    for eff in params
    for m in available_models
    if global_effect_sizes[eff].get(m, None) is not None
]
if len(all_global_draws) == 0:
    raise RuntimeError("No global ES draws were computed.")

all_global_draws = np.concatenate(all_global_draws)
q_lo, q_hi = np.nanpercentile(all_global_draws, [1.0, 99.0])
pad = 0.12 * (q_hi - q_lo + 1e-12)
x_min = min(q_lo - pad, 0.0)
x_max = max(q_hi + pad, 0.0)

# common y-limits for trajectory panels
traj_lo = float(traj["hdi_low"].min())
traj_hi = float(traj["hdi_high"].max())
traj_pad = 0.08 * max(traj_hi - traj_lo, 1e-12)
# traj_ymin = min(traj_lo - traj_pad, 0.0)
# traj_ymax = max(traj_hi + traj_pad, 0.0)
traj_ymin = traj_lo 
traj_ymax = traj_hi

for row_idx, eff in enumerate(params):
    axg, axt = axes[row_idx, 0], axes[row_idx, 1]
    row_bg = "gainsboro" if (row_idx % 2 == 0) else "whitesmoke"

    # -------------------- left: global ES
    axg.set_facecolor(row_bg)

    for m in available_models:
        draws = global_effect_sizes[eff].get(m, None)
        if draws is None:
            continue

        draws = np.asarray(draws, dtype=np.float64)
        y_pos = offsets[m]

        parts = axg.violinplot(
            draws,
            positions=[y_pos],
            vert=False,
            widths=0.10,
            showmeans=False,
            showextrema=False,
            showmedians=False,
        )
        body = parts["bodies"][0]
        body.set_facecolor(violin_colors[m])
        body.set_edgecolor("none")
        body.set_alpha(0.32)

        hdi_low, hdi_high = az.hdi(draws, hdi_prob=args.hdi_prob)
        med = float(np.median(draws))

        axg.plot([hdi_low, hdi_high], [y_pos, y_pos], color=hdi_colors[m], lw=2.0)
        axg.plot(med, y_pos, marker="o", color=hdi_colors[m], markersize=3.8)

    axg.axvline(0.0, linestyle="--", color="black", lw=1.0, zorder=-2)
    axg.set_xlim(x_min, x_max)
    axg.set_ylim(-0.42, 0.42)
    axg.xaxis.set_major_locator(MaxNLocator(4))
    axg.set_yticks([])
    #axg.grid(axis="x", linestyle=":", color="gray", alpha=0.9, zorder=-2)
    axg.tick_params(axis="y", length=0)
    axg.tick_params(axis="x", length=4)
    axg.set_ylabel(effect_labels[eff], va="center", rotation=0, labelpad=20)

    # -------------------- right: trajectories
    sub = traj[traj["effect"] == eff].copy()
    axt.set_facecolor(row_bg)

    for i, xv in enumerate(x_vals):
        axt.axvspan(
            xv - 0.45,
            xv + 0.55,
            color=("white" if i % 2 else "lightgray"),
            alpha=0.22,
            zorder=-100,
        )

    for k, m in enumerate(available_models):
        subm = sub[sub["model"] == m].sort_values("t")
        if subm.empty:
            continue

        x = subm["t"].to_numpy()
        med = subm["median"].to_numpy()
        lo = subm["hdi_low"].to_numpy()
        hi = subm["hdi_high"].to_numpy()
        c = violin_colors[m]

        axt.fill_between(x, lo, hi, color=c, alpha=0.16, linewidth=0)
        axt.plot(
            x,
            med,
            color=c,
            lw=2,
            alpha=1,
            marker="o",
            markersize=4.2
        )

    axt.axhline(0.0, color="black", lw=1.0, linestyle="--", zorder=-10)
    axt.set_xlim(x_vals.min() - 0.5, x_vals.max() + 0.5)
    axt.set_ylim(traj_ymin, 1)
    axt.set_xticks(x_vals)
    axt.yaxis.set_major_locator(MaxNLocator(5))
    axt.tick_params(axis="both", length=4)

    if row_idx < n_rows - 1:
        axg.tick_params(labelbottom=False)
        axt.tick_params(labelbottom=False)

axes[0, 0].set_title("Time-averaged", loc="left", pad=6)
axes[0, 1].set_title("Time-dependent", loc="left", pad=6)

axes[0, 1].set_ylabel("Standardized effect size",y=-1.3,x=10, rotation=90, labelpad=0)
axes[-1, 0].set_xlabel("Standardized effect size")
axes[-1, 1].set_xlabel("Discussion round")

axes[0, 0].text(-0.225, 1.125, "a", transform=axes[0, 0].transAxes, fontsize=15, fontweight="bold")
axes[0, 1].text(-0.225, 1.125, "b", transform=axes[0, 1].transAxes, fontsize=15, fontweight="bold")



handles = [
    Patch(facecolor=violin_colors[m], edgecolor="none", alpha=0.4, label=model_labels[m])
    for m in available_models
]
fig.legend(
    handles=handles,
    frameon=False,
    ncols=2,
    loc="upper center",
    bbox_to_anchor=(0.53, 1.15),
)

fig.subplots_adjust(top=0.86, left=0.15, right=0.98, hspace=0.18, wspace=0.5)
combined_stem = (
    "figure_5"
)
save_figure(fig, combined_stem)

plt.show()