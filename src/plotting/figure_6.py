import os
import math
import argparse
import warnings

import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.legend_handler import HandlerBase
from scipy import stats
from scipy.stats import gaussian_kde, rv_discrete


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()

# match inference.py naming
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
    default="../../data/traces/full_model"
)

parser.add_argument(
    "--save_dir",
    type=str,
    default="../../figures",
)
parser.add_argument("--save_name", type=str, default=None)

# broken-axis settings for tau_agree panel
parser.add_argument("--tau_break_lo", type=float, default=1.0)
parser.add_argument("--tau_break_hi", type=float, default=40.0)

args = parser.parse_args()


# ----------------------------------------------------------------------
# Models / labels
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
# Topic order
# ----------------------------------------------------------------------
# inference.py order
topics_inference = [
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
n_topics = len(topics_inference)

# display order preserved from your old script
topic_display_idx = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
topic_display_titles = [topics_inference[i].replace("_", " ") for i in topic_display_idx]
topic_param_names = [f"b{i}" for i in topic_display_idx]
topic_param_labels = [rf"$b_{{{i+1}}}$" for i in topic_display_idx]
topic_param_set = set(topic_param_names)


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
# Rescaling
# ----------------------------------------------------------------------
# inference.py uses:
#   delta_x_scaled = delta_x / 4
#   x_scaled       = x / 2
#   x0_scaled      = x0 / 2
#   t_scaled       = (t - 1) / 4
#
# Therefore:
# - coefficients multiplying (target - x_i) or (x_j - x_i) convert as 1/2
# - b lives on x-scale, so convert as *2
# - tau in original t-units is 4 / lambda
COEF_TO_ORIGINAL = 1 / 2.0
POSITION_TO_ORIGINAL = 2.0
TAU_TO_ORIGINAL = 4.0

# kept for possible later extension
SIGMA_TO_ORIGINAL = 1 / 4.0


# ----------------------------------------------------------------------
# Priors
# ----------------------------------------------------------------------
param_SD = 0.1
log_beta_mu_MU = 0.0
log_beta_mu_SD = 0.5

_rng = np.random.default_rng(123)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def prior_pdf_coef_normal(x):
    return stats.norm.pdf(x, loc=0.0, scale=param_SD * COEF_TO_ORIGINAL)


def prior_pdf_tau(x):
    return stats.lognorm.pdf(x, s=1.0, scale=TAU_TO_ORIGINAL)


def prior_pdf_b(x):
    return stats.uniform.pdf(
        x,
        loc=-1.0 * POSITION_TO_ORIGINAL,
        scale=2.0 * POSITION_TO_ORIGINAL,
    )


def make_beta_t_mu_kde(n=150_000):
    z = _rng.normal(loc=log_beta_mu_MU, scale=log_beta_mu_SD, size=n)
    beta_scaled = softplus(4.0 * z) / 4.0
    beta_orig = COEF_TO_ORIGINAL * beta_scaled
    return gaussian_kde(beta_orig)


kde_beta_t_mu = make_beta_t_mu_kde()


def prior_pdf_beta_t_mu(x):
    return kde_beta_t_mu(x)


prior_pdfs = {
    "alpha_mu": prior_pdf_coef_normal,
    "beta_t_mu": prior_pdf_beta_t_mu,
    "beta_a_mu": prior_pdf_coef_normal,
    "beta_c_mu": prior_pdf_coef_normal,
    "lambda_interact": prior_pdf_tau,
    "lambda_b_topic": prior_pdf_tau,
    "lambda_b_agree": prior_pdf_tau,
    "lambda_b_anchor": prior_pdf_tau,
    **{f"b{i}": prior_pdf_b for i in range(n_topics)},
}


class HandlerPrior(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        x = np.linspace(xdescent, xdescent + width, 100)
        t = np.linspace(-2.5, 2.5, 100)

        bell = np.exp(-0.5 * t**2)
        bell /= bell.max()

        y0 = ydescent + 0.12 * height
        y = y0 + 0.76 * height * bell

        verts = [(x[0], y0)] + list(zip(x, y)) + [(x[-1], y0)] + [(x[0], y0)]
        codes = [Path.MOVETO] + [Path.LINETO] * len(x) + [Path.LINETO, Path.CLOSEPOLY]

        fill = PathPatch(
            Path(verts, codes),
            facecolor="grey",
            edgecolor="none",
            alpha=0.15,
            transform=trans,
        )
        line = Line2D(
            x, y,
            color="grey",
            linestyle=":",
            linewidth=0.8,
            transform=trans,
        )
        return [fill, line]


class HandlerPosterior(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        xmid = xdescent + 0.5 * width
        y = np.linspace(ydescent + 0.02 * height, ydescent + 0.98 * height, 100)
        t = np.linspace(-2.5, 2.5, 100)

        prof = np.exp(-0.5 * t**2)
        prof /= prof.max()
        halfw = 0.6 * width * prof

        x_left = xmid - halfw
        x_right = xmid + halfw

        verts = (
            list(zip(x_left, y))
            + list(zip(x_right[::-1], y[::-1]))
            + [(x_left[0], y[0])]
        )
        codes = (
            [Path.MOVETO]
            + [Path.LINETO] * (len(y) - 1)
            + [Path.LINETO] * len(y)
            + [Path.CLOSEPOLY]
        )

        violin = PathPatch(
            Path(verts, codes),
            facecolor="0",
            edgecolor="none",
            alpha=0.3,
            transform=trans,
        )

        ymid = ydescent + 0.5 * height
        hdi = Line2D(
            [xdescent + 0.18 * width, xdescent + 0.82 * width],
            [ymid, ymid],
            color="0.2",
            linewidth=1.6,
            transform=trans,
        )
        median = Line2D(
            [xmid], [ymid],
            marker="o",
            markersize=4,
            color="0.2",
            linestyle="None",
            transform=trans,
        )

        return [violin, hdi, median]
    

def add_group_ylabel(fig_, axes_grid, row_start, row_end, text, x_pad=0.065, **kwargs):
    row_axes = [
        axes_grid[r, c]
        for r in range(row_start, row_end + 1)
        for c in range(axes_grid.shape[1])
    ]
    bboxes = [ax.get_position().frozen() for ax in row_axes]

    x0 = min(bb.x0 for bb in bboxes)
    y0 = min(bb.y0 for bb in bboxes)
    y1 = max(bb.y1 for bb in bboxes)

    fig_.text(
        x0 - x_pad,
        0.5 * (y0 + y1),
        text,
        rotation=90,
        va="center",
        ha="center",
        fontsize=18,
        **kwargs,
    )


# ----------------------------------------------------------------------
# Parameter blocks to plot
# ----------------------------------------------------------------------
strength_specs = []
tau_specs = []
topic_specs = []

if args.interaction != "none":
    strength_specs.append(
        dict(name="alpha_mu", title="Interaction strength", label=r"$\bar{\alpha}_{\mathrm{int}}$")
    )
if args.topic_bias != "none":
    strength_specs.append(
        dict(name="beta_t_mu", title="Topic bias strength", label=r"$\bar{\beta}_{\mathrm{top}}$")
    )
if args.agree_bias != "none":
    strength_specs.append(
        dict(name="beta_a_mu", title="Agreement bias strength", label=r"$\bar{\beta}_{\mathrm{agr}}$")
    )
if args.anchor_bias != "none":
    strength_specs.append(
        dict(name="beta_c_mu", title="Anchoring bias strength", label=r"$\bar{\beta}_{\mathrm{anc}}$")
    )

if args.interaction == "decay":
    tau_specs.append(
        dict(
            name="lambda_interact",
            title="Interaction\ndecay timescale",
            label=r"$\tau_{\mathrm{int}}$",
        )
    )
if args.topic_bias == "decay":
    tau_specs.append(
        dict(
            name="lambda_b_topic",
            title="Topic bias\ndecay timescale",
            label=r"$\tau_{\mathrm{top}}$",
        )
    )
if args.agree_bias == "decay":
    tau_specs.append(
        dict(
            name="lambda_b_agree",
            title="Agreement bias\ndecay timescale",
            label=r"$\tau_{\mathrm{agr}}$",
        )
    )
if args.anchor_bias == "decay":
    tau_specs.append(
        dict(
            name="lambda_b_anchor",
            title="Anchoring bias\ndecay timescale",
            label=r"$\tau_{\mathrm{anc}}$",
        )
    )

if args.topic_bias != "none":
    for idx, title, lab in zip(topic_display_idx, topic_display_titles, topic_param_labels):
        topic_specs.append(
            dict(name=f"b{idx}", title=title, label=lab, topic_index=idx)
        )

all_specs = strength_specs + tau_specs + topic_specs


# ----------------------------------------------------------------------
# Read posterior samples
# ----------------------------------------------------------------------
available_models = []
posterior = {}
beta_t_topic_median = {}

for m in models:
    path = trace_paths[m]
    if not os.path.exists(path):
        warnings.warn(f"Skipping missing trace: {path}")
        continue

    idata = az.from_netcdf(path)
    post = idata.posterior
    pm = {}

    if "alpha_mu" in post:
        pm["alpha_mu"] = post["alpha_mu"].values.ravel() * COEF_TO_ORIGINAL
    if "beta_t_mu" in post:
        pm["beta_t_mu"] = post["beta_t_mu"].values.ravel() * COEF_TO_ORIGINAL
    if "beta_a_mu" in post:
        pm["beta_a_mu"] = post["beta_a_mu"].values.ravel() * COEF_TO_ORIGINAL
    if "beta_c_mu" in post:
        pm["beta_c_mu"] = post["beta_c_mu"].values.ravel() * COEF_TO_ORIGINAL

    if "lambda_interact" in post:
        pm["lambda_interact"] = TAU_TO_ORIGINAL / post["lambda_interact"].values.ravel()
    if "lambda_b_topic" in post:
        pm["lambda_b_topic"] = TAU_TO_ORIGINAL / post["lambda_b_topic"].values.ravel()
    if "lambda_b_agree" in post:
        pm["lambda_b_agree"] = TAU_TO_ORIGINAL / post["lambda_b_agree"].values.ravel()
    if "lambda_b_anchor" in post:
        pm["lambda_b_anchor"] = TAU_TO_ORIGINAL / post["lambda_b_anchor"].values.ravel()

    if "b" in post:
        b_post = post["b"].values.reshape(-1, n_topics) * POSITION_TO_ORIGINAL
        for bi in range(n_topics):
            pm[f"b{bi}"] = b_post[:, bi]

    if "beta_t" in post:
        beta_t_post = post["beta_t"].values.reshape(-1, n_topics) * COEF_TO_ORIGINAL
        for bi in range(n_topics):
            beta_t_topic_median[(m, bi)] = float(np.median(beta_t_post[:, bi]))

    posterior[m] = pm
    available_models.append(m)

if len(available_models) == 0:
    raise RuntimeError("No trace files were found for the requested configuration.")


# ----------------------------------------------------------------------
# Marker sizes based on topic-specific beta_t
# ----------------------------------------------------------------------
if len(beta_t_topic_median) > 0:
    all_bt_meds = np.fromiter(beta_t_topic_median.values(), dtype=float)
    bt_min = float(all_bt_meds.min())
    bt_max = float(all_bt_meds.max())
else:
    bt_min = 0.0
    bt_max = 1.0

ms_min, ms_max = 2.5, 8.0


def marker_size_from_strength(strength):
    if bt_max <= bt_min + 1e-12:
        return 4.0
    u = (strength - bt_min) / (bt_max - bt_min)
    u = float(np.clip(u, 0.0, 1.0))
    return ms_min + u * (ms_max - ms_min)


# ----------------------------------------------------------------------
# Figure layout
# ----------------------------------------------------------------------
ncols = 4
n_strength_rows = 1 if len(strength_specs) > 0 else 0
n_tau_rows = 1 if len(tau_specs) > 0 else 0
n_topic_rows = math.ceil(len(topic_specs) / ncols) if len(topic_specs) > 0 else 0
nrows = n_strength_rows + n_tau_rows + n_topic_rows

fig_h = 2.05 * nrows + 0.8
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11, 10), squeeze=False)
fig.subplots_adjust(hspace=0.95, wspace=0.5)

param_to_ax = {}
used_axes = []
broken_axes = {}

row_cursor = 0

if len(strength_specs) > 0:
    for j, spec in enumerate(strength_specs):
        param_to_ax[spec["name"]] = axes[row_cursor, j]
        used_axes.append(axes[row_cursor, j])
    for j in range(len(strength_specs), ncols):
        axes[row_cursor, j].axis("off")
    row_cursor += 1

if len(tau_specs) > 0:
    for j, spec in enumerate(tau_specs):
        param_to_ax[spec["name"]] = axes[row_cursor, j]
        used_axes.append(axes[row_cursor, j])
    for j in range(len(tau_specs), ncols):
        axes[row_cursor, j].axis("off")
    tau_row_idx = row_cursor
    row_cursor += 1
else:
    tau_row_idx = None

if len(topic_specs) > 0:
    for k, spec in enumerate(topic_specs):
        r = row_cursor + k // ncols
        c = k % ncols
        param_to_ax[spec["name"]] = axes[r, c]
        used_axes.append(axes[r, c])

    n_used_topic_cells = len(topic_specs)
    for k in range(n_used_topic_cells, n_topic_rows * ncols):
        r = row_cursor + k // ncols
        c = k % ncols
        axes[r, c].axis("off")


# broken axis for tau_agree if present
if "lambda_b_agree" in param_to_ax:
    ax_orig = param_to_ax["lambda_b_agree"]
    spec = ax_orig.get_subplotspec()
    fig.delaxes(ax_orig)

    subspec = spec.subgridspec(1, 2, wspace=0.08, width_ratios=[1, 1])
    ax_lo = fig.add_subplot(subspec[0])
    ax_hi = fig.add_subplot(subspec[1], sharey=ax_lo)

    param_to_ax["lambda_b_agree"] = ax_lo
    broken_axes["lambda_b_agree"] = (ax_lo, ax_hi)

    used_axes = [ax for ax in used_axes if ax is not ax_orig] + [ax_lo, ax_hi]


def add_group_rect(fig_, axes_list, pad=(0.04, 0.03, 0.03, 0.03), **rect_kwargs):
    if len(axes_list) == 0:
        return
    l, r, b, t = pad
    bboxes = [ax.get_position().frozen() for ax in axes_list]
    x0 = min(bb.x0 for bb in bboxes) - l
    y0 = min(bb.y0 for bb in bboxes) - b
    x1 = max(bb.x1 for bb in bboxes) + r
    y1 = max(bb.y1 for bb in bboxes) + t
    fig_.patches.append(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            transform=fig_.transFigure,
            **rect_kwargs,
        )
    )


rect_kwargs = dict(
    linestyle="solid",
    linewidth=1.0,
    edgecolor="grey",
    facecolor="whitesmoke",
    alpha=0.8,
    zorder=-100,
)

if len(strength_specs) + len(tau_specs) > 0:
    top_axes_for_box = []
    if len(strength_specs) > 0:
        top_axes_for_box.extend([param_to_ax[s["name"]] for s in strength_specs])

    if len(tau_specs) > 0:
        for s in tau_specs:
            if s["name"] in broken_axes:
                top_axes_for_box.extend(list(broken_axes[s["name"]]))
            else:
                top_axes_for_box.append(param_to_ax[s["name"]])

    add_group_rect(fig, top_axes_for_box, pad=(0.04, 0.03, 0.03, 0.03), **rect_kwargs)

if len(topic_specs) > 0:
    topic_axes_for_box = [param_to_ax[s["name"]] for s in topic_specs]
    add_group_rect(fig, topic_axes_for_box, pad=(0.04, 0.03, 0.04, 0.042), **rect_kwargs)

# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------
offset_values = np.linspace(0.3, -0.3, len(available_models))
offsets = dict(zip(available_models, offset_values))
min_offset = float(offset_values.min())

VIOLIN_KW = dict(
    vert=False,
    widths=0.18,
    showmeans=False,
    showextrema=False,
    showmedians=False,
)


def style_bodies(parts, model_list):
    for body, m in zip(parts["bodies"], model_list):
        body.set_facecolor(violin_colors[m])
        body.set_edgecolor("none")
        body.set_alpha(0.3)


def get_prior_grid(param_name, x_min, x_max):
    if param_name.startswith("b"):
        return np.linspace(
            -2.05 * POSITION_TO_ORIGINAL,
            2.05 * POSITION_TO_ORIGINAL,
            2000,
        )

    if param_name.startswith("lambda"):
        hi = max(args.tau_break_hi * 1.1, x_max)
        return np.concatenate(
            [
                np.linspace(0.0, 1.0, 1000, endpoint=False),
                np.linspace(1.0, hi, 2000),
            ]
        )

    return np.linspace(x_min, x_max, 2000)


DISPLAY_XLIMS = {
    "alpha_mu": (-0.01, 0.06),
    "beta_t_mu": (-0.01, 0.09),
    "beta_a_mu": (-0.03, 0.03),
    "beta_c_mu": (-0.05, 0.01),
    "lambda_interact": (0.0, 4),
    "lambda_b_topic": (0.0, 14.0),
    "lambda_b_agree": (0.0, 1.0),
    "lambda_b_anchor": (0.0, 1.4),
}

# DISPLAY_XLIMS = {
#     "alpha_mu": (-0.05, 0.1),
#     "beta_t_mu": (-0.05, 0.1),
#     "beta_a_mu": (-0.05, 0.1),
#     "beta_c_mu": (-0.05, 0.1),
#     "lambda_interact": (0.0, 12),
#     "lambda_b_topic": (0.0, 12),
#     "lambda_b_agree": (0.0, 12),
#     "lambda_b_anchor": (0.0, 12),
# }

def get_display_xlim(param_name, draws):
    if param_name in topic_param_set:
        return (-1.1 * POSITION_TO_ORIGINAL, 1.1 * POSITION_TO_ORIGINAL)

    q_low, q_high = np.nanpercentile(draws, [1, 99])

    if param_name.startswith("lambda"):
        pad = 0.12 * (q_high - q_low + 1e-12)
        lo = max(0.0, q_low - pad)
        hi = q_high + pad
    else:
        pad = 0.18 * (q_high - q_low + 1e-12)
        lo = q_low - pad
        hi = q_high + pad

    if param_name in DISPLAY_XLIMS:
        ref_lo, ref_hi = DISPLAY_XLIMS[param_name]
        lo = min(lo, ref_lo)
        hi = max(hi, ref_hi)

    return lo, hi


# ----------------------------------------------------------------------
# Plot loop
# ----------------------------------------------------------------------
plot_specs = strength_specs + tau_specs + topic_specs

for idx, spec in enumerate(plot_specs):
    param = spec["name"]
    ax = param_to_ax[param]
    ax_list = broken_axes.get(param, (ax,))
    is_broken = param in broken_axes

    datasets = []
    plot_models = []
    for m in available_models:
        if param in posterior[m]:
            datasets.append(posterior[m][param])
            plot_models.append(m)

    if len(datasets) == 0:
        for ax_ in ax_list:
            ax_.axis("off")
        continue

    positions = [offsets[m] for m in plot_models]

    all_draws = np.concatenate(datasets)
    X_MIN, X_MAX = get_display_xlim(param, all_draws)

    if is_broken:
        ax_lo, ax_hi = ax_list
        lo_sets, lo_pos, lo_models = [], [], []
        hi_sets, hi_pos, hi_models = [], [], []

        for m, d in zip(plot_models, datasets):
            d_lo = d[d <= args.tau_break_lo]
            d_hi = d[d >= args.tau_break_hi]
            if d_lo.size > 0:
                lo_sets.append(d_lo)
                lo_pos.append(offsets[m])
                lo_models.append(m)
            if d_hi.size > 0:
                hi_sets.append(d_hi)
                hi_pos.append(offsets[m])
                hi_models.append(m)

        if len(lo_sets) > 0:
            parts = ax_lo.violinplot(lo_sets, positions=lo_pos, **VIOLIN_KW)
            style_bodies(parts, lo_models)

        if len(hi_sets) > 0:
            parts = ax_hi.violinplot(hi_sets, positions=hi_pos, **VIOLIN_KW)
            style_bodies(parts, hi_models)

        for m, d, y_pos in zip(plot_models, datasets, positions):
            hdi_low, hdi_high = az.hdi(d, hdi_prob=0.95)
            med = float(np.median(d))

            if hdi_low <= args.tau_break_lo:
                ax_lo.plot(
                    [hdi_low, min(hdi_high, args.tau_break_lo)],
                    [y_pos, y_pos],
                    color=hdi_colors[m],
                    lw=2,
                )
            if hdi_high >= args.tau_break_hi:
                ax_hi.plot(
                    [max(hdi_low, args.tau_break_hi), hdi_high],
                    [y_pos, y_pos],
                    color=hdi_colors[m],
                    lw=2,
                )

            if param in topic_param_set and (m, spec["topic_index"]) in beta_t_topic_median:
                ms = marker_size_from_strength(beta_t_topic_median[(m, spec["topic_index"])])
            else:
                ms = 3.5

            if med <= args.tau_break_lo:
                ax_lo.plot(med, y_pos, marker="o", color=hdi_colors[m], markersize=ms)
            elif med >= args.tau_break_hi:
                ax_hi.plot(med, y_pos, marker="o", color=hdi_colors[m], markersize=ms)

    else:
        for ax_ in ax_list:
            parts = ax_.violinplot(datasets, positions=positions, **VIOLIN_KW)
            style_bodies(parts, plot_models)

        for m, d, y_pos in zip(plot_models, datasets, positions):
            hdi_low, hdi_high = az.hdi(d, hdi_prob=0.95)
            med = float(np.median(d))

            for ax_ in ax_list:
                ax_.plot([hdi_low, hdi_high], [y_pos, y_pos], color=hdi_colors[m], lw=2)

            if param in topic_param_set and (m, spec["topic_index"]) in beta_t_topic_median:
                ms = marker_size_from_strength(beta_t_topic_median[(m, spec["topic_index"])])
            else:
                ms = 3.5

            for ax_ in ax_list:
                ax_.plot(med, y_pos, marker="o", color=hdi_colors[m], markersize=ms)

    # formatting
    for ax_ in ax_list:
        ax_.set_yticks([0.0])
        ax_.set_yticklabels([spec["label"]])
        ax_.set_ylim([-0.5, 0.5])
        ax_.spines[["top", "right"]].set_visible(False)
        ax_.tick_params(axis="x", labelsize=14)
        ax_.tick_params(axis="y", length=0, labelsize=14)
        if not param.startswith("lambda"):
            ax_.axvline(0.0, linestyle="--", color="black", zorder=-2, lw=1.0)

    if idx != 6:
        ax.set_title(spec["title"], fontsize=12, loc="center", pad=5)
    else:
        ax.set_title(spec["title"], fontsize=12, loc="center", pad=5, x=1)


    # prior overlay
    prior_x = get_prior_grid(param, X_MIN, X_MAX)
    pdf_fn = prior_pdfs.get(param)
    if pdf_fn is not None:
        pdf = pdf_fn(prior_x)
        mmax = float(np.nanmax(pdf))
        if mmax > 0:
            pdf_scaled = pdf / mmax * 0.9 - 0.5
            for ax_ in ax_list:
                ax_.plot(prior_x, pdf_scaled, ls=":", color="grey", zorder=-3, lw=0.8)
                ax_.fill_between(
                    prior_x,
                    min_offset - 0.5,
                    pdf_scaled,
                    color="grey",
                    alpha=0.15,
                    zorder=-4,
                )

    # x-limits / ticks
    if is_broken:
        ax_lo, ax_hi = ax_list

        # sensible fixed window for agreement-bias timescales
        ax_lo.set_xlim([0.0, 1.0])
        ax_hi.set_xlim([100, 1000])

        ax_lo.set_xticks([0.0, 0.6])
        ax_hi.set_xticks([500])

        ax_lo.spines["right"].set_visible(False)
        ax_hi.spines["left"].set_visible(False)
        ax_hi.tick_params(axis="y", left=False, labelleft=False)

        d = 0.03
        kw_lo = dict(transform=ax_lo.transAxes, color="k", clip_on=False, lw=1)
        kw_hi = dict(transform=ax_hi.transAxes, color="k", clip_on=False, lw=1)
        ax_lo.plot((1 - d, 1 + d), (-d, +d), **kw_lo)
        ax_hi.plot((-d, +d), (-d, +d), **kw_hi)

    else:
        if param in topic_param_set:
            ax.set_xlim([-1.1 * POSITION_TO_ORIGINAL, 1.1 * POSITION_TO_ORIGINAL])
            ax.set_xticks([-1.0 * POSITION_TO_ORIGINAL, 0.0, 1.0 * POSITION_TO_ORIGINAL])
        elif param == "alpha_mu":
            ax.set_xlim([X_MIN, X_MAX])
            ax.set_xticks([0.0, 0.03, 0.06])
        elif param in {"beta_c_mu"}:
            ax.set_xlim([X_MIN, X_MAX])
            ax.set_xticks([-0.06, -0.03, 0.0])
        else:
            ax.set_xlim([X_MIN, X_MAX])
            ax.xaxis.set_major_locator(MaxNLocator(3))



top_block_last_row = n_strength_rows + n_tau_rows - 1
topic_block_first_row = n_strength_rows + n_tau_rows

if top_block_last_row >= 0:
    add_group_ylabel(fig, axes, 0, top_block_last_row, "Global parameters")

if n_topic_rows > 0:
    add_group_ylabel(fig, axes, topic_block_first_row, nrows - 1, "Topic-specific attractors")


# ----------------------------------------------------------------------
# Legend / save
# ----------------------------------------------------------------------
handles = [
    Patch(
        facecolor=violin_colors[m],
        edgecolor="none",
        alpha=0.6,
        label=model_labels[m],
    )
    for m in available_models
]

prior_handle = object()
posterior_handle = object()

fig.legend(
    handles=handles + [prior_handle] + [posterior_handle],
    labels=[model_labels[m] for m in available_models] + ["Prior distribution", "Posterior distribution\n(median + 95% HDI)"],
    handler_map={prior_handle: HandlerPrior(), posterior_handle: HandlerPosterior()},
    frameon=False,
    fontsize=14,
    loc="upper center",
    ncols=4,
    bbox_to_anchor=(0.5, 1.01),
)

fig.supxlabel(
    r"Parameter value",
    fontsize=18,
    y=0.03,
)

os.makedirs(args.save_dir, exist_ok=True)
if args.save_name is None:
    save_name = f"figure_6.png"
else:
    save_name = args.save_name

save_path = os.path.join(args.save_dir, save_name)
plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved figure to: {save_path}")

plt.show()