import os
import argparse
import textwrap
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.interpolate import griddata
from scipy.ndimage import minimum_filter

DATA_PATH_DEFAULT = (
    "../../data/all_discussions_s=25_l=5.parquet"
)
PRIOR_PATH_DEFAULT = (
    "../../data/unprompted_opinion_priors/"
)
OUTPUT_DIR_DEFAULT = "../../figures"

BASE_TOPICS = [
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

PRIOR_KEYS = []
for topic in BASE_TOPICS:
    base = topic.replace("_", " ")
    PRIOR_KEYS.append(base)
    PRIOR_KEYS.append(f"{base} rev")


PLOT_TOPICS = [
    "Climate_Change",
    "Global_Wealth_Distribution",
    "Artificial_Intelligence",
    "Musical_Preference",
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

API_MODELS = {"gpt-4o-mini", "grok-4-1-fast-non-reasoning"}
TOPIC_TITLES = {
    "Climate_Change": (
        "\"Climate Change is an important\nproblem to address\"",
        "\"Climate Change is $\\mathbf{\\mathit{not}}$ an\nimportant problem to address\"",
    ),
    "Vaccination": (
        "\"Vaccinations are necessary\nfor public health\"",
        "\"Vaccinations are $\\mathbf{\\mathit{not}}$ necessary\nfor public health\"",
    ),
    "Shape_of_the_Earth": (
        "\"The Earth\nis flat\"",
        "\"The Earth is\n$\\mathbf{\\mathit{not}}$ flat\"",
    ),
    "Global_Wealth_Distribution": (
        "\"The global distribution of wealth\nis fair\"",
        "\"The global distribution of wealth\nis $\\mathbf{\\mathit{not}}$ fair\"",
    ),
    "Abortion": (
        "\"Abortions should\nbe legal\"",
        "\"Abortions should\n$\\mathbf{\\mathit{not}}$ be legal\"",
    ),
    "Social_Media": (
        "\"Social media does have a\npositive impact on society\"",
        "\"Social media does $\\mathbf{\\mathit{not}}$ have a\npositive impact on society\"",
    ),
    "Artificial_Intelligence": (
        "\"Artificial Intelligence\nis dangerous\"",
        "\"Artificial Intelligence is\n$\\mathbf{\\mathit{not}}$ dangerous\"",
    ),
    "Morality_and_Religion": (
        "\"You can only be a moral person\nif you believe in God\"",
        "\"You cannot only be a moral person\nif you believe in God\"",
    ),
    "Free_Will": (
        "\"Humans possess\nfree will\"",
        "\"Humans do $\\mathbf{\\mathit{not}}$ possess\nfree will\"",
    ),
    "Musical_Preference": (
        "\"Bach is a greater composer\nthan Stravinsky\"",
        "\"Bach is $\\mathbf{\\mathit{not}}$ a greater composer\nthan Stravinsky\"",
    ),
    "Food_Preference": (
        "\"Pizza is better\nthan sushi\"",
        "\"Pizza is $\\mathbf{\\mathit{not}}$ better\nthan sushi\"",
    ),
    "Art_Style_Preference": (
        "\"Modern art is more meaningful\nthan classical art\"",
        "\"Modern art is $\\mathbf{\\mathit{not}}$ more meaningful\nthan classical art\"",
    ),
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the 4x6 streamplot figure for one model using the new parquet dataset."
    )
    parser.add_argument("--model", default=MODELS[2], choices=MODELS, help="Model to plot.")
    parser.add_argument(
        "--data-path",
        default=DATA_PATH_DEFAULT,
        help="Path to the parquet file with processed discussion data.",
    )
    parser.add_argument(
        "--prior-path",
        default=PRIOR_PATH_DEFAULT,
        help="Directory containing the prior .npy files.",
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
        "--grid-size",
        type=int,
        default=100,
        help="Resolution of the interpolation grid per axis.",
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
        if value_norm in {
            "reverse",
            "reversed",
            "rev",
            "reverse_framing",
            "reversed_framing",
            "1",
            "true",
        }:
            return True
        if value_norm in {
            "normal",
            "base",
            "original",
            "forward",
            "-1",
            "0",
            "false",
        }:
            return False

    raise ValueError(f"Unexpected framing value: {value}")

def load_parquet_data(data_path: str) -> pd.DataFrame:
    df = pd.read_parquet(data_path).copy()
    df["__row_order__"] = np.arange(len(df))
    df["framing_bool"] = df["framing"].apply(coerce_reverse_framing)
    return df


def load_priors(prior_path: str, models: List[str]) -> Dict[str, Dict[str, float]]:
    priors = {}

    for model in models:
        mean_entropy_path = os.path.join(
            prior_path,
            f"unprompted_opinion_priors_{model}_mean_entropy.npy"
        )

        if not os.path.exists(mean_entropy_path):
            raise FileNotFoundError(f"Missing prior file: {mean_entropy_path}")

        arr = np.load(mean_entropy_path)   # shape: (24, 2)
        means = arr[:, 0]

        if len(means) != len(PRIOR_KEYS):
            raise ValueError(
                f"{model}: expected {len(PRIOR_KEYS)} prior entries, got {len(means)}"
            )

        priors[model] = {key: means[i] for i, key in enumerate(PRIOR_KEYS)}

    return priors


def compute_topic_priors(model: str, priors: Dict[str, Dict[str, float]]) -> Dict[Tuple[str, bool], float]:
    model_priors = priors[model]
    out = {}

    for topic in BASE_TOPICS:
        base = topic.replace("_", " ")
        out[(topic, False)] = model_priors[base]
        out[(topic, True)] = model_priors[f"{base} rev"]

    return out


def format_topic_title(topic: str, reverse: bool) -> str:
    normal, reversed_title = TOPIC_TITLES[topic]
    return reversed_title if reverse else normal



def get_panel_rows(df: pd.DataFrame, model: str, topic: str, reverse: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    panel_df = df[
        (df["model"] == model)
        & (df["topic"] == topic)
        & (df["framing_bool"] == reverse)
    ].sort_values("__row_order__")

    start_rows = panel_df[panel_df["t"] == 0].copy()
    end_rows = panel_df[panel_df["t"] == 5].copy()

    if len(start_rows) == 0:
        raise ValueError(f"No rows found for model={model}, topic={topic}, reverse={reverse}.")
    if len(end_rows) == 0:
        raise ValueError(f"No final rows found for model={model}, topic={topic}, reverse={reverse}.")
    if len(start_rows) != len(end_rows):
        raise ValueError(
            f"Start/end row mismatch for model={model}, topic={topic}, reverse={reverse}: "
            f"{len(start_rows)} starts vs {len(end_rows)} ends. "
            "Without an explicit discussion identifier, these cannot be aligned safely."
        )

    return start_rows.reset_index(drop=True), end_rows.reset_index(drop=True)


def interpolate_panel(
    start_rows: pd.DataFrame,
    end_rows: pd.DataFrame,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> Dict[str, np.ndarray]:
    x_start = start_rows["x_i"].to_numpy(dtype=float) 
    y_start = start_rows["x_j"].to_numpy(dtype=float) 

    x_final = end_rows["x_i"].to_numpy(dtype=float) 
    y_final = end_rows["x_j"].to_numpy(dtype=float) 

    u = x_final - x_start
    v = y_final - y_start

    # Final uncertainty from t=5, converted from natural log to log2
    entropy = (
        0.5 * (end_rows["H_i"].to_numpy(dtype=float) + end_rows["H_j"].to_numpy(dtype=float))
        / np.log(2.0)
    )

    valid = np.isfinite(x_start) & np.isfinite(y_start) & np.isfinite(u) & np.isfinite(v) & np.isfinite(entropy)

    x_start = x_start[valid]
    y_start = y_start[valid]
    u = u[valid]
    v = v[valid]
    entropy = entropy[valid]

    if x_start.size < 3:
        shape = grid_x.shape
        return {
            "grid_u": np.full(shape, np.nan),
            "grid_v": np.full(shape, np.nan),
            "grid_magnitude": np.full(shape, np.nan),
            "grid_entropy": np.full(shape, np.nan),
        }

    grid_u = griddata((x_start, y_start), u, (grid_x, grid_y), method="linear")
    grid_v = griddata((x_start, y_start), v, (grid_x, grid_y), method="linear")
    grid_entropy = griddata((x_start, y_start), entropy, (grid_x, grid_y), method="linear")
    grid_magnitude = np.sqrt(grid_u**2 + grid_v**2)

    return {
        "grid_u": grid_u,
        "grid_v": grid_v,
        "grid_magnitude": grid_magnitude,
        "grid_entropy": grid_entropy,
    }


def build_all_panels(
    df: pd.DataFrame,
    model: str,
    grid_size: int,
) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    grid_x, grid_y = np.meshgrid(
        np.linspace(-2, 2, grid_size),
        np.linspace(-2, 2, grid_size),
    )

    panel_specs = (
        [(topic, False) for topic in PLOT_TOPICS]
        + [(topic, True) for topic in PLOT_TOPICS]
    )

    panels = []
    for topic, reverse in panel_specs:
        start_rows, end_rows = get_panel_rows(df, model, topic, reverse)
        panel = interpolate_panel(start_rows, end_rows, grid_x, grid_y)
        panel["topic"] = topic
        panel["reverse"] = reverse
        panels.append(panel)

    return panels, grid_x, grid_y


def plot_panels(
    panels: List[Dict[str, np.ndarray]],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    topic_priors: Dict[str, float],
    model: str,
    output_dir: str,
    dpi: int,
) -> str:
    cmap_stream = "inferno"
    cmap_entropy = plt.cm.Greys.copy()
    cmap_entropy.set_bad((1, 1, 1, 0))

    magnitude_stack = np.stack([p["grid_magnitude"] for p in panels], axis=0)
    entropy_stack = np.stack([p["grid_entropy"] for p in panels], axis=0)

    norm_magnitude = Normalize(vmin=np.nanmin(magnitude_stack), vmax=np.nanmax(magnitude_stack))
    norm_entropy = Normalize(vmin=np.nanmin(entropy_stack), vmax=np.nanmax(entropy_stack))

    fig = plt.figure(figsize=(11.5, 6.4))
    gs = fig.add_gridspec(
        2,
        4,
        hspace=0.375,
        wspace=0,
    )

    axes = np.empty((2, 4), dtype=object)
    for r in range(2):
        for c in range(4):
            share_ax = axes[0, 0] if (r > 0 or c > 0) else None
            axes[r, c] = fig.add_subplot(gs[r, c], sharex=share_ax, sharey=share_ax)

    stream_mappable = None
    entropy_mappable = None

    for idx, ax in enumerate(axes.flat):
        panel = panels[idx]

        # Background: interpolated final uncertainty
        z = panel["grid_entropy"]
        z_ma = np.ma.masked_invalid(z)
        entropy_mappable = ax.imshow(
            z_ma,
            extent=(-2, 2, -2, 2),
            origin="lower",
            cmap=cmap_entropy,
            alpha=0.5,
            aspect="auto",
            norm=norm_entropy,
        )

        nodata = z_ma.mask.astype(float)
        # transparent fill
        ax.contourf(
            grid_x, grid_y, nodata,
            levels=[0.5, 1.5],
            colors=[(1, 0, 1, 0.025)],
            zorder=0,
        )

        # hatch overlay
        ax.contourf(
            grid_x,
            grid_y,
            nodata,
            levels=[0.5, 1.5],
            colors="none",
            hatches=["...."],
            zorder=1,
        )

        # Streamlines
        stream = ax.streamplot(
            grid_x,
            grid_y,
            panel["grid_u"],
            panel["grid_v"],
            color=panel["grid_magnitude"],
            linewidth=1.5,
            cmap=cmap_stream,
            density=0.6,
            arrowsize=1.4,
            norm=norm_magnitude,
        )
        stream_mappable = stream.lines

        # Topic prior
        topic_prior = topic_priors[(panel["topic"], panel["reverse"])]
       
        ax.scatter(
            topic_prior,
            topic_prior,
            s=100,
            color="lime",
            marker="P",
            edgecolors="black",
            lw=1.5,
            zorder=11,
        )

        
        # Fixed-point / attractor approximation for the discrete map
        u = panel["grid_u"]
        v = panel["grid_v"]

        dx = grid_x[0, 1] - grid_x[0, 0]
        dy = grid_y[1, 0] - grid_y[0, 0]

        dU_dy, dU_dx = np.gradient(u, dy, dx, edge_order=2)
        dV_dy, dV_dx = np.gradient(v, dy, dx, edge_order=2)

        speed = np.sqrt(u**2 + v**2)

        # Jacobian of T(x, y) = (x + U(x, y), y + V(x, y))
        J11 = 1.0 + dU_dx
        J12 = dU_dy
        J21 = dV_dx
        J22 = 1.0 + dV_dy

        trace = J11 + J22
        det = J11 * J22 - J12 * J21
        disc = trace**2 - 4.0 * det

        rho = np.full_like(speed, np.nan, dtype=float)

        real_mask = np.isfinite(disc) & (disc >= 0)
        sqrt_disc = np.zeros_like(speed)
        sqrt_disc[real_mask] = np.sqrt(disc[real_mask])

        lam1 = np.full_like(speed, np.nan, dtype=float)
        lam2 = np.full_like(speed, np.nan, dtype=float)

        lam1[real_mask] = 0.5 * (trace[real_mask] + sqrt_disc[real_mask])
        lam2[real_mask] = 0.5 * (trace[real_mask] - sqrt_disc[real_mask])

        complex_mask = np.isfinite(disc) & (disc < 0)
        rho[complex_mask] = np.sqrt(np.abs(det[complex_mask]))
        rho[real_mask] = np.maximum(np.abs(lam1[real_mask]), np.abs(lam2[real_mask]))

        speed_thresh = np.nanpercentile(speed[np.isfinite(speed)], 10)
        cand = np.where(
            np.isfinite(speed) &
            np.isfinite(rho) &
            (rho < 1.0) &
            (speed <= speed_thresh)
        )

        if cand[0].size:
            argmin = np.argmin(speed[cand])
            j0 = cand[0][argmin]
            k0 = cand[1][argmin]
            ax.scatter(
                grid_x[j0, k0],
                grid_y[j0, k0],
                s=150,
                color="cyan",
                marker="*",
                edgecolors="black",
                lw=1.25,
                zorder=12,
            )
        
        # # Numerical sink / attractor approximation
        # u = panel["grid_u"]
        # v = panel["grid_v"]

        # dx = grid_x[0, 1] - grid_x[0, 0]
        # dy = grid_y[1, 0] - grid_y[0, 0]

        # dU_dy, dU_dx = np.gradient(u, dy, dx, edge_order=2)
        # dV_dy, dV_dx = np.gradient(v, dy, dx, edge_order=2)
        # div = dU_dx + dV_dy

        # speed = panel["grid_magnitude"]
        # cand = np.where(np.isfinite(speed) & np.isfinite(div) & (div < 0))

        # if cand[0].size:
        #     argmin = np.argmin(speed[cand])
        #     j0 = cand[0][argmin]
        #     k0 = cand[1][argmin]
        #     ax.scatter(
        #         grid_x[j0, k0],
        #         grid_y[j0, k0],
        #         s=150,
        #         color="cyan",
        #         marker="*",
        #         edgecolors="black",
        #         lw=1.25,
        #         zorder=12,
        #     )

        # Axes styling
        ax.set_xlim(-2.05, 2.05)
        ax.set_ylim(-2.05, 2.05)

        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])

        ax.set_xticks(np.linspace(-2, 2, 9), minor=True)
        ax.set_yticks(np.linspace(-2, 2, 9), minor=True)

        ax.tick_params(axis="both", labelsize=12)
        ax.tick_params(axis="both", which="minor", labelbottom=False, labelleft=False)

        ax.set_aspect("equal")
        ax.set_title(format_topic_title(panel["topic"], panel["reverse"]), fontsize=10, y=1.0, pad=8)

        ax.spines[["right", "top"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("black")
        ax.plot([-2, 2], [-2, 2], ls="-", color="black", alpha=1, lw=1.25)

        if ax not in axes[-1, :]:
            ax.tick_params(labelbottom=False)
        if ax not in axes[:, 0]:
            ax.tick_params(labelleft=False)

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.11, top=0.77)

    # Global colorbars above panels
    cbar_ax_stream = fig.add_axes([0.1, 0.965, 0.18, 0.018])
    cbar_ax_entropy = fig.add_axes([0.37, 0.965, 0.18, 0.018])

    cbar_stream = fig.colorbar(stream_mappable, cax=cbar_ax_stream, orientation="horizontal")
    cbar_entropy = fig.colorbar(entropy_mappable, cax=cbar_ax_entropy, orientation="horizontal")

    cbar_stream.set_label("Magnitude of opinion shift", size=12)
    cbar_entropy.set_label("Final opinion uncertainty", size=12)

    cbar_ax_stream.tick_params(labelsize=12)
    cbar_ax_entropy.tick_params(labelsize=12)

    # Global legend above panels
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="P",
            color="none",
            markerfacecolor="lime",
            markeredgecolor="black",
            markeredgewidth=1.5,
            markersize=10,
            linestyle="None",
            label="Topic prior",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="cyan",
            markeredgecolor="black",
            markeredgewidth=1.25,
            markersize=14,
            linestyle="None",
            label="Attractor\n(numerical approx.)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.60, 0.99),
        frameon=False,
        fontsize=12,
        ncols=2,
    )

    fig.supxlabel("Opinion Agent A", y=0.01, x=0.5, fontsize=16)
    fig.supylabel("Opinion Agent B", x=0.04, y=0.5, fontsize=16)

    model_tag = MODEL_LABELS.get(model, model).replace(".", "p").replace("/", "_").replace(" ", "_")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"figure_4.pdf")

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()

    return output_path


def main() -> None:
    args = parse_args()

    df = load_parquet_data(args.data_path)
    priors = load_priors(args.prior_path, MODELS)
    topic_priors = compute_topic_priors(args.model, priors)

    panels, grid_x, grid_y = build_all_panels(df, args.model, args.grid_size)
    output_path = plot_panels(
        panels=panels,
        grid_x=grid_x,
        grid_y=grid_y,
        topic_priors=topic_priors,
        model=args.model,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()