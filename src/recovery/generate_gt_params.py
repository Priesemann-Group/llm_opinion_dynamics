#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_TOPICS = [
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


def softplus(x):
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def to_serializable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def broadcast_or_validate(value, n, name):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=float)
    if arr.shape == (n,):
        return arr.astype(float)
    raise ValueError(
        f"'{name}' must be a scalar or a list of length {n}, got shape {arr.shape}."
    )


def sample_or_use_fixed(spec, rng, n, name):
    """
    Supports:
      - scalar
      - list of length n
      - dict for sampling, e.g.
          {"dist": "uniform", "lower": -1, "upper": 1}
          {"dist": "normal", "mu": 0.1, "sd": 0.02}
    """
    if isinstance(spec, dict):
        dist = spec.get("dist")
        if dist == "uniform":
            lower = float(spec["lower"])
            upper = float(spec["upper"])
            return rng.uniform(lower, upper, size=n)
        if dist == "normal":
            mu = float(spec["mu"])
            sd = float(spec["sd"])
            return rng.normal(mu, sd, size=n)
        raise ValueError(f"Unsupported distribution for '{name}': {dist}")
    return broadcast_or_validate(spec, n, name)


def generate_hierarchical_param(hyper, rng, n, kind):
    if kind == "alpha":
        mu = float(hyper["alpha_mu"])
        sd = float(hyper["alpha_sd"])
        return rng.normal(mu, sd, size=n)

    if kind == "beta_a":
        mu = float(hyper["beta_a_mu"])
        sd = float(hyper["beta_a_sd"])
        return rng.normal(mu, sd, size=n)

    if kind == "beta_c":
        mu = float(hyper["beta_c_mu"])
        sd = float(hyper["beta_c_sd"])
        return rng.normal(mu, sd, size=n)

    if kind == "sigma0":
        mu = float(hyper["log_sigma0_mu"])
        sd = float(hyper["log_sigma0_sd"])
        latent = rng.normal(mu, sd, size=n)
        return softplus(latent)

    if kind == "beta_t":
        mu = float(hyper["log_beta_t_mu"])
        sd = float(hyper["log_beta_t_sd"])
        latent = rng.normal(mu, sd, size=n)
        return softplus(4.0 * latent) / 4.0

    raise ValueError(f"Unknown hierarchical parameter kind: {kind}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a ground-truth JSON for parameter recovery. "
            "Output is directly compatible with the recovery script."
        )
    )
    parser.add_argument("--config", type=str, required=True, help="Path to input config JSON.")
    parser.add_argument("--out", type=str, required=True, help="Path to output GT JSON.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--include_meta",
        action="store_true",
        help="Include input hyperparameters and topics under '_meta'.",
    )
    args = parser.parse_args()

    cfg = load_json(args.config)
    rng = np.random.default_rng(args.seed)

    topics = cfg.get("topics", DEFAULT_TOPICS)
    n_topics = len(topics)

    hyper = cfg.get("hyper", {})
    fixed = cfg.get("fixed", {})

    gt = {}

    # -------------------- hierarchical or explicit topic-level parameters
    hierarchical_params = ["sigma0", "alpha", "beta_t", "beta_a", "beta_c"]

    for name in hierarchical_params:
        if name in fixed:
            gt[name] = sample_or_use_fixed(fixed[name], rng, n_topics, name)
        else:
            gt[name] = generate_hierarchical_param(hyper, rng, n_topics, name)

    # -------------------- topic preference parameter b
    # b is non-hierarchical in the model, but still topic-specific
    # You can give it as:
    #   - scalar
    #   - list of length n_topics
    #   - {"dist": "uniform", "lower": ..., "upper": ...}
    #   - {"dist": "normal", "mu": ..., "sd": ...}
    if "b" not in fixed:
        raise ValueError(
            "Config must provide 'fixed.b' as a scalar, list, or distribution spec."
        )
    gt["b"] = sample_or_use_fixed(fixed["b"], rng, n_topics, "b")

    # -------------------- scalar non-hierarchical parameters
    scalar_names = [
        "epsilon",
        "lambda_interact",
        "lambda_b_topic",
        "lambda_b_agree",
        "lambda_b_anchor",
    ]
    for name in scalar_names:
        if name in fixed:
            gt[name] = float(fixed[name])

    # -------------------- optional metadata
    if args.include_meta:
        gt["_meta"] = {
            "seed": args.seed,
            "topics": topics,
            "hyper": hyper,
            "fixed_input": fixed,
        }

    # convert numpy types for JSON
    gt_json = {k: to_serializable(v) for k, v in gt.items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(gt_json, f, indent=2)

    print(f"Saved GT JSON to {out_path}")


if __name__ == "__main__":
    main()