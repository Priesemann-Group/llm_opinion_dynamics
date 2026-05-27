import os
import time
import json
import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

tick = time.time()
print("Libraries loaded.\n", flush=True)


parser = argparse.ArgumentParser()

# -------------------- data / sampler args
parser.add_argument("--llm", type=str, required=True)
parser.add_argument("--draws", type=int, default=2000)
parser.add_argument("--tune", type=int, default=1000)
parser.add_argument("--chains", type=int, default=4)
parser.add_argument("--cores", type=int, default=4)
parser.add_argument("--target_accept", type=float, default=0.9)
parser.add_argument("--max_treedepth", type=int, default=10)
parser.add_argument("--mode", choices=["pp", "sample"], type=str, default="sample")

# -------------------- recovery args
parser.add_argument(
    "--recovery",
    choices=["empirical", "synthetic"],
    default="empirical",
    help=(
        "Parameter recovery mode. "
        "'empirical' keeps the observed covariates and simulates only delta_x_i from fixed GT params. "
        "'synthetic' recursively simulates x_i/x_j/x_0/delta_x_i along each discussion template."
    ),
)
parser.add_argument(
    "--gt_file",
    type=str,
    default=None,
    help=(
        "Path to a JSON file with ground-truth parameters. "
        "Accepted natural-scale keys: sigma0, epsilon, alpha, b, beta_t, beta_a, beta_c, "
        "lambda_interact, lambda_b_topic, lambda_b_agree, lambda_b_anchor. "
        "Also accepted: log_sigma0, log_epsilon, log_beta_t, log_lambda_*."
    ),
)
parser.add_argument("--random_seed", type=int, default=42)

# -------------------- ablation / model-building args
# defaults reproduce the full model
parser.add_argument("--interaction", choices=["none", "static", "decay"], default="decay")
parser.add_argument("--topic_bias", choices=["none", "static", "decay"], default="decay")
parser.add_argument("--agree_bias", choices=["none", "static", "decay"], default="decay")
parser.add_argument("--anchor_bias", choices=["none", "static", "decay"], default="decay")

parser.add_argument("--epsilon", dest="epsilon", action="store_true")
parser.add_argument("--no_epsilon", dest="epsilon", action="store_false")
parser.set_defaults(epsilon=True)

args = parser.parse_args()

print("Arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
print("\n", flush=True)


models = [
    "Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "dolphin-2.7-mixtral-8x7b-AWQ",
    "Llama-3.3-70B-Instruct-AWQ", "Mixtral-8x7B-Instruct-v0.1-AWQ-INT4",
    "gpt-4o-mini", "grok-4-1-fast-non-reasoning"
]

topics = [
    "Climate_Change", "Vaccination", "Shape_of_the_Earth",
    "Global_Wealth_Distribution", "Abortion", "Social_Media",
    "Artificial_Intelligence", "Morality_and_Religion", "Free_Will",
    "Musical_Preference", "Food_Preference", "Art_Style_Preference"
]

A_TARGET = 1.0

param_SD = 0.1
log_beta_mu_MU = 0
log_beta_mu_SD = 0.5
log_beta_sd_SD = 0.3


def build_model_tag(args):
    eps_tag = "eps" if args.epsilon else "noeps"
    recovery_tag = f"__R-{args.recovery}" if args.recovery is not None else ""
    return (
        f"I-{args.interaction}"
        f"__T-{args.topic_bias}"
        f"__A-{args.agree_bias}"
        f"__C-{args.anchor_bias}"
        f"__E-{eps_tag}"
        f"{recovery_tag}"
    )


def softplus_np(x):
    x = np.asarray(x, dtype="float64")
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def broadcast_topic_param(value, n_topics, name):
    value = np.asarray(value, dtype="float64")
    if value.ndim == 0:
        return np.full(n_topics, float(value), dtype="float64")
    if value.shape == (n_topics,):
        return value.astype("float64")
    raise ValueError(
        f"Ground-truth parameter '{name}' must be scalar or length {n_topics}, got shape {value.shape}."
    )


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def normalize_gt_params(raw_gt, n_topics):
    gt = {}

    def first_available(*keys):
        for key in keys:
            if key in raw_gt:
                return raw_gt[key]
        return None

    sigma0 = first_available("sigma0")
    if sigma0 is None and "log_sigma0" in raw_gt:
        sigma0 = softplus_np(raw_gt["log_sigma0"])
    if sigma0 is None and "sigma0_mu" in raw_gt:
        sigma0 = raw_gt["sigma0_mu"]
    if sigma0 is None and "log_sigma0_mu" in raw_gt:
        sigma0 = softplus_np(raw_gt["log_sigma0_mu"])
    if sigma0 is not None:
        gt["sigma0"] = broadcast_topic_param(sigma0, n_topics, "sigma0")

    alpha = first_available("alpha", "alpha_mu")
    if alpha is not None:
        gt["alpha"] = broadcast_topic_param(alpha, n_topics, "alpha")

    b = first_available("b")
    if b is not None:
        gt["b"] = broadcast_topic_param(b, n_topics, "b")

    beta_t = first_available("beta_t", "beta_t_mu")
    if beta_t is None and "log_beta_t" in raw_gt:
        beta_t = softplus_np(4.0 * np.asarray(raw_gt["log_beta_t"], dtype="float64")) / 4.0
    if beta_t is None and "log_beta_t_mu" in raw_gt:
        beta_t = softplus_np(4.0 * np.asarray(raw_gt["log_beta_t_mu"], dtype="float64")) / 4.0
    if beta_t is not None:
        gt["beta_t"] = broadcast_topic_param(beta_t, n_topics, "beta_t")

    beta_a = first_available("beta_a", "beta_a_mu")
    if beta_a is not None:
        gt["beta_a"] = broadcast_topic_param(beta_a, n_topics, "beta_a")

    beta_c = first_available("beta_c", "beta_c_mu")
    if beta_c is not None:
        gt["beta_c"] = broadcast_topic_param(beta_c, n_topics, "beta_c")

    epsilon = first_available("epsilon")
    if epsilon is None and "log_epsilon" in raw_gt:
        epsilon = float(softplus_np(raw_gt["log_epsilon"]))
    if epsilon is not None:
        gt["epsilon"] = float(np.asarray(epsilon, dtype="float64"))

    lambda_interact = first_available("lambda_interact")
    if lambda_interact is None and "log_lambda_interact" in raw_gt:
        lambda_interact = float(np.exp(raw_gt["log_lambda_interact"]))
    if lambda_interact is not None:
        gt["lambda_interact"] = float(lambda_interact)

    lambda_b_topic = first_available("lambda_b_topic")
    if lambda_b_topic is None and "log_lambda_b_topic" in raw_gt:
        lambda_b_topic = float(np.exp(raw_gt["log_lambda_b_topic"]))
    if lambda_b_topic is not None:
        gt["lambda_b_topic"] = float(lambda_b_topic)

    lambda_b_agree = first_available("lambda_b_agree")
    if lambda_b_agree is None and "log_lambda_b_agree" in raw_gt:
        lambda_b_agree = float(np.exp(raw_gt["log_lambda_b_agree"]))
    if lambda_b_agree is not None:
        gt["lambda_b_agree"] = float(lambda_b_agree)

    lambda_b_anchor = first_available("lambda_b_anchor")
    if lambda_b_anchor is None and "log_lambda_b_anchor" in raw_gt:
        lambda_b_anchor = float(np.exp(raw_gt["log_lambda_b_anchor"]))
    if lambda_b_anchor is not None:
        gt["lambda_b_anchor"] = float(lambda_b_anchor)

    return gt


def validate_gt_params(gt, args):
    required = ["sigma0"]
    if args.epsilon:
        required.append("epsilon")

    if args.interaction != "none":
        required.append("alpha")
        if args.interaction == "decay":
            required.append("lambda_interact")

    if args.topic_bias != "none":
        required.extend(["b", "beta_t"])
        if args.topic_bias == "decay":
            required.append("lambda_b_topic")

    if args.agree_bias != "none":
        required.append("beta_a")
        if args.agree_bias == "decay":
            required.append("lambda_b_agree")

    if args.anchor_bias != "none":
        required.append("beta_c")
        if args.anchor_bias == "decay":
            required.append("lambda_b_anchor")

    missing = [k for k in required if k not in gt]
    if missing:
        raise ValueError(
            "Missing required GT parameters for the current model configuration: "
            + ", ".join(missing)
        )


def find_discussion_id_col(df):
    candidates = [
        "discussion_id", "discussion_idx", "discussion", "disc_id",
        "conversation_id", "conversation_idx", "dialogue_id", "pair_id", "chat_id"
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "Synthetic recursive recovery needs a discussion identifier column. "
        f"None of the expected columns were found: {candidates}"
    )


def preprocess_df(df, topics):
    out = {}

    out["delta_x"] = df["delta_x_i"].to_numpy(dtype="float64") / 4.0
    out["x_i"] = df["x_i"].to_numpy(dtype="float64") / 2.0
    out["H_i"] = df["H_i"].to_numpy(dtype="float64") / np.log(5)
    out["x_j"] = df["x_j"].to_numpy(dtype="float64") / 2.0
    out["H_j"] = (
        df["H_j"].to_numpy(dtype="float64") / np.log(5)
        if "H_j" in df.columns else np.zeros(df.shape[0], dtype="float64")
    )
    out["t_raw"] = df["t"].to_numpy(dtype="float64")
    out["t_data"] = (out["t_raw"] - 1.0) / 4.0
    out["d_data"] = df["framing"].to_numpy(dtype="float64")
    out["topic_matrix"] = df[[f"is_topic_{t}" for t in topics]].to_numpy(dtype="float64")
    out["n_topics"] = out["topic_matrix"].shape[1]
    out["x_0"] = df["x_0"].to_numpy(dtype="float64") / 2.0
    out["is_discussion"] = (
        df["is_discussion"].to_numpy(dtype="float64")
        if "is_discussion" in df.columns else np.ones(df.shape[0], dtype="float64")
    )
    out["is_responder"] = df["is_responder"].to_numpy(dtype="float64")
    topic_to_idx = {t: i for i, t in enumerate(topics)}
    out["topic_idx"] = df["topic"].map(topic_to_idx).to_numpy(dtype="int64")
    out["topic"] = df["topic"].to_numpy()
    out["n_obs"] = df.shape[0]
    return out


def get_topic_value(arr, topic_idx):
    arr = np.asarray(arr, dtype="float64")
    if arr.ndim == 0:
        return float(arr)
    return float(arr[topic_idx])


def compute_mu_sigma_numpy(
    x_i, x_j, x_0, H_i, t_scaled, d_value, topic_idx, is_responder, gt, args
):
    mu = 0.0

    if args.interaction != "none":
        a = get_topic_value(gt["alpha"], topic_idx)
        if args.interaction == "decay":
            a = a * np.exp(-gt["lambda_interact"] * t_scaled)
        mu += a * (x_j - x_i)

    if args.topic_bias != "none":
        b_t = get_topic_value(gt["b"], topic_idx)
        beta_t = get_topic_value(gt["beta_t"], topic_idx)
        if args.topic_bias == "decay":
            beta_t = beta_t * np.exp(-gt["lambda_b_topic"] * t_scaled)
        mu += beta_t * (d_value * b_t - x_i)

    if args.agree_bias != "none":
        beta_a = get_topic_value(gt["beta_a"], topic_idx)
        if args.agree_bias == "decay":
            beta_a = beta_a * np.exp(-gt["lambda_b_agree"] * t_scaled)
        mu += beta_a * (A_TARGET - x_i)

    if args.anchor_bias != "none":
        beta_c = get_topic_value(gt["beta_c"], topic_idx)
        if args.anchor_bias == "decay":
            beta_c = beta_c * np.exp(-gt["lambda_b_anchor"] * t_scaled)
        mu += beta_c * (x_0 - x_i) * float(is_responder)

    sigma = get_topic_value(gt["sigma0"], topic_idx)
    if args.epsilon:
        sigma += gt["epsilon"] * H_i
    sigma += 1e-6

    return mu, sigma


def get_summary_vars(args, recovery=False):
    if not recovery:
        summary_vars = []

        if args.interaction != "none":
            summary_vars.append("alpha_mu")
            if args.interaction == "decay":
                summary_vars.append("lambda_interact")

        if args.topic_bias != "none":
            summary_vars.extend(["b", "beta_t_mu"])
            if args.topic_bias == "decay":
                summary_vars.append("lambda_b_topic")

        if args.agree_bias != "none":
            summary_vars.append("beta_a_mu")
            if args.agree_bias == "decay":
                summary_vars.append("lambda_b_agree")

        if args.anchor_bias != "none":
            summary_vars.append("beta_c_mu")
            if args.anchor_bias == "decay":
                summary_vars.append("lambda_b_anchor")

        summary_vars.append("sigma0_mu")

        if args.epsilon:
            summary_vars.append("epsilon")

        return summary_vars

    summary_vars = ["sigma0"]
    if args.epsilon:
        summary_vars.append("epsilon")

    if args.interaction != "none":
        summary_vars.append("alpha")
        if args.interaction == "decay":
            summary_vars.append("lambda_interact")

    if args.topic_bias != "none":
        summary_vars.extend(["b", "beta_t"])
        if args.topic_bias == "decay":
            summary_vars.append("lambda_b_topic")

    if args.agree_bias != "none":
        summary_vars.append("beta_a")
        if args.agree_bias == "decay":
            summary_vars.append("lambda_b_agree")

    if args.anchor_bias != "none":
        summary_vars.append("beta_c")
        if args.anchor_bias == "decay":
            summary_vars.append("lambda_b_anchor")

    return summary_vars


def flatten_true_params(gt):
    flat = {}
    for name, value in gt.items():
        value = np.asarray(value)
        if value.ndim == 0:
            flat[name] = float(value)
        else:
            for i, v in enumerate(value):
                flat[f"{name}[{i}]"] = float(v)
    return flat


def recovery_report(trace, gt, args):
    summary_vars = get_summary_vars(args, recovery=True)
    summ = az.summary(trace, var_names=summary_vars, kind="stats", round_to=4)
    summ = summ.reset_index().rename(columns={"index": "parameter"})

    truth_flat = flatten_true_params(gt)
    summ["true_value"] = summ["parameter"].map(truth_flat)

    hdi_low_col = None
    hdi_high_col = None
    for c in summ.columns:
        if c.startswith("hdi_") and c.endswith("%"):
            if hdi_low_col is None:
                hdi_low_col = c
            else:
                hdi_high_col = c

    if hdi_low_col is not None and hdi_high_col is not None:
        summ["covered_hdi"] = summ.apply(
            lambda row: (
                row[hdi_low_col] <= row["true_value"] <= row[hdi_high_col]
                if pd.notnull(row["true_value"])
                else np.nan
            ),
            axis=1,
        )
    else:
        summ["covered_hdi"] = np.nan

    return summ


def build_empirical_init_pools(df, topic_to_idx):
    disc_col = find_discussion_id_col(df)
    work = df.copy().reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    work["_t_sort"] = work["t"].astype("float64")

    first_rows = (
        work.sort_values([disc_col, "_t_sort", "_orig_idx"])
            .groupby([disc_col, "is_responder"], as_index=False)
            .first()
    )

    x0_pools = {}
    xi_pools = {}
    xi_role_pools = {0: [], 1: []}
    x0_global = []

    for topic_name in topic_to_idx:
        x0_topic = df.loc[df["topic"] == topic_name, "x_0"].to_numpy(dtype="float64") / 2.0
        x0_pools[topic_name] = x0_topic
        x0_global.extend(x0_topic.tolist())

        for role in [0, 1]:
            vals = first_rows.loc[
                (first_rows["topic"] == topic_name) & (first_rows["is_responder"].astype(int) == role),
                "x_i"
            ].to_numpy(dtype="float64") / 2.0
            xi_pools[(topic_name, role)] = vals
            xi_role_pools[role].extend(vals.tolist())

    x0_global = np.asarray(x0_global, dtype="float64")
    xi_role_pools = {k: np.asarray(v, dtype="float64") for k, v in xi_role_pools.items()}

    return x0_pools, x0_global, xi_pools, xi_role_pools


def draw_idealized_from_pool(pool, fallback_pool, rng, lower=-1.0, upper=1.0):
    pool = np.asarray(pool, dtype="float64")
    fallback_pool = np.asarray(fallback_pool, dtype="float64")

    base = pool if pool.size > 1 else fallback_pool
    if base.size == 0:
        return 0.0

    mu = float(np.mean(base))
    sd = float(np.std(base))

    if not np.isfinite(sd) or sd < 1e-6:
        return float(np.clip(mu, lower, upper))

    return float(np.clip(rng.normal(mu, sd), lower, upper))


def simulate_empirical_delta_x_with_pymc(data, gt, args):
    print("Generating empirical-recovery synthetic outcomes with fixed GT parameters...\n", flush=True)
    model = build_pymc_model(
        data=data,
        args=args,
        observed_delta_x=None,
        fixed_params=gt,
        simulation_var_name="y_sim",
    )
    with model:
        prior_draw = pm.sample_prior_predictive(
            draws=1,
            var_names=["y_sim"],
            random_seed=args.random_seed,
            return_inferencedata=False,
        )
    return np.asarray(prior_draw["y_sim"], dtype="float64").reshape(-1)


def simulate_recursive_synthetic_df(df_template, gt, args, topics, rng):
    print("Generating fully synthetic recursive dataset...\n", flush=True)

    disc_col = find_discussion_id_col(df_template)
    topic_to_idx = {t: i for i, t in enumerate(topics)}

    x0_pools, x0_global, xi_pools, xi_role_pools = build_empirical_init_pools(df_template, topic_to_idx)

    work = df_template.copy().reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    work["_t_sort"] = work["t"].astype("float64")

    sim_rows = []

    for _, g in work.groupby(disc_col, sort=False):
        g = g.sort_values(["_t_sort", "_orig_idx"]).copy()

        topic_name = g["topic"].iloc[0]
        topic_idx = topic_to_idx[topic_name]

        responder_anchor = draw_idealized_from_pool(
            x0_pools.get(topic_name, np.array([])),
            x0_global,
            rng,
        )

        states = {
            0: draw_idealized_from_pool(
                xi_pools.get((topic_name, 0), np.array([])),
                xi_role_pools[0],
                rng,
            ),
            1: draw_idealized_from_pool(
                xi_pools.get((topic_name, 1), np.array([])),
                xi_role_pools[1],
                rng,
            ),
        }

        for row in g.itertuples(index=False):
            speaker = int(row.is_responder)
            other = 1 - speaker

            x_i_scaled = float(states[speaker])
            x_j_scaled = float(states[other])
            x_0_scaled = float(responder_anchor)
            H_i_scaled = float(row.H_i) / np.log(5)
            t_scaled = (float(row.t) - 1.0) / 4.0
            d_value = float(row.framing)

            mu, sigma = compute_mu_sigma_numpy(
                x_i=x_i_scaled,
                x_j=x_j_scaled,
                x_0=x_0_scaled,
                H_i=H_i_scaled,
                t_scaled=t_scaled,
                d_value=d_value,
                topic_idx=topic_idx,
                is_responder=speaker,
                gt=gt,
                args=args,
            )

            delta_x_scaled = rng.normal(mu, sigma)
            states[speaker] = x_i_scaled + delta_x_scaled

            row_dict = row._asdict()
            row_dict["x_i"] = 2.0 * x_i_scaled
            row_dict["x_j"] = 2.0 * x_j_scaled
            row_dict["x_0"] = 2.0 * x_0_scaled
            row_dict["delta_x_i"] = 4.0 * delta_x_scaled
            sim_rows.append(row_dict)

    sim_df = pd.DataFrame(sim_rows).sort_values("_orig_idx").drop(columns=["_orig_idx", "_t_sort"])
    return sim_df


def build_pymc_model(data, args, observed_delta_x=None, fixed_params=None, simulation_var_name=None):
    n_topics = data["n_topics"]

    def fixed_tensor(name, value, shape=None):
        value = np.asarray(value, dtype="float64")
        if shape is not None:
            if isinstance(shape, int):
                shape = (shape,)
            if value.ndim == 0:
                value = np.full(shape, float(value), dtype="float64")
            elif value.shape != shape:
                raise ValueError(
                    f"Fixed parameter '{name}' expected shape {shape}, got {value.shape}."
                )
        return pm.Data(name, value)

    with pm.Model() as model:
        x_i = pm.Data("x_i", np.asarray(data["x_i"], dtype="float64"))
        H_i = pm.Data("H_i", np.asarray(data["H_i"], dtype="float64"))
        x_j = pm.Data("x_j", np.asarray(data["x_j"], dtype="float64"))
        t_data = pm.Data("t_data", np.asarray(data["t_data"], dtype="float64"))
        d_data = pm.Data("d_data", np.asarray(data["d_data"], dtype="float64"))
        topic_matrix = pm.Data("topic_matrix", np.asarray(data["topic_matrix"], dtype="float64"))
        x_0 = pm.Data("x_0", np.asarray(data["x_0"], dtype="float64"))
        is_responder = pm.Data("is_responder", np.asarray(data["is_responder"], dtype="float64"))
        topic_idx = pm.Data("topic_idx", np.asarray(data["topic_idx"], dtype="int64"))

        def nc_normal(name, mu, sd, shape):
            z = pm.Normal(f"{name}_z", 0.0, 1.0, shape=shape)
            return pm.Deterministic(name, mu + sd * z)

        # -------------------- baseline SD
        if fixed_params is not None and "sigma0" in fixed_params:
            sigma0 = fixed_tensor("sigma0", fixed_params["sigma0"], shape=n_topics)
            sigma0_mu = pm.Deterministic("sigma0_mu", pt.mean(sigma0))
            s0 = sigma0[topic_idx]
        else:
            log_sigma0_mu = pm.Normal("log_sigma0_mu", np.log(0.1), 1)
            log_sigma0_sd = pm.HalfNormal("log_sigma0_sd", 0.3)
            log_sigma0 = nc_normal("log_sigma0", log_sigma0_mu, log_sigma0_sd, n_topics)
            sigma0 = pm.Deterministic("sigma0", pm.math.log1pexp(log_sigma0))
            sigma0_mu = pm.Deterministic("sigma0_mu", pm.math.log1pexp(log_sigma0_mu))
            s0 = sigma0[topic_idx]

        # -------------------- optional epsilon SD term
        if args.epsilon:
            if fixed_params is not None and "epsilon" in fixed_params:
                epsilon = fixed_tensor("epsilon", fixed_params["epsilon"])
            else:
                log_epsilon = pm.Normal("log_epsilon", np.log(0.1), 1)
                epsilon = pm.Deterministic("epsilon", pm.math.log1pexp(log_epsilon))
            e = epsilon
        else:
            e = 0.0

        # -------------------- build mu from selected blocks
        mu = pt.zeros_like(x_i, dtype="float64")

        # interaction term
        if args.interaction != "none":
            if fixed_params is not None and "alpha" in fixed_params:
                alpha = fixed_tensor("alpha", fixed_params["alpha"], shape=n_topics)
            else:
                alpha_mu = pm.Normal("alpha_mu", 0, param_SD)
                alpha_sd = pm.HalfNormal("alpha_sd", param_SD)
                alpha = nc_normal("alpha", alpha_mu, alpha_sd, n_topics)
            a = alpha[topic_idx]

            if args.interaction == "decay":
                if fixed_params is not None and "lambda_interact" in fixed_params:
                    lambda_interact = fixed_tensor("lambda_interact", fixed_params["lambda_interact"])
                else:
                    log_lambda_interact = pm.Normal("log_lambda_interact", mu=np.log(1), sigma=1)
                    lambda_interact = pm.Deterministic("lambda_interact", pm.math.exp(log_lambda_interact))
                interaction = a * pm.math.exp(-lambda_interact * t_data)
            else:
                interaction = a

            interaction_term = pm.Deterministic("interaction_term", interaction * (x_j - x_i))
            mu = mu + interaction_term

        # topic bias term
        if args.topic_bias != "none":
            if fixed_params is not None and "b" in fixed_params:
                b = fixed_tensor("b", fixed_params["b"], shape=n_topics)
            else:
                b = pm.Uniform("b", lower=-1, upper=1, shape=n_topics)
            b_t = pm.math.dot(topic_matrix, b)

            if fixed_params is not None and "beta_t" in fixed_params:
                beta_t = fixed_tensor("beta_t", fixed_params["beta_t"], shape=n_topics)
                beta_t_mu = pm.Deterministic("beta_t_mu", pt.mean(beta_t))
            else:
                log_beta_t_mu = pm.Normal("log_beta_t_mu", log_beta_mu_MU, log_beta_mu_SD)
                log_beta_t_sd = pm.HalfNormal("log_beta_t_sd", log_beta_sd_SD)
                log_beta_t = nc_normal("log_beta_t", log_beta_t_mu, log_beta_t_sd, n_topics)
                beta_t = pm.Deterministic("beta_t", pm.math.log1pexp(log_beta_t * 4) / 4)
                beta_t_mu = pm.Deterministic("beta_t_mu", pm.math.log1pexp(log_beta_t_mu * 4) / 4)
            bt = beta_t[topic_idx]

            if args.topic_bias == "decay":
                if fixed_params is not None and "lambda_b_topic" in fixed_params:
                    lambda_b_topic = fixed_tensor("lambda_b_topic", fixed_params["lambda_b_topic"])
                else:
                    log_lambda_b_topic = pm.Normal("log_lambda_b_topic", mu=np.log(1), sigma=1)
                    lambda_b_topic = pm.Deterministic("lambda_b_topic", pm.math.exp(log_lambda_b_topic))
                k_topic = bt * pm.math.exp(-lambda_b_topic * t_data)
            else:
                k_topic = bt

            topic_term = pm.Deterministic("topic_term", k_topic * (d_data * b_t - x_i))
            mu = mu + topic_term

        # agree bias term
        if args.agree_bias != "none":
            if fixed_params is not None and "beta_a" in fixed_params:
                beta_a = fixed_tensor("beta_a", fixed_params["beta_a"], shape=n_topics)
            else:
                beta_a_mu = pm.Normal("beta_a_mu", 0, param_SD)
                beta_a_sd = pm.HalfNormal("beta_a_sd", param_SD)
                beta_a = nc_normal("beta_a", beta_a_mu, beta_a_sd, n_topics)
            ba = beta_a[topic_idx]

            if args.agree_bias == "decay":
                if fixed_params is not None and "lambda_b_agree" in fixed_params:
                    lambda_b_agree = fixed_tensor("lambda_b_agree", fixed_params["lambda_b_agree"])
                else:
                    log_lambda_b_agree = pm.Normal("log_lambda_b_agree", mu=np.log(1), sigma=1)
                    lambda_b_agree = pm.Deterministic("lambda_b_agree", pm.math.exp(log_lambda_b_agree))
                k_agree = ba * pm.math.exp(-lambda_b_agree * t_data)
            else:
                k_agree = ba

            agree_term = pm.Deterministic("agree_term", k_agree * (A_TARGET - x_i))
            mu = mu + agree_term

        # anchor bias term
        if args.anchor_bias != "none":
            if fixed_params is not None and "beta_c" in fixed_params:
                beta_c = fixed_tensor("beta_c", fixed_params["beta_c"], shape=n_topics)
            else:
                beta_c_mu = pm.Normal("beta_c_mu", 0, param_SD)
                beta_c_sd = pm.HalfNormal("beta_c_sd", param_SD)
                beta_c = nc_normal("beta_c", beta_c_mu, beta_c_sd, n_topics)
            bc = beta_c[topic_idx]

            if args.anchor_bias == "decay":
                if fixed_params is not None and "lambda_b_anchor" in fixed_params:
                    lambda_b_anchor = fixed_tensor("lambda_b_anchor", fixed_params["lambda_b_anchor"])
                else:
                    log_lambda_b_anchor = pm.Normal("log_lambda_b_anchor", mu=np.log(1), sigma=1)
                    lambda_b_anchor = pm.Deterministic("lambda_b_anchor", pm.math.exp(log_lambda_b_anchor))
                k_anchor = bc * pm.math.exp(-lambda_b_anchor * t_data)
            else:
                k_anchor = bc

            anchor_term = pm.Deterministic(
                "anchor_term",
                k_anchor * (x_0 - x_i) * is_responder,
            )
            mu = mu + anchor_term

        mu_det = pm.Deterministic("mu", mu)
        sigma = pm.Deterministic("sigma", s0 + e * H_i + 1e-6)

        if simulation_var_name is not None:
            pm.Normal(
                simulation_var_name,
                mu=mu_det,
                sigma=sigma,
                shape=x_i.shape,
            )
        else:
            pm.Normal(
                "y_obs",
                mu=mu_det,
                sigma=sigma,
                observed=np.asarray(observed_delta_x, dtype="float64"),
            )

    return model


# -------------------- load empirical dataframe
df = pd.read_parquet(
    "../../data/inference_ready_data_all_discussions_s=25_l=5.parquet"
),
df = df[df["model"] == args.llm].copy()

print(f"Data loaded for model {args.llm}, n={df.shape[0]} rows.\n", flush=True)

data_for_fit = preprocess_df(df, topics)
gt = None
synthetic_df = None

if args.recovery is not None:
    if args.gt_file is None:
        raise ValueError("--gt_file is required when --recovery is used.")

    raw_gt = load_json(args.gt_file)
    gt = normalize_gt_params(raw_gt, n_topics=data_for_fit["n_topics"])
    validate_gt_params(gt, args)

    print("Loaded and normalized GT parameters from JSON.\n", flush=True)

    if args.recovery == "empirical":
        delta_x_sim = simulate_empirical_delta_x_with_pymc(data_for_fit, gt, args)
        data_for_fit["delta_x"] = delta_x_sim
        print("Empirical recovery dataset generated.\n", flush=True)

    elif args.recovery == "synthetic":
        rng = np.random.default_rng(args.random_seed)
        synthetic_df = simulate_recursive_synthetic_df(df, gt, args, topics, rng)
        data_for_fit = preprocess_df(synthetic_df, topics)
        print(f"Synthetic recursive dataset generated, n={synthetic_df.shape[0]} rows.\n", flush=True)

if args.mode == "sample":
    print("Starting model inference...\n", flush=True)

# -------------------- fit model
model = build_pymc_model(
    data=data_for_fit,
    args=args,
    observed_delta_x=data_for_fit["delta_x"],
    fixed_params=None,
    simulation_var_name=None,
)

if args.mode == "sample":
    with model:
        trace = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            max_treedepth=args.max_treedepth,
            nuts_sampler="nutpie",
            random_seed=args.random_seed,
        )

# -------------------- prior predictive
print("Running prior predictive checks...\n", flush=True)
with model:
    prior = pm.sample_prior_predictive(draws=500, random_seed=args.random_seed)

y_pp = prior.prior_predictive["y_obs"].values
y_flat = y_pp.reshape(-1)
p_out = np.mean(np.abs(y_flat) > 1.0)
minv, maxv = np.nanmin(y_flat), np.nanmax(y_flat)
qs = np.nanpercentile(y_flat, [1, 5, 50, 95, 99])

print("\nP(|y|>1):", p_out)
print("min/max:", minv, maxv)
print("quantiles:", qs, "\n")

if args.mode == "sample":
    model_tag = build_model_tag(args)
    gt_stem = Path(args.gt_file).stem if args.gt_file is not None else "nogt"

    trace_dir = Path("../../data/recovery/recovery_traces")
    trace_dir.mkdir(parents=True, exist_ok=True)

    if args.recovery is None:
        output_path = trace_dir / (
            f"{model_tag}_{args.llm}_dr{args.draws}_tu{args.tune}_"
            f"ch{args.chains}_co{args.cores}_ar{args.target_accept}_td{args.max_treedepth}.nc"
        )
    else:
        output_path = trace_dir / (
            f"{model_tag}_{args.llm}_{gt_stem}_dr{args.draws}_tu{args.tune}_"
            f"ch{args.chains}_co{args.cores}_ar{args.target_accept}_td{args.max_treedepth}.nc"
        )

    az.to_netcdf(trace, output_path)
    print(f"Trace saved to {output_path}\n", flush=True)

    # -------------------- save synthetic recovery dataset if relevant
    if synthetic_df is not None:
        recovery_dir = Path("../../data/recovery/recovery_data")
        recovery_dir.mkdir(parents=True, exist_ok=True)
        sim_path = recovery_dir / f"{model_tag}_{args.llm}_{gt_stem}_synthetic_recursive.parquet"
        synthetic_df.to_parquet(sim_path, index=False)
        print(f"Synthetic dataset saved to {sim_path}\n", flush=True)

    if args.recovery == "empirical":
        recovery_dir = Path("../../data/recovery/recovery_data")
        recovery_dir.mkdir(parents=True, exist_ok=True)
        empirical_df = df.copy()
        empirical_df["delta_x_i"] = 4.0 * data_for_fit["delta_x"]
        sim_path = recovery_dir / f"{model_tag}_{args.llm}_{gt_stem}_empirical_recovery.parquet"
        empirical_df.to_parquet(sim_path, index=False)
        print(f"Empirical-recovery dataset saved to {sim_path}\n", flush=True)

    # -------------------- summaries
    if args.recovery is None:
        summary_vars = get_summary_vars(args, recovery=False)
        print(pm.summary(trace, var_names=summary_vars, round_to=2), "\n", flush=True)
    else:
        report = recovery_report(trace, gt, args)
        print(report.to_string(index=False), "\n", flush=True)

        recovery_dir = Path("../../data/recovery/recovery_reports")
        recovery_dir.mkdir(parents=True, exist_ok=True)
        report_path = recovery_dir / f"{model_tag}_{args.llm}_{gt_stem}_recovery_report.csv"
        report.to_csv(report_path, index=False)
        print(f"Recovery report saved to {report_path}\n", flush=True)

    print("Divergences:", trace.sample_stats.diverging.values.sum(), "\n", flush=True)

tock = time.time()
print(f"Total time elapsed: {(tock - tick) / 60} minutes.", flush=True)