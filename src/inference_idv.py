import os
import time
import argparse

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

args = parser.parse_args()

print("Arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
print("\n", flush=True)

CLIMATE_TOPIC = "Climate_Change"
A_TARGET = 1.0


def build_model_tag():
    return "climate_idv_attractors_full_decay_eps"

# -------------------- load data
if args.llm == "mixtral-finetuned":
    df = pd.read_csv(
        "../../data/mixtral_tuned_clim_data.csv"
    ).copy()

    # old file is already climate-only; keep topic name consistent
    df["topic"] = CLIMATE_TOPIC
else:
    df = pd.read_parquet(
        "../../data/inference_ready_data_all_discussions_s=25_l=5.parquet"
    )
    df = df[(df["model"] == args.llm) & (df["topic"] == CLIMATE_TOPIC)].copy()

if df.empty:
    raise ValueError(
        f"No rows found for model={args.llm} and topic={CLIMATE_TOPIC}."
    )

if "init_x_i" not in df.columns:
    raise KeyError(
        "Column 'init_x_i' is required for individual attractors but was not found."
    )

print(
    f"Data loaded for model {args.llm}, topic {CLIMATE_TOPIC}, n={df.shape[0]} rows.\n",
    flush=True,
)

# -------------------- extract raw data
if args.llm == "mixtral-finetuned":
    delta_x = df["dx"].to_numpy(dtype="float64")
    x_i = df["x_i"].to_numpy(dtype="float64")
    H_i = df["H_i"].to_numpy(dtype="float64")
    x_j = df["x_j"].to_numpy(dtype="float64")
    t_data = df["t"].to_numpy(dtype="float64") + 1.0  # shift time to start from 1 for consistency with new script
    d_data = df["delta"].to_numpy(dtype="float64")
    init_x_i_raw = df["init_x_i"].to_numpy(dtype="float64")

    # reconstruct x_0 from the old climate-only dataset
    x_0 = np.empty(len(df), dtype="float64")
    current_x0 = np.nan
    for i in range(len(df)):
        if t_data[i] == 1 and df["is_initiator"].iloc[i] == 1:
            current_x0 = x_i[i]   # first opinion of the discussion
        x_0[i] = current_x0

    # old convention: responder iff not initiator
    is_responder = (df["is_initiator"].to_numpy(dtype="float64") != 1).astype("float64")
else:
    delta_x = df["delta_x_i"].to_numpy(dtype="float64")
    x_i = df["x_i"].to_numpy(dtype="float64")
    H_i = df["H_i"].to_numpy(dtype="float64")
    x_j = df["x_j"].to_numpy(dtype="float64")
    t_data = df["t"].to_numpy(dtype="float64") # already shifted to start from 1 in preprocessing
    d_data = df["framing"].to_numpy(dtype="float64")
    x_0 = df["x_0"].to_numpy(dtype="float64")
    is_responder = df["is_responder"].to_numpy(dtype="float64")
    init_x_i_raw = df["init_x_i"].to_numpy(dtype="float64")

# initial opinion used only for one-hot climate attractors
init_x_i_raw = df["init_x_i"].to_numpy(dtype="float64")

# -------------------- one-hot encoding for initial opinion in {-2, -1, 0, 1, 2}
init_levels = np.array([-2, -1, 0, 1, 2], dtype="float64")
op_matrix = np.column_stack(
    [np.isclose(init_x_i_raw, level).astype("float64") for level in init_levels]
)

row_sums = op_matrix.sum(axis=1)
if not np.all(row_sums == 1):
    bad_vals = np.unique(init_x_i_raw[row_sums != 1])
    raise ValueError(
        "Each row must map to exactly one initial-opinion bin in {-2,-1,0,1,2}. "
        f"Found problematic values: {bad_vals}"
    )

# -------------------- rescaling (same convention as new script)
delta_x = delta_x / 4.0
x_i = x_i / 2.0
x_j = x_j / 2.0
x_0 = x_0 / 2.0

t_data = (t_data - 1.0) / 4.0
H_i = H_i / np.log(5.0)

if args.mode == "sample":
    print("Starting model inference...\n", flush=True)

# -------------------- scalar prior scales (non-hierarchical analogue of new script)
coef_SD = 0.15
log_beta_SD = 0.5

with pm.Model() as model:
    # -------------------- baseline SD
    log_sigma0 = pm.Normal("log_sigma0", mu=np.log(0.1), sigma=1.0)
    sigma0 = pm.Deterministic("sigma0", pm.math.log1pexp(log_sigma0))

    # -------------------- epsilon SD term
    log_epsilon = pm.Normal("log_epsilon", mu=np.log(0.1), sigma=1.0)
    epsilon = pm.Deterministic("epsilon", pm.math.log1pexp(log_epsilon))

    # -------------------- interaction term (always included, with decay)
    alpha = pm.Normal("alpha", mu=0.0, sigma=coef_SD)
    log_lambda_interact = pm.Normal("log_lambda_interact", mu=np.log(1.0), sigma=1.0)
    lambda_interact = pm.Deterministic(
        "lambda_interact", pm.math.exp(log_lambda_interact)
    )
    interaction = alpha * pm.math.exp(-lambda_interact * t_data)
    interaction_term = pm.Deterministic(
        "interaction_term", interaction * (x_j - x_i)
    )

    # -------------------- climate topic bias with initial-opinion-specific attractors
    # attractors live on the rescaled opinion axis [-1, 1]
    b = pm.Uniform("b", lower=-1.0, upper=1.0, shape=5)
    b_t = pm.Deterministic("b_t", pm.math.dot(op_matrix, b))

    log_beta_t = pm.Normal("log_beta_t", mu=0.0, sigma=log_beta_SD)
    beta_t = pm.Deterministic("beta_t", pm.math.log1pexp(log_beta_t * 4.0) / 4.0)

    log_lambda_b_topic = pm.Normal("log_lambda_b_topic", mu=np.log(1.0), sigma=1.0)
    lambda_b_topic = pm.Deterministic(
        "lambda_b_topic", pm.math.exp(log_lambda_b_topic)
    )
    k_topic = beta_t * pm.math.exp(-lambda_b_topic * t_data)

    topic_term = pm.Deterministic(
        "topic_term",
        k_topic * (d_data * b_t - x_i),
    )

    # -------------------- agree bias (always included, with decay)
    beta_a = pm.Normal("beta_a", mu=0.0, sigma=coef_SD)
    log_lambda_b_agree = pm.Normal("log_lambda_b_agree", mu=np.log(1.0), sigma=1.0)
    lambda_b_agree = pm.Deterministic(
        "lambda_b_agree", pm.math.exp(log_lambda_b_agree)
    )
    k_agree = beta_a * pm.math.exp(-lambda_b_agree * t_data)

    agree_term = pm.Deterministic(
        "agree_term",
        k_agree * (A_TARGET - x_i),
    )

    # -------------------- anchor bias (always included, with decay)
    beta_c = pm.Normal("beta_c", mu=0.0, sigma=coef_SD)
    log_lambda_b_anchor = pm.Normal("log_lambda_b_anchor", mu=np.log(1.0), sigma=1.0)
    lambda_b_anchor = pm.Deterministic(
        "lambda_b_anchor", pm.math.exp(log_lambda_b_anchor)
    )
    k_anchor = beta_c * pm.math.exp(-lambda_b_anchor * t_data)

    anchor_term = pm.Deterministic(
        "anchor_term",
        k_anchor * (x_0 - x_i) * is_responder,
    )

    # -------------------- combined mean and SD
    mu = pt.zeros_like(x_i, dtype="float64")
    mu = mu + interaction_term
    mu = mu + topic_term
    mu = mu + agree_term
    mu = mu + anchor_term
    mu_det = pm.Deterministic("mu", mu)

    sigma = sigma0 + epsilon * H_i + 1e-6

    # -------------------- likelihood
    y_obs = pm.Normal(
        "y_obs",
        mu=mu_det,
        sigma=sigma,
        observed=delta_x,
    )

    if args.mode == "sample":
        trace = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            max_treedepth=args.max_treedepth,
            nuts_sampler="nutpie",
        )

# -------------------- prior predictive
print("Running prior predictive checks...\n", flush=True)
with model:
    prior = pm.sample_prior_predictive(samples=500, random_seed=42)

y_pp = prior.prior_predictive["y_obs"].values
y_flat = y_pp.reshape(-1)
p_out = np.mean(np.abs(y_flat) > 1.0)
minv, maxv = np.nanmin(y_flat), np.nanmax(y_flat)
qs = np.nanpercentile(y_flat, [1, 5, 50, 95, 99])

print("\nP(|y|>1):", p_out)
print("min/max:", minv, maxv)
print("quantiles:", qs, "\n")

if args.mode == "sample":
    # -------------------- save trace
    model_tag = build_model_tag()
    output_path = (
        f"../../data/traces/idv_model/"
        f"{model_tag}_{args.llm}_dr{args.draws}_tu{args.tune}_"
        f"ch{args.chains}_co{args.cores}_ar{args.target_accept}_td{args.max_treedepth}.nc"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    az.to_netcdf(trace, output_path)
    print(f"Trace saved to {output_path}\n", flush=True)

    summary_vars = [
        "alpha",
        "lambda_interact",
        "b",
        "beta_t",
        "lambda_b_topic",
        "beta_a",
        "lambda_b_agree",
        "beta_c",
        "lambda_b_anchor",
        "sigma0",
        "epsilon",
    ]

    print(pm.summary(trace, var_names=summary_vars, round_to=2), "\n", flush=True)
    print("Divergences:", trace.sample_stats.diverging.values.sum(), "\n", flush=True)

tock = time.time()
print(f"Total time elapsed: {(tock - tick) / 60} minutes.", flush=True)