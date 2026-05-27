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

# print all args for logging purposes
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


def build_model_tag(args):
    eps_tag = "eps" if args.epsilon else "noeps"
    return (
        f"I-{args.interaction}"
        f"__T-{args.topic_bias}"
        f"__A-{args.agree_bias}"
        f"__C-{args.anchor_bias}"
        f"__E-{eps_tag}"
    )

df = pd.read_parquet(
    "../data/inference_ready_data_all_discussions_s=25_l=5.parquet"
)
df = df[df["model"] == args.llm]

print(f"Data loaded for model {args.llm}, n={df.shape[0]} rows.\n", flush=True)

delta_x = df.loc[:, "delta_x_i"].to_numpy()
x_i = df.loc[:, "x_i"].to_numpy()
H_i = df.loc[:, "H_i"].to_numpy()
x_j = df.loc[:, "x_j"].to_numpy()
H_j = df.loc[:, "H_j"].to_numpy()
t_data = df.loc[:, "t"].to_numpy()
d_data = df.loc[:, "framing"].to_numpy()
topic_matrix = df[[f"is_topic_{t}" for t in topics]].to_numpy()
n_topics = topic_matrix.shape[1]
x_0 = df.loc[:, "x_0"].to_numpy()
is_discussion = df.loc[:, "is_discussion"].to_numpy()
is_responder = df.loc[:, "is_responder"].to_numpy()

topic_to_idx = {t: i for i, t in enumerate(topics)}
topic_idx = df["topic"].map(topic_to_idx).to_numpy(dtype="int64")

# -------------------- rescaling
delta_x = delta_x / 4
x_i = x_i / 2
x_j = x_j / 2
x_0 = x_0 / 2
A_TARGET = 1

t_data = t_data.astype("float64")
t_data = (t_data - 1) / 4

H_i = H_i.astype("float64")
H_i = H_i / np.log(5)

if args.mode == "sample":
    print("Starting model inference...\n", flush=True)


param_SD = 0.1
log_lam_mu_SD = 0.5
log_lam_sd_SD = 0.3
log_beta_mu_MU = 0
log_beta_mu_SD = 0.5
log_beta_sd_SD = 0.3
topic_SD = 1


with pm.Model() as model:

    def nc_normal(name, mu, sd, shape):
        z = pm.Normal(f"{name}_z", 0.0, 1.0, shape=shape)
        return pm.Deterministic(name, mu + sd * z)

    # -------------------- always-present baseline SD
    log_sigma0_mu = pm.Normal("log_sigma0_mu", np.log(0.1), 1)
    log_sigma0_sd = pm.HalfNormal("log_sigma0_sd", 0.3)
    log_sigma0 = nc_normal("log_sigma0", log_sigma0_mu, log_sigma0_sd, n_topics)
    sigma0 = pm.Deterministic("sigma0", pm.math.log1pexp(log_sigma0))
    sigma0_mu = pm.Deterministic("sigma0_mu", pm.math.log1pexp(log_sigma0_mu))
    s0 = sigma0[topic_idx]

    # -------------------- optional epsilon SD term
    if args.epsilon:
        log_epsilon = pm.Normal("log_epsilon", np.log(0.1), 1)
        epsilon = pm.Deterministic("epsilon", pm.math.log1pexp(log_epsilon))
        e = epsilon
    else:
        e = 0.0

    # -------------------- build mu from selected blocks
    mu = pt.zeros_like(x_i, dtype="float64")

    # interaction term
    if args.interaction != "none":
        alpha_mu = pm.Normal("alpha_mu", 0, param_SD)
        alpha_sd = pm.HalfNormal("alpha_sd", param_SD)
        alpha = nc_normal("alpha", alpha_mu, alpha_sd, n_topics)
        a = alpha[topic_idx]

        if args.interaction == "decay":
            log_lambda_interact = pm.Normal("log_lambda_interact", mu=np.log(1), sigma=1)
            lambda_interact = pm.Deterministic("lambda_interact", pm.math.exp(log_lambda_interact))
            interaction = a * pm.math.exp(-lambda_interact * t_data)
        else:
            interaction = a

        interaction_term = pm.Deterministic("interaction_term", interaction * (x_j - x_i))
        mu = mu + interaction_term

    # topic bias term
    if args.topic_bias != "none":
        b = pm.Uniform("b", lower=-1, upper=1, shape=n_topics)
        b_t = pm.math.dot(topic_matrix, b)

        log_beta_t_mu = pm.Normal("log_beta_t_mu", log_beta_mu_MU, log_beta_mu_SD)
        log_beta_t_sd = pm.HalfNormal("log_beta_t_sd", log_beta_sd_SD)
        log_beta_t = nc_normal("log_beta_t", log_beta_t_mu, log_beta_t_sd, n_topics)

        beta_t = pm.Deterministic("beta_t", pm.math.log1pexp(log_beta_t * 4) / 4)
        beta_t_mu = pm.Deterministic("beta_t_mu", pm.math.log1pexp(log_beta_t_mu * 4) / 4)
        bt = beta_t[topic_idx]

        if args.topic_bias == "decay":
            log_lambda_b_topic = pm.Normal("log_lambda_b_topic", mu=np.log(1), sigma=1)
            lambda_b_topic = pm.Deterministic("lambda_b_topic", pm.math.exp(log_lambda_b_topic))
            k_topic = bt * pm.math.exp(-lambda_b_topic * t_data)
        else:
            k_topic = bt

        topic_term = pm.Deterministic("topic_term", k_topic * (d_data * b_t - x_i))
        mu = mu + topic_term

    # agree bias term
    if args.agree_bias != "none":
        beta_a_mu = pm.Normal("beta_a_mu", 0, param_SD)
        beta_a_sd = pm.HalfNormal("beta_a_sd", param_SD)
        beta_a = nc_normal("beta_a", beta_a_mu, beta_a_sd, n_topics)
        ba = beta_a[topic_idx]

        if args.agree_bias == "decay":
            log_lambda_b_agree = pm.Normal("log_lambda_b_agree", mu=np.log(1), sigma=1)
            lambda_b_agree = pm.Deterministic("lambda_b_agree", pm.math.exp(log_lambda_b_agree))
            k_agree = ba * pm.math.exp(-lambda_b_agree * t_data)
        else:
            k_agree = ba

        agree_term = pm.Deterministic("agree_term", k_agree * (A_TARGET - x_i))
        mu = mu + agree_term

    # anchor bias term
    if args.anchor_bias != "none":
        beta_c_mu = pm.Normal("beta_c_mu", 0, param_SD)
        beta_c_sd = pm.HalfNormal("beta_c_sd", param_SD)
        beta_c = nc_normal("beta_c", beta_c_mu, beta_c_sd, n_topics)
        bc = beta_c[topic_idx]

        if args.anchor_bias == "decay":
            log_lambda_b_anchor = pm.Normal("log_lambda_b_anchor", mu=np.log(1), sigma=1)
            lambda_b_anchor = pm.Deterministic("lambda_b_anchor", pm.math.exp(log_lambda_b_anchor))
            k_anchor = bc * pm.math.exp(-lambda_b_anchor * t_data)
        else:
            k_anchor = bc

        anchor_term = pm.Deterministic(
            "anchor_term",
            k_anchor * (x_0 - x_i) * is_responder
        )
        mu = mu + anchor_term

    # full combined mean term and SD
    mu_det = pm.Deterministic("mu", mu)

    sigma = s0 + e * H_i + 1e-6  # numerical stability

    # likelihood
    y_obs = pm.Normal(
        "y_obs",
        mu=mu_det,
        sigma=sigma,
        observed=delta_x,
    )

    if args.mode == "sample":
        # start sampling
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
    model_tag = build_model_tag(args)
    output_dir = (
        f"../data/traces/"
        f"{model_tag}_{args.llm}_dr{args.draws}_tu{args.tune}_"
        f"ch{args.chains}_co{args.cores}_ar{args.target_accept}_td{args.max_treedepth}.nc"
    )
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    az.to_netcdf(trace, output_dir)
    print(f"Trace saved to {output_dir}\n", flush=True)

    summary_vars = []

    # -------------------- interaction
    if args.interaction != "none":
        summary_vars.append("alpha_mu")
        if args.interaction == "decay":
            summary_vars.append("lambda_interact")

    # -------------------- topic bias
    if args.topic_bias != "none":
        summary_vars.extend(["b", "beta_t_mu"])
        if args.topic_bias == "decay":
            summary_vars.append("lambda_b_topic")

    # -------------------- agree bias
    if args.agree_bias != "none":
        summary_vars.append("beta_a_mu")
        if args.agree_bias == "decay":
            summary_vars.append("lambda_b_agree")

    # -------------------- anchor bias
    if args.anchor_bias != "none":
        summary_vars.append("beta_c_mu")
        if args.anchor_bias == "decay":
            summary_vars.append("lambda_b_anchor")

    # -------------------- noise
    summary_vars.append("sigma0_mu")

    if args.epsilon:
        summary_vars.append("epsilon")

    print(pm.summary(trace, var_names=summary_vars, round_to=2), "\n", flush=True)
    print("Divergences:", trace.sample_stats.diverging.values.sum(), "\n", flush=True)

tock = time.time()
print(f"Total time elapsed: {(tock - tick) / 60} minutes.", flush=True)