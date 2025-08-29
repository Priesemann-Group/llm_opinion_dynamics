import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

data_paths = {"dm_all": "../data/simulations/processed/dolphin_mixtral_data.csv",
              "mixtral_all": "../data/simulations/processed/mixtral_8x7b_data.csv",
              "gpt_all": "../data/simulations/processed/gpt_4o_mini_data.csv",
              "tuned_clim": "../data/simulations/processed/mixtral_tuned_clim_data.csv",
              "dm_clim": "../data/simulations/processed/dolphin_mixtral_data_clim.csv",
              "mixtral_clim": "../data/simulations/processed/mixtral_8x7b_data_clim.csv",
              "gpt_clim": "../data/simulations/processed/gpt_4o_mini_data_clim.csv"}

save_path = "../data/bayesian_inference/raw/traces/"


llm = "gpt_all"  # Change this to the desired dataset key
df = pd.read_csv(data_paths[llm])

# Shift time, starting from 1, add terms relevant for the model
df.loc[:, "t"] = df.loc[:, "t"]+1
df["is_discussion"] = (df["t"] != 1).astype(int)
df["is_responder"] = (df["is_initiator"] != 1).astype(int)

# Add x_0 column for anchoring effect
for i in range(len(df)):
    if df.loc[i, "t"] == 1:
        df.loc[i, "x_0"] = df.loc[i, "x_j"]

    else:
        df.loc[i, "x_0"] = df.loc[i-2, "x_0"]


# Prepare data for model fitting
delta_x = df['dx'].to_numpy()

x_i = df['x_i'].to_numpy()
H_i = df["H_i"].to_numpy()
x_j = df['x_j'].to_numpy()
H_j = df['H_j'].to_numpy()

t_data = df['t'].to_numpy()
d_data = df['delta'].to_numpy()

is_initiator = df['is_initiator'].to_numpy()
is_responder = df['is_responder'].to_numpy()
is_climate = df['is_climate'].to_numpy()
is_ai = df['is_ai'].to_numpy()
is_gwd = df['is_gwd'].to_numpy()

x_0 = df['x_0'].to_numpy()
is_discussion = df['is_discussion'].to_numpy()


##### Full model #####

with pm.Model() as full_model:
    alpha = pm.Normal("alpha", sigma=1.0)
    tau = pm.HalfNormal("tau", sigma=1.0)

    beta_t = pm.HalfNormal("beta_t", sigma=1)    
    b_vec = pm.Uniform("b", lower=-2, upper=2, shape=3)

    beta_a = pm.HalfNormal("beta_a", sigma=1.0)
    a = pm.Uniform("a", lower=-2, upper=2)

    beta_c = pm.Normal("beta_c", sigma=1)
    
    sigma0 = pm.HalfNormal("sigma0", sigma=1)
    gamma = pm.HalfNormal("gamma", sigma=20)
    eps = pm.HalfNormal("eps", sigma=1)

    # Topic biases
    topic_matrix = pm.math.stack([is_climate, is_ai, is_gwd], axis=1)
    b_t = pm.math.dot(topic_matrix, b_vec) 
    
    # Interaction term
    interaction = alpha * pm.math.exp(-t_data / tau )

    # Mean opinion shift
    mu = interaction * ( x_j - x_i  )  + beta_t * (d_data * b_t - x_i) + beta_a * (a - x_i) + beta_c * (x_0 - x_i) * is_responder
    
    # Std term
    sigma = sigma0 + gamma * mu**2 + eps * H_i * is_discussion
    
    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=delta_x)
    
    # Posterior sampling
    trace = pm.sample(2000, tune=1000, target_accept=0.9, nuts_sampler='nutpie')

# Save trace
az.to_netcdf(trace, save_path + f"full_model_trace_{llm}.nc")





##### Full model with individual topic attractors (for climate change topic only) #####

llm = "gpt_clim"  # Change this to the desired dataset key
df = pd.read_csv(data_paths[llm])

# Shift time, starting from 1, add terms relevant for the model
df.loc[:, "t"] = df.loc[:, "t"]+1
df["is_discussion"] = (df["t"] != 1).astype(int)
df["is_responder"] = (df["is_initiator"] != 1).astype(int)

# Add x_0 column for anchoring effect
for i in range(len(df)):
    if df.loc[i, "t"] == 1:
        df.loc[i, "x_0"] = df.loc[i, "x_j"]

    else:
        df.loc[i, "x_0"] = df.loc[i-2, "x_0"]

# add 5 new zero columns for binary initial opinion encoding
df['is-2'] = 0
df['is-1'] = 0
df['is0'] = 0
df['is1'] = 0
df['is2'] = 0

# fill the new columns with binary encoding of initial opinion
df.loc[df['init_x_i'] == -2, 'is-2'] = 1
df.loc[df['init_x_i'] == -1, 'is-1'] = 1
df.loc[df['init_x_i'] == 0, 'is0'] = 1
df.loc[df['init_x_i'] == 1, 'is1'] = 1
df.loc[df['init_x_i'] == 2, 'is2'] = 1

# Prepare data for model fitting
delta_x = df['dx'].to_numpy()

x_i = df['x_i'].to_numpy()
H_i = df["H_i"].to_numpy()
x_j = df['x_j'].to_numpy()
H_j = df['H_j'].to_numpy()

t_data = df['t'].to_numpy()
d_data = df['delta'].to_numpy()

is_initiator = df['is_initiator'].to_numpy()
is_responder = df['is_responder'].to_numpy()
is_climate = df['is_climate'].to_numpy()
is_ai = df['is_ai'].to_numpy()
is_gwd = df['is_gwd'].to_numpy()

x_0 = df['x_0'].to_numpy()
is_discussion = df['is_discussion'].to_numpy()

is_minus_2 = df['is-2'].to_numpy() # binary indicator for initial opinion -2
is_minus_1 = df['is-1'].to_numpy() # binary indicator for initial opinion -1
is_0 = df['is0'].to_numpy() # binary indicator for initial opinion 0
is_1 = df['is1'].to_numpy() # binary indicator for initial opinion 1
is_2 = df['is2'].to_numpy() # binary indicator for initial opinion 2

# Infer parameters
with pm.Model() as full_model_clim_idv:
    alpha = pm.HalfNormal("alpha",sigma=1) 
    tau = pm.HalfNormal("tau", sigma=1)

    beta = pm.HalfNormal("beta", sigma=1)    
    beta_t = pm.Deterministic("beta_t", beta)
    b_vec = pm.Uniform("b", lower=-2, upper=2, shape=5)

    beta_a = pm.HalfNormal("beta_a", sigma=1)
    a = pm.Uniform("a", lower=-2, upper=2)

    beta_c = pm.Normal("beta_c", sigma=1)
    
    sigma0 = pm.HalfNormal("sigma0", sigma=1)
    gamma = pm.HalfNormal("gamma", sigma=20)
    eps = pm.HalfNormal("eps", sigma=1)

    # Topic biases
    op_matrix = pm.math.stack([is_minus_2, is_minus_1, is_0, is_1, is_2], axis=1)
    b_t = pm.math.dot(op_matrix, b_vec) 
    
    # Interaction term
    interaction = alpha * pm.math.exp(-t_data / tau )

    # Mean opinion shift
    mu = interaction * ( x_j - x_i  )  + beta_t * (d_data * b_t - x_i) + beta_a * (a - x_i) + beta_c * (x_0 - x_i) * is_responder
    
    # Std term
    sigma = sigma0 + gamma * mu**2 + eps * H_i * is_discussion
    
    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=delta_x)
    
    # Posterior sampling
    trace_clim_idv = pm.sample(2000, tune=1000, target_accept=0.9, nuts_sampler='nutpie')

# Save trace
az.to_netcdf(trace_clim_idv, save_path + f"full_model_trace_{llm}_idv.nc")