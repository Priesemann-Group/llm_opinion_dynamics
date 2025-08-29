import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

data_paths = {"dm": "../data/simulations/processed/dolphin_mixtral_data.csv",
              "mixtral": "../data/simulations/processed/mixtral_8x7b_data.csv",
              "gpt": "../data/simulations/processed/gpt_4o_mini_data.csv"}


llm = "dm"  # Change this to the desired dataset key

save_path = f"../data/bayesian_inference/raw/model_comparison/"


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


##### Run models for model comparison #####

### Full model ###

with pm.Model() as model_1:
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
    trace_1 = pm.sample(2000, tune=1000, target_accept=0.9, nuts_sampler='nutpie')

# Save trace
az.to_netcdf(trace_1, save_path + f"{llm}/model_1.nc")




### Without anchoring ###

with pm.Model() as model_2:
    alpha = pm.Normal("alpha", sigma=1.0)
    tau = pm.HalfNormal("tau", sigma=1.0)

    beta_t = pm.HalfNormal("beta_t", sigma=1)    
    b_vec = pm.Uniform("b", lower=-2, upper=2, shape=3)

    beta_a = pm.HalfNormal("beta_a", sigma=1.0)
    a = pm.Uniform("a", lower=-2, upper=2)

    # beta_c = pm.Normal("beta_c", sigma=1)
    
    sigma0 = pm.HalfNormal("sigma0", sigma=1)
    gamma = pm.HalfNormal("gamma", sigma=20)
    eps = pm.HalfNormal("eps", sigma=1)

    # Topic biases
    topic_matrix = pm.math.stack([is_climate, is_ai, is_gwd], axis=1)
    b_t = pm.math.dot(topic_matrix, b_vec) 
    
    # Interaction term
    interaction = alpha * pm.math.exp(-t_data / tau )

    # Mean opinion shift
    mu = interaction * ( x_j - x_i  )  + beta_t * (d_data * b_t - x_i) + beta_a * (a - x_i) 
    
    # Std term
    sigma = sigma0 + gamma * mu**2 + eps * H_i * is_discussion
    
    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=delta_x)
    
    # Posterior sampling
    trace_2 = pm.sample(2000, tune=1000, target_accept=0.9, nuts_sampler='nutpie')

# Save trace
az.to_netcdf(trace_2, save_path + f"{llm}/model_2.nc")




### Without temporal decay ###

with pm.Model() as model_3:
    alpha = pm.Normal("alpha", sigma=1.0)
    # tau = pm.HalfNormal("tau", sigma=1.0)

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
    interaction = alpha

    # Mean opinion shift
    mu = interaction * ( x_j - x_i  )  + beta_t * (d_data * b_t - x_i) + beta_a * (a - x_i) + beta_c * (x_0 - x_i) * is_responder

    # Std term
    sigma = sigma0 + gamma * mu**2 + eps * H_i * is_discussion
    
    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=delta_x)
    
    # Posterior sampling
    trace_3 = pm.sample(2000, tune=1000, target_accept=0.9, nuts_sampler='nutpie')

# Save trace
az.to_netcdf(trace_3, save_path + f"{llm}/model_3.nc")




### Without interaction term ###

with pm.Model() as model_4:
    #alpha = pm.Normal("alpha", sigma=1.0)
    #tau = pm.HalfNormal("tau", sigma=1.0)

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
    # interaction = alpha * pm.math.exp(-t_data / tau )

    # Mean opinion shift
    mu = beta_t * (d_data * b_t - x_i) + beta_a * (a - x_i) + beta_c * (x_0 - x_i) * is_responder
    
    # Std term
    sigma = sigma0 + gamma * mu**2 + eps * H_i * is_discussion
    
    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=delta_x)
    
    # Posterior sampling
    trace_4 = pm.sample(2000, tune=1000, target_accept=0.9, nuts_sampler='nutpie')

# Save trace
az.to_netcdf(trace_4, save_path + f"{llm}/model_4.nc")




### Without topic bias ###

with pm.Model() as model_5:
    alpha = pm.Normal("alpha", sigma=1.0)
    tau = pm.HalfNormal("tau", sigma=1.0)

    #beta_t = pm.HalfNormal("beta_t", sigma=1)    
    #b_vec = pm.Uniform("b", lower=-2, upper=2, shape=3)

    beta_a = pm.HalfNormal("beta_a", sigma=1.0)
    a = pm.Uniform("a", lower=-2, upper=2)

    beta_c = pm.Normal("beta_c", sigma=1)
    
    sigma0 = pm.HalfNormal("sigma0", sigma=1)
    gamma = pm.HalfNormal("gamma", sigma=20)
    eps = pm.HalfNormal("eps", sigma=1)

    # Topic biases
    #topic_matrix = pm.math.stack([is_climate, is_ai, is_gwd], axis=1)
    #b_t = pm.math.dot(topic_matrix, b_vec) 
    
    # Interaction term
    interaction = alpha * pm.math.exp(-t_data / tau )

    # Mean opinion shift
    mu = interaction * ( x_j - x_i  ) + beta_a * (a - x_i) + beta_c * (x_0 - x_i) * is_responder
    
    # Std term
    sigma = sigma0 + gamma * mu**2 + eps * H_i * is_discussion
    
    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=delta_x)
    
    # Posterior sampling
    trace_5 = pm.sample(2000, tune=1000, target_accept=0.9, nuts_sampler='nutpie')

# Save trace
az.to_netcdf(trace_5, save_path + f"{llm}/model_5.nc")




### without entropy contribution to std. ###

with pm.Model() as model_6:
    alpha = pm.Normal("alpha", sigma=1.0)
    tau = pm.HalfNormal("tau", sigma=1.0)

    beta_t = pm.HalfNormal("beta_t", sigma=1)    
    b_vec = pm.Uniform("b", lower=-2, upper=2, shape=3)

    beta_a = pm.HalfNormal("beta_a", sigma=1.0)
    a = pm.Uniform("a", lower=-2, upper=2)

    beta_c = pm.Normal("beta_c", sigma=1)
    
    sigma0 = pm.HalfNormal("sigma0", sigma=1)
    gamma = pm.HalfNormal("gamma", sigma=20)
    #eps = pm.HalfNormal("eps", sigma=1)

    # Topic biases
    topic_matrix = pm.math.stack([is_climate, is_ai, is_gwd], axis=1)
    b_t = pm.math.dot(topic_matrix, b_vec) 
    
    # Interaction term
    interaction = alpha * pm.math.exp(-t_data / tau )

    # Mean opinion shift
    mu = interaction * ( x_j - x_i  )  + beta_t * (d_data * b_t - x_i) + beta_a * (a - x_i) + beta_c * (x_0 - x_i) * is_responder
    
    # Std term
    sigma = sigma0 + gamma * mu**2
    
    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=delta_x)
    
    # Posterior sampling
    trace_6 = pm.sample(2000, tune=1000, target_accept=0.9, nuts_sampler='nutpie')

# Save trace
az.to_netcdf(trace_6, save_path + f"{llm}/model_6.nc")




### without agreement bias ###

with pm.Model() as model_7:
    alpha = pm.Normal("alpha", sigma=1.0)
    tau = pm.HalfNormal("tau", sigma=1.0)

    beta_t = pm.HalfNormal("beta_t", sigma=1)    
    b_vec = pm.Uniform("b", lower=-2, upper=2, shape=3)

    #beta_a = pm.HalfNormal("beta_a", sigma=1.0)
    #a = pm.Uniform("a", lower=-2, upper=2)

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
    mu = interaction * ( x_j - x_i  )  + beta_t * (d_data * b_t - x_i) + beta_c * (x_0 - x_i) * is_responder
    
    # Std term
    sigma = sigma0 + gamma * mu**2 + eps * H_i * is_discussion
    
    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=delta_x)
    
    # Posterior sampling
    trace_7 = pm.sample(2000, tune=1000, target_accept=0.9, nuts_sampler='nutpie')

# Save trace
az.to_netcdf(trace_7, save_path + f"{llm}/model_7.nc")




### without bias term ###

with pm.Model() as model_8:
    alpha = pm.Normal("alpha", sigma=1.0)
    tau = pm.HalfNormal("tau", sigma=1.0)

    # beta_t = pm.HalfNormal("beta_t", sigma=1)    
    # b_vec = pm.Uniform("b", lower=-2, upper=2, shape=3)

    # beta_a = pm.HalfNormal("beta_a", sigma=1.0)
    # a = pm.Uniform("a", lower=-2, upper=2)

    # beta_c = pm.Normal("beta_c", sigma=1)
    
    sigma0 = pm.HalfNormal("sigma0", sigma=1)
    gamma = pm.HalfNormal("gamma", sigma=20)
    eps = pm.HalfNormal("eps", sigma=1)

    # Topic biases
    # topic_matrix = pm.math.stack([is_climate, is_ai, is_gwd], axis=1)
    # b_t = pm.math.dot(topic_matrix, b_vec) 
    
    # Interaction term
    interaction = alpha * pm.math.exp(-t_data / tau )

    # Mean opinion shift
    mu = interaction * ( x_j - x_i  ) 
    
    # Std term
    sigma = sigma0 + gamma * mu**2 + eps * H_i * is_discussion
    
    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=delta_x)
    
    # Posterior sampling
    trace_8 = pm.sample(2000, tune=1000, target_accept=0.9, nuts_sampler='nutpie')

# Save trace
az.to_netcdf(trace_8, save_path + f"{llm}/model_8.nc")




### without opinion shift contribution to std. ###

with pm.Model() as model_9:
    alpha = pm.Normal("alpha", sigma=1.0)
    tau = pm.HalfNormal("tau", sigma=1.0)

    beta_t = pm.HalfNormal("beta_t", sigma=1)    
    b_vec = pm.Uniform("b", lower=-2, upper=2, shape=3)

    beta_a = pm.HalfNormal("beta_a", sigma=1.0)
    a = pm.Uniform("a", lower=-2, upper=2)

    beta_c = pm.Normal("beta_c", sigma=1)
    
    sigma0 = pm.HalfNormal("sigma0", sigma=1)
    #gamma = pm.HalfNormal("gamma", sigma=20)
    eps = pm.HalfNormal("eps", sigma=1)

    # Topic biases
    topic_matrix = pm.math.stack([is_climate, is_ai, is_gwd], axis=1)
    b_t = pm.math.dot(topic_matrix, b_vec) 
    
    # Interaction term
    interaction = alpha * pm.math.exp(-t_data / tau )

    # Mean opinion shift
    mu = interaction * ( x_j - x_i  )  + beta_t * (d_data * b_t - x_i) + beta_a * (a - x_i) + beta_c * (x_0 - x_i) * is_responder
    
    # Std term
    sigma = sigma0 + eps * H_i * is_discussion
    
    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=delta_x)
    
    # Posterior sampling
    trace_9 = pm.sample(2000, tune=1000, target_accept=0.9, nuts_sampler='nutpie')

# Save trace
az.to_netcdf(trace_9, save_path + f"{llm}/model_9.nc")





##### Compute log likelihoods #####

with model_1:
    pm.compute_log_likelihood(trace_1)

with model_2:
    pm.compute_log_likelihood(trace_2)

with model_3:
    pm.compute_log_likelihood(trace_3)

with model_4:
    pm.compute_log_likelihood(trace_4)

with model_5:
    pm.compute_log_likelihood(trace_5)

with model_6:
    pm.compute_log_likelihood(trace_6)

with model_7:
    pm.compute_log_likelihood(trace_7)

with model_8:
    pm.compute_log_likelihood(trace_8)

with model_9:
    pm.compute_log_likelihood(trace_9)


# Compare the models
df_comp_loo = az.compare({"full model": trace_1, "without\nanchoring bias": trace_2, "without\ntemporal decay": trace_3, 
                          "without\ninteraction term": trace_4, "without\ntopic bias": trace_5, "without\nentropy std.": trace_6,
                          "without\nagreement bias": trace_7, "without\nbias term": trace_8, "without\nopinion shift std.": trace_9})

# save comparison to file
df_comp_loo.to_csv(save_path + f"../data/bayesian_inference/processed/model_comparison_{llm}.csv")
