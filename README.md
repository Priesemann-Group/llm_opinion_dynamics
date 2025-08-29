# Project Information
This repository contains code and resources for the paper: "Disentangling Interaction and Bias Effects in Opinion Dynamics of Large Language Models", Vincent C. Brockers, David A. Ehrlich and Viola Priesemann, 2025

## Repository Structure
- `data/` : Results data. 
    - `bayesian_inference/`
        - `processed/` 
            - `params/` : mean and 94% HDI values of inferred parameters.
        - `raw/`
            - `model_comparison/` : traces of models for comparison.
            - `traces/` : traces of full models for every LLM (3 topics + just climate with individual attractors).
    - `finetuning/` : Five model checkpoints for each initial opinion.
    - `simulations/` 
        - `processed/` : Prepared data for bayesian inference (opinion differences / changes, etc.).
        - `raw/` : Every measured metrics and complete text of discussions.
    - `topic priors/` : Measured opinions of every LLM without character initialization.

- `envs/` : Conda environment files.

- `plots/` : Main figures.

- `src/` : Source code.
    - `plotting/` : Interactive jupyter notebooks to re-create the figures.

- `toy_example/` : Reduced simulation, inference and analysis (no GPU but API key required).

- `README.md`

## Installation
Clone the repository with `git clone https://github.com/username/repo-name.git` and change your current working directory to `cd repo-name` for the current data structure to work flawlessly.

Create two new conda environments from the pre-defined dependencies with `conda env create -f envs/<name>_environment.yml -n <name>_env`. Replace `<name>` with `bayes` and `llm`. 

## Running the simulations, inference and plotting
Inside the scripts, you can adapt the LLM and topics. The agent loading can easily be exchanged with an API of your choice. For running simulation scripts use `conda activate llm_env`, for bayesian inference and plotting scripts/notebooks use `conda activate bayes_env`. Run with `python src/simulations/<script>.py`. For plotting use the jupyter notebooks inside `\src\plotting\`. 

Note: Running simulations locally and sequentially takes a while (few hours on a NVIDIA A100 GPU). For quicker results, it is recommended to use parallel API calls, as shown in the toy example.

## Toy example
Within the `toy_example/` folder, you find a minimal setup with jupyter notebooks to explore simulation and Bayesian inference of the results. This setup per default is limited to analyzing one discussion topic at a time. You can either run simulations by yourself or start directly with the Bayesian inference.

Create the toy conda environment with `conda env create -f toy_example/toy_env.yml`. Both the simulation and inference notebook can be run with this. 

If you wish to perform the simulations by yourself, open `toy_simulation.ipynb`, insert your OpenAI API key and run the notebook. It contains a basic layout for llm agent discussions about climate change with normal and logically negated framing during opinion probing. The notebook will automatically store all the generated discussions in `toy_example/runs` and creates a Bayesian inference ready dataframe `your_data.csv`. 

If you wish to directly start performing Bayesian inference, use the pre-generated dataframe `toy_data.csv` within `toy_inference.ipynb`. This notebook infers the opinion model parameters, as described in the paper, and demonstrates a brief model comparison. 