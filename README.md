# LLM Opinion Dynamics

Code and data for the manuscript:

**Disentangling Interaction and Bias Effects in Opinion Dynamics of Large Language Models**  
Vincent C. Brockers, David A. Ehrlich, Viola Priesemann, 2026

This repository contains the simulation, Bayesian inference, parameter recovery, and plotting code used to analyse opinion dynamics in interacting large language model agents. It is intended to support inspection and reproduction of the analyses reported in the manuscript.

## Repository structure

```text
llm_opinion_dynamics/
├── data/
│   ├── inference_ready_data_all_discussions_s=25_l=5.parquet
│   ├── mixtral_tuned_clim_data.csv
│   ├── recovery/
│   │   ├── recovery_data/
│   │   └── recovery_reports/
│   ├── traces/
│   │   ├── full_model/
│   │   └── idv_model/
│   ├── unprompted_opinion_priors/
│   └── variance_explained/
│       ├── loo_r2_values/
│       └── nc_values/
├── envs/
│   ├── bayes_env.yml
│   └── vllm_env.yml
├── figures/
│   ├── figure_1.pdf
│   └── ...
├── src/
│   ├── inference.py
│   ├── inference_idv.py
│   ├── simulate-api.py
│   ├── simulate-vllm.py
│   ├── plotting/
│   └── recovery/
├── LICENSE
└── README.md
```

The main processed analysis dataset is stored in `data/inference_ready_data_all_discussions_s=25_l=5.parquet`. Posterior traces are stored in `data/traces/`. Exported manuscript figures are provided in `figures/`, and figure-generation scripts are in `src/plotting/`.

Large fine-tuned model checkpoints are available at:
https://doi.org/10.25625/4EKOMS

## Installation

Clone the repository and keep the directory structure unchanged, as several scripts use relative paths.

```bash
git clone https://github.com/Priesemann-Group/llm_opinion_dynamics.git
cd llm_opinion_dynamics
```

Create the conda environments:

```bash
conda env create -f envs/bayes_env.yml
conda env create -f envs/vllm_env.yml
```

The environment names defined in the files are:

```bash
conda activate pymc   # Bayesian inference, recovery, plotting
conda activate vllm   # local vLLM simulations and API simulations
```

## Reproducing the analyses

### Bayesian inference

The main hierarchical Bayesian model can be run from `src/`:

```bash
conda activate pymc
cd src

python inference.py --llm Llama-3.1-8B-Instruct
```

The script accepts additional sampler and model-ablation arguments. Inspect available options with:

```bash
python inference.py --help
```

The climate-only model with individual attractors is implemented in:

```text
src/inference_idv.py
```

### Parameter recovery

Parameter-recovery scripts and configuration files are in:

```text
src/recovery/
```

Corresponding recovery data and reports are stored in:

```text
data/recovery/
```

### Plotting

Manuscript figure PDFs are stored in `figures/`. Plotting scripts are in `src/plotting/`:

```bash
conda activate pymc
cd src/plotting

python figure_3.py
```

Analogous scripts are provided for the remaining plotted figures.

## Running new simulations

Two simulation backends are provided.

### API-based simulations

`src/simulate-api.py` supports OpenAI-compatible APIs. Set the required API key before running:

```bash
export OPENAI_API_KEY=<your_key>
# or, for xAI:
export XAI_API_KEY=<your_key>
```

Example:

```bash
conda activate vllm
cd src

python simulate-api.py \
  --backend openai \
  --model gpt-4o-mini \
  --topic "Climate Change" \
  --samples 25 \
  --length 5 \
  --output ../data/example_api_run.csv
```

### Local vLLM simulations

For local Hugging Face / vLLM inference, use:

```bash
conda activate vllm
cd src

python simulate-vllm.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --topic "Climate Change" \
  --samples 25 \
  --length 5 \
  --gpu 0 \
  --output ../data/example_vllm_run.csv
```

Local simulations require a CUDA-compatible GPU. Depending on model size and hardware, full sequential simulations can take several hours.

## Notes on reproducibility

The repository includes processed data, posterior traces, recovery outputs, and final figure PDFs, so the reported analyses can be inspected without rerunning all LLM simulations. Rerunning simulations may produce small numerical differences because LLM sampling and remote API backends can be stochastic or version-dependent.

## License

This repository is released under the BSD 3-Clause License.
