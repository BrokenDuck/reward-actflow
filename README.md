# Active Diffusion Models

Towards efficient mid-training and test-time discovery beyond the data via self-expansion.

## Installation

First you need to install [uv](https://docs.astral.sh/uv/getting-started/installation/):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then create a virtual environment and install the core dependencies:
```bash
uv venv
uv pip install -e .
```
Finally, run the install script to set up the remaining dependencies (bioinformatics tools, openfold, and the SGPO protein diffusion model). It takes a scratch directory as argument:
```bash
bash install.sh <install_dir>
```
To activate the environment for an interactive session:
```bash
source .venv/bin/activate
```

**Recommended**: Use a scratch disk for the experiments, e.g., by setting `--dir /cluster/scratch/<username>/adm/proteins` for the proteins experiment, because the experiment directories can get quite large.

## Command Line Interface

### Task-Agnostic Expansion

```bash
uv run python -m adm.task_agnostic <problem_setup> <uncertainty_estimator> --dir <experiment_dir> <other_args>
```
For `<problem_setup>`, currently supported options are `toy`, `qm9`, `geom_drugs`, `stable_diffusion`, and `proteins` (this branch). For `<uncertainty_estimator>`, currently supported options are `gp` (Gaussian process) and `ensemble` (ensemble of 5 MLPs with 3 hidden layers, 100 activations each). They all define their own arguments as well, which you can see by running:
```bash
uv run python -m adm.task_agnostic <problem_setup> <uncertainty_estimator> --help
```

### Task-Directed Expansion

```bash
uv run python -m adm.task_directed <problem_setup> <uncertainty_estimator> --dir <experiment_dir> --reward <task> --reward_opt <"min" | "max"> <other_args>
```
This command supports the same problem setups as the Task Agnostic Expansion, but also requires a `--reward` argument to specify the task. Also you can set the `--reward_opt` to control whether you minimize or maximize the reward.

### Arguments for Expansion

#### Baselines

 * `--no_uncertainty`: This flag disables uncertainty estimation and maximization, resulting in the Recursive Sampling (with verifier) scheme.
 * `--no_verifier`: This flag disables the verifier, resulting in the Recursive Sampling (without verifier) scheme in combination with `--no_uncertainty`.

**Recommended**: For any problem setup, first run the above two to get baseline results. `--no_uncertainty` can also be used to dial in the verifier function or other parts of your problem setup, such that you can differentiate between problems arising from expansion or the problem setup itself.

#### Uncertainty Estimation

All uncertainty estimators take features from the score network as input at a specified timestep (`--feat_timestep`). The weight given to the uncertainty estimator is set by `--dps_weight`, where we use Diffusion Posterior Sampling (DPS; https://arxiv.org/abs/2209.14687) to find high-uncertainty samples.

 * `--feat_timestep`: This option specifies the timestep from which features are obtained for the GP. For QM9, `0.9` works best, but this might be different for other problem setups. **Recommended**: Start from `0.9` and try different values to see what works best for your problem setup.
 * `--dps_weight`: This option specifies the weight of the uncertainty reward for Diffusion Posterior Sampling (DPS). Higher values will lead to more exploration, but also leads to lower validity. **Recommended**: Start from `10` and try different values to see what works best for your problem setup.

##### Gaussian Process

 * `--gp_kernel`: This option specifies the kernel for the uncertainty estimation. Supported options are `rbf` and `linear`.
 * `--gp_lengthscale`: If you set `--gp_kernel` to `rbf`, this option specifies the lengthscale for the RBF kernel. This controls how much you believe the generative model generalizes from a data point.

##### Ensemble

Here we use an ensemble defined over the input space of score network activations. We use the same ensemble architecture as in Active Learning-Assisted Directed Evolution (ALDE; https://www.nature.com/articles/s41467-025-55987-8).

#### Feasibility

 * `--sample_batch_size`: This option specifies the batch size for sampling. Setting it to a lower value reduces VRAM usage but increases runtime. For our purposes, this should be lower than what would be possible for normal inference due to running DPS. **Recommended**: Try to find the maximum value that would fit in your VRAM so you can maximally make use of parallelization. For Stable Diffusion, this value is `2` on a 24GB VRAM GPU.
 * `--ft_batch_size`: This option specifies the batch size for fine-tuning. Setting it to a lower value reduces VRAM usage but increases stochasticity in the gradients. **Recommended**: Try to find the maximum value that would fit in your VRAM. For Stable Diffusion, this value is `4` on a 24GB VRAM GPU.
 * `--ft_accumulate_steps`: This option specifies the number of steps for gradient accumulation during fine-tuning. By making use of this, we can get a bigger effective batch size for fine-tuning without increasing VRAM usage. For example, if you set `--ft_batch_size` to `4` and `--ft_accumulate_steps` to `4`, you get an effective batch size of 16. **Recommended**: Make sure that the effective batch size is at least 32.
 * `--ft_steps`: This option specifies the number of fine-tuning steps with the data collected thus far. _Note_ that the effective number of steps is `ft_steps` / `ft_accumulate_steps`.

#### Evaluation

 * `--eval_samples`: Number of samples to use for evaluation during training. Defaults to `0`.
 * `--eval_every`: This option specifies how often to run evaluation during training, in terms of iterations.

#### Other

 * `--ft_lr`: This option specifies the learning rate when fine-tuning on new samples. **Recommended**: Use the same learning rate as was used in pre-training. Lower it if you see overfitting.
 * `--ft_weight_decay`: This option specifies the weight decay of AdamW when fine-tuning on new samples. **Recommended**: If you use LoRA (e.g., Stable Diffusion), set this fairly high at around `1e-1` or `1e-2`. Otherwise, do not use this, as you will only drift further from the pre-trained weights then.
 * `--ft_min_dataset_size`: This option specifies the minimum dataset size before starting fine-tuning. In other words, it specifies how many samples we take from the prior before starting the expansion process. Setting it to a higher value could lead to less mode forgetting. **Recommended**: Start from `1024` and try higher values if you see mode forgetting.

### Evaluation

To evaluate the final model, you can run (make sure to set `--ckpt base_model.pt` to use the expanded model instead of the pre-trained model):
```bash
uv run python -m adm.evaluation.sample_many <experiment_dir> <other_args>
```
This script will generate many samples from the generative model. Alternatively, you can run:
```bash
uv run python -m adm.evaluation.sample_many_valids <experiment_dir> <other_args>
```
This script will keep sampling until the number of specified valid samples are sampled (instead of sampling both invalid and valid ones). This is important for metrics such as Vendi diversity that need a constant number of samples.

Then to evaluate the samples, you can run:
```bash
uv run python -m adm.evaluation.eval_samples <experiment_dir> <sample_dir> [--do_global_metrics] [--do_sample_metrics]
```
This will evaluate the samples using the specified global and sample metrics for the problem setup. You can specify which to do using the flags `--do_global_metrics` and `--do_sample_metrics`. For example, for the QM9 problem setup, it computes the Vendi diversity over all samples and GFN2-xTB properties.

### Proteins

The `proteins` problem setup models protein sequence generation using a continuous diffusion model over ESM token embeddings. Validity is assessed by folding each candidate sequence with ESMFold and thresholding on mean pLDDT. The following arguments are specific to this setup.

#### Problem Setup

 * `--cfg_path`: Path to the SGPO diffusion model config YAML. Defaults to the bundled `sample_config.yaml` from the `sgpo` package.
 * `--threshold`: Minimum mean pLDDT score (0–100) required for a sequence to be considered valid. Default: `65.0`. **Recommended**: Lower this if too few samples pass validity; raise it if you want stricter structural confidence.
 * `--validity_batch_size`: Number of sequences to fold per ESMFold call. Lower values reduce VRAM usage at the cost of runtime. Default: `32`.
 * `--esmfold_chunk_size`: Enables chunked attention in ESMFold. Smaller values reduce peak VRAM usage but slow down folding. Omit to disable chunking.
 * `--esmfold_fp16`: Run ESMFold in half-precision (fp16). Reduces VRAM usage significantly, but stubs out the pTM head (pTM scores will be zero). Use when VRAM is the bottleneck and pTM is not needed for validity.
 * `--lengthscale_vendi`: RBF kernel lengthscale used when computing the Vendi diversity score over ESM embeddings. Default: `2.0`.

#### Rewards (Task-Directed only)

Two rewards are available for the proteins problem setup, passed via `--reward`:

 * `proteins/creilov`: Predicts CreiLOV luciferase fitness using an ensemble of oracle models trained on the CreiLOV dataset. Sequences are penalised exponentially for Hamming distances above `70` from the wild type. Use `--reward_opt max` to maximise fitness.
 * `proteins/fitness`: Same as `proteins/creilov` but accepts a custom `oracle_path` via `--reward_kwargs`, e.g. `--reward_kwargs '{"oracle_path": "/path/to/oracle"}'`.

Example task-directed run maximising CreiLOV fitness:
```bash
uv run python -m adm.task_directed proteins gp --dir experiments/proteins --reward proteins/creilov --reward_opt max
```

## Running on Different Problem Setups

This repository abstracts away the interface with the specific problem at hand (e.g. protein sequence generation, unconditional molecular sequence & structure generation) into a `ProblemSetup` (see `adm/setups/problem_setup.py`). A `ProblemSetup` allows the user to define the base flow model, validity function, problem-relevant metrics, pre/post-processing functions and other important logic all in a centralised package. We have defined different problem setups in this repository corresponding to different applications (e.g. molecules/proteins). In order to separate dependencies each is defined in its own respective branch. To use each one, first check out the relevant branch and then use the CLI defined [above](#command-line-interface). To define your own Problem Setup for your own application see the section [below](#defining-new-problem-setups).

## Defining New Problem Setups

To define a new problem setup to run task-agnostic or task-directed expansion on, you only need to inherit the `ProblemSetup[D]` in `adm/setups/problem_setup.py` and implement the abstract methods. You can then run expansion on your problem setup by using the command line interface described below, and passing the name of your problem setup as an argument. This requires some knowledge of the [diffusiongym package](https://github.com/cristianpjensen/diffusiongym), so it is recommended to start with reading [the documentation](https://cristianpjensen.github.io/diffusiongym/) and looking at the existing problem setups for examples.
