# Running on the GH200 (IQ HPC, gh-aria)

This guide explains how to set up and run the Hidden Activations pipeline on the NVIDIA GH200 GraceHopper superchip at the IQ HPC cluster. The GH200 has an ARM64 (aarch64) CPU, which requires specific installation steps that differ from the Compute Canada x86 clusters.

---

## Hardware Overview

| Component | Spec |
|-----------|------|
| CPU | 72-core ARM Neoverse V2 (Grace) |
| GPU | 1x GH200 with 144 GB HBM3E |
| System memory | 465 GB LPDDR5X |
| Local storage | 3.4 TB |
| Interconnect | NVLink C2C at 900 GB/s (CPU-GPU) |
| SLURM partition | `gh-aria` |
| Node | `gh1305` (alias: `aria`) |
| Max job time | 7 days |

---

## Prerequisites

- SSH access to the IQ HPC login node
- Access to the `gh-aria` partition (contact sysadmin if needed)
- Git clone of this repository

---

## Step 1: Connect to the GH200 Node

From your local machine, SSH into the IQ HPC login node, then hop to the GH200:

```bash
ssh <your-username>@iq-hpc-login   # Replace with actual login hostname
ssh aria                            # Jump to the GH200 node
```

Alternatively, request an interactive session with GPU access:

```bash
salloc -p gh-aria --gres=gpu:1 --mem=32G
```

> **Note:** The GH200 node (`gh1305`/`aria`) has a separate home directory from the main cluster because it runs ARM64 Linux. Your files at `/net/nfs-iq/home-gh/<username>/` are distinct from your main cluster home.

---

## Step 2: Clone the Repository

On the GH200 node (or from the login node, since NFS is shared):

```bash
cd /net/nfs-iq/home-gh/armenta/
git clone <repo-url> Hidden-Activations-are-not-Enough
cd Hidden-Activations-are-not-Enough
git checkout over-GH200
```

If the repo is already cloned, just switch to the correct branch:

```bash
cd /net/nfs-iq/home-gh/armenta/Hidden-Activations-are-not-Enough
git fetch origin
git checkout over-GH200
git pull origin over-GH200
```

---

## Step 3: Run the Setup Script

The setup script creates a Python virtual environment, installs all dependencies with ARM64 CUDA wheels, downloads pretrained weights, and downloads datasets. **This only needs to be done once.**

```bash
bash setup_gh200.sh
```

This will:

1. **Create a virtual environment** at `./gh_env/`
2. **Install PyTorch and torchvision** using the ARM64 CUDA wheel index (`https://download.pytorch.org/whl/cu128`)
3. **Install all other dependencies** (torchattacks, scikit-learn, scipy, optuna, knowledgematrix, etc.)
4. **Verify CUDA availability** (prints GPU name, memory, compute capability)
5. **Download pretrained weights** (AlexNet, ResNet18, VGG11 from torchvision) into `experiments/*/weights/`
6. **Download datasets** (CIFAR-10, CIFAR-100) into `data/`

### Expected output (last section):

```
Verifying CUDA availability...
  PyTorch version: 2.7.x
  CUDA available:  True
  GPU:             GH200 480GB 120GB
  GPU memory:      142.5 GB
  Compute cap:     9.0
```

### If CUDA shows as unavailable

If `torch.cuda.is_available()` returns `False`:
- You may be running on the login node (no GPU). That's OK for setup — the weights and datasets will still download. CUDA will work when jobs run on the compute node.
- If running on the GH200 node and CUDA is still unavailable, PyTorch was installed without CUDA. Delete the venv and re-run:
  ```bash
  rm -rf gh_env
  bash setup_gh200.sh
  ```

### If you need to re-run setup

The script is idempotent. If the venv already exists, it skips creation. To start fresh:

```bash
rm -rf gh_env
bash setup_gh200.sh
```

---

## Step 4: Verify the Configuration

Before submitting real jobs, inspect the configuration to make sure paths and resource profiles are correct.

### 4a. Check `experiment_config.sh`

Open `experiment_config.sh` and verify these values match your environment:

```bash
ACCOUNT="${ACCOUNT:-def-xxxx}"         # Your SLURM account (find yours with: sacctmgr show associations user=$USER)
PARTITION="gh-aria"                    # Must be gh-aria
PROJECT_DIR="/net/nfs-iq/home-gh/armenta/Hidden-Activations-are-not-Enough"  # Your project path
ENV_NAME="gh_env"                      # Must match the venv created by setup_gh200.sh
MODULES=""                             # Must be empty (no module system on gh-aria)
```

If your username or project path differs, update `PROJECT_DIR` accordingly.

**SLURM account setup:** The `ACCOUNT` variable on line 15 sets the default SLURM billing account. Find your account with:

```bash
sacctmgr show associations user=$USER format=Account,Partition,QOS -P
```

Then set the value in `experiment_config.sh` line 15. **Watch out for the bash `:-` expansion pattern.** The correct form is:

```bash
# CORRECT — two dashes total (operator :- then value def-xxxx):
ACCOUNT="${ACCOUNT:-def-xxxx}"

# WRONG — three dashes (operator :- then --def-xxxx, producing an invalid account name):
ACCOUNT="${ACCOUNT:---def-xxxx}"
```

The `${VAR:-default}` operator uses `:-` (colon + hyphen). If your account name starts with a hyphen (unlikely), you'd get a triple-dash that looks like `---value`, producing an invalid `--value` default. This same triple-dash pattern is used intentionally for GPU flags (e.g., `A_GPU="${A_GPU:---gres=gpu:1}"`) where the value `--gres=gpu:1` genuinely starts with `--`.

If your cluster doesn't require billing accounts, leave the default empty: `ACCOUNT="${ACCOUNT:-}"`.

The account flows through `experiment_config.sh` into derived variables:
- Line 91: `GPU_ACCOUNT="${GPU_ACCOUNT:-$ACCOUNT}"` — account for GPU jobs
- Line 92: `CPU_ACCOUNT="${CPU_ACCOUNT:-$ACCOUNT}"` — account for CPU jobs
- Line 95: `ACCOUNT_LINE_GPU="${GPU_ACCOUNT:+#SBATCH --account=$GPU_ACCOUNT}"` — conditional SBATCH directive (only emitted when non-empty)

### 4b. Check `run_experiment.sh`

Open `run_experiment.sh` and verify the USER CONFIGURATION block (near the top):

```bash
ACCOUNT="def-xxxx"                     # Your SLURM account (same as experiment_config.sh)
EXPERIMENTS=("alexnet_cifar10")        # Which experiment(s) to run
```

The `ACCOUNT` here (line 32) is set **before** sourcing `experiment_config.sh`. Since `experiment_config.sh` uses `${ACCOUNT:-default}`, a non-empty value here takes precedence. If you set it here, it overrides the default in `experiment_config.sh`. If you leave it empty (`ACCOUNT=""`), the `:-` operator in `experiment_config.sh` will use its default value.

### 4c. Check `calibration.sh`

`calibration.sh` generates its own SLURM heredoc (lines 150-165) for calibration jobs. Verify that it uses the same derived variables as `run_experiment.sh`:

```bash
cat > "$JOB_DIR/calibrate.sh" << CALIB_EOF
#!/bin/bash
$ACCOUNT_LINE_GPU                      # Must use conditional variable (not hardcoded #SBATCH --account=...)
$PARTITION_LINE                        # Must be present (#SBATCH --partition=gh-aria)
$CHDIR_LINE                            # Must be present (#SBATCH --chdir=...)
#SBATCH $CALIB_GPU
...
$ENV_SETUP                             # Must use ENV_SETUP (not bare "module load" + "source")
```

**Common bugs in `calibration.sh`:**

| Bug | Wrong | Correct |
|-----|-------|---------|
| Account line | `#SBATCH --account=$GPU_ACCOUNT` (always emits, even when empty) | `$ACCOUNT_LINE_GPU` (conditional, omitted when empty) |
| Module loading | `module load $MODULES` (fails on GH200, no module system) | Remove, or guard with `if [ -n "$MODULES" ]` |
| Env activation | `source $ENV_NAME/bin/activate` (relative path) | `$ENV_SETUP` (absolute path via `$PROJECT_DIR`) |
| Missing partition | No `--partition` directive | Add `$PARTITION_LINE` |
| Missing chdir | No `--chdir` directive | Add `$CHDIR_LINE` |

### 4d. Dry run

Generate all SLURM scripts without actually submitting them:

```bash
bash run_experiment.sh --dry-run --test --skip-audit alexnet_cifar10
```

This creates job scripts in `experiments/alexnet_cifar10/orchestrator_jobs/`. Inspect a few to verify they look correct:

```bash
cat experiments/alexnet_cifar10/orchestrator_jobs/step_A.sh
cat experiments/alexnet_cifar10/orchestrator_jobs/step_B_chunk_0.sh
```

**What to check in each generated script:**
- `#SBATCH --partition=gh-aria` is present
- `#SBATCH --chdir=/net/nfs-iq/home-gh/armenta/Hidden-Activations-are-not-Enough` is present
- No `#SBATCH --account=` line (or the correct account if you set one)
- No `module load` line
- The environment activation line points to your venv (e.g., `source /net/nfs-iq/home-gh/armenta/Hidden-Activations-are-not-Enough/gh_env/bin/activate`)
- GPU steps have `#SBATCH --gres=gpu:1`
- A `SLURM_TMPDIR` fallback line is present in GPU steps

You can also run a quick verification:

```bash
# Should return nothing (no leftover Compute Canada references):
grep -r "h100\|def-assem\|module load" experiments/alexnet_cifar10/orchestrator_jobs/

# Every script should have gh-aria:
grep -l "gh-aria" experiments/alexnet_cifar10/orchestrator_jobs/*.sh
```

---

## Step 5: Run a Test Experiment

Start with a small test run to validate the full pipeline end-to-end:

```bash
bash run_experiment.sh --test --skip-audit alexnet_cifar10
```

The `--test` flag uses small sample sizes (10 samples/class, 2 chunks, short time limits) so the entire pipeline completes quickly. The `--skip-audit` flag skips the pre-flight audit step and submits the pipeline directly.

### Monitor job progress

```bash
squeue -u $USER                        # See all your queued/running jobs
squeue -u $USER -p gh-aria             # Filter to gh-aria partition only
watch -n 10 squeue -u $USER            # Auto-refresh every 10 seconds
```

### Check job output

SLURM logs go to `slurm_out_test/` (test mode) or `slurm_out/` (normal mode):

```bash
# List all output files:
ls -lt slurm_out_test/

# View the most recent job output:
cat slurm_out_test/PIPE_A_alexnet_cifar10_*.out

# Check for errors:
cat slurm_err_test/PIPE_A_alexnet_cifar10_*.err
```

### What to expect

The test pipeline submits jobs in this order:

1. **Step A** (Training) — GPU, trains the model for a few epochs
2. **Step B** (Matrices) — GPU, computes knowledge matrices (2 chunks in test mode)
3. **Step C** (Adversarial Examples) — GPU, generates adversarial examples (one job per attack)
4. **Step D** (Adversarial Matrices) — GPU, computes knowledge matrices for adversarial examples
5. **Step E** (Comparison) — GPU, runs all detectors on all representations
6. **Step G** (Theorem 4.5) — GPU, validates the theorem bound
7. **Step F** (LaTeX Tables) — CPU-only, generates result tables

Jobs are chained with SLURM dependencies (`afterok`), so each step waits for its prerequisites.

---

## Step 6: Run the Full Experiment

Once the test passes, run the full experiment:

```bash
bash run_experiment.sh --skip-audit alexnet_cifar10
```

Or run all four experiments:

```bash
# Edit run_experiment.sh to set:
# EXPERIMENTS=("alexnet_cifar10" "resnet_cifar10" "resnet_cifar100" "vgg_cifar100")

bash run_experiment.sh --skip-audit
```

### Full experiment resource summary

| Step | GPU | CPUs | Time | Memory | Jobs |
|------|-----|------|------|--------|------|
| A (Training) | 1x GH200 | 4 | 2h | 32G | 1 |
| B (Matrices) | 1x GH200 | 12 | 20m | 128G | 8 (chunked) |
| C (Adv Examples) | 1x GH200 | 4 | 3h | 32G | 17 (per-attack) |
| D (Adv Matrices) | 1x GH200 | 12 | 12h | 128G | 8 (chunked) |
| E (Comparison) | 1x GH200 | 8 | 8h | 128G | 1 |
| G (Theorem 4.5) | 1x GH200 | 4 | 6h | 64G | 1 |
| F (LaTeX Tables) | none | 2 | 15m | 4G | 1 |

> **Note:** Since gh-aria has a single GPU node, all GPU jobs run sequentially (only one job at a time on the node). CPU-only jobs (A, F) can overlap with GPU jobs if the node has available CPU resources.

---

## Step 7: Check Results

After the pipeline completes:

```bash
# Check for the main results file:
cat experiments/alexnet_cifar10/comparison/representation_comparison.json | python3 -m json.tool | head -50

# Check Theorem 4.5 results:
cat experiments/alexnet_cifar10/theorem45/theorem45_results.json | python3 -m json.tool | head -30

# Check generated LaTeX tables:
ls tables/*.tex

# Run the pipeline report for a summary:
python pipeline_report.py --experiment alexnet_cifar10
```

---

## Troubleshooting

### Job fails with "Invalid account or account/partition combination"

This means the SLURM `--account` value is wrong. Check the generated script:

```bash
head -5 experiments/*/orchestrator_jobs/step_A.sh
```

If you see `#SBATCH --account=--def-xxxx` (note the double-dash prefix `--`), the account name is malformed. This is caused by a triple-dash bug in `experiment_config.sh` line 15:

```bash
# BUG: three dashes — bash reads :- as operator, --def-xxxx as default
ACCOUNT="${ACCOUNT:---def-xxxx}"    # produces "--def-xxxx" (invalid)

# FIX: two dashes — bash reads :- as operator, def-xxxx as default
ACCOUNT="${ACCOUNT:-def-xxxx}"      # produces "def-xxxx" (correct)
```

To verify your account and fix:

```bash
# Find your account:
sacctmgr show associations user=$USER format=Account,Partition -P

# Fix experiment_config.sh line 15 with the correct account name:
# ACCOUNT="${ACCOUNT:-def-xxxx}"

# Regenerate and verify:
bash run_experiment.sh --dry-run --test --skip-audit <experiment>
head -5 experiments/<experiment>/orchestrator_jobs/step_A.sh
```

Also check `calibration.sh` — it may have a hardcoded `#SBATCH --account=$GPU_ACCOUNT` line instead of the conditional `$ACCOUNT_LINE_GPU` variable. See Section 4c for details.

### Job fails immediately with "Permission denied" or path errors

Check that `PROJECT_DIR` in `experiment_config.sh` matches your actual project directory. The `#SBATCH --chdir=` directive uses this path. Run:

```bash
echo $PROJECT_DIR   # Should print your project path
ls -la $PROJECT_DIR  # Should list your project files
```

### Job fails with "No devices were found" or CUDA errors

Make sure the job script includes `#SBATCH --gres=gpu:1`. Check the generated script:

```bash
head -15 experiments/alexnet_cifar10/orchestrator_jobs/step_B_chunk_0.sh
```

### Out of Memory (OOM) kills

The pipeline has automatic OOM retry: if a job is killed with exit code 137 (SIGKILL from cgroup), the next pipeline run doubles the memory allocation for that step. To manually increase memory:

```bash
# Override before running:
B_MEM=256G D_MEM=256G bash run_experiment.sh --skip-audit alexnet_cifar10
```

The GH200 node has 465 GB system memory. The `double_mem()` function caps at 480G.

### SLURM_TMPDIR not available

The generated scripts include a fallback:

```bash
SLURM_TMPDIR="${SLURM_TMPDIR:-/tmp/slurm-$SLURM_JOB_ID}"
mkdir -p "$SLURM_TMPDIR"
```

If SLURM does not set `$SLURM_TMPDIR` on gh-aria, jobs will use `/tmp/slurm-<jobid>/` instead. This is local storage on the node.

### torch.cuda.is_available() returns False in jobs

The PyTorch installation has CPU-only wheels instead of CUDA wheels. Re-run setup:

```bash
rm -rf gh_env
bash setup_gh200.sh
```

Verify with:

```bash
source gh_env/bin/activate
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Jobs stay in PENDING state

Check if the partition is available:

```bash
sinfo -p gh-aria
```

If the node shows as `drain` or `down`, contact the sysadmin.

### Re-running after partial completion

The pipeline supports checkpointing. If some steps completed successfully, re-running the same command will skip those steps:

```bash
bash run_experiment.sh --skip-audit alexnet_cifar10
```

Completed steps show as `SKIPPED (already complete)` in the output.

To force re-running everything, delete the checkpoints:

```bash
rm -rf experiments/alexnet_cifar10/checkpoints/
bash run_experiment.sh --skip-audit alexnet_cifar10
```

---

## Configuration Reference

All configuration lives in `experiment_config.sh`. Every variable can be overridden via environment variables:

```bash
# Example: override partition and memory for a single run
PARTITION=other-partition B_MEM=256G bash run_experiment.sh --skip-audit alexnet_cifar10
```

### Key variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ACCOUNT` | `""` | SLURM billing account (find yours with `sacctmgr show associations user=$USER`) |
| `PARTITION` | `gh-aria` | SLURM partition |
| `PROJECT_DIR` | `/net/nfs-iq/home-gh/armenta/Hidden-Activations-are-not-Enough` | Absolute project path |
| `ENV_NAME` | `gh_env` | Python venv directory name |
| `ENV_SETUP` | `source $PROJECT_DIR/$ENV_NAME/bin/activate` | Shell command to activate the environment |
| `MODULES` | `""` | Module system packages (empty = no modules) |
| `TOTAL_CHUNKS` | `8` | Parallel chunks for matrix computation |
| `BATCH_SIZE` | `1800` | Knowledge matrix columns per GPU pass |
| `NUM_SAMPLES_PER_CLASS` | `500` | Training samples per class (Step B) |
| `SAMPLES_PER_ATTACK` | `500` | Adversarial examples per attack (Step D) |

### Reverting to Compute Canada

To run on Compute Canada instead of GH200, override the GH200 defaults:

```bash
ACCOUNT="def-assem" \
PARTITION="" \
PROJECT_DIR="$(pwd)" \
MODULES="StdEnv/2023 python/3.11.5 scipy-stack/2025a" \
ENV_NAME="env" \
ENV_SETUP="module load StdEnv/2023 python/3.11.5 scipy-stack/2025a && source env/bin/activate" \
B_GPU="--gpus=h100:1" C_GPU="--gpus=h100:1" D_GPU="--gpus=h100:1" E_GPU="--gpus=h100:1" G_GPU="--gpus=h100:1" \
B_MEM=280G D_MEM=280G \
bash run_experiment.sh --skip-audit alexnet_cifar10
```

---

## Files Added for GH200 Support

| File | Purpose |
|------|---------|
| `setup_gh200.sh` | One-time setup script (venv, deps, weights, datasets) |
| `requirements-gh200.txt` | ARM64-compatible Python dependencies |
| `README-GH200.md` | This file |

## Files Modified for GH200 Support

| File | What changed |
|------|-------------|
| `experiment_config.sh` | New variables (`PARTITION`, `PROJECT_DIR`, `ENV_SETUP`, etc.), GPU flags `--gres=gpu:1`, conditional module load, derived SBATCH directive helpers |
| `run_experiment.sh` | All ~20 heredoc SLURM scripts updated with partition/chdir/env setup, SLURM_TMPDIR fallback in GPU steps, dispatch section updated |
