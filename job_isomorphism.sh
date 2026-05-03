#!/bin/bash
#SBATCH --array=0-0
#SBATCH --time=12:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --output=slurm_out/A_iso_%A_%a.out
#SBATCH --error=slurm_err/A_iso_%A_%a.err

# Pillar 3 alignment (TMLR resubmission): Step A migrated to the same three
# architectures used by Pillar 3 KM-feature-viz (commit 9e5e9f4) and Step B
# teleportation (commit 369c9dc). Resource bumped to 8h/192G/h100:1 to match
# job_kmfv_kms.sh sizing for the larger pretrained models on full ImageNet val.
#
# IMPORTANT — array bound to 0-0 (resnet152 only):
#   * resnet152_imagenet: random neuron permutation supported via the new
#     wide-face walker in isomorphism_experiment.py (works for bottleneck
#     ResNets where the existing channel-group walker breaks at the
#     bottleneck expansion conv).
#   * densenet121_imagenet / googlenet_imagenet: NOT supported in Step A.
#     DenseNet's dense connections produce post-pool channels via implicit
#     concat (channels have specific provenance, non-interchangeable).
#     GoogLeNet's last Inception block concatenates 4 parallel branches
#     with distinct channel counts. Pillar-1 evidence on these architectures
#     comes from Step B (teleportation; see teleportation_experiment.py).
#   * If/when a more sophisticated permutation strategy is implemented for
#     concat-based topologies, expand the array to 0-2.

mkdir -p $SLURM_SUBMIT_DIR/slurm_out
mkdir -p $SLURM_SUBMIT_DIR/slurm_err

# Map array task ID to experiment. Only index 0 (resnet152) is enabled by the
# array=0-0 bound above; indices 1 and 2 are documented for completeness and
# would fail with a clear NotImplementedError if invoked manually.
EXPERIMENTS=("resnet152_imagenet" "densenet121_imagenet" "googlenet_imagenet")
EXPERIMENT=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}

echo "Step A: Isomorphism experiment for $EXPERIMENT (task $SLURM_ARRAY_TASK_ID)"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

python isomorphism_experiment.py \
    --experiment $EXPERIMENT \
    --num_permutations 5 \
    --num_samples 500 \
    --num_matrix_samples 50 \
    --matrix_batch_size 400    # Step A keeps TWO KnowledgeMatrixComputers in
                                # GPU memory at once (orig + permuted model).
                                # 1800 OOMed at ~78 GB on H100; 200 worked but
                                # used 7h59m of an 8h budget. 400 ≈ 17 GB GPU
                                # use (well under 80 GB), ~2× faster than 200.

echo "Task $SLURM_ARRAY_TASK_ID ($EXPERIMENT) completed"
