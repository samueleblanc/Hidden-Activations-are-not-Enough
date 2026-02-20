#!/bin/bash
#SBATCH --account=def-assem
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --output=slurm_out/AUDIT_%A.out
#SBATCH --error=slurm_err/AUDIT_%A.err

# ==============================================================
# job_audit.sh — Full experiment audit
#
# Audits all artifacts for an experiment, prints a color-coded
# report, saves a JSON report, and generates a recovery plan.
# ==============================================================

# --- User-configurable variables ---
EXPERIMENT="alexnet_cifar10"
TOTAL_CHUNKS=8

# --- Environment ---
mkdir -p $SLURM_SUBMIT_DIR/slurm_out
mkdir -p $SLURM_SUBMIT_DIR/slurm_err

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env_rorqual/bin/activate

# --- Copy experiment to $SLURM_TMPDIR ---
echo "Copying experiment data to temporary directory..."
mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/
cp -r experiments/$EXPERIMENT/* $SLURM_TMPDIR/experiments/$EXPERIMENT/
echo "Copy complete."

# --- Run audit and save JSON report ---
cd $SLURM_SUBMIT_DIR

python << 'AUDIT_EOF'
import sys
import os
import json

sys.path.insert(0, os.environ["SLURM_SUBMIT_DIR"])

from utils.data_integrity import verify_experiment, _print_report
from constants.constants import ATTACKS, DEFAULT_EXPERIMENTS

experiment = os.environ["EXPERIMENT"]
total_chunks = int(os.environ["TOTAL_CHUNKS"])
tmpdir = os.environ["SLURM_TMPDIR"]

experiment_dir = os.path.join(tmpdir, "experiments", experiment)

# Determine num_classes from the experiment's dataset
num_classes = 10
num_samples_per_class = 1000
num_samples_rejection_level = 10000

if experiment in DEFAULT_EXPERIMENTS:
    dataset = DEFAULT_EXPERIMENTS[experiment].get("dataset", "cifar10")
    if dataset == "cifar100":
        num_classes = 100
    elif dataset == "imagenet":
        num_classes = 1000

report = verify_experiment(
    experiment_dir=experiment_dir,
    experiment_name=experiment,
    num_classes=num_classes,
    num_samples_per_class=num_samples_per_class,
    total_chunks=total_chunks,
    num_samples_rejection_level=num_samples_rejection_level,
    attacks_list=ATTACKS,
    sample_ratio=1.0,
)

_print_report(report)

report_path = os.path.join(experiment_dir, "audit_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nJSON report saved to: {report_path}")
AUDIT_EOF

echo "Audit complete. Generating recovery plan..."

# --- Generate recovery plan ---
python << 'RECOVERY_PLAN_EOF'
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.environ["SLURM_SUBMIT_DIR"])

experiment = os.environ["EXPERIMENT"]
total_chunks = int(os.environ["TOTAL_CHUNKS"])
tmpdir = os.environ["SLURM_TMPDIR"]
submit_dir = os.environ["SLURM_SUBMIT_DIR"]

experiment_dir = os.path.join(tmpdir, "experiments", experiment)
report_path = os.path.join(experiment_dir, "audit_report.json")

with open(report_path, "r") as f:
    report = json.load(f)

now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
summary = report["summary"]

# --- Determine per-step failures ---

# Step A: weights
weights_status = report["steps"]["weights"]["status"]
recover_a = weights_status != "OK"
reason_a = []
if recover_a:
    reason_a.append(f"weights: {weights_status}")

# Step B: matrices_task_*.zip
recover_b = False
recover_b_chunks = []
reason_b = []
for i, entry in enumerate(report["steps"]["matrices_zips"]):
    if entry["status"] != "OK":
        recover_b = True
        recover_b_chunks.append(str(i))
        reason_b.append(f"matrices_task_{i}.zip: {entry['status']}")

# Step C: adversarial_examples
recover_c = False
reason_c = []
for entry in report["steps"]["adversarial_examples"]:
    if entry["status"] != "OK":
        recover_c = True
        fname = os.path.basename(os.path.dirname(entry["path"]))
        reason_c.append(f"adversarial_examples/{fname}: {entry['status']}")

# Step D: rejection_level_zips
recover_d = False
recover_d_chunks = []
reason_d = []
for i, entry in enumerate(report["steps"]["rejection_level_zips"]):
    if entry["status"] != "OK":
        recover_d = True
        recover_d_chunks.append(str(i))
        reason_d.append(f"rejection_levels/matrices_task_{i}.zip: {entry['status']}")

# Step E: matrix_statistics.json
mat_stats = report["steps"]["matrix_statistics"]
recover_e = mat_stats["status"] != "OK"
reason_e = []
if recover_e:
    reason_e.append(f"matrix_statistics.json: {mat_stats['status']}")

# Step F: adv_matrices_task_*.zip
recover_f = False
recover_f_chunks = []
reason_f = []
for i, entry in enumerate(report["steps"]["adv_matrices_zips"]):
    if entry["status"] != "OK":
        recover_f = True
        recover_f_chunks.append(str(i))
        reason_f.append(f"adv_matrices_task_{i}.zip: {entry['status']}")

# Step G: grid_search
grid_status = report["steps"]["grid_search"]["status"]
recover_g = grid_status != "OK"
reason_g = []
if recover_g:
    reason_g.append(f"grid_search: {grid_status}")

# Also check rejection_level_jsons for G
for entry in report["steps"].get("rejection_level_jsons", []):
    if entry["status"] != "OK":
        recover_g = True
        reason_g.append(f"rejection_level_jsons: {entry['status']}")

# --- Propagate dependencies ---

# If A is bad, everything downstream needs re-run
if recover_a:
    if not recover_b:
        recover_b = True
        recover_b_chunks = [str(i) for i in range(total_chunks)]
        reason_b.append("Propagated: depends on Step A")
    if not recover_c:
        recover_c = True
        reason_c.append("Propagated: depends on Step A")
    if not recover_d:
        recover_d = True
        recover_d_chunks = [str(i) for i in range(total_chunks)]
        reason_d.append("Propagated: depends on Step A")

# If any B chunk is bad, E needs re-run
if recover_b and not recover_e:
    recover_e = True
    reason_e.append("Propagated: depends on Step B")

# If C is bad, F needs re-run
if recover_c:
    if not recover_f:
        recover_f = True
        recover_f_chunks = [str(i) for i in range(total_chunks)]
        reason_f.append("Propagated: depends on Step C")

# If E, F, or D changed, G needs re-run
if (recover_e or recover_f or recover_d) and not recover_g:
    recover_g = True
    deps = []
    if recover_e:
        deps.append("Step E")
    if recover_f:
        deps.append("Step F")
    if recover_d:
        deps.append("Step D")
    reason_g.append(f"Propagated: depends on {' + '.join(deps)}")

recovery_needed = any([recover_a, recover_b, recover_c, recover_d,
                       recover_e, recover_f, recover_g])

# --- Write recovery_plan.sh ---
plan_path = os.path.join(submit_dir, "experiments", experiment, "recovery_plan.sh")
os.makedirs(os.path.dirname(plan_path), exist_ok=True)

with open(plan_path, "w") as f:
    f.write("#!/bin/bash\n")
    f.write("# ==============================================================\n")
    f.write(f"# Recovery Plan for experiment: {experiment}\n")
    f.write(f"# Generated: {now}\n")
    f.write(f"# Audit summary: {summary}\n")
    f.write("# ==============================================================\n\n")
    f.write(f'RECOVERY_EXPERIMENT="{experiment}"\n')
    f.write(f'RECOVERY_TOTAL_CHUNKS={total_chunks}\n')
    f.write(f'RECOVERY_GENERATED_AT="{now}"\n\n')

    def write_step(f, name, label, recover, reasons, chunks=None):
        f.write(f"# --- Step {name}: {label} ---\n")
        f.write(f"RECOVER_STEP_{name}={'true' if recover else 'false'}\n")
        if chunks is not None:
            f.write(f'RECOVER_STEP_{name}_CHUNKS="{" ".join(chunks) if recover else ""}"\n')
        for r in reasons:
            f.write(f"#   {r}\n")
        f.write("\n")

    write_step(f, "A", "Training", recover_a, reason_a)
    write_step(f, "B", "Generate matrices", recover_b, reason_b, recover_b_chunks)
    write_step(f, "C", "Adversarial examples", recover_c, reason_c)
    write_step(f, "D", "Rejection level matrices", recover_d, reason_d, recover_d_chunks)
    write_step(f, "E", "Matrix statistics", recover_e, reason_e)
    write_step(f, "F", "Adversarial matrices", recover_f, reason_f, recover_f_chunks)
    write_step(f, "G", "Grid search", recover_g, reason_g)

    f.write(f"RECOVERY_NEEDED={'true' if recovery_needed else 'false'}\n")

print(f"Recovery plan saved to: {plan_path}")
if recovery_needed:
    print("RECOVERY NEEDED — run job_recovery.sh to submit recovery jobs.")
else:
    print("No recovery needed — all artifacts OK.")
RECOVERY_PLAN_EOF

# --- Copy audit_report.json to permanent storage ---
echo "Copying audit report to permanent storage..."
cp $SLURM_TMPDIR/experiments/$EXPERIMENT/audit_report.json \
   $SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/audit_report.json

echo "Audit job complete."
