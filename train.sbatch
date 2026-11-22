#!/bin/bash
#SBATCH --job-name=deeplob
#SBATCH --partition=gpu
#SBATCH --qos=allocated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=experiments/%x-%j.log

# Submit from the project root, after creating experiments/:
# sbatch --account=YOUR_ACCOUNT --job-name=experiment_03_KGHM train.sbatch --stock KGHM \
#   --batch-norm --dropout 0.2 --initialization he_normal --learning-rate 0.0001 \
#   --batch-size 64 --max-epochs 50 --patience 20 --run-dir experiments/experiment_03/KGHM
# Set TRAINING_PYTHON if the environment is installed elsewhere.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
training_python=${TRAINING_PYTHON:-/N/slate/$USER/envs/deep-learning/bin/python}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

date -Is
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
srun "$training_python" -u train_deeplob.py --device cuda "$@"
