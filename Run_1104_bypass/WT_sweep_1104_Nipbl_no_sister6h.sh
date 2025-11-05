#!/bin/bash
#SBATCH --account=fudenber_735
#SBATCH --partition=qcb 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --array=0-190
# turn off numba JIT
export NUMBA_DISABLE_JIT=1

module load conda

eval "$(conda shell.bash hook)"
conda activate polychrom-hoomd1

python3 WT_sweep_1104_Nipbl_no_sister6h.py ${SLURM_ARRAY_TASK_ID}
