#!/bin/sh
#SBATCH --partition=dev_gpu_4
#SBATCH --mail-user=ueojl@student.kit.edu
#SBATCH --error=stupid_errors.txt
#SBATCH --output=stupid_output.txt
#SBATCH --job-name=eval_seamless_bundle
#SBATCH --mail-type=ALL
#SBATCH --time=18
#SBATCH --gres=gpu:2
#SBATCH --mem=9000

echo "Starting Stupid Model Evaluation"
tar -C $TMPDIR/ -xvzf $(ws_find data-ssd)/IWSLT23_data.tgz

python evaluation.py --model "stupid_model" --input $TMPDIR/IWSLT23_with_files.csv --output $TMPDIR/results
rsync -av $TMPDIR/results $(ws_find data-ssd)/results-${SLURM_JOB_ID}/

echo "------------------------------------------------------------"