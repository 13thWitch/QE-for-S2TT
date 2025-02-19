#!/bin/sh
#SBATCH --partition=gpu_4
#SBATCH --mail-user=ueojl@student.kit.edu
#SBATCH --error=best_errors.txt
#SBATCH --output=best_output.txt
#SBATCH --job-name=eval_seamless_bundle_best
#SBATCH --mail-type=ALL
#SBATCH --time=110
#SBATCH --gres=gpu:1
#SBATCH --mem=20000

echo "Starting job"
variant="best"
tar -C $TMPDIR/ -xvzf $(ws_find data-ssd)/IWSLT23_data.tgz
tar -C $TMPDIR/ -xvzf $(ws_find data-ssd)/configs_$variant.tgz
mkdir $TMPDIR/results
for config in $TMPDIR/configs_$variant/*
do
    echo "Evaluating ""$config"
    python evaluation.py --model "facebook/seamless-m4t-v2-large" --input $TMPDIR/IWSLT23_with_files.csv --output $TMPDIR/results --config "$config"
    echo "Done Evaluating ""$config"
done

rsync -av $TMPDIR/results $(ws_find data-ssd)/results-${SLURM_JOB_ID}/
echo "----------------------------------------------"