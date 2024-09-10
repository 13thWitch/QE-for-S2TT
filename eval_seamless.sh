#!/bin/sh
#SBATCH --partition=gpu_4
#SBATCH --mail-user=ueojl@student.kit.edu
#SBATCH --error=joberrors.txt
#SBATCH --output=joboutput.txt
#SBATCH --job-name=eval_seamless_bundle
#SBATCH --mail-type=ALL
#SBATCH --time=100
#SBATCH --gres=gpu:4
#SBATCH --mem=30000

echo "Starting job"
tar -C $TMPDIR/ -xvzf $(ws_find data-ssd)/IWSLT23_data.tgz
tar -C $TMPDIR/ -xvzf $(ws_find data-ssd)/configs.tgz
mkdir $TMPDIR/results
for config in $TMPDIR/configs/*
do
    echo "Evaluating ""$config"
    python evaluation.py --model "facebook/seamless-m4t-v2-large" --input $TMPDIR/IWSLT23_with_files.csv --output $TMPDIR/results --config "$config"
    echo "Done Evaluating ""$config"
done
# tar -cvzf $(ws_find data-ssd)/results-${SLURM_JOB_ID}.tar.gz $TMPDIR/results
rsync -av $TMPDIR/results $(ws_find data-ssd)/results-${SLURM_JOB_ID}/
echo "----------------------------------------------"