#!/bin/sh
#SBATCH --partition=dev_gpu_4
#SBATCH --mail-user=ueojl@student.kit.edu
#SBATCH --error=joberrors.txt
#SBATCH --output=joboutput.txt
#SBATCH --job-name=eval_seamless_test
#SBATCH --mail-type=ALL
#SBATCH --time=30
#SBATCH --gres=gpu:2

echo "Starting job"
tar -C $TMPDIR/ -xvzf $(ws_find data-ssd)/IWSLT23_data.tgz
mkdir $TMPDIR/results

python evaluation.py --model "facebook/seamless-m4t-v2-large" --input $TMPDIR/IWSLT23_with_files.csv --output $TMPDIR/results

# tar -cvzf $(ws_find data-ssd)/results-${SLURM_JOB_ID}.tar.gz $TMPDIR/results
rsync -av $TMPDIR/results $(ws_find data-ssd)/results-${SLURM_JOB_ID}/
echo "----------------------------------------------"