#!/bin/sh
#SBATCH --partition=dev_gpu_4
#SBATCH --mail-user=ueojl@student.kit.edu
#SBATCH --error=test_inference_errors.txt
#SBATCH --output=test_inference_output.txt
#SBATCH --job-name=test_inference
#SBATCH --mail-type=ALL
#SBATCH --time=7
#SBATCH --gres=gpu:1
#SBATCH --mem=10000

echo "Starting Inference Test."

python "./inference.py" --model "facebook/seamless-m4t-v2-large" --audio $(ws_find data-ssd)/audio_original.wav --source_lang "por" --metric "chrf" --as_corpus true

echo "-------------------------------------------------------------"