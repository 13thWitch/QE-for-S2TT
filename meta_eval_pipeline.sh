#!/bin/sh
echo "Starting Meta-Evaluation"
homePath="/home/kit/stud/ueojl/Projects/QE-for-S2TT"

# Move to the data-ssd directory
cd $(ws_find data-ssd) 

# Loop through all directories (hidden and non-hidden)
for dir in */
do
    # Check if the results directory exists in each dir before moving files
    if [ -d "$dir/results" ]; then
        mv "$dir""results/"* "$homePath/current_results"
    fi
done

# Change to home path
cd "$homePath"

# Loop over files in current_results and run eval_eval.py
for file in ./current_results/*
do
    echo "Evaluating $file..."
    python eval_eval.py --result_file "$file"
done

# cleanup
mv ./current_results/* ./results/

echo "Meta-Evaluation complete"
