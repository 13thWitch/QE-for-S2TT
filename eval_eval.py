import pandas as pd
import numpy as np
import json

scores = pd.read_csv("IWSLT23_seamless_scores.csv")
qe_scores_np = scores["result"].to_numpy()
comet_scores_np = scores["comet"].to_numpy() * 100
# confidence_np = scores["confidence"].to_numpy()

# Pearson correlation coefficient
qe_comet_corr = np.corrcoef(qe_scores_np, comet_scores_np)
print(f"Correlation between QE and COMET scores: {qe_comet_corr[0][1]}")

# Mean Absolute Error
mae = np.mean(np.abs(qe_scores_np - comet_scores_np))
print(f"Mean Absolute Error between QE and COMET scores: {mae}")

# Root Mean Squared Error
rmse = np.sqrt(np.mean((qe_scores_np - comet_scores_np) ** 2))
print(f"Root Mean Squared Error between QE and COMET scores: {rmse}")

# Add Run to Log
run_name = "Seamless_1-145_initial"
config = {
        "perturbation": {
            "random_noise": {
                "std_ns": [0.001]
            }, 
            "resampling": {
                "target_sample_rates": [8000, 32000]
            },
            "speed_warp": {
                "speeds": [0.7, 1, 1.7]
            }, 
            "frequency_filtering": {
                "pass_cutoffs": [(300, 3000), (1000, 10000)],
                "stop_cutoffs": [(300, 3000), (1000, 10000)]
            }
        }, 
    "weights": {},
    }

run_info = {
    "pearson": qe_comet_corr[0][1],
    "mae": mae,
    "rmse": rmse, 
    "config": config
}

with open("log_bleu.json", "r", encoding="utf-8") as f:
    log = json.load(f)
with open("log_bleu.json", "w", encoding="utf-8") as f:
    log[run_name] = run_info
    json.dump(log, f, ensure_ascii=False, indent=4)