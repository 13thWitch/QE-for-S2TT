from config import config, weights
import pandas as pd
import numpy as np
import json
import re

def extract_float_from_tensor_string(tensor_string):
    """
    If extracting from txt (printed output), set if True to if False.
    """
    # Define the regular expression pattern to match the floating-point number
    if False:
        return float(tensor_string)
    pattern = r'tensor\(\[([-+]?\d*\.\d+|\d+\.?\d*[eE][-+]?\d+)\]\)'
    # pattern = r'tensor\(([-+]?\d*\.\d*|\d+\.?\d*[eE][-+]?\d+)\)'
    # Search for the pattern in the input string
    match = re.search(pattern, tensor_string)
    
    if match:
        # Extract the matched floating-point number
        float_str = match.group(1)
        # Convert the string to a float
        return float(float_str)
    else:
        raise ValueError(f"{tensor_string} is not a valid tensor string.")

scores = pd.read_csv("IWSLT23_seamless_scores.csv")
qe_scores_np = scores["result"].to_numpy()
comet_scores_np = scores["comet"].to_numpy() * 100
confidence_np = scores["confidence"].map(lambda x: extract_float_from_tensor_string(x)).to_numpy()

# Pearson correlation coefficient
qe_comet_corr = np.corrcoef(qe_scores_np, comet_scores_np)
qe_loglik_corr = np.corrcoef(qe_scores_np, confidence_np)
loglik_comet_corr = np.corrcoef(confidence_np, comet_scores_np)
print(f"Correlation between QE and COMET scores: {qe_comet_corr[0][1]}")
print(f"Correlation between QE and Loglik scores: {qe_loglik_corr[0][1]}")
print(f"Correlation loglik and comet: {loglik_comet_corr[0][1]}")


# Mean Absolute Error
mae_comet = np.mean(np.abs(qe_scores_np - comet_scores_np))
mae_loglik = np.mean(np.abs(qe_scores_np - confidence_np))
mae_loglik_comet = np.mean(np.abs(confidence_np - comet_scores_np))
print(f"Mean Absolute Error between QE and COMET scores: {mae_comet}")
print(f"MAE between QE and Loglik scores: {mae_loglik}")
print(f"MAE loglik and comet: {mae_loglik_comet}")


# Root Mean Squared Error
rmse_comet = np.sqrt(np.mean((qe_scores_np - comet_scores_np) ** 2))
rmse_loglik = np.sqrt(np.mean((qe_scores_np - confidence_np) ** 2))
rmse_loglik_comet = np.sqrt(np.mean((comet_scores_np - confidence_np) ** 2))
print(f"Root Mean Squared Error between QE and COMET scores: {rmse_comet}")
print(f"Root Mean Squared Error between QE and Loglik scores: {rmse_loglik}")
print(f"RMSE loglik and comet: {rmse_loglik_comet}")

# Add Run to Log
run_name = "Seamless_noise1357_warp_05-2_centerweight_hflogprob"
metric = "bleu"
config = {
    "perturbation": config,
    "weights": weights
}

run_info = {
    "qe-comet": {
        "pearson": qe_comet_corr[0][1],
        "mae": mae_comet,
        "rmse": rmse_comet, 
    },
    "qe-loglik": {
        "pearson": qe_loglik_corr[0][1],
        "mae": mae_loglik,
        "rmse": rmse_loglik
    },
    "loglik-comet": {
        "pearson": loglik_comet_corr[0][1],
        "mae": mae_loglik_comet,
        "rmse": rmse_loglik_comet
    },
    "config": config,
    "metric": metric
}

with open(f"log_{metric}_complete.json", "r", encoding="utf-8") as f:
    log = json.load(f)
with open(f"log_{metric}_complete.json", "w", encoding="utf-8") as f:
    log[run_name] = run_info
    json.dump(log, f, ensure_ascii=False, indent=4)