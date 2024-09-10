import argparse
import pandas as pd
import numpy as np
import json
import re
import os

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
    
def get_file_name_from_path(path):
    return re.split(r"/|\\", path)[-1]
    

def get_config(input_path):
    file_name = get_file_name_from_path(input_path)
    config_file = "_".join(file_name.split("_")[-2:]).removesuffix(".csv") + ".json"
    with open(os.path.join("configs", config_file), "r", encoding="utf-8") as f:
        return json.load(f)
    
def evaluate(input_path):
    if not os.path.isfile(input_path):
        raise ValueError(f"The input file is not a file.")
    scores = pd.read_csv(input_path)
    qe_scores_np = scores["result"].to_numpy()
    comet_scores_np = scores["comet"].to_numpy() * 100
    confidence_np = scores["confidence"].map(lambda x: extract_float_from_tensor_string(x)).to_numpy()

    # Pearson correlation coefficient
    qe_comet_corr = np.corrcoef(qe_scores_np, comet_scores_np)
    qe_loglik_corr = np.corrcoef(qe_scores_np, confidence_np)
    loglik_comet_corr = np.corrcoef(confidence_np, comet_scores_np)
    print(f"Correlation between QE and COMET scores: {qe_comet_corr[0][1]}")


    # Mean Absolute Error
    mae_comet = np.mean(np.abs(qe_scores_np - comet_scores_np))
    mae_loglik = np.mean(np.abs(qe_scores_np - confidence_np))
    mae_loglik_comet = np.mean(np.abs(confidence_np - comet_scores_np))
    print(f"Mean Absolute Error between QE and COMET scores: {mae_comet}")


    # Root Mean Squared Error
    rmse_comet = np.sqrt(np.mean((qe_scores_np - comet_scores_np) ** 2))
    rmse_loglik = np.sqrt(np.mean((qe_scores_np - confidence_np) ** 2))
    rmse_loglik_comet = np.sqrt(np.mean((comet_scores_np - confidence_np) ** 2))
    print(f"Root Mean Squared Error between QE and COMET scores: {rmse_comet}")

    # Add Run to Log
    characteristics = get_file_name_from_path(input_path).split("_")
    run_name = "_".join(characteristics[1:])

    config = get_config(input_path)

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
        "config": config
    }

    with open(f"result_{config["metric"] if config["metric"] else "bleu"}.json", "r", encoding="utf-8") as f:
        log = json.load(f)
    with open(f"result_{config["metric"] if config["metric"] else "bleu"}.json", "w", encoding="utf-8") as f:
        log[run_name] = run_info
        json.dump(log, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    """
    This script takes a result file containing columns result, comet and confidence. 
    It calculates Pearson correlation, RMSE and MAE for (result, comet), (result, confidence) and (comet, confidence).
    These values are saved to the metric-specific log, if available, else to the general log.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str)
    args = parser.parse_args()
    evaluate(args.result_file)