import math
from Quality_Estimator import QualityEstimator
from comet import download_model, load_from_checkpoint
import soundfile as sf
import pandas as pd
import argparse
import json
import time
import re
import numpy as np
import os

# pathing
path = os.path.dirname(os.path.realpath(__file__))
ted_path = os.path.join(path, "datasets", "IWSLT23.tst2023.en-de", "benchmark_IWSLT-23.en-de", "benchmark", "en-de", "tst2023", "segmented_wavs")
acl_path = os.path.join(path, "datasets", "2023.iwslt-1.2.dataset", "2", "acl_6060", "eval", "segmented_wavs", "gold")
input_path = ""
output_path = ""
config_name = "default"
start_time = 0

# load TED and ACL audio files
def load_audio(file, domain):
    if domain == "TED":
        return sf.read(os.path.join(ted_path, file))
    elif domain == "ACL":
        return sf.read(os.path.join(acl_path, file))
    else:
        try:
            return sf.read(os.path.join(ted_path, file))
        except:
            try:
                return sf.read(os.path.join(acl_path, file))
            except:
                print(f"File {file} not found in either domain.")
                return None, None
            
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not "frequency_filtering" in config["perturbation"]:
        return config
    else:
        new_config = config.copy()
        perturbations = config["perturbation"]
        for type, cutoffs in config["perturbation"]["frequency_filtering"].items():
            new_cutoffs = [(pair[0], pair[1]) for pair in cutoffs]
            perturbations["frequency_filtering"][type] = new_cutoffs
        new_config["perturbation"] = perturbations
        return new_config
    
def get_file_name_from_path(path):
    return re.split(r"/|\\", path)[-1]

def exract_scenario(input_path):
    file_name = get_file_name_from_path(input_path)
    return file_name.split(".")[0]

def evaluate(model=str, source_language="eng", target_language="deu", passed_config={}):
    # load Quality Estimation model
    if passed_config:
        QE_Model = QualityEstimator(model, source_language, target_language, **passed_config)
    else:
        QE_Model = QualityEstimator(model, source_language, target_language)

    # set up COMET 
    comet_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_path)

    def comet(transcription, translation, reference):
        data = [{
                "src": transcription,
                "mt": translation,
                "ref": reference
        }]
        return comet_model.predict(data).system_score

    # prepare for evaluation iteration
    scores = dict()
    iwslt23 = pd.read_csv(input_path)

    inference_times = np.array([])
    index = 1
    # Iterate through eval data and predict
    for _, row in iwslt23.loc[iwslt23['lp'] == 'en-de'].iterrows():
        # early stopping because of bwUniClusterv2 job length restriction
        if index == 90:
            break
        index += 1
        # load audio for eval datum 
        audio, sr = load_audio(row['audio_file'], row['domain'])
        if audio is None:
            continue
        
        starting_inference = time.time()
        # predict
        score, eval_data = QE_Model.estimate_quality(audio, sr, eval=True)
        inference_times.append(time.time() - starting_inference)
        if score is None:
            print(f"Score for {row['audio_file']} not available.")
            continue

        # record
        scores[row['audio_file']] = {
            "result": score,
            "comet": comet(row['src'], eval_data['translation'], row['ref']),
            "confidence": eval_data['confidence']
        }
        print(f"File {row['audio_file']} evaluated. Results: {scores[row['audio_file']]}")
        print(f"Median inference time: {np.median(inference_times)}")
        print(f"Mean inference time: {np.mean(inference_times)}")

    # save scores to csv
    scores_df = pd.DataFrame(columns=['audio_file', 'result', 'comet', 'confidence'])
    scores_df['audio_file'] = list(scores.keys())
    scores_df['result'] = [value['result'] for value in scores.values()]
    scores_df['comet'] = [value['comet'] for value in scores.values()]
    scores_df['confidence'] = [value['confidence'] for value in scores.values()]
    scores_df.to_csv(os.path.join(output_path, f"IWSLT23_{get_file_name_from_path(model)}_{time.time()}_{config_name}.csv"), index=False)

def main():
    global start_time
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--source_lang", type=str, default="eng")
    parser.add_argument("--target_lang", type=str, default="deu")
    parser.add_argument("--input", type=str, default=os.path.join("eval-prep", "IWSLT23_with_files.csv"))
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()

    # set eval data input and result output paths
    global input_path, output_path
    input_path = args.input
    output_path = args.output

    # If config file was given, load and pass it
    if args.config:
        config = load_config(args.config)
        global config_name
        config_name = exract_scenario(args.config)
        evaluate(args.model, args.source_lang, args.target_lang, passed_config=config)
    else:
        evaluate(args.model, args.source_lang, args.target_lang)
    
    print(f"Execution Time: {time.time() - start_time}")


if __name__ == "__main__":
    main()