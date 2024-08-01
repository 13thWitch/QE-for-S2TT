import math
from Quality_Estimator import QualityEstimator
from comet import download_model, load_from_checkpoint
import soundfile as sf
import pandas as pd
import argparse
import os

# pathing
path = os.path.dirname(os.path.realpath(__file__))
ted_path = os.path.join(path, "datasets", "IWSLT23.tst2023.en-de", "benchmark_IWSLT-23.en-de", "benchmark", "en-de", "tst2023", "segmented_wavs")
acl_path = os.path.join(path, "datasets", "2023.iwslt-1.2.dataset", "2", "acl_6060", "eval", "segmented_wavs", "gold")
input_path = ""
output_path = ""

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

def evaluate(model=str, source_language="eng", target_language="deu"):
    # load Quality Estimation model
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

    # evaluate audio files
    scores = dict()
    iwslt23 = pd.read_csv(input_path)
    for _, row in iwslt23.loc[iwslt23['lp'] == 'en-de'].iterrows():
        audio, sr = load_audio(row['audio_file'], row['domain'])
        if audio is None:
            continue
        score, eval_data = QE_Model.estimate_quality(audio, sr, eval=True)
        if math.isnan(score):
            print(f"Score for {row['audio_file']} not available.")
            continue
        scores[row['audio_file']] = {
            "result": score,
            "comet": comet(row['src'], eval_data['translation'], row['ref']),
            "confidence": eval_data['confidence']
        }
        print(f"File {row['audio_file']} evaluated. Results: {scores[row['audio_file']]}")

    # save scores to csv
    scores_df = pd.DataFrame(columns=['audio_file', 'result', 'comet', 'confidence'])
    scores_df['audio_file'] = list(scores.keys())
    scores_df['result'] = [value['result'] for value in scores.values()]
    scores_df['comet'] = [value['comet'] for value in scores.values()]
    scores_df['confidence'] = [value['confidence'] for value in scores.values()]
    scores_df.to_csv(os.path.join(output_path, "IWSLT23_seamless_scores.csv"), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--source_lang", type=str, default="eng")
    parser.add_argument("--target_lang", type=str, default="deu")
    parser.add_argument("--input", type=str, default=os.path.join("eval-prep", "IWSLT23_with_files.csv"))
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    global input_path, output_path
    input_path = args.input
    output_path = args.output
    evaluate(args.model, args.source_lang, args.target_lang)


if __name__ == "__main__":
    main()