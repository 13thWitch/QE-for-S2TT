from Quality_Estimator import QualityEstimator
from ModelWrapper import STModel
import soundfile as sf
import argparse
import json
import os

path = os.path.dirname(os.path.realpath(__file__))
model = None
    
def load_STModel(given_model, target_language, source_language=None, additional_args=None):
    global model
    model = STModel(model_key=given_model, target_language=target_language, source_language=source_language, additional_args=additional_args)


def load_audio(audio_path):
    assert os.path.exists(audio_path), f"Audio file not found: {audio_path}"
    audio, sample_rate = sf.read(audio_path)
    print(f"Loaded audio from {audio_path} with sample rate {sample_rate} Hz")
    return audio, sample_rate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--target_lang", type=str, default="eng")
    parser.add_argument("--source_lang", type=str, default="eng")
    parser.add_argument("--metric", type=str, required=False)
    parser.add_argument("--as_corpus", type=bool, default=False)
    parser.add_argument("--from_config", type=str, default="")
    args = parser.parse_args()

    audio, sampling_rate = load_audio(args.audio)
    if args.from_config:
        with open(args.from_config, "r") as f:
            config = json.load(f)
        QE_Model = QualityEstimator(
            args.model, 
            args.source_lang, 
            args.target_lang, 
            perturbation=config["perturbation"], 
            weights=config["weights"], 
            as_corpus=config["as_corpus"], 
            metric=config["metric"]
            )
    else:
        QE_Model = QualityEstimator(args.model, args.source_lang, args.target_lang)
    score = QE_Model.estimate_quality(audio, sampling_rate, metric=args.metric, as_corpus=args.as_corpus)
    print(f"Quality Estimation: {score}")


if __name__ == "__main__":
    """
    This is an inference script for the Quality Estimation model. 
    Its configuration is the default configuration unless a config file is passed. 
    Metric and variant for translation comparison may be customized independently.
    """
    main()

