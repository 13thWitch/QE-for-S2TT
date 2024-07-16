from ModelWrapper import STModel
from Perturbation import Perturbator
from QEHead import QEHead
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

def document(result, reference_QE=None, run_data={}):
    """
    Save the run information, including result (and expected result if available) to a json file.
    """
    cummulated_run_info = run_data
    cummulated_run_info["result"] = result
    if reference_QE is not None:
        cummulated_run_info["reference"] = reference_QE
    
    with open("documentation.json", "w", encoding="utf-8") as f:
        json.dump(cummulated_run_info, f, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--target_lang", type=str, default="en")
    parser.add_argument("--source_lang", type=str, default="en")
    parser.add_argument("--transcript", type=str, default="")
    args = parser.parse_args()

    """
    TODO: Implement seeding & device selection
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    """

    load_STModel(args.model, args.target_lang, args.source_lang)
    audio, sampling_rate = load_audio(args.audio)

    config = {
        "random_noise": {
            "std_ns": [0.001]
        }, 
        "resampling": {
            "target_sample_rates": [8000, 32000]
        },
        "speed_warp": {
            "speeds": [0.5, 1, 2]
        }, 
        "frequency_filtering": {
            "pass_cutoffs": [(100, 1000), (1000, 10000)],
            "stop_cutoffs": [(100, 1000), (1000, 10000)]
        }
    }
    perturbations = Perturbator(config=config).get_perturbations(audio=audio, sample_rate=sampling_rate, transcription=args.transcript)
    perturbations["original"] = audio
    predictions = model.infer(perturbations, sampling_rate)
    # TODO: add weights
    weights = {strat: 1 for strat in predictions.keys()}
    evaluator = QEHead(weights)
    score = evaluator.get_QE_score(predictions=predictions, metric="bleu", interpret_as_corpus=False)
    document(result=score, reference_QE=None, run_data={"config": config, "weights": weights, "used_transcription": bool(args.transcript)})
    print(f"Quality Estimation: {score}")

def S2TT_inference_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--target_lang", type=str, default="en")
    parser.add_argument("--source_lang", type=str, default="en")
    args = parser.parse_args()
    load_STModel(args.model, target_language=args.target_lang, source_language=args.source_lang)
    audio, sampling_rate = load_audio(args.audio)
    print(model.infer([audio], sampling_rate))
    print("Test successful")

def perturbation_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True)
    args = parser.parse_args()
    audio, sampling_rate = load_audio(args.audio)
    perturbator = Perturbator({
        "random_noise": {
            "std_ns": [0.001]
        }, 
        "resampling": {
            "target_sample_rates": [8000, 32000]
        },
        "speed_warp": {
            "speeds": [0.5, 1, 2]
        }, 
        "frequency_filtering": {
            "pass_cutoffs": [(100, 1000), (1000, 10000)],
            "stop_cutoffs": [(100, 1000), (1000, 10000)]
        }
    })
    result = perturbator.get_perturbations(audio=audio, sample_rate=sampling_rate, transcription="Somos grandes fans de nosso clubo de futebol.")
    for strategy, perturbed_audio in result.items():
        sf.write(f"perturbator_{strategy}.wav", perturbed_audio, samplerate=sampling_rate)
    print("Test successful")

def qe_test():
    head = QEHead(weights={})
    predictions = {
        "original": "This is a test",
        "model2": "That is a test",
        "model3": "This is the test",
        "model4": "That is the test",
        "model5": "That's a test"
    }
    print(f'Score: {head.get_QE_score(predictions, metric="ter")}')


if __name__ == "__main__":
    # perturbation_test()
    # S2TT_inference_test()
    # qe_test()
    main()
