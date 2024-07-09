from ModelWrapper import STModel
from Perturbation import Perturbator
import soundfile as sf
import argparse
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
    parser.add_argument("--target_lang", type=str, default="en")
    parser.add_argument("--source_lang", type=str, default="en")
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

    perturbations = Perturbator(audio).get_perturbations()
    model.infer(perturbations, sampling_rate)

def S2TT_inference_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--target_lang", type=str, default="en")
    parser.add_argument("--source_lang", type=str, default="en")
    args = parser.parse_args()
    load_STModel(args.model, target_language=args.target_lang, source_language=args.source_lang)
    audio, sampling_rate = load_audio(args.audio)
    print(f'Audio is {type(audio)}')
    print(f'Audio after loading contains {type(audio[0])}s')
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
    result = perturbator.frequency_filtering(audio=audio, sample_rate=sampling_rate)
    for filter_spec, filtered_audio in result.items():
        sf.write(f"perturbator_filter_{filter_spec}.wav", filtered_audio, samplerate=sampling_rate)
    print("Test successful")


if __name__ == "__main__":
    perturbation_test()
    # S2TT_inference_test()
    # main()
