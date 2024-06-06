from utils import huggingface_models
from transformers import AutoModel, AutoProcessor
import soundfile as sf
import torch
import argparse
import os

path = os.path.dirname(os.path.realpath(__file__))
hf_model = {
    "model": "",
    "processor": "",
}
audio = []


def load_model(model_path):
    if model_path in huggingface_models:
        # Load from Hugging Face
        global hf_model
        hf_model["model"] = AutoModel.from_pretrained(model_path)
        hf_model["processor"] = AutoProcessor.from_pretrained(model_path)
        print(f"Loaded model from Hugging Face: {model_path}")
    elif os.path.exists(model_path):
        print(f"Loading model from {model_path}")
    elif os.path.exists(os.path.join(path, model_path)):
        print(f"Loading model from {os.path.join(path, model_path)}")
        # TODO: Load from local path
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    else:
        # Model not found
        print(f"Model {model_path} not found")
        return


def load_audio(audio_path):
    assert os.path.exists(audio_path), f"Audio file not found: {audio_path}"
    global audio
    audio, sample_rate = sf.read(audio_path)
    print(f"Loaded audio from {audio_path} with sample rate {sample_rate} Hz")


def infer():
    # TODO: Implement inference
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
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

    load_model(args.model)
    load_audio(args.audio)

    infer()


if __name__ == "__main__":
    main()
