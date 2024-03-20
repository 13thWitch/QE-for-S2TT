from datasets import load_dataset, list_datasets
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os

path = os.path.dirname(os.path.realpath(__file__))

ds = load_dataset("covost2", "pt_en", data_dir="./datasets/pt", split="test", trust_remote_code=True)

# Select an audio file and read it:
audio_sample = ds[4]["audio"]
waveform = audio_sample["array"]
sampling_rate = audio_sample["sampling_rate"]

# Load the Whisper model in Hugging Face format:
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

# Use the model and processor to transcribe the audio:
input_features = processor(
    waveform, sampling_rate=sampling_rate, return_tensors="pt"
).input_features

# Generate token ids
predicted_ids = model.generate(input_features)

# Decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(transcription[0])
