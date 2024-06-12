from datasets import load_dataset
from transformers import AutoProcessor, SeamlessM4Tv2Model
import soundfile as sf
import os

path = os.path.dirname(os.path.realpath(__file__))

ds = load_dataset(
    "covost2", "pt_en", data_dir="./datasets/pt", split="test", trust_remote_code=True
)

# Select an audio file and read it:
audio_sample = ds[7]["audio"]
waveform = audio_sample["array"]
sampling_rate = audio_sample["sampling_rate"]

# Save the audio file
sf.write("seamless_audio.wav", waveform, sampling_rate)

# Load the Seamless model
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
print(f"Seamless SR:{model.config.sampling_rate}")
print(f"Audio SR:{sampling_rate}")

# Use the model and processor to transcribe the audio:
audio_inputs = processor(
    audios=waveform, sampling_rate=sampling_rate, return_tensors="pt"
)

# from audio
output_tokens = model.generate(**audio_inputs, tgt_lang="fra", generate_speech=False)
translated_text_from_audio = processor.decode(
    output_tokens[0].tolist()[0], skip_special_tokens=True
)

print(f"Translated Text: {translated_text_from_audio}")
