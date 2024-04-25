from datasets import load_dataset
from augment import SpecAugment
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os

path = os.path.dirname(os.path.realpath(__file__))

ds = load_dataset("covost2", "pt_en", data_dir="./datasets/pt", split="test", trust_remote_code=True)

# Select an audio file and read it:
audio_sample = ds[4]["audio"]
waveform = audio_sample["array"]
sampling_rate = audio_sample["sampling_rate"]

# Extract Mel Spectrogram Features from the audio file
mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sampling_rate, n_mels=256, hop_length=128, fmax=8000)
plt.figure(figsize=(14, 6))
lib_return = librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), x_axis='time', y_axis='mel', fmax=8000) # Base


# Apply SpecAugment
apply = SpecAugment(mel_spectrogram, policy='LB')

# mask frequency
freq_masked = apply.freq_mask()
plt.figure(figsize=(14, 6))
librosa.display.specshow(librosa.power_to_db(freq_masked, ref=np.max), x_axis='time', y_axis='mel', fmax=8000)

plt.savefig('freq_mask_spec.png')