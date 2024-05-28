from datasets import load_dataset
from augment_notfa import SpecAugment
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import math
import os

path = os.path.dirname(os.path.realpath(__file__))

ds = load_dataset("covost2", "pt_en", data_dir="./datasets/pt", split="test", trust_remote_code=True)

# Select an audio file and read it:
audio_sample = ds[4]["audio"]
waveform = audio_sample["array"]
sampling_rate = audio_sample["sampling_rate"]

# Save the audio file
sf.write('unperturbed_audio_file.wav', waveform, sampling_rate)

# Extract Mel Spectrogram Features from the audio file
print(f'SampleRate: {sampling_rate}')
mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sampling_rate, n_mels=80, hop_length=256, fmax=8000)
print(f'Mel shape: {mel_spectrogram.shape}')
plt.figure(figsize=(14, 6))
lib_return = librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), x_axis='time', y_axis='mel', fmax=8000) # Base


# Apply SpecAugment
apply = SpecAugment(mel_spectrogram, policy='LB')

# mask frequency
freq_masked = apply.freq_mask()

print(freq_masked.shape)
# save the masked mel spectrogram
with open('freq_masked_mel_transposed.npy', 'wb') as f:
    np.save(f, freq_masked.T)

# convert back to audio signal
# This is unavoidably lossy and introduces some noise/metallic sound. Minimal at about 2048 n_fft and 128 hop_length
# Apparently, larger n_fft and smaller hop_length values are better for time precision (needed for speech recognition)
perturbed_audio = librosa.feature.inverse.mel_to_audio(freq_masked, sr=sampling_rate, n_fft=1024, hop_length=256)
# sf.write('perturbed_audio.wav', perturbed_audio[:math.floor(len(perturbed_audio)/3)], sampling_rate)
sf.write('perturbed_audio.wav', perturbed_audio, sampling_rate)

plt.figure(figsize=(14, 6))
# librosa.display.specshow(librosa.power_to_db(freq_masked, ref=np.max), x_axis='time', y_axis='mel', fmax=8000)
librosa.display.specshow(librosa.power_to_db(freq_masked[:,:], ref=np.max), x_axis='time', y_axis='mel', fmax=8000)
plt.savefig('test.png')