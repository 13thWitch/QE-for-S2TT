from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import soundfile as sf
import numpy as np

# TODO: optimize performance. Replace with current Perturbator implementation.
# TODO: think about sense of band_stop filter

def plot_spectrogram_librosa(audio, sample_rate, title="Spectrogram"):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=80, hop_length=128, fmax=sample_rate/2)
    plt.figure(figsize=(10, 6))
    lib_return = librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), x_axis='time', y_axis='mel', fmax=sample_rate/2) # Base
    plt.savefig(f"{title}.png")
    #plt.show()

lower = 100
upper = 3000


# Load the audio file
audio, sample_rate = sf.read("unperturbed_audio_file.wav")
sample_count = len(audio)

plot_spectrogram_librosa(audio, sample_rate, "original_spectrogram")

audio_fft = fft(audio)
# print(audio_fft[sample_count//2 + sample_count//])

freqs = np.fft.fftfreq(sample_count, 1/sample_rate)

for i in range(sample_count//2):
    if freqs[i] < lower or freqs[i+1] > upper:
        audio_fft[i] = 0
        audio_fft[-(i-1)] = 0


# Inverse Fourier Transform to convert back to the time domain
filtered_audio_pass = ifft(audio_fft).real
# filtered_audio_stop = ifft(audio_fft).real

sf.write(f"filtered_audio_pass{lower}-{upper}.wav", filtered_audio_pass, sample_rate)
# sf.write(f"filtered_audio_stop{lower}-{upper}.wav", filtered_audio_stop, sample_rate)

plot_spectrogram_librosa(filtered_audio_pass, sample_rate, f"filtered_spectrogram_pass{lower}-{upper}")
# plot_spectrogram_librosa(filtered_audio_stop, sample_rate, f"filtered_spectrogram_stop{lower}-{upper}")
