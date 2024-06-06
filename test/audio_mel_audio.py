import soundfile as sf
import librosa as lb
import numpy as np

# import scipy.signal

"""
Compares direct IFT from mel to audio with the one from the hifi-gan. 
Generates: audio_original.wav, audio_resampled.wav, mel_spec_resampled.npy, reconstructed_audio_original.wav(, audio_filtered.wav)
"""

audio, sr = sf.read(".\..\datasets\pt\clips\common_voice_pt_19277040.mp3")
print(f"Sample Rate original: {sr}")

with open("audio_original.wav", "wb") as f:
    sf.write(f, audio, sr)

settings = {
    "normal": {
        # semi-empirically optimized values
        "n_mels": 256,
        "hop_length": 128,
        "sr": sr,
        "n_fft": 2048,
    },
    "hifi": {
        # values used in the hifi-gan
        "n_mels": 80,
        "hop_length": 256,
        "fmax": 8000,
        "sr": 22050,
        "fmin": 0,
        "n_fft": 1024,
    },
}

resampled = lb.resample(y=audio, orig_sr=sr, target_sr=settings["hifi"]["sr"])

with open("audio_resampled.wav", "wb") as f:
    sf.write(f, resampled, settings["hifi"]["sr"])

"""
# Define the low-pass filter
nyquist_rate = settings["hifi"]["sr"] / 2.0
cutoff_freq = 3000  # Change this to the desired cutoff frequency
normal_cutoff = cutoff_freq / nyquist_rate
b, a = scipy.signal.butter(5, normal_cutoff, btype="low", analog=False)

# Apply the filter to the resampled audio signal
filtered_audio = scipy.signal.lfilter(b, a, resampled)

# Now you can write the filtered audio to a file
with open("audio_filtered.wav", "wb") as f:
    sf.write(f, filtered_audio, settings["hifi"]["sr"])
"""

mel_spec_original = lb.feature.melspectrogram(
    y=audio,
    sr=settings["normal"]["sr"],
    n_mels=settings["normal"]["n_mels"],
    hop_length=settings["normal"]["hop_length"],
    n_fft=settings["normal"]["n_fft"],
)

mel_spec_resampled = lb.feature.melspectrogram(
    y=resampled,
    sr=settings["hifi"]["sr"],
    n_mels=settings["hifi"]["n_mels"],
    hop_length=settings["hifi"]["hop_length"],
    fmax=settings["hifi"]["fmax"],
    fmin=settings["hifi"]["fmin"],
    n_fft=settings["hifi"]["n_fft"],
)

with open("mel_spec_resampled.npy", "wb") as f:
    np.save(f, mel_spec_resampled)


reconstructed_audio = lb.feature.inverse.mel_to_audio(
    mel_spec_original,
    sr=settings["normal"]["sr"],
    hop_length=settings["normal"]["hop_length"],
    n_fft=settings["normal"]["n_fft"],
)

with open("reconstructed_audio_original.wav", "wb") as f:
    sf.write(f, reconstructed_audio, settings["normal"]["sr"])
