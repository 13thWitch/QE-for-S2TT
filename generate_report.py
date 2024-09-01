from Perturbation import Perturbator, trim_silence
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import librosa
import os

base_audio, sr = sf.read("audio_original.wav")
def plot_spectrogram_librosa(audio, sample_rate, save_as="Spectrogram"):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=80, hop_length=128, fmax=sample_rate/2)
    plt.figure(figsize=(10, 6))
    lib_return = librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), x_axis='time', y_axis='mel', fmax=sample_rate/2) # Base
    plt.savefig(os.path.join(f"{save_as}.png"))
    plt.close()
    # plt.show()

base_config = {
    "random_noise": {
        "std_ns": [0.007]
    },
    "speed_warp": {
        "speeds": [0.5, 2]
    }, 
    "frequency_filtering": {
         "pass_cutoffs": [(500, 3000)],
        "stop_cutoffs": [(500, 3000)]
    }
}

# Whole Audio Perturbation
save_folder_whole = os.path.join("exhibition_data", "whole_perturbed")
custom_config = base_config.copy()
custom_config["resampling"] = {
    "target_sample_rates": [4000, 32000]
}
perturbator_whole = Perturbator(custom_config)
whole_audio_perturb = perturbator_whole.get_perturbations(audio=base_audio, sample_rate=sr)

for spec, perturbed_audio in whole_audio_perturb.items():
    current_sr = sr
    if "resampling" in spec:
        current_sr = int(spec.split("-")[1])
    plot_spectrogram_librosa(audio=perturbed_audio, sample_rate=current_sr, save_as=os.path.join(save_folder_whole, spec))
    sf.write(file=os.path.join(save_folder_whole, f"{spec}.wav"), data=perturbed_audio, samplerate=current_sr)
    print(f"Saved {os.path.join(save_folder_whole, spec)}")

print("Whole-audio completed. Moving on to segment-wise perturbation.")

# Segment-wise Perturbation
save_folder_segmented = os.path.join("exhibition_data", "segment_perturbed")
perturbator_segmented = Perturbator(base_config)
segmented_perturb = perturbator_segmented.get_perturbations(audio=base_audio, sample_rate=sr, transcription="Nós somos grandes fans de nosso clobo de fútebol.")

for spec, perturbed_audio in segmented_perturb.items():
    plot_spectrogram_librosa(audio=perturbed_audio, sample_rate=sr, save_as=os.path.join(save_folder_segmented, spec))
    sf.write(file=os.path.join(save_folder_segmented, f"{spec}.wav"), data=perturbed_audio, samplerate=sr)
    print(f"Saved {os.path.join(save_folder_segmented, spec)}")

plot_spectrogram_librosa(audio=trim_silence(base_audio, sr), sample_rate=sr, save_as=os.path.join("exhibition_data","original_trimmed"))
plot_spectrogram_librosa(audio=base_audio, sample_rate=sr, save_as=os.path.join("exhibition_data","original"))