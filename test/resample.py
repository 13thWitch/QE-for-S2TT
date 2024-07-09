import soundfile as sf
import resampy

# Load a sound file
audio, sample_rate = sf.read("unperturbed_audio_file.wav")
print(f"Original sample rate: {sample_rate}")
# Resample the audio to 16 kHz
audio32k = resampy.resample(audio, sample_rate, 16000)

# Save the resampled audio
sf.write("resampled_audio_file.wav", audio32k, 16000)