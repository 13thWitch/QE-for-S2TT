import time

import whisper

"""
An example of inference with Whisper. 
Whisper can only do X -> en and X -> X
"""
start_time = time.time()
model = whisper.load_model("base")
loaded_time = time.time()
print(f"Model Loading time: {loaded_time - start_time}")

result_mp3 = model.transcribe(
    "../datasets/pt/clips/common_voice_pt_19273360.mp3",
    language="pt",
    task="translate",
    fp16=False,
    verbose=True,
)
inference_one = time.time()
print(f"MP3 Inference time: {inference_one - loaded_time}")

result_wav = model.transcribe(
    "./audio_original.wav",
    language="pt",
    fp16=False,
    verbose=True,
    task="translate",
)
print(f"WAV Inference Time: {time.time() - inference_one}")

print(f"MP3-result: {result_mp3['text']}")
print(f"WAV-result: {result_wav['text']}")
"""
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import librosa as li

start_time = time.time()
# Load the Whisper model in Hugging Face format:
processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")
# forced_decoder_ids = processor.get_decoder_prompt_ids(language="portuguese", task="transcribe")
loaded = time.time()

audio_wav_og, sr_wav_og = sf.read("./audio_original.wav")
audio_mp3_og, sr_mp3_og = sf.read("../datasets/pt/clips/common_voice_pt_19273360.mp3")
sr_mp3 = 16000
sr_wav = sr_mp3
audio_mp3 = li.resample(audio_mp3_og, orig_sr=sr_mp3_og, target_sr=sr_mp3)
audio_wav = li.resample(audio_wav_og, orig_sr=sr_wav_og, target_sr=sr_wav)


# MP3
inf_start_mp3 = time.time()
input_features_mp3 = processor(
    audio_mp3, sampling_rate=sr_mp3, return_tensors="pt"
).input_features
predicted_ids_mp3 = model.generate(input_features_mp3)
transcription_mp3 = processor.batch_decode(predicted_ids_mp3, skip_special_tokens=True)

mp3_done = time.time()

# WAV
input_features_wav = processor(
    audio_wav, sampling_rate=sr_wav, return_tensors="pt"
).input_features
predicted_ids_wav = model.generate(input_features_wav)
transcription_wav = processor.batch_decode(predicted_ids_wav, skip_special_tokens=True)
wav_done = time.time()

print(f"Model Loading Time: {loaded - start_time}")
print(f"MP3 inference time: {mp3_done - inf_start_mp3}")
print(f"WAV inference time: {wav_done - mp3_done}")
print(f"MP3 Result: {transcription_mp3[0]}")
print(f"WAV Result: {transcription_wav[0]}")
"""
