import soundfile as sf
import librosa as li
import numpy as np
import math

"""
OMGTHIS!!!


"""
data, sr = sf.read('datasets/pt/clips/common_voice_pt_19273358.mp3')
print(f'librosa audio: {data}')
print(f'librosa sr: {sr}')

RMS = math.sqrt(np.mean(data**2))

STD_n = 0.001
noise = np.random.normal(0, STD_n, data.shape[0])
noisy = data + noise

sf.write('noisy.wav', noisy, sr)