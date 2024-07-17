import os
import yaml
from pydub import AudioSegment 

# pathing
path = os.path.dirname(os.path.realpath(__file__))
save_path = f"{path}\\datasets\\IWSLT23.tst2023.en-de\\benchmark_IWSLT-23.en-de\\benchmark\\en-de\\tst2023\\segmented_wavs\\"

# loading a talk
def load_audio(talk_num):
    print(f"Loading audio file {talk_num}")
    return AudioSegment.from_wav(f"{path}\\datasets\\IWSLT23.tst2023.en-de\\benchmark_IWSLT-23.en-de\\benchmark\\en-de\\tst2023\\wav\\ted_{talk_num}.wav")

# starting values
talk_num = 13587
segment_num = 0
audio = load_audio(talk_num)

with open(f"{path}\\datasets\\IWSLT23.tst2023.en-de\\benchmark_IWSLT-23.en-de\\benchmark\\en-de\\tst2023\\IWSLT.TED.tst2023.en-de.yaml", 'r') as stream:
    try:
        segments = yaml.safe_load(stream)
        for segment in segments:
            segment_num += 1
            if f"ted_{talk_num}.wav" != segment['wav']:
                # new talk
                talk_num = int(segment['wav'].split('_')[1].split('.')[0])
                segment_num = 1
                assert f"ted_{talk_num}.wav" == segment['wav']
                audio = load_audio(talk_num)
            # extract audio segment
            start = segment['offset'] * 1000
            audio_segment = audio[start : (start + segment['duration'] * 1000)]
            audio_segment.export(f"{save_path}{talk_num}_{segment_num}.wav", bitrate="256k", format="wav")
    
    except yaml.YAMLError as exc:
        print(exc)
