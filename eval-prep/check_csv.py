import pandas as pd
import math
import os

# pathing
path = os.path.dirname(os.path.realpath(__file__))
ted_path = os.path.join(path, "..\\datasets\\IWSLT23.tst2023.en-de\\benchmark_IWSLT-23.en-de\\benchmark\\en-de\\tst2023\\segmented_wavs")
acl_path = os.path.join(path, "..\\datasets\\2023.iwslt-1.2.dataset\\2\\acl_6060\\eval\\segmented_wavs\\gold")

# load TED and ACL audio files
def load_audio(domain, file=""):
    if type(file) != str and math.isnan(file):
        return False
    if domain == "TED":
        return os.path.isfile(os.path.join(ted_path, file))
    elif domain == "ACL":
        return os.path.isfile(os.path.join(acl_path, file))
    else:
        print(f"File {file} not found in either domain.")
        return False
    
csv = pd.read_csv("IWSLT23_with_files.csv")
for index, row in csv.loc[csv['lp'] == 'en-de'].iterrows():
    if not load_audio(row['domain'], row['audio_file']):
        print(f"Missing file {row['audio_file']} at index {index} from set {row['domain']}") 
