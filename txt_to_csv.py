import pandas as pd
import re

# Read the txt file
with open('joboutput.txt', 'r') as file:
    data = file.readlines()

file_names = list()
qe_scores = list()
comet_scores = list()
confidences = list()

for line in data:
    file_name = re.search(r'File (.+?) evaluated', line).group(1)
    qe_score = re.search(r'\'result\': (.+?), ', line).group(1)
    comet_score = re.search(r'\'comet\': (.+?), ', line).group(1)
    confidence = re.search(r'\'confidence\': tensor\(\[(.+?)\]\)', line).group(1)
    file_names.append(file_name)
    qe_scores.append(qe_score)  
    comet_scores.append(comet_score)
    confidences.append(confidence)

# Create a DataFrame
df = pd.DataFrame({'audio_file': file_names, 'result': qe_scores, 'comet': comet_scores, 'confidence': confidences})
df.to_csv(f'IWSLT23_scores_1-{len(file_names)}.csv', index=False)
