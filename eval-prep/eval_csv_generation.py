import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os
import re

"""
This script generates a csv file based on the IWSLT23 csv dataset file. It additionally contains the audio file names for each segment in the en-de dataset.
Since the file structure and contents of the TED and ACL segment files are different, the script is split into two parts, one for each domain.
"""

def load_xml(file):
    with open(file, "r", encoding="utf-8") as f:
        xml = f.read()
    return ET.fromstring(re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) + "</root>")

# TED
tree = load_xml("IWSLT.TED.tst2023.en-de.en_formatted.xml") # create an ElementTree object 

dict_text_to_file = {}
for doc in tree.iter('doc'):
    talk_id = doc.attrib['docid']
    for seg in doc.iter('seg'):
        seg_id = seg.attrib['id']
        text = seg.text
        dict_text_to_file[text] = f"{talk_id}_{seg_id}.wav"

text_to_file = pd.DataFrame(columns=['src', 'audio_file'])
text_to_file['src'] = list(dict_text_to_file.keys())
text_to_file['audio_file'] = list(dict_text_to_file.values())
        


csv_input = pd.read_csv('..\\datasets\IWSLT23.tst2023.en-de\iwslt2023da.csv')

joined_csv_ted = csv_input.merge(text_to_file, how='left', left_on='src', right_on='src')

# ACL
tree = load_xml("ACL.6060.eval.en-xx.en_formatted.xml")
dict_text_to_file = {}
for seg in tree.iter('seg'):
    dict_text_to_file[seg.text] = f"sent_{seg.attrib['id']}.wav"

text_to_file = pd.DataFrame(columns=['src', 'audio_file'])
text_to_file['src'] = list(dict_text_to_file.keys())
text_to_file['audio_file'] = list(dict_text_to_file.values())

csv_input = joined_csv_ted
joined_csv_ted_acl = csv_input.merge(text_to_file, how='left', left_on='src', right_on='src')

# Right now, the joined_csv has two columns for audio_file, audio_file_x and audio_file_y. We need to clean this up.
clean_cols = ['lp','src','mt','ref','raw','annotators','domain','year','task','sys_id','audio_file']
clean_csv = pd.DataFrame(columns=['lp','src','mt','ref','raw','annotators','domain','year','task','sys_id','audio_file'])
for col in clean_cols[:-1]:
    clean_csv[col] = joined_csv_ted_acl[col]

# fill the clean_csv audio_file column with value in audio_file_x if the domain is TED, otherwise fill with audio_file_y
clean_csv['audio_file'] = np.where(clean_csv['domain'] == 'TED', joined_csv_ted_acl['audio_file_x'], joined_csv_ted_acl['audio_file_y'])

clean_csv.to_csv('IWSLT23_with_files.csv', index=False)
