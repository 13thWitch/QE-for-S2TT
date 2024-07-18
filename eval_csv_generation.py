import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import csv
import os
import io
import re

path = os.getcwd()
ted_path = "datasets\\IWSLT23.tst2023.en-de\\benchmark_IWSLT-23.en-de\\benchmark\\en-de\\tst2023\\"

def load_xml(file):
    with open(file, "r", encoding="utf-8") as f:
        xml = f.read()
    return ET.fromstring(re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) + "</root>")

"""
def iter_segs(root):
    author_attr = root.attrib
    for seg in root.iter('seg'):
        seg_dict = author_attr.copy()
        seg_dict.update(seg.attrib)
        seg_dict['data'] = seg.text
        yield seg_dict
"""

# TED
tree = load_xml(f"{ted_path}IWSLT.TED.tst2023.en-de.en.xml") # create an ElementTree object 

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
        


csv_input = pd.read_csv('datasets\IWSLT23.tst2023.en-de\iwslt2023da.csv')

joined_csv = csv_input.merge(text_to_file, how='left', left_on='src', right_on='src')

joined_csv.to_csv('output.csv', index=False)

# ACL
tree = load_xml(f"C:\\Users\\ivaan\\Projects\\QE-for-S2TT\\datasets\\2023.iwslt-1.2.dataset\\2\\acl_6060\\eval\\text\\xml\\ACL.6060.eval.en-xx.en.xml")
dict_text_to_file = {}
for seg in tree.iter('seg'):
    dict_text_to_file[seg.text] = f"sent_{seg.attrib['id']}"

text_to_file = pd.DataFrame(columns=['src', 'audio_file'])
text_to_file['src'] = list(dict_text_to_file.keys())
text_to_file['audio_file'] = list(dict_text_to_file.values())

csv_input = pd.read_csv('output.csv')
joined_csv = csv_input.merge(text_to_file, how='left', left_on='src', right_on='src')

clean_cols = ['lp','src','mt','ref','raw','annotators','domain','year','task','sys_id','audio_file']
clean_csv = pd.DataFrame(columns=['lp','src','mt','ref','raw','annotators','domain','year','task','sys_id','audio_file'])
for col in clean_cols[:-1]:
    clean_csv[col] = joined_csv[col]

# fill the clean_csv audio_file column with column audio_file_x from joined_csv, or if that value is empty, with audio_file_y from joined_csv
clean_csv['audio_file'] = joined_csv['audio_file_x'].fillna(joined_csv['audio_file_y'])

clean_csv.to_csv('IWSLT23_with_files.csv', index=False)
