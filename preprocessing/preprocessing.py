#1. Data Preprocessing
#Remove whitespace from JSON text body and adjust annotation offset positions#  
import os
import pprint
import json
import pandas as pd
import numpy as np
import re

#1.1 Load JSON Annotation File
with open ('breast_cancer.json','r',encoding='utf-8') as f:
    df = json.load(f)
print(len(df))

#Exclude reports that have not been annotated
df1 =[]
for item in df:
    for i in item['annotations']:
        if i['was_cancelled'] ==False :
            df1.append(item) 
print(len(df1))

#1.2 Data Cleaning (Remove unnecessary content and adjust annotation offsets)
#Data cleaning: remove content before the DIAGNOSIS section
for item in df1:
    aaa=item['data']['ner']
    item['data']['ner']=aaa[aaa.find('\nDIAGNOSIS'):]
    for qa in [p['result'] for p in item['annotations']]:
        for q in qa:
            q['value']['start'] = q['value']['start']-aaa.find('\nDIAGNOSIS')
            q['value']['end'] = q['value']['end']-aaa.find('\nDIAGNOSIS')

#1.3  Data Cleaning (Whitespace normalization and annotation offset recalculation)
for item in df1:
    aaa=item['data']['ner']
    item['data']['ner'] = " ".join(str(aaa).split())
    for qa in [p['result'] for p in item['annotations']]:
        for q in qa:
            q['value']['text']=" ".join(str(q['value']['text']).split())
            q['value']['start'] = item['data']['ner'].find(q['value']['text'])
            q['value']['end'] = item['data']['ner'].find(q['value']['text'])+len(q['value']['text'])+1

#1.4. Save Preprocessed Data
with open('breast_cancer.json','w') as f:
    json.dump(df1,f,ensure_ascii=False)

