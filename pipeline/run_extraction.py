import os
import re
import string
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.client import device_lib
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel


# load JSON File
with open ('eval.json','r',encoding='latin-1') as d:
    eval = json.load(d)

MAX_LEN = 384  
tokenizer = BertWordPieceTokenizer("../clinicalbert/vocab.txt", lowercase=True,strip_accents=False)


class SquadExample:
    def __init__(self, question, context, start_char_idx, answer_text,end_char_idx):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.end_char_idx = end_char_idx
        self.skip = False

    def preprocess(self):
        context = self.context
        question = self.question
        answer_text = self.answer_text
        start_char_idx = self.start_char_idx
        end_char_idx = self.end_char_idx

        # Clean context, answer and question
        context = " ".join(str(context).split())
        question = " ".join(str(question).split())
        answer = " ".join(str(answer_text).split())

        # Find end character index of answer in context
        end_char_idx = start_char_idx + len(answer_text)
        if end_char_idx >= len(context):
            self.skip = True
            return

        # Mark the character indexes in context that are in answer
        is_char_in_ans = [0] * len(context)

        for idx in range(start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1

        # Tokenize context
        tokenized_context = tokenizer.encode(context)

        # Find tokens that were created from answer characters
        ans_token_idx = []
        for idx, (start, end) in enumerate(tokenized_context.offsets):
            if sum(is_char_in_ans[start:end]) > 0:
                ans_token_idx.append(idx)

        if len(ans_token_idx) == 0:
            self.skip = True
            return

        # Find start and end token index for tokens from answer
        start_token_idx = ans_token_idx[0]
        end_token_idx = ans_token_idx[-1]

        # Tokenize question
        tokenized_question = tokenizer.encode(question)

        # Create inputs
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(
            tokenized_question.ids[1:]
        )
        attention_mask = [1] * len(input_ids)

        # Pad and create attention masks.
        # Skip if truncation is needed
        padding_length = MAX_LEN - len(input_ids)
        if padding_length > 0:  # pad
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:  # skip
            self.skip = True
            return

        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.context_token_to_char = tokenized_context.offsets

def create_squad_examples(raw_data):
    squad_examples=[]
    for item in raw_data:
        context = item['data']['ner']
        for qa in [p['result'] for p in item['annotations']]:
            for q in qa:
                question = q['value']['labels']
                answer_text = q['value']['text']
                start_char_idx = q['value']['start']
                end_char_idx = q['value']['end']
                squad_eg = SquadExample(question,context,start_char_idx,answer_text,end_char_idx)
                squad_eg.preprocess()
                squad_examples.append(squad_eg)
    return squad_examples

def create_inputs_targets(squad_examples):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in squad_examples:
        if item.skip == False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    ]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y

def normalized_answer(s):    
    def remove_(text):
        ''' Remove unnecessary symbols '''
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub('《', " ", text)
        text = re.sub('》', " ", text)
        text = re.sub('<', " ", text)
        text = re.sub('>', " ", text) 
        text = re.sub('〈', " ", text)
        text = re.sub('〉', " ", text)   
        text = re.sub("\(", " ", text)
        text = re.sub("\)", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)      
        return text

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_(s))))


eval_squad_examples = create_squad_examples(eval)
x_eval, y_eval = create_inputs_targets(eval_squad_examples)
print(f"{len(eval_squad_examples)} evaluation points created.")

# load model
m = tf.saved_model.load('model/Thyroid-clinicalbert/Thyroid_Cancer/')
print(list(m.signatures.keys()))
infer = m.signatures['serving_default']

input_ids = np.array(x_eval[2])
attention_mask = np.array(x_eval[0])
token_type_ids = np.array(x_eval[1])

inputs = {
    'input_3' : input_ids,
    'input_1' : attention_mask,
    'input_2' : token_type_ids
}

outputs = infer(**inputs)


s = outputs['output_1']
e = outputs['output_2']
count = 0
ttt=pd.DataFrame()
eval_examples_no_skip = [_ for _ in eval_squad_examples if _.skip == False]
for idx, (start, end) in enumerate(zip(s, e)):
    ttt_1=[]
    abc=0
    squad_eg = eval_examples_no_skip[idx]
    offsets = squad_eg.context_token_to_char
    start = np.argmax(start)
    end = np.argmax(end)
    if start >= len(offsets):
        continue
    pred_char_start = offsets[start][0]
    if end < len(offsets):
        pred_char_end = offsets[end][1]
        pred_ans = squad_eg.context[pred_char_start:pred_char_end]
    else:
        pred_ans = squad_eg.context[pred_char_start:]
    normalized_pred_ans = normalized_answer(pred_ans)
    normalized_true_ans = normalized_answer(squad_eg.answer_text)
    if normalized_pred_ans in normalized_true_ans:
        count += 1
        abc=1
    print('Question :',squad_eg.question,'True answer : ',squad_eg.answer_text,'Predicted', pred_ans) 
    ttt_1.append([str(squad_eg.question),squad_eg.answer_text,pred_ans,abc])
    #ttt=ttt.append(ttt_1)
    
    ttt= pd.concat([ttt,pd.DataFrame(ttt_1)],ignore_index=True)
ttt.rename(columns={0:'variable', 1:'True_value', 2:'pred_value', 3:'is_correct'}, inplace=True)
display(ttt)


for idx, (start, end) in enumerate(zip(s, e)):
    ttt_1=[]
    abc=0
    squad_eg = eval_examples_no_skip[idx]
    offsets = squad_eg.context_token_to_char
    start = np.argmax(start)
    end = np.argmax(end)
    if start >= len(offsets):
        continue
    pred_char_start = offsets[start][0]
    if end < len(offsets):
        pred_char_end = offsets[end][1]
        pred_ans = squad_eg.context[pred_char_start:pred_char_end]
    else:
        pred_ans = squad_eg.context[pred_char_start:]
    normalized_pred_ans = normalized_answer(pred_ans)
    normalized_true_ans = normalized_answer(squad_eg.answer_text)
    print(normalized_pred_ans,"    ", normalized_true_ans)

ttt.to_csv("Thyroid_Cancer_pred.csv", index = False, encoding='utf-8-sig')
