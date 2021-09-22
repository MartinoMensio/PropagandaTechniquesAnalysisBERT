from typing import Dict
import uvicorn
import torch
import spacy
from collections import defaultdict
from spacy.tokens import Span
from transformers import BertTokenizerFast
from tokenizations import get_alignments

import random
import string
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks

from model import BertForTokenAndSequenceJointClassification

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForTokenAndSequenceJointClassification.from_pretrained(
    "QCRI/PropagandaTechniquesAnalysis-en-BERT",
    revision="v0.1.0",
)

nlp = spacy.blank('en')
nlp.add_pipe('sentencizer')

app = FastAPI()

class AnalysisRequest(BaseModel):
    text: str

class BackgroundTaskState:
    def __init__(self, target, **args):
        self.state = 'created'
        self.target = target
        self.args = args
        self.result = None
        self.key = ''.join(random.choices(string.ascii_uppercase + string.digits, k=32))
        print(args)

    def __call__(self):
        self.state = 'running'
        try:
            self.result = self.target(**self.args)
            self.state = 'done'
        except Exception as e:
            print(e)
            self.state = 'error'

    def get_state(self):
        return self.state
    
    def get_result(self):
        return self.result

tasks: Dict[str, BackgroundTaskState] ={}


@app.post('/article/propaganda/sentences')
def create_async_task(request: AnalysisRequest, background_tasks: BackgroundTasks):
    task = BackgroundTaskState(compute_on_text, request=request)
    tasks[task.key] = task
    background_tasks.add_task(task)
    return {"key": task.key}

@app.get('/article/propaganda/sentences')
def get_async_task(key: str):
    if key not in tasks:
        return 404, 'Task not found'
    task = tasks[key]
    if task.state == 'done':
        return task.result
    elif task.state == 'error':
        return {
            'error': 'some other errors',
            'state': task.state
        }
    else:
        return {
            'error': 'result is not ready',
            'state': task.state
        }
    

@app.post('/compute-now')
def compute_on_text(request: AnalysisRequest):
    """Split the text into sentences, then use compute_on_sentence"""
    text = request.text
    sents = text.split('\n\n')
    doc = nlp(text)
    sentence_propaganda = []
    sentences = list(doc.sents)
    print('number of sentences', len(sentences))
    print('number of sentences (newlines delimited)', len(sents))
    for sent in sents:
        sent = nlp(sent)
        sent_res = compute_on_sentence(sent)
        if sent_res:
            sentence_propaganda.append(sent_res)
        else:
            raise ValueError('Something is wrong')
            # more than 512 tokens
            for s in sent.sents:
                # try with sentencizer
                sent_res = compute_on_sentence(sent, trim=True)
                sentence_propaganda.append(sent_res)
    #
    return {
        # 'article_key'
        'content': '\n<br/>'.join(f'<div>{s}</div>' for s in sentences), # TODO span and sup annotations of techniques
        'sentence_propaganda': sentence_propaganda,
        'success': True
    }


def compute_on_sentence(sentence: Span, trim=True):
    text = sentence.text
    if trim:
        # TODO: can recognise trimmmed one in db if len(tokens_old) == 510
        inputs = tokenizer.encode_plus(text, return_tensors="pt", max_length=512, truncation=True)
    else:
        inputs = tokenizer.encode_plus(text, return_tensors="pt")
    if inputs.input_ids.shape[1] > 512:
        print('there will be an error soon! Input sequence too long')
        return False
    outputs = model(**inputs)

    sequence_class_index = torch.argmax(outputs.sequence_logits, dim=-1)
    sequence_class = model.sequence_tags[sequence_class_index[0]]
    probabilities = torch.softmax(outputs.sequence_logits, dim=-1)
    sequence_confidence = torch.max(probabilities).tolist()

    token_class_index = torch.argmax(outputs.token_logits, dim=-1)
    tokens_probabilities = torch.softmax(outputs.token_logits, dim=-1)
    tokens_confidences = torch.max(tokens_probabilities, axis=-1).values[0][1:-1].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0][1:-1])
    tags = [model.token_tags[i] for i in token_class_index[0].tolist()[1:-1]]
    # now align spacy and wordpieces, copied and adapted from https://github.com/explosion/spacy-transformers/blob/7fd2a013b007c2a9d808bda46dd485de74e61672/spacy_transformers/align.py
    sp_toks = [token.text for token in sentence]
    wp_toks_filtered = tokens
    span2wp, wp2span = get_alignments(sp_toks, wp_toks_filtered)
    # for each spacy_token, voting from the wordpieces tags
    tags_rearranged_spacy = [None] * len(sp_toks)
    for tag, align in zip(tags, wp2span):
        # print(tag, align)
        if len(align) > 1:
            print('error realigning')
        if not align:
            continue
        spacy_index = align[0]
        votes = tags_rearranged_spacy[spacy_index]
        if votes == None:
            votes = defaultdict(int)
            tags_rearranged_spacy[spacy_index] = votes
        votes[tag] += 1
    tags_final_spacy = []
    for tags_cnt in tags_rearranged_spacy:
        if tags_cnt == None:
            # print(sp_toks)
            # print(wp_toks_filtered)
            # print(tags)
            print('no tags for a word')
            tags_final_spacy.append('O')
        else:
            if len(tags_cnt) > 1:
                print('not the same tag, majority will be used', tags_cnt)
            max_cnt = max(tags_cnt.values())
            tags_final_spacy.append(next(k for k, v in tags_cnt.items() if v == max_cnt))
    return {
            'confidence': sequence_confidence,
            'prediction': sequence_class,
            'tags': tags_final_spacy,
            'tags_old': tags,
            # 'token_confidences': tokens_confidences, # TODO why so long on their API? I don't use it anyways
            'tokens': sp_toks,
            'tokens_old': tokens
        }

if __name__ == '__main__':
    uvicorn.run(app, port=8899, host='0.0.0.0')