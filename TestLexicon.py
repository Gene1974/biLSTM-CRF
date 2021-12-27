import json
import numpy as np
import os
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

from BiLSTM_CRF import NER
from CCKSData import CCKSDataset, CCKSVocab
from TagVocab import TagVocab
from pytorchtools import EarlyStopping
from Utils import logger, label_chinese_entity

torch.manual_seed(1)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from LexiconEmbedding import LexiconEmbedding

data_path = './data_small/'
data_path = '/home/gene/Documents/Data/CCKS2019/'

model_time = '12251734'
model_path = './results/{}_lexicon'.format(model_time)
with open(model_path + '/vocab_' + model_time, 'rb') as f:
    vocab = pickle.load(f)
    tag_vocab = vocab.tag_vocab            
test_set = CCKSDataset(data_path, vocab, tag_vocab, mod = 'test')
logger('Load data. Test data: {}'.format(len(test_set)))
lexicon_embeds = LexiconEmbedding(256, tag_vocab)

batch_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gold_num, predict_num, correct_num = 0, 0, 0
relax_correct_num = 0
logger('Begin testing.')
i = 0
while i < len(test_set):
    if i + batch_size < len(test_set):
        batch = test_set[i: i + batch_size]
    else:
        batch = test_set[i:]
    i += batch_size
    text, char_ids, char_mask, tag_ids = batch

    sen_len = max([len(sentence) for sentence in text])
    char_ids = char_ids[:, : sen_len].to(device)
    tag_ids = tag_ids[:, : sen_len].to(device)
    char_mask = char_mask[:, : sen_len].to(device)

    
    predict = lexicon_embeds.map_batch_to_typeid(text)
    
    for j in range(tag_ids.shape[0]):
        gold_entity = label_chinese_entity(text[j], tag_ids[j].tolist(), tag_vocab.ix_to_tag)
        pred_entity = lexicon_embeds.map_typeids_to_entity(text[j], predict[j])
        gold_num += len(gold_entity)
        predict_num += len(pred_entity)
        correct_entity = []
        relax_correct_entity = []
        relax_correct_entity_gold = []
        relax_correct_entity_pred = []
        print(''.join(text[j]))
        for gold in gold_entity:
            for pred in pred_entity:
                # [start, end)
                if max(pred['start_pos'], gold['start_pos']) < min(pred['end_pos'], gold['end_pos']) and pred['label'] == gold['label']:
                    relax_correct_num += 1
                    if gold == pred:
                        correct_num += 1
                        correct_entity.append(gold)
                    else:
                        relax_correct_entity.append([gold, pred])
                        relax_correct_entity_gold.append(gold)
                        relax_correct_entity_pred.append(pred)
        
        print('correct:')
        for e in correct_entity:
            print(e)
        print('relax correct:')
        for e in relax_correct_entity:
            print(e[0], e[1])
        print('gold:')
        for e in gold_entity:
            if e not in correct_entity and e not in relax_correct_entity_gold:
                print(e)
        print('predict:')
        for e in pred_entity:
            if e not in correct_entity and e not in relax_correct_entity_pred:
                print(e)
        print()

precision = correct_num / (predict_num + 0.000000001)
recall = correct_num / (gold_num + 0.000000001)
f1 = 2 * precision * recall / (precision + recall + 0.000000001)
logger('[Test] Precisely matching:')
logger('[Test] Precision: {:.8f} Recall: {:.8f} F1: {:.8f}'.format(precision, recall, f1))
precision = relax_correct_num / (predict_num + 0.000000001)
recall = relax_correct_num / (gold_num + 0.000000001)
f1 = 2 * precision * recall / (precision + recall + 0.000000001)
logger('[Test] Relaxation matching:')
logger('[Test] Precision: {:.8f} Recall: {:.8f} F1: {:.8f}'.format(precision, recall, f1))

