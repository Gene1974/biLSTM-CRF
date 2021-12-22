import functools
import json
import random
from torch.utils import data
import torch
from torch.utils.data import Dataset

# load pretrains
class CCKSVocab():
    def __init__(self, dataset_path = None):
        self.OOV_TAG = '<OOV>'
        self.PAD_TAG = '<PAD>'
        if dataset_path is None:
            dataset_path = '/home/gene/Documents/Data/CCKS2019/'
        self.type_list = set()
        self.char_list = set() # 1980
        self.load_dataset(dataset_path)
        self.tag_vocab = TagVocab(self.type_list)

    def load_dataset(self, path):
        self.load_ccks_word(path + 'subtask1_train/subtask1_training_part1.txt')
        self.load_ccks_word(path + 'subtask1_train/subtask1_training_part2.txt')
        self.load_ccks_word(path + 'subtask1_test/subtask1_test_set_with_answer.json')
        self.char_list = [self.PAD_TAG, self.OOV_TAG] + list(self.char_list)
        self.char_to_ix = {self.char_list[index]: index for index in range(len(self.char_list))}
        self.type_list = list(self.type_list)

    def load_ccks_word(self, path):
        f = open(path, 'r', encoding = 'utf-8-sig')
        for line in f.readlines():
            line = line.strip()
            if line:
                data = json.loads(line)
                self.char_list |= set(data['originalText'])
                self.type_list |= set([entity['label_type'] for entity in data['entities']])
        f.close()
    
    def map_char(self, words, dim = 3):
        if dim == 3: # words
            return [[self.char_to_ix[char] if char in self.char_to_ix else self.char_to_ix[self.OOV_TAG] for char in word] for word in words]
        if dim == 2: # word
            return [self.char_to_ix[char] if char in self.char_to_ix else self.char_to_ix[self.OOV_TAG] for char in words]
    
    def map_word(self, words, dim = 2):
        if dim == 2: # words
            return [self.word_to_ix[word] if word in self.word_to_ix else self.word_to_ix[self.OOV_TAG] for word in words]
        if dim == 1:
            return self.word_to_ix[words] if words in self.word_to_ix else self.word_to_ix[self.OOV_TAG]

# map BIOES tags
class TagVocab():
    def __init__(self, type_list = None):
        self.START_TAG = 'START'
        self.STOP_TAG = 'STOP'
        self.PAD_TAG = '<PAD>'

        pos_list = ['B', 'I', 'E', 'S']
        if type_list == None:
            type_list = ['疾病和诊断', '检查', '检验', '手术', '药物', '解剖部位']
        tag_list = []
        for pos in pos_list:
            tag_list = tag_list + [pos + '-' + types for types in type_list]
        tag_list.append('O')
        self.tag_to_ix = {tag_list[i]: i + 1 for i in range(len(tag_list))}
        self.tag_to_ix[self.PAD_TAG] = 0
        self.tag_to_ix[self.START_TAG] = len(self.tag_to_ix)
        self.tag_to_ix[self.STOP_TAG] = len(self.tag_to_ix)
        self.ix_to_tag = {value: key for key, value in self.tag_to_ix.items()}

    def map_tag(self, tags, dim = 2):
        if dim == 2:
            return [self.tag_to_ix[tag] for tag in tags]
        if dim == 1:
            return self.tag_to_ix[tags]

class CCKSDataset(Dataset):
    def __init__(self, path = None, word_vocab = None, tag_vocab = None, mod = 'train', ratio = 0.8, shuffle = False):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_sen_len = 128
        self.PAD_TAG = '<PAD>'
        if path == None:
            path = '/home/gene/Documents/Data/CCKS2019/'
        self.path = path
        self.mod = mod
        self.shuffle = shuffle

        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab

        self.originalText = []
        self.entities = []
        self.text = []
        self.char_ids = []
        self.char_masks = []
        self.tag_ids = []

        if mod == 'train':
            self.load_ccks(path + 'subtask1_train/subtask1_training_part1.txt')
            self.load_ccks(path + 'subtask1_train/subtask1_training_part2.txt')
            self.char_ids = torch.tensor(self.char_ids, dtype = torch.long)
            self.char_masks = torch.tensor(self.char_masks, dtype = torch.bool)
            self.tag_ids = torch.tensor(self.tag_ids, dtype = torch.long)
            self.split_data(ratio = ratio, shuffle = shuffle)
        elif mod == 'test':
            self.load_ccks(path + 'subtask1_test/subtask1_test_set_with_answer.json')
            self.char_ids = torch.tensor(self.char_ids, dtype = torch.long)
            self.char_masks = torch.tensor(self.char_masks, dtype = torch.bool)
            self.tag_ids = torch.tensor(self.tag_ids, dtype = torch.long)

    def load_ccks(self, path):
        length = self.max_sen_len
        f = open(path, 'r', encoding = 'utf-8-sig')
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            text = data['originalText']
            entity = data['entities']
            
            begin_index = 0 # index of text
            item_index = 0  # index of entity
            '''
            text:     original text
            sentence: modified text
            '''
            sentence = text.replace('；', '。').replace('？', '。').replace('！', '。').replace(';', '。').replace('?', '。').replace('!', '。').split('。')
            for sen in sentence:
                sub_entity = [] # entity in a sub_sentence
                end_index = begin_index + len(sen) + 1
                while len(sen) > length - 1: # 加上句号小于128
                    sub_sen = sen[:length]
                    pos = sub_sen.replace(',', '，').rfind('，')
                    if pos == -1 or pos == 0:
                        pos = length - 1
                    end_index = begin_index + pos + 1
                    while item_index < len(entity) and entity[item_index]['start_pos'] < end_index:
                        item = entity[item_index]
                        item['text'] = text[item['start_pos']: item['end_pos']]
                        item['start_pos'] = item['start_pos'] - begin_index
                        item['end_pos'] = item['end_pos'] - begin_index
                        sub_entity.append(item)
                        item_index += 1
                    self.label_data(text[begin_index: end_index], sub_entity)
                    sub_entity = []
                    sen = sen[pos + 1:]
                    begin_index = end_index
                    end_index = begin_index + len(sen) + 1
                if not sen: # sen = ''
                    continue
                
                # sen < 128
                while item_index < len(entity) and entity[item_index]['start_pos'] < end_index:
                    item = entity[item_index]
                    item['text'] = text[item['start_pos']: item['end_pos']]
                    item['start_pos'] = item['start_pos'] - begin_index
                    item['end_pos'] = item['end_pos'] - begin_index
                    sub_entity.append(item)
                    item_index += 1
                self.label_data(text[begin_index: end_index], sub_entity)
                sub_entity = []
                begin_index = end_index
                end_index = begin_index + len(sen) + 1
    
    def label_data(self, text, entities):
        '''
        input: 
            text: str
            enities: list(dict)
        '''
        sen_len = len(text)
        text = list(text)
        tags = ['O' for _ in range(sen_len)]
        for entity in entities:
            start_pos = entity['start_pos']
            end_pos = entity['end_pos']
            end_pos = min(end_pos, sen_len)
            if end_pos == start_pos + 1: # [start_pos, end_pos)
                tags[start_pos] = 'S-' + entity['label_type']
            else:
                tags[start_pos] = 'B-' + entity['label_type']
                tags[end_pos - 1] = 'E-' + entity['label_type']
                for i in range(start_pos + 1, end_pos - 1):
                    tags[i] = 'I-' + entity['label_type']
        self.map_and_pad(text, tags)
    
    def map_and_pad(self, text, tags):
        '''
        input: 
            text: list(char)
            tags: list(char)
        '''
        char_ids = self.word_vocab.map_char(text, dim = 2)
        tag_ids = self.tag_vocab.map_tag(tags)
        
        char_ids, char_masks = self.padding_fixed(char_ids, padding_value = 0, dim = 2)
        tag_ids, _ = self.padding_fixed(tag_ids, padding_value = 0, dim = 2)

        self.text.append(text) # Elmo 不用 cut 多余的 char
        self.char_ids.append(char_ids)
        self.char_masks.append(char_masks)
        self.tag_ids.append(tag_ids)
    
    def padding_fixed(self, sentence, padding_value = 0, dim = 2):
        '''
        sentences: list, (list(list))
        dim: 
            dim = 2, word padding, result = (sen_len)
            dim = 3, char padding, result = (sen_len, word_len)
        '''
        max_sen_len = self.max_sen_len
        if dim == 2: # word padding
            padded_data = sentence + [padding_value] * (max_sen_len - len(sentence))
            padded_mask = [1] * len(sentence) + [0] * (max_sen_len - len(sentence))
            return padded_data[: max_sen_len], padded_mask[: max_sen_len]

    # split data into trainset and testset
    def split_data(self, ratio = 0.8, shuffle = False):
        self.total_num = len(self.char_ids)
        if shuffle:
            index = list(range(self.total_num))
            random.shuffle(index)
            self.text = self.text[index]
            self.char_ids = self.char_ids[index]
            self.char_masks = self.char_masks[index]
            self.tag_ids = self.tag_ids[index]
        if self.mod == 'train':
            self.train_num = int(ratio * self.total_num)
            self.valid_num = self.total_num - self.train_num

            self.valid_set = CCKSDataset(mod = 'valid')
            self.valid_set.text = self.text[self.train_num:]
            self.valid_set.char_ids = self.char_ids[self.train_num:]
            self.valid_set.char_masks = self.char_masks[self.train_num:]
            self.valid_set.tag_ids = self.tag_ids[self.train_num:]
            self.text = self.text[: self.train_num]
            self.char_ids = self.char_ids[: self.train_num]
            self.char_masks = self.char_masks[: self.train_num]
            self.tag_ids = self.tag_ids[: self.train_num]
    
    def __len__(self):
        return len(self.text) # number of sentence

    def __getitem__(self, index):
        return self.text[index], self.char_ids[index], self.char_masks[index], self.tag_ids[index]


# def collate_conll(batch_data):
#     '''
#     input:
#         batch_data: [dataset[i] for i in indices], (batch_size, sen_len)
#     output:
#         word_ids: tensor: (batch_size, sen_len)
#         tag_ids: tensor: (batch_size, sen_len)
#     '''
#     text, word_ids, tag_ids, word_mask = batch_data
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     sen_len = torch.max(torch.sum(word_mask, dim = 1, dtype = torch.int64)).item()
#     print(sen_len, word_ids.shape, tag_ids.shape)

#     word_ids = word_ids[:, : sen_len].to(device)
#     tag_ids = tag_ids[:, : sen_len].to(device)
#     word_mask = word_mask[:, : sen_len].to(device)
#     return text, word_ids, tag_ids, word_mask

if __name__ == '__main__':
    # path = '/home/gene/Documents/Data/CCKS2019/'
    # file_name = 'subtask1_train/subtask1_training_part1.txt'
    # dataset = CCKSDataset()
    # dataset.load_ccks(path + file_name)
    tag_vocab = TagVocab()