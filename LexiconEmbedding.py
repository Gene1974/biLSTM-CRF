import re
import torch
import torch.nn as nn
from TagVocab import TagVocab
from Utils import logger

class LexiconVocab():
    def __init__(self, tag_vocab = None, dict_path = None, expand_lexicon = False):
        if dict_path is None:
            dict_path = '/data/Lexicon/lexicon.txt'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if tag_vocab is not None:
            self.tag_vocab = tag_vocab
        else:
            self.tag_vocab = TagVocab()
        
        self.rules = [
            ['解剖部位', 1, ['癌', '肿瘤'], '疾病和诊断'], 
            ['解剖部位', 1, ['活检术', '切除术', '植入术', '造口术', '造痿术'], '手术']
            ]
        self.make_type_ids()
        self.make_tag_ids()
        self.load_dict(dict_path, expand_lexicon)
        self.type_size = len(self.type_to_id)
        self.lexicon_size = len(self.word_list)
        logger('Load Lexicon. Size = {}'.format(self.lexicon_size))

    def make_type_ids(self):    # only type, no BIOES
        self.type_list = self.tag_vocab.type_list
        self.type_to_id = {self.type_list[i]: i + 1 for i in range(len(self.type_list))}
        self.type_to_id['O'] = 0
        self.id_to_type = {self.type_to_id[k]: k for k in self.type_to_id}
    
    def make_tag_ids(self): # BIOES-type
        self.type_list = self.tag_vocab.type_list
        #self.pos_list = self.tag_vocab.pos_list
        self.tag_to_ix = self.tag_vocab.tag_to_ix
        self.tagid_to_tag = self.tag_vocab.ix_to_tag
        #self.type_to_tags = {type: {pos: pos + '-' + type for pos in self.pos_list} for type in self.type_list}
        #self.type_to_tagid = {type: {pos: self.tag_to_ix[pos + '-' + type] for pos in self.pos_list} for type in self.type_list}
    
    def load_dict(self, dict_path, expand_lexicon = False):
        '''
        output:
            self.word_list      sorted by len
            self.word_to_type   {左股二头肌: 解剖部位}
            self.word_to_typeid
        '''
        f = open(dict_path, 'r')
        self.word_list = []
        self.word_to_type = {} # {左股二头肌: 解剖部位}
        self.word_to_typeid = {}
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            text, tag = line.split('\t')
            if tag in self.type_list:
                self.word_list.append(text)
                self.word_to_type[text] = tag
                self.word_to_typeid[text] = self.type_to_id[tag]
        if expand_lexicon:
            self.expand_lexicon() # 16452 -> 85138
        self.word_list.sort(key = lambda x: len(x), reverse = True) # desceding order
        f.close()

    def expand_lexicon(self): # 16452 -> 85138
        rules = self.rules
        for word in self.word_list:
            for rule in rules:
                type, gap, chars, new_type = rule
                if type != self.word_to_type[word] or gap != 1:
                    continue
                for char in chars:
                    if word + char not in self.word_list:
                        new_word = word + char
                        self.word_list.append(new_word)
                        self.word_to_type[new_word] = new_type
                        self.word_to_typeid[new_word] = self.type_to_id[new_type]
        
    def map_batch_to_typeid(self, batch, use_rule = False):
        '''
        input:
            batch:      list(list)
        output:
            batch_ids:  (batch_size, max_sen_len)
        '''
        batch_ids = []
        sen_len = max([len(sen) for sen in batch])
        for text in batch:
            batch_ids.append(self.map_sentence_to_typeid(''.join(text), sen_len, use_rule))
        return batch_ids

    def map_sentence_to_typeid(self, text, sen_len, use_rule):
        '''
        input:
            text:       str
            sen_len:    max_sen_len
        output:
            type_ids:   (max_sen_len)
        '''
        if use_rule:
            rules = self.rules
        else:
            rules = []
        type_ids = [0 for _ in range(sen_len)]
        marked = [False for _ in range(sen_len)]
        for word in self.word_list:
            base = 0
            pos = text.find(word)
            while pos != -1:
                start_pos = base + pos
                end_pos = start_pos + len(word)
                if not marked[start_pos] and not marked[end_pos]:
                    type_id = self.word_to_typeid[word]
                    for rule in rules:
                        type, gap, chars, new_type = rule
                        if type != self.word_to_type[word]:
                            continue
                        for char in chars:
                            char_pos = text[end_pos:].find(char)
                            if char_pos != -1 and char_pos < gap:
                                end_pos += char_pos + len(char)
                                type_id = self.type_to_id[new_type]
                                break
                    for i in range(start_pos, end_pos):
                        type_ids[i] = type_id
                        marked[i] = True
                base = end_pos
                pos = text[base:].find(word)
        return type_ids
        
    def map_typeids_to_entity(self, text, type_ids):
        '''
        input:
            text:       str
            type_ids:   (max_sen_len)
        output:
            entity:     list(dict)
        '''
        entity = []
        count = len(type_ids)
        i = 0
        while i < count:
            if type_ids[i] != 0:
                j = i + 1
                while j < count:
                    if type_ids[j] != type_ids[i]:
                        break
                    else:
                        j += 1
                entity.append({
                    "text": ''.join(text[i: j]),
                    "start_pos": i,
                    "end_pos": j,
                    "label": self.id_to_type[type_ids[i]]
                })
                i = j
            else:
                i += 1
        return entity

    def clean_tagid_with_lexicon(self, text, tag_ids, use_rule = False):
        '''
        input:
            text:       str
            tag_ids:    (max_sen_len)
        output:
            tag_ids:    (max_sen_len)
        '''
        if use_rule:
            rules = self.rules
        else:
            rules = []
        marked = [False if t == self.tag_to_ix['O'] else True for t in range(tag_ids.shape[0])]
        
        for word in self.word_list: # 遍历 lexicon
            base = 0
            pos = text.find(word) # 找句中有没有对应的 word
            while pos != -1:
                start_pos = base + pos
                end_pos = start_pos + len(word)
                if not marked[start_pos] and not marked[end_pos]: # 标记没有标出的词汇
                    word_type = self.word_to_type[word] # 没有 rule 的时候用 word 原始的 type
                    for rule in rules:
                        type, gap, chars, new_type = rule
                        if type != self.word_to_type[word]:
                            continue
                        for char in chars:
                            char_pos = text[end_pos:].find(char)
                            if char_pos != -1 and char_pos < gap:
                                end_pos += char_pos + len(char)
                                word_type = new_type # rule 匹配的时候用新的 type
                    
                    for i in range(start_pos, end_pos):
                        marked[i] = True
                    if end_pos - start_pos == 1:
                        tag_ids[start_pos] = self.tag_to_ix['S-' + word_type]
                    else:
                        tag_ids[start_pos] = self.tag_to_ix['B-' + word_type]
                        tag_ids[end_pos - 1] = self.tag_to_ix['E-' + word_type]
                        for i in range(start_pos + 1, end_pos - 1):
                            tag_ids[i] = self.tag_to_ix['I-' + word_type]
                base = end_pos
                pos = text[base:].find(word)
        return tag_ids



class LexiconEmbedding(nn.Module):
    def __init__(self, embed_size = 100, tag_vocab = None, lexicon_vocab = None, dict_path = None):
        super().__init__()
        if dict_path is None:
            dict_path = '/data/word_dict/lexicon.txt'
        if lexicon_vocab is not None:
            self.lexicon_vocab = lexicon_vocab
        else:
            self.lexicon_vocab = LexiconVocab()
        if tag_vocab is not None:
            self.tag_vocab = tag_vocab
        else:
            self.tag_vocab = TagVocab()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embed_size = embed_size
        self.type_size = len(self.lexicon_vocab.type_to_id)
        #self.lexicon_size = len(self.lexicon_vocab.word_list)
        self.lexicon_embeds = nn.Embedding(self.type_size, embed_size)
        #logger('Load lexicon. Size = {}'.format(self.lexicon_size))
        #self.tag

    def forward(self, text):
        '''
        input:
            text: list(list)
        output:
            lexicon_emb: (batch_size, sen_len, emb_size)
        '''
        type_ids = self.lexicon_vocab.map_batch_to_typeid(text)
        type_ids = torch.tensor(type_ids, dtype = torch.long).to(self.device)
        lexicon_emb = self.lexicon_embeds(type_ids)
        return lexicon_emb
    
    def get_emb_dim(self):
        return self.embed_size



if __name__ == '__main__':
    lex_vocab = LexiconVocab()
    # lex_vocab.make_tag_ids()
    # dict_embeds = LexiconEmbedding()
    # text = '，外院B超示：盆腔内2个巨大团块，考虑卵巢来源可能性大；进行双侧盆腔淋巴结清扫手术。'
    # type_ids = dict_embeds.map_sentence_to_typeid_with_rules(text, 50)
    # print(dict_embeds.map_typeids_to_entity(text, type_ids))
    #dict_embeds.map_sentence_to_typeid_with_rules([str('，外院B超示：盆腔内2个巨大团块，考虑卵巢来源可能性大；'), str('进行双侧盆腔淋巴结清扫手术。')], 50)
