import torch
import torch.nn as nn
from TagVocab import TagVocab
from Utils import logger

class LexiconEmbedding(nn.Module):
    def __init__(self, embed_size = 100, tag_vocab = None, dict_path = None):
        super().__init__()
        if dict_path is None:
            dict_path = '/data/word_dict/lexicon.txt'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if tag_vocab is not None:
            self.tag_vocab = tag_vocab
        else:
            self.tag_vocab = TagVocab()
        self.make_type_ids()
        self.load_dict(dict_path)
        
        self.type_size = len(self.type_to_id)
        self.lexicon_size = len(self.word_list)
        self.lexicon_embeds = nn.Embedding(self.type_size, embed_size)
        logger('Load lexicon. Size = {}'.format(self.lexicon_size))

    def make_type_ids(self):
        self.type_list = self.tag_vocab.type_list
        self.type_to_id = {self.type_list[i]: i + 1 for i in range(len(self.type_list))}
        self.type_to_id['O'] = 0
    
    def load_dict(self, dict_path):
        f = open(dict_path, 'r')
        self.word_list = []
        self.word_to_type = {} # {左股二头肌: 解剖部位}
        self.word_to_typeid = {} # {左股二头肌: 解剖部位}
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            text, tag = line.split('\t')
            if tag in self.type_list:
                self.word_list.append(text)
                self.word_to_type[text] = tag
                self.word_to_typeid[text] = self.type_to_id[tag]
        self.word_list.sort(key = lambda x: len(x), reverse = True) # desceding order
        f.close()

    def forward(self, text):
        '''
        input:
            text: list(list)
        output:
            lexicon_emb: (batch_size, sen_len, emb_size)
        '''
        type_ids = self.map_batch_to_typeid(text)
        type_ids = torch.tensor(type_ids, dtype = torch.long).to(self.device)
        lexicon_emb = self.lexicon_embeds(type_ids)
        return lexicon_emb

    def map_batch_to_typeid(self, batch):
        batch_ids = []
        sen_len = max([len(sen) for sen in batch])
        for text in batch:
            batch_ids.append(self.map_sentence_to_typeid(''.join(text), sen_len))
        return batch_ids

    def map_sentence_to_typeid(self, text, sen_len):
        type_ids = [0 for _ in range(sen_len)]
        marked = [False for _ in range(sen_len)]
        for word in self.word_list:
            base = 0
            pos = text.find(word)
            while pos != -1:
                start_pos = base + pos
                end_pos = start_pos + len(word)
                if not marked[start_pos] and not marked[end_pos]:
                    for i in range(base + pos, base + pos + len(word)):
                        type_ids[i] = self.word_to_typeid[word]
                        marked[i] = True
                base = end_pos
                pos = text[base:].find(word)
        # print(type_ids)
        return type_ids
    
    # def make_tag_ids(self): # BIOES
    #     self.type_list = self.tag_vocab.type_list
    #     self.pos_list = self.tag_vocab.pos_list
    #     self.tag_to_ix = self.tag_vocab.tag_to_ix
    #     self.type_to_tags = {type: {pos: pos + '-' + type for pos in self.pos_list} for type in self.type_list}
    #     self.type_to_tagid = {type: {pos: self.tag_to_ix[pos + '-' + type] for pos in self.pos_list} for type in self.type_list}
    #     print(self.type_to_tagid)

if __name__ == '__main__':
    dict_embeds = LexiconEmbedding()
    #dict_embeds.map_sentence_to_typeid(str('，外院B超示：盆腔内2个巨大团块，考虑卵巢来源可能性大；进行双侧盆腔淋巴结清扫手术。'))
    dict_embeds.map_batch_to_typeid([str('，外院B超示：盆腔内2个巨大团块，考虑卵巢来源可能性大；'), str('进行双侧盆腔淋巴结清扫手术。')])
