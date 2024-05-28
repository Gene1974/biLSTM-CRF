import copy
import torch
from torch.utils.data import Dataset

def bio1_bioes(tags):
    # [tag1, tag2, ..., tag n]
    new_tags = copy.deepcopy(tags)
    i = 0
    while i < len(tags):
        if tags[i][0] == 'I':
            new_tags[i] = 'B' +  tags[i][1:]
        if new_tags[i][0] == 'B':
            j = i + 1
            while j < len(tags) and tags[j][0] == 'I':
                j += 1
            entity_length = j - i
            if entity_length == 1:
                new_tags[i] = 'S' + tags[i][1:]
            else:
                new_tags[j - 1] = 'E' + tags[j - 1][1:]
            i = j
        else:
            i += 1
    return new_tags
    

# load pretrains
class WordVocab():
    def __init__(self, dataset_path = None):
        self.OOV_TAG = '<OOV>'
        self.PAD_TAG = '<PAD>'
        if dataset_path is None:
            dataset_path = '/data/CoNLL2003/'
        pretrained_path = '/data/Glove/glove.6B.100d.txt'
        self.load_dataset(dataset_path)
        #self.load_glove(pretrained_path)
        self.load_senna(pretrained_path)

    def load_dataset(self, path):
        self.word_list = [self.PAD_TAG, self.OOV_TAG] # 没有大写
        self.word_to_ix = {self.PAD_TAG: 0, self.OOV_TAG: 1} # 没有大写
        self.char_to_ix = {self.PAD_TAG: 0, self.OOV_TAG: 1} # 有大写
        self.load_conll_word(path + 'eng.train')
        self.load_conll_word(path + 'eng.testa')
        self.load_conll_word(path + 'eng.testb')

    def load_conll_word(self, path):
        f = open(path, 'r')
        for line in f.readlines():
            line = line.strip()
            if line:
                word, _, _, _ = line.split(' ')
                for char in word: # Char Embedding 要保留大写
                    if char not in self.char_to_ix:
                        self.char_to_ix[char] = len(self.char_to_ix)
                word = word.lower()
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)
                    self.word_list.append(word)
        f.close()
        
    def load_glove(self, path):
        word_to_ix = {self.PAD_TAG: 0, self.OOV_TAG: 1}
        word_emb = []
        with open(path, 'r') as glove:
            for line in glove.readlines():
                data = line.strip().split(' ') # [word emb1 emb2 ... emb n]
                word = data[0]
                embeds = [float(i) for i in data[1:]]
                word_to_ix[word] = len(word_to_ix)
                word_emb.append(embeds)
        
        word_emb.insert(0, [0.] * len(word_emb[0]))
        word_emb.insert(0, [0.] * len(word_emb[0]))
        
        used_idx = [word_to_ix[word] if word in word_to_ix else word_to_ix[self.OOV_TAG] for word in self.word_list]
        self.word_emb = torch.tensor(word_emb, dtype = torch.float)[used_idx]
        return self.word_emb
        
    def load_senna(self, path):
        path = '/data/Senna/'
        word_to_ix = {self.PAD_TAG: 0, self.OOV_TAG: 1}
        word_emb = []
        with open(path + 'word_list.txt', 'r') as f:
            for line in f.readlines():
                word = line.strip()
                word_to_ix[word] = len(word_to_ix)
        with open(path + 'embeddings.txt', 'r') as f:
            for line in f.readlines():
                embeds = line.strip().split(' ')
                embeds = [float(i) for i in embeds]
                word_to_ix[word] = len(word_to_ix)
                word_emb.append(embeds)
        
        word_emb.insert(0, [0.] * len(word_emb[0]))
        word_emb.insert(0, [0.] * len(word_emb[0]))
        
        used_idx = [word_to_ix[word] if word in word_to_ix else word_to_ix[self.OOV_TAG] for word in self.word_list]
        self.word_emb = torch.tensor(word_emb, dtype = torch.float)[used_idx]
        return self.word_emb
    
    def map_word(self, words, dim = 2):
        if dim == 2: # words
            return [self.word_to_ix[word.lower()] if word.lower() in self.word_to_ix else self.word_to_ix[self.OOV_TAG] for word in words]
        if dim == 1:
            return self.word_to_ix[words] if words in self.word_to_ix else self.word_to_ix[self.OOV_TAG]
    
    def map_char(self, words, dim = 2):
        if dim == 2: # words
            return [[self.char_to_ix[char] if char in self.char_to_ix else self.char_to_ix[self.OOV_TAG] for char in word] for word in words]
        if dim == 1: # word
            return [self.char_to_ix[char] if char in self.char_to_ix else self.char_to_ix[self.OOV_TAG] for char in words]

# map BIOES tags
# class TagVocab():
#     def __init__(self):
#         self.START_TAG = 'START'
#         self.STOP_TAG = 'STOP'
#         self.PAD_TAG = '<PAD>'

#         tag_list = ['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER',
#                     'E-LOC', 'E-MISC', 'E-ORG', 'E-PER', 'S-LOC', 'S-MISC', 'S-ORG', 'S-PER',
#                     'O']
#         self.tag_to_ix = {tag_list[i]: i + 1 for i in range(len(tag_list))}
#         self.tag_to_ix[self.PAD_TAG] = 0
#         self.tag_to_ix[self.START_TAG] = len(self.tag_to_ix)
#         self.tag_to_ix[self.STOP_TAG] = len(self.tag_to_ix)
#         self.ix_to_tag = {value: key for key, value in self.tag_to_ix.items()}

#     def map_tag(self, tags, dim = 2):
#         if dim == 2:
#             return [self.tag_to_ix[tag] for tag in tags]
#         if dim == 1:
#             return self.tag_to_ix[tags]

class ConllDataset(Dataset):
    def __init__(self, path, word_vocab = None, tag_vocab = None):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_sen_len = 64
        self.max_word_len = 16
        self.PAD_TAG = '<PAD>'

        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab
            
        self.text = []
        self.word_ids = []
        self.char_ids = []
        self.tag_ids = []
        self.word_masks = []
        #self.char_masks = []
        self.load_conll(path)
        
        self.word_ids = torch.tensor(self.word_ids, dtype = torch.long)
        self.char_ids = torch.tensor(self.char_ids, dtype = torch.long)
        self.tag_ids = torch.tensor(self.tag_ids, dtype = torch.long)
        self.word_masks = torch.tensor(self.word_masks, dtype = torch.bool)
        #self.char_masks = torch.tensor(self.char_masks, dtype = torch.bool, device = self.device)
    
    def load_conll(self, path):
        f = open(path, 'r')
        text = []
        tags = []
        for line in f.readlines():
            line = line.strip()
            if not line:
                self.map_and_pad(text, tags)
                text = []
                tags = []
            else:
                word, _, _, ner = line.split(' ')
                text.append(word) # 保留大写
                tags.append(ner)
        f.close()
    
    def map_and_pad(self, text, tags):
        tags = bio1_bioes(tags)
        word_ids = self.word_vocab.map_word(text, dim = 2)
        char_ids = self.word_vocab.map_char(text, dim = 2)
        tag_ids = self.tag_vocab.map_tag(tags)
        
        word_ids, word_mask = self.padding_fixed(word_ids, padding_value = 0, dim = 2)
        char_ids, char_mask = self.padding_fixed(char_ids, padding_value = 0, dim = 3)
        tag_ids, _ = self.padding_fixed(tag_ids, padding_value = 0, dim = 2)

        self.text.append(text[:self.max_sen_len]) # Elmo 不用 cut 多余的 char
        self.word_ids.append(word_ids)
        self.char_ids.append(char_ids)
        self.tag_ids.append(tag_ids)
        self.word_masks.append(word_mask)
        #self.char_masks.append(char_mask)
    
    def padding_fixed(self, sentence, padding_value = 0, dim = 2):
        '''
        sentences: list, (list(list))
        dim: 
            dim = 2, word padding, result = (sen_len)
            dim = 3, char padding, result = (sen_len, word_len)
        '''
        max_sen_len = self.max_sen_len
        max_word_len = self.max_word_len
        if dim == 2: # word padding
            padded_data = sentence + [padding_value] * (max_sen_len - len(sentence))
            padded_mask = [1] * len(sentence) + [0] * (max_sen_len - len(sentence))
            return padded_data[: max_sen_len], padded_mask[: max_sen_len]
        if dim == 3: # char padding, [[char1, char2, ..], [], ...]
            zero_padding = [padding_value] * max_word_len # [0, 0, 0, ..]
            zero_mask = [0] * max_word_len
            padded_data = [word[: max_word_len] + [padding_value] * (max_word_len - len(word)) for word in sentence] + [zero_padding] * (max_sen_len - len(sentence))
            padded_mask = [[1] * len(word[: max_word_len]) + [0] * (max_word_len - len(word)) for word in sentence] + [zero_mask] * (max_sen_len - len(sentence))
            return padded_data[: max_sen_len], padded_mask[: max_sen_len]

    def __len__(self):
        return len(self.text) # number of sentence

    def __getitem__(self, index):
        # [word1, word2, ..., word n], [tag1, tag2, ..., tag n] (without mapping)
        #return self.text[index], self.word_ids[index], self.char_ids[index], self.tag_ids[index], self.word_masks[index], self.char_masks[index]
        return self.text[index], self.word_ids[index], self.char_ids[index], self.tag_ids[index], self.word_masks[index]

    def get_mapping(self):
        return self.word_vocab, self.tag_vocab


def collate_conll(batch_data):
    '''
    input:
        batch_data: [dataset[i] for i in indices], (batch_size, sen_len)
    output:
        padded_data: dict
        word_ids: tensor: (batch_size, sen_len)
        char_ids: tensor: (batch_size, sen_len, word_len)
        tag_ids: tensor: (batch_size, sen_len)
    '''
    print(len(batch_data))
    word_ids, char_ids, tag_ids, word_mask = batch_data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sen_len = torch.max(torch.sum(word_mask, dim = 1, dtype = torch.int64)).item()

    word_ids = word_ids[:, : sen_len].to(device)
    char_ids = char_ids[:, : sen_len, :].to(device)
    tag_ids = tag_ids[:, : sen_len].to(device)
    word_mask = word_mask[:, : sen_len].to(device)
    return word_ids, char_ids, tag_ids, word_mask

if __name__ == '__main__':
    loader = WordVocab()
