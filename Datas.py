import copy
import torch
from torch.utils.data import Dataset, Sampler
from Utils import get_charset, get_wordset, get_tagset, padding, load_glove

def bio1_bio2_simple(tags):
    for i in range(len(tags)):
        if i != 0 and tags[i][0] == 'I' and tags[i - 1][0] != 'I':
            tags[i] = 'B'
        else:
            tags[i] = tags[i][0]
    return tags

def bio1_bio2(tags):
    for i in range(len(tags)):
        if i != 0 and tags[i][0] == 'I' and tags[i - 1][0] != 'I':
            tags[i] = 'B' + tags[i][1:]
    return tags

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
    
def load_conll(path, scheme = 'BIOES'):
    # [[[word1, word2, ..., word n], [tag1, tag2, ..., tag n]], [[], []], ...]
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    datas = [[]]
    tags = [[]]
    seq_length = []
    for line in lines:
        line = line.strip()
        if not line:
            datas.append([])
            tags.append([])
        else:
            word, pos, syntactic, ner = line.split(' ')
            datas[-1].append(word)
            tags[-1].append(ner)
    if not datas[-1]:
        datas.pop()
        tags.pop()
    if scheme.upper() == 'BIOES':
        tags = [bio1_bioes(tag) for tag in tags]
    return datas, tags
    
def load_conll_dict(path, scheme = 'BIOES'):
    # [[{'word': word, 'tag': tag}, {}, ...], [], ...]
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    sentences = [[]]
    for line in lines:
        line = line.strip()
        if not line:
            sentences.append([])
        else:
            word, pos, syntactic, ner = line.split(' ')
            sentences[-1].append({"word": word, "tag": ner})
    if not sentences[-1]:
        sentences.pop()
    if scheme.upper() == 'BIOES':
        sentences = [[sentence[0], bio1_bioes(sentence[1])] for sentence in sentences]
    return sentences 

class WordVocab():
    def __init__(self):
        self._make_mapping_from_pretrained()

    def _make_mapping_from_pretrained(self):
        path = '/home/gene/Documents/Data/Glove/glove.6B.100d.txt'
        embed_list, embed_word_list, embed_word_to_ix = load_glove(path)
        self.embed_list = embed_list
        self.word_list = embed_word_list
        self.word_to_ix = embed_word_to_ix

        self.OOV_WORD = '<OOV>'
        self.PAD_WORD = '<PAD>'
        if self.PAD_WORD not in self.word_to_ix:
            self.word_to_ix[self.PAD_WORD] = 0
            self.word_list.insert(0, '<PAD>')
            torch.cat((torch.tensor([[0. for _ in range(len(self.embed_list[0]))]]), self.embed_list), 0)
        if self.OOV_WORD not in self.word_to_ix:
            self.word_to_ix[self.OOV_WORD] = len(self.word_to_ix)
            self.word_list.append('<OOV>')
            torch.cat((self.embed_list, torch.tensor([[0. for _ in range(len(self.embed_list[0]))]])), 0)
        # self.word_list = [key for key in self.word_to_ix]
        
    def _make_mapping(self, sentences):
        self.word_to_ix = get_wordset(sentences)
        self.OOV_WORD = '<OOV>'
        self.PAD_WORD = '<PAD>'
        if self.PAD_WORD not in self.word_to_ix:
            self.word_to_ix[self.PAD_WORD] = 0
        if self.OOV_WORD not in self.word_to_ix:
            self.word_to_ix[self.OOV_WORD] = len(self.word_to_ix)
        self.word_list = [key for key in self.word_to_ix]

    def map_to_ix(self, sentence):
        return [self.word_to_ix[word] if word in self.word_to_ix else self.word_to_ix[self.OOV_WORD] for word in sentence]

class CharVocab():
    def __init__(self, sentences):
        self._make_mapping(sentences)
    
    def _make_mapping(self, sentences):
        self.char_to_ix = get_charset(sentences)
        self.OOV_WORD = '<OOV>'
        self.PAD_WORD = '<PAD>'
        if self.PAD_WORD not in self.char_to_ix:
            self.char_to_ix[self.PAD_WORD] = 0
        if self.OOV_WORD not in self.char_to_ix:
            self.char_to_ix[self.OOV_WORD] = len(self.char_to_ix)

    def map_to_ix(self, sentence):
        return [[self.char_to_ix[char] if char in self.char_to_ix else self.char_to_ix[self.OOV_WORD] for char in word] for word in sentence]

# map BIOES tags
class TagVocab():
    def __init__(self, scheme):
        self.scheme = scheme
        self._make_mapping()
    
    def _make_mapping(self):
        self.tag_to_ix = get_tagset(self.scheme)
        self.START_TAG = 'START'
        self.STOP_TAG = 'STOP'
        self.PAD_TAG = '<PAD>'
        if self.PAD_TAG not in self.tag_to_ix:
            self.tag_to_ix[self.PAD_TAG] = 0
        if self.START_TAG not in self.tag_to_ix:
            self.tag_to_ix[self.START_TAG] = len(self.tag_to_ix)
        if self.STOP_TAG not in self.tag_to_ix:
            self.tag_to_ix[self.STOP_TAG] = len(self.tag_to_ix)
        self.ix_to_tag = {value: key for key, value in self.tag_to_ix.items()}

    def map_to_ix(self, tags):
        return [self.tag_to_ix[tag] for tag in tags]

# 
class ConllDataset(Dataset):
    def __init__(self, path, word_vocab = None, char_vocab = None, tag_vocab = None, scheme = 'BIOES', make_map = True):
        super().__init__()
        self.sentences, self.tags = load_conll(path, scheme)
        #self.data = load_conll_dict(path, scheme)
        self.scheme = scheme
        if make_map:
            self.word_vocab = WordVocab()
            self.char_vocab = CharVocab(self.sentences)
            self.tag_vocab = TagVocab(self.scheme)
        else:
            self.word_vocab = word_vocab
            self.char_vocab = char_vocab
            self.tag_vocab = tag_vocab
        self.data = []
        for i in range(len(self.sentences)): # sentence
            self.data.append({
                'text': self.sentences[i],
                'word_ids': self.word_vocab.map_to_ix(self.sentences[i]),
                'char_ids': self.char_vocab.map_to_ix(self.sentences[i]),
                'tag_ids':  self.tag_vocab.map_to_ix(self.tags[i])
            })

    def __len__(self):
        return len(self.data) # number of sentence

    def __getitem__(self, index):
        # [word1, word2, ..., word n], [tag1, tag2, ..., tag n] (without mapping)
        return self.data[index]
    
    def get_mapping(self):
        return self.word_vocab, self.char_vocab, self.tag_vocab
    
    def set_mapping(self, word_vocab, char_vocab, tag_vocab):
        self.word_vocab, self.char_vocab, self.tag_vocab = word_vocab, char_vocab, tag_vocab

def collate_fn_conll(batch_data):
    '''
    input:
        batch_data: [dataset[i] for i in indices], (batch_size, sen_len)
    output:
        padded_data: dict
        word_ids: tensor: (batch_size, sen_len)
        char_ids: tensor: (batch_size, sen_len, word_len)
        tag_ids: tensor: (batch_size, sen_len)
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    padded_batch = {}
    padded_batch["text"] = [d["text"] for d in batch_data]
    word_ids, word_mask = padding([d["word_ids"] for d in batch_data], dim = 2)
    padded_batch["word_ids"] = torch.tensor(word_ids, dtype = torch.long, device = device)
    padded_batch["word_mask"] = torch.tensor(word_mask, dtype = torch.bool, device = device)
    char_ids, char_mask = padding([d["char_ids"] for d in batch_data], dim = 3)
    padded_batch["char_ids"] = torch.tensor(char_ids, dtype=torch.long, device = device)
    padded_batch["char_mask"] = torch.tensor(char_mask, dtype=torch.bool, device = device)
    dim = 2 if type(batch_data[0]["tag_ids"][0]) == int else 3
    padded_batch["tag_ids"] = torch.tensor(padding([d["tag_ids"] for d in batch_data], dim=dim)[0], dtype=torch.long, device=device)
    return padded_batch

