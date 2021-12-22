import copy
import sys
import time
import torch

# load dataset
def bio1_bio2_simple(tags):
    for i in range(len(tags)):
        if i != 0 and tags[i][0] == 'I' and tags[i - 1][0] != 'I':
            tags[i] = 'B'
        else:
            tags[i] = tags[i][0]
    return tags

def bio1_bio2(tags):
    # tag_to_ix = {'B': 0, 'I': 1, 'O': 2, '<START>': 3, '<STOP>': 4}
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

    sentences = [[[], []]]
    for line in lines:
        line = line.strip()
        if not line:
            sentences.append([[], []])
        else:
            word, pos, syntactic, ner = line.split(' ')
            sentences[-1][0].append(word)
            sentences[-1][1].append(ner)
    if not sentences[-1]:
        sentences.pop()
    if scheme.upper() == 'SIMPLE':
        sentences = [[sentence[0], bio1_bio2_simple(sentences[1])] for sentence in sentences]
    elif scheme.upper() == 'BIO2':
        sentences = [[sentence[0], bio1_bio2(sentence[1])] for sentence in sentences]
    elif scheme.upper() == 'BIOES':
        sentences = [[sentence[0], bio1_bioes(sentence[1])] for sentence in sentences]
    return sentences 
    
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

def label_batch_entity(tags, tag_list, scheme="BIOES"):
    entity = []
    for i in tags.shape[0]:
        entity += label_sentence_entity(tags[i], tag_list, scheme)

def label_sentence_entity(text, tags, tag_list):
    if type(tags) == torch.Tensor:
        tags = tags.tolist()
    tags = [tag_list[tag] for tag in tags]
    entity = []
    count = len(tags)
    i = 0
    while i < count:
        if tags[i].startswith("B-"):
            j = i + 1
            while j < count:
                if tags[j].startswith("E-"):
                    break
                else:
                    j += 1
            entity.append({
                "text": ''.join(text[i: j]),
                "start_index": i,
                "end_index": j,
                "label": tags[i][2:]
            })
            i = j + 1
        elif tags[i].startswith("S-"):
            entity.append({
                "text": text[i],
                "start_index": i,
                "end_index": i,
                "label": tags[i][2:]
            })
            i += 1
        else:
            i += 1
    return entity

def logger(content):
    time_stamp = time.strftime("%m-%d %H:%M:%S", time.localtime())
    sys.stdout.write('[{}] {}\n'.format(time_stamp, content))
    sys.stderr.write('[{}] {}\n'.format(time_stamp, content))

def get_charset(sentences):
    # [[word1, word2, ..., word n], [], ...]
    char_set = set()
    for sentence in sentences:
        for word in sentence:
            char_set |= set([c for c in word])
    char_list = list(char_set)
    char_to_ix = {char_list[idx]: idx + 1 for idx in range(len(char_list))} # 0 for padding
    return char_to_ix

def get_wordset(sentences):
    # [[word1, word2, ..., word n], [], ...]
    word_to_ix = {}
    for sentence in sentences:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix) + 1 # 0 for padding
    return word_to_ix

def get_tagset(scheme = 'BIOES'):
    if scheme.upper() == 'BIOES':
        tag_list = ['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER',
                    'E-LOC', 'E-MISC', 'E-ORG', 'E-PER', 'S-LOC', 'S-MISC', 'S-ORG', 'S-PER',
                    'O']
    else:
        tag_list = ['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O']
    #tag_to_ix = {tag_list[i]: i for i in range(len(tag_list))}
    tag_to_ix = {tag_list[i]: i + 1 for i in range(len(tag_list))}  # 0 for padding
    return tag_to_ix

def padding(sentences, padding_value = 0, dim = 2):
    '''
    sentences: list(list), (list(list(list)))
    dim: 
        dim = 2, word padding, result = (batch_size, sen_len)
        dim = 3, char padding, result = (batch_size, sen_len, word_len)
    '''
    max_sen_len = max([len(sentence) for sentence in sentences])
    if dim == 2: # word padding
        padded_data = [sentence + [padding_value] * (max_sen_len - len(sentence)) for sentence in sentences]
        padded_mask = [[1] * len(sentence) + [0] * (max_sen_len - len(sentence)) for sentence in sentences]
        return padded_data, padded_mask
    if dim == 3: # char padding, [[[char1, char2, ..], [], ...]]
        max_word_len = max([max([len(word) for word in sentence]) for sentence in sentences])
        zero_padding = [padding_value] * max_word_len # [0, 0, 0, ..]
        zero_mask = [0] * max_word_len
        padded_data = [[word + [padding_value] * (max_word_len - len(word))for word in sentence] + [zero_padding] * (max_sen_len - len(sentence)) for sentence in sentences]
        padded_mask = [[[1] * len(word) + [0] * (max_word_len - len(word))for word in sentence] + [zero_mask] * (max_sen_len - len(sentence)) for sentence in sentences]
        return padded_data, padded_mask

def check_input_word(word_ids, word_vocab):
    for sentence in word_ids:
        print(sentence)
        print([word_vocab.word_list[word] for word in sentence])
