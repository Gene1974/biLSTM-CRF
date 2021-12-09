import torch
import torch.nn as nn
from Utils import logger

class WordEmbedding(nn.Module):
    def __init__(self, word_vocab, use_pretrained):
        super().__init__()
        self.word_emb_size = 100
        self.word_vocab = word_vocab
        self.n_words = len(word_vocab.word_to_ix)
        if use_pretrained:
            # path = '/home/gene/Documents/Data/Glove/glove.6B.100d.txt'
            # embed_list, embed_word_list, embed_word_to_ix = load_glove(path)
            self.embed_list = word_vocab.embed_list
            self.word_embeds = nn.Embedding.from_pretrained(self.embed_list)
        else:
            self.word_embeds = nn.Embedding(self.n_words, 100)
        logger('Load pretrained word embedding. Shape: {}'.format(self.embed_list.shape))

    def forward(self, word_ids):
        '''
        input:
            word_ids: (batch_size, max_sen_len)
        output:
            word_embeds: (batch_size, max_sen_len, emb_len)
        '''
        word_emb = self.word_embeds(word_ids)
        return word_emb