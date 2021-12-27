import torch
import torch.nn as nn
#from CharEmbedding import CharEmbedding
from Utils import logger

class WordEmbedding(nn.Module):
    def __init__(self, word_vocab, word_emb_dim = 256, use_pretrained_word = True, fine_tune = False):
        super().__init__()
        self.word_vocab = word_vocab
        self.n_words = len(word_vocab.word_to_ix)
        self.freeze = not fine_tune
        if use_pretrained_word:
            self.word_emb_dim = word_vocab.word_emb.shape[1]
            self.word_emb = word_vocab.word_emb
            self.word_embeds = nn.Embedding.from_pretrained(self.word_emb, freeze = self.freeze)
            logger('Load pretrained word embedding, freeze = {}. Shape: {}'.format(self.freeze, self.word_emb.shape))
        else:
            self.word_emb_dim = word_emb_dim
            #self.word_embeds = CharEmbedding(self.n_words, self.word_emb_dim, use_pretrained_char)
            self.char_embeds = nn.Embedding(self.n_chars, word_emb_dim)

    def forward(self, word_ids):
        '''
        input:
            word_ids: (batch_size, max_sen_len)
        output:
            word_embeds: (batch_size, max_sen_len, emb_len)
        '''
        word_emb = self.word_embeds(word_ids)
        return word_emb