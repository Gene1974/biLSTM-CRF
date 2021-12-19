import torch
import torch.nn as nn
from CharEmbedding import CharEmbedding

class WordEmbedding(nn.Module):
    def __init__(self, word_vocab, use_pretrained = True, word_emb_dim = 100):
        super().__init__()
        self.word_vocab = word_vocab
        self.n_words = len(word_vocab.word_to_ix)
        if use_pretrained:
            self.word_emb_dim = word_vocab.word_emb.shape[1]
            self.word_emb = word_vocab.word_emb
            self.word_embeds = nn.Embedding.from_pretrained(self.word_emb)
        else:
            self.word_emb_dim = word_emb_dim
            self.word_embeds = CharEmbedding(self.n_words, self.word_emb_dim)

    def forward(self, word_ids):
        '''
        input:
            word_ids: (batch_size, max_sen_len)
        output:
            word_embeds: (batch_size, max_sen_len, emb_len)
        '''
        word_emb = self.word_embeds(word_ids)
        return word_emb