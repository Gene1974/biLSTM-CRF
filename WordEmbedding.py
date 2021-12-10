import torch
import torch.nn as nn

class WordEmbedding(nn.Module):
    def __init__(self, word_vocab, use_pretrained):
        super().__init__()
        self.word_vocab = word_vocab
        self.word_emb_size = word_vocab.word_emb.shape[1]
        self.n_words = len(word_vocab.word_to_ix)
        if use_pretrained:
            self.word_emb = word_vocab.word_emb
            self.word_embeds = nn.Embedding.from_pretrained(self.word_emb)
        else:
            self.word_embeds = nn.Embedding(self.n_words, self.word_emb_size)

    def forward(self, word_ids):
        '''
        input:
            word_ids: (batch_size, max_sen_len)
        output:
            word_embeds: (batch_size, max_sen_len, emb_len)
        '''
        word_emb = self.word_embeds(word_ids)
        return word_emb