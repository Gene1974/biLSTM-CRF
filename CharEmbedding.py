import numpy as np
import torch
from torch._C import dtype
import torch.nn as nn
from gensim.models import Word2Vec
from AttentionPooling import AttentionPooling
from Utils import logger


class CharEmbedding(nn.Module):
    def __init__(self, char_to_ix, char_emb_dim, use_pretrained_char = False, fine_tune = False, use_cnn = True, attention_pooling = False, dropout = 0.5):
        super().__init__()
        path = '/data/word_embedding/cnchar.model'
        self.n_chars = len(char_to_ix)
        self.char_emb_dim = char_emb_dim
        self.use_pretrained_char = use_pretrained_char
        self.use_cnn = use_cnn
        self.attention_pooling = attention_pooling

        if use_pretrained_char:
            self.freeze = not fine_tune
            self.embeddings = self.init_char_embedding(path, char_to_ix)
            self.embeddings = torch.tensor(self.embeddings, dtype = torch.float)
            self.char_embeds = nn.Embedding.from_pretrained(self.embeddings, freeze = self.freeze) # fine-tune
            self.char_emb_dim = self.embeddings.shape[1]
            logger('pretrained char path: {}'.format(path))
            logger('Load pretrained char embedding. Shape: {}'.format(self.embeddings.shape))
            if attention_pooling:
                self.atten_pool = AttentionPooling(self.embeddings.shape[1], char_emb_dim)
        else:
            self.char_embeds = nn.Embedding(self.n_chars, char_emb_dim)
            if use_cnn:
                self.dropout = nn.Dropout(p = dropout)
                self.cnn = nn.Conv1d(in_channels = char_emb_dim, out_channels = char_emb_dim, kernel_size = 3, padding = 1)
            if attention_pooling:
                self.atten_pool = AttentionPooling(char_emb_dim, char_emb_dim)

    def init_char_embedding(self, path, vocab_to_ix):
        word2index = Word2Vec.load(path) # Word2Vec(vocab=20022, vector_size=200, alpha=0.025)
        matrix=np.random.normal(size=(len(vocab_to_ix)+1,200))
        #pretrained = 0

        for word in vocab_to_ix.keys():
            index=vocab_to_ix[word]
            if word in word2index.wv:
                matrix[index,:]=word2index.wv[word]
                #pretrained += 1
        #print(pretrained)
        return matrix

    def get_emb_dim(self):
        return self.char_emb_dim
        
    def _init(self):
        if not self.use_pretrained_char:
            nn.init.kaiming_uniform_(self.char_embeds, mode = 'fan_out')
            if self.use_cnn:
                nn.init.xavier_uniform_(self.cnn.weight)
                nn.init.zeros_(self.cnn.bias)

    def forward(self, char_ids):
        '''
        input:
            char_ids: (batch_size, max_sen_len, max_word_len), dim = 3
                      (batch_size, max_sen_len), dim = 2
        output:
            char_embeds: (batch_size, max_sen_len, embed_size)
        '''
        if self.use_pretrained_char:
            char_emb = self.char_embeds(char_ids) # (batch_size, max_sen_len, embed_size)
            return char_emb
        
        dim = char_ids.dim()
        if dim == 2:
            char_ids = torch.unsqueeze(char_ids, dim = 2) # (batch_size, max_sen_len, 1), max_word_len = 1
        
        batch_size, max_sen_len, max_word_len = char_ids.shape # (batch_size, max_sen_len, max_word_len)
        emb_size = self.char_emb_dim
        char_emb = self.char_embeds(char_ids) # (batch_size, max_sen_len, max_word_len, embed_size)
        if self.use_cnn:
            char_emb = char_emb.reshape(batch_size * max_sen_len, max_word_len, emb_size)
            char_emb = char_emb.permute(0, 2, 1) # (batch_size * max_sen_len, embed_size, max_word_len)
            char_emb = self.dropout(char_emb)
            char_emb = self.cnn(char_emb) # (batch_size * max_sen_len, embed_size, max_word_len)
            char_emb = char_emb.reshape(batch_size, max_sen_len, -1, emb_size) # (batch_size, max_sen_len, max_word_len, embed_size)
        
        if dim == 2: # 二维情况下不用 pooling
            char_emb = torch.squeeze(char_emb, dim = 2) # (batch_size, max_sen_len, embed_size)
        elif self.attention_pooling:
            char_emb = self.atten_pool(char_emb)
        else:
            char_emb = torch.max(char_emb, dim = 2).values # (batch_size, max_sen_len, embed_size)
        return char_emb
    
    def __len__(self):
        return self.embeddings.shape
