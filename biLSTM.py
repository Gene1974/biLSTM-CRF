import sys
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from crf import CRF
from CharEmbedding import CharEmbedding
from WordEmbedding import WordEmbedding
from LMEmbedding import LMEmbedding
from Utils import *

class BiLSTM_CRF(nn.Module):
    def __init__(self, word_vocab, tag_vocab, 
            char_emb_dim, word_emb_dim, lm_emb_dim, emb_dim, hidden_dim, num_layers, 
            batch_size, device, dropout = 0.5, 
            use_word = True, use_char = True, use_lm = True, use_crf = True, use_cnn = True,
            use_pretrained_word = True, use_pretrained_char = True, 
            attention_pooling = False):
        super(BiLSTM_CRF, self).__init__()
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.tagset_size = len(tag_vocab.tag_to_ix)
        self.charset_size = len(word_vocab.char_to_ix)
        
        self.batch_size = batch_size
        self.device = device
        self.use_pretrained_word = use_pretrained_word
        self.use_pretrained_char = use_pretrained_char
        self.use_word = use_word
        self.use_char = use_char
        self.use_cnn = use_cnn
        self.use_crf = use_crf
        self.use_lm = use_lm
        self.attention_pooling = attention_pooling
        
        # self.word_emb_dim = word_emb_dim
        # self.char_emb_dim = char_emb_dim
        self.lm_emb_dim = lm_emb_dim
        self.emb_dim = emb_dim # lstm input dim
        self.raw_emb_dim = 0 # emb_dim after concat
        if use_word:
            if use_pretrained_word:
                self.word_emb_dim = word_vocab.word_emb.shape[1]
            else:
                self.word_emb_dim = char_emb_dim
            self.word_embeds = WordEmbedding(word_vocab, self.word_emb_dim, use_pretrained_word, use_pretrained_char)
            self.raw_emb_dim += self.word_emb_dim
        if use_char:
            self.char_embeds = CharEmbedding(word_vocab.char_to_ix, char_emb_dim, use_pretrained_char, use_cnn, attention_pooling, dropout = 0.5)
            self.char_emb_dim = self.char_embeds.get_emb_dim()
            self.raw_emb_dim += self.char_emb_dim 
        
        if use_lm:
            self.lm_embeds = LMEmbedding(lm_emb_dim)
            self.raw_emb_dim += self.lm_emb_dim
            
        #self.embed_dense = nn.Linear(self.raw_emb_dim, self.emb_dim)
        self.emb_dim = self.raw_emb_dim
        
        self.dropout1 = nn.Dropout(p = dropout)
        self.dropout2 = nn.Dropout(p = dropout)
        self.lstm = nn.LSTM(self.emb_dim, hidden_dim // 2,
                            num_layers = num_layers, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        if use_crf:
            self.crf = CRF(self.tag_vocab.ix_to_tag)
        
    def _init_weight(self):
        for name, param in self.lstm.named_parameters():
            if name.startswith('bias'): # b_i|b_f|b_g|b_o
                nn.init.zeros_(param)
                param.data[self.lstm.hidden_size: 2 * self.lstm.hidden_size] = 1
            else:
                nn.init.xavier_uniform_(param)

    '''
    input:
        dim == 3: (English NER)
            text:   list(list)
            word_ids:   (batch_size, sen_len)
            word_mask:  (batch_size, sen_len)
            char_ids:   (batch_size, sen_len, max_word_len)
            label:      (batch_size, sen_len)
        dim == 2: (Chinese NER)
            text:   list(list)
            char_ids:   (batch_size, sen_len)
            char_mask:  (batch_size, sen_len)
    '''
    def forward(self, text, word_ids, word_mask, char_ids, char_mask, label = None): # (batch_size, sen_len)
        embeds = None
        if self.use_word:
            word_emb = self.word_embeds(word_ids) # (batch_size, sen_len, emb_size)
            embeds = word_emb
        
        if self.use_char:
            '''
            char_emb: (batch_size, sen_len, emb_size)
            '''
            char_emb = self.char_embeds(char_ids) # (batch_size, sen_len, emb_size)
            if embeds is None:
                embeds = char_emb
            else:
                embeds = torch.cat((embeds, char_emb), dim = -1)
        
        if self.use_lm:
            lm_embeds = self.lm_embeds(text) # (batch_size, sen_len, 256)
            if embeds is None:
                embeds = lm_embeds
            else:
                embeds = torch.cat((embeds, lm_embeds), dim = -1)
            #embeds = self.embed_dense(embeds) # (batch_size, sen_len, 256)
        
        embeds = self.dropout1(embeds) # (batch_size, sen_len, 256)
        if self.use_char and char_ids.dim() == 2: # no word embedding
            word_mask = char_mask
        
        sen_len = torch.sum(word_mask, dim = 1, dtype = torch.int64).to('cpu') # (batch_size)
        pack_seq = pack_padded_sequence(embeds, sen_len, batch_first = True, enforce_sorted = False)
        lstm_out, _ = self.lstm(pack_seq)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first = True) # (batch_size, seq_len, hidden_size)
        lstm_feats = self.hidden2tag(lstm_out) # （batch_size, seq_len, tagset_size) d
        
        if not self.use_crf:
            if label is not None:
                lstm_feats = self.dropout2(lstm_feats)
            return lstm_feats
        else:
            if label is None:
                predict = self.crf.viterbi_tags(lstm_feats, word_mask)
                return predict
            else:
                lstm_feats = self.dropout2(lstm_feats)
                log_likelihood = self.crf(lstm_feats, label, word_mask)
                batch_size = label.shape[0]
                loss = -log_likelihood / batch_size
                return loss
