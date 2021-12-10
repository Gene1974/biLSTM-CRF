import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from crf import CRF
from CharEmbedding import CharEmbedding
from WordEmbedding import WordEmbedding
from Utils import *

class BiLSTM_CRF(nn.Module):
    def __init__(self, word_vocab,tag_vocab, 
            char_emb_dim, hidden_dim, num_layers, 
            batch_size, device, dropout = 0.5, 
            use_pretrained = True, use_char = True, use_crf = True, use_cnn = False):
        super(BiLSTM_CRF, self).__init__()
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab
        
        self.char_emb_dim = char_emb_dim
        self.word_emb_dim = word_vocab.word_emb.shape[1]
        if use_char:
            self.emb_dim = char_emb_dim + self.word_emb_dim
        else:
            self.emb_dim = self.word_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.tagset_size = len(tag_vocab.tag_to_ix)
        self.charset_size = len(word_vocab.char_to_ix)
        
        self.batch_size = batch_size
        self.device = device
        self.use_pretrained = use_pretrained
        self.use_char = use_char
        self.use_cnn = use_cnn
        self.use_crf = use_crf

        self.word_embeds = WordEmbedding(word_vocab, use_pretrained)
        self.char_embeds = CharEmbedding(self.charset_size, char_emb_dim, use_cnn = use_cnn)
        
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

    def forward(self, word_ids, word_mask, char_ids, char_mask, label = None): # (batch_size, sen_len)
        word_emb = self.word_embeds(word_ids) # (batch_size, sen_len, 100)
        if self.use_char:
            char_emb = self.char_embeds(char_ids) # (batch_size, sen_len, 30)
            embeds = torch.cat((word_emb, char_emb), dim = -1)
        else: 
            embeds = word_emb
        
        embeds = self.dropout1(embeds)
        sen_len = torch.sum(word_mask, dim = 1, dtype = torch.int64).to('cpu') # (batch_size)
        pack_seq = pack_padded_sequence(embeds, sen_len, batch_first = True, enforce_sorted = False)
        lstm_out, _ = self.lstm(pack_seq)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first = True) # (batch_size, seq_len, hidden_size)
        lstm_feats = self.hidden2tag(lstm_out) # （batch_size, seq_len, tagset_size)
        lstm_feats = self.dropout2(lstm_feats)
        
        if not self.use_crf:
            return lstm_feats
        else:
            if label is None:
                predict = self.crf.viterbi_tags(lstm_feats, word_mask)
                return predict
            else:
                log_likelihood = self.crf(lstm_feats, label, word_mask)
                batch_size = word_ids.shape[0]
                loss = -log_likelihood / batch_size
                return loss
