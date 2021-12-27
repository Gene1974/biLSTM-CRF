import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from crf import CRF
from Utils import logger

class BiLSTM(nn.Module):
    def __init__(self, vocab, tag_vocab, 
            emb_dim, hidden_dim, num_layers, dropout = 0.5
            ):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vocab = vocab
        self.tag_vocab = tag_vocab

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.tagset_size = len(tag_vocab.tag_to_ix)
        self.charset_size = len(vocab.char_to_ix)
        
        self.dropout1 = nn.Dropout(p = dropout)
        self.dropout2 = nn.Dropout(p = dropout)
        self.lstm = nn.LSTM(self.emb_dim, hidden_dim // 2,
                            num_layers = num_layers, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

    def forward(self, embeds, word_mask, label = None): # (batch_size, sen_len)
        '''
        input:
            embeds:     (batch_size, sen_len, emb_size)
            word_mask:  (batch_size, sen_len)
            label:      (batch_size, sen_len)
        output:
            crf, predict: (label is None)
                predict:    (batch_size, sen_len)
            crf, train: (label is Not None)
                loss
            no-crf:
                lstm_feats: (batch_size, sen_len, tagset_size)
        '''
        embeds = self.dropout1(embeds) # (batch_size, sen_len, 256)
        sen_len = torch.sum(word_mask, dim = 1, dtype = torch.int64).to('cpu') # (batch_size)
        pack_seq = pack_padded_sequence(embeds, sen_len, batch_first = True, enforce_sorted = False)
        lstm_out, _ = self.lstm(pack_seq)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first = True) # (batch_size, seq_len, hidden_size)
        lstm_feats = self.hidden2tag(lstm_out) # （batch_size, seq_len, tagset_size)
        lstm_feats = self.dropout2(lstm_feats)

        return lstm_feats
