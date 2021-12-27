import sys
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from BiLSTM import BiLSTM

from crf import CRF
from CharEmbedding import CharEmbedding
from WordEmbedding import WordEmbedding
from LMEmbedding import LMEmbedding
from LexiconEmbedding import LexiconEmbedding
from Utils import logger

class BiLSTM_CRF(nn.Module):
    def __init__(self, word_vocab, tag_vocab, 
            char_emb_dim, word_emb_dim, lm_emb_dim, lexicon_emb_dim, emb_dim, hidden_dim, num_layers, 
            batch_size, device, dropout = 0.5, 
            use_word = True, use_char = True, use_lm = True, use_crf = True, use_cnn = True, use_lexicon = True,
            use_pretrained_word = True, use_pretrained_char = True, 
            attention_pooling = False):
        super().__init__()
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
        self.fine_tune_word = True
        self.fine_tune_char = True
        self.use_word = use_word
        self.use_char = use_char
        self.use_lm = use_lm
        self.use_lexicon = use_lexicon
        self.use_cnn = use_cnn
        self.use_crf = use_crf
        self.attention_pooling = attention_pooling
        
        # self.word_emb_dim = word_emb_dim
        # self.char_emb_dim = char_emb_dim
        self.lm_emb_dim = lm_emb_dim
        self.lexicon_emb_dim = lexicon_emb_dim
        self.emb_dim = emb_dim # lstm input dim
        self.raw_emb_dim = 0 # emb_dim after concat
        if use_word:
            if use_pretrained_word:
                self.word_emb_dim = word_vocab.word_emb.shape[1]
            else:
                self.word_emb_dim = char_emb_dim
            self.word_embeds = WordEmbedding(word_vocab, self.word_emb_dim, use_pretrained_word, self.fine_tune_word)
            self.raw_emb_dim += self.word_emb_dim
        if use_char:
            self.char_embeds = CharEmbedding(word_vocab.char_to_ix, char_emb_dim, use_pretrained_char, self.fine_tune_char, use_cnn, attention_pooling, dropout = 0.5)
            self.char_emb_dim = self.char_embeds.get_emb_dim()
            self.raw_emb_dim += self.char_emb_dim 
        if use_lm:
            self.lm_embeds = LMEmbedding(lm_emb_dim)
            self.raw_emb_dim += self.lm_emb_dim
        if use_lexicon:
            self.lexicon_embeds = LexiconEmbedding(lexicon_emb_dim, self.tag_vocab)
            self.raw_emb_dim += self.lexicon_emb_dim
            
        # if self.raw_emb_dim >= 512:
        #     self.embed_dense = nn.Linear(self.raw_emb_dim, self.emb_dim)
        # else:
        #     self.emb_dim = self.raw_emb_dim
        self.emb_dim = self.raw_emb_dim
        
        self.bilstm = BiLSTM(
            word_vocab, tag_vocab, self.emb_dim, self.hidden_dim, num_layers, dropout
            )
        if use_crf:
            tag_dict = {value: key for key, value in self.tag_vocab.tag_to_ix.items()}
            self.crf = CRF(tag_dict)

    
    def forward(self, text, word_ids, word_mask, char_ids, char_mask, label = None): # (batch_size, sen_len)
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
        embeds = None
        if self.use_word:
            word_emb = self.word_embeds(word_ids) # (batch_size, sen_len, emb_size)
            embeds = word_emb
        
        if self.use_char:
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
        
        if self.use_lexicon:
            lexicon_embeds = self.lexicon_embeds(text) # (batch_size, sen_len, 256)
            if embeds is None:
                embeds = lexicon_embeds
            else:
                embeds = torch.cat((embeds, lexicon_embeds), dim = -1)
            
        # if self.raw_emb_dim >= 512:
        #     embeds = self.embed_dense(embeds) # (batch_size, sen_len, 256)
        
        if self.use_char and char_ids.dim() == 2: # no word embedding
            word_mask = char_mask
        
        lstm_out = self.bilstm(embeds, word_mask, label)
        
        if not self.use_crf:
            return lstm_out
        else:
            if label is None:
                predict = self.crf.viterbi_tags(lstm_out, word_mask)
                return predict
            else:
                log_likelihood = self.crf(lstm_out, label, word_mask)
                batch_size = label.shape[0]
                loss = -log_likelihood / batch_size
                return loss
