import configparser
import copy
import numpy as np
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class WordEmbedding(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.word_emb_size = 100
    
    def load_glove(self):
        # [word num1 num2 ... num100]
        path = '/home/gene/Documents/Data/Glove/glove.6B.100d.txt'
        word_list = []  # words
        word_emb = []   # embeddings, [tensor1, tensor2, ..., tensor n]
        word_index = {} # word: index
        idx = 0
        with open(path, 'r') as glove:
            for line in glove.readlines():
                data = line.strip().split(' ')
                word = data[0]
                embeds = [float(i) for i in data[1:]]
                word_list.append(word)
                word_emb.append(embeds)
                word_index[word] = idx
                idx += 1
        word_emb = torch.tensor(word_emb, dtype = torch.float) # [400000, 100]
        return word_emb, word_list, word_index

class biLSTM_CNN_CRF(nn.Module):
    def __init__(self,
                tag_to_idx,
                char_length,
                char_emb_dim,
                char_dropout, 
                cnn_channel, 
                cnn_kernel_size, 
                cnn_padding,
                lstm_hidden_size,
                lstm_layers,
                lstm_dropout
                ):
        super().__init__()

        self.tag_to_ix = tag_to_idx
        self.char_length = char_length
        self.tagset_size = len(tag_to_idx)
        self.START_TAG = '<START>'
        self.STOP_TAG = '<STOP>'
        #self.ner_map = {'O': 0, 'I-LOC': 1, 'I-MISC': 2, 'I-ORG': 3, 'I-PER': 4,
        #                        'B-LOC': 5, 'B-MISC': 6, 'B-ORG': 7, 'B-PER': 8}
        

        # word embedding
        self.word_emb, self.word_list, self.word_to_idx = WordEmbedding().load_glove()
        self.word_emb_dim = len(self.word_to_idx)
        
        self.word_emb_layer = nn.Embedding.from_pretrained(self.word_emb)
        self.char_emb_layer = nn.Sequential(
            nn.Embedding(num_embeddings = char_length,
                        embedding_dim = char_emb_dim
                        ),
            nn.Dropout(p = char_dropout),
            nn.Conv1d(in_channels = char_emb_dim, 
                    out_channels = cnn_channel, 
                    kernel_size = cnn_kernel_size, 
                    padding = cnn_padding
                    ),
            nn.MaxPool1d(kernel_size = 3)
        )
        self.lstm_layer = nn.Sequential(
            nn.Dropout(p = lstm_dropout),
            nn.LSTM(input_size = char_emb_dim + self.word_emb_dim,
                    hidden_size = lstm_hidden_size // 2,
                    num_layers = lstm_layers,
                    batch_first = True,
                    bidirectional = True
                    )
        )
        self.hidden2tag = nn.Linear(lstm_hidden_size, self.tagset_size),
        self.dropout = nn.Dropout(p = lstm_dropout)
        # self.crf_layer = self.CRF()
        self.transitions = torch.zeros((self.tag_size, self.tag_size))
    
    def _init_weight(self):
        nn.init.kaiming_uniform_(self.char_emb_layer.weights, mode='fan_out')
        nn.init.xavier_uniform_(self.cnn_layer.weights)
        nn.init.zeros_(self.cnn_layer.bias)
        for name, param in self.lstm_layer.named_parameters():
            if name.startswith('bias'): # b_i|b_f|b_g|b_o
                nn.init.zeros_(param)
                param.data[self.lstm_layer.hidden_size: 2 * self.lstm_layer.hidden_size] = 1
            else:
                nn.init.xavier_uniform_(param)
        self.transitions[self.ner_map[self.START_TAG], :] = -10000
        self.transitions[:, self.ner_map[self.STOP_TAG]] = -10000

    '''
    def _get_lstm_features(self, sentence): # forwarding
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1) # forward calculate embes
        lstm_out, self.hidden = self.lstm(embeds, self.hidden) # pass throuth lstm
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats'''
    
    def _get_lstm_features(self, sentence):
        word_emb = self.word_emb_layer(sentence)
        #char_emb = self.char_emb_layer(sentence)
        char_emb = self.char_emb_layer(sentence).view(len(sentence), 1, -1) # forward calculate embes
        emb = torch.cat((char_emb, word_emb), 1) # [batch_size, seq_len, feature]
        # emb_packed = pack_padded_sequence(emb, emb.shape[1], batch_first=True)
        lstm_out, self.hidden = self.lstm_layer(emb)
        #lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = lstm_out.view(len(sentence), self.lstm_hidden_size)
        lstm_feats = self.hidden2tag(lstm_out)
        lstm_feats = self.dropout(lstm_feats)
        return lstm_feats

    def argmax(self, vec):
        _, idx = torch.max(vec, dim = 1)
        return idx.item()

    def log_sum_exp(self, vec):
        max_score = vec[0, self.argmax(vec)]
        max_score_broadcast = max_score.view(-1, 1).expand(1, vec.size()[1])
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
    
    def prepare_sequence(self, seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    # calculate log(sum(e^S))
    def _forward_alg(self, feats): # feats is output of lstm, indicates A
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.) # tagset_size is number of tags
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[self.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        # feats is output of lstm, mapping whole sentence's word to tag
        # represent A[i, j], word i -> tag j
        for feat in feats: 
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score # add three scores
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        # only 
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        alpha = self.log_sum_exp(terminal_var)
        return alpha

    # calcu s(X, y)
    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[self.START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[self.STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[self.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = self.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        best_tag_id = self.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[self.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    # return loss = S(X, y) - log(\sum(exp(S(X, y))))
    def neg_log_likelihood(self, sentence, tags): # input, predict
        feats = self._get_lstm_features(sentence) # forwarding, get A
        forward_score = self._forward_alg(feats) # get log(\sum(exp(S(X, y))))
        gold_score = self._score_sentence(feats, tags) # get S(X, y)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence) # forwarding

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats) # predict
        return score, tag_seq




        
if __name__ == '__main__':
    trainer = Trainer()
