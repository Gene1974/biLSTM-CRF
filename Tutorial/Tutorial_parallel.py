import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import sys

from DataLoader import DataLoader
from pytorchtools import EarlyStopping

torch.manual_seed(1)
#os.environ["CUDA_VISIBLE_DEVICES"] = '4'

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, device):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.device = device
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Transition matrix, P[i, j]: tag j(y_i-1) -> tag i(y_i)
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
            
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2).to(self.device),
                torch.randn(2, 1, self.hidden_dim // 2).to(self.device))

    # calculate log(sum(e^S))
    def _forward_alg(self, feats): # feats(A) is output of lstm, word i -> tag j, [input_size, tagset_size]
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((self.tagset_size, ), -10000.).to(self.device) # tagset_size is number of tags
        init_alphas[self.tag_to_ix[START_TAG]] = 0. # START_TAG has all of the score.

        # Wrap in a variable so that we will get automatic backprop
        forward_var_list = []
        forward_var_list.append(init_alphas)

        # Iterate through the sentence
        for i in range(feats.shape[0]): 
            forward_var = torch.stack([forward_var_list[i]] * self.tagset_size)
            #emit_score = torch.stack([feats[i]] * self.tagset_size).transpose(0, 1)
            emit_score = torch.unsqueeze(feats[i], dim = 0).transpose(0, 1)
            trans_score = self.transitions
            next_tag_var = forward_var + emit_score + trans_score
            forward_var_list.append(torch.logsumexp(next_tag_var, dim = 1))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]] # last word
        alpha = torch.logsumexp(torch.unsqueeze(terminal_var, dim = 0), dim = 1)
        return alpha

    def _get_lstm_features(self, sentence): # forwarding
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1) # forward calculate embes
        lstm_out, self.hidden = self.lstm(embeds, self.hidden) # pass throuth lstm
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    # calcu s(X, y)
    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(self.device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(self.device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((self.tagset_size, ), -10000.).to(self.device)
        init_vvars[self.tag_to_ix[START_TAG]] = 0

        
        forward_var_list = [init_vvars]
        for i in range(feats.shape[0]):
            # forward_var at step i holds the viterbi variables for step i-1
            forward_var = torch.stack([forward_var_list[i]] * self.tagset_size)
            # next_tag_var[i] holds the viterbi variable for tag i at the previous step
            next_tag_var = forward_var + self.transitions
            viterbivars_t, best_tag_id = torch.max(next_tag_var, dim = 1)
            bptrs_t = best_tag_id.tolist()
            # new forward_var
            forward_var = viterbivars_t + feats[i] # [tagset_size], add the emission scores
            forward_var_list.append(forward_var)
            backpointers.append(best_tag_id.tolist())

        # Transition to STOP_TAG
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = torch.argmax(terminal_var).tolist()
        path_score = terminal_var[best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
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




START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5 # tag 共有 B, I, O, <START>, <STOP> 五个，因此 embedding_dim = 5
HIDDEN_DIM = 4 # biLSTM隐藏层特征数量为2，双向为4

# Make up some training data
# [[[word1, word2, ..., word n], [tag1, tag2, ..., tag n]]]
data_loader = DataLoader()
# print('[{}] Loading data...'.format(time.strftime("%m-%d %H:%M:%S", time.localtime())))
# train_data = data_loader.load_conll_sentence('./train_30', scheme = "SIMPLE")
# valid_data = data_loader.load_conll_sentence('./train_30', scheme = "SIMPLE")
# test_data = data_loader.load_conll_sentence('./train_30', scheme = "SIMPLE")

#data_path = '/data/hyz/CoNLL2003/'
data_path = '/home/gene/Documents/Data/CoNLL2003/'
print('[{}] Loading data...'.format(time.strftime("%m-%d %H:%M:%S", time.localtime())))
train_data = data_loader.load_conll_sentence(data_path + 'eng.train', scheme = "SIMPLE")
test_data = data_loader.load_conll_sentence(data_path + 'eng.testb', scheme = "SIMPLE")
valid_data = data_loader.load_conll_sentence(data_path + 'eng.testa', scheme = "SIMPLE")
sentences = train_data + test_data + valid_data

# encoding all the words in training data
word_to_ix = {}
for sentence, tags in sentences:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device = {}'.format(device))
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, device).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

for i in range(len(train_data)):
    train_data[i][0] = torch.tensor([word_to_ix[w] for w in train_data[i][0]]).to(device)
    train_data[i][1] = torch.tensor([tag_to_ix[t] for t in train_data[i][1]]).to(device)

for i in range(len(test_data)):
    test_data[i][0] = torch.tensor([word_to_ix[w] for w in test_data[i][0]]).to(device)
    test_data[i][1] = torch.tensor([tag_to_ix[t] for t in test_data[i][1]]).to(device)

for i in range(len(valid_data)):
    valid_data[i][0] = torch.tensor([word_to_ix[w] for w in valid_data[i][0]]).to(device)
    valid_data[i][1] = torch.tensor([tag_to_ix[t] for t in valid_data[i][1]]).to(device)

early_stopping = EarlyStopping(patience = 7, verbose = False)
avg_train_losses = []
avg_valid_losses = []
for epoch in range(1000):
    train_losses = []
    valid_losses = []
    model.train()
    for sentence, tags in train_data:
        model.zero_grad()
        loss = model.neg_log_likelihood(sentence, tags)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        for sentence, tags in valid_data:
            valid_losses.append(model.neg_log_likelihood(sentence, tags).item())
        avg_train_loss = np.average(train_losses)
        avg_valid_loss = np.average(valid_losses)
        avg_train_losses.append(avg_train_loss)
        avg_valid_losses.append(avg_valid_loss)
        sys.stderr.write('[{}][epoch {:3d}] train_loss: {:.8f}  valid_loss: {:.8f}\n'.format(
            time.strftime("%m-%d %H:%M:%S", time.localtime()), epoch + 1, avg_train_loss, avg_valid_loss))
        sys.stdout.write('[{}][epoch {:3d}] train_loss: {:.8f}  valid_loss: {:.8f}\n'.format(
           time.strftime("%m-%d %H:%M:%S", time.localtime()), epoch + 1, avg_train_loss, avg_valid_loss))
        early_stopping(avg_valid_loss, model)
        if early_stopping.early_stop:
            print('Early stopping')
            sys.stderr.write('Early stopping\n')
            break

#torch.save(model.state_dict(), './model/biLSTM_CRF_test_{}'.format(time.strftime('%m%d_%H%M%S', time.localtime())))
torch.save(model.state_dict(), './model/biLSTM_CRF_{}'.format(time.strftime('%m%d_%H%M%S', time.localtime())))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(avg_train_losses)
plt.plot(avg_valid_losses)
plt.legend(['train_loss', 'valid_loss'])
plt.savefig('./loss/biLSTM_CRF_{}.png'.format(time.strftime('%m%d_%H%M%S', time.localtime())), format = 'png')
    
# test
model.eval()
test_losses = []
with torch.no_grad():
    correct = 0
    total = 0
    for sentence, tags in test_data:
        score, predict = model(sentence)
        test_losses.append(model.neg_log_likelihood(sentence, tags).item())
        correct += torch.sum(torch.tensor(predict).to(device) == tags).item()
        total += sentence.shape[0]
    avg_test_loss = np.average(test_losses)
    sys.stderr.write('[{}][Test] loss: {:.8f}  test_accu: {:.8f}\n'.format(
            time.strftime("%m-%d %H:%M:%S", time.localtime()), avg_test_loss, correct / total))
    sys.stdout.write('[{}][Test] loss: {:.8f}  test_accu: {:.8f}\n'.format(
            time.strftime("%m-%d %H:%M:%S", time.localtime()), avg_test_loss, correct / total))


# reference:
# https://github.com/mali19064/LSTM-CRF-pytorch-faster/blob/master/LSTM_CRF_faster.py
# https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
# https://blog.csdn.net/qq_39526294/article/details/104055944