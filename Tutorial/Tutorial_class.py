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

        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Transition matrix, P[i, j]: tag j(y_i-1) -> tag i(y_i)
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
            
        self.transitions.data[tag_to_ix[self.START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[self.START_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2).to(self.device),
                torch.randn(2, 1, self.hidden_dim // 2).to(self.device))

    # def is_transition_allowed(self, from_tag, from_entity, to_tag, to_entity):
    #     '''
    #     transition rules of BIOES tagging scheme
    #     from_tag & to_tag: string, ['B', 'I', 'O', 'E', 'S']
    #     from_entity & to_entity: string, ['PER', 'LOC', 'ORG', 'REL']s
    #     '''
    #     if from_tag == "START":
    #         return to_tag in ['O', 'B', 'S']
    #     if to_tag == "END":
    #         return from_tag in ['O', 'E', 'S']
    #     if from_tag in ['O', 'E', 'S'] and to_tag in ["O", "B", "S"]:
    #         return True
    #     elif from_tag in ['B', 'I'] and to_tag in ['I', 'E'] and from_entity == to_entity:
    #         return True
    #     else:
    #         return False

    # calculate log(sum(e^S))
    def _forward_alg(self, feats): # feats(A) is output of lstm, word i -> tag j, [input_size, tagset_size]
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((self.tagset_size, ), -10000.).to(self.device) # tagset_size is number of tags
        init_alphas[self.tag_to_ix[self.START_TAG]] = 0. # START_TAG has all of the score.

        # Wrap in a variable so that we will get automatic backprop
        forward_var_list = []
        forward_var_list.append(init_alphas)

        # Iterate through the sentence
        for i in range(feats.shape[0]): 
            forward_var = torch.stack([forward_var_list[i]] * self.tagset_size)
            emit_score = torch.unsqueeze(feats[i], dim = 0).transpose(0, 1)
            trans_score = self.transitions
            next_tag_var = forward_var + emit_score + trans_score
            forward_var_list.append(torch.logsumexp(next_tag_var, dim = 1))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[self.STOP_TAG]] # last word
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
        tags = torch.cat([torch.tensor([self.tag_to_ix[self.START_TAG]], dtype=torch.long).to(self.device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[self.STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((self.tagset_size, ), -10000.).to(self.device)
        init_vvars[self.tag_to_ix[self.START_TAG]] = 0

        
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
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        best_tag_id = torch.argmax(terminal_var).tolist()
        path_score = terminal_var[best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        #assert start == self.tag_to_ix[self.START_TAG]  # Sanity check
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


class Trainer():
    def __init__(self, epochs = 100, type = 'test', scheme = 'SIMPLE') -> None:
        self.scheme = scheme
        self.epochs = epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('device = {}'.format(self.device))

        self.load_data()

        self.embedding_dim = len(self.tag_to_ix) # embedding_dim = tagset_size
        self.hidden_dim = 4 # biLSTM隐藏层特征数量为2，双向为4
        
    def load_data(self):
        # Make up some training data
        # [[[word1, word2, ..., word n], [tag1, tag2, ..., tag n]]]
        data_loader = DataLoader()
        print('[{}] Loading data...'.format(time.strftime("%m-%d %H:%M:%S", time.localtime())))

        #data_path = '/data/hyz/CoNLL2003/'
        data_path = '/home/gene/Documents/Data/CoNLL2003/'
        self.train_data = data_loader.load_conll_sentence(data_path + 'eng.train', scheme = "IOBES")
        self.test_data = data_loader.load_conll_sentence(data_path + 'eng.testb', scheme = "IOBES")
        self.valid_data = data_loader.load_conll_sentence(data_path + 'eng.testa', scheme = "IOBES")

        # self.train_data = data_loader.load_conll_sentence('./train_30', scheme = "IOBES")
        # self.valid_data = data_loader.load_conll_sentence('./train_30', scheme = "IOBES")
        # self.test_data = data_loader.load_conll_sentence('./train_30', scheme = "IOBES")

        self.encode()
        self.train_data = [[torch.tensor([self.word_to_ix[w] for w in sentence[0]]).to(self.device),  
                            torch.tensor([self.tag_to_ix[t] for t in sentence[1]]).to(self.device)] for sentence in self.train_data]
        self.test_data = [[torch.tensor([self.word_to_ix[w] for w in sentence[0]]).to(self.device),  
                            torch.tensor([self.tag_to_ix[t] for t in sentence[1]]).to(self.device)] for sentence in self.test_data]
        self.valid_data = [[torch.tensor([self.word_to_ix[w] for w in sentence[0]]).to(self.device),  
                            torch.tensor([self.tag_to_ix[t] for t in sentence[1]]).to(self.device)] for sentence in self.valid_data]
    
    def encode(self):
        sentences = self.train_data + self.test_data + self.valid_data
        word_to_ix = {}
        for sentence, tags in sentences:
            for word in sentence:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        self.tag_to_ix = {'I-LOC': 0, 'I-MISC': 1, 'I-ORG': 2, 'I-PER': 3,
                          'B-LOC': 4, 'B-MISC': 5, 'B-ORG': 6, 'B-PER': 7,
                          'E-LOC': 8, 'E-MISC': 9, 'E-ORG': 10, 'E-PER': 11,
                          'S-LOC': 12, 'S-MISC': 13, 'S-ORG': 14, 'S-PER': 15,
                          'O': 16, '<START>': 17, '<STOP>': 18}
        self.START_TAG = '<START>'
        self.STOP_TAG = '<STOP>'
        self.entity_list = list(self.tag_to_ix.keys())
        self.word_to_ix = word_to_ix


    def train(self):
        model = BiLSTM_CRF(len(self.word_to_ix), self.tag_to_ix, self.embedding_dim, self.hidden_dim, self.device).to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
        early_stopping = EarlyStopping(patience = 7, verbose = False)

        avg_train_losses = []
        avg_valid_losses = []
        for epoch in range(self.epochs):
            train_losses = []
            valid_losses = []
            model.train()
            for sentence, tags in self.train_data:
                model.zero_grad()
                loss = model.neg_log_likelihood(sentence, tags)
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                for sentence, tags in self.valid_data:
                    valid_losses.append(model.neg_log_likelihood(sentence, tags).item())
                avg_train_loss = np.average(train_losses)
                avg_valid_loss = np.average(valid_losses)
                avg_train_losses.append(avg_train_loss)
                avg_valid_losses.append(avg_valid_loss)
                sys.stderr.write('[{}][epoch {:3d}] train_loss: {:.8f}  valid_loss: {:.8f}\n'.format(
                    time.strftime("%m-%d %H:%M:%S", time.localtime()), epoch + 1, avg_train_loss, avg_valid_loss))
                #sys.stdout.write('[{}][epoch {:3d}] train_loss: {:.8f}  valid_loss: {:.8f}\n'.format(
                #    time.strftime("%m-%d %H:%M:%S", time.localtime()), epoch + 1, avg_train_loss, avg_valid_loss))
                early_stopping(avg_valid_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        torch.save(model.state_dict(), './model/biLSTM_CRF_test_{}'.format(time.strftime('%m%d_%H%M%S', time.localtime())))
        self.model = model
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(avg_train_losses)
        plt.plot(avg_valid_losses)
        plt.legend(['train_loss', 'valid_loss'])
        plt.savefig('./loss/biLSTM_CRF_test_{}.png'.format(time.strftime('%m%d_%H%M%S', time.localtime())), format = 'png')

        self.test(model)
            
    def test(self, model):
        if not model:
            model = self.model
        model.eval()
        test_losses = []
        gold_num, predict_num, correct_num = 0, 0, 0
        correct = 0
        total = 0
        with torch.no_grad():
            for sentence, tags in self.test_data:
                score, predict = model(sentence)
                # predict = torch.tensor(predict).to(self.device)
                loss = model.neg_log_likelihood(sentence, tags).item()
                test_losses.append(loss)

                correct += torch.sum(torch.tensor(predict).to(self.device) == tags).item()
                total += sentence.shape[0]

                gold_entity = self.label_sentence_entity(tags.tolist())
                pred_entity = self.label_sentence_entity(predict)
                gold_num += len(gold_entity)
                predict_num += len(pred_entity)
                for entity in gold_entity:
                    if entity in pred_entity:
                        correct_num += 1
            avg_test_loss = np.average(test_losses)
            precision = correct_num / predict_num
            recall = correct_num / gold_num
            f1 = 2 * precision * recall / (precision + recall + 0.000000001)
            sys.stdout.write('[{}][Test] loss: {:.8f} Accuracy: {:.8f}\n'.format(
                    time.strftime("%m-%d %H:%M:%S", time.localtime()), avg_test_loss, correct / total))
            sys.stdout.write('[{}][Test] Precision: {:.8f} Recall: {:.8f} F1: {:.8f}\n'.format(
                    time.strftime("%m-%d %H:%M:%S", time.localtime()), precision, recall, f1))
            sys.stderr.write('[{}][Test] loss: {:.8f} Accuracy: {:.8f}\n'.format(
                    time.strftime("%m-%d %H:%M:%S", time.localtime()), avg_test_loss, correct / total))
            sys.stderr.write('[{}][Test] Precision: {:.8f} Recall: {:.8f} F1: {:.8f}\n'.format(
                    time.strftime("%m-%d %H:%M:%S", time.localtime()), precision, recall, f1))

    def label_sentence_entity(self, tags, scheme="BIOES"):
        if scheme.upper() != "BIOES":
            return
        if type(tags) == torch.Tensor:
            tags = tags.tolist()
        tags = [self.entity_list[tag] for tag in tags]
        entity = []
        count = len(tags)
        i = 0
        while i < count:
            if tags[i].startswith("B-"):
                j = i + 1
                while j < count:
                    if tags[j].startswith("E-"):
                        break
                    else:
                        j += 1
                entity.append({
                    "start_index": i,
                    "label": tags[i][2:]
                })
                i = j + 1
            elif tags[i].startswith("S-"):
                entity.append({
                    "start_index": i,
                    "label": tags[i][2:]
                })
                i += 1
            else:
                i += 1
        return entity

if __name__ == '__main__':
    sys.stdout.write('model: BiLSTM_CRF WordEmb: random\n')
    sys.stderr.write('model: BiLSTM_CRF WordEmb: random\n')
    trainer = Trainer(epochs = 300)
    trainer.train()

# reference:
# https://github.com/mali19064/LSTM-CRF-pytorch-faster/blob/master/LSTM_CRF_faster.py
# https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
# https://blog.csdn.net/qq_39526294/article/details/104055944