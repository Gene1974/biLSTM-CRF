import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import sys

from DataLoader import DataLoader

torch.manual_seed(1)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# return word_emb of each word in seq
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size) # output of whole sentence

        # Matrix of transition, P[i, j] is transitioning from tag j(y_i-1) to tag i(y_i)
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2).to(self.device),
                torch.randn(2, 1, self.hidden_dim // 2).to(self.device))

    # calculate log(sum(e^S))
    def _forward_alg(self, feats): # feats is output of lstm, indicates A
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(self.device) # tagset_size is number of tags
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

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
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        # only 
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]] # [1, tagset_size]
        alpha = log_sum_exp(terminal_var)
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
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(self.device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

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
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

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
# [([word1, word2, ..., word n], [tag1, tag2, ..., tag n])]
data_loader = DataLoader()
sentences = data_loader.load_conll_sentence('./train_30', scheme = "SIMPLE")

# encoding all the words in training data
word_to_ix = {}
for sentence, tags in sentences:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

#device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device = {}'.format(device))
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

for i in range(len(sentences)):
    sentences[i][0] = torch.tensor([word_to_ix[w] for w in sentences[i][0]]).to(device)
    sentences[i][1] = torch.tensor([tag_to_ix[t] for t in sentences[i][1]]).to(device)
training_data = sentences

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(30):  # again, normally you would NOT do 300 epochs, it is toy data
    total_loss = 0
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        #sentence_in = prepare_sequence(sentence, word_to_ix) # map sentence to word_index
        #targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        #loss = model.neg_log_likelihood(sentence_in, targets)
        loss = model.neg_log_likelihood(sentence, tags)
        total_loss += loss.item()

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

    # Check predictions after training
    with torch.no_grad():
        #precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        #print(model(precheck_sent))
        correct = 0
        total = 0
        for sentence, tags in training_data:
            score, predict = model(sentence)
            correct += torch.sum(torch.tensor(predict).to(device) == tags).item()
            total += sentence.shape[0]
        sys.stderr.write('[{}][epoch {:3d}] loss: {:.8f}  train_accu: {:.8f}\n'.format(
            time.strftime("%m-%d %H:%M:%S", time.localtime()), epoch + 1, total_loss / total, correct / total))
        if epoch % 20 == 19:
            sys.stdout.write('[{}][epoch {:3d}] loss: {:.8f}  train_accu: {:.8f}\n'.format(
                time.strftime("%m-%d %H:%M:%S", time.localtime()), epoch + 1, total_loss / total, correct / total))
    
# # detect
# sentences = data_loader.load_conll_sentence('./test_1000', scheme = "SIMPLE")
# for i in range(len(sentences)):
#     sentences[i][0] = torch.tensor([word_to_ix[w] for w in sentences[i][0]]).to(device)
#     sentences[i][1] = torch.tensor([tag_to_ix[t] for t in sentences[i][1]]).to(device)
# test_data = sentences

# with torch.no_grad():
#     correct = 0
#     total = 0
#     for sentence, tags in test_data:
#         score, predict = model(sentence)
#         correct += torch.sum(torch.tensor(predict).to(device) == tags).item()
#         total += sentence.shape[0]
#     sys.stdout.write('[{}][Test] loss: {:.8f}  train_accu: {:.8f}\n'.format(
#             time.strftime("%m-%d %H:%M:%S", time.localtime()), total_loss / total, correct / total))
