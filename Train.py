import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from biLSTM import BiLSTM_CRF
from Datas import ConllDataset, collate_fn_conll
from pytorchtools import EarlyStopping
from Utils import *

torch.manual_seed(1)
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class Trainer():
    def __init__(self, data_path, epochs = 100, use_pretrained = True, use_char = True, use_crf = True, use_cnn = True, scheme = 'BIOES'):
        super().__init__()

        self.scheme = scheme
        self.char_emb_dim = 30
        self.word_emb_dim = 100
        self.hidden_dim = 256
        self.lstm_layers = 1
        self.dropout = 0.1
        self.epochs = epochs
        self.batch_size = 32
        self.use_pretrained = use_pretrained
        self.use_char = use_char
        self.use_cnn = use_cnn
        self.use_crf = use_crf
        self.lr = 0.0001
        self.momentum = 0.9
        self.decay_rate = 0.05
        self.gradient_clip = 5.0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger('device = {}'.format(self.device))
        logger('use_pretrained = {}, use_char = {}, use_crf = {}, use_cnn = {}'.format(use_pretrained, use_char, use_crf, use_cnn))

        self.load_data(data_path)

        self.model = BiLSTM_CRF(
            self.word_vocab, self.char_vocab, self.tag_vocab, 
            self.char_emb_dim, self.word_emb_dim, self.hidden_dim, self.lstm_layers, 
            self.batch_size, self.device, self.dropout, 
            use_pretrained = use_pretrained, use_char = use_char, use_crf = use_crf, use_cnn = use_cnn
        ).to(self.device)

    def load_data(self, data_path):
        train_set = ConllDataset(data_path + 'eng.train', scheme = 'BIOES', make_map = True)
        self.word_vocab, self.char_vocab, self.tag_vocab = train_set.get_mapping()
        test_set = ConllDataset(data_path + 'eng.testb', self.word_vocab, self.char_vocab, self.tag_vocab, scheme = 'BIOES')
        valid_set = ConllDataset(data_path + 'eng.testa', self.word_vocab, self.char_vocab, self.tag_vocab, scheme = 'BIOES')
        logger('Load data. Train data: {}, Valid data: {}, Test data: {}'.format(len(train_set.data), len(valid_set.data), len(test_set.data)))
        
        self.train_loader = DataLoader(train_set, batch_size = self.batch_size, collate_fn = collate_fn_conll, shuffle = False)
        self.test_loader  = DataLoader(test_set,  batch_size = self.batch_size, collate_fn = collate_fn_conll, shuffle = False)
        self.valid_loader = DataLoader(valid_set, batch_size = self.batch_size, collate_fn = collate_fn_conll, shuffle = False)

    def train(self):
        model = self.model
        # optimizer = optim.SGD(model.parameters(), lr = self.lr, weight_decay = self.decay_rate, momentum = self.momentum)
        optimizer = optim.Adam(model.parameters(), lr = 1e-4)
        early_stopping = EarlyStopping(patience = 20, verbose = False)
        entrophy = nn.CrossEntropyLoss()

        avg_train_losses = []
        avg_valid_losses = []
        for epoch in range(self.epochs):
            train_losses = []
            valid_losses = []
            model.train()
            for i, batch in enumerate(self.train_loader):
                #print(batch)
                optimizer.zero_grad()
                word_ids = batch["word_ids"] # (batch_size, sen_len)
                word_mask = batch["word_mask"]
                char_ids = batch["char_ids"]
                char_mask = batch["char_mask"]
                label = batch["tag_ids"] # (batch_size, sen_len)
                # for i in range(len(word_ids)):
                #     sentence = word_ids[i]
                #     print(sentence)
                #     print([self.word_vocab.word_list[id] for id in sentence])
                #     print(label[i])
                #     print([self.tag_vocab.ix_to_tag[id] for id in label[i]])
                #     return
                
                if self.use_crf:
                    loss = model(word_ids, word_mask, char_ids, char_mask, label) # (batch_size, sen_len, tagset_size)
                else:
                    output = model(word_ids, word_mask, char_ids, char_mask) # (batch_size, sen_len, tagset_size)
                    output = output.permute(0, 2, 1) # (batch_size, tagset_size, sen_len)
                    loss = entrophy(output, label)
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(self.valid_loader):
                    word_ids = batch["word_ids"] # (batch_size, sen_len)
                    word_mask = batch["word_mask"]
                    char_ids = batch["char_ids"]
                    char_mask = batch["char_mask"]
                    label = batch["tag_ids"]
                    if self.use_crf:
                        loss = model(word_ids, word_mask, char_ids, char_mask, label) # (batch_size, sen_len, tagset_size)
                    else:
                        output = model(word_ids, word_mask, char_ids, char_mask) # (batch_size, sen_len, tagset_size)
                        output = output.permute(0, 2, 1) # (batch_size, tagset_size, sen_len)
                        loss = entrophy(output, label)
                    valid_losses.append(loss.item())
                avg_train_loss = np.average(train_losses)
                avg_valid_loss = np.average(valid_losses)
                avg_train_losses.append(avg_train_loss)
                avg_valid_losses.append(avg_valid_loss)
                logger('[epoch {:3d}] train_loss: {:.8f}  valid_loss: {:.8f}'.format(epoch + 1, avg_train_loss, avg_valid_loss))
                early_stopping(avg_valid_loss, model)
                if early_stopping.early_stop:
                    logger("Early stopping")
                    break

        model_time = '{}'.format(time.strftime('%m%d%H%M', time.localtime()))
        torch.save(model.state_dict(), './model/model_{}'.format(model_time))
        self.model = model
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(avg_train_losses)
        plt.plot(avg_valid_losses)
        plt.legend(['train_loss', 'valid_loss'])
        plt.savefig('./loss/loss_{}.png'.format(model_time), format = 'png')
        logger('Save result {}'.format(model_time))

        self.test()
            
    def test(self):
        model = self.model
        model.eval()
        gold_num, predict_num, correct_num = 0, 0, 0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                word_ids = batch["word_ids"] # (batch_size, sen_len)
                word_mask = batch["word_mask"]
                char_ids = batch["char_ids"]
                char_mask = batch["char_mask"]
                label = batch["tag_ids"] # (batch_size, sen_len)
                
                if self.use_crf:
                    predict = model(word_ids, word_mask, char_ids, char_mask) # (batch_size, sen_len)
                else:
                    output = model(word_ids, word_mask, char_ids, char_mask) # (batch_size, sen_len, tagset_size)
                    predict = torch.max(output, dim = 2).indices # (batch_size, sen_len)
                correct += torch.sum(predict[word_mask] == label[word_mask]).item()
                total += torch.sum(word_mask).item()
                
                for i in range(label.shape[0]):
                    gold_entity = label_sentence_entity(label[i].tolist(), self.tag_vocab.ix_to_tag)
                    pred_entity = label_sentence_entity(predict[i], self.tag_vocab.ix_to_tag)
                    gold_num += len(gold_entity)
                    predict_num += len(pred_entity)
                    for entity in gold_entity:
                        if entity in pred_entity:
                            correct_num += 1
                    print(gold_entity)
                    print(pred_entity)
                return
            precision = correct_num / (predict_num + 0.000000001)
            recall = correct_num / (gold_num + 0.000000001)
            f1 = 2 * precision * recall / (precision + recall + 0.000000001)
            logger('[Test] Tagging accuracy: {:.8f}'.format(correct / total))
            logger('[Test] Precision: {:.8f} Recall: {:.8f} F1: {:.8f}'.format(precision, recall, f1))

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.test()

if __name__ == '__main__':
    #data_path = '/data/hyz/CoNLL2003/'
    #data_path = './data_small/'
    data_path = '/home/gene/Documents/Data/CoNLL2003/'

    trainer = Trainer(data_path, epochs = 100, use_pretrained = True, use_char = True, use_crf = True)
    trainer.train()
    #trainer.load_model('model/model_12091232')
