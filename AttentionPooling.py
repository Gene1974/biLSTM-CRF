import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        #self.max_word_len = 16
        self.output_dim = output_dim
        '''
        a = q^T tanh(V * c + v)
        alpha = softmax(a)
        '''
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias = False),
            nn.Softmax(dim = 3)
        )
    
    def forward(self, char_emb):
        '''
        input:
            char_emb: (batch_size, max_sen_len, max_word_len, char_embed_size)
        output:
            atten_emb: (batch_size, max_sen_len, char_embed_size)
        '''
        char_emb = char_emb.permute(0, 1, 3, 2) # (batch_size, max_sen_len, char_embed_size, max_word_len)
        alpha = self.attention(char_emb) # (batch_size, max_sen_len, char_embed_size, max_word_len)
        atten_emb = torch.sum(torch.mul(alpha, char_emb), dim = 3)
        return atten_emb

# reference:
# https://rstudio-pubs-static.s3.amazonaws.com/254996_c29015147c01403b958ea3c1a4c3b70d.html
