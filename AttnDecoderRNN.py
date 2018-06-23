from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 10

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length) 
        #            output will be ( input row dim, output column dim )
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs): 
        # dimesions : [1,256], [1,256], [10,256] # why 10? because it's max_length, and you want to 
        # input is a decoder output. # encoder_outputs is stacked h's.
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # [1,10] = [ 1, 256*2 ] * [256*2, 10]

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),   # torch.Size([1, 1, 10])
                                 encoder_outputs.unsqueeze(0))# torch.Size([1, 10, 256])
        # torch.Size([1, 1, 256]) =  torch.Size([1, 1, 10]) * torch.Size([1, 10, 256])
        
        output = torch.cat((embedded[0], attn_applied[0]), 1) 
        #                    [1,256]         [1,256]
        # Why concatenate? This is how we code for
        # P(y_i|y1,y2,y3...) = g(y_i-1, S_i-1, C_i)
        output = self.attn_combine(output).unsqueeze(0)
        # Then we use linear combination to make this combind 'S' and 'C' dimension same as that of hidden. 
        # (so that we can put it in GRU.)
        # [1,1,256]       
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        #                         [1,256],[1,256]                            
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)