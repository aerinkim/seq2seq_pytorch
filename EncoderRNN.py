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


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        #                          num_embeddings, embedding_dim
        # nn.Embedding is a simple lookup table. 
        # used to store word embeddings and retrieve them using indices. 
        # The input to the module is a list of indices, and the output is the corresponding word embeddings.       
        self.gru = nn.GRU(hidden_size, hidden_size)
        #                  input_size, hidden_size
        #                  # of expected features in the input x, # of features in hidden state h
        # Why does GRU input look like this? (2 hidden_sizes param)
        # because # of expected features in the input x = hidden dim
        #
        # Inputs: input, h_0
        # input of shape (seq_len, batch, input_size): tensor containing the features of the input sequence. 
        #                 # of words, 32, input_size like 27 (character + 1)
        # h_0 of shape (num_layers * num_directions, batch, hidden_size): 
        # tensor containing the initial hidden state for each element in the batch. 
        # Outputs: output, h_n


    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


"""
Dimensions:

encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device) 

input_size = input_lang.n_words
hidden_size = hidden_size
hidden_size = 256

In [120]: encoder1 = EncoderRNN(input_lang.n_words, hidden_size)

In [121]: encoder1.hidden_size
Out[121]: 256

In [122]: encoder1.embedding
Out[122]: Embedding(4489, 256)

In [123]: encoder1.gru
Out[123]: GRU(256, 256)



==== LSTM Comment ====

>>> rnn = nn.LSTMCell(10, 20)
>>> input = torch.randn(6, 3, 10)
>>> hx = torch.randn(3, 20) # 3: batch , h :[1, dim], therefore 20 is dim
>>> cx = torch.randn(3, 20) # 3: batch , c :[1, dim], therefore 20 is dim
>>> output = []
>>> for i in range(6):
        hx, cx = rnn(input[i], (hx, cx))
        output.append(hx)


"""