import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, code_vocab_size, emb_dim, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(code_vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_size, dropout=0.5, num_layers=2).to(device)

    def forward(self, input):
        embedded = self.embedding(input)
        _, hidden_state = self.gru(embedded)
        return hidden_state

class Decoder(nn.Module):
    def __init__(self, docstring_vocab_size, emb_dim, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(docstring_vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_size, dropout=0.5, num_layers=2).to(device)
        self.linear = nn.Linear(hidden_size, docstring_vocab_size)

    def forward(self, input, initial_state):
        embedded = self.embedding(input)
        output, _ = self.gru(embedded, initial_state)
        output = self.linear(output)
        return F.softmax(output)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, code, docstring):
        dec_initial_state = self.encoder(code)
        return self.decoder(docstring, dec_initial_state)



def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

