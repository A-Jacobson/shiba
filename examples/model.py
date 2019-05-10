from torch import nn

class LSTMLM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, vocab_size, embedding_size, hidden_size, nlayers, dropout=0.5, tie_weights=False):
        super(LSTMLM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, nlayers, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if embedding_size != hidden_size:
                raise ValueError('When using the tied flag, embedding_size must be equal to hidden_size')
            self.decoder.weight = self.encoder.weight
            
        self.init_weights()
        self.hidden_size = hidden_size
        self.nlayers = nlayers

    def init_weights(self):
        self.encoder.weight.data.uniform_(-0.1, 0.1)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input, hidden):
        embedding = self.drop(self.encoder(input)) # embed input
        output, hidden = self.lstm(embedding, hidden) # encode input given previous state
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, batch_size, self.hidden_size),
                weight.new_zeros(self.nlayers, batch_size, self.hidden_size))
    
