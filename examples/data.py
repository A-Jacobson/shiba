import os
import torch
import numpy as np

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
    
    
class LMLoader:
    """Language Model Loader
      Starting from sequential data, arrange the data into columns.
      For instance, with the alphabet as the sequence and batch size 4, we'd get
      ┌ a g m s ┐
      │ b h n t │
      │ c i o u │
      │ d j p v │
      │ e k q w │
      └ f l r x ┘.
      One batchified, return chunks of (batch_size, seq_len), no overlap,
      with the next word in the sequence as the target.
      
      
    """
    def __init__(self, data, batch_size=4, seq_len=35, variable_length=False, device='cpu'):
        self.data = self.batchify(data, batch_size, device)
        self.seq_len = seq_len
        self.variable_length = variable_length
        self.batch_size = batch_size
        self.i = 0
    
    @staticmethod
    def batchify(data, batch_size, device):
        # Work out how cleanly we can divide the dataset into batch_size parts.
        nbatch = data.size(0) // batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * batch_size)
        # Evenly divide the data across the batch_size batches.
        data = data.view(batch_size, -1).t().contiguous()
        return data.to(device)
        
    def get_batch(self, i, seq_len):
        seq_len = min(seq_len, len(self.data) - 1 - i)
        inputs = self.data[i:i+seq_len]
        targets = self.data[i+1:i+1+seq_len].view(-1) # get the next words
        return inputs, inputs
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.i < len(self.data) - 1:
            if self.variable_length:
                # Variable length backpropagation sequences https://arxiv.org/pdf/1708.02182.pdf
                seq_len = self.seq_len if np.random.random() < 0.95 else self.seq_len / 2.
                seq_len = max(5, int(np.random.normal(seq_len, 5)))

            else:
                seq_len = self.seq_len
            batch = self.get_batch(self.i, seq_len)
            self.i += self.seq_len
            return batch
        else:
            self.i = 0
            raise StopIteration
            
    def __len__(self):
        ## will slightly overestimate if shuffle_seq_len == True
        if self.variable_length:
            return len(self.data) // (self.seq_len + 2)
        return len(self.data) // self.seq_len 
        
