import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def pad_sequence(sequences: list,
                 batch_first: bool =True) -> Variable:
    """
    :param sequences: list of Variable: [chars]
    :param batch_first: Whether if batch index comes first in the output
    :return: Variable of added sequences
    """
    max_size = sequences[0].size()
    max_len, trailing_dims = max_size[0], max_size[1:]
    prev_l = max_len
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_variable = Variable(sequences[0].data.new(*out_dims).zero_())
    for i, variable in enumerate(sequences):
        length = variable.size(0)
        # temporary sort check, can be removed when we handle sorting internally
        if prev_l < length:
                raise ValueError("lengths array has to be sorted in decreasing order")
        prev_l = length
        # use index notation to prevent duplicate references to the variable
        if batch_first:
            out_variable[i, :length, ...] = variable
        else:
            out_variable[:length, i, ...] = variable

    return out_variable

class RNNMimick(nn.Module):
    """
    An RNN module to train character embeddings that mimicks word embeddings
    """
    def __init__(self, args) -> None:
        """
        :param args: argparse contains following fields:
        itos: dict, indices of words
        ctoi: dict, character to indices
        char_dict: character to index dictionary
        char_size: character vocabulary size
        vocab_size: word vocabulary size
        char_embed: dimension of character embedding
        word_embed: dimension of word embedding
        cell_size: cell size of LSTM
        hidden_size: size of hidden layer
        cuda: if use GPU
        """
        super(RNNMimick, self).__init__()
        self.itos = args.itos
        self.stoi = args.stoi
        self.ctoi = args.ctoi
        self.char_size = args.char_size
        self.vocab_size = args.vocab_size
        self.char_embed = args.char_embed
        self.word_embed = args.word_embed
        self.hidden_size = args.hidden_size
        self.gpu = args.gpu
        self.nonlinear = F.tanh
        self.data = args.data
        self.words = set(self.stoi.keys())

        # Model
        self.embedding = nn.Embedding(self.char_size, self.char_embed, padding_idx=0)
        self.cell_size = args.cell_size
        self.rnn = nn.LSTM(input_size=self.char_embed,
                           hidden_size=self.cell_size,
                           batch_first=True,
                           bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(in_features=2 * self.cell_size,
                                            out_features=self.hidden_size),
                                  nn.Tanh(),
                                  nn.Linear(in_features=self.hidden_size,
                                            out_features=self.word_embed))

    @classmethod
    def word2chars(self, word):
        """
        :param word: A batch of words [batch]
        :return: A batch of char collections [batch, max_word_len]
        """
        if not self.cuda:
            word = word.numpy()
        else:
            word = word.cpu().numpy()

        word = [self.itos[w] for w in word]
        word = [[self.ctoi[c] for c in w] for w in word]
        word = sorted(word, key=lambda x:-len(x))
        word = [Variable(torch.LongTensor(w)) for w in word]
        word = pad_sequence(word)
        if not self.gpu:
            return word
        else:
            return word.cuda()

    def forward(self,
                word: torch.LongTensor) -> torch.FloatTensor:
        """
        :param word: A batch of words [batch]
        :return: mimicked word embedding
        """
        chars = self.word2chars(word) # [batch] -> [batch, cidx]
        char_embed = self.embedding(chars)
        char_rnn = self.rnn(char_embed)
        char_rnn = char_rnn[0][:, -1, :].view(-1, 2 * self.cell_size) # [batch, 2 * cell_size]
        mimick = self.proj(char_rnn)
        return mimick

    def mimick_embedding(self, word: str) -> torch.FloatTensor:
        """
        :param word: string of the word
        :return:
        """
        word = [self.ctoi[c] for c in word]
        word = Variable(torch.LongTensor([word]))
        word = self.embedding(word)
        word = self.rnn(word)
        word = word[0][:, -1, :].view(-1, 2 * self.cell_size)
        word = self.proj(word)
        return word.squeeze().data