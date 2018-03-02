import os
import argparse
import torch
import datetime

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import RNNMimick
from torch.autograd import Variable

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--embedding', type=str, default='./data/glove.txt', help="embedding path")
parser.add_argument('--unk', type=str, default='<unk>', help="unknown token")
parser.add_argument('--cpad', type=str, default='<cpad>', help="Padding character for tokens")
parser.add_argument('--batch_size', type=int, default=256, help="batch size of the dataset")
parser.add_argument('--epochs', type=int, default=100, help="number of epochs")
parser.add_argument('--cell_size', type=int, default=200, help="hidden size of the LSTM")
parser.add_argument('--hidden_size', type=int, default=512, help="hidden size of the MLP")
parser.add_argument('--char_embed', type=int, default=128, help="embedding size of char vectors")
parser.add_argument('--no_cuda', action='store_true', default=False, help='disable the gpu')
parser.add_argument('--save_dir', type=str, default='snapshot', help='where to save the snapshot')
args = parser.parse_args()

class dataset():
    def __init__(self, args):
        self.data = []
        self.stoi = {}
        self.embedding = args.embedding

    def prepare(self):
        with open(args.embedding, "rb") as f:
            f.readline()
            idx = 0
            for line in f:
                dataline = line.rstrip().split()
                try:
                    key = dataline[0].decode("utf-8")
                    vector = list(map(float, dataline[1:]))
                    if (len(key) > 1) & (key != self.unk):
                        self.data.append(vector)
                        self.stoi.update({key: idx})
                        idx += 1
                except IndexError:
                    pass

        self.itos = {self.stoi[word]: word for word in self.stoi.keys()}
        self.chars_set = set(''.join(filter(lambda x: x != args.unk, self.stoi.keys())))
        self.chars_set = [args.cpad] + list(self.chars_set)
        self.ctoi = {char: i for i, char in enumerate(self.chars_set)}


print("\nLoading data...")

dataset = dataset(args)
dataset.prepare()

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class to train character embeddings

    Arguments:
        word index (Tensor): contains word index
        word embedding (Tensor): contains the corresponding word embedding
    """

    def __init__(self, word_index, word_embedding):
        assert word_index.size(0) == word_embedding.size(0)
        self.word = word_index
        self.embedding = word_embedding

    def __getitem__(self, index):
        return self.word[index], self.embedding[index]

    def __len__(self):
        return self.word.size(0)

args.char_size = len(dataset.ctoi)
args.vocab_size = len(dataset.stoi)
args.data = torch.FloatTensor(dataset.data)
args.word_embed = args.data.size()[1]

train_idx, test_idx, train_embed, test_embed = train_test_split(range(args.vocab_size),
                                                                dataset.data,
                                                                test_size=0.05)

train_dset = Dataset(torch.LongTensor(train_idx),
                     torch.FloatTensor(train_embed))

test_dset = Dataset(torch.LongTensor(test_idx),
                    torch.FloatTensor(test_embed))

train_dloader, test_dloader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=False), \
                              DataLoader(test_dset, batch_size=args.batch_size, shuffle=False)

args.ctoi = dataset.ctoi
args.stoi = dataset.stoi
args.itos = dataset.itos
args.gpu = (not args.no_cuda) and torch.cuda.is_available()
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

rnnmimick = RNNMimick(args)

if args.gpu:
    rnnmimick = rnnmimick.cuda()

optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, rnnmimick.parameters()), lr=0.001)

global_step = 0
criterion = torch.nn.CosineEmbeddingLoss(size_average=False)


for epoch in range(args.epochs):
    loss_ = 0
    for sample in tqdm(train_dloader):
        mimick_emb = rnnmimick(sample[0])
        embedding = Variable(sample[1])

        if args.gpu:
            embedding = embedding.cuda()

        optimizer.zero_grad()
        y = Variable(torch.cuda.FloatTensor([1.0] * mimick_emb.size()[0]))
        loss = criterion(mimick_emb, embedding, y)
        loss_ += loss.cpu().data.numpy() / len(train_dloader)
        global_step += 1
        loss.backward()
        optimizer.step()

    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
    save_prefix = os.path.join(args.save_dir, 'snapshot')
    save_path = '{}_steps_{}.pt'.format(save_prefix, global_step)
    torch.save(rnnmimick.state_dict(), save_path)
    print('Epoch:[%d/%d], Step:[%d/%d], Loss: %.4f' % (epoch + 1, args.epochs,
                                                       global_step + 1, len(train_dloader) * args.epochs, loss_))

    loss_ = 0
    for sample in tqdm(test_dloader):
        mimick_emb = rnnmimick(sample[0])
        embedding = Variable(sample[1])
        y = Variable(torch.cuda.FloatTensor([1.0] * mimick_emb.size()[0]))
        loss = criterion(mimick_emb, embedding, y)
        loss_ += loss.cpu().data.numpy() / len(test_dloader)

    print('Test Performance:\nEpoch:[%d/%d], Loss: %.4f' % (epoch + 1, args.epochs, loss_))