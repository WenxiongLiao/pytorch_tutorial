import pandas as pd
import numpy as np
import torch

#loading the data
pd.read_csv('data/train.csv').head()

from torchtext.data import Field

tokenize = lambda x: x.split()
#lowercased, whitespace-tokenized, and preprocessed
TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
#labels is numeric already
LABEL = Field(sequential=False, use_vocab=False)

from torchtext.data import TabularDataset
trvld_datafields = [("id", None),
                ("comment_text", TEXT), ("toxic", LABEL),
                ("severe_toxic", LABEL), ("threat", LABEL),
                ("obscene", LABEL), ("insult", LABEL),
                ("identity_hate", LABEL)]

trn, vld = TabularDataset.splits(
            path="data",
            train="train.csv", validation="valid.csv",
            format="csv", skip_header=True,
            fields=trvld_datafields)

tst_datafields = [("id", None), ("comment_text", TEXT)]
tst = TabularDataset(
        path="data/test.csv", format="csv", skip_header=True, fields=tst_datafields)

print ('size of the train, dev, test dataset:', len(trn), len(vld), len(tst))
#print (trn.fields.items())
#print (trn[0].comment_text)
#print (tst[0].comment_text) 

from torchtext import vocab
vec = vocab.Vectors('E:\\语料\word2vec\\glove.6B\\glove.6B.100d.txt', '.\\data\\')
TEXT.build_vocab(trn, vld, max_size=200, vectors=vec)
print ('size of the vocab and embedding dim:', TEXT.vocab.vectors.shape) #size=202,50 (max vocab size=200+2 for <unk> and <pad>, and glove vector dim=50)
print ('index of the in the vocab:', TEXT.vocab.stoi['the']) #output:2, so the index 2 in vocab is for 'the'
print ('index 0 in the vocab:', TEXT.vocab.itos[0]) #output:<unk>, so the index 0 in vocab is for '<unk>'
#print ('embedding vector for 'the':', TEXT.vocab.vectors[TEXT.vocab.stoi['the']])

#print (TEXT.vocab.freqs.most_common(10))
#print (trn[0].__dict__.keys())
#print (trn[0].comment_text[:5])

#BucketIterator groups sequence of similar lengths text in a batch together to minimize padding! how cool is that!
from torchtext.data import Iterator, BucketIterator
train_iter, val_iter = BucketIterator.splits((trn, vld),
                                            batch_sizes=(3,3),
                                            device='cuda',
                                            sort_key=lambda x: len(x.comment_text), #tell the bucketIterator how to group the sequences
                                            sort_within_batch=False,
                                            repeat=False) #we want to wrap this Iterator layer
test_iter = Iterator(tst, batch_size=3, device='cuda', sort=False, sort_within_batch=False, repeat=False)

print ('number of batch (size: 3):', len(train_iter), len(val_iter)) #output: 9,9. because our train and val data only has 25 examples. since the batch size is 3, it means we have 25/3: 9 batches for each train and val

batch = next(iter(train_iter))
print ('details of batch:', batch)
print ("the content of 'toxic' for the first 3 examples in batch 1:", batch.toxic)
#print ("the content of 'comment_text' for the first 3 examples in batch 1, size depends on the longest sequence:", batch.comment_text)
#print (batch.dataset.fields)

class BatchGenerator:
    def __init__(self, dl, x, y):
        self.dl, self.x, self.y = dl, x, y
    def __len__(self):
        return len(self.dl)
    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x) #assuming one input
            if self.y is not None: #concat the y into single tensor
                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y], dim=1).float()
            else:
                y = torch.zeros((1))
            yield (X,y)

train_batch_it = BatchGenerator(train_iter, 'comment_text', ['toxic', 'threat'])
#print ('get data x and y out of batch object:', next(iter(train_batch_it)))
valid_batch_it = BatchGenerator(val_iter, 'comment_text', ['toxic', 'threat'])
test_batch_it = BatchGenerator(test_iter, 'comment_text', None)
#print (next(test_batch_it.__iter__()))

import torch.optim as optim
import torch.nn as nn
from model import BiLSTM

vocab_size = len(TEXT.vocab)
emb_dim = 50
hidden_dim = 50
out_dim = 2 #only use 'toxic' and 'threat'
pretrained_vec = trn.fields['comment_text'].vocab.vectors
model = SimpleLSTM(vocab_size, hidden_dim, emb_dim, out_dim, pretrained_vec)
print (model)
model.cuda()

import tqdm
opt = optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.BCEWithLogitsLoss()
epochs = 100
train_loss = []
valid_loss = []

for epoch in range(1, epochs+1):
    training_loss = 0.0
    training_corrects = 0
    model.train()
    for x, y in tqdm.tqdm(train_batch_it):
        opt.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)

        loss.backward()
        opt.step()
        training_loss += loss.item() * x.size(0)
    epoch_loss = training_loss/ len(trn)

    val_loss = 0.0
    model.eval()
    for x,y in valid_batch_it:
        preds = model(x)
        loss = criterion(preds, y)
        val_loss += loss.item() * x.size(0)
    val_loss /= len(vld)
    train_loss.append(epoch_loss)
    valid_loss.append(val_loss)
    print ('Epoch: {}, Training loss: {:.4f}, Validation loss: {:.4f}'.format(epoch, epoch_loss, val_loss))
    
#predictions. note that the preds is the probability of the comment belong in each category output
test_preds = []
for x, y in tqdm.tqdm(test_batch_it):
    preds = model(x)
    preds = preds.data.cpu().numpy()
    preds = 1/(1+np.exp(-preds)) #actual output of the model are logits, so we need to pass into sigmoid function
    test_preds.append(preds)
    print (y, ' >>> ',  preds)

test_preds = np.hstack(test_preds)

import matplotlib.pyplot as plt
ep = range(1, epochs+1)
print (ep)
plt.plot(ep, train_loss, 'bo', label='Training loss')
plt.plot(ep, valid_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('plt.png')
#print (test_preds)

import sys
sys.exit()

#write predictions to csv
df = pd.read_csv("data/test.csv") 
for i, col in enumerate(["toxic", "threat"]):
    df[col] = test_preds[:, i]
df.head()
       

