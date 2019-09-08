#TRAINING CLASSIFIER
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, emb_dim, out_dim, pretrained_vec):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.embedding.weight.data.copy_(pretrained_vec) #load pretrained vec
        self.embedding.weight.requires_grad = False #make embedding non-trainable    
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=2, dropout=0.1, bidirectional=True)
        self.linear = nn.Linear(hidden_dim*2, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, seq):
        emb = self.embedding(seq)
        hdn, _ = self.encoder(emb) #initial h0, c0 is zero. _ : (hn,cn), hdn=output features for each timestep
        #print (seq.shape, emb.shape, hdn.shape)
        feature = hdn[-1, :, :] #take the last timestep of the encoder output
        #print (feature.shape)
        feature = self.linear(feature)
        #print (feature.shape)
        preds = self.predictor(feature)
        #print (preds.shape)
        return preds
