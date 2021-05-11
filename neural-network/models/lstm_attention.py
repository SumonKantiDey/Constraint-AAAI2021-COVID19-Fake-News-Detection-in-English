import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

class FCNet(nn.Module):
    def __init__(self,in_dim,out_dim,dropout):
        super(FCNet,self).__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.relu=nn.ReLU()
        self.linear=weight_norm(nn.Linear(in_dim,out_dim),dim=None)
        self.dropout=nn.Dropout(dropout)
 
    def forward(self,x):
        logits=self.dropout(self.linear(x))
        return logits

class opt:
    NUM_HIDDEN = 128
    PROJ_DIM = 128
    FC_DROPOUT = 0.1

class Attention(nn.Module):
    def __init__(self,opt):
        super(Attention,self).__init__()
        self.opt=opt
        self.v_proj=FCNet(self.opt.NUM_HIDDEN,self.opt.PROJ_DIM,self.opt.FC_DROPOUT)
        self.q_proj=FCNet(self.opt.NUM_HIDDEN,self.opt.PROJ_DIM,self.opt.FC_DROPOUT)
        self.att=FCNet(self.opt.PROJ_DIM,1,self.opt.FC_DROPOUT)
        self.softmax=nn.Softmax()
 
    def forward(self,v,q, mask):
        v_proj=self.v_proj(v)
        q_proj=torch.unsqueeze(self.q_proj(q),1)
        vq_proj=F.relu(v_proj +q_proj)
        proj=torch.squeeze(self.att(vq_proj))
        mask = (1.0 - mask) * -10000.0
        proj = proj + mask
        #print("proj shape = ",proj.shape)
        soft_pro = self.softmax(proj)
        #print("sodt = ",soft_pro.shape, soft_pro)
        w_att=torch.unsqueeze(self.softmax(proj),2)
        #print("watt shape = ", w_att.shape)
        vatt=v * w_att
        att=torch.sum(vatt,1)
        return att, w_att

class LSTMAttention(nn.Module):
    def __init__(self,embedding_matrix):
        super(LSTMAttention, self).__init__()        
        self.embedding = nn.Embedding(args.max_features, args.embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.LSTM(
            args.embed_size, 
            args.hidden_size, 
            bidirectional=True, 
            batch_first=True)
        #self.gru = nn.GRU(hidden_size*2, hidden_size, bidirectional=True, batch_first=True) 
        self.lstm_attention = Attention(opt)
        # self.gru_attention = Attention(hidden_size*2, config.MAX_LEN)
        
        self.linear = nn.Linear(args.hidden_size*2,args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.out = nn.Linear(args.hidden_size, args.num_layers)
        
    def forward(self, x, x_length, mask):
        h_embedding = self.embedding(x)
        #print("Embedding shape = ",h_embedding.shape)
        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))
        if len(h_embedding.shape) == 2:
            h_embedding = h_embedding.unsqueeze(0)
        #print("h_embedding shape = ", h_embedding.shape)
        h_lstm, (hn, cn) = self.lstm(h_embedding)
        hn = hn.view(1,2,h_embedding.size(0),args.hidden_size)[-1]
        hidden = torch.cat((hn[0], hn[1]), 1)
        h_lstm_atten, weights = self.lstm_attention(h_lstm,hidden, mask)
        #print("h_lstm_atten = ", h_lstm_atten.shape, weights.shape)
        linear = self.linear(h_lstm_atten)
        conc = F.relu(linear)
        conc = self.dropout(conc)
        out = self.out(conc)
        return out, weights