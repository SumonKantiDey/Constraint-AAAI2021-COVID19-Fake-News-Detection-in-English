import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from flag import get_parser
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
parser = get_parser()
args = parser.parse_args()

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
 
    def forward(self,v,q):
        v_proj=self.v_proj(v)
        q_proj=torch.unsqueeze(self.q_proj(q),1)
        vq_proj=F.relu(v_proj +q_proj)
        proj=torch.squeeze(self.att(vq_proj))
        w_att=torch.unsqueeze(self.softmax(proj),2)
        vatt=v * w_att
        att=torch.sum(vatt,1)
        return att

class LSTMAttention(nn.Module):
    def __init__(self,embedding_matrix):
        super(LSTMAttention, self).__init__()        
        self.embedding = nn.Embedding(args.max_features, args.embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm_layers = args.num_layers
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.num_directions = 2
        self.lstm = nn.LSTM(
                            args.embed_size, 
                            args.hidden_size, 
                            bidirectional=True, 
                            batch_first=True)
        #self.gru = nn.GRU(hidden_size*2, hidden_size, bidirectional=True, batch_first=True) 
        self.lstm_attention = Attention(opt)
        # self.gru_attention = Attention(hidden_size*2, config.MAX_LEN)
        
        self.linear = nn.Linear(args.hidden_size*2, args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.out = nn.Linear(args.hidden_size, args.num_layers)
    
    def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(self.lstm_layers * self.num_directions, batch_size, args.hidden_size)),
                Variable(torch.zeros(self.lstm_layers * self.num_directions, batch_size, args.hidden_size)))
        return h.to("cuda"), c.to("cuda")
        
    def forward(self, x, x_length):
        batch_size = x.shape[0]
        h_0, c_0 = self.init_hidden(batch_size)
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))
        packed_embedded = pack_padded_sequence(h_embedding, x_length, batch_first=True)
        h_lstm, (hn, cn) = self.lstm(packed_embedded,(h_0, c_0))
        hn = hn.view(1,2,h_embedding.size(0),args.hidden_size)[-1]
        hidden = torch.cat((hn[0], hn[1]), 1)
        h_lstm_atten = self.lstm_attention(h_lstm,hidden)
        linear = self.linear(h_lstm_atten)
        conc = F.relu(linear)
        conc = self.dropout(conc)
        out = self.out(conc)
        return out