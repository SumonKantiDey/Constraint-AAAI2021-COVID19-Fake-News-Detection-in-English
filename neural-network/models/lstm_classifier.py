import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

class LSTMClassifier(nn.Module):
    def __init__(self,embedding_matrix):        
        super(LSTMClassifier, self).__init__()
        self.embedding_size = args.embed_size
        self.max_features = args.max_features

        self.embedding = nn.Embedding(self.max_features, self.embedding_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(args.dropout)

        
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.reduction_size = args.reduction_size
        self.ih2h = nn.LSTM(
                        self.embedding_size, 
                        self.hidden_size,
                        self.num_layers,
                        bidirectional=True, 
                        batch_first=True
                        )
        self.fc1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 1)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x, x_length, mask):

        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0))) #(bs x sq * embed_size)
        packed_embedded = pack_padded_sequence(h_embedding, x_length, batch_first=True) 
        o, (h_n, c_n) = self.ih2h(h_embedding)
        # print(h_n[-2, :, :].shape, h_n[-1, :, :].shape)
        cat = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        #print("cat = ", cat.shape)
        rel = self.relu(cat)
        dense1 = self.fc1(rel)
        #print("dens1 = ", dense1.shape)
        drop = self.dropout(dense1)
        preds = self.fc2(drop)
        #print("preds = ", preds.shape)
        return preds, None

class PureLstm(nn.Module):
    def __init__(self,embedding_matrix):        
        super(PureLstm, self).__init__()
        self.embedding_size = args.embed_size
        self.max_features = args.max_features

        self.embedding = nn.Embedding(self.max_features, self.embedding_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(0.1)

        
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.reduction_size = args.reduction_size
        self.ih2h = nn.LSTM(
                        self.embedding_size, 
                        self.hidden_size,
                        self.num_layers,
                        bidirectional=True, 
                        batch_first=True
                        )
        self.h2r = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.r2o = nn.Linear(self.hidden_size, 1)
    
    def forward(self, x, x_length, mask):

        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0))) #(bs x sq * embed_size)
        o, (h_n, c_n) = self.ih2h(h_embedding) # o = bs x sq x 2*hidden, o[-1] = sq x 2*hidden
        o = o.transpose(0, 1).squeeze()
        linear = self.h2r(o[-1])
        conc = self.r2o(linear)
        output = F.relu(conc)
        #print("output shape = ", output.shape)
        return output, None

class LSTMHidden(nn.Module):
    def __init__(self,embedding_matrix):        
        super(LSTMHidden, self).__init__()
        self.embedding_size = args.embed_size
        self.max_features = args.max_features

        self.embedding = nn.Embedding(self.max_features, self.embedding_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(0.1)
        self.num_directions = 2
        self.lstm_layers = args.num_layers
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.reduction_size = args.reduction_size
        self.ih2h = nn.LSTM(
                        self.embedding_size, 
                        self.hidden_size,
                        self.num_layers,
                        bidirectional=True, 
                        batch_first=True
                        )
        self.fc1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 1)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(self.lstm_layers * self.num_directions, batch_size, self.hidden_size)),
                Variable(torch.zeros(self.lstm_layers * self.num_directions, batch_size, self.hidden_size)))
        return h.to("cuda"), c.to("cuda")

    
    def forward(self, x, x_length, mask):
        batch_size = x.shape[0]
        h_0, c_0 = self.init_hidden(batch_size)
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0))) #(bs x sq * embed_size)
        packed_embedded = pack_padded_sequence(h_embedding, x_length, batch_first=True) 
        o, (h_n, c_n) = self.ih2h(packed_embedded,(h_0, c_0))
        output_unpacked, output_lengths = pad_packed_sequence(o, batch_first=True)
        out = output_unpacked[:, -1, :]
        rel = self.relu(out)
        dense1 = self.fc1(rel)
        drop = self.dropout(dense1)
        preds = self.fc2(drop)
        return preds, None