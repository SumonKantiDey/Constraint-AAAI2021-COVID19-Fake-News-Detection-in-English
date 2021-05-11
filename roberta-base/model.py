import config
import transformers
import torch
import numpy as np
import torch.nn as nn
from transformers import AutoModel
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

class RobertaBase(nn.Module):
    """
    Model class that combines a pretrained bert model with a linear later
    """
    def __init__(self):
        super(RobertaBase, self).__init__()
        # Load the pretrained RobBERTa model
        self.roberta = AutoModel.from_pretrained(args.pretrained_model_name,output_hidden_states=True)
        # Set 10% dropout to be applied to the RobBERTa backbone's output
        self.drop_out = nn.Dropout(args.dropout)
        # 768 is the dimensionality of roberta-base's hidden representations
        # Multiplied by 2 since the forward pass concatenates the last two hidden representation layers
        self.l0 = nn.Linear(args.roberta_hidden * 2, 1)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, ids, attention_mask):

        _, _, out = self.roberta(
            ids,
            attention_mask=attention_mask,
        )
        out = torch.cat((out[-1], out[-2]), dim=-1) 
        out = out[:,0,:]
        out = self.drop_out(out)
        logits = self.l0(out)
        return logits

class RobertaBaseNext(nn.Module):
    """
    Model class that combines a pretrained bert model with a linear later
    """
    def __init__(self):
        super(RobertaBaseNext, self).__init__()
        # Load the pretrained RobBERTa model
        self.roberta = AutoModel.from_pretrained(args.pretrained_model_name,output_hidden_states=True)
        # Set 10% dropout to be applied to the RobBERTa backbone's output
        self.drop_out = nn.Dropout(args.dropout)
        # 768 is the dimensionality of roberta-base's hidden representations
        # Multiplied by 2 since the forward pass concatenates the last two hidden representation layers
        self.l0 = nn.Linear(args.roberta_hidden * 4, 1)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
    def _get_cls_vec(self, vec):
        return vec[:,0,:].view(-1, args.roberta_hidden)
    def forward(self, ids, attention_mask):
        _, _, hidden_states = self.roberta(
            ids,
            attention_mask=attention_mask
        )
        vec1 = self._get_cls_vec(hidden_states[-1])
        vec2 = self._get_cls_vec(hidden_states[-2])
        vec3 = self._get_cls_vec(hidden_states[-3])
        vec4 = self._get_cls_vec(hidden_states[-4])

        out = torch.cat([vec1, vec2, vec3, vec4], dim=1)
        #out = self.drop_out(out)
        logits = self.l0(out)
        return logits
