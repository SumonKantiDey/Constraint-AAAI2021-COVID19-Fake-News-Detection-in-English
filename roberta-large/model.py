import torch
import torch.nn as nn
import transformers
import numpy as np
from transformers import AutoModel
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
class RobertaLarge(nn.Module):
    def __init__(self):
        super(RobertaLarge, self).__init__()
        self.roberta = AutoModel.from_pretrained(args.pretrained_model_name,output_hidden_states=True)
        self.drop_out = nn.Dropout(args.dropout) 
        self.l0 =  nn.Linear(args.roberta_hidden * 2, 1)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def forward(self,ids,attention_mask):
        _, _, out = self.roberta(
            ids,
            attention_mask=attention_mask
        )
        out = torch.cat((out[-1], out[-2]), dim=-1)
        #out = self.drop_out(out)
        out = out[:,0,:]
        logits = self.l0(out)
        return logits


class RobertaLargeNext(nn.Module):
    def __init__(self):
        super(RobertaLargeNext, self).__init__()
        self.roberta = AutoModel.from_pretrained(args.pretrained_model_name,output_hidden_states=True)
        self.drop_out = nn.Dropout(args.dropout) 
        self.l0 =  nn.Linear(args.roberta_hidden * 4, 1)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def _get_cls_vec(self, vec):
        return vec[:,0,:].view(-1, args.roberta_hidden)

    def forward(self,ids,attention_mask):
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
