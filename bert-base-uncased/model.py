import torch
import torch.nn as nn
import numpy as np
import transformers
from flag import get_parser

parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

class BertBaseUncased(nn.Module) :
    def __init__(self) : 
        super(BertBaseUncased,self).__init__() 
        self.bert = transformers.BertModel.from_pretrained(args.bert_path, output_hidden_states=True) 
        self.drop_out = nn.Dropout(args.dropout) 
        self.l0 =  nn.Linear(args.bert_hidden * 2, 1)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def forward(self,ids,attention_mask,token_type_ids): 
        _, _, out = self.bert(
            ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        out = torch.cat((out[-1], out[-2]), dim=-1)
        #out = self.drop_out(out)
        out = out[:,0,:]
        logits = self.l0(out)
        return logits

class BertBaseUncasedNext(nn.Module) :
    def __init__(self) : 
        super(BertBaseUncasedNext,self).__init__() 
        self.bert = transformers.BertModel.from_pretrained(args.bert_path, output_hidden_states=True) 
        self.drop_out = nn.Dropout(args.dropout) 
        self.l0 =  nn.Linear(args.bert_hidden * 4, 1)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
        
    def _get_cls_vec(self, vec):
        return vec[:,0,:].view(-1, args.bert_hidden)
    def forward(self,ids,attention_mask,token_type_ids):
        _, _, hidden_states = self.bert(
            ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        vec1 = self._get_cls_vec(hidden_states[-1])
        vec2 = self._get_cls_vec(hidden_states[-2])
        vec3 = self._get_cls_vec(hidden_states[-3])
        vec4 = self._get_cls_vec(hidden_states[-4])

        out = torch.cat([vec1, vec2, vec3, vec4], dim=1)
        #out = self.drop_out(out)
        logits = self.l0(out)
        return logits