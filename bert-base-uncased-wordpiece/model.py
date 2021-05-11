import torch
import torch.nn as nn
import transformers
import numpy as np
import config
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
class BertBaseUncased(transformers.BertPreTrainedModel) :
    def __init__(self, conf) : 
        super(BertBaseUncased,self).__init__(conf) 
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH, config=conf) 
        self.drop_out = nn.Dropout(0.01) 
        self.l0 =  nn.Linear(768 * 2, 1)
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

class BertBaseUncasedNext(transformers.BertPreTrainedModel) :
    def __init__(self, conf) : 
        super(BertBaseUncasedNext,self).__init__(conf) 
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH, config=conf) 
        self.drop_out = nn.Dropout(0.01) 
        self.l0 =  nn.Linear(768 * 4, 1)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
    def _get_cls_vec(self, vec):
        return vec[:,0,:].view(-1, 768)
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
        #out = out[:,0,:]
        logits = self.l0(out)
        return logits