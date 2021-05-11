import torch
import torch.nn as nn
import transformers
import numpy as np
from transformers import AutoModel
class BertLargeUncased(nn.Module):
    def __init__(self):
        super(BertLargeUncased, self).__init__()
        self.roberta = AutoModel.from_pretrained("/content/drive/My Drive/pre_trained_model/bert-large-uncased/",output_hidden_states=True)
        self.drop_out = nn.Dropout(0.1) 
        self.l0 =  nn.Linear(1024 * 2, 1)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertLargeUncased()
model.to(device)
