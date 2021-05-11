import torch 
import torch.nn as nn
import numpy as np
import time
import os
import json
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
def f1_score(y_pred, y_true):
    y_true = y_true.squeeze()
    y_pred = torch.round(nn.Sigmoid()(y_pred)).squeeze()
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    recall = tp / (tp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    return 2*(precision*recall) / (precision + recall + epsilon)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_masks(tokens,length):
        masks = [0]*(length-len(tokens)) + [1]*(len(tokens))
        masks = masks[:length]
        return masks

def padding_sent(tokens,length):
        if len(tokens)<length:
            padding=[0]*(length-len(tokens))
            tokens = padding + tokens
        else:
            tokens = tokens[:length]
        return tokens

def collate_fn(data):
    """This function will be used to pad the tweets to max length
       in the batch and transpose the batch from 
       batch_size x max_seq_len to max_seq_len x batch_size.
       It will return padded vectors, labels and lengths of each tweets (before padding)
       It will be used in the Dataloader
    """
    data.sort(key=lambda x: len(x[0]), reverse=True)
  
    lens = [len(sent) for sent, label, _id in data]
    labels = []
    att_mask = []
    padded_sents = []
    ids = []
    for i, (sent, label, _id) in enumerate(data):
        ids.append(_id)
        text_padded = padding_sent(sent, max(lens))
        labels.append(label)
        mask = get_masks(sent, max(lens))
        att_mask.append(mask)
        padded_sents.append(text_padded)
    #padded_sents = padded_sents.transpose(0,1)
    return torch.FloatTensor(att_mask), torch.LongTensor(padded_sents), torch.FloatTensor(labels), lens, torch.LongTensor(ids)


def _store_data(store_data):
    path = "/content/drive/My Drive/COVID19 Fake News Detection in English/neural-network/text-attn-vis/"
    with open(f'{path}attentions.json', 'w') as fp:
        json.dump(store_data, fp)











#Function to pad and transpose data (to be used in Dataloader)
# def collate_fn(data):
#     """This function will be used to pad the tweets to max length
#        in the batch and transpose the batch from 
#        batch_size x max_seq_len to max_seq_len x batch_size.
#        It will return padded vectors, labels and lengths of each tweets (before padding)
#        It will be used in the Dataloader
#     """
#     data.sort(key=lambda x: len(x[0]), reverse=True)
  
#     lens = [len(sent) for sent, label in data]
#     labels = []
#     padded_sents = torch.zeros(len(data), max(lens)).long()
#     att_mask = []
#     for i, (sent, label) in enumerate(data):
#         padded_sents[i,:lens[i]] = torch.LongTensor(sent)
#         labels.append(label)
#         mask = get_masks(sent, max(lens))
#         att_mask.append(mask)
#     #padded_sents = padded_sents.transpose(0,1)
#     return torch.FloatTensor(att_mask), padded_sents, torch.FloatTensor(labels), lens



