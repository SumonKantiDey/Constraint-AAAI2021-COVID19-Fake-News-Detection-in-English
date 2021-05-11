import torch
import numpy as np
import torch.nn as nn
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def loss_fn(y_pred, y_true):
    return nn.BCEWithLogitsLoss()(y_pred, y_true.view(-1,1))

def get_predictions(data_loader, model, device):
        model.eval()
        predictions = []
        with torch.no_grad():
            for bi, d in enumerate(data_loader):
                ids = d["ids"]
                mask = d["mask"]
                token_type_ids = d["token_type_ids"]
                targets = d["targets"]
                ids = ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)
                token_type_ids = token_type_ids.to(device, dtype=torch.long)
                targets = targets.to(device, dtype=torch.float)

                outputs = model(
                    ids=ids,
                    attention_mask=mask,
                    token_type_ids = token_type_ids 
                )
                preds = torch.round(nn.Sigmoid()(outputs)).squeeze()
                predictions.extend(preds)
        predictions = torch.stack(predictions).cpu()
        return predictions