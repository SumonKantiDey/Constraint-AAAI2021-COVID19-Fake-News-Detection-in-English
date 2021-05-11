import torch
import torch.nn as nn
import numpy as np
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def get_predictions(data_loader, model, device):
        model.eval()
        predictions = []
        with torch.no_grad():
            for bi, d in enumerate(data_loader):
                ids = d["ids"]
                mask = d["mask"]
                targets = d["targets"]
                ids = ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)
                targets = targets.to(device, dtype=torch.float)

                outputs = model(
                    ids=ids,
                    attention_mask=mask
                )
                preds = torch.round(nn.Sigmoid()(outputs)).squeeze()
                predictions.extend(preds)
        predictions = torch.stack(predictions).cpu()
        return predictions