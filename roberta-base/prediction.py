import torch
import torch.nn as nn
import numpy as np 
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
        prediction_probs = []
        real_values = []
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
                    attention_mask=mask,
                )
                #print("Outputs = ",outputs)
                loss = loss_fn(outputs, targets)
                preds = torch.round(nn.Sigmoid()(outputs)).squeeze()
                predictions.extend(preds)
                prediction_probs.extend(outputs.squeeze())
                real_values.extend(targets)
        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
        # print(predictions)
        # print(real_values)
        # print(prediction_probs)
        return predictions, prediction_probs, real_values