import torch
import torch.nn as nn
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
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