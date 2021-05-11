import utils
import engine
import torch
import pandas as pd
from sklearn import metrics
import torch.nn as nn
import numpy as np
from models import LSTMClassifier, LSTMAttention
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import model_selection
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from dataloader import TweetDataset
from utils import collate_fn
from settings import get_module_logger
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
logger = get_module_logger(__name__)

parser = get_parser()
args = parser.parse_args()
def loss_fn(y_pred, y_true):
    return nn.BCEWithLogitsLoss()(y_pred, y_true.view(-1,1))

def testatten_evaluation(tokenizer,model, device):
    #shuffling the data
    np.random.seed(args.seed)

    ids = []
    predictions = []
    prediction_probs = []
    real_values = []
    store_vis = []
    dfx = pd.read_csv(args.test).dropna().reset_index(drop=True)
    test = TweetDataset(dfx, tokenizer)
    logger.info("test data len {}".format(len(dfx)))
    test_loader = DataLoader(
                                test, 
                                batch_size=args.valid_batch_size, 
                                shuffle=False, 
                                collate_fn=collate_fn,
                                pin_memory = True
                             )

    with torch.no_grad():
        for i, (mask, x_batch,y_batch, x_length, _id) in enumerate(test_loader):
            _id = _id.to(device)
            x_batch =  x_batch.to(device)
            y_batch = y_batch.to(device)
            mask = mask.to(device)
            y_pred,weight = model(x_batch, x_length, mask)#.to(device)
            loss = loss_fn(y_pred, y_batch)
            preds = torch.round(nn.Sigmoid()(y_pred)).squeeze()
            # Get decoded text and labels
            id2word = dict(map(reversed, tokenizer.word_index.items()))
            batch_len = len(x_batch)
            for bi in range(batch_len):
                json_store = dict()
                batch = x_batch[bi].cpu().detach().numpy().tolist()
                batch = np.trim_zeros(batch)
                decoded_text = [id2word[word] for word in batch]
                json_store['words'] = decoded_text
                json_store['weights'] = weight[bi].squeeze().cpu().detach().numpy().tolist()
                label = y_batch[bi].cpu().detach().numpy().tolist()
                json_store['label'] = "Real" if label == 0 else "Fake"
                pred = preds[bi].cpu().detach().numpy().tolist()
                json_store['prediction'] = "Real" if pred == 0 else "Fake"
                # store json file
                store_vis.append(json_store)
            ids.extend(_id)
            predictions.extend(preds)
            prediction_probs.extend(y_pred.squeeze())
            real_values.extend(y_batch)
    ids = torch.stack(ids).cpu()
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    
    dfx['y_real'] = real_values
    dfx['y_pred'] = predictions
    dfx['id'] = ids

    pred_test = dfx[['id','y_real','y_pred']]
    pred_test = pred_test.sort_values(by=['id'])
    pred_test = pred_test.reset_index(drop=True)
    pred_test.to_csv(f'../neural-network/output/{args.output}',index = False)
    
    print('Accuracy::', metrics.accuracy_score(real_values,predictions))
    print('Mcc Score::', matthews_corrcoef(real_values,predictions))
    print('Precision::', metrics.precision_score(real_values,predictions, average='weighted'))
    print('Recall::', metrics.recall_score(real_values,predictions, average='weighted'))
    print('F_score::', metrics.f1_score(real_values,predictions, average='weighted'))
    print('classification_report:: ', metrics.classification_report(real_values,predictions))#target_names=["real", "fake"]))
    logger.info('Accuracy:: {}'.format(metrics.accuracy_score(real_values,predictions)))
    logger.info('Mcc Score:: {}'.format(matthews_corrcoef(real_values,predictions)))
    logger.info('Precision:: {}'.format(metrics.precision_score(real_values,predictions, average='weighted')))
    logger.info('Recall:: {}'.format(metrics.recall_score(real_values,predictions, average='weighted')))
    logger.info('F_score:: {}'.format(metrics.f1_score(real_values,predictions, average='weighted')))
    logger.info('classification_report:: {}'.format(classification_report(real_values,predictions)))


if __name__ == "__main__":
    testatten_evaluation()