import utils
import config
import dataset
import engine
import torch
from sklearn import metrics
import transformers
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn import model_selection
from transformers import AdamW
from dataset import TweetDataset
from prediction import get_predictions
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from settings import get_module_logger
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
logger = get_module_logger(__name__)

def test_evaluation(model, device):
    dfx = pd.read_csv(args.testing_file).dropna().reset_index(drop=True)
    test_dataset = TweetDataset(
        tweet=dfx.clean_tweet.values,
        targets=dfx.target.values
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=4
    )
    # device = torch.device("cuda")
    # model = BertBaseUncased()
    # model.to(device)
    # model.load_state_dict(torch.load("../models/model.bin"))
    y_pred, y_pred_probs, y_test = get_predictions(test_data_loader, model, device)
    
    dfx['y_real'] = y_test
    dfx['y_pred'] = y_pred
    
    pred_test = dfx[['id','tweet','target','y_real','y_pred']]
    pred_test.to_csv(f'../roberta-base/output/{args.output}',index = False)

    print('Accuracy::', metrics.accuracy_score(y_test, y_pred))
    print('Mcc Score::', matthews_corrcoef(y_test, y_pred))
    print('Precision::', metrics.precision_score(y_test, y_pred, average='weighted'))
    print('Recall::', metrics.recall_score(y_test, y_pred, average='weighted'))
    print('F_score::', metrics.f1_score(y_test, y_pred, average='weighted'))
    print('classification_report:: ', metrics.classification_report(y_test, y_pred))#target_names=["real", "fake"]))
    logger.info('Accuracy:: {}'.format(metrics.accuracy_score(y_test, y_pred)))
    logger.info('Mcc Score:: {}'.format(matthews_corrcoef(y_test, y_pred)))
    logger.info('Precision:: {}'.format(metrics.precision_score(y_test, y_pred, average='weighted')))
    logger.info('Recall:: {}'.format(metrics.recall_score(y_test, y_pred, average='weighted')))
    logger.info('F_score:: {}'.format(metrics.f1_score(y_test, y_pred, average='weighted')))
    logger.info('classification_report:: {}'.format(classification_report(y_test, y_pred)))


if __name__ == "__main__":
    test_evaluation()