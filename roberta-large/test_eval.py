import utils
import dataset
import engine
import torch
import transformers
import pandas as pd
from sklearn import metrics
import torch.nn as nn
import numpy as np
from sklearn import model_selection
from transformers import AdamW
from dataset import TweetDataset
from prediction import get_predictions 
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
    dfx['target'] = 0
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
    y_pred = get_predictions(test_data_loader, model, device)

    dfx['y_pred'] = y_pred
    pred_test = dfx[['id','tweet','y_pred']]
    
    pred_test.to_csv(f'../roberta-large/output/{args.output}',index = False)
if __name__ == "__main__":
    test_evaluation()