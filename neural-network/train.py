import os
import utils
import gc
import random
import engine
import re
import string
import time
import warnings
from models import LSTMClassifier, LSTMAttention, LSTMHidden, PureLstm
from sklearn import model_selection
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
import torch.utils.data
from torch.optim.optimizer import Optimizer 
from sklearn.model_selection import train_test_split
from  embedding import load_glove, load_fasttext
from dataloader import TweetDataset
from utils import collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from collections import defaultdict
tqdm.pandas()

from settings import get_module_logger
from test_eval import test_evaluation
from testatten_eval import testatten_evaluation
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
logger = get_module_logger(__name__)
def run():
    np.random.seed(args.seed)
    dfx =  pd.read_csv(args.train).dropna().reset_index(drop = True)
    dfx['id'] = [val+1 for val in range(len(dfx))]
    text = dfx['clean_tweet'].fillna('_##_').values

    tokenizer = Tokenizer(num_words=args.max_features)
    tokenizer.fit_on_texts(list(text))
    word_index = tokenizer.word_index

    train_df, val_df = train_test_split(
                                        dfx, 
                                        test_size=0.2, 
                                        random_state=42, 
                                        stratify=dfx['target']
                                        )

    train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    logger.info("train shape {} valid shape {}".format(train_df.shape, val_df.shape))
    
    #embedding_matrix, _ = load_glove(word_index, args.max_features, unk_uni=False, create = False)
    embedding_matrix, _ = load_fasttext(word_index, args.max_features, unk_uni=True, create = False)
   
    train = TweetDataset(train_df, tokenizer)
    valid = TweetDataset(val_df, tokenizer)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_name == "lstm_attention":
        model = LSTMAttention(embedding_matrix)
    else:
        model = PureLstm(embedding_matrix)

    model.to(device)
    
    train_loader = DataLoader(
                                train, 
                                batch_size=args.train_batch_size, 
                                shuffle=True, 
                                collate_fn=collate_fn,
                                pin_memory = True
                             )
    
    
    valid_loader = DataLoader(
                                valid, 
                                batch_size=args.valid_batch_size, 
                                shuffle=False, 
                                collate_fn=collate_fn,
                                pin_memory = True
                             )


    optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    


    logger.info("{}".format("STARTING TRAINING"))
    logger.info("{} - {}".format("STARTING TRAINING",args.model_specification))
    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(args.epochs):
        logger.info(f'Epoch {epoch + 1}/{args.epochs}')
        logger.info('-' * 10)

        train_acc, train_loss = engine.train_fn(
                                                train_loader, 
                                                model, 
                                                optimizer, 
                                                device, 
                                                len(train_df)
                                                )
        logger.info(f'Train loss {train_loss} accuracy {train_acc}')
        val_acc, val_loss = engine.eval_fn( 
                                            valid_loader, 
                                            model, 
                                            device, 
                                            len(val_df)
                                            )
        logger.info(f'Val loss {val_loss} accuracy {val_acc}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            #torch.save(model.state_dict(), args.model_path)
            best_accuracy = val_acc
            if args.model_name == "lstm_attention":
                testatten_evaluation(tokenizer, model, device)
            else:
                test_evaluation(tokenizer, model, device)
    logger.info("##################################### Task End ############################################")
    
    
if __name__ == "__main__":
    run()