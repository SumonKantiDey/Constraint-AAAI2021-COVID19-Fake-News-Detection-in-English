import utils
import config
import dataset
import engine
import torch
import transformers
import pandas as pd
import torch.nn as nn
import numpy as np
from settings import get_module_logger
from model import BertBaseUncased, BertBaseUncasedNext
from sklearn import model_selection
from transformers import AdamW
from dataset import TweetDataset
from transformers import get_linear_schedule_with_warmup
from vis import display_acc_curves, display_loss_curves
from collections import defaultdict
from test_eval import test_evaluation
import gc
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
logger = get_module_logger(__name__)


def run():
    dfx = pd.read_csv(config.TRAINING_FILE).dropna().reset_index(drop=True)
    df_train, df_valid = model_selection.train_test_split(
        dfx, 
        test_size=0.1, 
        random_state=42, 
        stratify=dfx.target.values
    )

    logger.info("train len - {} valid len - {}".format(len(df_train), len(df_valid)))

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = TweetDataset(
        tweet=df_train.clean_tweet.values,
        targets=df_train.target.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    valid_dataset = TweetDataset(
        tweet=df_valid.clean_tweet.values,
        targets=df_valid.target.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=2
    )
    
    device = torch.device("cuda")
    model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)
    model_config.output_hidden_states = True
    model = BertBaseUncased(conf=model_config)
    model.to(device)
   

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # Define two sets of parameters: those with weight decay, and those without
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(
        len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    '''
    Create a scheduler to set the learning rate at each training step
    "Create a schedule with a learning rate that decreases linearly after linearly increasing during a warmup period." (https://pytorch.org/docs/stable/optim.html)
    Since num_warmup_steps = 0, the learning rate starts at 3e-5, and then linearly decreases at each training step
    '''
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )
    #es = utils.EarlyStopping(patience=15, mode="max")
    print("STARTING TRAINING ...\n")
    logger.info("{}".format("STARTING TRAINING"))
    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        logger.info(f'Epoch {epoch + 1}/{config.EPOCHS}')
        logger.info('-' * 10)

        train_acc, train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler, len(df_train))
        logger.info(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = engine.eval_fn(valid_data_loader, model, device, len(df_valid))
        logger.info(f'Val loss {val_loss} accuracy {val_acc}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            #torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = val_acc
            test_evaluation(model, device)

    
    display_acc_curves(history, "acc_curves")
    display_loss_curves(history, "loss_curves")
    del model, train_data_loader, valid_data_loader, train_dataset, valid_dataset
    print(gc.collect())
if __name__ == "__main__":
    run()