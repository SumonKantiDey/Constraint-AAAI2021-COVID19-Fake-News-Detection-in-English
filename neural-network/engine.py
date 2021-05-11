import utils
import torch
import time
import string
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from settings import get_module_logger
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
logger = get_module_logger(__name__)

def loss_fn(y_pred, y_true):
    return nn.BCEWithLogitsLoss()(y_pred, y_true.view(-1,1))

def train_fn(data_loader, model, optimizer, device, n_examples):
    model.train()
    losses = utils.AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    start = time.time()
    train_losses = []
    correct_predictions = 0
    for bi, (mask, x_batch, y_batch, x_length, _id) in enumerate(tk0):
        optimizer.zero_grad()
        x_batch =  x_batch.to(device)
        y_batch = y_batch.to(device)
        mask = mask.to(device)
        y_pred, weight = model(x_batch, x_length, mask)
        loss = loss_fn(y_pred, y_batch)
        preds = torch.round(nn.Sigmoid()(y_pred)).squeeze().to(device)
        correct_predictions += torch.sum(preds.to(device) == y_batch.to(device))
        train_losses.append(loss.item())

        train_f1 = utils.f1_score(y_pred, y_batch)
        
        f1 = np.round(train_f1.item(), 3)
        end = time.time()

        if (bi % 20 == 0 and bi != 0) or (bi == len(data_loader) - 1) :
            logger.info(f'bi={bi}, Train F1={f1},Train loss={loss}, time={end-start}')
        
        loss.backward() # Calculate gradients based on loss
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step() # Adjust weights based on calculated gradients
        losses.update(loss.item(),x_batch.size(0))
        tk0.set_postfix(loss = losses.avg)
    return correct_predictions.double() / n_examples, np.mean(train_losses)

def eval_fn(data_loader, model, device, n_examples):
    model.eval()
    start = time.time()
    losses = utils.AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    val_losses = []
    correct_predictions = 0
    with torch.no_grad():
        for bi, (mask, x_batch, y_batch, x_length, _id) in enumerate(tk0):
            x_batch =  x_batch.to(device)
            y_batch = y_batch.to(device)
            mask = mask.to(device)
            #print("x_batch = ",x_batch.shape)
            y_pred, weight = model(x_batch, x_length, mask)#.to(device)
            loss = loss_fn(y_pred, y_batch)
            preds = torch.round(nn.Sigmoid()(y_pred)).squeeze()
            correct_predictions += torch.sum(preds == y_batch)
            val_losses.append(loss.item())
            losses.update(loss.item(),x_batch.size(0))
            tk0.set_postfix(loss=losses.avg)
        # model.train()
    return correct_predictions.double() / n_examples, np.mean(val_losses)
