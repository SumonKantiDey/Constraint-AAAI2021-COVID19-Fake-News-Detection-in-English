import utils
import torch
import time
import string
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from settings import get_module_logger
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from flag import get_parser
from sklearn.metrics import f1_score
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
logger = get_module_logger(__name__)


def loss_fn(y_pred, y_true):
    return nn.BCEWithLogitsLoss()(y_pred, y_true.view(-1,1))

def train_fn(data_loader, model, optimizer, device, scheduler, n_examples):
    model.train()
    losses = utils.AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    start = time.time()
    train_losses = []
    fin_targets = []
    fin_outputs = []
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        mask = d["mask"]
        targets = d["targets"]
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        # Reset gradients
        model.zero_grad()
        outputs = model(
            ids=ids,
            attention_mask=mask
        )
        loss = loss_fn(outputs, targets)
        train_losses.append(loss.item())

        outputs = torch.round(nn.Sigmoid()(outputs)).squeeze()
        targets = targets.squeeze()
        outputs = outputs.cpu().detach().numpy().tolist()
        targets = targets.cpu().detach().numpy().tolist()
        

        train_f1 = f1_score(outputs, targets)
        end = time.time()

        f1 = np.round(train_f1.item(), 3)
        if (bi % 100 == 0 and bi != 0) or (bi == len(data_loader) - 1) :
            logger.info(f'bi={bi}, Train F1={f1},Train loss={loss.item()}, time={end-start}')
        
        loss.backward() # Calculate gradients based on loss
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step() # Adjust weights based on calculated gradients
        scheduler.step() # Update scheduler
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss = losses.avg)
        fin_targets.extend(targets) 
        fin_outputs.extend(outputs)
    f1 = f1_score(fin_outputs, fin_targets, average='weighted')
    f1 = np.round(f1.item(), 3)
    return f1, np.mean(train_losses)

def eval_fn(data_loader, model, device, n_examples):
        model.eval()
        start = time.time()
        losses = utils.AverageMeter()
        val_losses = []
        fin_targets = []
        fin_outputs = []
        with torch.no_grad():
            #tk0 = tqdm(data_loader, total=len(data_loader))
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
                loss = loss_fn(outputs, targets)
                val_losses.append(loss.item())
                targets = targets.squeeze()
                outputs = torch.round(nn.Sigmoid()(outputs)).squeeze()
                
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
        f1 = f1_score(fin_outputs, fin_targets, average='weighted')
        f1 = np.round(f1.item(), 3)
        logger.info('Accuracy:: {}'.format(metrics.accuracy_score(fin_targets, fin_outputs)))
        logger.info('Mcc Score:: {}'.format(matthews_corrcoef(fin_targets, fin_outputs)))
        logger.info('Precision:: {}'.format(metrics.precision_score(fin_targets, fin_outputs, average='weighted')))
        logger.info('Recall:: {}'.format(metrics.recall_score(fin_targets, fin_outputs, average='weighted')))
        logger.info('F_score:: {}'.format(metrics.f1_score(fin_targets, fin_outputs, average='weighted')))
        return f1, np.mean(val_losses)