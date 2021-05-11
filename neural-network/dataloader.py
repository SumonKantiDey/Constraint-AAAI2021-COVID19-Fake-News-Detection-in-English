import torch
import numpy as np
from torch.utils.data import Dataset
from settings import get_module_logger
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
logger = get_module_logger(__name__)

class TweetDataset(Dataset):
  def __init__(self, df, tokenizer):
    self.df = df
    self.text = df['clean_tweet'].fillna('_##_').values
    self.text_seq = tokenizer.texts_to_sequences(self.text)
    # logger.info('*'*100)
    # logger.info('Dataset info:')
    # logger.info(f'Number of Tweets: {self.df.shape[0]}')
    # logger.info(f'Vocab Size: {len(word_index)}')
    # logger.info('*'*100)

  def __len__(self):
        return self.df.shape[0]  

  def __getitem__(self, idx):
        return self.text_seq[idx], self.df.target[idx], self.df.id[idx] #return text seq and target