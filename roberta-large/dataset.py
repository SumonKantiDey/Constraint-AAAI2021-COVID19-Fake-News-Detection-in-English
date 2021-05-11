import utils
import pandas as pd 
import torch
import numpy as np
import warnings
from transformers import AutoTokenizer
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
warnings.filterwarnings("ignore")

class TweetDataset:
    def __init__(self, tweet, targets):
        self.tweet = tweet
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, do_lower_case = args.do_lower_case)
        self.max_length = args.max_len
        self.targets = targets

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        
        tweet = str(self.tweet[item])
        tweet = " ".join(tweet.split())

        inputs = self.tokenizer.encode_plus(
            tweet,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation_strategy="longest_first",
            pad_to_max_length=True,
            truncation=True
        )
        
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
       
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[item], dtype=torch.float)
        }


if __name__ == "__main__":
    df = pd.read_csv(args.training_file).dropna().reset_index(drop = True)
    dset = TweetDataset(
        tweet=df.tweet.values,
        targets=df.target.values
        )
    print(df.iloc[1]['tweet'])
    #print(dset[1])
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, do_lower_case=args.do_lower_case)
    print(tokenizer.tokenize(df.iloc[1]['tweet']))
    print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(df.iloc[1]['tweet'])))