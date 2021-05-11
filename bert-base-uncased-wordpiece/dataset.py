import config
import utils
import pandas as pd 
import torch
import numpy as np
import warnings

np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
warnings.filterwarnings("ignore")

class TweetDataset:
    def __init__(self, tweet, targets):
        self.tweet = tweet
        self.tokenizer = config.TOKENIZER
        self.max_length = config.MAX_LEN
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
        token_type_ids = inputs["token_type_ids"]
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[item], dtype=torch.float)
        }


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE).dropna().reset_index(drop = True)
    dset = TweetDataset(
        tweet=df.tweet.values,
        targets=df.target.values
        )
    print(df.iloc[1])
    print(dset[1])
    print(config.TOKENIZER.tokenize(df.iloc[1]['clean_tweet']))