import config
import utils
import pandas as pd 
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
print(config.TOKENIZER.tokenize("The CDC currently reports 99031 deaths"))