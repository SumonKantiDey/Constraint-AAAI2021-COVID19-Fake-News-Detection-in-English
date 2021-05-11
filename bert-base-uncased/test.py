import config
import utils
import pandas as pd 
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
#print(config.TOKENIZER.tokenize("The CDC currently reports 99031 deaths"))

from flag import get_parser

parser = get_parser()
args = parser.parse_args()
print(args)