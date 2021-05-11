import transformers

from flag import get_parser

parser = get_parser()
args = parser.parse_args()
# MAX_LEN = 128
# TRAIN_BATCH_SIZE = 32
# VALID_BATCH_SIZE = 32
# EPOCHS = 3
# ROBERTA_PATH = "/content/drive/My Drive/pre_trained_model/roberta-base/"
# MODEL_PATH = "../store_model/roberta_model.bin"
# TRAINING_FILE = "../input/train_final.csv"   
# TESTING_FILE = "../input/test_final.csv"
# TOKENIZER = transformers.RobertaTokenizer(
#     vocab_file=f"{ROBERTA_PATH}/vocab.json", 
#     merges_file=f"{ROBERTA_PATH}/merges.txt", 
#     lowercase=args.do_lower_case
#     )