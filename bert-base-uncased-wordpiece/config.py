import transformers
import tokenizers
SEED = 42
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 5
BERT_PATH = "/content/drive/My Drive/pre_trained_model/bert-base-uncased/"
MODEL_PATH = "../store_model/model.bin"
TRAINING_FILE = "../input/train_final.csv"   
TESTING_FILE = "../input/test_final.csv"
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    f"{BERT_PATH}/vocab.txt",
    lowercase=True
)