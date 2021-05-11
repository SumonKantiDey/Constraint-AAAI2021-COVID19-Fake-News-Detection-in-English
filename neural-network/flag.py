import argparse

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--epochs", default=15, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument("-train_batch", "--train_batch_size", default=32, type=int, metavar='N', help='train-mini-batch size')
    parser.add_argument("-valid_batch", "--valid_batch_size", default=32, type=int, metavar='N', help='valid-mini-batch size')
    parser.add_argument("-mf", "--max_features", default=120000, type=int, metavar='N', help='how many unique words to use (i.e num rows in embedding vector)')
    parser.add_argument("-ml", "--max_len", default=120, type=int, metavar='N', help='max number of words in a question to use')
    parser.add_argument("-es","--embed_size", default=300, type=int, metavar='N', help='embedding size')
    
    parser.add_argument("-lr", "--learning_rate", default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument("-wd","--weight_decay", default=1e-4, type=float, metavar='W', help='weight decay')
    
    parser.add_argument("-hs", "--hidden_size", default=64, type=int, metavar='N', help='rnn hidden size')
    parser.add_argument("-rs", "--reduction_size", default=16, type=int, metavar='N', help='rnn reduction size')
    parser.add_argument("-nl", "--num_layers", default=1, type=int, metavar='N', help='number of rnn layers')
    parser.add_argument("--classes", default=1, type=int, metavar='N', help='number of output classes')
    parser.add_argument("--glove", default='glove/glove.6B.100d.txt', help='path to glove txt')
    parser.add_argument("--train", default='../input/train_final.csv',type=str, help='path to train file')
    parser.add_argument("--test", default='../input/test_final.csv', type=str,help='path to test file')
    parser.add_argument("-lm", "--lstm", type=str, default="lstm",help="Model to be used for lstm")
    parser.add_argument("-gr", "--gru", type=str, default="gru",help="Model to be used for gru")
    parser.add_argument("-do", "--dropout", type=float, default=0.1,help="Dropout")
    parser.add_argument("-s", "--save_log", type=str, default='./save',help="Save folder")

    parser.add_argument("--output", default='pred.csv',type=str, help='Path to output file')
    parser.add_argument("--model_specification", default="Single LSTM", required=False, help="model name")
    parser.add_argument("--model_name", default="lstm_attention", required=False, help="model name")

    parser.add_argument("-m_path", "--model_path", type=str, default='../store_model/lstm.bin',help="save best model")
    parser.add_argument("--seed", type=int, default=42,help="Seed for reproducibility")
    parser.add_argument("--clip", type=float, default=0.25, help='gradient clipping')

    #args = parser.parse_args()
    return parser

