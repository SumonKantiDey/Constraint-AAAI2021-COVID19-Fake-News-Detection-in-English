#Bert Base Uncased (max_len=128, train_batch=16, test_batch=32, epochs = 4, lr=3e-5)
python train.py --bert_hidden 768 \
    --epochs 5 \
    --learning_rate 3e-5 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_two"\
    --model_specification "bert_base_two_train_16_3e5" \
    --output "bert_base_two_train_16_3e5.csv"

python train.py --bert_hidden 768 \
    --epochs 5 \
    --learning_rate 2e-5 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_two"\
    --model_specification "bert_base_two_train_16_2e5" \
    --output "bert_base_two_train_16_2e5.csv"