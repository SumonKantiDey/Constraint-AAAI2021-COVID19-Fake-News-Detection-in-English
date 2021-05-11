#RobertaLarge[Last Four Hidden] (max_len=128, train_batch=16, test_batch=32, epochs = 4, lr = 2e-5)
python train.py --roberta_hidden 1024 \
    --epochs 4 \
    --learning_rate 3e-5 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_two" \
    --model_specification "roerta_large_two_train_16_3e5" \
    --output "roerta_large_two_train_16_3e5.csv"

python train.py --roberta_hidden 1024 \
    --epochs 4 \
    --learning_rate 3e-5 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_four" \
    --model_specification "roberta_large_four_train_16_3e5" \
    --output "roberta_large_four_train_16_3e5.csv"

python train.py --roberta_hidden 1024 \
    --epochs 4 \
    --learning_rate 2e-5 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_four" \
    --model_specification "roberta_large_four_train_16_2e5" \
    --output "roberta_large_four_train_16_2e5.csv"
