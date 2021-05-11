## Bert Large last two hidden
python train.py --bert_hidden 1024 \
    --epochs 5 \
    --learning_rate 2e-5 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_two"\
    --model_specification "bert_large_two_train_16_2e5" \
    --output "bert_large_two_train_16_2e5.csv"

python train.py --bert_hidden 1024 \
    --epochs 5 \
    --learning_rate 3e-5 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_two"\
    --model_specification "bert_large_two_train_16_3e5" \
    --output "bert_large_two_train_16_3e5.csv"

python train.py --bert_hidden 1024 \
    --epochs 5 \
    --learning_rate 2e-5 \
    --max_len 128 \
    --train_batch_size 32 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_two"\
    --model_specification "bert_large_two_train_32_2e5" \
    --output "bert_large_two_train_32_2e5.csv" 


python train.py --bert_hidden 1024 \
    --epochs 5 \
    --learning_rate 3e-5 \
    --max_len 128 \
    --train_batch_size 32 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_two"\
    --model_specification "bert_large_two_train_32_3e5" \
    --output "bert_large_two_train_32_3e5.csv" 


## Bert Large last four hidden
python train.py --bert_hidden 1024 \
    --epochs 5 \
    --learning_rate 2e-5 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_four"\
    --model_specification "bert_large_four_train_16_2e5" \
    --output "bert_large_four_train_16_2e5.csv"

python train.py --bert_hidden 1024 \
    --epochs 5 \
    --learning_rate 3e-5 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_four"\
    --model_specification "bert_large_four_train_16_3e5" \
    --output "bert_large_four_train_16_3e5.csv"

python train.py --bert_hidden 1024 \
    --epochs 5 \
    --learning_rate 2e-5 \
    --max_len 128 \
    --train_batch_size 32 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_four"\
    --model_specification "bert_large_four_train_32_2e5" \
    --output "bert_large_four_train_32_2e5.csv" 

python train.py --bert_hidden 1024 \
    --epochs 5 \
    --learning_rate 3e-5 \
    --max_len 128 \
    --train_batch_size 32 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_four"\
    --model_specification "bert_large_four_train_32_3e5" \
    --output "bert_large_four_train_32_3e5.csv" 