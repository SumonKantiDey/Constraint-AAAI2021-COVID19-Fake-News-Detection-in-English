# # #last 2 hidden
# python train.py --roberta_hidden 768 \
#     --epochs 4 \
#     --learning_rate 3e-5 \
#     --max_len 128 \
#     --train_batch_size 16 \
#     --valid_batch_size 32 \
#     --do_lower_case \
#     --model_layer "last_two" \
#     --model_specification "train file train_batch 16 valid_batch 32 learning_rate 3e-5" \
#     --output "roberta_base_train_16_3e5.csv"

# python train.py --roberta_hidden 768 \
#     --epochs 4 \
#     --learning_rate 2e-5 \
#     --max_len 128 \
#     --train_batch_size 16 \
#     --valid_batch_size 32 \
#     --do_lower_case \
#     --model_layer "last_two" \
#     --model_specification "train file train_batch 16 valid_batch 32 learning_rate 2e-5" \
#     --output "roberta_base_train_16_2e5.csv"

python train.py --roberta_hidden 768 \
    --epochs 6 \
    --learning_rate 3e-5 \
    --max_len 128 \
    --train_batch_size 32 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_two" \
    --model_specification "train file train_batch 32 valid_batch 32 learning_rate 3e-5" \
    --output "roberta_base_train_32_3e5.csv" 

# python train.py --roberta_hidden 768 \
#     --epochs 4 \
#     --learning_rate 2e-5 \
#     --max_len 128 \
#     --train_batch_size 32 \
#     --valid_batch_size 32 \
#     --do_lower_case \
#     --model_layer "last_two" \
#     --model_specification "train file train_batch 32 valid_batch 32 learning_rate 2e-5" \
#     --output "roberta_base_train_32_2e5.csv" 

# #last 4 hidden

# python train.py --roberta_hidden 768 \
#     --epochs 4 \
#     --learning_rate 3e-5 \
#     --max_len 128 \
#     --train_batch_size 16 \
#     --valid_batch_size 32 \
#     --do_lower_case \
#     --model_layer "last_four" \
#     --model_specification "Last Four hidden Layer train_batch 16 valid_batch 32 learning_rate 3e-5" \
#     --output "roberta_basenext_train_16_3e5.csv"

# python train.py --roberta_hidden 768 \
#     --epochs 4 \
#     --learning_rate 2e-5 \
#     --max_len 128 \
#     --train_batch_size 16 \
#     --valid_batch_size 32 \
#     --do_lower_case \
#     --model_layer "last_four" \
#     --model_specification "Last Four hidden Layer train_batch 16 valid_batch 32 learning_rate 2e-5" \
#     --output "roberta_basenext_train_16_2e5.csv"

# python train.py --roberta_hidden 768 \
#     --epochs 4 \
#     --learning_rate 3e-5 \
#     --max_len 128 \
#     --train_batch_size 32 \
#     --valid_batch_size 32 \
#     --do_lower_case \
#     --model_layer "last_four" \
#     --model_specification "Last Four hidden Layer train_batch 32 valid_batch 32 learning_rate 3e-5" \
#     --output "roberta_basenext_train_32_3e5.csv"

# python train.py --roberta_hidden 768 \
#     --epochs 4 \
#     --learning_rate 2e-5 \
#     --max_len 128 \
#     --train_batch_size 32 \
#     --valid_batch_size 32 \
#     --do_lower_case \
#     --model_layer "last_four" \
#     --model_specification "Last Four hidden Layer train_batch 32 valid_batch 32 learning_rate 2e-5" \
#     --output "roberta_basenext_train_32_2e5.csv"