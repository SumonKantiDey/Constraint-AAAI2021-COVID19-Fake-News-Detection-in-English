############################ Last Two Layer ###########################

## Covid twitter Bert
python train.py --bert_hidden 1024 \
    --pretrained_model_name "digitalepidemiologylab/covid-twitter-bert-v2" \
    --learning_rate 2e-5 \
    --epochs 4 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_two" \
    --model_specification "covid_bert_two_train_16_2e5" \
    --output "covid_bert_two_train_16_2e5.csv"
    
python train.py --bert_hidden 1024 \
    --pretrained_model_name "digitalepidemiologylab/covid-twitter-bert-v2" \
    --learning_rate 3e-5 \
    --epochs 4 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_two" \
    --model_specification "covid_bert_two_train_16_3e5" \
    --output "covid_bert_two_train_16_3e5.csv"


## Bio Bert
python train.py --bert_hidden 768 \
    --pretrained_model_name "seiya/oubiobert-base-uncased" \
    --learning_rate 2e-5 \
    --epochs 8 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_two" \
    --model_specification "bio_bert_two_train_16_2e5" \
    --output "bio_bert_two_train_16_2e5.csv"

python train.py --bert_hidden 768 \
    --pretrained_model_name "seiya/oubiobert-base-uncased" \
    --learning_rate 3e-5 \
    --epochs 8 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_two" \
    --model_specification "bio_bert_two_train_16_3e5" \
    --output "bio_bert_two_train_16_3e5.csv"

## Sci Bert
python train.py --bert_hidden 768 \
    --pretrained_model_name "allenai/scibert_scivocab_uncased" \
    --learning_rate 2e-5 \
    --epochs 4 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_two" \
    --model_specification "sci_bert_two_train_16_2e5" \
    --output "sci_bert_two_train_16_2e5.csv"

python train.py --bert_hidden 768 \
    --pretrained_model_name "allenai/scibert_scivocab_uncased" \
    --learning_rate 3e-5 \
    --epochs 4 \
    --max_len 128 \
    --train_batch_size 32 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_two" \
    --model_specification "sci_bert_two_train_32_3e5" \
    --output "sci_bert_two_train_32_3e5.csv"

################################# Last Four Layer ###########################
## Covid twitter Bert
python train.py --bert_hidden 1024 \
    --pretrained_model_name "digitalepidemiologylab/covid-twitter-bert-v2" \
    --learning_rate 2e-5 \
    --epochs 4 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_four" \
    --model_specification "covid_bert_four_train_16_2e5" \
    --output "covid_bert_four_train_16_2e5.csv"
    
python train.py --bert_hidden 1024 \
    --pretrained_model_name "digitalepidemiologylab/covid-twitter-bert-v2" \
    --learning_rate 3e-5 \
    --epochs 4 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_four" \
    --model_specification "covid_bert_four_train_16_3e5" \
    --output "covid_bert_four_train_16_3e5.csv"


## Bio Bert
python train.py --bert_hidden 768 \
    --pretrained_model_name "seiya/oubiobert-base-uncased" \
    --learning_rate 2e-5 \
    --epochs 8 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_four" \
    --model_specification "bio_bert_four_train_16_2e5" \
    --output "bio_bert_four_train_16_2e5.csv"

python train.py --bert_hidden 768 \
    --pretrained_model_name "seiya/oubiobert-base-uncased" \
    --learning_rate 3e-5 \
    --epochs 8 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_four" \
    --model_specification "bio_bert_four_train_16_3e5" \
    --output "bio_bert_four_train_16_3e5.csv"

## Sci Bert
python train.py --bert_hidden 768 \
    --pretrained_model_name "allenai/scibert_scivocab_uncased" \
    --learning_rate 2e-5 \
    --epochs 4 \
    --max_len 128 \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_four" \
    --model_specification "sci_bert_four_train_16_2e5" \
    --output "sci_bert_four_train_16_2e5.csv"

python train.py --bert_hidden 768 \
    --pretrained_model_name "allenai/scibert_scivocab_uncased" \
    --learning_rate 3e-5 \
    --epochs 4 \
    --max_len 128 \
    --train_batch_size 32 \
    --valid_batch_size 32 \
    --do_lower_case \
    --model_layer "last_four" \
    --model_specification "sci_bert_four_train_32_3e5" \
    --output "sci_bert_four_train_32_3e5.csv"