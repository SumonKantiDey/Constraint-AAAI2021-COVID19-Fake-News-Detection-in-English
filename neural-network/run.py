python train.py --epochs 15 \
    --train_batch_size 32 \
    --valid_batch_size 32 \
    --model_specification "Vanilla LSTM train_batch 32 valid_batch 32" \
    --model_name "lstm_classifier" \
    --output "vanilla_lstm32.csv"

python train.py --epochs 15 \
    --train_batch_size 32 \
    --valid_batch_size 32 \
    --model_specification "Attention  LSTM train_batch 32 valid_batch 32" \
    --model_name "lstm_attention" \
    --output "attention_lstm32.csv"