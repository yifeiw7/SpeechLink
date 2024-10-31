# Finetuning 


## Preprocessing

Example usage:

`/usr/local/bin/python3 "/Users/elizzy/Desktop/Speechlink/finetune/preprocessing/data_prep.py"
 --source_data_file "/Users/elizzy/Desktop/Speechlink/data/train_data.json"
 --output_data_dir "/Users/elizzy/Desktop/Speechlink/data/prep_train"
 --base_audio_dir "/Users/elizzy/Desktop/Speechlink/data"`

## Training

Example usage:

`/usr/local/bin/python3 /Users/elizzy/Desktop/Speechlink/finetune/train/fine-tune_on_custom_dataset.py \
    --model_name "openai/whisper-tiny" \
    --language "English" \
    --sampling_rate 16000 \
    --num_proc 2 \
    --train_strategy epoch \
    --learning_rate 3e-3 \
    --warmup 1000 \
    --train_batchsize 16 \
    --eval_batchsize 8 \
    --num_epochs 10 \
    --resume_from_ckpt None \
    --output_dir /Users/elizzy/Desktop/Speechlink/finetune/train/checkpoints \
    --train_datasets /Users/elizzy/Desktop/Speechlink/data/prep_train \
    --eval_datasets /Users/elizzy/Desktop/Speechlink/data/prep_eval`

    
