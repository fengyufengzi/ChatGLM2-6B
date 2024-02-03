PRE_SEQ_LEN=128  #128,
LR=1e-4
NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_predict \
    --train_file ../data/train_less.json \
    --validation_file ../data/val_less.json \
    --test_file ../data/val_less.json \
    --preprocessing_num_workers 1 \
    --prompt_column query \
    --response_column answer \
    --overwrite_cache \
    --model_name_or_path /home/ubuntu/Documents/ai/model/chatglm2-6b \
    --output_dir output/adgen-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR-2 \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 256 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 10 \
    --logging_steps 1 \
    --save_steps 10 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit=8 \
    --report_to wandb \
    --source_prefix [支付宝专用]请用最简洁的语言回答如下问题: \
    #
    #


