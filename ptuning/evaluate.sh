PRE_SEQ_LEN=128
CHECKPOINT=adgen-chatglm2-6b-pt-128-1e-4
STEP=1000
NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_predict \
    --validation_file ../data/val.json \
    --test_file ../data/test.json \
    --overwrite_cache \
    --prompt_column query \
    --response_column answer \
    --model_name_or_path /home/ubuntu/Documents/ai/model/chatglm2-6b \
    --ptuning_checkpoint ./output/$CHECKPOINT/checkpoint-$STEP \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 256 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 8
