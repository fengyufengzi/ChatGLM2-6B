PRE_SEQ_LEN=128
CHECKPOINT=adgen-chatglm2-6b-pt-128-1e-4
STEP=1400
NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_predict \
    --validation_file ../data/val.json \
    --test_file ../data/val_less.json \
    --overwrite_cache \
    --prompt_column query \
    --response_column answer \
    --model_name_or_path /data/ai/model/ChatGLM3/chatglm3-6b \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 128 \
    --max_target_length 256 \
    --per_device_eval_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit=8 \
    --report_to wandb \
#    --source_prefix "你是专业客服人员，请用最简洁的语言回答用户的问题：
#
#
#    问题：湖北高考成绩会发短信吗？
#    回答：不会。可自行在网上查询。
#    问题：长沙在读大学生落户有补贴吗？
#    回答：长沙在读大学生落户没有补贴，本科及以上大学生毕业之后落户长沙且在长沙工作符合相关条件的可以申领租房和生活补贴。
#    问题：青岛市老年人高龄补贴标准是多少？
#    回答：对不起，根据参考资料无法回答。
#
#
#    问题：" \
#    --ptuning_checkpoint ./output/$CHECKPOINT/checkpoint-$STEP \