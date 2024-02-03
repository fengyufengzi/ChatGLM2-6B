PRE_SEQ_LEN=128  #128,
LR=1e-4  #
NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_eval \
    --do_predict \
    --train_file ../data/train.json \
    --validation_file ../data/val.json \
    --test_file ../data/val_less.json \
    --preprocessing_num_workers 1 \
    --prompt_column query \
    --response_column answer \
    --overwrite_cache \
    --model_name_or_path /data/ai/model/chatglm2-6b \
    --output_dir output/adgen-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 512 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 2100 \
    --logging_steps 10 \
    --save_steps 700 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit=8 \
    --report_to wandb \
    --max_eval_samples 20 \
    --ptuning_checkpoint  /data/ai/ChatGLM2-6B/ptuning/output/adgen-chatglm2-6b-pt-128-1e-4/checkpoint-2100 \
    --source_prefix "你是蚂蚁金服公司的客服人员,用清晰简洁的语言回答老年人的问题.如果因为知识库受限无法回答,只需回答'对不起，根据参考资料无法回答'

    "
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
#    --source_prefix [支付宝专用]请用最简洁的语言回答如下问题: \




