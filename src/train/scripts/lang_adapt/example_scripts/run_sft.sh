for lr in 1e-5; do
    for data_dir in translated_bespoke_stratos_17k; do

        pretrained_model=NousResearch/DeepHermes-3-Llama-3-8B-Preview
        irish_tokenizer_path=NousResearch/DeepHermes-3-Llama-3-8B-Preview
        # pretrained_model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
        # irish_tokenizer_path=deepseek-ai/DeepSeek-R1-Distill-Llama-8B

        output_dir=output_model/irish_deephermes_llama31_8b_${data_dir}
        # output_dir=output_model/irish_deepseek_llama31_8b_${data_dir}
        deepspeed_config_file=./scripts/lang_adapt/example_scripts/ds_zero2_offload.json

        dataset_dir=${data_dir}
        data_cache=./cache
        cache_dir=./cache
        per_device_train_batch_size=1
        per_device_eval_batch_size=1
        gradient_accumulation_steps=12

        lora_rank=16
        lora_alpha=32
        lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
        modules_to_save="embed_tokens,lm_head"
        lora_dropout=0.0
        peft_method="lora"

        DS_SKIP_CUDA_CHECK=1 CUDA_VISIBLE_DEVICES="5" torchrun --nnodes 1 --nproc_per_node 1 --master_port 29502 ./scripts/lang_adapt/run_sft.py \
            --deepspeed ${deepspeed_config_file} \
            --model_name_or_path ${pretrained_model} \
            --tokenizer_name_or_path ${irish_tokenizer_path} \
            --dataset_dir ${dataset_dir} \
            --data_cache_dir ${data_cache} \
            --cache_dir ${cache_dir} \
            --validation_split_percentage 0.001 \
            --per_device_train_batch_size ${per_device_train_batch_size} \
            --per_device_eval_batch_size ${per_device_eval_batch_size} \
            --do_train \
            --seed 24259 \
            --bf16 \
            --num_train_epochs 3 \
            --lr_scheduler_type cosine \
            --learning_rate ${lr} \
            --warmup_ratio 0.05 \
            --weight_decay 0.01 \
            --logging_strategy steps \
            --logging_steps 10 \
            --save_strategy epoch \
            --save_total_limit 10 \
            --gradient_accumulation_steps ${gradient_accumulation_steps} \
            --preprocessing_num_workers 8 \
            --max_seq_length 16384 \
            --output_dir ${output_dir} \
            --ddp_timeout 30000 \
            --logging_first_step True \
            --torch_dtype bfloat16 \
            --gradient_checkpointing \
            --ddp_find_unused_parameters False \
            --overwrite_output_dir \
            --use_peft False \
            --lora_rank ${lora_rank} \
            --lora_alpha ${lora_alpha} \
            --trainable ${lora_trainable} \
            --method ${peft_method} \
            --modules_to_save ${modules_to_save} \
            --lora_dropout ${lora_dropout} \
            --use_liger True \
            --packing False \
            --langdeplay None \
            --optim paged_adamw_32bit \
            --reasoning_collator True \
            --log_extra_losses True \
            --weight_before_think_loss 1.0 \
            --weight_after_think_loss 1.0
    done
done
