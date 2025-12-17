#!/bin/bash
# Medusa 5-head 训练脚本

# ================= 配置 =================
MODEL_PATH=".."  # OpenPangu 模型路径(相对于train目录)
TRAIN_DATA="../pangu_distilled_10k.json"  # 自蒸馏数据(在根目录)
OUTPUT_DIR="../medusa_5heads_lr0.001_layers1"  # 输出到根目录
NUM_HEADS=5  # 论文推荐 5 个 heads
NUM_LAYERS=1  # 每个 head 的层数

# 训练参数
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
NUM_EPOCHS=5
LEARNING_RATE=1e-3
SAVE_STEPS=1000

# GPU 设置 (使用2个GPU)
export CUDA_VISIBLE_DEVICES=2,5

# ================= 训练 =================
echo "========================================"
echo "Medusa 5-Head Training (Self-Distilled)"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Data: $TRAIN_DATA"
echo "Heads: $NUM_HEADS"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# 切换到 train 目录
cd "$(dirname "$0")"

nohup torchrun --nproc_per_node=2 --master_port=29501 train_medusa.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$TRAIN_DATA" \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 3 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --medusa_num_heads $NUM_HEADS \
    --medusa_num_layers $NUM_LAYERS \
    --deepspeed ../deepspeed.json \
    --report_to none > ../1210_train.log 2>&1 & echo $! > ../1210_train.pid

echo "========================================"
echo "Training completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "========================================"
