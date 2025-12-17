#!/bin/bash

# ================= 配置 =================
TOTAL_SAMPLES=10000  # 方案1：10k 样本
MAX_NEW_TOKENS=512    # 优化后的长度
GPU_LIST=(2 3)        # 使用 GPU 2 和 3
NUM_GPUS=${#GPU_LIST[@]}

# 计算每个 GPU 处理的样本数
SAMPLES_PER_GPU=$((TOTAL_SAMPLES / NUM_GPUS))

echo "=========================================="
echo "快速自蒸馏（双 GPU 并行）"
echo "=========================================="
echo "总样本数: $TOTAL_SAMPLES"
echo "Max tokens: $MAX_NEW_TOKENS"
echo "使用 GPUs: ${GPU_LIST[@]}"
echo "每GPU样本: $SAMPLES_PER_GPU"
echo "预计时间: 5-7 小时"
echo "=========================================="

# 清理旧文件
rm -f pangu_distilled_10k_part_*.json

# 启动多个进程
for i in "${!GPU_LIST[@]}"; do
    GPU_ID=${GPU_LIST[$i]}
    START_IDX=$((i * SAMPLES_PER_GPU))
    END_IDX=$(((i + 1) * SAMPLES_PER_GPU))
    OUTPUT_FILE="pangu_distilled_10k_part_${i}.json"
    LOG_FILE="gen_10k_gpu${GPU_ID}.log"
    
    echo "启动 GPU $GPU_ID: 样本 $START_IDX 到 $END_IDX"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python train/gen_self_distill_data_fast.py \
        --start_idx $START_IDX \
        --end_idx $END_IDX \
        --output_data $OUTPUT_FILE \
        --max_new_tokens $MAX_NEW_TOKENS \
        > $LOG_FILE 2>&1 &
    
    PID=$!
    echo "  进程 PID: $PID (日志: $LOG_FILE)"
    sleep 2
done

echo ""
echo "监控进度："
echo "  tail -f gen_10k_gpu2.log"
echo "  tail -f gen_10k_gpu3.log"
echo ""
echo "查看状态："
echo "  ps aux | grep gen_self_distill_data_fast"
echo ""
echo "完成后合并结果："
echo "  python train/merge_distilled_data.py --pattern 'pangu_distilled_10k_part_*.json' --output pangu_distilled_10k.json"
echo ""
echo "预计完成时间: $(date -d '+6 hours' '+%Y-%m-%d %H:%M:%S')"
