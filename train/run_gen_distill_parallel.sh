#!/bin/bash
# 并行生成自蒸馏数据脚本
# 使用 GPU 1 和 GPU 2，生成 50K 样本

DATE=$(date +%Y%m%d)
OUTPUT_DIR="../test"
BASE_NAME="pangu_self_distilled_${DATE}"

# 每个 GPU 生成 25K 样本
SAMPLES_PER_GPU=25000

echo "=============================================="
echo "并行自蒸馏数据生成"
echo "=============================================="
echo "日期: $DATE"
echo "输出目录: $OUTPUT_DIR"
echo "总样本数: $((SAMPLES_PER_GPU * 2))"
echo "GPU 1: 样本 0-$((SAMPLES_PER_GPU - 1))"
echo "GPU 2: 样本 $SAMPLES_PER_GPU-$((SAMPLES_PER_GPU * 2 - 1))"
echo "=============================================="

# 确保输出目录存在
mkdir -p $OUTPUT_DIR

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate medusa-train

# GPU 1: 生成前半部分 (0-24999)
echo ""
echo "[$(date '+%H:%M:%S')] 启动 GPU 1 (样本 0-$((SAMPLES_PER_GPU - 1)))..."
CUDA_VISIBLE_DEVICES=1 python gen_self_distill_data_fast.py \
    --model_path .. \
    --input_data ../third_party/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --output_data "${OUTPUT_DIR}/${BASE_NAME}_part0.json" \
    --num_samples $SAMPLES_PER_GPU \
    --start_idx 0 \
    --end_idx $SAMPLES_PER_GPU \
    --device cuda:0 \
    --max_new_tokens 512 \
    > "${OUTPUT_DIR}/gen_log_gpu1_${DATE}.log" 2>&1 &

PID1=$!
echo "GPU 1 PID: $PID1"

# GPU 2: 生成后半部分 (25000-49999)
echo "[$(date '+%H:%M:%S')] 启动 GPU 2 (样本 $SAMPLES_PER_GPU-$((SAMPLES_PER_GPU * 2 - 1)))..."
CUDA_VISIBLE_DEVICES=2 python gen_self_distill_data_fast.py \
    --model_path .. \
    --input_data ../third_party/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --output_data "${OUTPUT_DIR}/${BASE_NAME}_part1.json" \
    --num_samples $((SAMPLES_PER_GPU * 2)) \
    --start_idx $SAMPLES_PER_GPU \
    --end_idx $((SAMPLES_PER_GPU * 2)) \
    --device cuda:0 \
    --max_new_tokens 512 \
    > "${OUTPUT_DIR}/gen_log_gpu2_${DATE}.log" 2>&1 &

PID2=$!
echo "GPU 2 PID: $PID2"

echo ""
echo "=============================================="
echo "后台任务已启动!"
echo "=============================================="
echo ""
echo "监控命令:"
echo "  tail -f ${OUTPUT_DIR}/gen_log_gpu1_${DATE}.log"
echo "  tail -f ${OUTPUT_DIR}/gen_log_gpu2_${DATE}.log"
echo ""
echo "检查进度:"
echo "  ps aux | grep gen_self_distill"
echo ""
echo "等待完成..."

# 等待两个进程完成
wait $PID1
STATUS1=$?
echo "[$(date '+%H:%M:%S')] GPU 1 完成 (退出码: $STATUS1)"

wait $PID2
STATUS2=$?
echo "[$(date '+%H:%M:%S')] GPU 2 完成 (退出码: $STATUS2)"

# 合并结果
if [ $STATUS1 -eq 0 ] && [ $STATUS2 -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "合并数据文件..."
    echo "=============================================="
    
    python3 << EOF
import json

# 读取两部分数据
with open("${OUTPUT_DIR}/${BASE_NAME}_part0.json", 'r') as f:
    part0 = json.load(f)
with open("${OUTPUT_DIR}/${BASE_NAME}_part1.json", 'r') as f:
    part1 = json.load(f)

# 合并
merged = part0 + part1

# 保存合并后的数据
output_file = "${OUTPUT_DIR}/${BASE_NAME}_50k.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print(f"✅ 合并完成!")
print(f"   Part 0: {len(part0)} 样本")
print(f"   Part 1: {len(part1)} 样本")
print(f"   总计: {len(merged)} 样本")
print(f"   输出: {output_file}")

# 验证数据格式
sample = merged[0]
content = sample['conversations'][1]['value']
print(f"\n📋 样本格式验证:")
print(f"   包含 [unused16]: {'[unused16]' in content}")
print(f"   包含 [unused17]: {'[unused17]' in content}")
print(f"   包含 [unused10]: {'[unused10]' in content}")
EOF

    echo ""
    echo "=============================================="
    echo "✅ 全部完成!"
    echo "=============================================="
    echo "输出文件: ${OUTPUT_DIR}/${BASE_NAME}_50k.json"
    echo ""
else
    echo "❌ 生成过程中出现错误，请检查日志"
    echo "   GPU 1 日志: ${OUTPUT_DIR}/gen_log_gpu1_${DATE}.log"
    echo "   GPU 2 日志: ${OUTPUT_DIR}/gen_log_gpu2_${DATE}.log"
fi
