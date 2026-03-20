## test分支相较于主分支添加了debug信息打印
### 运行benchmark:
```
CUDA_VISIBLE_DEVICES=4 nohup python inference/benchmark.py     --base_model .      --medusa_dir medusa_heads_5      --prompt "Give me a short introduction to LLM." > log/benchmark.log 2>&1 &
```

### 构建优化树文件位于`train/medusa_tree_builder.py`
构建优化树：
```
CUDA_VISIBLE_DEVICES=4 nohup python train/medusa_tree_builder.py \
    --model_path . \
    --medusa_head medusa_heads_5/medusa_lm_head.safetensors \
    --num_heads 5 \
    --data_path test/pangu_self_distilled_20251223_50k.json \
    --num_samples 1000 \
    --max_seq_len 256 \
    --top_k 10 \    # 此处修改TOPK个数
    --max_candidates 32 \   # 此处修改叶子节点个数
    --output tree_res.py \
    > log/tree_build.log 2>&1 &
```
