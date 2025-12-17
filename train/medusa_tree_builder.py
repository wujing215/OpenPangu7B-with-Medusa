#!/usr/bin/env python3
"""
Medusa Tree 构建工具 - 基于贪婪搜索优化 medusa_choices

原理：
1. 对验证集样本，用训练好的 Medusa 模型生成
2. 记录每个 head 的 Top-K token 概率
3. 使用贪婪算法构建最优树结构（最大化接受率）

参考论文：Medusa: Simple LLM Inference Acceleration Framework
"""

import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from medusa_model import MedusaModelPangu
from transformers import AutoTokenizer


class MedusaTreeBuilder:
    """构建最优 Medusa Tree"""
    
    def __init__(self, model_path: str, medusa_head_path: str, num_heads: int, 
                 device: str = "cuda:0", top_k: int = 10):
        """
        Args:
            model_path: 基础模型路径
            medusa_head_path: Medusa head 权重路径
            num_heads: Medusa head 数量
            device: 设备
            top_k: 每个 head 保留的 top-k 候选
        """
        self.device = device
        self.top_k = top_k
        self.num_heads = num_heads
        
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        
        # 加载基础模型配置并添加 medusa 参数
        from transformers import AutoConfig
        from medusa_model import MedusaConfig
        
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.medusa_num_heads = num_heads
        config.medusa_num_layers = 1
        
        # 使用修改后的配置加载模型
        print(f"Loading Medusa model with {num_heads} heads...")
        self.model = MedusaModelPangu.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        
        # 加载训练好的 Medusa heads
        print(f"Loading Medusa heads from {medusa_head_path}...")
        from safetensors.torch import load_file
        medusa_weights = load_file(medusa_head_path)
        
        # 添加 'medusa_head.' 前缀(训练时保存的键没有前缀)
        medusa_weights_with_prefix = {
            f'medusa_head.{k}': v for k, v in medusa_weights.items()
        }
        
        missing_keys, unexpected_keys = self.model.load_state_dict(medusa_weights_with_prefix, strict=False)
        print(f"  Loaded {len(medusa_weights)} medusa head parameters")
        if unexpected_keys:
            print(f"  Warning: Unexpected keys: {unexpected_keys[:3]}...")
        
        self.model.eval()
        
        print("Model loaded successfully!")
        
    def collect_statistics(self, prompts: List[str], max_new_tokens: int = 256) -> Dict:
        """收集 Medusa heads 的预测统计"""
        stats = {
            'head_accuracy': [[] for _ in range(self.num_heads)],  # 每个 head 的准确率
            'head_top_k_hit': [[] for _ in range(self.num_heads)],  # Top-K 命中率
            'position_patterns': defaultdict(lambda: defaultdict(int)),  # 位置模式
        }
        
        print(f"Collecting statistics on {len(prompts)} prompts...")
        
        for prompt in tqdm(prompts):
            # 准备输入
            messages = [
                {"role": "system", "content": "你是一个有帮助的助手。"},
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # 贪婪解码，记录每步的 ground truth
                outputs = self.model.base_model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=45892,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                
                ground_truth_ids = outputs.sequences[0, input_ids.shape[1]:].cpu().tolist()
                
                # 逐步预测，收集 Medusa heads 的表现
                current_ids = input_ids
                for step_idx in range(min(len(ground_truth_ids) - self.num_heads, max_new_tokens)):
                        # Forward pass 获取 Medusa logits
                    medusa_logits, outputs, _ = self.model(
                        current_ids, 
                        output_orig=True, 
                        medusa_forward=True
                    )
                    # medusa_logits shape: [num_heads, batch, seq_len, vocab]
                    
                    # 检查每个 head 的预测
                    for head_idx in range(self.num_heads):
                        # 当前 head 预测的是未来第 (head_idx+1) 个 token
                        if step_idx + head_idx + 1 >= len(ground_truth_ids):
                            break
                        
                        gt_token = ground_truth_ids[step_idx + head_idx + 1]
                        head_logits = medusa_logits[head_idx, 0, -1, :]  # 最后一个位置的预测
                        
                        # Top-1 准确率
                        pred_token = head_logits.argmax().item()
                        is_correct = (pred_token == gt_token)
                        stats['head_accuracy'][head_idx].append(float(is_correct))
                        
                        # Top-K 命中率
                        top_k_tokens = head_logits.topk(self.top_k).indices.cpu().tolist()
                        is_in_top_k = (gt_token in top_k_tokens)
                        stats['head_top_k_hit'][head_idx].append(float(is_in_top_k))
                    
                    # 移动到下一个 token
                    next_token = torch.tensor([[ground_truth_ids[step_idx]]], device=self.device)
                    current_ids = torch.cat([current_ids, next_token], dim=1)
        
        return stats
    
    def build_tree_greedy(self, stats: Dict, max_candidates: int = 64) -> List[List[int]]:
        """基于统计数据贪婪构建最优树
        
        返回 medusa_choices 格式：[[0], [0, 0], [0, 1], [1], ...]
        """
        print("\nBuilding Medusa tree greedily...")
        
        # 计算每个 head 的平均 Top-K 命中率
        head_hit_rates = []
        for head_idx in range(self.num_heads):
            hit_rate = np.mean(stats['head_top_k_hit'][head_idx])
            head_hit_rates.append(hit_rate)
            print(f"  Head {head_idx}: Top-{self.top_k} hit rate = {hit_rate:.3f}")
        
        # 贪婪策略：优先扩展命中率高的 head
        medusa_choices = []
        
        # 第一层：所有 head 的直接预测
        for head_idx in range(self.num_heads):
            medusa_choices.append([head_idx])
        
        # 逐层扩展
        current_layer = [[i] for i in range(self.num_heads)]
        
        while len(medusa_choices) < max_candidates:
            next_layer = []
            
            # 为当前层的每个路径尝试扩展
            for path in current_layer:
                last_head = path[-1]
                
                # 尝试扩展到下一个 head
                for next_head in range(self.num_heads):
                    new_path = path + [next_head]
                    
                    # 计算该路径的期望命中率（各 head 命中率的乘积）
                    expected_hit = 1.0
                    for i, h in enumerate(new_path):
                        expected_hit *= head_hit_rates[h]
                    
                    # 如果命中率足够高，加入候选
                    if expected_hit > 0.01:  # 阈值可调
                        next_layer.append((new_path, expected_hit))
            
            # 按期望命中率排序，保留 top 候选
            next_layer.sort(key=lambda x: x[1], reverse=True)
            
            for path, score in next_layer:
                if len(medusa_choices) >= max_candidates:
                    break
                medusa_choices.append(path)
            
            # 更新当前层
            current_layer = [path for path, _ in next_layer[:max_candidates // 2]]
            
            if not current_layer:
                break
        
        # 按长度和索引排序（Medusa 要求）
        medusa_choices.sort(key=lambda x: (len(x), x))
        
        return medusa_choices
    
    def save_tree_config(self, medusa_choices: List[List[int]], output_path: str, 
                         stats: Dict, tree_name: str = "pangu_optimized"):
        """保存树配置到 Python 文件"""
        with open(output_path, 'w') as f:
            f.write(f'''# Medusa Tree Configuration: {tree_name}
# Auto-generated by medusa_tree_builder.py
# Based on {len(stats['head_accuracy'][0])} validation samples

# Statistics:
''')
            for head_idx in range(self.num_heads):
                acc = np.mean(stats['head_accuracy'][head_idx])
                hit = np.mean(stats['head_top_k_hit'][head_idx])
                f.write(f'#   Head {head_idx}: Accuracy={acc:.4f}, Top-{self.top_k} Hit={hit:.4f}\n')
            
            f.write(f'''
{tree_name} = {medusa_choices}

# Tree info:
#   Total candidates: {len(medusa_choices)}
#   Max depth: {max(len(path) for path in medusa_choices)}
#   Avg depth: {np.mean([len(path) for path in medusa_choices]):.2f}
''')
        
        print(f"\nTree configuration saved to: {output_path}")
        print(f"  Total candidates: {len(medusa_choices)}")
        print(f"  Max depth: {max(len(path) for path in medusa_choices)}")


def main():
    parser = argparse.ArgumentParser(description="Build optimal Medusa tree")
    parser.add_argument("--model_path", type=str, default=".", help="Base model path")
    parser.add_argument("--medusa_head", type=str, required=True, 
                        help="Medusa head checkpoint path (medusa_lm_head.safetensors)")
    parser.add_argument("--num_heads", type=int, default=5, help="Number of Medusa heads")
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Validation data (JSON with prompts)")
    parser.add_argument("--num_samples", type=int, default=1000, 
                        help="Number of samples to use for statistics")
    parser.add_argument("--max_new_tokens", type=int, default=128, 
                        help="Max tokens to generate per sample")
    parser.add_argument("--top_k", type=int, default=10, 
                        help="Top-K candidates per head")
    parser.add_argument("--max_candidates", type=int, default=64, 
                        help="Maximum tree candidates")
    parser.add_argument("--output", type=str, default="medusa_tree_optimized.py", 
                        help="Output tree configuration file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    args = parser.parse_args()
    
    # 加载验证数据
    print(f"Loading validation data from {args.data_path}...")
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    
    # 提取 prompts
    prompts = []
    for item in data[:args.num_samples]:
        if "conversations" in item:
            for conv in item["conversations"]:
                if conv.get("from") == "human":
                    prompts.append(conv["value"])
                    break
    
    print(f"Collected {len(prompts)} prompts")
    
    # 构建树
    builder = MedusaTreeBuilder(
        args.model_path, 
        args.medusa_head, 
        args.num_heads, 
        args.device, 
        args.top_k
    )
    
    stats = builder.collect_statistics(prompts, args.max_new_tokens)
    medusa_choices = builder.build_tree_greedy(stats, args.max_candidates)
    
    tree_name = f"pangu_{args.num_heads}heads_top{args.top_k}"
    builder.save_tree_config(medusa_choices, args.output, stats, tree_name)
    
    print("\n✅ Done! Use this tree in inference:")
    print(f"   from {Path(args.output).stem} import {tree_name}")
    print(f"   model.generate(..., medusa_choices={tree_name})")


if __name__ == "__main__":
    main()
