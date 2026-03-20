#!/usr/bin/env python3
"""
Medusa Tree 构建工具 - 优化版本（快速）

优化点：
1. 单次 forward pass 收集所有 head 的统计信息（不逐 token 生成）
2. 使用已有的 ground truth 数据（从蒸馏数据集）
3. 并行计算多个位置的统计

原理：
1. 对验证集样本，直接用完整的 input+output 做一次 forward
2. 对每个位置，检查 Medusa head 的预测是否匹配下一个 token
3. 使用贪婪算法构建最优树结构

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


class MedusaTreeBuilderFast:
    """构建最优 Medusa Tree - 优化版本"""
    
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
    
    def collect_statistics_fast(self, data: List[Dict], max_samples: int = 1000, 
                                 max_seq_len: int = 2048) -> Dict:
        """
        快速收集 Medusa heads 的预测统计
        
        核心优化：对每个样本只做一次 forward pass，同时评估所有位置的预测
        """
        stats = {
            'head_accuracy': [[] for _ in range(self.num_heads)],  # 每个 head 的准确率
            'head_top_k_hit': [[] for _ in range(self.num_heads)],  # Top-K 命中率
            'joint_accuracy': defaultdict(list),  # 多 head 联合准确率
        }
        
        print(f"Collecting statistics on {min(len(data), max_samples)} samples (fast mode)...")
        
        processed = 0
        for item in tqdm(data[:max_samples]):
            try:
                # 从蒸馏数据中提取 prompt 和 response
                if "conversations" in item:
                    prompt = ""
                    response = ""
                    for conv in item["conversations"]:
                        if conv.get("from") == "human":
                            prompt = conv["value"]
                        elif conv.get("from") == "gpt":
                            response = conv["value"]
                    
                    if not prompt or not response:
                        continue
                else:
                    continue
                
                # 构建完整的对话（包含 prompt 和 response）
                messages = [
                    {"role": "system", "content": "你是一个有帮助的助手。"},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
                
                # Tokenize 完整对话
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                
                input_ids = self.tokenizer.encode(text, return_tensors="pt", 
                                                   truncation=True, 
                                                   max_length=max_seq_len).to(self.device)
                
                seq_len = input_ids.shape[1]
                if seq_len < self.num_heads + 10:  # 太短的序列跳过
                    continue
                
                with torch.no_grad():
                    # 单次 forward pass 获取所有 Medusa logits
                    medusa_logits, _, _ = self.model(
                        input_ids, 
                        output_orig=True, 
                        medusa_forward=True
                    )
                    # medusa_logits shape: [num_heads, batch, seq_len, vocab]
                    
                    # 对每个位置,检查 Medusa heads 的预测
                    # position i 的 head j 预测的是 position i+j+2 的 token (与训练对齐: labels[..., 2+j:])
                    for pos in range(seq_len - self.num_heads - 2):
                        for head_idx in range(self.num_heads):
                            target_pos = pos + head_idx + 2  # 修正：提前 (head_idx+2) 步，与训练一致
                            if target_pos >= seq_len:
                                break
                            
                            # Ground truth token
                            gt_token = input_ids[0, target_pos].item()
                            
                            # Medusa head 在位置 pos 的预测
                            head_logits = medusa_logits[head_idx, 0, pos, :]
                            
                            # Top-1 准确率
                            pred_token = head_logits.argmax().item()
                            is_correct = (pred_token == gt_token)
                            stats['head_accuracy'][head_idx].append(float(is_correct))
                            
                            # Top-K 命中率
                            top_k_tokens = head_logits.topk(self.top_k).indices.cpu().tolist()
                            is_in_top_k = (gt_token in top_k_tokens)
                            stats['head_top_k_hit'][head_idx].append(float(is_in_top_k))
                
                processed += 1
                
                # 定期打印进度统计
                if processed % 100 == 0:
                    print(f"\n  Progress: {processed} samples processed")
                    for head_idx in range(self.num_heads):
                        if stats['head_accuracy'][head_idx]:
                            acc = np.mean(stats['head_accuracy'][head_idx])
                            hit = np.mean(stats['head_top_k_hit'][head_idx])
                            print(f"    Head {head_idx}: Acc={acc:.4f}, Top-{self.top_k}={hit:.4f}")
                            
            except Exception as e:
                print(f"  Warning: Error processing sample: {e}")
                continue
        
        print(f"\nTotal samples processed: {processed}")
        return stats
    
    def build_tree_greedy(self, stats: Dict, max_candidates: int = 64) -> List[List[int]]:
        """基于统计数据构建最优树（完全遵循 Medusa 论文原版期望贪心算法）"""
        print("\nBuilding Medusa tree (Greedy Selection of Maximum Expected Acceptance Length)...")
        
        # 计算每个 head 的准确率及其 decay rate
        head_accuracies = []
        head_decay_rates = []
        for head_idx in range(self.num_heads):
            if stats['head_accuracy'][head_idx]:
                acc = np.mean(stats['head_accuracy'][head_idx])
                hit = np.mean(stats['head_top_k_hit'][head_idx])
            else:
                acc, hit = 0.01, 0.01
            head_accuracies.append(acc)
            # 通过理论公式估算 token 排列的衰减系数 r: A * (1-r^K)/(1-r) ≈ Hit -> r ≈ (Hit - Acc)/Hit
            r = (hit - acc) / max(hit, 1e-6)
            r = max(0.01, min(0.95, r))
            head_decay_rates.append(r)
            print(f"  Head {head_idx}: Top-1 Accuracy = {acc:.4f}, Decay Rate = {r:.4f}")
        
        medusa_choices = []
        # candidates: 存储格式为 (路径_list, 真实联合概率, 用于排序的分数)
        candidates = []
        
        # 初始化候选池：第一层（Head 0）的所有可能预测
        for token_idx in range(self.top_k):
            # 第一层的期望准确率 = head 0 的准确率 * 该层特定的衰减因子
            expected_prob = head_accuracies[0] * (head_decay_rates[0] ** token_idx)
            # 根据论文，期望贡献恰好等于联合准确率
            candidates.append(([token_idx], expected_prob, expected_prob))
            
        # 使用 Best-First Search 扩展节点，直到选满 max_candidates 个节点
        for _ in range(max_candidates):
            if not candidates:
                break
                
            # 全局按期望概率从大到小排序
            candidates.sort(key=lambda x: x[2], reverse=True)
            
            # 取出当前期望概率最高的最优节点
            best_path, best_prob, best_score = candidates.pop(0)
            medusa_choices.append(best_path)
            
            # 若该路径还能继续往深层扩展，则生成其所有子节点加入候选池
            depth = len(best_path)
            if depth < self.num_heads:
                for token_idx in range(self.top_k):
                    child_path = best_path + [token_idx]
                    # 子连乘概率 = 父联合概率 * 当前层预测的概率
                    step_prob = head_accuracies[depth] * (head_decay_rates[depth] ** token_idx)
                    child_prob = best_prob * step_prob
                    candidates.append((child_path, child_prob, child_prob))
        
        # 验证前缀属性（双重保险）
        medusa_choices = self._validate_prefix_property(medusa_choices)
        
        # 按长度和值排序（Medusa 框架约定俗成的易读格式）
        medusa_choices.sort(key=lambda x: (len(x), x))
        
        return medusa_choices
    
    def _validate_prefix_property(self, medusa_choices: List[List[int]]) -> List[List[int]]:
        """验证并修复前缀属性：如果路径存在，其所有前缀也必须存在"""
        choices_set = set(tuple(c) for c in medusa_choices)
        
        # 对所有路径，确保其前缀都在集合中
        all_required = set()
        for path in choices_set:
            # 添加该路径的所有前缀
            for i in range(1, len(path) + 1):
                all_required.add(path[:i])
        
        # 转换回列表格式
        result = sorted([list(p) for p in all_required])
        
        num_added = len(result) - len(medusa_choices)
        if num_added > 0:
            print(f"  ⚠️  Added {num_added} required prefixes to satisfy prefix property")
        
        return result
    
    def build_tree_from_accuracy(self, stats: Dict, max_candidates: int = 64) -> List[List[int]]:
        """基于准确率的简化树构建方法（基于 Medusa 论文思想连乘评估全量组合）"""
        print("\nBuilding Medusa tree (accuracy-based with actual decay)...")
        
        # 计算每个 head 的准确率及其 decay rate
        head_accuracies = []
        head_decay_rates = []
        for head_idx in range(self.num_heads):
            if stats['head_accuracy'][head_idx]:
                acc = np.mean(stats['head_accuracy'][head_idx])
                hit = np.mean(stats['head_top_k_hit'][head_idx])
            else:
                acc, hit = 0.01, 0.01
            head_accuracies.append(acc)
            r = (hit - acc) / max(hit, 1e-6)
            r = max(0.01, min(0.95, r))
            head_decay_rates.append(r)
            print(f"  Head {head_idx}: Top-1 accuracy = {acc:.4f}, Decay Rate = {r:.4f}")
        
        medusa_choices = []
        
        # 笛卡尔积（评估所有可能的路径组合）
        from itertools import product
        ranges = [range(self.top_k) for _ in range(self.num_heads)]
        
        all_paths = []
        for depth in range(1, self.num_heads + 1):
            for combo in product(*ranges[:depth]):
                path = list(combo)
                # 计算论文中定义的连乘期望贡献
                expected_value = 1.0
                for i, k in enumerate(path):
                    expected_value *= head_accuracies[i] * (head_decay_rates[i] ** k)
                all_paths.append((path, expected_value))
        
        # 按期望值排序，取 top candidates
        all_paths.sort(key=lambda x: x[1], reverse=True)
        
        for path, score in all_paths[:max_candidates]:
            medusa_choices.append(path)
        
        # 排序
        medusa_choices.sort(key=lambda x: (len(x), x))
        
        return medusa_choices
    def save_tree_config(self, medusa_choices: List[List[int]], output_path: str, 
                         stats: Dict, tree_name: str = "pangu_optimized"):
        """保存树配置到 Python 文件"""
        with open(output_path, 'w') as f:
            f.write(f'''# Medusa Tree Configuration: {tree_name}
# Auto-generated by medusa_tree_builder_fast.py
# Based on {len(stats['head_accuracy'][0]) if stats['head_accuracy'][0] else 0} validation samples

# Statistics:
''')
            for head_idx in range(self.num_heads):
                if stats['head_accuracy'][head_idx]:
                    acc = np.mean(stats['head_accuracy'][head_idx])
                    hit = np.mean(stats['head_top_k_hit'][head_idx])
                else:
                    acc, hit = 0.0, 0.0
                f.write(f'#   Head {head_idx}: Accuracy={acc:.4f}, Top-{self.top_k} Hit={hit:.4f}\n')
            
            f.write(f'''
{tree_name} = {medusa_choices}

# Tree info:
#   Total candidates: {len(medusa_choices)}
#   Max depth: {max(len(path) for path in medusa_choices) if medusa_choices else 0}
#   Avg depth: {np.mean([len(path) for path in medusa_choices]) if medusa_choices else 0:.2f}
''')
        
        print(f"\nTree configuration saved to: {output_path}")
        print(f"  Total candidates: {len(medusa_choices)}")
        if medusa_choices:
            print(f"  Max depth: {max(len(path) for path in medusa_choices)}")


def main():
    parser = argparse.ArgumentParser(description="Build optimal Medusa tree (fast version)")
    parser.add_argument("--model_path", type=str, default=".", help="Base model path")
    parser.add_argument("--medusa_head", type=str, required=True, 
                        help="Medusa head checkpoint path (medusa_lm_head.safetensors)")
    parser.add_argument("--num_heads", type=int, default=5, help="Number of Medusa heads")
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Validation data (JSON with conversations)")
    parser.add_argument("--num_samples", type=int, default=1000, 
                        help="Number of samples to use for statistics")
    parser.add_argument("--max_seq_len", type=int, default=2048, 
                        help="Max sequence length per sample")
    parser.add_argument("--top_k", type=int, default=10, 
                        help="Top-K candidates per head")
    parser.add_argument("--max_candidates", type=int, default=32, 
                        help="Maximum tree candidates")
    parser.add_argument("--output", type=str, default="medusa_tree_optimized.py", 
                        help="Output tree configuration file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--method", type=str, default="greedy", 
                        choices=["greedy", "accuracy"],
                        help="Tree building method: greedy or accuracy-based")
    args = parser.parse_args()
    
    # 加载验证数据
    print(f"Loading validation data from {args.data_path}...")
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # 构建树
    builder = MedusaTreeBuilderFast(
        args.model_path, 
        args.medusa_head, 
        args.num_heads, 
        args.device, 
        args.top_k
    )
    
    # 快速收集统计信息
    stats = builder.collect_statistics_fast(data, args.num_samples, args.max_seq_len)
    
    # 构建树
    if args.method == "greedy":
        medusa_choices = builder.build_tree_greedy(stats, args.max_candidates)
    else:
        medusa_choices = builder.build_tree_from_accuracy(stats, args.max_candidates)
    
    tree_name = f"pangu_{args.num_heads}heads_top{args.top_k}"
    builder.save_tree_config(medusa_choices, args.output, stats, tree_name)
    
    print("\n✅ Done! Use this tree in inference:")
    print(f"   from {Path(args.output).stem} import {tree_name}")
    print(f"   model.generate(..., medusa_choices={tree_name})")


if __name__ == "__main__":
    main()
