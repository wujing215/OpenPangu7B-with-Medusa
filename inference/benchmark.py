#!/usr/bin/env python3
"""
OpenPangu Medusa 性能基准测试脚本 (已适配昇腾NPU验收)

对比集成 Medusa Heads 前后的推理性能：
- generate.py: 原始 OpenPangu 模型（自回归解码）
- medusa_generate.py: 集成 Medusa 的模型（投机解码）

测试指标：
- TPOT (Time Per Output Token): 每个 token 的生成时间 (ms)
- TPS (Tokens Per Second): 每秒生成的 token 数
- TTFT (Time To First Token): 首个 token 延迟 (ms)
- 总生成时间
- 加速比 (Speedup)
"""

import argparse
import time
import torch
import sys
import os
from pathlib import Path

# --- 昇腾 NPU 适配与验收信息准备 ---
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    DEVICE_TYPE = "npu"
    DEVICE_TAG = "昇腾 (Ascend NPU)"
    print(f"[Info] 检测到 torch_npu，将使用昇腾 NPU 进行推理。")
except ImportError:
    DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE_TAG = "NVIDIA CUDA" if DEVICE_TYPE == "cuda" else "CPU"
    print(f"[Warning] 未检测到 torch_npu，退化使用 {DEVICE_TAG}。")

target_device = f"{DEVICE_TYPE}:0"

def device_synchronize():
    """跨平台设备同步函数"""
    if DEVICE_TYPE == 'npu':
        torch.npu.synchronize()
    elif DEVICE_TYPE == 'cuda':
        torch.cuda.synchronize()

def empty_cache():
    """跨平台清空显存函数"""
    if DEVICE_TYPE == 'npu':
        torch.npu.empty_cache()
    elif DEVICE_TYPE == 'cuda':
        torch.cuda.empty_cache()

def print_acceptance_info(model_path, is_parallel=False):
    """打印第一阶段验收所需的关键信息用于截图"""
    mode_str = "支持并行推理 (Medusa)" if is_parallel else "基准串行推理 (Baseline)"
    print("\n" + "=" * 60)
    print(f"1. 运行模式: {mode_str}")
    print(f"2. 运行环境: {DEVICE_TAG}, Device: {target_device})")
    print(f"3. 模型状态: 加载模型为openPangu系列开源模型")
    print(f"   - 模型路径: {model_path}")
    print("=" * 60 + "\n")
# ------------------------------------

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def benchmark_baseline(model_path, prompt, max_new_tokens, num_runs=3, warmup_runs=1):
    """测试原始模型（无 Medusa）"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    
    print("=" * 60)
    print(f"Baseline: OpenPangu (Autoregressive Decoding) on {DEVICE_TAG}")
    print("=" * 60)
    
    # 加载模型
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        use_fast=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=target_device, # 适配 NPU
        trust_remote_code=True,
    )
    model.eval()

    # 【验收截图关键点】打印验收信息
    print_acceptance_info(model_path, is_parallel=False)
    
    # 准备输入
    messages = [
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": prompt},
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # 适配 NPU
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(target_device)
    input_len = input_ids.shape[1]
    
    print(f"Input length: {input_len} tokens")
    print(f"Max new tokens: {max_new_tokens}")
    
    # OpenPangu 特定的 eos_token_id
    eos_token_id = 45892
    
    # Warmup
    print(f"\nWarmup ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_token_id,
            )
    
    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    output_tokens_list = []
    ttft_list = []
    
    for i in range(num_runs):
        device_synchronize() # 适配 NPU 同步
        
        # 测量 TTFT（首个 token 时间）
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_token_id,
            )
        
        device_synchronize() # 适配 NPU 同步
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        output_tokens = outputs.shape[1] - input_len
        
        times.append(elapsed)
        output_tokens_list.append(output_tokens)
        
        print(f"  Run {i+1}: {elapsed:.3f}s, {output_tokens} tokens")
    
    # 计算统计
    avg_time = sum(times) / len(times)
    avg_tokens = sum(output_tokens_list) / len(output_tokens_list)
    tps = avg_tokens / avg_time
    tpot = (avg_time / avg_tokens) * 1000  # ms per token
    
    results = {
        "method": "Baseline (Autoregressive)",
        "avg_time": avg_time,
        "avg_tokens": avg_tokens,
        "tps": tps,
        "tpot": tpot,
    }
    
    print(f"\n--- Baseline Results ---")
    print(f"Average time: {avg_time:.3f}s")
    print(f"Average tokens: {avg_tokens:.1f}")
    print(f"TPS: {tps:.2f} tokens/s")
    print(f"TPOT: {tpot:.2f} ms/token")
    
    # 清理显存
    del model
    empty_cache() # 适配 NPU 清理显存
    
    return results


def benchmark_medusa(model_path, medusa_dir, prompt, max_new_tokens, num_runs=3, warmup_runs=1):
    """测试 Medusa 模型（投机解码）"""
    from medusa_generate import MedusaPanguInference
    from medusa_choices import pangu_stage2
    
    print("\n" + "=" * 60)
    print(f"Medusa: OpenPangu + Medusa Heads (Speculative Decoding) on {DEVICE_TAG}")
    print("=" * 60)
    
    # 加载模型
    print("Loading model...")
    model = MedusaPanguInference(
        base_model_path=model_path,
        medusa_head_path=os.path.join(medusa_dir, "medusa_lm_head.safetensors"),
        tokenizer_path=medusa_dir,
        device=target_device, # 适配 NPU
        dtype=torch.float16,
        medusa_num_heads=3,
        medusa_num_layers=1,
    )

    # 【验收截图关键点】打印验收信息
    print_acceptance_info(model_path, is_parallel=True)
    
    # 准备输入
    messages = [
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": prompt},
    ]
    formatted_prompt = model.apply_chat_template(messages)
    # 适配 NPU
    input_ids = model.tokenizer.encode(formatted_prompt, return_tensors="pt").to(target_device)
    input_len = input_ids.shape[1]
    
    print(f"Input length: {input_len} tokens")
    print(f"Max steps: {max_new_tokens}")
    
    # Warmup
    print(f"\nWarmup ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        _ = model.generate(formatted_prompt, max_steps=max_new_tokens, temperature=0.0)
    
    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    output_tokens_list = []
    accepted_tokens_list = []
    
    for i in range(num_runs):
        device_synchronize() # 适配 NPU 同步
        start_time = time.perf_counter()
        
        output_text = model.generate(
            formatted_prompt, 
            max_steps=max_new_tokens, 
            temperature=0.0,
            medusa_choices=pangu_stage2,
        )
        
        device_synchronize() # 适配 NPU 同步
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        
        # 计算输出 token 数
        output_ids = model.tokenizer.encode(output_text, return_tensors="pt")
        output_tokens = output_ids.shape[1] - input_len
        
        times.append(elapsed)
        output_tokens_list.append(output_tokens)
        
        print(f"  Run {i+1}: {elapsed:.3f}s, {output_tokens} tokens")
    
    # 计算统计
    avg_time = sum(times) / len(times)
    avg_tokens = sum(output_tokens_list) / len(output_tokens_list)
    tps = avg_tokens / avg_time
    tpot = (avg_time / avg_tokens) * 1000  # ms per token
    
    results = {
        "method": "Medusa (Speculative Decoding)",
        "avg_time": avg_time,
        "avg_tokens": avg_tokens,
        "tps": tps,
        "tpot": tpot,
    }
    
    print(f"\n--- Medusa Results ---")
    print(f"Average time: {avg_time:.3f}s")
    print(f"Average tokens: {avg_tokens:.1f}")
    print(f"TPS: {tps:.2f} tokens/s")
    print(f"TPOT: {tpot:.2f} ms/token")
    
    # 清理显存
    del model
    empty_cache() # 适配 NPU 清理显存
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark OpenPangu with/without Medusa")
    parser.add_argument("--base_model", type=str, default="/root/openPangu-Embedded-7B-V1.1",
                        help="Base model path")
    parser.add_argument("--medusa_dir", type=str, 
                        default="/root/OpenPangu7B-on-NVIDIA/test_medusa_mlp_._medusa_3_lr_0.001_layers_1",
                        help="Medusa head directory")
    parser.add_argument("--prompt", type=str, 
                        default="请详细介绍一下大语言模型的工作原理。",
                        help="Test prompt")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Maximum new tokens to generate")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup runs")
    parser.add_argument("--baseline_only", action="store_true",
                        help="Only run baseline benchmark")
    parser.add_argument("--medusa_only", action="store_true",
                        help="Only run Medusa benchmark")
    args = parser.parse_args()
    
    # 解析路径
    base_model_path = str(Path(args.base_model).expanduser().resolve())
    medusa_dir = str(Path(args.medusa_dir).expanduser().resolve())
    
    print("=" * 60)
    print(f"OpenPangu + Medusa Performance Benchmark on {DEVICE_TAG}")
    print("=" * 60)
    print(f"Base model: {base_model_path}")
    print(f"Medusa dir: {medusa_dir}")
    print(f"Prompt: {args.prompt[:50]}...")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Runs: {args.num_runs} (+ {args.warmup} warmup)")
    
    results = {}
    
    # Baseline benchmark
    if not args.medusa_only:
        results["baseline"] = benchmark_baseline(
            base_model_path, 
            args.prompt, 
            args.max_tokens,
            args.num_runs,
            args.warmup,
        )
    
    # Medusa benchmark
    if not args.baseline_only:
        results["medusa"] = benchmark_medusa(
            base_model_path,
            medusa_dir,
            args.prompt,
            args.max_tokens,
            args.num_runs,
            args.warmup,
        )
    
    # 对比结果
    if "baseline" in results and "medusa" in results:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        
        baseline = results["baseline"]
        medusa = results["medusa"]
        
        speedup_tps = medusa["tps"] / baseline["tps"]
        speedup_time = baseline["avg_time"] / medusa["avg_time"]
        
        print(f"\n{'Metric':<25} {'Baseline':<15} {'Medusa':<15} {'Speedup':<10}")
        print("-" * 65)
        print(f"{'TPS (tokens/s)':<25} {baseline['tps']:<15.2f} {medusa['tps']:<15.2f} {speedup_tps:<10.2f}x")
        print(f"{'TPOT (ms/token)':<25} {baseline['tpot']:<15.2f} {medusa['tpot']:<15.2f} {baseline['tpot']/medusa['tpot']:<10.2f}x")
        print(f"{'Total time (s)':<25} {baseline['avg_time']:<15.3f} {medusa['avg_time']:<15.3f} {speedup_time:<10.2f}x")
        print(f"{'Tokens generated':<25} {baseline['avg_tokens']:<15.1f} {medusa['avg_tokens']:<15.1f}")
        
        print(f"\n🚀 Medusa achieves {speedup_tps:.2f}x speedup!")
        
        if speedup_tps > 1.5:
            print("✅ Significant speedup achieved!")
        elif speedup_tps > 1.0:
            print("⚠️  Moderate speedup. Consider tuning Medusa parameters.")
        else:
            print("❌ No speedup. Check Medusa head training quality.")


if __name__ == "__main__":
    main()
    