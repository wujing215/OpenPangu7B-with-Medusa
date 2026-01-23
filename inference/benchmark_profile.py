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
import random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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

def benchmark_baseline(model_path, prompt, max_new_tokens, num_runs=3, warmup_runs=1, profile=True):
    """
    测试原始模型（无 Medusa）的逐步性能
    Args:
        profile (bool): 是否开启详细的逐步计时（Step 00xx ...）
    """

    print("=" * 60)
    print(f"Baseline: OpenPangu (Manual Loop) on {DEVICE_TAG}")
    print(f"Profile Mode: {profile} (Per-step timing enabled)")
    print("=" * 60)
    
    # 1. 加载模型
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        use_fast=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=target_device,
        trust_remote_code=True,
    )
    model.eval()

    # 打印验收信息
    print_acceptance_info(model_path, is_parallel=False)
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(target_device)
    input_len = input_ids.shape[1]
    print(f"Prompt Length: {input_len}")
    
    # OpenPangu 特定的 eos_token_id (如果不确定，可以从 config 获取)
    eos_token_id = getattr(model.config, "eos_token_id", 45892)
    
    # 3. Warmup
    print(f"\nWarmup ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                max_new_tokens=100, # 稍微跑一点即可
                do_sample=False,
                eos_token_id=eos_token_id,
                use_cache=True
            )
    
    # 4. Benchmark Loop
    print(f"Benchmarking ({num_runs} runs)...")
    total_times = []
    total_tokens_list = []
    
    for run_i in range(num_runs):
        print(f"\n--- Run {run_i + 1} ---")
        
        # 重置状态
        curr_input_ids = input_ids.clone()
        past_key_values = None
        curr_step = 0
        
        # 计时器
        if profile: device_synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            # ==========================================
            # A. Prefill 阶段 (处理 Prompt)
            # ==========================================
            # 第一次前向传播，计算整个 Prompt 的 KV Cache
            outputs = model(
                input_ids=curr_input_ids,
                past_key_values=None,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            # 取最后一个 token 的 logits 预测下一个
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            curr_input_ids = torch.cat([curr_input_ids, next_token], dim=-1)
            
            # ==========================================
            # B. Decoding 阶段 (逐 Token 生成)
            # ==========================================
            for step in range(max_new_tokens - 1): # -1 因为 Prefill 已经生成了 1 个
                
                step_start = time.time()
                
                # 核心推理：只传入最新的 token 和 缓存
                outputs = model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                # 更新状态
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                # 记录 ID (用于最终解码，虽然 Benchmark 不太关心内容)
                # curr_input_ids = torch.cat([curr_input_ids, next_token], dim=-1)
                
                # --- Profiling 打印 ---
                if profile:
                    device_synchronize()
                    step_end = time.time()
                    elapsed_ms = (step_end - step_start) * 1000
                    
                    # 打印格式对齐 Medusa
                    if step % 50 == 0:
                    # if step % 1 == 0:
                        print(f"Step {step:04d} | Total: {elapsed_ms:.2f}ms")
                    
                curr_step += 1
                
                if not profile and next_token.item() == eos_token_id:
                    break

        if profile: device_synchronize()
        end_time = time.time()
        
        total_elapsed = end_time - start_time
        generated_tokens = curr_step + 1 # +1 for the prefill token
        
        total_times.append(total_elapsed)
        total_tokens_list.append(generated_tokens)
        
        print(f"Run {run_i + 1} Finished: {total_elapsed:.3f}s, {generated_tokens} tokens")

    # 5. 统计结果
    avg_time = sum(total_times) / len(total_times)
    avg_tokens = sum(total_tokens_list) / len(total_tokens_list)
    tps = avg_tokens / avg_time
    tpot = (avg_time / avg_tokens) * 1000
    
    results = {
        "method": "Baseline (Manual Loop)",
        "avg_time": avg_time,
        "avg_tokens": avg_tokens,
        "tps": tps,
        "tpot": tpot,
    }
    
    print(f"\n--- Baseline Results ---")
    print(f"TPS: {tps:.2f} tokens/s")
    print(f"TPOT: {tpot:.2f} ms/token")
    
    # 清理
    del model
    if DEVICE_TYPE == 'npu':
        torch.npu.empty_cache()
    elif DEVICE_TYPE == 'cuda':
        torch.cuda.empty_cache()
        
    return results


def benchmark_medusa(model_path, medusa_dir, prompt, max_new_tokens, num_runs=3, warmup_runs=1, profile=False):
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
        medusa_num_heads=5,
        medusa_num_layers=1,
    )

    # 【验收截图关键点】打印验收信息
    print_acceptance_info(model_path, is_parallel=True)
    
    # 准备输入
    # messages = [
    #     {"role": "system", "content": "你是一个有帮助的助手。"},
    #     {"role": "user", "content": prompt},
    # ]
    # formatted_prompt = model.apply_chat_template(messages)
    # print(f"Prompt: {formatted_prompt}")
    formatted_prompt = prompt

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
            profile=profile
            # medusa_choices=pangu_stage2,
        )
        
        device_synchronize() # 适配 NPU 同步
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        
        # 计算输出 token 数
        output_ids = model.tokenizer.encode(output_text, return_tensors="pt")
        output_tokens = output_ids.shape[1]
        
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
                        default="/root/OpenPangu7B-with-Medusa/medusa_5heads_layers1_1231_medusa_mlp_openPangu-Embedded-7B-V1.1_medusa_5_lr_0.001_layers_1/",
                        help="Medusa head directory")
    parser.add_argument("--prompt", type=str, 
                        default="Write an immersive, long-form science fiction narrative about a galactic 'Silk Road' that connects the far reaches of the universe. Focus on building a vivid world that showcases advanced speculative technology, breathtaking alien environments, and the complex socio-economic trade between multiple non-human civilizations. Develop a multi-layered plot with rich character interactions and descriptive world-building, ensuring a substantial length.",
                        help="The input prompt for non-profile mode")
    parser.add_argument("--prompt_len", type=int, default=512,
                        help="Length of the input prompt (number of tokens)")
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
    parser.add_argument("--profile", action="store_true",
                        help="run with profile mode")
    args = parser.parse_args()
    
    # 解析路径
    base_model_path = str(Path(args.base_model).expanduser().resolve())
    medusa_dir = str(Path(args.medusa_dir).expanduser().resolve())
    
    print("=" * 60)
    print(f"OpenPangu + Medusa Performance Benchmark on {DEVICE_TAG}")
    print("=" * 60)
    print(f"Base model: {base_model_path}")
    print(f"Medusa dir: {medusa_dir}")
    print(f"Prompt Length: {args.prompt_len} tokens") # 显示长度
    print(f"Decode step: {args.max_tokens}")
    print(f"Runs: {args.num_runs} (+ {args.warmup} warmup)")
    

    if args.profile:
        # Profile 模式：生成指定长度的随机 Prompt (用于压测性能，不考虑语义)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        print(f"[{DEVICE_TAG}] Profile mode: Generating a random prompt of {args.prompt_len} tokens...")
        vocab_range = range(100, tokenizer.vocab_size - 10) 
        random_token_ids = random.choices(vocab_range, k=args.prompt_len)
        generated_prompt = tokenizer.decode(random_token_ids, skip_special_tokens=True)
    else:
        # 正常模式：使用 --prompt 传入的文本
        print(f"[{DEVICE_TAG}] Normal mode: Using provided prompt.")
        generated_prompt = args.prompt
    results = {}
    
    # Baseline benchmark
    if not args.medusa_only:
        results["baseline"] = benchmark_baseline(
            base_model_path, 
            generated_prompt, 
            args.max_tokens,
            args.num_runs,
            args.warmup,
            args.profile
        )
    
    # Medusa benchmark
    if not args.baseline_only:
        results["medusa"] = benchmark_medusa(
            base_model_path,
            medusa_dir,
            generated_prompt,
            args.max_tokens,
            args.num_runs,
            args.warmup,
            args.profile
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
    