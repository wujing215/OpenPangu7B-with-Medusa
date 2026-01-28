#!/usr/bin/env python3
"""
快速自蒸馏数据生成脚本 - 优化版

优化点：
1. 减少 max_new_tokens（512 而不是 1024）
2. 支持多进程并行（多 GPU）
3. 添加超时机制
4. 更频繁的保存
"""

import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import time

# ================= 配置 =================
MODEL_PATH = str(Path(__file__).parent.parent.resolve())
INPUT_DATA = "third_party/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json"
OUTPUT_DATA = "pangu_self_distilled_data.json"
DEVICE = "cuda"
NUM_SAMPLES = 50000
MAX_NEW_TOKENS = 512  
BATCH_SIZE = 1
TIMEOUT_SECONDS = 60  # 单个样本超时时间

# 全局变量
tokenizer = None
model = None


def generate_response(prompt_text, timeout=TIMEOUT_SECONDS):
    """使用 OpenPangu 原模型生成回复（自蒸馏）+ 超时保护"""
    messages = [
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": prompt_text}
    ]
    
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_text, return_tensors="pt").to(DEVICE)
    
    start_time = time.time()
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=MAX_NEW_TOKENS,  # 512
                do_sample=False,
                eos_token_id=45892,
                return_dict_in_generate=True
            )
        
        # 检查是否超时
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"\n  Generation timeout ({elapsed:.1f}s), skipping...")
            return None
        
        input_length = inputs.input_ids.shape[1]
        generated_ids = outputs.sequences[0, input_length:]
        output_sent = tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        # 保留完整输出，包括 [unused16]...[unused17]...[unused10]
        # 模型生成格式: [unused16]<thinking>[unused17]<response>[unused10]
        # 
        # [unused10](EOS)也要保留
        # 1. Medusa heads 需要学会预测何时结束生成
        # 2. train_medusa.py 不会额外添加 [unused10]
        return output_sent.strip()
    
    except Exception as e:
        print(f"\n Generation error: {e}")
        return None


def main():
    global MODEL_PATH, INPUT_DATA, OUTPUT_DATA, NUM_SAMPLES, MAX_NEW_TOKENS, DEVICE, tokenizer, model
    
    parser = argparse.ArgumentParser(description="Generate self-distillation data (Fast version)")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--input_data", type=str, default=INPUT_DATA)
    parser.add_argument("--output_data", type=str, default=OUTPUT_DATA)
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for parallel processing")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for parallel processing")
    args = parser.parse_args()
    
    MODEL_PATH = args.model_path
    INPUT_DATA = args.input_data
    OUTPUT_DATA = args.output_data
    NUM_SAMPLES = args.num_samples
    MAX_NEW_TOKENS = args.max_new_tokens
    DEVICE = args.device
    
    print("=" * 60)
    print("Fast Self-Distillation Data Generator")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Max tokens: {MAX_NEW_TOKENS}")
    print(f"Device: {DEVICE}")
    
    # 加载模型
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        device_map="auto", 
        torch_dtype=torch.float16, 
        trust_remote_code=True
    )
    model.eval()
    print(" Model loaded!")
    
    # 加载数据
    with open(INPUT_DATA, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # 确定处理范围
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx else min(NUM_SAMPLES, len(original_data))
    data_to_process = original_data[start_idx:end_idx]
    
    print(f"\nProcessing samples {start_idx} to {end_idx} ({len(data_to_process)} total)")
    
    new_data = []
    skipped = 0
    
    for i, conversation in enumerate(tqdm(data_to_process, desc="Generating")):
        actual_idx = start_idx + i
        
        try:
            user_msg = next((msg for msg in conversation["conversations"] if msg["from"] == "human"), None)
            
            if user_msg:
                prompt = user_msg["value"]
                
                # 生成回复
                pangu_response = generate_response(prompt)
                
                if pangu_response is None:
                    skipped += 1
                    continue
                
                new_conv_entry = {
                    "id": conversation.get("id", f"gen_{actual_idx}"),
                    "conversations": [
                        {"from": "human", "value": prompt},
                        {"from": "gpt", "value": pangu_response}
                    ]
                }
                
                new_data.append(new_conv_entry)
                
                # 每 50 个样本保存一次（更频繁）
                if len(new_data) % 50 == 0 and len(new_data) > 0:
                    with open(OUTPUT_DATA, 'w', encoding='utf-8') as out_f:
                        json.dump(new_data, out_f, indent=2, ensure_ascii=False)
                    print(f"\n Checkpoint: Saved {len(new_data)} samples (skipped: {skipped})")
                        
        except Exception as e:
            print(f"\n Error at {actual_idx}: {e}")
            skipped += 1
            continue
    
    # 最终保存
    with open(OUTPUT_DATA, 'w', encoding='utf-8') as out_f:
        json.dump(new_data, out_f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print(" Generation Complete!")
    print(f"   Generated: {len(new_data)} samples")
    print(f"   Skipped: {skipped} samples")
    print(f"   Output: {OUTPUT_DATA}")
    print("=" * 60)


if __name__ == "__main__":
    main()
