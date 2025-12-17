#!/usr/bin/env python3
"""
合并多个 GPU 生成的部分数据文件
"""

import json
import glob
from pathlib import Path

def merge_distilled_data(pattern="pangu_distilled_part_*.json", output="pangu_self_distilled_data.json"):
    """合并所有部分文件"""
    
    part_files = sorted(glob.glob(pattern))
    
    if not part_files:
        print(f"❌ No files found matching: {pattern}")
        return
    
    print(f"Found {len(part_files)} part files:")
    for f in part_files:
        print(f"  - {f}")
    
    all_data = []
    
    for part_file in part_files:
        with open(part_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.extend(data)
            print(f"  Loaded {len(data)} samples from {part_file}")
    
    # 保存合并结果
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Merged {len(all_data)} total samples")
    print(f"   Output: {output}")
    
    # 统计
    print(f"\n📊 Statistics:")
    print(f"   Total conversations: {len(all_data)}")
    
    total_tokens = 0
    for item in all_data:
        for conv in item.get("conversations", []):
            if conv.get("from") == "gpt":
                total_tokens += len(conv.get("value", "").split())
    
    avg_tokens = total_tokens / len(all_data) if all_data else 0
    print(f"   Avg tokens per response: {avg_tokens:.1f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="pangu_distilled_part_*.json")
    parser.add_argument("--output", default="pangu_self_distilled_data.json")
    args = parser.parse_args()
    
    merge_distilled_data(args.pattern, args.output)
