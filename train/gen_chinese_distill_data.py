#!/usr/bin/env python3
"""
生成中文自蒸馏数据用于 Medusa 训练

Medusa 训练的关键：
1. 使用目标模型自身生成的数据（自蒸馏）
2. 数据要与推理时使用的语言/领域匹配
3. 足够多的数据（至少 10K+ 样本）
"""

import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import random

# ================= 中文 Prompt 模板 =================
# 这些是多样化的中文提示词，涵盖各种主题和风格
CHINESE_PROMPTS = [
    # 知识问答类
    "请详细介绍一下人工智能的发展历史。",
    "什么是机器学习？它和深度学习有什么区别？",
    "请解释一下大语言模型的工作原理。",
    "区块链技术的核心特点是什么？",
    "量子计算的基本原理是什么？",
    "什么是云计算？它有哪些优势？",
    "请介绍一下5G技术的特点和应用场景。",
    "物联网是什么？它如何改变我们的生活？",
    "什么是元宇宙？它的发展前景如何？",
    "请解释一下神经网络的基本结构。",
    
    # 编程相关
    "请用Python写一个快速排序算法。",
    "如何在Python中读取和处理JSON文件？",
    "请解释Python中的装饰器是什么，并给出一个例子。",
    "什么是REST API？如何设计一个好的REST API？",
    "请介绍一下Git的基本使用方法。",
    "什么是Docker？它解决了什么问题？",
    "请解释一下什么是微服务架构。",
    "如何进行代码性能优化？",
    "请介绍一下数据库索引的原理和使用场景。",
    "什么是设计模式？请举例说明。",
    
    # 日常生活类
    "如何保持良好的睡眠质量？",
    "有哪些有效的时间管理方法？",
    "如何提高工作效率？",
    "请推荐一些健康的饮食习惯。",
    "如何缓解工作压力？",
    "怎样培养阅读习惯？",
    "如何提高英语口语水平？",
    "请分享一些理财的基本知识。",
    "怎样保持身体健康？",
    "如何提高专注力？",
    
    # 写作辅助类
    "请写一篇关于环保的短文。",
    "帮我写一封求职信。",
    "请写一个关于友情的故事。",
    "帮我写一份项目总结报告。",
    "请写一段产品介绍文案。",
    "帮我写一篇读书笔记的模板。",
    "请写一个工作计划的大纲。",
    "帮我写一份会议纪要。",
    "请写一段自我介绍。",
    "帮我写一个活动策划方案。",
    
    # 分析思考类
    "请分析一下远程办公的优缺点。",
    "中国传统文化有哪些值得传承的内容？",
    "如何看待人工智能对就业的影响？",
    "请分析一下电动汽车的发展前景。",
    "新能源技术的发展趋势是什么？",
    "如何平衡工作和生活？",
    "请分析一下在线教育的优势和挑战。",
    "数字化转型对企业有什么影响？",
    "如何看待社交媒体对社会的影响？",
    "请分析一下共享经济的商业模式。",
    
    # 创意类
    "请设计一个手机App的创意方案。",
    "帮我想一个公司的名字和slogan。",
    "请设计一个团建活动方案。",
    "帮我想几个短视频的创意主题。",
    "请设计一个节日促销活动方案。",
    "帮我想一个新产品的营销策略。",
    "请设计一个公司年会的流程。",
    "帮我想一个品牌故事。",
    "请设计一个用户调研方案。",
    "帮我想一个社区运营的活动方案。",
    
    # 技术问答
    "请解释什么是transformer架构。",
    "什么是注意力机制？它是如何工作的？",
    "请介绍一下BERT模型的特点。",
    "什么是GPT？它和BERT有什么区别？",
    "请解释什么是预训练和微调。",
    "什么是强化学习？请举例说明。",
    "请介绍一下CNN和RNN的区别。",
    "什么是模型蒸馏？它有什么作用？",
    "请解释什么是batch normalization。",
    "什么是过拟合？如何防止？",
    
    # 商业分析
    "请分析一下抖音的商业模式。",
    "如何做好一个创业项目的商业计划？",
    "请分析一下直播带货的发展趋势。",
    "什么是精益创业？如何实践？",
    "请分析一下SaaS商业模式的特点。",
    "如何进行用户画像分析？",
    "请介绍一下A/B测试的方法。",
    "什么是增长黑客？有哪些常用策略？",
    "请分析一下会员制商业模式的优势。",
    "如何提高用户留存率？",
    
    # 学习方法
    "如何高效地学习一门新技能？",
    "请分享一些记忆方法和技巧。",
    "如何制定有效的学习计划？",
    "请介绍一下费曼学习法。",
    "如何克服学习中的拖延症？",
    "请分享一些有效的笔记方法。",
    "如何保持学习的动力？",
    "请介绍一下番茄工作法。",
    "如何进行自我评估和反思？",
    "请分享一些提高阅读效率的方法。",
    
    # 历史文化
    "请介绍一下中国的四大发明。",
    "唐朝为什么被称为盛世？",
    "请介绍一下丝绸之路的历史意义。",
    "孔子的主要思想是什么？",
    "请介绍一下中国古代的科举制度。",
    "长城有哪些历史和文化价值？",
    "请介绍一下中国传统节日的由来。",
    "什么是儒家思想？它对中国文化有什么影响？",
    "请介绍一下中国古代的四大名著。",
    "中国茶文化有哪些特点？",
    
    # 科学知识
    "请解释一下相对论的基本概念。",
    "什么是黑洞？它是如何形成的？",
    "请介绍一下DNA的结构和功能。",
    "地球的内部结构是什么样的？",
    "什么是光合作用？它的过程是怎样的？",
    "请解释一下进化论的核心观点。",
    "什么是牛顿三大定律？",
    "请介绍一下元素周期表的规律。",
    "什么是温室效应？它有什么影响？",
    "请解释一下声音是如何产生和传播的。",
    
    # 数学相关
    "请解释一下什么是微积分。",
    "什么是概率论？它有哪些实际应用？",
    "请介绍一下线性代数的基本概念。",
    "什么是统计学中的正态分布？",
    "请解释一下什么是博弈论。",
    "什么是数学归纳法？请举例说明。",
    "请介绍一下傅里叶变换的概念。",
    "什么是蒙特卡洛方法？",
    "请解释一下什么是贝叶斯定理。",
    "什么是数值分析？它有什么用途？",
    
    # 更多技术话题
    "请介绍一下Linux操作系统的特点。",
    "什么是TCP/IP协议？",
    "请解释一下HTTP和HTTPS的区别。",
    "什么是数据库的ACID特性？",
    "请介绍一下NoSQL数据库的类型和特点。",
    "什么是分布式系统？它有哪些挑战？",
    "请解释一下什么是负载均衡。",
    "什么是CDN？它是如何工作的？",
    "请介绍一下消息队列的使用场景。",
    "什么是CI/CD？如何实施？",
    
    # 职场相关
    "如何准备一次技术面试？",
    "请分享一些团队管理的经验。",
    "如何进行有效的项目管理？",
    "请介绍一下敏捷开发的方法。",
    "如何提升自己的领导力？",
    "请分享一些职业规划的建议。",
    "如何处理职场中的人际关系？",
    "请介绍一下OKR的设定方法。",
    "如何进行有效的向上沟通？",
    "请分享一些跨部门协作的技巧。",
]

# 扩展prompts的变体
def expand_prompts(base_prompts, target_count=10000):
    """扩展基础prompts到目标数量"""
    expanded = []
    
    # 添加前缀变体
    prefixes = [
        "",
        "请详细解答：",
        "我想了解",
        "能否帮我解释一下",
        "请问",
        "帮我分析一下",
        "简单介绍一下",
        "详细说明",
    ]
    
    # 添加后缀变体
    suffixes = [
        "",
        "谢谢！",
        "请详细说明。",
        "希望能给出具体的例子。",
        "越详细越好。",
    ]
    
    for prompt in base_prompts:
        for prefix in prefixes:
            for suffix in suffixes:
                new_prompt = f"{prefix}{prompt}{suffix}".strip()
                if new_prompt:
                    expanded.append(new_prompt)
    
    # 如果还不够，随机组合
    while len(expanded) < target_count:
        base = random.choice(base_prompts)
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        expanded.append(f"{prefix}{base}{suffix}".strip())
    
    return expanded[:target_count]

# ================= 配置 =================
MODEL_PATH = str(Path(__file__).parent.parent.resolve())
OUTPUT_DATA = "pangu_chinese_distilled_data.json"
DEVICE = "cuda"
NUM_SAMPLES = 10000  # 目标生成数量
MAX_NEW_TOKENS = 1024
BATCH_SIZE = 1

tokenizer = None
model = None

def generate_response(prompt_text):
    """使用 OpenPangu 原模型生成回复（自蒸馏）"""
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
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=MAX_NEW_TOKENS, 
            do_sample=False,  # 贪婪解码
            eos_token_id=45892,  # OpenPangu 的 EOS token
            return_dict_in_generate=True
        )
    
    input_length = inputs.input_ids.shape[1]
    generated_ids = outputs.sequences[0, input_length:]
    output_sent = tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    # 返回完整输出（包含 thinking 和 content）
    # 因为 Medusa 需要学习整个输出序列
    return output_sent

def main():
    global MODEL_PATH, OUTPUT_DATA, NUM_SAMPLES, MAX_NEW_TOKENS, DEVICE, tokenizer, model
    
    parser = argparse.ArgumentParser(description="生成中文自蒸馏数据用于 Medusa 训练")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Base model path")
    parser.add_argument("--output_data", type=str, default=OUTPUT_DATA, help="Output distilled data")
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES, help="Number of samples to generate")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS, help="Max generation length")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device to use")
    args = parser.parse_args()
    
    MODEL_PATH = args.model_path
    OUTPUT_DATA = args.output_data
    NUM_SAMPLES = args.num_samples
    MAX_NEW_TOKENS = args.max_new_tokens
    DEVICE = args.device
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        device_map="auto", 
        torch_dtype=torch.float16, 
        trust_remote_code=True
    )
    model.eval()
    print("Model loaded successfully!")
    
    # 扩展 prompts
    print(f"Expanding prompts from {len(CHINESE_PROMPTS)} to {NUM_SAMPLES}...")
    prompts = expand_prompts(CHINESE_PROMPTS, NUM_SAMPLES)
    random.shuffle(prompts)  # 打乱顺序
    
    results = []
    failed_count = 0
    
    print(f"Generating {NUM_SAMPLES} samples...")
    for idx, prompt in enumerate(tqdm(prompts)):
        try:
            response = generate_response(prompt)
            
            # 检查回复长度
            if len(response) < 10:
                failed_count += 1
                continue
            
            # 保存为 Medusa 训练格式
            results.append({
                "id": f"chinese_{idx}",
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": response}
                ]
            })
            
            # 每 100 条保存一次
            if (idx + 1) % 100 == 0:
                with open(OUTPUT_DATA, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"\nSaved {len(results)} samples, failed: {failed_count}")
                
        except Exception as e:
            print(f"\nError at index {idx}: {e}")
            failed_count += 1
            continue
    
    # 最终保存
    with open(OUTPUT_DATA, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== Generation Complete ===")
    print(f"Total samples generated: {len(results)}")
    print(f"Failed samples: {failed_count}")
    print(f"Output saved to: {OUTPUT_DATA}")

if __name__ == "__main__":
    main()
