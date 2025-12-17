# coding=utf-8
# Copyright (c) 2025
# Medusa 投机推理脚本 for OpenPangu-Embedded-7B
# 
# 基于 Medusa 的投机解码（Speculative Decoding）流程：
# 1. 多个 Medusa heads 并行预测未来多个位置的 token
# 2. 构建候选树，每个路径代表一种可能的 token 序列
# 3. 使用基础模型验证候选，接受正确的前缀
# 4. 一次前向传播可能接受多个 token，加速推理
#
# 使用方法:
#   python medusa_generate.py --prompt "你的问题"
#   python medusa_generate.py --interactive  # 交互模式

import torch
import sys
import os
import argparse
from pathlib import Path

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(parent_dir))

# 导入 Medusa 模型（已集成 Pangu 支持）
from medusa_model import MedusaModel, MedusaConfig
from medusa_choices import pangu_stage2
from medusa_choices import pangu_5heads_top10
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_file


class MedusaPanguInference:
    """
    Medusa 投机推理封装类
    
    核心流程（在 medusa_generate 函数中）:
    1. initialize_medusa: 初始化 KV cache 和树状注意力掩码
    2. generate_candidates: 从 Medusa heads 生成候选 token 树
    3. tree_decoding: 使用树状注意力一次验证所有候选
    4. evaluate_posterior: 评估候选序列，选择最佳接受前缀
    5. update_inference_inputs: 更新输入序列和 KV cache
    """
    
    def __init__(
        self,
        base_model_path: str,
        medusa_head_path: str,
        tokenizer_path: str = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
        medusa_num_heads: int = 5,
        medusa_num_layers: int = 1,
    ):
        self.device = device
        self.dtype = dtype
        self.medusa_num_heads = medusa_num_heads
        
        # 判断是本地路径还是 HuggingFace repo ID
        def is_local_path(path):
            """检查是否是本地路径（而非 HF repo ID）"""
            # 如果包含路径分隔符且以 . 或 / 开头，或者路径存在，则是本地路径
            return (
                path.startswith('.') or 
                path.startswith('/') or 
                path.startswith('~') or
                os.path.exists(path)
            )
        
        # 只对本地路径进行解析，HF repo ID 保持原样
        if is_local_path(base_model_path):
            base_model_path = str(Path(base_model_path).expanduser().resolve())
            local_files_only = True
        else:
            local_files_only = False
            
        if is_local_path(medusa_head_path):
            medusa_head_path = str(Path(medusa_head_path).expanduser().resolve())
        
        tokenizer_path = tokenizer_path or (os.path.dirname(medusa_head_path) if is_local_path(medusa_head_path) else medusa_head_path)
        
        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=False,
            trust_remote_code=True,
            local_files_only=local_files_only if is_local_path(tokenizer_path) else False,
        )
        
        # 加载基础模型配置并添加 Medusa 参数
        print(f"Loading config from {base_model_path}...")
        config = AutoConfig.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )
        config.medusa_num_heads = medusa_num_heads
        config.medusa_num_layers = medusa_num_layers
        
        # 加载带 Medusa heads 的模型
        print(f"Loading MedusaModelPangu from {base_model_path}...")
        
        # 直接使用 MedusaModelPangu 类
        from medusa_model import MedusaModelPangu
        self.model = MedusaModelPangu.from_pretrained(
            base_model_path,
            config=config,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        
        # 加载训练好的 Medusa heads 权重
        print(f"Loading Medusa heads from {medusa_head_path}...")
        
        # 判断是本地文件还是 HF repo
        if is_local_path(medusa_head_path):
            # 本地文件
            if medusa_head_path.endswith('.safetensors'):
                medusa_state_dict = load_file(medusa_head_path)
            elif medusa_head_path.endswith('.pt'):
                medusa_state_dict = torch.load(medusa_head_path, map_location='cpu')
            else:
                # 尝试两种格式
                safetensors_path = os.path.join(medusa_head_path, 'medusa_lm_head.safetensors')
                pt_path = os.path.join(medusa_head_path, 'medusa_lm_head.pt')
                if os.path.exists(safetensors_path):
                    medusa_state_dict = load_file(safetensors_path)
                elif os.path.exists(pt_path):
                    medusa_state_dict = torch.load(pt_path, map_location='cpu')
                else:
                    raise FileNotFoundError(f"Cannot find medusa_lm_head.safetensors or .pt in {medusa_head_path}")
        else:
            # HF repo - 下载文件
            from huggingface_hub import hf_hub_download
            try:
                local_path = hf_hub_download(repo_id=medusa_head_path, filename="medusa_lm_head.safetensors")
                medusa_state_dict = load_file(local_path)
            except:
                local_path = hf_hub_download(repo_id=medusa_head_path, filename="medusa_lm_head.pt")
                medusa_state_dict = torch.load(local_path, map_location='cpu')
        
        self.model.medusa_head.load_state_dict(medusa_state_dict, strict=False)
        self.model.eval()
        
        # 设置 tokenizer
        self.model.tokenizer = self.tokenizer
        
        print(f"Model loaded successfully! Device: {device}, Dtype: {dtype}")
        print(f"Medusa config: {medusa_num_heads} heads, {medusa_num_layers} layers")

    def apply_chat_template(self, messages: list) -> str:
        """应用 OpenPangu 的对话模板"""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def generate(
        self,
        prompt: str,
        max_steps: int = 512,
        temperature: float = 0.0,
        medusa_choices: list = None,
        posterior_threshold: float = 0.09,
        posterior_alpha: float = 0.3,
        top_p: float = 0.8,
        sampling: str = 'typical',
        fast: bool = True,
        stream: bool = False,
    ):
        """
        使用 Medusa 投机解码生成文本
        
        Args:
            prompt: 输入文本（已格式化）
            max_steps: 最大生成步数
            temperature: 采样温度（0=greedy）
            medusa_choices: 投机解码树配置，None 使用 pangu_stage2
            posterior_threshold: 后验验证阈值
            posterior_alpha: 后验验证 alpha 参数
            top_p: nucleus 采样阈值
            sampling: 采样策略 ('typical' 或 'nucleus')
            fast: 是否使用快速解码
            stream: 是否流式输出
            
        Returns:
            生成的文本，或流式生成器
        """
        # 使用 pangu_5heads_top10 作为默认配置（适合 5 个 Medusa heads）
        if medusa_choices is None:
            medusa_choices = pangu_5heads_top10
        
        # Tokenize 输入
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.model.base_model.device)
        
        # 调用 Medusa 的投机生成函数
        generator = self.model.medusa_generate(
            input_ids,
            temperature=temperature,
            max_steps=max_steps,
            medusa_choices=medusa_choices,
            posterior_threshold=posterior_threshold,
            posterior_alpha=posterior_alpha,
            top_p=top_p,
            sampling=sampling,
            fast=fast,
        )
        
        if stream:
            # 流式输出：返回生成器
            return self._stream_generate(generator)
        else:
            # 非流式：收集所有输出
            final_text = ""
            for output in generator:
                final_text = output["text"]
            return final_text
    
    def _stream_generate(self, generator):
        """流式生成的辅助方法"""
        for output in generator:
            yield output["text"]


def parse_output(output_text: str) -> dict:
    """解析 OpenPangu 的输出，分离 thinking 和 content"""
    result = {"raw": output_text, "thinking": "", "content": ""}
    
    # 尝试解析 thinking content
    if "[unused16]" in output_text and "[unused17]" in output_text:
        try:
            thinking = output_text.split("[unused17]")[0].split("[unused16]")[-1].strip()
            result["thinking"] = thinking
        except:
            pass
    
    # 尝试解析主要 content
    if "[unused17]" in output_text:
        try:
            content = output_text.split("[unused17]")[-1].split("[unused10]")[0].strip()
            result["content"] = content
        except:
            pass
    elif "[unused10]" in output_text:
        try:
            content = output_text.split("[unused10]")[0].strip()
            for marker in ["[unused9]助手：", "助手："]:
                if content.startswith(marker):
                    content = content[len(marker):]
            result["content"] = content
        except:
            pass
    
    # 如果没有解析到 content，使用原始文本
    if not result["content"]:
        result["content"] = output_text
    
    return result


def main():
    parser = argparse.ArgumentParser(description="OpenPangu + Medusa 投机推理")
    parser.add_argument("--base_model", type=str, default="~/work/openPangu-Embedded-7B-V1.1",
                        help="Base OpenPangu model path")
    parser.add_argument("--medusa_dir", type=str, 
                        default="~/work/openPangu-Embedded-7B-V1.1/medusa_5heads_lr0.001_layers1_medusa_mlp_._medusa_5_lr_0.001_layers_1",
                        help="Directory containing medusa_lm_head.safetensors")
    parser.add_argument("--medusa_file", type=str, default="medusa_lm_head.safetensors",
                        help="Medusa weights filename")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_steps", type=int, default=512,
                        help="Maximum generation steps")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0=greedy)")
    parser.add_argument("--top_p", type=float, default=0.8,
                        help="Nucleus sampling threshold")
    parser.add_argument("--posterior_threshold", type=float, default=0.09,
                        help="Medusa posterior validation threshold")
    parser.add_argument("--posterior_alpha", type=float, default=0.3,
                        help="Medusa posterior alpha (recommended: sqrt(threshold))")
    parser.add_argument("--sampling", type=str, default="typical",
                        choices=["typical", "nucleus"],
                        help="Sampling strategy")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive chat mode")
    parser.add_argument("--stream", action="store_true",
                        help="Stream output tokens")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt for non-interactive mode")
    args = parser.parse_args()

    # 判断是本地路径还是 HuggingFace repo ID
    def is_local_path(path):
        """检查是否是本地路径（而非 HF repo ID）"""
        return (
            path.startswith('.') or 
            path.startswith('/') or 
            path.startswith('~') or
            (os.path.exists(path) and not '/' in path.split('/')[-1])  # 避免 user/repo 被误判
        )
    
    # 解析路径 - 只对本地路径使用 resolve()
    if is_local_path(args.base_model):
        base_model_path = str(Path(args.base_model).expanduser().resolve())
    else:
        base_model_path = args.base_model
        
    if is_local_path(args.medusa_dir):
        medusa_dir = str(Path(args.medusa_dir).expanduser().resolve())
        medusa_head_path = os.path.join(medusa_dir, args.medusa_file)
        tokenizer_path = medusa_dir
    else:
        medusa_dir = args.medusa_dir
        medusa_head_path = args.medusa_dir  # HF repo，让 __init__ 处理下载
        tokenizer_path = args.medusa_dir

    # 加载模型
    model = MedusaPanguInference(
        base_model_path=base_model_path,
        medusa_head_path=medusa_head_path,
        tokenizer_path=tokenizer_path,
        device=args.device,
        dtype=torch.float16,
        medusa_num_heads=5,
        medusa_num_layers=1,
    )

    # 系统提示
    sys_prompt = (
        "你必须严格遵守法律法规和社会道德规范。"
        "生成任何内容时，都应避免涉及暴力、色情、恐怖主义、种族歧视、性别歧视等不当内容。"
    )

    if args.interactive:
        # 交互模式
        print("\n" + "="*60)
        print("OpenPangu + Medusa 投机推理 (Speculative Decoding)")
        print("="*60)
        print("命令: 输入 'exit' 或 'quit' 退出, 'clear' 清空")
        print(f"配置: temperature={args.temperature}, max_steps={args.max_steps}")
        print(f"Medusa: posterior_threshold={args.posterior_threshold}, "
              f"posterior_alpha={args.posterior_alpha}")
        print("="*60 + "\n")

        while True:
            try:
                user_input = input("用户: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n退出...")
                break

            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print("退出...")
                break
            if user_input.lower() == 'clear':
                print("已清空")
                continue

            # 构建消息
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_input},
            ]
            prompt = model.apply_chat_template(messages)

            print("\n助手: ", end="", flush=True)

            if args.stream:
                # 流式输出
                last_text = ""
                for text in model.generate(
                    prompt,
                    max_steps=args.max_steps,
                    temperature=args.temperature,
                    posterior_threshold=args.posterior_threshold,
                    posterior_alpha=args.posterior_alpha,
                    top_p=args.top_p,
                    sampling=args.sampling,
                    stream=True,
                ):
                    # 增量输出
                    new_text = text[len(last_text):]
                    print(new_text, end="", flush=True)
                    last_text = text
                print()
            else:
                # 非流式输出
                output = model.generate(
                    prompt,
                    max_steps=args.max_steps,
                    temperature=args.temperature,
                    posterior_threshold=args.posterior_threshold,
                    posterior_alpha=args.posterior_alpha,
                    top_p=args.top_p,
                    sampling=args.sampling,
                    stream=False,
                )
                parsed = parse_output(output)
                if parsed["thinking"]:
                    print(f"\n[思考]: {parsed['thinking']}")
                print(f"{parsed['content']}")
            
            print()

    else:
        # 单次推理模式
        if args.prompt is None:
            args.prompt = "Give me a short introduction to large language model."

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": args.prompt},
        ]
        prompt = model.apply_chat_template(messages)

        print(f"\n输入: {args.prompt}")
        print("-" * 40)

        if args.stream:
            print("回答: ", end="", flush=True)
            last_text = ""
            for text in model.generate(
                prompt,
                max_steps=args.max_steps,
                temperature=args.temperature,
                posterior_threshold=args.posterior_threshold,
                posterior_alpha=args.posterior_alpha,
                top_p=args.top_p,
                sampling=args.sampling,
                stream=True,
            ):
                new_text = text[len(last_text):]
                print(new_text, end="", flush=True)
                last_text = text
            print()
        else:
            output = model.generate(
                prompt,
                max_steps=args.max_steps,
                temperature=args.temperature,
                posterior_threshold=args.posterior_threshold,
                posterior_alpha=args.posterior_alpha,
                top_p=args.top_p,
                sampling=args.sampling,
                stream=False,
            )
            parsed = parse_output(output)
            if parsed["thinking"]:
                print(f"\n思考过程:\n{parsed['thinking']}")
            print(f"\n回答:\n{parsed['content']}")


if __name__ == "__main__":
    main()
