import torch
import torch.nn as nn
import sys
import os

# 使用兼容层初始化 Medusa（处理 transformers 版本兼容性）
_current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _current_dir)
import medusa_compat  # 必须在导入 medusa 之前

# 从 third_party/Medusa 导入 Llama 和 Mistral 的 KV 版本
from medusa.model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from medusa.model.modeling_mistral_kv import MistralForCausalLM as KVMistralForCausalLM

# 从当前目录导入 Pangu 模型（已修改支持 Medusa）
from modeling_openpangu_dense import PanguEmbeddedForCausalLM

from transformers import PreTrainedModel, PretrainedConfig

# 从 third_party/Medusa 导入工具函数和 KV Cache
from medusa.model.utils import *
from medusa.model.kv_cache import initialize_past_key_values

# 从当前目录导入修改过的 medusa_choices（包含 Pangu 配置）
from medusa_choices import *

from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download
import warnings



def evaluate_posterior_test(
    logits, candidates, temperature, posterior_threshold=0.3, posterior_alpha=0.09, top_p=0.8, sampling='typical', fast=True
):
    """
    【测试专用 - 强制接受模式】
    强制接受固定长度的 token,用于测试理想情况下的最大吞吐量
    
    策略:
    1. 对于每条 candidate 路径,计算它与 base model 预测的匹配度
    2. 选择匹配最长的路径
    3. 强制接受该路径的前 FORCE_ACCEPT_LEN+1 个 token
    
    Args:
    - logits (torch.Tensor): shape [num_paths, tree_depth, vocab_size]
    - candidates (torch.Tensor): shape [num_paths, tree_depth]
    """
    # ================= 配置区 =================
    # 论文中 5 heads 的平均接受长度约为 3（包含 base model 的 1 个 + medusa 的 2 个）
    # 当前只有 3 heads,尝试强制接受 2 个额外 token
    FORCE_ACCEPT_LEN = 2
    # =========================================
    
    # 【调试】打印形状信息（第一次调用时）
    if not hasattr(evaluate_posterior_test, '_debug_printed'):
        print(f"\n[DEBUG evaluate_posterior_test]")
        print(f"  logits.shape = {logits.shape}")
        print(f"  candidates.shape = {candidates.shape}")
        print(f"  FORCE_ACCEPT_LEN = {FORCE_ACCEPT_LEN}")
        evaluate_posterior_test._debug_printed = True
    
    # 计算每条路径与 greedy 预测的匹配度
    # logits[:, :-1] 是所有路径的前 n-1 个位置
    # torch.argmax(logits[:, :-1], dim=-1) 得到每个位置的 greedy token
    # 形状: [num_paths, tree_depth-1]
    greedy_tokens = torch.argmax(logits[:, :-1], dim=-1)
    
    # candidates[:, 1:] 是每条路径的第 2 到最后一个 token
    # 形状: [num_paths, tree_depth-1]
    
    # 比较是否匹配
    posterior_mask = (candidates[:, 1:] == greedy_tokens).int()
    
    # 计算每条路径的连续匹配长度
    # torch.cumprod 会在第一个 0 之后全变成 0
    candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
    
    # 找到匹配最长的路径
    # 如果有多条路径长度相同,选第一条
    accept_length = candidates_accept_length.max().item()
    
    if accept_length == 0:
        # 没有任何匹配,只接受 base model 的 1 个 token
        best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        accept_length = 0
    else:
        # 强制接受 min(FORCE_ACCEPT_LEN, accept_length) 个额外 token
        accept_length = min(FORCE_ACCEPT_LEN, accept_length)
        # 在所有能接受这么多 token 的路径中,选第一条
        valid_candidates = (candidates_accept_length >= accept_length).nonzero(as_tuple=False)
        best_candidate = valid_candidates[0].item()
        best_candidate = torch.tensor(best_candidate, dtype=torch.long, device=candidates.device)
    
    return best_candidate, accept_length


class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model.

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 2.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        medusa_num_heads=5,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path

class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class MedusaModelABC(nn.Module):
    """The Medusa Language Model Head.

    This module creates a series of prediction heads (based on the 'medusa' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    # Load the base model
    # base_model_prefix = "model"
    # supports_gradient_checkpointing = True
    # _no_split_modules = ["LlamaDecoderLayer", "MistralDecoderLayer"]
    # _skip_keys_device_placement = "past_key_values"
    # _supports_flash_attn_2 = True

    def __init__(
        self,
        config,
    ):
        """
        Args:
            config (PretrainedConfig): The configuration of the MedusaModel.
        """
        super().__init__(config)
        # For compatibility with the old APIs

        medusa_num_heads = config.medusa_num_heads
        medusa_num_layers = config.medusa_num_layers
        base_model_name_or_path = config._name_or_path
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.medusa = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        # [Modified] Tokenizer will be set later in from_pretrained or by the user
        # self.tokenier = AutoTokenizer.from_pretrained(base_model_name_or_path)
        self.tokenizer = None
        # Create a list of Medusa heads
        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * medusa_num_layers),
                    nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                )
                for _ in range(medusa_num_heads)
            ]
        )
    # Add a link named base_model to self
    @property
    def base_model(self):
        return self
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # [New] 如果已经传入了 config，直接使用它，否则尝试加载
        if 'config' in kwargs and kwargs['config'] is not None:
            # 直接使用传入的 config，调用父类的 from_pretrained
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        
        # Manually load config to ensure that the medusa_num_heads parameter is loaded
        trust_remote_code = kwargs.get('trust_remote_code', False)
        try:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, 
                trust_remote_code=trust_remote_code
            )
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
                config=config,
            )
        # [Modified] 兼容旧版 Medusa 配置加载
        except Exception as e:
            print(f"AutoConfig loading failed: {e}, trying MedusaConfig...")
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(
                config.base_model_name_or_path,
                trust_remote_code=trust_remote_code
            )
            base_model_config.medusa_num_heads = 5 # TODO: fix the uploaded config (only include 2 heads)
            base_model_config.medusa_num_layers = config.medusa_num_layers
            model = super().from_pretrained(
                config.base_model_name_or_path,
                *args,
                **kwargs,
                config=base_model_config,
            )
            medusa_head_path = os.path.join(pretrained_model_name_or_path, "medusa_lm_head.pt")
            if os.path.exists(medusa_head_path):
                filename = medusa_head_path
            else:
                filename = hf_hub_download(pretrained_model_name_or_path, "medusa_lm_head.pt")
            medusa_head_state_dict = torch.load(filename, map_location=model.device)
            model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)
            return model
        

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        medusa_forward=False,
        **kwargs,
    ):
        """Forward pass of the MedusaModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
        if not medusa_forward:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
        
        # [MODIFIED] Ensure position_ids is 2D [batch, seq_len] as expected by Pangu model
        if position_ids is not None and position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        
        with torch.inference_mode():
            # Pass input through the base model
            # 基础模型前向传播
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
            # 原始模型预测（当 output_orig=True 时）,使用原始的语言模型头对隐藏状态进行预测，得到原始的 token 预测 logits
            # outputs[0] = 最后一个transformer层的隐藏状态，outputs[1:] = 其他可选输出（past_key_values、attentions等）
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
        # Clone the output hidden states
        hidden_states = outputs[0].clone()  #outputs[0] 被克隆作为Medusa模型多个头的输入
        medusa_logits = []
        # TODO: Consider parallelizing this loop for efficiency?
        for i in range(self.medusa):
            # 对每个 Medusa Head 应用隐藏状态，得到每个头的预测 logits
            medusa_logits.append(self.medusa_head[i](hidden_states))
        if output_orig:
            return torch.stack(medusa_logits, dim=0), outputs, orig
        return torch.stack(medusa_logits, dim=0)
    def get_medusa_choice(self, model_name):
        if 'vicuna' in model_name:
            if '7b' in model_name:
                return vicuna_7b_stage2
            elif '13b' in model_name:
                return vicuna_13b_stage2
            elif '33b' in model_name:
                return vicuna_33b_stage2
        elif 'zephyr' in model_name:
            return zephyr_stage2
        elif 'pangu' in model_name:  # 新增Pangu配置
            return pangu_5heads_top10  # 见 medusa_choices.py
        warnings.warn('Please specify medusa choice configuration!')
        return mc_sim_7b_63

    def medusa_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the Medusa
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        medusa_choices=None,
        posterior_threshold=0.09,  # threshold validation of Medusa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
        top_p=0.8, 
        sampling = 'typical', 
        fast = True
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            medusa_choices (list, optional): A list of integers indicating the number of choices for each Medusa head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
            top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
            sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
            fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Cache medusa buffers (the fixed patterns for tree attention)
        if medusa_choices is None:
            medusa_choices = self.get_medusa_choice(self.base_model_name_or_path)

        if hasattr(self, "medusa_choices") and self.medusa_choices == medusa_choices:
            # Load the cached medusa buffer
            medusa_buffers = self.medusa_buffers
        else:
            # Initialize the medusa buffer
            # medusa_buffers 是一个字典，包含了 Medusa 树状注意力机制所需的预计算缓冲区
            medusa_buffers = generate_medusa_buffers(
                medusa_choices, device=self.base_model.device
            )
        self.medusa_buffers = medusa_buffers
        self.medusa_choices = medusa_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        reset_medusa_mode(self)
        # Initialize tree attention mask and process prefill tokens
        # medusa_logits是 Medusa Heads 对每个token的预测logits，logits是原始 LM Head 对每个token的预测logits
        medusa_logits, logits = initialize_medusa(
            input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values
        )

        new_token = 0
        last_round_token = 0

        for idx in range(max_steps):
            # Generate candidates with topk predictions from Medusa heads
            candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_buffers["tree_indices"],
                medusa_buffers["retrieve_indices"],
                temperature=temperature,
                posterior_alpha=posterior_alpha,
                posterior_threshold=posterior_threshold,
                top_p=top_p,
                sampling=sampling,
                fast=fast,
            )

            # Use tree attention to verify the candidates and get predictions
            medusa_logits, logits, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                medusa_buffers["medusa_position_ids"],
                input_ids,
                medusa_buffers["retrieve_indices"],
            )

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast
            )
            '''best_candidate, accept_length = evaluate_posterior_test(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling = sampling, fast = fast
            )'''

            # Update the input_ids and logits
            input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                outputs,
                logits,
                medusa_logits,
                new_token,
                past_key_values_data,
                current_length_data,
            )

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break


class MedusaModelLlama(MedusaModelABC, KVLlamaForCausalLM):
    pass

class MedusaModelMistral(MedusaModelABC, KVMistralForCausalLM):
    pass


class MedusaModelPangu(MedusaModelABC, PanguEmbeddedForCausalLM):
    """Medusa model for Pangu architecture"""
    pass

class MedusaModel():
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the medusa_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        except:
            # MEDUSA-v0.1 load
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            config.model_type = base_model_config.model_type

        if config.model_type == "llama":
            return MedusaModelLlama.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        elif config.model_type == "mistral":
            return MedusaModelMistral.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        elif config.model_type == "pangu":  # 新增Pangu支持
            return MedusaModelPangu.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        else:
            raise ValueError("Only support llama and mistral for now!!")
