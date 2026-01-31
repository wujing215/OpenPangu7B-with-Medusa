# 基于 Medusa 投机解码的 OpenPangu-7B 在 Ascend NPU 的推理加速实现

> **OpenPangu-7B 基础模型**请参考：  
> 👉 https://atomgit.com/ascend-tribe/openPangu-Embedded-7B-V1.1.git



本仓库围绕 **OpenPangu-7B**，提供了一套**基于 Medusa 的端到端投机推理（Speculative Inference）加速实现**，面向 **昇腾（Ascend）硬件平台**，对大模型推理阶段的自回归解码进行优化加速。

---

## 1. 项目背景

在标准自回归解码中，大模型需要逐 Token 生成文本，每一步都执行完整前向计算，实际推理性能往往受限于显存带宽。

**投机推理（Speculative Inference）** 通过在一次前向传播中预测并验证多个 Token，以减少解码步数，是当前提升大模型推理效率的重要方向之一。

**本仓库项目目标：**

- 在昇腾架构上实现端到端的 Medusa 投机推理优化 。
- 构建 OpenPangu-with-Medusa，提升推理吞吐量 。



## 2. Medusa 方法概览

Medusa 在主模型输出的隐藏状态上引入多个轻量级预测头（Medusa Heads）：

- 不同 Head 预测不同步长的未来 Token
- 多个 Head 的预测结果构成候选 Token 集合

在一次解码迭代中：

1. Medusa Heads 并行生成候选 Token
2. 构造候选 Token 树
3. 通过 Tree Attention 一次性验证所有候选
4. 接受最长可行 Token 前缀并继续解码

该方式在保证生成一致性的前提下，有效减少了解码轮数。



## 3. 投机推理实现

本仓库实现了**完整可用的 Medusa 推理流程**，主要包含：

- **Prefill 阶段**：处理输入 Prompt
- **候选生成**：Medusa Heads 预测未来 Token
- **Tree Attention 解码**：并行验证候选 Token
- **后验评估与状态更新**：确定可接受 Token 并进入下一轮

整体接口对用户仍表现为标准的文本生成流程。



## 4. Ascend 平台工程化

针对昇腾硬件特性，仓库在实现中对投机推理流程进行了工程化优化：

- 将候选 Token 树及相关索引提前构建为**静态 Tensor 结构**
- 使用固定 Attention Mask 与索引映射，减少动态控制流
- 降低 Host–Device 交互开销，适配 Ascend 图执行模式

上述设计使 Medusa 投机推理能够稳定运行在 Ascend 平台上。



## 5. 代码目录结构

```text
OpenPangu7B-with-Medusa/
├── 核心模型文件
│   ├── config.json
│   ├── configuration_openpangu_dense.py
│   ├── modeling_openpangu_dense.py
│   ├── modular_openpangu_dense.py
│   ├── tokenization_openpangu.py
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
│
├── Medusa 相关实现
│   ├── medusa_model.py                  # Medusa 核心推理实现
│   ├── medusa_compat.py                 # Transformers 兼容适配
│   ├── medusa_choices.py                # 投机推理候选树配置
│   └── third_party/
│       └── Medusa/                      # Medusa 原始实现
│
├── inference/
│   ├── generate.py                      # 基础模型推理
│   ├── medusa_generate.py               # Medusa 投机推理（Ascend）
│   └── benchmark.py                     # 推理性能测试
│
├── train/
│   ├── train_medusa.py                  # Medusa Heads 训练
│   ├── train_medusa_5heads.sh
│   └── medusa_tree_builder.py
│
├── patches/
│   └── medusa_transformers_compat.patch
│
├── deepspeed.json
├── generation_config.json
└── apply_patches.sh
```




## 6. 实验结果

### 6.1 加速效果

| 解码长度 | Accept Rate | Speedup |
|:--------:|:-----------:|:-------:|
| 64       | 1.84        | **1.43×** |
| 128      | 1.78        | 1.32×   |
| 256      | 1.69        | 1.13×   |

### 6.2 结果分析

- **端到端加速显著**：在短序列生成（Decode Length = 64）场景下，获得最高 **1.43 倍**加速比，有效利用了昇腾 NPU 的空闲算力。

- **解码长度的影响**：随着生成序列变长，Accept Rate 从 1.84 下降至 1.69，导致加速比相应回落。这是因为上下文增长带来的文本分布不确定性增大，轻量级 Medusa Head 的预测准确率略有下降。

- **静态图优化是关键**：得益于静态候选树构建与零拷贝路径回溯机制，在 Ascend 910B 上将初始计算开销（Overhead）控制在较低水平，使得解码长度 < 1024 的场景均能实现正向加速。

### 6.3 结论

实验证明 OpenPangu-with-Medusa 方案在昇腾硬件上是有效的，**特别适合中短文本生成的低延迟场景**。



## 7. 环境部署与使用指南

### 7.1 环境准备

```bash
git clone https://github.com/wujing215/OpenPangu7B-with-Medusa.git
cd OpenPangu7B-with-Medusa/third_party
git clone https://github.com/FasterDecoding/Medusa.git
pip install -e .

```

### 7.2 运行方法

 ```Bash
 cd OpenPangu7B-with-Medusa
 # 单次推理
 python inference/medusa_generate.py --device npu \    # 选择运行设备
     --base_model /path/to/openpangu \    # 基础模型权重文件所在路径
     --medusa_dir /path/to/medusa_head \    # Medusa Heads权重文件所在路径
     --prompt xxxx    # xxxx为用户单次提问输入
 # 交互式推理
 python inference/medusa_generate.py --device npu \    # 选择运行设备
     --base_model /path/to/openpangu \    # 基础模型权重文件所在路径
     --medusa_dir /path/to/medusa_head \    # Medusa Heads权重文件所在路径
     --interactive    # 启动交互式连续问答模式
 # benchmark
 python inference/benchmark.py \
     --base_model /path/to/openpangu \    # 基础模型权重文件所在路径
     --medusa_dir /path/to/medusa_head     # Medusa Heads权重文件所在路径
 ```

 - 支持huggingface权重加载：

 ```bash
 # 通过huggingface仓库加载
 python inference/medusa_generate.py --device npu \    # 选择运行设备
     --base_model Ivy0525/openPangu7B-with-Medusa \    # hf仓库名
     --medusa_dir Ivy0525/openPangu7B-with-Medusa \    # hf仓库名
     --prompt "Give me a short intruduction to LLM."    # 示例prompt
 ```





