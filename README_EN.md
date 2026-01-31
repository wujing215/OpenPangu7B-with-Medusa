# Medusa-based Speculative Decoding Acceleration for OpenPangu-7B on Ascend NPU

> **Base OpenPangu-7B model**:  
> 👉 https://atomgit.com/ascend-tribe/openPangu-Embedded-7B-V1.1.git

---

This repository provides an **end-to-end Medusa-based speculative inference acceleration implementation** for **OpenPangu-7B**, targeting the **Ascend hardware platform**. The goal is to optimize autoregressive decoding during large language model inference.

---

## 1. Background

In standard autoregressive decoding, large language models generate tokens sequentially, requiring a full forward pass at each step. In practice, inference performance is often limited by memory bandwidth.

**Speculative Inference** reduces the number of decoding iterations by predicting and validating multiple future tokens within a single forward pass.

**Objectives:**

- Enable end-to-end Medusa speculative decoding optimization on Ascend architecture.
- Develop OpenPangu-with-Medusa for improved inference throughput.

---

## 2. Overview of Medusa

Medusa augments the backbone model with multiple lightweight prediction heads (Medusa Heads):

- Each head predicts tokens at different future offsets
- Predictions jointly form a set of candidate tokens

Within one decoding iteration:

1. Medusa Heads generate candidate tokens in parallel  
2. A candidate token tree is constructed  
3. Tree Attention validates all candidates in a single forward pass  
4. The longest valid token prefix is accepted  

This approach reduces decoding steps while preserving output correctness.

---

## 3. Speculative Inference Workflow

This repository implements a **complete Medusa inference workflow**, including:

- **Prefill stage** for prompt processing  
- **Candidate generation** via Medusa Heads  
- **Tree Attention decoding** for parallel validation  
- **Posterior evaluation and state update**

From the user perspective, the interface remains consistent with standard text generation.

---

## 4. Ascend-Oriented Engineering

To adapt speculative inference to Ascend hardware, the implementation includes:

- **Static tensor** representations of candidate token trees
- Fixed attention masks to reduce dynamic control flow
- Minimized host–device interaction for graph execution

These optimizations enable stable Medusa-based inference on Ascend platforms.

---

## 5. Code Structure

```text
OpenPangu7B-with-Medusa/
├── Core model files
│   ├── config.json
│   ├── configuration_openpangu_dense.py
│   ├── modeling_openpangu_dense.py
│   ├── modular_openpangu_dense.py
│   ├── tokenization_openpangu.py
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
│
├── Medusa implementation
│   ├── medusa_model.py
│   ├── medusa_compat.py
│   ├── medusa_choices.py
│   └── third_party/Medusa/
│
├── inference/
│   ├── generate.py
│   ├── medusa_generate.py
│   └── benchmark.py
│
├── train/
│   ├── train_medusa.py
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

---

## 6. Experimental Results


- **Short to medium-length generation** achieves about 1.3×–1.4× end-to-end speedup
- **Accept rate** decreases slowly as generation length increases
- **Long-sequence gains are limited** due to additional validation overhead

This approach is best suited for latency-sensitive workloads with moderate output lengths.

---

## 7. Environment Setup and Usage

### 7.1 Environment

```bash
git clone https://github.com/wujing215/OpenPangu7B-with-Medusa.git
cd OpenPangu7B-with-Medusa/third_party
git clone https://github.com/FasterDecoding/Medusa.git
pip install -e .

```

### 7.2 Inference Examples

> ```Bash
> cd OpenPangu7B-with-Medusa
> # Single inference
> python inference/medusa_generate.py --device npu \    # Select device
>     --base_model /path/to/openpangu \    # Path to base model weights
>     --medusa_dir /path/to/medusa_head \    # Path to Medusa Heads weights
>     --prompt xxxx    # User input prompt
> # Interactive inference
> python inference/medusa_generate.py --device npu \    # Select device
>     --base_model /path/to/openpangu \    # Path to base model weights
>     --medusa_dir /path/to/medusa_head \    # Path to Medusa Heads weights
>     --interactive    # Enable interactive Q&A mode
> # Benchmark
> python inference/benchmark.py \
>     --base_model /path/to/openpangu \    # Path to base model weights
>     --medusa_dir /path/to/medusa_head     # Path to Medusa Heads weights
> ```
>
> - Loading from HuggingFace:
>
> ```bash
# Load from HuggingFace repository
python inference/medusa_generate.py --device npu \    # Select device
    --base_model Ivy0525/openPangu7B-with-Medusa \    # HF repo name
    --medusa_dir Ivy0525/openPangu7B-with-Medusa \    # HF repo name
    --prompt "Give me a short intruduction to LLM."    # Example prompt
> ```

---



