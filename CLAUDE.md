# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

MiniMind 是一个从零开始训练超小语言模型的开源项目，仅用3块钱成本 + 2小时即可训练出仅为25.8M的模型。项目实现了完整的训练流程，包括数据预处理、预训练、监督微调(SFT)、LoRA微调、直接偏好优化(DPO)、强化学习(RLAIF)和知识蒸馏等。

## 常用开发命令

### 环境设置
```bash
# 安装依赖
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

# 检查CUDA可用性
python -c "import torch; print(torch.cuda.is_available())"
```

### 模型推理测试
```bash
# 加载原生torch权重模型测试
python eval_llm.py --weight full_sft

# 加载transformers格式模型
python eval_llm.py --load_from ./MiniMind2

# 启动Web UI界面（需要Python >= 3.10）
streamlit run scripts/web_demo.py

# 启动OpenAI兼容API服务
python scripts/serve_openai_api.py

# 测试API服务
python scripts/chat_openai_api.py
```

### 模型训练流程
```bash
# 所有训练脚本都在 trainer 目录下执行
cd trainer

# 1. 预训练（学知识）
python train_pretrain.py
# 或多卡训练
torchrun --nproc_per_node N train_pretrain.py

# 2. 监督微调（学对话）
python train_full_sft.py

# 3. LoRA微调（领域适配）
python train_lora.py

# 4. 直接偏好优化
python train_dpo.py

# 5. 强化学习训练
python train_ppo.py    # PPO算法
python train_grpo.py   # GRPO算法
python train_spo.py    # SPO算法

# 6. 推理模型蒸馏
python train_distill_reason.py

# 7. 知识蒸馏
python train_distillation.py
```

### 断点续训
所有训练脚本都支持断点续训，只需添加 `--from_resume 1` 参数：
```bash
python train_pretrain.py --from_resume 1
```
检查点文件保存在 `./checkpoints/` 目录，支持跨不同GPU数量恢复训练。

### 实验记录
```bash
# 使用wandb记录训练过程（需要能直连）
python train_pretrain.py --use_wandb

# 使用swanlab记录（国内推荐，API完全兼容wandb）
# 默认已使用swanlab，无需额外配置
```

## 核心架构

### 模型架构特点
- **基于Transformer Decoder-Only**：与Llama3.1架构相同
- **超轻量级设计**：模型参数范围26M-145M
- **自定义分词器**：仅6400词汇大小（vs Llama3的128K）
- **关键技术组件**：
  - RMSNorm归一化
  - SwiGLU激活函数
  - 旋转位置编码(RoPE)
  - Flash Attention支持
  - YaRN算法支持长文本外推
  - 混合专家(MoE)支持

### 目录结构说明
```
minimind/
├── model/                 # 模型架构定义
│   ├── model_minimind.py  # 核心模型实现
│   ├── model_lora.py      # LoRA实现
│   └── tokenizer.*        # 分词器文件
├── trainer/               # 训练脚本
│   ├── train_pretrain.py  # 预训练
│   ├── train_full_sft.py  # 监督微调
│   ├── train_lora.py      # LoRA微调
│   ├── train_dpo.py       # DPO训练
│   ├── train_ppo.py       # PPO强化学习
│   ├── train_grpo.py      # GRPO强化学习
│   ├── train_spo.py       # SPO强化学习
│   └── train_*.py         # 其他训练方法
├── dataset/               # 数据处理
│   └── lm_dataset.py      # 数据加载和预处理
├── scripts/               # 工具脚本
│   ├── web_demo.py        # Streamlit Web界面
│   ├── serve_openai_api.py # OpenAI兼容API
│   ├── convert_model.py   # 模型格式转换
│   └── train_tokenizer.py # 分词器训练
└── out/                   # 模型输出目录
    └── *.pth              # 训练好的权重文件
```

### 模型配置
模型配置在 `model/model_minimind.py` 中的 `MiniMindConfig` 类：

- **MiniMind2-Small (26M)**: hidden_size=512, num_layers=8
- **MiniMind2 (104M)**: hidden_size=768, num_layers=16
- **MiniMind2-MoE (145M)**: 启用MoE，shared=1, routed=4

### 数据格式
- **预训练数据**: `{"text": "文本内容"}`
- **SFT数据**:
  ```json
  {
    "conversations": [
      {"role": "user", "content": "问题"},
      {"role": "assistant", "content": "回答"}
    ]
  }
  ```
- **DPO数据**: 包含chosen和rejected字段
- **推理模型数据**: 使用 `<think>` 和 `<answer>` 标签

## 开发注意事项

### 训练配置
- 默认每隔100步自动保存模型到 `./out/` 目录（每次覆盖旧文件）
- 支持单机多卡DDP训练：`torchrun --nproc_per_node N train_xxx.py`
- 自动保存检查点到 `./checkpoints/` 目录，命名格式：`<权重名>_<维度>_resume.pth`
- 支持跨GPU数量恢复训练
- 推理时支持历史对话：`--historys 2`（偶数轮数）

### 推理部署
- 项目完全兼容Hugging Face transformers
- 支持llama.cpp、vLLM、Ollama等推理引擎
- 提供OpenAI兼容API服务
- 支持RoPE长度外推（通过YaRN算法）

### 性能优化
- 默认开启Flash Attention
- 支持混合精度训练
- 动态批处理
- 梯度检查点选项

## 数据集准备

数据集需要下载到 `./dataset/` 目录：
- `pretrain_hq.jsonl` - 预训练数据（必需）
- `sft_mini_512.jsonl` - 快速SFT数据（必需）
- `sft_512.jsonl` - 完整SFT数据
- `dpo.jsonl` - DPO训练数据
- `r1_mix_1024.jsonl` - 推理模型数据
- `rlaif-mini.jsonl` - 强化学习数据

## 第三方集成

### WandB/SwanLab集成
- 使用 `--use_wandb` 参数启用训练可视化
- 国内环境默认使用SwanLab（无需特殊配置）
- 自动记录loss、learning rate、eval metrics等

### 模型格式转换
```bash
# torch -> transformers
python scripts/convert_model.py

# 转换后可在transformers生态中使用
# 如vLLM、llama.cpp等
```

### 第三方推理引擎
```bash
# vLLM部署（推荐）
vllm serve ./MiniMind2 --model-impl transformers --served-model-name "minimind"

# llama.cpp（需要先转换为GGUF格式）
# 参考README中的详细步骤

# Ollama（一键运行）
ollama run jingyaogong/minimind2
```

## 调试技巧

### 检查模型输出
```bash
# 查看模型参数
python eval_llm.py --weight pretrain --max_new_tokens 100

# 携带历史对话
python eval_llm.py --weight full_sft --historys 2

# 调整生成参数
python eval_llm.py --weight dpo --temperature 0.7 --top_p 0.9
```

### 常见问题
- 确保数据集文件放在 `./dataset/` 目录
- 检查CUDA是否可用：`python -c "import torch; print(torch.cuda.is_available())"`
- 多卡训练使用 `torchrun` 而非 `python`
- 断点续训确保检查点文件存在于 `./checkpoints/` 目录
- 模型文件路径：原生权重在 `./out/*.pth`，transformers格式需要先转换
- 注意：项目本身没有传统的单元测试，主要通过模型输出效果来验证