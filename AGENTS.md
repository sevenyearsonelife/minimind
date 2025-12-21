# Repository Guidelines

## 项目结构与模块组织
- `model/`：模型结构与配置（MiniMind、LoRA、MoE 等实现）。
- `trainer/`：训练脚本（Pretrain、SFT、DPO、PPO/GRPO/SPO、蒸馏）。
- `dataset/`：数据集文件与说明（优先参考 `dataset/dataset.md`）。
- `scripts/`：工具脚本（tokenizer、Web Demo、OpenAI API 兼容服务）。
- `eval_llm.py`：本地推理/评估入口。
- `images/`：文档与可视化资源。

## 构建、测试与本地运行命令
- 安装依赖：`pip install -r requirements.txt`。
- 预训练/微调（在 `trainer/` 下执行）：
  - `python train_pretrain.py`
  - `python train_full_sft.py`
- 多卡训练示例：`torchrun --nproc_per_node N train_pretrain.py`。
- 推理评估：`python eval_llm.py --weight full_sft`。
- API 服务（可选）：`python scripts/serve_openai_api.py`。

## 编码风格与命名约定
- 以 Python 为主，保持 4 空格缩进，沿用现有脚本结构。
- 文件/函数使用 `snake_case`，类使用 `CamelCase`。
- 权重命名前缀保持一致：`pretrain_*`、`full_sft_*`、`dpo_*`、`reason_*`、`ppo_actor_*`、`grpo_*`、`spo_*`、`lora_xxx_*`。

## 数据与权重约定
- 数据文件放在 `./dataset/`，格式与路径以 `dataset/dataset.md` 为准。
- 训练过程会在 `./checkpoints/` 保存完整检查点；权重默认输出到 `./out/`。
- 评估前确保需要的 `*.pth` 权重在 `./out/` 内。

## 测试指南
- 仓库未提供专用测试框架或覆盖率要求。
- 变更后建议用 `eval_llm.py` 做最小推理验收；训练相关改动可用小数据/少步数验证是否可跑通。

## 提交与 Pull Request 指南
- 近期提交信息常见前缀为 `[feat]`、`[fix]`，建议保持一致。
- PR 描述建议包含：变更摘要、关键命令/复现步骤、影响范围；若涉及训练曲线或 UI，附截图或对比结果。
