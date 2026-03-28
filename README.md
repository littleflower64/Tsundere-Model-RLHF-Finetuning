# Tsundere-Model-RLHF-Finetuning
Fine-tuning Qwen2.5 with SFT and DPO for Character Roleplay.

# Tsundere-RLHF-Finetuning

基于 Qwen2.5 (或你使用的模型) 的角色扮演大模型微调项目。该项目实现了从数据清洗、SFT (监督微调) 到 DPO (直接偏好优化) 对齐的完整流程。

## 🎯 项目目标

在标准的监督微调 (SFT) 中，模型倾向于生成通用、安全的回复，导致角色扮演 (Roleplay) 场景中的人设崩塌（例如：傲娇角色会说出违和的礼貌话语）。

本项目引入 **DPO 算法**，通过构建偏好对数据集，直接优化模型生成的偏好，使其输出更符合特定性格的预期。

## 🛠️ 技术栈

- **基础模型**: Qwen2.5 (或其他你使用的模型)
- **微调方法**: LoRA
- **对齐算法**: DPO
- **框架**: PyTorch, Transformers, PEFT

## 📂 项目结构

```text
Tsundere-RLHF-Finetuning/
├── README.md
├── requirements.txt
├── data/                     # 数据目录
│   ├── sft_data.jsonl        # SFT 训练数据
│   └── dpo_data.jsonl        # DPO 训练数据 (偏好对)
├── scripts/
│   ├── clean_data.py         # 数据清洗脚本
│   ├── generate_data.py      # 数据合成脚本
│   ├── train_sft.py          # SFT 训练脚本
│   └── train_dpo.py          # DPO 训练脚本
└── .gitignore
