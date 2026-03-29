# Tsundere-RLHF-Finetuning

基于 **Qwen2.5-0.5B-Instruct** 开发的角色扮演大模型微调项目。受限于单卡 RTX 4070 (12G 显存) 的硬件环境，本项目完整复现了从数据清洗、SFT (监督微调) 到 DPO (直接偏好优化) 对齐的全流程，成功生成了语言风格高度个性化（傲娇）的轻量级对话模型。

## 🎯 项目目标

在标准的监督微调 (SFT) 中，通过1000条高质量的个性化问答对赋予模型个性化的语言风格，完成微调后引入 DPO ，通过构建混合偏好对数据集，直接优化模型生成的偏好，同时保证其正常的逻辑能力，使其输出更符合特定性格的预期，旨在解决长尾场景下的指令遵循稳定性问题。

结果表明，DPO 有效抑制了 SFT 模型在复杂语境下偶尔出现的“过度傲娇”（把“傲娇”演变成“没礼貌”或者“发疯”）或“人设漂移”（回归基座模型的语言风格）现象。

## 🛠️ 技术栈

- **基础模型**: Qwen2.5-0.5B-Instruct
- **微调方法**: LoRA
- **对齐算法**: DPO
- **框架**: PyTorch, TRL, Transformers, PEFT

## 📂 项目结构

```text
Tsundere-RLHF-Finetuning/
├── DPO_data/                    # DPO 训练数据目录
│   ├── DPO_data1.json           
│   ├── DPO_data2.json           
│   └── manual_general_20.json   # 通用手动标注数据
├── SFT_data/                    # SFT 训练数据目录
│   ├── SFT_train_data.jsonl     # SFT 主训练数据
│   ├── self_introduction.json   # 自我介绍数据
│   ├── tsundere_data.json       # 傲娇数据
│   ├── tsundere_data1.json    
│   └── tsundere_data2.json    
├── scripts/                     # 脚本目录
│   ├── combine_data_SFT.py      # SFT 数据合并脚本
│   ├── data_clean.py            # 数据清洗脚本
│   ├── json2jsonl.py            # JSON 转 JSONL 脚本
│   ├── mix_data_DPO.py          # DPO 数据混合脚本
│   ├── train_dpo.py             # DPO 训练脚本
│   └── train_sft.py             # SFT 训练脚本
├── test/                        # 测试目录
│   ├── DPO_test.py              # DPO 测试脚本
│   └── SFT_test.py              # SFT 测试脚本
├── .gitignore                 
├── LICENSE                    
├── README.md                  
└── requirements.txt           
```

## 🚀 快速开始

### 1. 环境配置
建议使用 Python 3.10+ 环境，推荐使用 `conda` 或 `venv` 创建独立虚拟环境。

```bash
# 创建并激活 conda 环境 (示例)
conda create -n tsundere python=3.10
conda activate tsundere

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据预处理
运行以下命令以准备训练数据：

```bash
# 1. 清洗原始数据
python scripts/data_clean.py

# 2. 将 JSON 数据转换为训练所需的 JSONL 格式
python scripts/json2jsonl.py

# 3. 合并 SFT 数据集
python scripts/combine_data_SFT.py

# 4. 混合 DPO 偏好对数据
python scripts/mix_data_DPO.py
```

### 3. 模型训练
训练脚本已封装好，直接运行即可：

```bash
# 训练 SFT 模型
python scripts/train_sft.py

# 基于 SFT 检查点，继续训练 DPO 模型
python scripts/train_dpo.py
```

### 4. 推理与测试
训练完成后，可以使用测试脚本查看生成效果：

```bash
# SFT 模型测试
python test/SFT_test.py

# DPO 模型测试
python test/DPO_test.py
```
