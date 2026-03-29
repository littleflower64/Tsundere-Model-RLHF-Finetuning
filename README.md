# Tsundere-RLHF-Finetuning

基于 Qwen2.5 开发的角色扮演大模型微调项目。由于设备首先，该项目使用 RTX 4070 显卡 12G 显存，以 Qwen2.5-0.5B-Instruct 模型为基座模型，完成从数据清洗、SFT (监督微调) 到 DPO (直接偏好优化) 对齐的完整流程，生成了语言风格个性化（傲娇）的模型。

## 🎯 项目目标

基于 **Qwen2.5-0.5B-Instruct** 开发的角色扮演大模型微调项目。受限于单卡 RTX 4070 (12G 显存) 的硬件环境，本项目完整复现了从数据清洗、SFT (监督微调) 到 DPO (直接偏好优化) 对齐的全流程，成功生成了语言风格高度个性化（傲娇）的轻量级对话模型。

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
