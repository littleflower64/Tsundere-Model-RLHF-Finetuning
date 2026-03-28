import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

# 配置你的本地路径
base_model_path = "/home/littleflower/models/qwen/Qwen2.5-0.5B-Instruct"
sft_lora_path = "/home/littleflower/rlhf_project/models/qwen-0.5b-tsundere-sft-v2-final"
dpo_data_path = "/home/littleflower/rlhf_project/data/DPO_train_data.jsonl"
output_dpo_path = "/home/littleflower/rlhf_project/models/qwen-0.5b-tsundere-dpo"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前设备: {device}")

print("正在加载基础模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("正在合并 SFT LoRA 权重...")
# 将 SFT 的 LoRA 权重加载到基座模型上
sft_model = PeftModel.from_pretrained(base_model, sft_lora_path)
# 合并权重并卸载 LoRA 适配器，使其变回一个完整的独立模型
model = sft_model.merge_and_unload()

print("合并完成！")

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载你的 dpo jsonl 文件
raw_dpo_dataset = load_dataset("json", data_files=dpo_data_path, split="train")

def format_dpo_data(example):
    # 注入傲娇 System Prompt，洗脑 Qwen 的默认设定
    prompt_message = [
        {"role": "system",
         "content": "你是一个傲娇少女，说话总是口是心非，虽然嘴上不饶人、脾气有点大，但内心其实很在乎对方。"},
        {"role": "user", "content": example["prompt"]}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        prompt_message,
        tokenize=False,
        add_generation_prompt=True
    )

    return {
        "prompt": formatted_prompt,
        "chosen": example["chosen"] + "<|im_end|>",
        "rejected": example["rejected"] + "<|im_end|>"
    }

# 应用格式化并打乱数据
dpo_dataset = raw_dpo_dataset.map(format_dpo_data).shuffle(seed=42)

print(f"DPO 数据集加载完毕，共 {len(dpo_dataset)} 条。")
print("【格式化后的 Prompt 样例】:\n", dpo_dataset[0]['prompt'])
print("【Chosen 样例】:\n", dpo_dataset[0]['chosen'])
print("【Rejected 样例】:\n", dpo_dataset[0]['rejected'])

# 为 DPO 阶段注入全新的 LoRA
dpo_peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, dpo_peft_config)
model.print_trainable_parameters()

# 配置 DPO 训练参数
dpo_args = DPOConfig(
    output_dir=output_dpo_path,
    per_device_train_batch_size=2,  # DPO 显存占用较大，设为 2
    gradient_accumulation_steps=8,  # 2 * 8 = 16 的全局 Batch Size
    learning_rate=5e-6,  # DPO 学习率要比 SFT 小，防止模型崩坏
    logging_steps=10,
    num_train_epochs=1,  # 偏好对齐通常跑 1-2 个 epoch 即可
    save_steps=200,
    bf16=True,  # 4070 使用 bf16
    optim="adamw_torch",
    beta=0.1,  # KL 惩罚系数，控制模型偏离原模型的程度，0.1是推荐经验值
    max_length=512,  # prompt + 回复 的最大总长度

)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,  # 设置为 None 即可，TRL会自动使用 Base Weights 作为参考
    args=dpo_args,
    train_dataset=dpo_dataset,
    processing_class=tokenizer,
)

print("开始 DPO (人类偏好对齐) 训练...")
dpo_trainer.train()

# 保存最终模型！
final_save_path = output_dpo_path + "-final"
dpo_trainer.save_model(final_save_path)
print(f"DPO 训练完成！模型已保存至: {final_save_path}")
