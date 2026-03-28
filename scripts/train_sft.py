import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# --- 1. 路径配置 ---
model_path = "/home/littleflower/models/qwen/Qwen2.5-0.5B-Instruct"
data_path = "/home/littleflower/rlhf_project/data/SFT_train_data.jsonl"  # 纯傲娇数据
output_sft_path = "/home/littleflower/rlhf_project/models/qwen-0.5b-tsundere-sft-v2"

# --- 2. 设备检查 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前设备: {device}")

# --- 3. 加载数据集 ---
sft_dataset = load_dataset("json", data_files=data_path, split="train")
print(f"加载了 {len(sft_dataset)} 条训练数据。")

# --- 4. 加载 Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- 5. 模型加载 (针对 12GB 显存优化) ---

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4"
# )

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,  # 4070 支持 bf16，比 fp16 更稳
    device_map="auto",
    trust_remote_code=True,
    # quantization_config=quantization_config # 如果上面开启量化，这里也要取消注释
)


# --- 6. 注入 LoRA ---
# 针对 Qwen2.5，我们主要关注注意力模块和 MLP 模块
peft_config = LoraConfig(
    r=16,  # 秩设为 16 足够小模型使用
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # 这里的 target_modules 覆盖了 Qwen 的关键层
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# --- 7. 定义格式化函数  ---
# 将数据从 instruction/output 格式转换成 ChatML 格式
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []

    # Qwen2.5 的 ChatML 格式
    for instruction, input_text, output_text in zip(instructions, inputs, outputs):
        # 处理 input 为空的情况
        user_content = instruction
        if input_text and input_text.strip():
            user_content += "\n" + input_text

        text = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
        texts.append(text)

    return {"text": texts}


# 应用格式化
print("正在格式化数据...")
sft_dataset = sft_dataset.map(formatting_prompts_func, batched=True)
print("数据样例预览:", sft_dataset[0]["text"][:100] + "...")

# --- 8. SFT 配置 (针对小数据集 & 12GB 显存) ---
sft_args = SFTConfig(
    output_dir=output_sft_path,
    per_device_train_batch_size=2,  # 4070 12G 建议从 2 开始，如果 OOM 改 1
    gradient_accumulation_steps=8,  # 累积步数增加，保证有效 Batch Size 足够大
    learning_rate=2e-4,  # LoRA 学习率可以稍大一点
    num_train_epochs=5,  # 【重要修改】数据少，多跑几轮，让模型背下来
    logging_steps=10,
    save_steps=100,  # 数据少，可以频繁保存
    save_total_limit=2,  # 只保留最后两个检查点
    bf16=True,  # 开启 bf16
    optim="adamw_torch",
    max_length=512,  # 限制长度省显存
    gradient_checkpointing=True,  # 【重要修改】开启梯度检查点，用计算换显存，防止 OOM
    dataset_text_field="text",  # 指定上面格式化后的字段
    packing=False,  # 数据量少时建议 False，避免拼接导致长度截断
)

sft_trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=sft_dataset,
    processing_class=tokenizer,
)

print("开始 SFT 微调...")
sft_trainer.train()

# 保存模型（准确说是 LoRA 权重，不是完整模型）
sft_trainer.save_model(output_sft_path + "-final")
print(f"SFT 训练完成！模型已保存至: {output_sft_path}-final")
