import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 你的三个关键路径
base_model_path = "/home/littleflower/models/qwen/Qwen2.5-0.5B-Instruct"
sft_lora_path = "/home/littleflower/rlhf_project/models/qwen-0.5b-tsundere-sft-v2-final"
dpo_lora_path = "/home/littleflower/rlhf_project/models/qwen-0.5b-tsundere-dpo-final"

print("1. 正在加载原始基座模型...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("2. 正在挂载并融合 SFT 傲娇补丁...")
# 加载 SFT LoRA
model = PeftModel.from_pretrained(model, sft_lora_path)
# 必须合并！把 SFT 知识焊死在基座里
model = model.merge_and_unload()

print("3. 正在挂载 DPO 偏好对齐补丁...")
# 在融合后的模型上，继续挂载 DPO LoRA（推理时直接挂载即可，不用合并）
model = PeftModel.from_pretrained(model, dpo_lora_path)

# 切记：推理时开启评估模式，关闭 Dropout 随机丢弃
model.eval()

print("模型加载完毕！\n")

# 提问测试
prompt = "12加15等于多少。"
print(f"提问: {prompt}")

# 构建消息格式
messages = [
    {"role": "user", "content": prompt}
]

# 应用聊天模板
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 生成回复
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1  # 防止复读
    )

response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

print(f"回复: {response}")
