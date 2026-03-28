import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型路径
# model_path = "/home/littleflower/models/qwen/Qwen2.5-0.5B-Instruct"
model_path = "/home/littleflower/rlhf_project/models/qwen-0.5b-tsundere-sft-v2-final"

print("正在加载模型...")
# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 直接写死的问题
prompt = "你是谁。"
print(f"提问: {prompt}")

# 构建消息格式
messages = [{"role": "user", "content": prompt}]


# 应用聊天模板
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 生成回复
outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

print(f"回复: {response}")
