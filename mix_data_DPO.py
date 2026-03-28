import json
import random

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def normalize_format(data_list):
    """统一数据格式：将 question 转换为 prompt"""
    normalized = []
    for item in data_list:
        if 'question' in item:
            item['prompt'] = item.pop('question')
        if 'prompt' in item and 'chosen' in item and 'rejected' in item:
            normalized.append(item)
    return normalized

# --- 配置区域 ---
tsundere_file = '../RLHF_data/final_dpo_train.jsonl'  # 傲娇数据
general_file = '../RLHF_data/manual_general_20.jsonl'  # 刚才保存的通用数据
output_file = 'DPO_train_data.jsonl'

print("🚀 开始数据混合流程...")

# 1. 加载傲娇数据
tsundere_data = load_jsonl(tsundere_file)
print(f"✅ 傲娇数据加载完成: {len(tsundere_data)} 条")

# 2. 加载通用数据
general_data = load_jsonl(general_file)
print(f"✅ 通用数据加载完成: {len(general_data)} 条")

# 3. 扩充通用数据（可选）
# 为了让通用数据量级接近傲娇数据（假设傲娇有500条），我们可以把20条通用数据重复25次
# 这样就有 500 条通用数据，形成 1:1 的完美对抗
repeat_factor = max(1, len(tsundere_data) // len(general_data))
general_data_expanded = general_data * repeat_factor
print(f"🔄 通用数据扩充后: {len(general_data_expanded)} 条 (重复倍数: {repeat_factor})")

# 4. 格式统一化
tsundere_normalized = normalize_format(tsundere_data)
general_normalized = normalize_format(general_data_expanded)

# 5. 混合与打乱
final_dataset = tsundere_normalized + general_normalized
random.shuffle(final_dataset)

# 6. 保存
with open(output_file, 'w', encoding='utf-8') as f:
    for item in final_dataset:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"🎉 混合完成！")
print(f"📊 最终数据量: {len(final_dataset)} 条")
print(f"💾 已保存至: {output_file}")