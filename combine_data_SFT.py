import json
import random


def standardize_data(raw_data, source_name):
    """
    将数据统一为标准格式：{"instruction": "...", "input": "...", "output": "..."}
    """
    standardized = []

    for item in raw_data:
        # 1. 提取基础字段
        instruction = item.get('instruction', '')
        output = item.get('output', '')

        # 2. 处理 input 字段
        # 如果原本就有 input，就用原本的
        if 'input' in item:
            input_val = item['input']
        else:
            # 如果没有 input，设为空字符串（这是最安全的做法）
            # 这样训练框架就不会报错
            input_val = ""

        # 3. 构建标准对象
        # 确保所有数据都有这三个键
        standard_item = {
            "instruction": instruction,
            "input": input_val,
            "output": output
        }

        standardized.append(standard_item)

    print(f"处理 [{source_name}]: {len(raw_data)} 条 -> 标准化完成")
    return standardized


def main():
    # 1. 加载数据
    try:
        with open('../SFT_data/self_introduction.json', 'r', encoding='utf-8') as f:
            data1 = json.load(f)
        with open('../SFT_data/tsundere_data2.json', 'r', encoding='utf-8') as f:
            data2 = json.load(f)
        with open('../SFT_data/tsundere_data1.json', 'r', encoding='utf-8') as f:
            data3 = json.load(f)
    except FileNotFoundError:
        print("错误：找不到文件，请确保 json 文件和脚本在同一目录下")
        return

    # 2. 标准化处理 (关键步骤)
    # 无论原数据有没有 input，这里都会补齐
    std_data1 = standardize_data(data1, "self_introduction")
    std_data2 = standardize_data(data2, "tsundere_data2")
    std_data3 = standardize_data(data3, "tsundere_data1")

    # 3. 增强人设 (可选)
    # 让纯傲娇对话 (data2) 的权重翻倍，防止被知识问答稀释
    std_data2 = std_data2 * 2

    # 4. 合并与打乱
    combined_data = std_data1 + std_data2 + std_data3
    random.shuffle(combined_data)

    # 5. 保存
    output_file = 'final_merged_dataset.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

    print("-" * 30)
    print(f"处理完成！总数据量: {len(combined_data)} 条")
    print(f"已保存为: {output_file}")
    print("-" * 30)
    print("数据样例预览 (前2条):")
    for i in range(min(2, len(combined_data))):
        print(combined_data[i])


if __name__ == "__main__":
    main()