import json
import re
import os


def clean_text(text):
    """
    清洗文本的核心函数
    """
    if not isinstance(text, str):
        return text

    # 1. 解决转义符问题：将 JSON 保存时转义的字符还原（如果是在字符串中）
    # 注意：如果文件已经是正确解析的 JSON，这一步主要处理字符串内的 \n
    text = text.replace('\\n', '\n')

    # 2. 合并连续的换行符：将两个或以上的换行符替换为一个换行符
    # 这能解决“每条都有 \n\n”的问题
    text = re.sub(r'\n{2,}', '\n', text)

    # 3. 去除每行开头和结尾的空白字符（空格、制表符）
    lines = [line.strip() for line in text.split('\n')]

    # 4. 去除首尾空行
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()

    # 5. 重新组合文本
    cleaned_text = '\n'.join(lines)

    # 6. 针对性处理类似 "S= \"vv Int {" 这种不规范的开头/结尾
    # 移除开头的非中文/非字母数字字符（除非是标点）
    # cleaned_text = re.sub(r'^[^a-zA-Z\u4e00-\u9fff0-9\(\)\[\]{}【】]+', '', cleaned_text)

    # 如果文本看起来像代码片段且不完整（例如以 { 开头，} 结尾不全），你可以选择保留或标记
    # 这里我们只做简单的空格清理
    return cleaned_text


def clean_dataset(input_file, output_file):
    """
    清洗整个数据集文件
    """
    if not os.path.exists(input_file):
        print(f"错误：找不到文件 {input_file}")
        return

    cleaned_data = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())

                # 清洗特定的字段
                if 'question' in item:
                    item['question'] = clean_text(item['question'])
                if 'response_chosen' in item:
                    item['response_chosen'] = clean_text(item['response_chosen'])
                if 'response_rejected' in item:
                    item['response_rejected'] = clean_text(item['response_rejected'])
                if 'system' in item:
                    item['system'] = clean_text(item['system'])

                # 如果 history 是列表，且包含字符串，也可以清洗，但通常 history 是空列表或结构化数据
                # 这里假设 history 是文本列表
                if 'history' in item and isinstance(item['history'], list):
                    item['history'] = [clean_text(msg) if isinstance(msg, str) else msg for msg in item['history']]

                cleaned_data.append(item)

            except json.JSONDecodeError as e:
                print(f"解析 JSON 错误: {e}")
                continue

    # 保存清洗后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in cleaned_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"清洗完成！共处理 {len(cleaned_data)} 条数据。")
    print(f"已保存至: {output_file}")


# --- 执行清洗 ---
input_path = "../general_DPO.jsonl"  # 你的输入文件名
output_path = "../general_DPO_cleaned.jsonl"  # 输出文件名

clean_dataset(input_path, output_path)