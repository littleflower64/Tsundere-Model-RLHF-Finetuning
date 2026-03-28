import json
import os


def convert_json_to_jsonl_with_boost(input_file, output_file, boost_factor=2):
    """
    将 JSON 数组转换为 JSONL 格式，并支持对特定数据进行增强。

    参数:
    boost_factor: 针对 tsundere_data2 类型数据的重复倍数。
                  设为 1 表示不增强，设为 2 表示写入两次（推荐）。
    """

    if not os.path.exists(input_file):
        print(f"错误：找不到文件 {input_file}")
        return

    count = 0

    print(f"正在读取 {input_file} ...")
    with open(input_file, 'r', encoding='utf-8') as f_in:
        data_list = json.load(f_in)

    print(f"正在写入 {output_file} (开启人设增强模式 x{boost_factor})...")

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in data_list:
            # 简单的增强逻辑：
            # 如果 output 中包含 "笨蛋" 或者 "哼" 等典型傲娇词汇（或者你可以根据需要修改判断条件）
            # 这里我们简单粗暴一点：如果是 tsundere_data2 里的数据，通常比较短且情绪化。
            # 为了演示，我们假设所有数据都正常写入，但你可以通过修改这里来筛选特定数据。

            # 写入当前行
            # ensure_ascii=False 保证中文正常显示，不被转义成 \uXXXX
            json_str = json.dumps(item, ensure_ascii=False)
            f_out.write(json_str + '\n')
            count += 1

            # 【核心技巧】人设增强
            # 如果这条数据看起来像纯傲娇对话（这里用 output 长度小于 50 且包含 "哼" 或 "笨蛋" 作为简单特征）
            # 你可以根据实际情况调整这个判断逻辑，或者直接对所有数据增强
            output_text = item.get('output', '')
            if boost_factor > 1 and ("哼" in output_text or "笨蛋" in output_text or "才不" in output_text):
                for _ in range(boost_factor - 1):
                    f_out.write(json_str + '\n')
                    count += 1

    print("-" * 30)
    print(f"转换完成！")
    print(f"原始数据行数: {len(data_list)}")
    print(f"增强后总行数: {count}")
    print(f"文件已保存: {output_file}")


# 运行转换
# 输入是你刚才合并好的文件
input_filename = 'SFT_data/SFT_train_data.json'
output_filename = 'SFT_data/SFT_train_data.jsonl'

convert_json_to_jsonl_with_boost(input_filename, output_filename, boost_factor=2)
