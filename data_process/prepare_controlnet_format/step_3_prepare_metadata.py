import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================

# 1. Configure Root
ROOT = r"./dataset/demo_dataset"

MASK_DIR_READ = os.path.join(ROOT, r"train/conditioning_images")   # single channle segmentation mask with class index
OUTPUT_FILE = os.path.join(ROOT, r"train/metadata.jsonl")

# 2. JSONL for directory prefix
PREFIX_IMAGE = 'images'
PREFIX_MASK = 'conditioning_images'  # 对应 masks_filtered
PREFIX_COLOR_MASK = 'colorful_conditioning_images'  # 对应 masks_color (可视化图)

# 4. ID-to-name mapping, 0 for background
class_mapping_json = os.path.join(ROOT, r"class_map.json")
with open(class_mapping_json, "r") as f:
    ID_TO_NAME = json.load(f)
    ID_TO_NAME = {int(k):v for k,v in ID_TO_NAME.items()}

# Prompt template
default_prompt = "A realistic ultrasound image, fetal cardiac, standard four-chamber view."
PROMPT_TEMPLATE = "A realistic ultrasound image, fetal cardiac, standard four-chamber view, segmented: {}"

# file name suffix
VALID_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']

# ===========================================

def get_files(folder):
    if not os.path.exists(folder):
        return []
    return [f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]


def analyze_mask_classes(mask_path):
    """
    读取掩码，返回其中包含的所有唯一类别 ID (剔除背景 0)
    """
    img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"Warning: 无法读取 {mask_path}")
        return []

    # 降维处理 (H, W, 3) -> (H, W)
    if img.ndim == 3:
        img = img[:, :, 0]

    # 获取唯一值
    unique_ids = np.unique(img)

    # 转换为列表，并排序
    # 过滤掉 0 (背景)，如果你想包含背景，去掉 'if x != 0'
    valid_ids = [int(x) for x in unique_ids if x != 0]
    return sorted(valid_ids)


def main():
    # 获取所有掩码文件名
    mask_files = get_files(MASK_DIR_READ)
    if not mask_files:
        print(f"错误: 在 {MASK_DIR_READ} 中没找到文件。")
        return

    print(f"正在构建 {OUTPUT_FILE}，共 {len(mask_files)} 条数据...")

    # 打开文件准备写入
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:

        for filename in tqdm(mask_files):
            # 1. 分析掩码内容
            mask_path = os.path.join(MASK_DIR_READ, filename)
            class_ids = analyze_mask_classes(mask_path)

            # 2. 构建类别文本列表
            class_names = []
            for uid in class_ids:
                if uid in ID_TO_NAME:
                    class_names.append(ID_TO_NAME[uid])
                else:
                    # 如果出现了映射表中没有的 ID，打印警告
                    print(f"\n[警告] 文件 {filename} 包含未定义的 ID: {uid}")
                    class_names.append(f"object_{uid}")  # 兜底策略

            # 3. 构建 Prompt
            # 如果该图全是背景（没有前景物体），prompt 怎么写？
            if class_names:
                classes_str = ", ".join(class_names)
                prompt = PROMPT_TEMPLATE.format(classes_str)
            else:
                # 假如全是背景，可以写个默认词，或者保持原样
                prompt = default_prompt

                # 4. 构建 JSON 条目
            # 注意：这里假设你的图片文件夹下的文件名和掩码文件名是完全一致的
            entry = {
                "file_name": f"{PREFIX_IMAGE}/{filename}",
                "default_prompt": default_prompt,
                "conditioning_image": f"{PREFIX_MASK}/{filename}",
                # "enhanced_prompt": prompt,
                "colorful_conditioning_image": f"{PREFIX_COLOR_MASK}/{filename}",
            }

            # 5. 写入一行 (JSONL 格式)
            f_out.write(json.dumps(entry) + '\n')

    print(f"\n成功生成: {OUTPUT_FILE}")
    print("请务必检查 metadata.jsonl 的前几行是否符合预期。")


if __name__ == "__main__":
    main()