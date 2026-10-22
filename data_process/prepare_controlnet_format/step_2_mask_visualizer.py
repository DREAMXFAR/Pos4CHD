import os
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
import json


# ================= Utils Functions =================
def get_files(folder):
    if not os.path.exists(folder):
        print(f"错误: 输入文件夹 '{folder}' 不存在")
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]


def generate_uniform_colors(class_ids):
    """
    根据类别数量，在 HSV 空间均匀生成颜色。
    Class 0 (背景) 固定为黑色。
    其他类别均匀分布在色相环上。
    """
    color_map = {}

    # 过滤掉 0，只计算前景类别的数量
    foreground_classes = [c for c in class_ids if c != 0]
    num_foreground = len(foreground_classes)

    # 处理 Class 0 (背景)
    if 0 in class_ids:
        color_map[0] = np.array([0, 0, 0], dtype=np.uint8)  # 黑色

    # 如果没有前景类别，直接返回（避免后续除以0）
    if num_foreground == 0:
        return color_map

    # 均匀分布生成颜色
    for i, cls_id in enumerate(foreground_classes):
        # 1. 计算色相 (Hue): OpenCV 的 Hue 范围是 0-179
        # 我们把 180 度平分给所有前景类别
        hue = int(180 * i / num_foreground)

        # 2. 构造 HSV 颜色: [Hue, Saturation=255, Value=255]
        # 创建一个 1x1 的像素来借用 cv2.cvtColor 进行转换
        hsv_pixel = np.uint8([[[hue, 255, 255]]])

        # 3. 转为 BGR
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]

        color_map[cls_id] = bgr_pixel

    ### gray color visualization
    # for i, cls_id in enumerate(foreground_classes):
    #     bgr_pixel = cls_id * int(255 / 7)
        
    #     color_map[cls_id] = bgr_pixel

    return color_map


def apply_color_map_fast(gray_img, color_dict):
    """
    使用 NumPy LUT (查找表) 进行快速映射
    修复点：
    1. 处理空color_dict的情况
    2. 防止max_id为负数导致lut空数组
    3. 限制gray_img的数值类型为整数，避免浮点ID
    4. 优化越界ID的处理逻辑
    """
    # 空颜色字典：直接转为3通道灰度图
    if not color_dict:
        return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    # 确保color_dict的key是整数，避免浮点型ID
    color_dict = {int(k): v for k, v in color_dict.items()}
    max_id = max(color_dict.keys()) if color_dict else 0

    # 核心修复1：强制max_id至少为0，避免lut数组为空（shape=(0,3)）
    max_id = max(max_id, 0)

    # 创建查找表 LUT (默认为黑色)，指定dtype避免溢出
    try:
        lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
    except OverflowError:
        # 极端情况：max_id过大，降级为256个类别
        print(f"警告：类别ID {max_id} 过大，LUT初始化溢出，降级为256个类别")
        max_id = 255
        lut = np.zeros((256, 3), dtype=np.uint8)

    # 填充 LUT
    for cls_id, color in color_dict.items():
        if 0 <= cls_id <= max_id:  # 防止ID超出LUT范围
            lut[cls_id] = color
        else:
            print(f"警告：类别ID {cls_id} 超出LUT范围（0-{max_id}），忽略该类别")

    # 核心修复2：处理掩码数据
    # 1. 转为整数型，避免浮点ID导致索引错误
    safe_img = gray_img.astype(np.int32)
    # 2. 防止图片中的值超出 LUT 范围，超出部分设为0（背景）
    safe_img = np.clip(safe_img, 0, max_id)

    # 应用映射
    colored_img = lut[safe_img]

    return colored_img


def save_dict_to_json(data, file_path):
    # 使用字典推导式，将所有的 ndarray 转换为普通的嵌套 list
    serializable_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in data.items()}
    
    with open(file_path, "w", encoding="utf-8") as f:
        # indent=4 可以让 JSON 文件格式化输出，方便人类阅读
        json.dump(serializable_dict, f, indent=4, ensure_ascii=False)
    print(f"数据已成功保存至: {file_path}")


def load_dict_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    # 将读取出来的嵌套 list 重新包装回 np.array
    # 此时它会自动恢复原本的 3D 形状（Shape）
    recovered_dict = {k: np.array(v) for k, v in json_data.items()}
    return recovered_dict



# ================= 配置区域 =================
ROOT = "./dataset/demo_dataset"

INPUT_FOLDER = os.path.join(ROOT, r"train/conditioning_images")
OUTPUT_FOLDER = os.path.join(ROOT, r"train/tmp_colorful_conditioning_images")

# COLOR_MAP=None
COLOR_MAP=load_dict_from_json(r".dataset/demo_dataset/color_map.json")

# 支持的扩展名
VALID_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']


def main():
    # 1. 准备输出目录
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    files = get_files(INPUT_FOLDER)
    if not files:
        return

    # ================= 阶段 1: 全局类别分析 & 出现频率统计 =================
    total_images = len(files)
    print(f"--- 正在分析 {total_images} 张图片中的类别与出现频率 ---")

    # 使用字典存储：{类别ID: 包含该类别的图片数量}
    class_image_counts = defaultdict(int)

    for f in tqdm(files, desc="Step 1 Analysis"):
        # 读取图片
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"警告：无法读取文件 {f}，跳过")
            continue

        # 降维处理（彩色掩码转单通道）
        if img.ndim == 3:
            img = img[:, :, 0]

        # 核心修复3：处理空掩码（避免np.unique报错）
        if img.size == 0:
            print(f"警告：文件 {f} 是空掩码，跳过")
            continue

        # 核心修改：这里不需要 return_counts=True，只需要知道这张图里“有没有”这个类
        unique_vals = np.unique(img)

        # 遍历这张图里出现的所有唯一类别，给全局计数器 +1
        for val in unique_vals:
            # 过滤非整数ID（比如浮点型），转为整数
            val = int(val)
            class_image_counts[val] += 1

    # 获取排序后的类别列表
    sorted_classes = sorted(list(class_image_counts.keys()))

    # 核心修复4：处理无有效类别的情况
    if not sorted_classes:
        print("错误: 所有掩码文件中未检测到有效类别ID")
        return

    print(f"\n[分析结果]")
    print(f"总图片数: {total_images}")
    print(f"{'Class ID':<10} | {'Images Count':<15} | {'Image Freq':<10}")
    print("-" * 45)

    # 计算概率(频率)并打印
    class_freqs = {}
    for cls_id in sorted_classes:
        count = class_image_counts[cls_id]
        freq = count / total_images  # 出现该类别的图片数 / 总图片数
        class_freqs[cls_id] = freq
        print(f"{cls_id:<10} | {count:<15} | {freq:.4%}")

    # ================= 阶段 2: 生成均匀色表 =================
    print(f"\n--- 生成 HSV 均匀色表 ---")
    if COLOR_MAP is None:
        color_map = generate_uniform_colors(sorted_classes)
    else:
        color_map = COLOR_MAP    

    # ================= 阶段 3: 映射并保存 =================
    print(f"--- 开始转换并保存 ---")
    for f in tqdm(files, desc="Step 2 Converting"):
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"警告：无法读取文件 {f}，跳过")
            continue

        if img.ndim == 3:
            img = img[:, :, 0]

        # 处理空掩码
        if img.size == 0:
            print(f"警告：文件 {f} 是空掩码，跳过")
            continue

        # 染色
        colored_img = apply_color_map_fast(img, color_map)

        # 保存
        basename = os.path.basename(f)
        save_path = os.path.join(OUTPUT_FOLDER, basename)
        cv2.imwrite(save_path, colored_img)

    print(f"\n全部完成，输出路径: {OUTPUT_FOLDER}")
    
    return color_map


if __name__ == "__main__":
    
    color_map = main()
    
    ### save color map for recording
    # save_path = r"./output_dir/color_map.json"
    # save_dict_to_json(color_map, save_path)
    # print("save color map in: {}".format(save_path))