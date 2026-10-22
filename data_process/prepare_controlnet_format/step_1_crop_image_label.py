import os
import cv2
import numpy as np
from tqdm import tqdm

# ================= Config =================
ROOT = r"/path/to/raw_dataset"

INPUT_FOLDER = os.path.join(ROOT, r"train/raw_images")  
OUTPUT_FOLDER = os.path.join(ROOT, r"train/images")

INPUT_FOLDER = os.path.join(ROOT, r"train/raw_conditioning_images") 
OUTPUT_FOLDER = os.path.join(ROOT, r"train/conditioning_images")


# Crop Configuration (0.0 ~ 1.0)
CROP_TOP = 0.15      # Top
CROP_BOTTOM = 0.0    # Bottom
CROP_LEFT = 0.18     # Left
CROP_RIGHT = 0.18    # Right

# Suffix
VALID_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']

def get_files(folder):
    if not os.path.exists(folder):
        print(f"Wrong! '{folder}' not exist!")
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]


def center_crop_to_512(img):
    target = 512
    h, w = img.shape[:2]

    # —— 1. black padding to 512×512 ——
    pad_top = max((target - h) // 2, 0)
    pad_bottom = max(target - h - pad_top, 0)
    pad_left = max((target - w) // 2, 0)
    pad_right = max(target - w - pad_left, 0)

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        img = cv2.copyMakeBorder(
            img,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)  # black
        )

    # —— 2. center crop to 512×512 ——
    h, w = img.shape[:2]
    center_y, center_x = h // 2, w // 2

    y1 = center_y - target // 2
    y2 = y1 + target
    x1 = center_x - target // 2
    x2 = x1 + target

    crop = img[y1:y2, x1:x2]
    return crop


def main():
    # 1. output dir
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"create output dir: {OUTPUT_FOLDER}")

    files = get_files(INPUT_FOLDER)
    if not files:
        print("Image file not found!")
        return

    print(f"--- Processing {len(files)} Images ---")
    print(f"Crop config: Top={CROP_TOP:.0%}, Left={CROP_LEFT:.0%}, Right={CROP_RIGHT:.0%}")

    for i, f in enumerate(tqdm(files, desc="Cropping")):
        # read image
        img = cv2.imread(f)
        if img is None:
            print(f"Not readable: {f}")
            continue

        # Raw size (Height, Width)
        h, w = img.shape[:2]

        # compute size (Pixel indices)
        # y轴: 从上往下增加
        y_start = int(h * CROP_TOP)              # 起点：去掉上面 CROP_TOP
        y_end = int(h * (1 - CROP_BOTTOM))       # 终点：如果底部不裁，就是 h

        # x轴: 从左往右增加
        x_start = int(w * CROP_LEFT)             # 起点：去掉左边 CROP_LEFT
        x_end = int(w * (1 - CROP_RIGHT))        # 终点：去掉右边 CROP_RIGHT

        # 核心操作：Numpy 数组切片 [y:y, x:x]
        # 注意：这里仅仅是截取，没有任何 resize 操作
        cropped_img = img[y_start:y_end, x_start:x_end]
        # cropped_img = center_crop_to_512(img)
        
        print("cropped_image size: {}".format(cropped_img.shape))

        # 保存图片
        basename = os.path.basename(f)
        save_path = os.path.join(OUTPUT_FOLDER, basename)
        cv2.imwrite(save_path, cropped_img)

        # 打印第一张图的尺寸变化，供你核对
        if i == 0:
            new_h, new_w = cropped_img.shape[:2]
            print(f"\n[示例检查] 文件: {basename}")
            print(f"  原始尺寸: {w} x {h}")
            # print(f"  裁剪范围: Y[{y_start}:{y_end}], X[{x_start}:{x_end}]")
            print(f"  新尺寸:   {new_w} x {new_h}")
            print(f"  (宽度减少了 {(w-new_w)/w:.1%}, 高度减少了 {(h-new_h)/h:.1%})\n")

    print(f"全部完成，结果保存在: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()