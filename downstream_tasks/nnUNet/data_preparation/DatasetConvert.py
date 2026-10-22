import os
import json
import shutil
from pathlib import Path

import argparse
import numpy as np
from PIL import Image
import ipdb 


def prepare_png_dataset_fixed_with_channels(args):
    source_dir = args.source_dir
    output_dir = args.output_dir
    addition_image_dir = args.addition_image_dir
    dataset_id = args.dataset_id
    mode = args.mode
    extend = args.extend
    extend_dir = args.extend_dir
    dataset_name = args.dataset_name
    modality_name = args.modality_name

    # 创建目录
    dataset_folder = Path(output_dir) / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_tr = dataset_folder / ("imagesTr" if mode == "train" else "imagesTs")
    labels_tr = dataset_folder / ("labelsTr" if mode == "train" else "labelsTs")

    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    images = sorted((Path(source_dir) / mode / "images").glob("*.png"))
    if extend:
        images += sorted((Path(source_dir) / extend_dir / addition_image_dir).glob("*.png"))

    masks = get_matching_masks(images, Path(source_dir) / mode / "conditioning_images")
    
    # ipdb.set_trace()

    if extend:
        addition_masks_dir = Path(source_dir) / extend_dir / "conditioning_images"
        addition_images = [img for img in images if extend_dir in str(img)]
        addition_masks = get_matching_masks(addition_images, addition_masks_dir)

        # 将addition的mask合并到主mask列表的对应位置
        for img, mask in zip(addition_images, addition_masks):
            img_index = images.index(img)
            if img_index < len(masks):
                masks.insert(img_index, mask)
            else:
                masks.append(mask)

    # 验证images和masks的数量匹配
    if len(images) != len(masks):
        print(f"警告: images数量 ({len(images)}) 和 masks数量 ({len(masks)}) 不匹配")

        # 移除没有对应mask的image
        valid_pairs = []
        for img, mask in zip(images, masks):
            if mask is not None:
                valid_pairs.append((img, mask))

        if valid_pairs:
            images, masks = zip(*valid_pairs)
            images, masks = list(images), list(masks)
            print(f"调整后: {len(images)} 个有效图像-mask对")

    print(f"找到 {len(images)} 个样本")

    # 首先检查文件格式
    print("\n🔍 检查文件格式...")
    sample_info = []
    for i in range(min(5, len(images))):  # 检查前5个样本
        img_path = images[i]
        mask_path = masks[i] if i < len(masks) else None

        img = Image.open(img_path)
        img_array = np.array(img)

        info = {
            "image": img_path.name,
            "image_shape": img_array.shape,
            "image_mode": img.mode
        }

        if mask_path:
            mask = Image.open(mask_path)
            mask_array = np.array(mask)
            info.update({
                "mask_shape": mask_array.shape,
                "mask_mode": mask.mode,
                "mask_values": np.unique(mask_array).tolist()
            })

        sample_info.append(info)
        print(
            f"  样本{i}: 图像={img_array.shape}({img.mode}), 标签={info.get('mask_shape', 'N/A')}({info.get('mask_mode', 'N/A')})")

    # 分析标签
    print("\n分析标签值...")
    all_labels = set()
    for mask_path in masks:  # 分析所有
        mask = np.array(Image.open(mask_path))
        all_labels.update(np.unique(mask).tolist())

    label_values = sorted(list(all_labels))
    print(f"发现标签值: {label_values}")

    # 创建标签映射
    if 0 in label_values:
        label_map = {"background": 0}
        other_labels = [l for l in label_values if l != 0]
        for i, lbl in enumerate(other_labels, 1):
            label_name = f"class_{i}"
            if len(other_labels) == 1:
                label_name = "target"  # 二分类时使用更友好的名称
            label_map[label_name] = int(lbl)
    else:
        label_map = {"background": 0}
        for i, lbl in enumerate(label_values, 1):
            label_map[f"class_{i}"] = int(lbl)
        print("⚠️  注意: 标签中没有0值，已添加background:0")

    print(f"标签映射: {label_map}")

    # 复制并转换文件
    print("\n📁 复制并转换文件...")
    converted_images = 0
    converted_masks = 0

    for i, (img_path, mask_path) in enumerate(zip(images, masks)):
        case_id = f"{i:04d}"

        try:
            # ========== 处理图像 ==========
            img = Image.open(img_path)
            img_array = np.array(img)

            # 检查并转换图像通道
            if len(img_array.shape) == 3:  # RGB或RGBA
                if img_array.shape[2] == 3:  # RGB
                    # RGB转灰度
                    gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
                    img_array = gray.astype(np.uint8)
                    converted_images += 1
                elif img_array.shape[2] == 4:  # RGBA
                    # RGBA转灰度（忽略alpha）
                    gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
                    img_array = gray.astype(np.uint8)
                    converted_images += 1
                else:
                    # 其他多通道，取第一个通道
                    img_array = img_array[:, :, 0]
                    converted_images += 1

            # 保存转换后的图像
            Image.fromarray(img_array).save(images_tr / f"case_{case_id}_0000.png")

            # ========== 处理标签 ==========
            mask = Image.open(mask_path)
            mask_array = np.array(mask)

            # 检查并转换标签通道
            if len(mask_array.shape) == 3:  # 多通道标签
                # 取第一个通道（假设是灰度图保存为RGB）
                mask_array = mask_array[:, :, 0]
                converted_masks += 1

            # 确保标签值是整数
            mask_array = mask_array.astype(np.uint8)

            # 保存转换后的标签
            Image.fromarray(mask_array).save(labels_tr / f"case_{case_id}.png")

            if (i + 1) % 100 == 0:  # 每100个显示进度
                print(f"  已处理 {i + 1}/{len(images)} 个样本")

        except Exception as e:
            print(f"❌ 处理 {img_path.name} 失败: {e}")

    print(f"\n📊 转换统计:")
    print(f"  转换的图像: {converted_images}/{len(images)}")
    print(f"  转换的标签: {converted_masks}/{len(masks)}")

    # 创建dataset.json
    dataset_info = {
        "name": f"Dataset{dataset_id:03d}_{dataset_name}",
        "description": "PNG ultrasound dataset (converted to single channel)",
        "reference": "Prepared with channel conversion",
        "licence": "CC-BY-NC-SA 4.0",
        "release": "1.0",
        "tensorImageSize": "2D",
        "modality": {"0": modality_name},
        "labels": label_map,
        "numTraining": len(images),
        "numTest": 0,
        "file_ending": ".png",
        # 使用NaturalImage2DIO，专门处理2D自然图像
        "overwrite_image_reader_writer": "Python2DIO",
        "channel_names": {"0": modality_name}
    }

    # 训练才创建dataset.json
    if mode == "train":
        with open(dataset_folder / "dataset.json", 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=4)

    print(f"\n✅ 数据集准备完成: {dataset_folder}")

    # 验证转换结果
    print("\n🔍 验证转换结果:")
    for i in range(min(3, len(images))):
        try:
            img = np.array(Image.open(images_tr / f"case_{i:04d}_0000.png"))
            mask = np.array(Image.open(labels_tr / f"case_{i:04d}.png"))

            print(f"  样本{i}:")
            print(f"    图像形状: {img.shape}, 值范围: [{img.min()}, {img.max()}]")
            print(f"    标签形状: {mask.shape}, 唯一值: {np.unique(mask)}")

            # 检查标签值是否在允许范围内
            allowed_values = set(label_map.values())
            actual_values = set(np.unique(mask))

            if actual_values.issubset(allowed_values):
                print(f"    ✅ 标签值有效")
            else:
                print(f"    ⚠️  标签包含无效值: {actual_values - allowed_values}")

        except Exception as e:
            print(f"    ❌ 验证失败: {e}")


def get_matching_masks(images, mask_dir):
    """
    根据图像文件名在mask目录中寻找对应的mask文件
    """
    matched_masks = []
    mask_dir = Path(mask_dir)

    for image_path in images:
        # 获取图像文件名（不带路径）
        image_name = image_path.name

        # 构建对应的mask文件路径
        mask_path = mask_dir / image_name

        if mask_path.exists():
            matched_masks.append(mask_path)
        else:
            # 如果找不到完全匹配的文件，尝试其他可能的命名模式
            # 例如：image_001.png -> mask_001.png 或 image_001_mask.png
            found = False

            # 尝试1: 添加后缀
            name_without_ext = image_path.stem
            ext = image_path.suffix

            # 可能的mask命名模式
            possible_names = [
                f"{name_without_ext}_mask{ext}",  # image_001.png -> image_001_mask.png
                f"mask_{name_without_ext}{ext}",  # image_001.png -> mask_image_001.png
                f"{name_without_ext}_seg{ext}",  # image_001.png -> image_001_seg.png
                f"seg_{name_without_ext}{ext}",  # image_001.png -> seg_image_001.png
                f"{name_without_ext}_label{ext}",  # image_001.png -> image_001_label.png
                f"label_{name_without_ext}{ext}",  # image_001.png -> label_image_001.png
                f"{name_without_ext}_generated{ext}",  # image_001.png -> image_001_label.png
                f"{name_without_ext}{ext}".replace("_generated", ""),  # image_001.png -> image_001_label.png
            ]

            for possible_name in possible_names:
                possible_path = mask_dir / possible_name
                if possible_path.exists():
                    matched_masks.append(possible_path)
                    found = True
                    print(f"警告: 使用替代命名 {possible_name} 匹配 {image_name}")
                    break

            if not found:
                # 如果还是找不到，使用占位符或报错
                print(f"错误: 无法找到 {image_name} 对应的mask文件")
                matched_masks.append(None)  # 或者 raise Exception

    return matched_masks


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='处理数据集用于nnUNet训练')

    # 必需参数
    parser.add_argument('--source_dir', type=str, required=True,
                        help='源数据目录路径')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='nnUNet raw数据输出目录')

    # 可选参数
    parser.add_argument('--dataset_id', type=int, default=1,
                        help='数据集ID')
    parser.add_argument('--dataset_name', type=str, default='train_nnUNet_data',
                        help='数据集名称')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'],
                        help='处理模式: train, val, test')
    parser.add_argument('--modality_name', type=str, default='Ultrasound',
                        help='模态名称')

    # addition相关参数
    parser.add_argument('--addition', action='store_true',
                        help='是否包含addition数据')
    parser.add_argument('--addition_image_dir', type=str, default='full_filtered',
                        help='addition图像目录 (默认: full_filtered)')

    # extend相关参数
    parser.add_argument('--extend', action='store_true',
                        help='是否扩展数据集')
    parser.add_argument('--extend_dir', type=str, default='extend',
                        help='扩展数据目录 (默认: extend)')

    # 其他参数
    parser.add_argument('--image_ext', type=str, default='.png',
                        help='图像文件扩展名 (默认: .png)')
    parser.add_argument('--mask_ext', type=str, default='.png',
                        help='掩码文件扩展名 (默认: .png)')
    parser.add_argument('--image_dir', type=str, default='images',
                        help='图像子目录名 (默认: images)')
    parser.add_argument('--mask_dir', type=str, default='conditioning_images',
                        help='掩码子目录名 (默认: conditioning_images)')

    # 调试参数
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细信息')
    parser.add_argument('--dry_run', action='store_true',
                        help='只显示将要执行的操作，不实际执行')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    prepare_png_dataset_fixed_with_channels(args)