from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Literal


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

MissingLabelPolicy = Literal["error", "empty", "skip"]


def find_labels_dir(dataset_root: Path) -> Path:
    """
    自动查找 labels 或 lables 目录。

    优先使用标准名称 labels；如果不存在，则尝试 lables。
    """
    labels_dir = dataset_root / "labels"
    lables_dir = dataset_root / "lables"

    if labels_dir.is_dir():
        return labels_dir

    if lables_dir.is_dir():
        print(
            f"[Warning] 检测到目录名为 'lables'，"
            f"建议修改为标准名称 'labels': {lables_dir}"
        )
        return lables_dir

    raise FileNotFoundError(
        f"未找到标签目录：\n"
        f"  {labels_dir}\n"
        f"  {lables_dir}"
    )


def get_image_files(image_dir: Path) -> list[Path]:
    """获取目录下支持的图像文件。"""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"图像目录不存在：{image_dir}")

    image_files = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(image_files)


def build_output_name(prefix: str, original_stem: str) -> str:
    """
    构造添加前缀后的文件名。

    示例：
        prefix='dataset1'
        original_stem='case001'
        输出='dataset1_case001'
    """
    prefix = prefix.strip()

    if not prefix:
        return original_stem

    prefix = prefix.rstrip("_- ")
    return f"{prefix}_{original_stem}"


def copy_one_dataset(
    dataset_root: str | Path,
    output_root: str | Path,
    prefix: str,
    splits: tuple[str, ...] = ("train", "test"),
    missing_label_policy: MissingLabelPolicy = "error",
    overwrite: bool = False,
) -> dict[str, dict[str, int]]:
    """
    将一个数据集合并到目标目录。

    Args:
        dataset_root:
            输入数据集根目录。

        output_root:
            合并后数据集根目录。

        prefix:
            添加到文件名前的前缀，例如 dataset1、dataset2。

        splits:
            需要处理的数据划分，默认为 train 和 test。

        missing_label_policy:
            图像缺少对应标签时的处理方式：

            - "error": 抛出异常；
            - "empty": 创建空标签文件，适合YOLO负样本；
            - "skip": 跳过该图像。

        overwrite:
            如果目标文件已存在，是否覆盖。

    Returns:
        每个 split 的统计信息。
    """
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"数据集根目录不存在：{dataset_root}")

    input_images_root = dataset_root / "images"
    input_labels_root = find_labels_dir(dataset_root)

    statistics: dict[str, dict[str, int]] = {}

    for split in splits:
        source_image_dir = input_images_root / split
        source_label_dir = input_labels_root / split

        target_image_dir = output_root / "images" / split
        target_label_dir = output_root / "labels" / split

        target_image_dir.mkdir(parents=True, exist_ok=True)
        target_label_dir.mkdir(parents=True, exist_ok=True)

        image_files = get_image_files(source_image_dir)

        copied_images = 0
        copied_labels = 0
        empty_labels = 0
        skipped_images = 0

        for image_path in image_files:
            source_label_path = source_label_dir / f"{image_path.stem}.txt"

            new_stem = build_output_name(
                prefix=prefix,
                original_stem=image_path.stem,
            )

            target_image_path = (
                target_image_dir / f"{new_stem}{image_path.suffix.lower()}"
            )
            target_label_path = target_label_dir / f"{new_stem}.txt"

            # 处理标签缺失情况
            if not source_label_path.is_file():
                if missing_label_policy == "error":
                    raise FileNotFoundError(
                        f"图像缺少对应标签：\n"
                        f"  image: {image_path}\n"
                        f"  label: {source_label_path}"
                    )

                if missing_label_policy == "skip":
                    print(
                        f"[Skip] 缺少标签，跳过图像：{image_path.name}"
                    )
                    skipped_images += 1
                    continue

                if missing_label_policy != "empty":
                    raise ValueError(
                        "missing_label_policy 必须是 "
                        "'error'、'empty' 或 'skip'"
                    )

            # 防止覆盖同名文件
            if not overwrite:
                if target_image_path.exists():
                    raise FileExistsError(
                        f"目标图像已经存在：{target_image_path}\n"
                        f"请更换前缀，或者设置 overwrite=True。"
                    )

                if target_label_path.exists():
                    raise FileExistsError(
                        f"目标标签已经存在：{target_label_path}\n"
                        f"请更换前缀，或者设置 overwrite=True。"
                    )

            # 复制图像
            shutil.copy2(image_path, target_image_path)
            copied_images += 1

            # 复制或创建标签
            if source_label_path.is_file():
                shutil.copy2(source_label_path, target_label_path)
                copied_labels += 1
            else:
                # YOLO中空txt文件可表示没有目标的负样本
                target_label_path.write_text("", encoding="utf-8")
                empty_labels += 1

        statistics[split] = {
            "images": copied_images,
            "labels": copied_labels,
            "empty_labels": empty_labels,
            "skipped": skipped_images,
        }

        print(
            f"[{dataset_root.name} | {split}] "
            f"images={copied_images}, "
            f"labels={copied_labels}, "
            f"empty_labels={empty_labels}, "
            f"skipped={skipped_images}"
        )

    return statistics


def merge_two_datasets(
    dataset1_root: str | Path,
    dataset2_root: str | Path,
    output_root: str | Path,
    prefix1: str = "dataset1",
    prefix2: str = "dataset2",
    missing_label_policy: MissingLabelPolicy = "error",
    overwrite: bool = False,
) -> None:
    """
    合并两个YOLO格式数据集。
    """
    dataset1_root = Path(dataset1_root).resolve()
    dataset2_root = Path(dataset2_root).resolve()
    output_root = Path(output_root).resolve()

    if dataset1_root == dataset2_root:
        raise ValueError("dataset1_root 和 dataset2_root 不能相同。")

    if output_root == dataset1_root or output_root == dataset2_root:
        raise ValueError("输出目录不能与任意输入数据集目录相同。")

    if prefix1 == prefix2:
        print(
            "[Warning] 两个数据集使用了相同前缀，"
            "如果原文件名相同，可能发生命名冲突。"
        )

    print("=" * 70)
    print("合并数据集1")
    print(f"输入路径：{dataset1_root}")
    print(f"文件前缀：{prefix1}")
    print("=" * 70)

    stats1 = copy_one_dataset(
        dataset_root=dataset1_root,
        output_root=output_root,
        prefix=prefix1,
        missing_label_policy=missing_label_policy,
        overwrite=overwrite,
    )

    print()
    print("=" * 70)
    print("合并数据集2")
    print(f"输入路径：{dataset2_root}")
    print(f"文件前缀：{prefix2}")
    print("=" * 70)

    stats2 = copy_one_dataset(
        dataset_root=dataset2_root,
        output_root=output_root,
        prefix=prefix2,
        missing_label_policy=missing_label_policy,
        overwrite=overwrite,
    )

    print()
    print("=" * 70)
    print("合并完成")
    print(f"输出路径：{output_root}")

    for split in ("train", "test"):
        total_images = (
            stats1.get(split, {}).get("images", 0)
            + stats2.get(split, {}).get("images", 0)
        )
        total_labels = (
            stats1.get(split, {}).get("labels", 0)
            + stats2.get(split, {}).get("labels", 0)
        )
        total_empty = (
            stats1.get(split, {}).get("empty_labels", 0)
            + stats2.get(split, {}).get("empty_labels", 0)
        )

        print(
            f"{split}: "
            f"images={total_images}, "
            f"labels={total_labels}, "
            f"empty_labels={total_empty}"
        )

    print("=" * 70)


def parse_args() -> argparse.Namespace:

    ### config, merge two yolo dataset
    dataset_1_path = r"./dataset/real_dataset"
    dataset_2_path = r"./dataset/gen_dataset"
    
    # prefix  
    dataset_1_prefix = "a_"
    dataset_2_prefix = "b_"
    
    output_path = r"./dataset/Final_mixing_VSD_ctrlnet_genpos_freeaug_cls8_trainset_4CH_target14_SpatextCtrlNet_seed42_cfg2p0_vsdonly_humanfilter"
    
    
    parser = argparse.ArgumentParser(
        description="合并两个YOLO格式的数据集，并添加来源前缀。"
    )

    parser.add_argument(
        "--dataset1",
        type=str,
        # required=True,
        default=dataset_1_path, 
        help="第一个数据集根目录",
    )
    parser.add_argument(
        "--dataset2",
        type=str,
        # required=True,
        default=dataset_2_path, 
        help="第二个数据集根目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        # required=True,
        default=output_path, 
        help="合并后的输出目录",
    )
    parser.add_argument(
        "--prefix1",
        type=str,
        default=dataset_1_prefix,
        help="第一个数据集的文件名前缀",
    )
    parser.add_argument(
        "--prefix2",
        type=str,
        default=dataset_2_prefix,
        help="第二个数据集的文件名前缀",
    )
    parser.add_argument(
        "--missing-label-policy",
        type=str,
        choices=["error", "empty", "skip"],
        default="error",
        help=(
            "缺失标签处理方式："
            "error=报错，empty=创建空标签，skip=跳过图像"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖目标目录中的同名文件",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    merge_two_datasets(
        dataset1_root=args.dataset1,
        dataset2_root=args.dataset2,
        output_root=args.output,
        prefix1=args.prefix1,
        prefix2=args.prefix2,
        missing_label_policy=args.missing_label_policy,
        overwrite=args.overwrite,
    )