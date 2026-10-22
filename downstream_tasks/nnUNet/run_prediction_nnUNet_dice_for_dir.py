import os 
import sys
import shutil
import subprocess
import argparse
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image
import cv2
import json 

from metrics.Dice_with_acc import MultiClassDiceEvaluator


def get_dice_score(pred_mask, gt_mask, target_classes):
    """
    计算单个预测掩码与真实掩码的 Dice 分数

    Args:
        pred_mask: 预测掩码 (numpy array)
        gt_mask: 真实掩码 (numpy array)
        target_classes: 目标类别列表

    Returns:
        平均 Dice 分数
    """
    # 确保尺寸一致
    if pred_mask.shape != gt_mask.shape:
        # 使用最近邻插值调整大小
        import cv2
        pred_mask = cv2.resize(
            pred_mask,
            (gt_mask.shape[1], gt_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    # 计算每个类别的 Dice
    dice_scores = []
    for class_id in target_classes:
        # 二值化
        pred_binary = (pred_mask == class_id).astype(np.uint8).flatten()
        gt_binary = (gt_mask == class_id).astype(np.uint8).flatten()

        intersection = np.sum(pred_binary * gt_binary)
        union = np.sum(pred_binary) + np.sum(gt_binary)

        if union > 0:
            dice = (2.0 * intersection) / union
        else:
            dice = 0.0

        dice_scores.append(dice)

    # 返回 Dice
    return dice_scores


def run_nnunet_prediction(image_paths, output_dir, dataset_id, fold, config,
                          nnUNet_preprocessed=None, nnUNet_results=None, nnUNet_raw=None):
    """
    对一组图像运行 nnUNet 预测

    Args:
        image_paths: 图像路径列表
        output_dir: 输出目录
        dataset_id: 数据集 ID
        fold: 折数
        config: 配置 (2d, 3d_lowres 等)
        nnUNet_preprocessed: nnUNet 预处理数据目录
        nnUNet_results: nnUNet 结果目录
        nnUNet_raw: nnUNet 原始数据目录

    Returns:
        dict: {filename: prediction_path}
    """
    # 设置环境变量
    env = os.environ.copy()
    if nnUNet_preprocessed is not None:
        env['nnUNet_preprocessed'] = str(nnUNet_preprocessed)
        print(f"设置环境变量 nnUNet_preprocessed={nnUNet_preprocessed}")
    if nnUNet_results is not None:
        env['nnUNet_results'] = str(nnUNet_results)
        print(f"设置环境变量 nnUNet_results={nnUNet_results}")
    if nnUNet_raw is not None:
        env['nnUNet_raw'] = str(nnUNet_raw)
        print(f"设置环境变量 nnUNet_raw={nnUNet_raw}")

    # 创建临时输入目录
    temp_input = tempfile.mkdtemp(prefix="nnunet_input_")
    temp_input_path = Path(temp_input)

    # 复制图像到临时目录（确保转换为灰度图）
    name_mapping = {}  # {新文件名: 原始文件名}
    for img_path in image_paths:
        # 使用统一格式的文件名（防止重复）
        # nnUNet期望的格式：文件名_0000.png (单通道)
        base_name = f"input_{len(name_mapping):04d}"
        new_name = f"{base_name}_0000.png"
        dst = temp_input_path / new_name

        # 读取图像并转换为灰度图
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")

        # 保存为灰度图
        cv2.imwrite(str(dst), img)

        # 存储映射关系（不带_0000后缀，便于后续查找）
        # name_mapping[base_name] = img_path.name   # the name are the same
        name_mapping[base_name] = img_path # 

    try:
        # 构建 nnUNet 命令
        cmd = [
            "nnUNetv2_predict",
            "-i", str(temp_input_path),
            "-o", str(output_dir),
            "-d", str(dataset_id),
            "-f", str(fold),
            "-c", config,
            "-tr", "nnUNetTrainer_200epoches",
            "-chk", "checkpoint_best.pth", 
            # "--disable_tta",  # 禁用测试时增强，提高速度
        ]

        print(f"运行 nnUNet 预测: {' '.join(cmd)}")

        # 运行命令（使用设置的环境变量）
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=env
        )

        # 查找输出文件并映射回原始文件名
        output_dir_path = Path(output_dir)
        pred_files = sorted(output_dir_path.glob("*.png")) + sorted(output_dir_path.glob("*.jpg"))

        # 构建映射: {原始文件名: 预测文件路径}
        pred_mapping = {}
        for pred_file in pred_files:
            # nnUNet 输出文件名格式: input_XXXX_0000.png 或 input_XXXX.png
            stem = pred_file.stem
            # if stem.endswith("_0000"):
            #     stem = stem[:-5]

            # 找到对应的原始文件名
            original_name = name_mapping.get(stem)
            if original_name:
                pred_mapping[original_name] = pred_file

        return pred_mapping

    finally:
        # 清理临时目录
        shutil.rmtree(temp_input_path, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='计算某个文件夹中图像的预测结果，以及dice score'
    )

    # 必需参数
    parser.add_argument(
        '--dir_name',
        type=str,
        required=True,
        help='directory name of input images'
    )
    parser.add_argument(
        '--input_root',
        type=str,
        required=True,
        help='输入的图像目录'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        required=True,
        help='输出目录（重新排序后的数据）'
    )
    parser.add_argument(
        '--gt_mask_dir',
        type=str,
        required=True,
        help='源数据目录（用于查找真实掩码）'
    )

    # nnUNet 参数
    parser.add_argument(
        '--dataset_id',
        type=int,
        default=325,
        help='nnUNet 数据集 ID'
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=0,
        help='nnUNet 折数'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='2d',
        help='nnUNet 配置 (2d, 3d_lowres 等)'
    )
    parser.add_argument(
        '--target_classes',
        type=str,
        default='1,2,3,4,5,6,7',
        help='目标类别，用逗号分隔（例如：1,2,3）'
    )

    # nnUNet 环境变量参数
    parser.add_argument(
        '--nnUNet_preprocessed',
        type=str,
        default=None,
        help='nnUNet 预处理数据目录路径'
    )
    parser.add_argument(
        '--nnUNet_results',
        type=str,
        default=None,
        help='nnUNet 结果目录路径'
    )
    parser.add_argument(
        '--nnUNet_raw',
        type=str,
        default=None,
        help='nnUNet 原始数据目录路径'
    )

    args = parser.parse_args()
    return args


def main():
    
    args = parse_args()
    
    # 参数
    dir_name = args.dir_name
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    gt_mask_dir = Path(args.gt_mask_dir)
    # nnUNet checkpoint settings
    dataset_id = args.dataset_id
    fold = args.fold
    config = args.config
    target_classes = list(map(int, args.target_classes.split(','))) 

    # nnUNet 环境变量参数
    nnUNet_preprocessed = args.nnUNet_preprocessed
    nnUNet_results = args.nnUNet_results
    nnUNet_raw = args.nnUNet_raw

    # 打印环境变量设置
    if nnUNet_preprocessed:
        print(f"nnUNet_preprocessed: {nnUNet_preprocessed}")
    if nnUNet_results:
        print(f"nnUNet_results: {nnUNet_results}")
    if nnUNet_raw:
        print(f"nnUNet_raw: {nnUNet_raw}")
    
    generated_data_dir = os.path.join(input_root, dir_name)
    
    output_dir = os.path.join(output_root, dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    ### collect images
    generated_data_list = []
    for idx, adata in enumerate(os.listdir(generated_data_dir)):
        if not adata.endswith("png"):
            continue
        cur_data_path = os.path.join(generated_data_dir, adata)
        generated_data_list.append(cur_data_path)
        # debug usage
        # if idx > 5: break 
    
    ### run prediction    
    tmp_dir_path = "./output_dir/tmp"
    os.makedirs(tmp_dir_path, exist_ok=True)
    print("==> temp results saved in: {}".format(tmp_dir_path))
    
    temp_output_dir = tempfile.mkdtemp(dir=tmp_dir_path, prefix="nnunet_resort_")
    # 运行 nnUNet 预测
    pred_output = Path(temp_output_dir) / "predictions"
    pred_output.mkdir(parents=True, exist_ok=True)

    pred_mapping = run_nnunet_prediction(
        generated_data_list,
        str(pred_output),
        dataset_id,
        fold,
        config,
        nnUNet_preprocessed,
        nnUNet_results,
        nnUNet_raw
    )
    
    ### copy predictions back to dst_path 
    pred_output_dir = os.path.join(output_dir, "prediction")
    if not os.path.exists(pred_output_dir):
        os.makedirs(pred_output_dir)
    
    for img_path, pred_path in pred_mapping.items():
        raw_filename = os.path.basename(img_path)
        src_pred_path = pred_path
        dst_pred_path = os.path.join(pred_output_dir, raw_filename)
        shutil.copy(src_pred_path, dst_pred_path)

    shutil.rmtree(temp_output_dir, ignore_errors=True)  # clear tmp dir
    print("pred results are saved in: {}".format(output_dir))
    
    ### evaluation (dice), and save jsonl
    metrics_jsonl = []
    dice_list = []
    cnt = 0
    for afile in os.listdir(pred_output_dir):
        cur_filename = afile
        cur_pred_path = os.path.join(pred_output_dir, cur_filename)
        cur_gt_path = os.path.join(gt_mask_dir, cur_filename).replace("_generated", "")
        
        evaluator = MultiClassDiceEvaluator("", "", target_classes)
        gt_mask = evaluator._read_mask(str(cur_gt_path))
        pred_mask = evaluator._read_mask(str(cur_pred_path))

        # 计算 Dice
        class_dice = get_dice_score(pred_mask, gt_mask, target_classes)
        
        cur_metric = {
            "filename": afile, 
            "pred": cur_pred_path,
            "gt": cur_gt_path,  
            "dice": [float(i) for i in class_dice], 
        }
        metrics_jsonl.append(cur_metric)
        dice_list.append(np.mean(class_dice))
        cnt += 1
        pass
    
    save_metric_jsonl_path = os.path.join(output_dir, "metric_class_dice.jsonl")
    with open(save_metric_jsonl_path, "w") as f:
        json.dump(metrics_jsonl, f, indent=4)
    print(f"save metrics: {cnt} items")
    print("==> mean dice: {}".format(np.mean(dice_list)))
    
    print("==> Finish! ")


if __name__ == "__main__":
    main()