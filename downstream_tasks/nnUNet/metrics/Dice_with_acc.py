import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from loguru import logger


class MultiClassDiceEvaluator:
    def __init__(self, gt_folder, pred_folder, target_classes):
        """
        :param gt_folder: 真实掩码文件夹
        :param pred_folder: 预测掩码文件夹
        :param target_classes: 需要计算的类别列表，例如 [1, 2, 3]。通常跳过背景 0。
        """
        self.gt_folder = gt_folder
        self.pred_folder = pred_folder
        self.target_classes = target_classes  # 比如 [1, 2]

        # 支持的图片扩展名
        self.exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']

    def _load_files(self, folder):
        files = []
        for ext in self.exts:
            files.extend(glob.glob(os.path.join(folder, ext)))
        return sorted(files)

    def _read_mask(self, path):
        """
        读取多类别掩码，保持原始整数值 (0, 1, 2...)
        千万不要在这里做 >127 的二值化，否则类别信息就丢了
        """
        # 读取为灰度图 (0-255)
        # 注意：nnUNet 输出通常是 0,1,2，肉眼看是全黑的，但数据是对的
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"无法读取图片: {path}")

        return img

    def calculate_dice_for_class(self, y_true, y_pred, class_id):
        """
        计算特定类别的 Dice
        """
        # 【关键】这里进行二值化：只把等于当前 class_id 的像素设为 1，其他设为 0
        y_true_binary = (y_true == class_id).astype(np.uint8).flatten()
        y_pred_binary = (y_pred == class_id).astype(np.uint8).flatten()

        intersection = np.sum(y_true_binary * y_pred_binary)
        union = np.sum(y_true_binary) + np.sum(y_pred_binary)

        # 纯数学公式，不处理空掩码返回1的情况
        dice = (2.0 * intersection) / (union + 1e-8)
        return dice
    
    
    def run(self):
        gt_files = self._load_files(self.gt_folder)
        gt_map = {os.path.splitext(os.path.basename(f))[0]: f for f in gt_files}
        pred_files = self._load_files(self.pred_folder)

        print(f"找到 GT 图片: {len(gt_files)} 张")
        print(f"找到 Pred 图片: {len(pred_files)} 张")
        print(f"目标计算类别: {self.target_classes}")

        # 存储每个类别的所有 Dice 分数
        # 结构: {1: [0.9, 0.8...], 2: [0.7, 0.6...]}
        class_dice_scores = {c: [] for c in self.target_classes}

        # 记录每张图片的平均 Dice，用于画分布图
        mean_dice_per_image = []
        filename_log = []

        for pred_path in tqdm(pred_files, desc="Processing"):
            fname = os.path.splitext(os.path.basename(pred_path))[0]
            if fname not in gt_map: continue

            gt_path = gt_map[fname]

            try:
                # 1. 读取原始掩码 (包含 0,1,2...)
                mask_gt = self._read_mask(gt_path)
                mask_pred = self._read_mask(pred_path)

                # 2. 尺寸对齐
                if mask_gt.shape != mask_pred.shape:
                    mask_pred = cv2.resize(mask_pred, (mask_gt.shape[1], mask_gt.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)

                # 3. 逐个类别计算 Dice
                image_scores = []
                for c in self.target_classes:
                    dice = self.calculate_dice_for_class(mask_gt, mask_pred, class_id=c)

                    class_dice_scores[c].append(dice)
                    image_scores.append(dice)

                # 计算这张图所有类别的平均分
                avg_score_this_img = np.mean(image_scores) if image_scores else 0.0
                mean_dice_per_image.append(avg_score_this_img)
                filename_log.append((fname, avg_score_this_img))

            except Exception as e:
                print(f"Err {fname}: {e}")

        if not mean_dice_per_image:
            print("未计算任何数据。")
            return

        # ================= 结果统计 =================
        print("\n" + "=" * 40)
        print("📊 多类别评估结果 (Multi-Class Evaluation)")
        print("=" * 40)

        total_means = []

        # 1. 输出每个类别的详细表现
        for c in self.target_classes:
            scores = class_dice_scores[c]
            mean_c = np.mean(scores)
            std_c = np.std(scores)
            total_means.append(mean_c)
            print(f"Class {c}: Mean Dice = {mean_c:.4f} ± {std_c:.4f}")

        # 2. 输出总体平均 (mDice)
        final_mDice = np.mean(total_means)
        print("-" * 40)
        print(f"✅ Overall mDice (所有类别平均): {final_mDice:.4f}")
        print("-" * 40)

        # 3. 找出平均表现最差的样本
        mean_dice_per_image = np.array(mean_dice_per_image)
        sorted_indices = np.argsort(mean_dice_per_image)
        print("\n⚠️ 表现最差的 5 个样本 (Worst Cases, Avg of all classes):")
        for idx in sorted_indices[:5]:
            print(f"   File: {filename_log[idx][0]} | Dice: {mean_dice_per_image[idx]:.4f}")

        # 画直方图 (使用每张图的平均Dice)
        # self.plot_histogram(mean_dice_per_image, final_mDice)


    def run_acc(self):
        gt_files = self._load_files(self.gt_folder)
        gt_map = {os.path.splitext(os.path.basename(f))[0]: f for f in gt_files}
        pred_files = self._load_files(self.pred_folder)

        print(f"找到 GT 图片: {len(gt_files)} 张")
        print(f"找到 Pred 图片: {len(pred_files)} 张")
        print(f"目标计算类别: {self.target_classes}")

        # 存储每个类别的所有 是否出现
        gt_acc_recorder = {c: [] for c in self.target_classes}
        pred_acc_recorder = {c: [] for c in self.target_classes}

        for pred_path in tqdm(pred_files, desc="Processing"):
            fname = os.path.splitext(os.path.basename(pred_path))[0]
            if fname not in gt_map: continue

            gt_path = gt_map[fname]

            # 1. 读取原始掩码 (包含 0,1,2...)
            mask_gt = self._read_mask(gt_path)
            mask_pred = self._read_mask(pred_path)
            
            mask_gt_classes = np.unique(mask_gt)
            mask_pred_classes = np.unique(mask_pred)

            # 3. 逐个类别记录acc
            for c in self.target_classes:
                if c in mask_gt_classes:
                    gt_acc_recorder[c].append(1)
                else:
                    gt_acc_recorder[c].append(0)
                
                if c in mask_pred_classes:
                    pred_acc_recorder[c].append(1)
                else:
                    pred_acc_recorder[c].append(0)
                    
            pass

        # ================= 结果统计 =================
        print("\n" + "=" * 40)
        print("📊 多类别评估结果 (Multi-Class Evaluation)")
        print("=" * 40)

        total_means = []

        # 1. 输出每个类别的详细表现
        for c in self.target_classes:
            
            tp = tn = fp = fn = 0
            
            for gt, pred in zip(gt_acc_recorder[c], pred_acc_recorder[c]):
                if gt == 1 and pred == 1:
                    tp += 1
                elif gt == 0 and pred == 0:
                    tn += 1
                elif gt == 0 and pred == 1:
                    fp += 1
                elif gt == 1 and pred == 0:
                    fn += 1

            accuracy = (tp + tn) / len(gt_acc_recorder[c])
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
  
            print(f"Class {c}: Acc = {accuracy:.4f}, Recall = {recall:.4f}")

        # info
        print("-" * 40)


    def calculate_pixel_prf_for_class(self, y_true, y_pred, class_id):
        """
        计算特定类别的像素级 Precision / Recall / F1 / Acc。

        对当前类别 class_id 做 one-vs-rest 二值化：
        - positive: 像素值 == class_id
        - negative: 像素值 != class_id
        """
        y_true_binary = (y_true == class_id)
        y_pred_binary = (y_pred == class_id)

        tp = int(np.logical_and(y_true_binary, y_pred_binary).sum())
        tn = int(np.logical_and(~y_true_binary, ~y_pred_binary).sum())
        fp = int(np.logical_and(~y_true_binary, y_pred_binary).sum())
        fn = int(np.logical_and(y_true_binary, ~y_pred_binary).sum())

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        return {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
        }

    def run_precision_recall(self):
        """
        计算像素级 per-class Precision / Recall / F1。

        这里不是 case-level 类别是否出现，而是把所有匹配图像的像素累计起来，
        对每个类别 c 进行 one-vs-rest 统计：
        - TP: gt == c 且 pred == c 的像素数
        - FP: gt != c 但 pred == c 的像素数
        - FN: gt == c 但 pred != c 的像素数
        - TN: gt != c 且 pred != c 的像素数
        """
        gt_files = self._load_files(self.gt_folder)
        gt_map = {os.path.splitext(os.path.basename(f))[0]: f for f in gt_files}
        pred_files = self._load_files(self.pred_folder)

        print(f"找到 GT 图片: {len(gt_files)} 张")
        print(f"找到 Pred 图片: {len(pred_files)} 张")
        print(f"目标计算类别: {self.target_classes}")

        # 累计所有图像的像素级 TP/TN/FP/FN
        pixel_stats = {
            c: {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
            for c in self.target_classes
        }

        matched_cnt = 0

        for pred_path in tqdm(pred_files, desc="Processing pixel-level PRF"):
            fname = os.path.splitext(os.path.basename(pred_path))[0]
            if fname not in gt_map:
                continue

            matched_cnt += 1
            gt_path = gt_map[fname]

            try:
                mask_gt = self._read_mask(gt_path)
                mask_pred = self._read_mask(pred_path)

                # 尺寸对齐，语义标签图必须用最近邻插值
                if mask_gt.shape != mask_pred.shape:
                    mask_pred = cv2.resize(
                        mask_pred,
                        (mask_gt.shape[1], mask_gt.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

                for c in self.target_classes:
                    result = self.calculate_pixel_prf_for_class(
                        mask_gt,
                        mask_pred,
                        class_id=c,
                    )
                    pixel_stats[c]["tp"] += result["tp"]
                    pixel_stats[c]["tn"] += result["tn"]
                    pixel_stats[c]["fp"] += result["fp"]
                    pixel_stats[c]["fn"] += result["fn"]

            except Exception as e:
                print(f"Err {fname}: {e}")

        if matched_cnt == 0:
            print("未找到可以匹配的 GT/Pred 文件，请检查文件名是否一致。")
            return

        print("\n" + "=" * 104)
        print("📊 多类别像素级评估结果 (Pixel-level One-vs-Rest Evaluation)")
        print("=" * 104)
        print("Class |        TP        TN        FP        FN |   Acc    Prec   Recall     F1     IoU")
        print("-" * 104)

        acc_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []

        for c in self.target_classes:
            tp = pixel_stats[c]["tp"]
            tn = pixel_stats[c]["tn"]
            fp = pixel_stats[c]["fp"]
            fn = pixel_stats[c]["fn"]

            total = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

            acc_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            iou_list.append(iou)

            print(
                f"{c:5d} | "
                f"{tp:9d} {tn:9d} {fp:9d} {fn:9d} | "
                f"{accuracy:7.4f} {precision:7.4f} {recall:8.4f} {f1:7.4f} {iou:7.4f}"
            )

        print("-" * 96)
        print(
            f"Macro | "
            f"{'-':>9} {'-':>9} {'-':>9} {'-':>9} | "
            f"{np.mean(acc_list):7.4f} "
            f"{np.mean(precision_list):7.4f} "
            f"{np.mean(recall_list):8.4f} "
            f"{np.mean(f1_list):7.4f}"
            f"{np.mean(iou_list):7.4f}"
        )
        print("=" * 96)
        print(f"matched cases: {matched_cnt}")


    def plot_histogram(self, scores, mean_val):
        output_dir = './results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.figure(figsize=(8, 5))
        plt.hist(scores, bins=20, range=(0, 1), color='lightgreen', edgecolor='black', alpha=0.7)
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
        plt.title('Distribution of Image-level mDice')
        plt.xlabel('Dice Coefficient')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'multiclass_dice_dist.png'))
        plt.close()


def arg_parser():
    parser = argparse.ArgumentParser(description="Compute Dice")

    parser.add_argument("--gt_dir", type=str, default=None, help="label Ground Truth root")
    parser.add_argument("--pred_dir", type=str, default=None, help="pred label root")
    parser.add_argument("--num_class", type=int, default=9, help="pred label root")
    
    args = parser.parse_args()    
    return args


# ================= 运行配置 =================
if __name__ == "__main__":
    
    # args
    args = arg_parser()
    
    gt_dir = args.gt_dir
    pred_dir = args.pred_dir
    ### debug usage
    # gt_dir = r"./nnUNet_raw/Dataset401_4CH_target14_processed_other_val100_v2_controlnet_format/labelsTs"
    # pred_dir = r"./output_dir/nnUNet_inference_results/Dataset401_4CH_target14_processed_other_val100_v2_controlnet_format_fold0_test_best"
    # args.num_class = 14
    
    logger.info(f"[config] gt_dir: {gt_dir}")
    logger.info(f"[config] pred_dir: {pred_dir}")
    
    # TARGET_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9]   # FormatFuncData
    TARGET_CLASSES = [i+1 for i in range(args.num_class)]   # VSD_png_v1
    logger.info(f"[config] target classes: {TARGET_CLASSES}")
    
    # process
    evaluator = MultiClassDiceEvaluator(gt_dir, pred_dir, target_classes=TARGET_CLASSES)
    
    # compute Acc
    evaluator.run_acc()
    # compute P, R, F1, pixel-level
    evaluator.run_precision_recall()
    # compute Dice
    evaluator.run()
