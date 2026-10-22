import os
import sys
sys.path.append(r"/dat03/xuanwenjie/code/UltraSound_Image_Generation/ultralytics")

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


import os
import math
import numpy as np
import torch
import random

from torch.utils.data import WeightedRandomSampler

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.build import InfiniteDataLoader, build_dataloader, seed_worker
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import torch_distributed_zero_first


from ultralytics import settings

settings.update({
    "weights_dir": "./pretrained_models"
})


class VSDWeightedSamplerMAP50Trainer(DetectionTrainer):
    """
    use WeightedRandomSampler for class-balanced training
    """

    target_class_id = 1      # multi class VSD=1
    positive_weight = 4.0    # for VSD sampling weight
    negative_weight = 1.0    # for IVS sampling weight
    num_samples_scale = 1.0  # sampling ratio per epoch，1.0 for the size of raw dataset

    def build_image_weights(self, dataset):
        weights = []
        n_pos = 0
        n_neg = 0

        for lb in dataset.labels:
            cls = lb.get("cls", np.array([]))

            if len(cls) > 0:
                cls = cls.reshape(-1).astype(int)
                has_target = np.any(cls == self.target_class_id)
            else:
                has_target = False

            if has_target:
                weights.append(self.positive_weight)
                n_pos += 1
            else:
                weights.append(self.negative_weight)
                n_neg += 1

        weights = torch.as_tensor(weights, dtype=torch.double)

        LOGGER.info(
            f"Weighted sampler: target_class_id={self.target_class_id}, "
            f"positive_images={n_pos}, negative_or_other_images={n_neg}, "
            f"positive_weight={self.positive_weight}, negative_weight={self.negative_weight}"
        )

        return weights

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."

        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        # val with no weighted sampling
        if mode != "train":
            return build_dataloader(
                dataset,
                batch=batch_size,
                workers=self.args.workers * 2,
                shuffle=False,
                rank=rank,
                drop_last=False,
            )

        # only for single GPU training, not consider multi-gpu
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            raise RuntimeError("This simple WeightedRandomSampler version is for single-GPU training only.")

        weights = self.build_image_weights(dataset)

        num_samples = int(len(dataset) * self.num_samples_scale)

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples,
            replacement=True,
        )

        nd = torch.cuda.device_count()
        nw = min((os.cpu_count() or 1) // max(nd, 1), self.args.workers)

        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + max(rank, 0))

        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=False,          # turn off shuffle when using sampler
            sampler=sampler,
            num_workers=nw,
            pin_memory=nd > 0,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=self.args.compile and len(dataset) % batch_size != 0,
            prefetch_factor=4 if nw > 0 else None,
        )

    def validate(self):
        old_best_fitness = self.best_fitness

        metrics, default_fitness = super().validate()

        if metrics is None:
            return metrics, default_fitness

        self.best_fitness = old_best_fitness

        # overall mAP50
        map50 = metrics.get("metrics/mAP50(B)", None)

        if map50 is None:
            fitness = default_fitness
        else:
            fitness = float(map50)

        metrics["fitness"] = fitness

        if self.best_fitness is None or fitness > self.best_fitness:
            self.best_fitness = fitness

        return metrics, fitness

# config
save_root = "./output_dir"
project_name = "demo_vsd_dataset"

seed = random.randint(0, 10000)
# seed = 42
exp_name = f"exp_random_seed_{seed}"

# Load a COCO-pretrained YOLOv5n model
model_path = r"./pretrained_models/yolov5su.pt"
model = YOLO(model_path)

data_config = r"./configs/datasets/{}.yaml".format(project_name)

# Display model information (optional)
# model.info()

# Train the model on the dataset for 100 epochs
results = model.train(
    data=data_config, 
    task="detect", 
    epochs=250,
    batch=16,  
    imgsz=640, 
    amp=False, 
    project=os.path.join(save_root, "VSDWeightedSamplerMAP50Trainer_weight4_pretrain_" + project_name), 
    name=exp_name, 
    pretrained=True,
    optimizer="auto",
    workers=8,
    seed=seed,
    trainer=VSDWeightedSamplerMAP50Trainer, 
)

# Evaluate the model's performance on the validation set
metrics = model.val()
