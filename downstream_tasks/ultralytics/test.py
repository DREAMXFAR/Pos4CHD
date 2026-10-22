from ultralytics import YOLO

# config
save_root = "./output_dir"

project_name = "demo_vsd_dataset"
exp_name = "debug_test"


# pre-trained model path
model_path = r"./pretrained_models/VSDWeightedSamplerMAP50Trainer_weight4_pretrain_Final_mixing_VSD_genpos_freeaug_cls12_trainset_4CH_target14_SpatextCtrlNet_seed42_cfg2p0_humanfilter_vsdonly/exp_0_seed_2930/weights/best.pt"

# dataset config
data_yaml = r"./configs/datasets/{}.yaml".format(project_name)

# load model
model = YOLO(model_path)

# test
metrics = model.val(
    data=data_yaml,
    split="val", 
    task="detect",
    imgsz=640,
    batch=16,
    device=0,
    project=project_name,
    name=exp_name,
    plots=False,
    save_json=False,
)
