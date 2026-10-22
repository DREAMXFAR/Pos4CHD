import os 
import numpy as np 
import cv2
import json 
from PIL import Image
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm

from loguru import logger


#####################################################################################################
# Config
#####################################################################################################
IVS_LABEL = 13
VSD_LABEL = 99


#####################################################################################################
# Functions
#####################################################################################################
def seg2yolo_onlyvsd(bbox_type="hbb"): 
    """
    only consider vsd and non-vsd detection
    """
    data_name = r"VSD_ctrlnet_genpos_freeaug_cls8_trainset_4CH_target14_SpatextCtrlNet_seed42_cfg2p0_humanfilter"
     
    # path
    src_root = r"./datasets/demo_vsd_dataset"
    src_data_path = os.path.join(src_root, data_name)
     
    dst_root = r"./data"
    dst_data_path = os.path.join(dst_root, data_name + "_vsdonly")
    if not os.path.exists(dst_data_path):
        os.makedirs(dst_data_path)
    
    # prepare source info
    class_dict_path = os.path.join(src_data_path, "class_map.json")
    assert os.path.isfile(class_dict_path)
    class_dict = json.load(open(class_dict_path, "r"))
    
    # splits = ["train", "test"]
    splits = ["train"]
    
    # loop 
    for cur_split in splits:
        
        cur_src_image_dir = os.path.join(src_data_path, cur_split, "images")
        cur_src_label_dir = os.path.join(src_data_path, cur_split, "conditioning_images")
        
        cur_dst_image_dir = os.path.join(dst_data_path, "images", cur_split)
        cur_dst_label_dir = os.path.join(dst_data_path, "labels", cur_split)
        os.makedirs(cur_dst_image_dir, exist_ok=True)
        os.makedirs(cur_dst_label_dir, exist_ok=True)
        
        # if cur_split == "test":
        #     print("==> create empty test dir, but skip looping")
        #     break
        
        for afile in tqdm(os.listdir(cur_src_image_dir)):
            
            if afile.endswith("txt"):
                continue
            
            cur_filename_wosuffix = os.path.splitext(afile)[0]
            cur_src_image_path = os.path.join(cur_src_image_dir, afile)
            cur_src_label_path = os.path.join(cur_src_label_dir, afile)
        
            # process label: get bbox and label
            cur_src_label_array = cv2.imread(cur_src_label_path)
            cur_src_label_array = cv2.cvtColor(cur_src_label_array, cv2.COLOR_BGR2GRAY)
            height, width = cur_src_label_array.shape
            
            cur_label_list = np.unique(cur_src_label_array)
            cur_annos = []
            combine_flag = 0   
            
            for class_id in [IVS_LABEL, VSD_LABEL]:
                class_id = int(class_id)
                # skip background classI
                if class_id == 0:
                    continue  
                            
                if class_id == IVS_LABEL and VSD_LABEL in cur_label_list:
                    cur_class_seg_vs = (cur_src_label_array == IVS_LABEL)
                    cur_class_seg_vsd = (cur_src_label_array == VSD_LABEL)
                    cur_class_seg = np.logical_or(cur_class_seg_vs, cur_class_seg_vsd).astype(np.uint8)*255
                    combine_flag = 1
                    class_id = 1  # vsd
                elif class_id == VSD_LABEL and combine_flag == 1:
                    continue
                else:
                    cur_class_seg = (cur_src_label_array == class_id).astype(np.uint8)*255
                    class_id = 0  # non-vsd
                
                if np.sum(cur_class_seg) == 0:
                    continue
                
                ys, xs = np.where(cur_class_seg > 0)
                x_min = int(xs.min())
                y_min = int(ys.min())
                x_max = int(xs.max())
                y_max = int(ys.max())
                
                center_x = (x_min + x_max) * 1.0 / 2 / width
                center_y = (y_min + y_max) * 1.0 / 2 / height
                box_width = (x_max - x_min) * 1.0 / width
                box_height = (y_max - y_min) * 1.0 / height
                cur_line = " ".join([str(i) for i in [class_id, center_x, center_y, box_width, box_height]])
                cur_annos.append(cur_line)
                pass
                
            # process labels: write lines
            cur_label_path = os.path.join(cur_dst_label_dir, f"{cur_filename_wosuffix}.txt")
            with open(cur_label_path, "w") as f:
                for aline in cur_annos:
                    f.write(aline + "\n")
                    
            # process images: copy 
            cur_dst_image_path = os.path.join(cur_dst_image_dir, afile)
            shutil.copy(cur_src_image_path, cur_dst_image_path)
    
            # break 

    # info 
    logger.info(f"Processing {data_name}")
    logger.success("Finish :)")


def main():
    # convert segmentation label to yolo dataset format
    seg2yolo_onlyvsd()


if __name__ == "__main__":
    main()