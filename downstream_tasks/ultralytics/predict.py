from ultralytics import YOLO
import os 
import cv2 
from tqdm import tqdm 
import numpy as np
import json 


class YoloInferencePipeline():
    def __init__(self, pretrained_model_path):
        self.model = YOLO(pretrained_model_path)
        
        # info
        print(f"Load model from: {pretrained_model_path}")
        
    def predict_image(self, image_path):
        results = self.model(image_path)  # predict on an image

        # Access the results
        for result in results:
            xywh = result.boxes.xywh  # center-x, center-y, width, height
            xywhn = result.boxes.xywhn  # normalized
            xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
            xyxyn = result.boxes.xyxyn  # normalized
            names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
            confs = result.boxes.conf  # confidence score of each box
            
            print(f"names:{names}")
            
    def predict_dir(self, dir_path, output_json_path):
        ### info 
        print(f"... Run predictions for dir: {dir_path}")
        
        ### get image list
        image_list = os.listdir(dir_path)
        num_images_raw = len(image_list)
        image_list = [i for i in image_list if (i.endswith("png") or i.endswith("jpg") or i.endswith("jpeg"))]
        num_images = len(image_list)
        
        print("Files in the target dir: {} not ends with png/jpg/jepg.".format(num_images_raw - num_images))
        print("#Num of images for prediction: {}".format(num_images))
        
        ### loop for prediction
        results_json = []
        
        cnt_vsd = 0
        
        for afile in tqdm(image_list):
            cur_file_path = os.path.join(dir_path, afile)
            
            cur_results = self.model.predict(  # predict on an image
                source=cur_file_path,
                imgsz=640,
                conf=0.25,      # confidence threshold
                iou=0.5,        # NMS IoU threshold
                save=False,
                save_txt=False,
                save_conf=False,
            )
            
            for aresult in cur_results:
                xywh = aresult.boxes.xywh  # center-x, center-y, width, height
                xywhn = aresult.boxes.xywhn  # normalized
                xyxy = aresult.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
                xyxyn = aresult.boxes.xyxyn  # normalized
                names = [aresult.names[cls.item()] for cls in aresult.boxes.cls.int()]  # class name of each box
                confs = aresult.boxes.conf  # confidence score of each box

            if len(names) > 0:                
                cur_res_json = {
                    "filename": afile,
                    "pred_cls": tuple([int(i) for i in aresult.boxes.cls.int().cpu().numpy()]), 
                    "confs": tuple([float(i) for i in confs.cpu().numpy().astype(np.float32)]), 
                }
                results_json.append(cur_res_json)
                
                # record vsd
                if np.sum(cur_res_json["pred_cls"]) > 0:
                    cnt_vsd += 1
            
            # if len(results_json) > 20:
            #     break
        
        ### post process
        with open(output_json_path, "w") as f:
            json.dump(results_json, f, indent=4)
            
        print(f"The num of possible VSD: {cnt_vsd}")

  
if __name__ == "__main__":
    
    ### config
    model_path = r"./pretrained_models/VSD_mixing_genpos_freeaug_humanfilter_vsdonly/weights/best.pt"
    
    ### inference
    pipeline = YoloInferencePipeline(model_path)
    
    ### for directory inference
    dir_path = r"./dataset/demo_vsd_dataset/images/test"
    output_json_path = r"./output_dir/results.json"
    
    """
    two class prediciton, label 1 for ivs, label 2 for vsd
    Note that images should be preprocessed by cropping first, to crop surrounding marks
    """
    pipeline.predict_dir(
        dir_path=dir_path, 
        output_json_path=output_json_path, 
    )
    
    
    