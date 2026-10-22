import os 
import shutil
import json 


def main():
    DST_DATA_ROOT = r"./dataset"
    DST_DATA_NAME = "demo_dataset"

    dst_data_path = os.path.join(DST_DATA_ROOT, DST_DATA_NAME)
    os.makedirs(dst_data_path, exist_ok=True) 
    
    ### 1. save generate cls_id - name mapping
    name2clsid_json = {
        "background": 0,
        "chest": 1,
        "left_lung": 2,
        "right_lung": 3,
        "heart": 4,
        "spine": 5,
        "left_atrium": 6,
        "right_atrium": 7,
        "left_ventricle": 8,
        "right_ventricle": 9,
        "tricuspid_valve": 10,
        "mitral_valve": 11,
        "foramen_ovale": 12,
        "interventricular_septum": 13,
        "dao": 14
    }
    
    class_map_json = {v: k for k, v in name2clsid_json.items()}
    
    save_json_path = os.path.join(dst_data_path, "class_map.json")
    with open(save_json_path, "w") as f:
        json.dump(class_map_json, f, indent=4)
    print("save class json map in: {}".format(save_json_path))
    

if __name__ == "__main__":
    main()
