import os 
import shutil
from tqdm import tqdm
from PIL import Image


if __name__ == "__main__":
    
    src_dir = r"/path/to/train/conditioning_images"
    dst_dir = r"/path/to/train/conditioning_images_resized"
    
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        
    for afile in tqdm(os.listdir(src_dir)):
        if afile.endswith("txt"):
            continue
        
        cur_src_path = os.path.join(src_dir, afile)

        cur_filename_wosuffix = os.path.splitext(afile)[0]
        
        ### rename purposes
        # cur_dst_path = os.path.join(dst_dir, f"{cur_filename_wosuffix}.png".replace("_generated", ""))
        # cur_dst_path = os.path.join(dst_dir, f"gen_{cur_filename_wosuffix}.png")
        # shutil.copy(cur_src_path, cur_dst_path) 
        
        ### resize purposes
        cur_dst_path = os.path.join(dst_dir, f"{cur_filename_wosuffix}.png")
        cur_src_image = Image.open(cur_src_path).convert("RGB")
        cur_dst_image = cur_src_image.resize((512, 512), Image.NEAREST)
        cur_dst_image.save(cur_dst_path)
        
        # break 
    
    print(":) Finish!")