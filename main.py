import ray
import os
import utils
from PIL import Image

# Folder contains the gaze data
BASE_PATHS =[".../__allData/gaze_data/..._1",
              ".../__allData/gaze_data/..._2"]
# Acess to all folders in the BASE_PATHS, i.e. rad_name_date
FOLDER_NAMES = [os.path.join(base_path, entry) 
                 for base_path in BASE_PATHS 
                 for entry in os.listdir(base_path)
                 if os.path.isdir(os.path.join(base_path, entry))]

# Folder contains chest X-ray images
IMAGE_PATH = '.../__allData/converted_images'
# Folder to save the masked images
OUTPUT_DIR_FOR_MASKS = '.../__allData/miccai2025/masked_images_512size'
# Folder to save the anatomical patches
ANATOMICAL_PATH = '.../__allData/miccai2025/anatomical_patches_16_img224'

PATCH_SIZE = 16


def chunkify(lst, n):
    """Divide a list `lst` into `n` chunks."""
    return [lst[i::n] for i in range(n)]

@ray.remote
def _process(func, *args, **kwargs):
        func(*args, **kwargs)

if __name__ == "__main__":
    # Initialize Ray with the number of workers
    # Adjust the number of workers based on your system's capabilities
    num_workers = 16
    ray.init(num_cpus=num_workers)

    # Generate segmentation masks.
    # For each image in the IMAGE_PATH, create a mask
    # and save it to the OUTPUT_DIR_FOR_MASKS.
    # If you will skip this part, please check utils.get_mask_and_pixels function.
    png_files = [os.path.join(IMAGE_PATH, f) for f in os.listdir(IMAGE_PATH) if f.endswith('.png')]
    png_files_chunks = chunkify(png_files, int(num_workers))
    ray.get([
         _process.remote(
                utils.create_mask,
                png_files=chunk,
                output_dir=OUTPUT_DIR_FOR_MASKS
            )
            for chunk in png_files_chunks
        ])
    
    # Generate anatomical patches.
    # MAIN_IMAGE is a sample image from the IMAGE_PATH. 
    # It will serve as I_{ref}.
    MAIN_IMAGE = Image.open(os.path.join(IMAGE_PATH, '0a4fbc9ade84a7abd1680eb8ba031a9d.png'))
    mask, white_pixels, target = utils.get_mask_and_pixels('0a4fbc9ade84a7abd1680eb8ba031a9d.png')#, dist1=20, dist2=2) #dist1=20, dist2=2 for 8x8, 360 patches
    MAIN_IMAGE  = MAIN_IMAGE.resize((224,224))
    squares, squares_coord = utils.get_target_box_position(target, PATCH_SIZE)

    for folder_path in FOLDER_NAMES:
        folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
        folder_chunks = chunkify(folders, int(num_workers))
        ray.get([
                _process.remote(
                        utils.prepare_anatomical_patches,
                        image_path = IMAGE_PATH, 
                        folder_path = folder_path,
                        folder_names = chunk, 
                        target_squares_coord = squares_coord,
                        patch_size = PATCH_SIZE,
                        anatomical_path = ANATOMICAL_PATH
                    )
                    for chunk in folder_chunks
                ])
    

    



