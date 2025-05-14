import os
import ray
import scipy
import pandas as pd
import matplotlib.image as mpimg
from  features import gaze_statistics_case


# Folder contains the gaze data
BASE_PATHS = [".../__allData/gaze_data/..._1",
              ".../__allData/gaze_data/..._2"]
# Acess to all folders in the BASE_PATHS, i.e. rad_name_date
FOLDER_NAMES = [os.path.join(base_path, entry) 
                 for base_path in BASE_PATHS 
                 for entry in os.listdir(base_path)
                 if os.path.isdir(os.path.join(base_path, entry))]

# Folder contains chest X-ray images
IMAGE_PATH = '.../__allData/converted_images'
# Folder contains segmentation masks for left and right lung
mask_folder = '.../__allData/segmentation_masks/all_masks_machine'

def chunkify(lst, n):
    """Divide a list `lst` into `n` chunks."""
    return [lst[i::n] for i in range(n)]

@ray.remote
def process_folder(folder_path, chunks):
    for entry in chunks:
        fix_points_path = os.path.join(folder_path, entry, 'fix_points.csv')
        features_outpath = os.path.join(folder_path, entry, 'features.csv')
        if os.path.exists(fix_points_path) and not os.path.exists(features_outpath):
            fix_points_df = pd.read_csv(fix_points_path)

            # case_name = str(fix_points_df['case'].iloc[0])
            fname = fix_points_df['fname'].iloc[0]

            image = mpimg.imread(os.path.join(IMAGE_PATH, fname)) 
            mask_L = mpimg.imread(os.path.join(mask_folder, fname[0:-4] + '_left_lung.png'))
            mask_L = scipy.ndimage.zoom(mask_L, image.shape[0] / mask_L.shape[0], order = 0)
            mask_R = mpimg.imread(os.path.join(mask_folder, fname[0:-4] + '_right_lung.png'))
            mask_R = scipy.ndimage.zoom(mask_R, image.shape[0] / mask_R.shape[0], order = 0)

            seq=[]
            for i in range(0, len(fix_points_df)):
                x_seq = fix_points_df[['x', 'y']].loc[:i].values
                stats = gaze_statistics_case(x_seq, cut_off = 75)
                seq.append(stats.get_all_statistics(image, mask_L, mask_R, x_seq))

            col = ['fixation_dist', 'fixation_dist_std' , 
                    'fixation_angle' , 'fixation_angle_std' , 
                    '#_fixations', 'switches_between_objects', "total_length",
                    'info_gain_per_fixation_mean', 'info_gain_per_fixation_std',   
                    'acceleration_mean', 'acceleration_std' , 
                    '#_fixation_below_75pix' , '#_fixation_below_150pix', 
                    'visits_lung_L' , 'visits_lung_R' , 
                    'gaze_coverage_abs_L' , 'gaze_coverage_L',
                    'gaze_coverage_abs_R' , 'gaze_coverage_R' , 

                    '#_fixations_infogain_below_50pix' , 
                    '#_fixations_infogain_below_100pix', 
                    '#_fixations_infogain_below_200pix', 
                    '#_fixations_infogain_below_300pix', 
                    '#_fixations_infogain_below_5000pix', 
                    '%_fixations_infogain_below_50pix', 
                    '%_fixations_infogain_below_100pix', 
                    '%_fixations_infogain_below_200pix', 
                    '%_fixations_infogain_below_300pix', 
                    '%_fixations_infogain_below_5000pix']
            
            df = pd.DataFrame(seq, columns=col)
            df['x'] = fix_points_df['x']
            df['y'] = fix_points_df['y']
            columns_to_copy = ['case', 'radiologist', 'fname', 'win_width', 'win_height', 'label']
            values = fix_points_df[columns_to_copy].iloc[0]
            df = df.assign(**values.to_dict())
            df.to_csv(features_outpath, index=False)
            print(f"Preprocessed: {fix_points_path}")


if __name__ == "__main__":
    num_workers = 16
    ray.init(num_cpus=num_workers)
    
    for folder_path in FOLDER_NAMES:
        folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
        folder_chunks = chunkify(folders, int(num_workers))
        ray.get([process_folder.remote(folder_path, chunk) for chunk in folder_chunks])
