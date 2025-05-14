import os
import torch
import pickle
import torchvision
import numpy as np
from PIL import Image, ImageDraw
import torchxrayvision as xrv
from pycpd import RigidRegistration
import pandas as pd

# Load the pre-trained model
# https://github.com/mlmed/torchxrayvision 
model = xrv.baseline_models.chestx_det.PSPNet()

def save_dicts(my_dict, json_file_name):
    # Save to JSON file
    with open(json_file_name+'.pkl', 'wb') as json_file:
        pickle.dump(my_dict, json_file)

def draw_on_image(image, points, fill='#a31d20'):
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    point_size = 2
    for i, j in points:
        draw.ellipse((j - point_size, i - point_size, j + point_size, i + point_size), fill=fill)
    return image

def find_white_pixels(img_array):
    white_pixels = np.argwhere(img_array == img_array.max())
    return white_pixels

def return_binary_predictions(model, image):
    img = xrv.datasets.normalize(np.asarray(image), 255) # convert 8-bit image to [-1024, 1024] range
    img = img[None, ...] # Make single color channel
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(512)])

    img = transform(img)
    img = torch.from_numpy(img)

    with torch.no_grad():
        pred = model(img)

    pred = 1 / (1 + np.exp(-pred))  # sigmoid
    pred[pred < 0.5] = 0
    pred[pred > 0.5] = 1
    return pred

def return_binary_mask(pred, target_indices): #[0, 1, 4, 5, 6, 7, 8, 9, 11, 12]
    selected_masks = pred[:, target_indices, :, :]
    final_mask = torch.any(selected_masks.type(torch.bool), dim=1, keepdim=True)
    return final_mask.float().squeeze().numpy()

def cpd(target, source):
    reg = RigidRegistration(X=target, Y=source)
    TY, (s_reg, R_reg, t_reg) = reg.register()
    return TY, (s_reg, R_reg, t_reg)

def delete_points(target, dist=10):
    filtered_points = []
    prev_point = target[0]
    filtered_points.append(prev_point)

    for point in target[1:]:
        if distance(point, prev_point) >= dist:
            filtered_points.append(point)
            prev_point = point
    return np.array(filtered_points)

def distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def define_dist(target, dist=10):
    filtered_points = []
    for point in target:
        if all(distance(point, prev_point) >= dist for prev_point in filtered_points):
            filtered_points.append(point)
    return np.array(filtered_points)

def reduce_number_of_points(coord, dist1, dist2):
    target = delete_points(coord[coord[:, 0].argsort()], dist1)
    target = define_dist(target, dist2)
    return target

def get_mask_and_pixels(image_name, 
                        dist1=50, 
                        dist2=5, 
                        target_indices=[0, 1, 4, 5, 6, 7, 8, 9, 11, 12]):
    # If you do not have prepared masks, uncomment the following lines:
    # image = image.resize((512,512))
    # pred = return_binary_predictions(model, image)
    # mask = return_binary_mask(pred, target_indices)
    # mask = Image.fromarray(mask.astype(np.uint8)).resize((224,224))
    # Otherwise, load the mask from the directory:
    mask = Image.open(os.path.join('.../__allData/miccai2025/masked_images_512size', image_name)).resize((224,224))
    white_pixels = find_white_pixels(np.asarray(mask))
    points = reduce_number_of_points(white_pixels, dist1, dist2)
    return mask, white_pixels, points

def calculate_intersection_area(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    return intersection_area

def should_draw_box(new_box, squares, square_size, threshold=0.3):
    new_box_area = square_size * square_size
    for box in squares:
        intersection_area = calculate_intersection_area(new_box, box)
        if intersection_area / new_box_area > threshold:
            return False
    return True

def get_target_box_position(target, square_size, threshold=0.3):
    squares = []
    squares_coord = []
    for i, (y,x) in enumerate(target):
        left = x - square_size // 2
        top = y - square_size // 2
        right = x + square_size // 2
        bottom = y + square_size // 2
        new_box = [left, top, right, bottom]
        
        if should_draw_box(new_box, squares, square_size, threshold):
            # draw1.rectangle(new_box, outline='blue')
            squares.append(new_box )
            squares_coord.append([y, x])
    return squares , squares_coord

def is_point_in_box(point, box):
    x, y = point
    left, top, right, bottom = box
    return left <= x < right and top <= y < bottom

def prepare_anatomical_patches(image_path, 
                            folder_path,
                            folder_names, 
                            target_squares_coord,
                            patch_size,
                            anatomical_path):
    for patient_case_name in folder_names:
        path_fixpoints = os.path.join(folder_path, patient_case_name, 'fix_points.csv')
        path_savefolder = os.path.join(folder_path, patient_case_name, 'miccai2025')
        if os.path.exists(path_fixpoints):
            df_fixpoints = pd.read_csv(path_fixpoints).reset_index(drop=True)
            source_image = Image.open(os.path.join(image_path, patient_case_name))

            source_mask, source_white_pixels, source = get_mask_and_pixels(patient_case_name)
            source_image = source_image.resize((224,224))
            scale_x = 224 / df_fixpoints['win_width'].iloc[0]
            scale_y = 224 / df_fixpoints['win_height'].iloc[0]

            TY, (s_reg, R_reg, t_reg) = cpd(source, np.array(target_squares_coord))

            important_patches = []
            cropped_patches = []
            patch_numbers = np.zeros(len(df_fixpoints), dtype=int)
            patch_numbers = patch_numbers.astype(float)
            patch_numbers[patch_numbers == 0] = np.nan

            for i, (y, x) in enumerate(TY):
                left = x - patch_size // 2
                top = y - patch_size // 2
                right = x + patch_size // 2
                bottom = y + patch_size // 2
                new_box = [left, top, right, bottom]
                cropped_img = source_image.crop(new_box)
                cropped_patches.append(np.asarray(cropped_img))

                contains_points = False
                for idx, (data_y, data_x) in enumerate(df_fixpoints[['y', 'x']].values):
                    data_x = data_x * scale_x
                    data_y = data_y * scale_y
                    if is_point_in_box((data_x, data_y), new_box):
                        contains_points = True
                        patch_numbers[idx] = i
                important_patches.append(1 if contains_points else 0)

            if not os.path.exists(path_savefolder):
                os.makedirs(path_savefolder)
            np.save(os.path.join(path_savefolder, 'anatomic_important_patches_' + str(patch_size)),  np.array(important_patches))
            np.save(os.path.join(path_savefolder, 'anatomic_patch_number_' + str(patch_size)),  np.array(patch_numbers))  
            if not os.path.exists(os.path.join(anatomical_path, patient_case_name)):
                np.save(os.path.join(anatomical_path, patient_case_name), np.array(cropped_patches)) 



def create_mask(png_files, output_dir):
    # png_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.png')]

    target_indices = [0, 1, 4, 5, 6, 7, 8, 9, 11, 12]

    for pngfile in png_files:
        image = Image.open(pngfile).resize((512,512))
        img = xrv.datasets.normalize(np.array(image), 255) # convert 8-bit image to [-1024, 1024] range
        img = img[None, ...] # Make single color channel
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(512)])
        img = transform(img)
        img = torch.from_numpy(img)

        with torch.no_grad():
            pred = model(img)
        
        pred = 1 / (1 + np.exp(-pred))  # sigmoid
        pred[pred < 0.5] = 0
        pred[pred > 0.5] = 1

        selected_masks = pred[:, target_indices, :, :]
        final_mask = torch.any(selected_masks.type(torch.bool), dim=1, keepdim=True)
        mask = final_mask.float().squeeze().numpy()
        final_mask = Image.fromarray(mask.astype('uint8')*255)

        png_name = os.path.basename(pngfile)
        final_mask.save(os.path.join(output_dir, png_name))
