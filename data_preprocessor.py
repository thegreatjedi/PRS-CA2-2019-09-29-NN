from os import listdir, makedirs
from os.path import exists
from typing import Tuple

import numpy as np
from cv2 import cvtColor, imread, imwrite, IMREAD_UNCHANGED, COLOR_BGR2RGB
from h5py import File


def crop_images(root_img_dir: str = "./images") -> None:
    # Things to note:
    # cv2.imread image nparray output format:
    # Shape-3 images like JPG: [rows (height), columns (width), channels]
    # Shape-4 images like PNG: [rows, columns, channels, alpha (opacity)]
    # Image coordinate system used: [0,0] is top-left corner
    
    subdir_list = listdir(root_img_dir)
    for subdir in subdir_list:
        print(f"Cropping images in {subdir}...")
        dest_dir = f"./cropped_images/{subdir}"
        if not exists(dest_dir):
            makedirs(dest_dir)
        
        # Crop images in centre folder
        file_list = listdir(f"./images/{subdir}/centre")
        for file in file_list:
            print(f"Cropping centre folder: {file}", end="\r")
            # Only process if .png file
            if file.split(".")[-1] != "png":
                continue
            img = imread(f"./images/{subdir}/centre/{file}", IMREAD_UNCHANGED)
            if img is None:
                continue
            img_height = img.shape[0]
            img_width = img.shape[1]
            
            if img_height < img_width:
                # Crop centre of landscape
                left_x = int((img_width - img_height) / 2)
                cropped_img = img[:, left_x:left_x + img_height]
            elif img_width < img_height:
                # Crop centre of portrait
                top_y = int((img_height - img_width) / 2)
                cropped_img = img[top_y:top_y + img_width, :]
            else:
                cropped_img = img
                
            assert cropped_img.shape[0] == cropped_img.shape[1], \
                "Assert error in centre"
            imwrite(f"{dest_dir}/{file}", cropped_img)
        print("All centre images cropped.")
                
        # Crop images in landscape_left folder
        file_list = listdir(f"./images/{subdir}/landscape_left")
        for file in file_list:
            print(f"Cropping landscape_left folder: {file}", end="\r")
            # Only process if .png file
            if file.split(".")[-1] != "png":
                continue
            img = imread(f"./images/{subdir}/landscape_left/{file}",
                         IMREAD_UNCHANGED)
            img_height = img.shape[0]
            cropped_img = img[:, :img_height]
            assert cropped_img.shape[0] == cropped_img.shape[1], \
                "Assert error in landscape_left"
            imwrite(f"{dest_dir}/{file}", cropped_img)
        print("All landscape_left images cropped.")

        # Crop images in landscape_right folder
        file_list = listdir(f"./images/{subdir}/landscape_right")
        for file in file_list:
            print(f"Cropping landscape_right folder: {file}", end="\r")
            # Only process if .png file
            if file.split(".")[-1] != "png":
                continue
            img = imread(f"./images/{subdir}/landscape_right/{file}",
                         IMREAD_UNCHANGED)
            img_height = img.shape[0]
            img_width = img.shape[1]
            left_x = img_width - img_height
            cropped_img = img[:, left_x:]
            assert cropped_img.shape[0] == cropped_img.shape[1], \
                "Assert error in landscape_right"
            imwrite(f"{dest_dir}/{file}", cropped_img)
        print("All landscape_right images cropped.")

        # Crop images in portrait_bottom folder
        file_list = listdir(f"./images/{subdir}/portrait_bottom")
        for file in file_list:
            print(f"Cropping portrait_bottom folder: {file}", end="\r")
            # Only process if .png file
            if file.split(".")[-1] != "png":
                continue
            img = imread(f"./images/{subdir}/portrait_bottom/{file}",
                         IMREAD_UNCHANGED)
            img_height = img.shape[0]
            img_width = img.shape[1]
            top_y = img_height - img_width
            cropped_img = img[top_y:, :]
            assert cropped_img.shape[0] == cropped_img.shape[1], \
                "Assert error in portrait_bottom"
            imwrite(f"{dest_dir}/{file}", cropped_img)
        print("All portrait_bottom images cropped.")

        # Crop images in portrait_top folder
        file_list = listdir(f"./images/{subdir}/portrait_top")
        for file in file_list:
            print(f"Cropping portrait_top folder: {file}", end="\r")
            # Only process if .png file
            if file.split(".")[-1] != "png":
                continue
            img = imread(f"./images/{subdir}/portrait_top/{file}",
                         IMREAD_UNCHANGED)
            img_width = img.shape[1]
            cropped_img = img[:img_width, :]
            assert cropped_img.shape[0] == cropped_img.shape[1], \
                "Assert error in portrait_top"
            imwrite(f"{dest_dir}/{file}", cropped_img)
        print("All portrait_top images cropped.")
        print(f"All images in {subdir} cropped.")


def convert_images_to_dataset(root_img_dir: str = "./cropped_images",
                              output_filename: str = "data") -> None:
    images = None
    labels = None
    
    subdir_list = listdir(root_img_dir)
    for subdir in subdir_list:
        # HDF5 files cannot store Python-default UTF-8 encoded strings
        class_label = subdir.encode("ascii")
        print(f"Converting {subdir}...", end="\r")
        file_list = listdir(f"{root_img_dir}/{subdir}")
        num_files = len(file_list)
        for i in range(num_files):
            print(f"{subdir} images converted: {i}/{num_files}", end="\r")
            file = file_list[i]
            file_path = f"{root_img_dir}/{subdir}/{file}"
            # This only reads the height, width, and channel data
            # Alpha is ignored as it is not used by the neural network
            image_file = imread(file_path)
            # Change channel order from default BGR to RGB
            image_file = cvtColor(image_file, COLOR_BGR2RGB)
            
            # Save all data as ndarrays
            if images is None:
                images = np.array([image_file])
            else:
                images = np.append(images, [image_file], 0)
                
            if labels is None:
                labels = np.array([class_label])
            else:
                labels = np.append(labels, [class_label], 0)
        print(f"All {subdir} images converted.")
    
    hdf5_file = File(f"./{output_filename}.hdf5", mode="w")
    hdf5_file.create_dataset(name="image_data", data=images)
    hdf5_file.create_dataset(name="class_labels", data=labels)
    hdf5_file.close()
    
    
def import_dataset(filename: str, dir_path: str = ".") \
        -> Tuple[np.ndarray, np.ndarray]:
    filepath = f"{dir_path}/{filename}.hdf5"
    hdf5_file = File(filepath, "r")
    image_data = hdf5_file.get("image_data")[()]
    class_labels = hdf5_file.get("class_labels")[()]
    
    return image_data, class_labels


'''
if __name__ == "__main__":
    crop_images()
    convert_images_to_dataset()
    data, tags = import_dataset("data")
'''
