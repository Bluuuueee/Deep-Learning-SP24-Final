import os
import numpy as np
from shutil import copy2
import re

base_path_train = '/teamspace/studios/this_studio/Ragnor/dataset/train'
images_path_train = '/teamspace/studios/this_studio/Ragnor/dataset/train_new/image'
labels_path_train = '/teamspace/studios/this_studio/Ragnor/dataset/train_new/label'

base_path_val = '/teamspace/studios/this_studio/Ragnor/dataset/val'
images_path_val = '/teamspace/studios/this_studio/Ragnor/dataset/val_new/image'
labels_path_val = '/teamspace/studios/this_studio/Ragnor/dataset/val_new/label'

def preprocess(base_path, images_path, labels_path):
    # Iterate through each folder in the train directory
    def extract_number(filename):
        return int(re.search(r'\d+', filename).group())

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            try:
                base_filename = os.path.basename(folder)  # Define base_filename here

                mask_file = os.path.join(folder_path, 'mask.npy')
                mask = np.load(mask_file)

                # Verify the mask shape
                if mask.shape == (22, 160, 240):
                    # Split the mask and save each as a corresponding file
                    for i in range(mask.shape[0]):
                        new_mask_filename = f'{base_filename}_frame_{i}.npy'
                        np.save(os.path.join(labels_path, new_mask_filename), mask[i])

                # Sort and rename images to correspond with mask names
                image_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]
                print('A:', image_files)
                image_files = sorted(image_files, key=extract_number)  # Sort files to maintain consistent ordering
                print('B:', image_files)
                for j, file in enumerate(image_files):
                    new_image_filename = f'{base_filename}_frame_{j}.png'
                    copy2(os.path.join(folder_path, file), os.path.join(images_path, new_image_filename))
            except Exception as e:
                print(f"Error processing folder {folder}: {e}")

preprocess(base_path_train, images_path_train, labels_path_train)
preprocess(base_path_val, images_path_val, labels_path_val)




