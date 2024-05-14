import json
import os
import shutil
import random

# Load the JSON file
with open('data/class_samples.json') as f:
    class_distribution = json.load(f)

# Define the paths
dataset_folder = 'data/gen_img_ldm_ori/'
output_folder = 'data/gen_img_ldm/'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get the total number of images
total_images = len(os.listdir(dataset_folder))
print(total_images)

# Create a list of image names
image_names = [f"{i:06d}.png" for i in range(total_images)]

# Iterate over each class
for class_index, (class_label, num_samples) in enumerate(class_distribution.items()):
    # Get the indices of the images belonging to the current class
    start_index = class_index
    class_image_indices = list(range(start_index, total_images, len(class_distribution)))
    
    # Check if the required number of samples is greater than the available images
    if num_samples > len(class_image_indices):
        print(f"Warning: Class {class_label} has fewer images than the required samples. Sampling all available images.")
        num_samples = len(class_image_indices)
    
    # Randomly sample the required number of images
    sampled_image_indices = random.sample(class_image_indices, num_samples)
    sampled_image_names = [image_names[i] for i in sampled_image_indices]
    
    # Copy the sampled images to the output folder
    for image_name in sampled_image_names:
        src_path = os.path.join(dataset_folder, image_name)
        dst_path = os.path.join(output_folder, image_name)
        shutil.copy(src_path, dst_path)

print("Sampling completed.")

# DEBUG
total_sampled_images = len(os.listdir(output_folder))
print(total_sampled_images)

import sys
sys.exit()
