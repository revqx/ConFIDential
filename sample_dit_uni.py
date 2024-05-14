import os
import shutil
import random

# Define the paths
dataset_folder = 'path/to/dataset/folder'
output_folder = 'path/to/output/folder'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get the list of all image names in the dataset folder
all_image_names = os.listdir(dataset_folder)

# Extract the class indices from the image names
class_indices = sorted(set(int(name.split('_')[1]) for name in all_image_names))

# Iterate over each class
for class_index in class_indices:
    # Get the list of image names for the current class
    image_names = [f"sample_{class_index}_{i}.png" for i in range(51)]
    
    # Randomly sample 50 images from the class
    sampled_images = random.sample(image_names, 50)
    
    # Copy the sampled images to the output folder
    for image_name in sampled_images:
        src_path = os.path.join(dataset_folder, image_name)
        dst_path = os.path.join(output_folder, image_name)
        shutil.copy(src_path, dst_path)

print("Sampling completed.")