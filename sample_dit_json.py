import json
import os
import shutil
import random

# Load the JSON file
with open('data/class_samples.json') as f:
    class_distribution = json.load(f)

# Define the paths
dataset_folder = 'data/gen_img_dit_ori'
output_folder = 'data/gen_img_dit'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over each class
for class_index, (class_label, num_samples) in enumerate(class_distribution.items()):
    # Get the list of image names for the current class
    image_names = [f"sample_{class_index}_{i}.png" for i in range(51)]
    
    # Check if the required number of samples is greater than the available images
    if num_samples > len(image_names):
        print(f"Warning: Class {class_label} has fewer images than the required samples. Sampling all available images.")
        num_samples = len(image_names)
    
    # Randomly sample the required number of images
    sampled_images = random.sample(image_names, num_samples)
    
    # Copy the sampled images to the output folder
    for image_name in sampled_images:
        src_path = os.path.join(dataset_folder, image_name)
        dst_path = os.path.join(output_folder, image_name)
        shutil.copy(src_path, dst_path)

print("Sampling completed.")