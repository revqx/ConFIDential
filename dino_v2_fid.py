# tar -xf ILSVRC2012_img_train.tar

### ---debug--- ###
"""
import os
print(len(os.listdir("data/ILSVRC2012_img_train")))
import sys
sys.exit()
"""
### ---debug--- ###

import os
import tarfile
import random

from tqdm import tqdm

import json
# from class_dict.py import class_dict

# Directory containing the extracted tar files
tar_dir = "data/ILSVRC2012_img_train"

# Dictionary to store the class counts
class_counts = {}

# Iterate over the tar files
for tar_file in tqdm(os.listdir(tar_dir)):
    if tar_file.endswith(".tar"):
        class_name = tar_file[:-4]  
        tar_path = os.path.join(tar_dir, tar_file)
        
        # Open the tar file and count the number of images
        with tarfile.open(tar_path, "r") as tar:
            num_images = len(tar.getnames())
            class_counts[class_name] = num_images

# Get the total number of images in each class
"""
for class_name, count in class_counts.items():
    print(f"Class: {class_name}, Count: {count}")
"""

### ---debug--- ###
"""
print(class_counts)
import sys
sys.exit()
"""
### ---debug--- ###

# Directory containing the extracted tar files
tar_dir = "data/ILSVRC2012_img_train"

# Directory to store the new validation set
val_dir = "data/ILSVRC2012_img_val"

# Desired size of the validation set
val_size = 50000

# Calculate the total number of images for all classes
total_images = sum(class_counts.values())

### ---debug--- ###
"""
print(f"Total number of images: {total_images}")
import sys
sys.exit()
"""
### ---debug--- ###

# Calculate the number of images to sample from each class to create the new validation set
# Ensure that in the end, the sum of samples is 50000 by distributing the remaining samples

class_samples = {}
for class_name, count in class_counts.items():
    sample_ratio = count / total_images
    num_samples = int(sample_ratio * val_size)
    class_samples[class_name] = num_samples

# Calculate the remaining samples
remaining_samples = val_size - sum(class_samples.values())

# Distribute the remaining samples to the classes with the highest sample ratios
sorted_samples = sorted(class_samples.items(), key=lambda x: x[1], reverse=True)
for i in range(remaining_samples):
    class_name, num_samples = sorted_samples[i]
    class_samples[class_name] += 1

# save the class samples to a JSON file
class_samples_path = "data/class_samples.json"
with open(class_samples_path, "w") as f:
    json.dump(class_samples, f)

# Save the class counts to a JSON file
class_counts_path = "data/class_counts.json"
with open(class_counts_path, "w") as f:
    json.dump(class_counts, f)

### ---debug--- ###
"""
print(class_samples)
import sys
sys.exit()
"""
### ---debug--- ###

# Create the new validation set
for class_name, num_samples in tqdm(class_samples.items()):
    tar_path = os.path.join(tar_dir, f"{class_name}.tar")
    
    # Open the tar file and randomly sample images
    with tarfile.open(tar_path, "r") as tar:
        image_names = tar.getnames()
        sampled_images = random.sample(image_names, num_samples)
        
        # Extract the sampled images to the validation directory
        for image_name in sampled_images:
            tar.extract(image_name, path=val_dir)

print("New validation set created successfully.")

"""
python -m dgm_eval path/to/new_valid_dataset path/to/generated_dataset \
--model dinov2 \
--metrics fd kd prdc ct
"""
