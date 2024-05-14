import os
import shutil
import random

# Define the paths
dataset_folder = 'data/gen_img_ldm_ori'
output_folder = 'data/gen_img_ldm_uni'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get the total number of images
total_images = len(os.listdir(dataset_folder))

# Create a list of image names
image_names = [f"{i:06d}.jpg" for i in range(total_images)]

# Get the number of classes
num_classes = total_images // 51

# Iterate over each class
for class_index in range(num_classes):
    # Get the indices of the images belonging to the current class
    start_index = class_index
    class_image_indices = list(range(start_index, total_images, num_classes))
    
    # Randomly sample 50 images from the class
    sampled_image_indices = random.sample(class_image_indices, 50)
    sampled_image_names = [image_names[i] for i in sampled_image_indices]
    
    # Copy the sampled images to the output folder
    for image_name in sampled_image_names:
        src_path = os.path.join(dataset_folder, image_name)
        dst_path = os.path.join(output_folder, image_name)
        shutil.copy(src_path, dst_path)

print("Sampling completed.")