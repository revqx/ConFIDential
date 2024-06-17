import os
import shutil
import random

import argparse

# Define the paths
# dataset_folder = 'data/gen_img_MDT'
# output_folder = 'data/gen_img_MDT_uni'

def main(args):
    # Create the output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Get the total number of images
    total_images = len(os.listdir(args.dataset_folder))

    # Create a list of file names inside dataset_folder
    # image_names = os.listdir(dataset_folder)

    # example: 0_000000.jpg

    image_names = [f"{class_}_{i:06d}.png" for class_ in range(1000) for i in range(51)]

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
            src_path = os.path.join(args.dataset_folder, image_name)
            dst_path = os.path.join(args.output_folder, image_name)
            shutil.copy(src_path, dst_path)

    print("Sampling completed!.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, default='data/gen_img_MDT')
    parser.add_argument('--output_folder', type=str, default='data/gen_img_MDT_uni')
    args = parser.parse_args()
    main(args)
