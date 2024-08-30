"""
This script prepares image datasets for deep generative model evaluation by creating several distributions given a folder of generated images:
1. A real distribution as specified by an input file.
2. A uniform distribution with exactly 50 images per class.
3. A randomly uniform distribution with a total of 50,000 images sampled across all classes.
The datasets are structured for use with specific evaluation frameworks.
Random seed 42 is used for reproducibility.

Usage:
python create_distribution_folders.py --generated_images_folder <path_to_generated_images> --dist_file <path_to_distribution_indices_txt>

Args:
--generated_images_folder (str): Path to the folder containing the generated images.
--dist_file (str): Path to the txt file containing the real image distribution. This can be generated using generate_distribution_files.py

Supports two file formats:
    - 'class_{class_num}_{...}.png': Direct class number in the filename.
    - '{000000-999999}.png': Sequential filenames where every 80 files belong to the next class.

Author: Marius Jacobs
"""

import os
import random
import argparse
from tqdm import tqdm
import re

def copy_images(source_paths, target_folder, start_index=0):
    """
    Copies images from source paths to a target folder with a new naming scheme and displays a progress bar.
    
    Args:
    source_paths (list): List of paths to source images.
    target_folder (str): Directory to copy the images to.
    start_index (int): Starting index for renaming images in the target directory.
    """
    for idx, path in tqdm(enumerate(source_paths, start=start_index), total=len(source_paths), desc='Copying images'):
        target_path = os.path.join(target_folder, f'{idx:06d}.png')
        os.system(f'cp {path} {target_path}')

def sample_images(class_image_paths_dict, num_samples):
    """
    Samples a specified number of images from each class in the dictionary.
    """
    sampled_paths_dict = {}
    for class_num, paths in class_image_paths_dict.items():
        if len(paths) < num_samples:
            raise ValueError(f"Class {class_num} has less than {num_samples} images.")
        sampled_paths_dict[class_num] = random.sample(paths, num_samples)
    return sampled_paths_dict

def prepare_class_image_dict(source_folder):
    """
    Organizes images by class from a specified folder into a path dictionary.
    Supports two file formats.
        - 'class_{class_num}_{...}.png': Direct class number in the filename.
        - '{000000-999999}.png': Sequential filenames where every 80 files belong to the next class.
    """
    files = os.listdir(source_folder)
    class_image_paths_dict = {}
    sample_file = files[0] if files else None

    if sample_file and re.match(r'class_\d+_.+\.png', sample_file):
        for file in files:
            class_num = int(file.split('_')[1])
            full_path = os.path.join(source_folder, file)
            class_image_paths_dict.setdefault(class_num, []).append(full_path)
    elif sample_file and re.match(r'\d{6}\.png', sample_file):
        for index, file in enumerate(files):
            class_num = index // 80
            full_path = os.path.join(source_folder, file)
            class_image_paths_dict.setdefault(class_num, []).append(full_path)
    else:
        raise ValueError("Unsupported file format detected or directory is empty.")
    return class_image_paths_dict

def main(args):
    """
    Main function to prepare image distributions.
    """
    generated_images_folder = os.path.abspath(args.generated_images_folder)
    if not os.path.exists(generated_images_folder):
        raise FileNotFoundError(f'{generated_images_folder} not found')
    
    # Set random seed 42 for reproducibility
    random.seed(42)

    # Create folders for the different distributions at the directory level of the generated images folder
    base_folder = os.path.dirname(generated_images_folder)
    real_folder = os.path.join(base_folder, 'real')
    uniform_50_folder = os.path.join(base_folder, 'uniform_50')
    uniform_random_folder = os.path.join(base_folder, 'uniform_random')
    
    os.makedirs(real_folder, exist_ok=True)
    os.makedirs(uniform_50_folder, exist_ok=True)
    os.makedirs(uniform_random_folder, exist_ok=True)

    class_image_paths_dict = prepare_class_image_dict(generated_images_folder)

    try:
        with open(args.dist_file, 'r') as f:
            real_distribution = list(map(int, f.read().strip().split(',')))
    except FileNotFoundError:
        print(f"Error: Distribution file {args.dist_file} not found.")
        return

    real_class_counts = {class_num: 0 for class_num in real_distribution}
    for class_num in real_distribution:
        real_class_counts[class_num] += 1

    real_samples = {}
    for class_num, count in real_class_counts.items():
        real_samples[class_num] = random.sample(class_image_paths_dict[class_num], count)

    copy_images([img for sublist in real_samples.values() for img in sublist], real_folder)
    uniform_50_samples = sample_images(class_image_paths_dict, 50)
    copy_images([img for sublist in uniform_50_samples.values() for img in sublist], uniform_50_folder)
    all_images = [path for paths in class_image_paths_dict.values() for path in paths]
    random.shuffle(all_images)
    copy_images(all_images[:50000], uniform_random_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_images_folder', required=True, type=str, help='Path to the folder containing the generated images')
    parser.add_argument('--dist_file', type=str, default='distribution_files/dist=real_total=50234_seed=42_num_classes=1000.txt', help='Path to the txt file containing the real image distribution')
    args = parser.parse_args()
    main(args)