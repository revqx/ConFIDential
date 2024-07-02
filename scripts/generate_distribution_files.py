"""
This script generates a list of class indices according to a specified distribution.

- specify the number of classes (default: 1000)
- specify the total number of images (default: 50000)
- select the distribution to follow (required)
    - uniform-static (a fixed number of images per class, just integer divsion of total by num of classes)
    - uniform-random (for each sample the class will be chosen randomly)
    - real-distribution according to a given json that contains class labels as keys and its number of images
      (resulting number of images will be total images specified in 2. plus minus something due to rounding)
- set a random seed (default: 42)

Further implementation details:
    - argparse to specify the arguments 
    - the list of labels should be printed in the terminal
    - the list of labels should be saved as a comma-seperated txt file into /class_lists whose name contains the arguments

Random seed is set to 42 for reproducibility.

Usage:
    python generate_distribution_files.py --dist uniform-static
    python generate_distribution_files.py --dist uniform-random
    python generate_distribution_files.py --dist real --dist-file distribution_files/real_distribution.json

Author: Marius Jacobs
"""

import argparse
import json
import os
import random
from typing import List

def parse_args():
    """
    Parse the arguments
    """

    parser = argparse.ArgumentParser(description='Generate a list of class indices according to a specified distribution.')
    parser.add_argument('--dist', type=str, required=True, help='Distribution to follow')
    parser.add_argument('--total_images', type=int, default=50000, help='Total number of images')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of classes')
    parser.add_argument('--dist-file', type=str, default=None, help='File containing the real distribution, required if distribution is real-distribution')
    args = parser.parse_args()

    return args

def generate_class_list(num_classes: int, total_images: int, distribution: str, distribution_file: str, seed: int) -> List[int]:
    """
    Generate a list of class indices according to a specified distribution
    """

    random.seed(42)

    if distribution == 'uniform-static':
        images_per_class = total_images // num_classes
        class_list = [i for i in range(num_classes) for _ in range(images_per_class)]
    elif distribution == 'uniform-random':
        class_list = [random.randint(0, num_classes - 1) for _ in range(total_images)]
    elif distribution == 'real':
        if distribution_file is None:
            raise ValueError('Distribution file required for real-distribution')
        if not os.path.exists(distribution_file):
            raise FileNotFoundError(f'{distribution_file} not found')
        
        with open(distribution_file, 'r') as f:
            real_distribution = json.load(f)

        class_list = []
        total_real_distribution = sum(real_distribution.values())
        for i, count in enumerate(real_distribution.values()):
            class_count = round(count / total_real_distribution * total_images)
            class_list += [i] * class_count

    else:
        raise ValueError('Invalid distribution')

    return class_list

def save_class_list(class_list: List[int], num_classes: int, total_images: int, distribution: str, seed: int):
    """
    Save the list of class indices as a comma-seperated txt file
    """

    class_list_file = f'distribution_files/dist={distribution}_{len(class_list)}_seed={seed}_num_classes={num_classes}.txt'
    with open(class_list_file, 'w') as f:
        f.write(','.join(map(str, class_list)))

    print('Total number of images:', len(class_list))
    print(f'Class list saved as {class_list_file}')

def main():
    args = parse_args()
    class_list = generate_class_list(args.num_classes, args.total_images, args.dist, args.dist_file, args.seed)
    save_class_list(class_list, args.num_classes, args.total_images, args.dist, args.seed)

if __name__ == '__main__':
    main()