import os
import numpy as np

IMAGENET_TRAIN_PATH = '/home/ra78lof/fid-flaws/data/imagenet_train/'
IMAGE_NET_TRAIN_SUBSET_PATH = '/home/ra78lof/fid-flaws/data/imagenet_train_subset/'

# load all the tar files inside
tar_files = [os.path.join(IMAGENET_TRAIN_PATH, f) for f in os.listdir(IMAGENET_TRAIN_PATH) if f.endswith('.tar')]

# untar all the files into a new directory called like the tar file
# if the folder already exists, it will not be created
for tar_file in tar_files[:10]:
    # if the folder already exists and contains the files, skip
    if os.path.exists(tar_file[:-4]) and len(os.listdir(tar_file[:-4])) > 0:
        continue

    os.makedirs(tar_file[:-4], exist_ok=True)
    os.system('tar -xf ' + tar_file + ' -C ' + tar_file[:-4])

# load all the folders
folders = [f for f in os.listdir(IMAGENET_TRAIN_PATH) if os.path.isdir(os.path.join(IMAGENET_TRAIN_PATH, f))][:10]
print(len(folders))
assert(len(folders) == 10)

# get the number of images per folder
num_images_per_class = []
images_per_class = {}

for folder in folders:
    num_images_per_class.append(len(os.listdir(IMAGENET_TRAIN_PATH + folder)))
    images_per_class[folder] = len(os.listdir(IMAGENET_TRAIN_PATH + folder))

# sample 50000 images according to the distribution
num_images = np.array(num_images_per_class)
prob = num_images / num_images.sum()

# sample 50000 images according to the distribution from images_per_class
sampled_images = {}
for folder in images_per_class:
    sampled_images[folder] = int(np.round(images_per_class[folder] * 50000 / num_images.sum()))

# copy the sampled images to a new directory
os.makedirs(IMAGE_NET_TRAIN_SUBSET_PATH, exist_ok=True)

for folder in sampled_images:
    sampled_images_folder = sampled_images[folder]
    images = os.listdir(IMAGENET_TRAIN_PATH + folder)
    np.random.shuffle(images)

    for i in range(sampled_images_folder):
        os.system('cp ' + IMAGENET_TRAIN_PATH + folder + '/' + images[i] + ' ' + IMAGE_NET_TRAIN_SUBSET_PATH)

