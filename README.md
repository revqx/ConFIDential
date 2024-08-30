# fid-flaws
In this repo, we are going to explore the performance of several newest SOTA model in the filed of Image Generation, this includes [DiT](https://arxiv.org/abs/2212.09748), [latent-diffusion](https://arxiv.org/abs/2112.10752), [MDTv2](https://arxiv.org/abs/2303.14389), [MaskDiT](https://arxiv.org/abs/2306.09305), [StyleGAN-XL](https://arxiv.org/abs/2202.00273), [U-DiTs](https://arxiv.org/abs/2405.02730), [VAR](https://arxiv.org/abs/2404.02905), and the newest SOTA paper [mar](https://arxiv.org/abs/2406.11838). 

## Starting Point ⭐
To start with understanding this repo's structure, here are the folders inside this repo:
```shell
├── DiT
├── MDT
├── MaskDiT
├── README.md
├── U-DiT
├── VAR
├── imagenet_class_index.json
├── latent-diffusion
├── mar
├── scripts
├── src
└── stylegan-xl
``` 
Each Model has an independent folder, if you want to use those models to generate images based on your own research demands, please first run `git clone  https://github.com/revqx/fid-flaws`, then `cd` into the model folder you want to test.

The `scripts` folder includes some helper files: 
```shell
├── create_distribution_folders.py
└── generate_distribution_files.py
```
The `generate_distribution_files.py` script is used to generate distribution `.txt` files, while the `create_distribution_folders.py` script is used to create different distribution folders for FiD score evaluation.

## Terminal Command Guidance 🔥
First, you need to run the following commands to make sure all the submodules are activated correctly:
```shell
git clone https://github.com/revqx/fid-flaws
cd fid-flaws
git submodule update --init --recursive
```

To generate 50 images for each class in ImageNet by using [StyleGAN-XL](https://arxiv.org/abs/2202.00273) model, please run the following commands:
```shell
cd stylegan-xl
python generation_single_gpu.py \
--outdir=samplesheet --trunc=1.0 \
--network=https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet256.pkl \
--num-classes 1000 \
--num-samples-per-class 50 \
--batch-size 32
```

To generate 50 images for each class in ImageNet by using [latent-diffusion](https://arxiv.org/abs/2112.10752) model, please run the following commands:
```shell
cd latent-diffusion
conda env create -f environment.yaml
conda activate ldm
python generation_single_gpu.py
```

To generate 50 images for each class in ImageNet by using [MDTv2](https://arxiv.org/abs/2303.14389) model, please run the following commands:
```shell
conda create -n MDT python==3.10
conda init
conda activate MDT

pip install -r requirements.txt

wget https://huggingface.co/shgao/MDT-XL2/resolve/main/mdt_xl2_v2_ckpt.pt

python generation_single_gpu.py --tf32
```
You can also run the default setting generation file(_50 images for each class, 50k in total_) by one line of command:
```shell
sh generation.bash
```

To generate 50 images for each class in ImageNet by using [MaskDiT](https://arxiv.org/abs/2306.09305) model, please run the following commands:
```shell
conda create -n MaskDiT python==3.10
conda activate MaskDiT
pip install -r requirements.txt

wget https://hzpublic-hub.s3.us-west-2.amazonaws.com/maskdit/checkpoints/imagenet256-ckpt-best_with_guidance.pt
python3 download_assets.py --name vae --dest assets/stable_diffusion
bash scripts/download_assets.sh

python generate_single_gpu.py \
--config configs/test/maskdit-256.yaml \
--ckpt_path PATH_TO_CHECKPOINTS \
--cfg_scale GUIDANCE_SCALE \
--num_images_per_class IMAGE_PER_CLASS \ 
--tf 32
```
You can also run the default setting generation file(_50 images for each class, 50k in total_) by one line of command:
```shell
sh buildup.bash
```

To generate 50 images for each class in ImageNet by using [VAR](https://arxiv.org/abs/2404.02905) model, please run the following commands:
```shell
conda create -n var python==3.10
pip install -r requirements.txt
python generation_single_gpu.py
```

To generate 50 images for each class in ImageNet by using [DiT](https://arxiv.org/abs/2212.09748) model, please run the following commands:
```shell
conda env create -f environment.yml
conda activate DiT
wget https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt
python sample.py \
--num_classes 1000 \ 
--cfg-scale 1.5 \ 
--batch_size 32 \
--images_per_class 50 \
--ckpt /path/to/model.pt
```

To generate 50 images for each class in ImageNet by using [mar](https://arxiv.org/abs/2406.11838) model, please run the following commands:
```shell
conda env create -f environment.yaml
conda activate mar

python util/download.py

python generation_single_gpu.py \
--batch-size 32 \
--cfg-scale 1.5 \
--cfg-schedule constant \
--samples-per-class 50 \
--tf32 # the tf32 will accelerate the generation 
```


## Preliminary FID Results

| Model                   | Uniform (50 per class) | Uniform (50k times random choice of 1000 classes) | Real (Underlying ImageNet distribution ~50k) |
|-------------------------|------------------------|---------------------------------------------------|----------------------------------------------|
| VAR (seed=42)           | 5.36 | 5.41 | 5.42 |
| MDT (seed=42)           | 2.28 | 2.30 | 2.27 |
| DiT                     | 2.82 | 2.83 | 2.79 |
| LDM                     | 3.56 | 3.54 | 3.53 |
| StyleGAN-XL (seed=1000) | 2.60 | 2.56 | 2.60 |
| StyleGAN-XL (seed=42)   | 2.61 | 2.55 | 2.56 |
| MaskedDiT               | 2.32 | 2.34 | 2.30 |
| LlamaGen                | 2.81 | 2.79 | 2.78 |
| U-ViT                   | 2.73 | 2.70 | 2.66 |
| U-DiT                   | 2.98 | 2.95 | 2.93 |
| Mar                     | 2.18 | 2.21 | 2.15 |

## Preliminary FDD Result

| Model                   | Uniform (50 per class) | Uniform (50k times random choice of 1000 classes) | Real (Underlying ImageNet distribution ~50k) |
|-------------------------|------------------------|---------------------------------------------------|----------------------------------------------|
| VAR (seed=42)           | 117.5 | 118.0 | 117.39 |
| MDT (seed=42)           | 57.82 | 58.0 | 57.5 |
| DiT                     | 68.0 | 68.5 | 67.5 |
| LDM                     | 132.45 | 133.53 | 133.56 |
| StyleGAN-XL (seed=1000) | 133.85 | 133.80 | 132.88 |
| StyleGAN-XL (seed=42)   | 133.56 | 133.53 | 132.45 |
| MaskedDiT               | 59.0 | 59.5 | 58.5 |
| LlamaGen                | 68.0 | 67.5 | 67.0 |
| U-ViT                   | 64.87 | 65.36 | 65.56 |
| U-DiT                   | 70.5 | 70.0 | 69.5 |
| Mar                     | 55.0 | 56.0 | 54.5 |
