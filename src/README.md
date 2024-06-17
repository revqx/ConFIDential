# How to use files in this folder to generate images beased on different models

### DiT image generation commands
To generate images based on pretrained DiT model, please run the following commands:
```shell
cd /fid-flaws/src/models
python DiT_sample.py --cfg-scale 1.5 --seed 42
```