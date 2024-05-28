# Command Line for running experiment.

## Compute the FID score
### resampled valid & resampled generated 
run the following command to get the FID score based on real distribution valid set and real distribution generated set:
```shell
python -m dgm_eval /personal_storage/scout/fid-flaws/data/ILSVRC2012_img_val /personal_storage/scout/fid-flaws/data/g
en_img_stylegan_all  --model inception  --metrics fd
```
Our original experiment results for FID score is: `3.475`

### uniform valid & resampled generated
run the following command to get the FID score based on uniform distribution and real distribution generated set:
```shell
python -m dgm_eval /personal_storage/scout/fid-flaws/data/ILSVRC2012_img_uni /personal_storage/scout/fid-flaws/data/g
en_img_stylegan_all  --model inception  --metrics fd
```
Our original experiment results for FID score is: `3.484`

### resampled valid & uniform generated
run the following command to get the FID score based on real distribution and uniformly generated set:
```shell
python -m dgm_eval /personal_storage/scout/fid-flaws/data/ILSVRC2012_img_val  /personal_storage/scout/fid-flaws/data/gen_img_stylegan_uni --model inception --metrics fd
```

### resampled valid & uniform generated
run the following command to get the FID score based on uniformly distribution and uniformly generated set:
```shell
python -m dgm_eval /personal_storage/scout/fid-flaws/data/ILSVRC2012_img_uni  /personal_storage/scout/fid-flaws/data/gen_img_stylegan_uni --model inception --metrics fd
```

### command for dinov2
```shell
python -m dgm_eval /personal_storage/scout/fid-flaws/data/ILSVRC2012_img_uni  /personal_storage/scout/fid-flaws/data/gen_img_stylegan --model dinov2 --metrics fd
```
Our original experiment results for FID score is: `217.07`

### command for calculating FID scores by using dinov2 for ImageNet dataset and DiT generated dataset.
```shell
python -m dgm_eval /personal_storage/scout/fid-flaws/data/ILSVRC2012_img_uni  /personal_storage/scout/fid-flaws/data/gen_img_dit --model dinov2 --metrics fd
```
Our original experiment results for FID score is: `98.69`

### command for calculating FID scores by using dinov2 for ImageNet dataset and latent diffusion model generated dataset.
```shell
python -m dgm_eval /personal_storage/scout/fid-flaws/data/ILSVRC2012_img_uni  /personal_storage/scout/fid-flaws/data/gen_img_ldm --model dinov2 --metrics fd
```
Our original experiment results for FID score is: `99.98`

### command for calculating FID scores by using dinov2 for ImageNet dataset and latent diffusion model generated dataset with unified distribution.
```shell
python -m dgm_eval /personal_storage/scout/fid-flaws/data/ILSVRC2012_img_uni  /personal_storage/scout/fid-flaws/data/gen_img_ldm_ori/ --model dinov2 --metrics fd
```
Our original experiment results for FID score is `99.56`


## Compute the FID score 50k mentioned in those papers
### command for calculating fid50k based on real distribution ldm
```shell
python -m dgm_eval /personal_storage/scout/fid-flaws/data/ILSVRC2012_img_jpeg  /personal_storage/scout/fid-flaws/data/gen_img_ldm_ori/ --model inception --metrics fd --save --nsample 1500000
```
Our result for FiD50k is `12.67538`.

### command for calculating fid50k based on uniform distribution ldm
```shell
python -m dgm_eval /personal_storage/scout/fid-flaws/data/ILSVRC2012_img_jpeg  /personal_storage/scout/fid-flaws/data/gen_img_ldm/ --model inception --metrics fd --save --nsample 1500000
```
Our result for FiD50k is `12.73546`.

### command for calculating fid50k based on uniform distribution dit
```shell
# uniform dist
python -m dgm_eval /personal_storage/scout/fid-flaws/data/ILSVRC2012_img_jpeg  /personal_storage/scout/fid-flaws/data/gen_img_dit_ori/ --model inception --metrics fd --save --nsample 1500000
```
Our result for FiD50k is `18.85919`.

### command for calculating fid50k based on uniform distribution dit
```shell
python -m dgm_eval /personal_storage/scout/fid-flaws/data/ILSVRC2012_img_jpeg  /personal_storage/scout/fid-flaws/data/gen_img_dit/ --model inception --metrics fd --save --nsample 1500000
```
Our result for FiD50k is `18.84164`.

### command for calculating fid50k based on uniform distribution stylegan
```shell
python -m dgm_eval /personal_storage/scout/fid-flaws/data/ILSVRC2012_img_jpeg  /personal_storage/scout/fid-flaws/data/gen_img_stylegan_uni/ --model inception --metrics fd --save --nsample 1500000
```
Our result for FiD50k is `2.86858`.

### command for calculating fid50k based on real distribution stylegan
```shell
python -m dgm_eval /personal_storage/scout/fid-flaws/data/ILSVRC2012_img_jpeg  /personal_storage/scout/fid-flaws/data/gen_img_stylegan/ --model inception --metrics fd --save --nsample 1500000
```
Our result for FiD50k is `2.36722`.
