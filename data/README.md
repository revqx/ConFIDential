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