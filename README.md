# fid-flaws
True-distribution FID evaluation

## For Running commands, check `data` folder's README.md

## Command line guidance
run the following command to generate your `json` files for real image reference statistics
```shell
python imagenet_json.py 
```

run the following command to generate uniformly distributed samples from the generated data
```shell
python sample_image_uni.py --data_folder DATA_FOLDER_PATH --output_folder OUTPUT_FOLDER_PATH
```

run the following command to sample according to the real distribution json file:  \
```shell
python sample_imagenet_json.py
```

## To Do:
- [ ] : add the argparse into those helper function files so we can use it from command line. like `sample_ldm_uni.py`.