# DLCV Final Project ( 3D ScanNet200 long-tail segmentation )

# How to run your code?

train.txt and test.txt should be placed in the same folder of dataset (i.e. the original dataset structure from CodaLab).

## Train

```
bash scripts/train.sh path_to_dataset
```


## Download checkpoints

```
bash scripts/download.sh
```

## Inference for final submission

```
bash scripts/test.sh path_to_dataset path_to_output_txts
```

It will take about 1~2 hours (test time augmentation & ensemble)

### Two step inference scripts

We also provide two step inference scripts in case that the inference script above does not works for some reasons. The first command is to generate 120 test-time augments, and the second command is to ensemble all the augments.

```
# generate test time augment predictions
bash scripts/tta_all.sh A_only_22000.pth C_inst_48000.pth C_aug3d_59500.pth path_to_aug_predictions path_to_dataset

# ensemble all test time predictions
bash scripts/ensemble.sh path_to_aug_predictions path_to_final_predictions
```

- path_to_dataset: path to scannet dataset
- path_to_aug_predictions: temp folders for test-time augmentation. This is just a temp folder; you can name it whatever you want.
- path_to_final_predictions: final prediction folder

# Checkpoints

[Google drive folder](https://drive.google.com/drive/folders/1UmnU38mBBGaR_6RsI_nS-0f6H1uzIAPf?usp=share_link)

## Full training data

```
# model A mix3d only
gdown "1QMIqGSxYTd0lgETZkKIf6WRcKoW6SWVE&confirm=t" -O A_only_20000.pth
gdown "14mhOFfxiLGcofwfRTvrWbruO11WAaJF3&confirm=t" -O A_only_22000.pth

# model C mix3d + inst
gdown "1rT1KX3g2vKAVEWTWgXYuSzsDOTlmA86K&confirm=t" -O C_inst_45500.pth
gdown "1UNrZINer6r_eIXxBrXLJJhCxguYoJE7h&confirm=t" -O C_inst_48000.pth

# model C mix3d + aug3d
gdown "19n2vpca9v19kzWDKw06R_-5czzgfWI__&confirm=t" -O C_aug3d_45000.pth
gdown "1j1xiVShOtEbUAfJAzfnAEV4bkjOjtKKR&confirm=t" -O C_aug3d_52000.pth
gdown "1WzzeB3R_o5SbqXoRXeQyzgV0c1wk-N9x&confirm=t" -O C_aug3d_59500.pth
```


# Env

## Mix3d

### Minkowski Engine

I failed to install Minkowski Engine using pip. Installing via conda or docker are recommended.

Conda installation example (on my computer), following [Minkowski Engine guide](https://github.com/NVIDIA/MinkowskiEngine#cuda-11x)

1. install openblas-devel
2. install pytorch (1.12.1 and cuda 11.6 is ok)
3. install Minkowski Engine.

### Other packages for Mix3d

pip install the remaining packages from requirement.txt

### Installation example on server

```
conda install openblas-devel -c anaconda

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.X -c pytorch -c conda-forge

# nvcc for conda
# Uncomment the following line if your do not have permission to install cuda on your device
# conda install -c conda-forge cudatoolkit-dev=11.X.X

pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

pip install -r requirement.txt
```

### Note

- If you can install cuda on your device, use your local cuda since It will take a *very very very long time* (about 1 hour) to install `cudatoolkit-dev`

- AttributeError: module 'numpy' has no attribute 'int'
  - This problem is caused by numpy version
  - run `conda install numpy=1.23.5 -c conda-forge` to downgrade numpy version

## LGround

### yml and requirement

```
conda env create -f LanguageGroundedSemseg-master/config/lg_semseg.yml
conda activate lg_semseg
```

then install other packages from requirement.txt

### Minkowski Engine

```
conda install openblas-devel -c anaconda
conda install pytorch=1.12.1 torchvision cudatoolkit=11.6 -c pytorch -c nvidia
# Install MinkowskiEngine

# Uncomment the following line to specify the cuda home. Make sure `$CUDA_HOME/nvcc --version` is 11.X
# export CUDA_HOME=/usr/local/cuda-11.6
export MAX_JOBS=16; pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="-
```



# Preprocess for tail instance augmentation

- utilize spatio_temporal/lib/constants/dataset_sets.py of this branch(LGround) instead of the original one

- run preprocess code

  ```
  python LanguageGroundedSemseg-master/instance_preprocess.py --dataroot /path/to/preprocessed/scannet200
  ```

### Dataset structure

After the ScanNet200 dataset is preprocessed we provide [extracted data files](https://kaldir.vc.in.tum.de/rozenberszki/language_grounded_semseg/feature_data.zip) that we preprocessed for our method. The Zip file with all the necessary content can be downloaded from here and should be placed in the same folder where the processed data files live. Please refer to our paper on how these files were created and what they are used for. So the preprocessed dataset should look something like this:

```
    feature_data/
        |--clip_feats_scannet_200.pkl
        |--dataset_frequencies.pkl
        |--scannet200_category_weights.pkl
        |-- ...
    train/
       |--train_instances/
       |--val_instances/
       |--scene0000_00.ply
       |--scene0000_01.ply
       |--...
    test/
       |--scenexxxx_00.ply
       |--scenexxxx_01.ply
       |--...
    train.txt
    val.txt
    test.txt
```

# Final experiments

## Important flags

- `--merge`: Turn on Mix3d augmentation
- `--data_aug_3d`: Turn on basic random 3D augmentation(rotate, flip and scale)
- `--sample_tail_instances`: Turn on instance augmentation
- `--threads`: num_workers of dataloader. This has very big impact with mix3d augmentation (intense CPU computing)

## Some useful flags

- `start_save_iter`: after `start_save_iter`, every ckpt will be saved separately (ckpt_30, ckpt_60, ...)
- `val_freq`: you can set val_freq to very large if you don’t want to perform validation

## Commands

put all training data in train.txt (i.e. original train.txt)

To perform instance augmentation, first you need to preprocess data following [instance augmentation part](#Instance Augmentation)

```
# 12G-1: modelA / mix3d
export BATCH_SIZE=1; export ITER_SIZE=12; export MODEL=Res16UNet34A;
bash run.sh 0 final "--scannet_path /path/to/preprocessed/scannet200 --merge True --start_save_iter 35000 --save_freq 500 --threads=4 --train_phase=train --val_freq=1000000"

# 12G-2: model A / mix3d / instance augmentation
export BATCH_SIZE=1; export ITER_SIZE=12; export MODEL=Res16UNet34A;
bash run.sh 0 final "--scannet_path /path/to/preprocessed/scannet200 --merge True --sample_tail_instances True --start_save_iter 35000 --save_freq 500 --threads=4 --train_phase=train --val_freq=1000000"

# server-1: model C / mix3d / basic augmentation
export BATCH_SIZE=3; export ITER_SIZE=4; export MODEL=Res16UNet34C;
bash run.sh 0 final_0 "--scannet_path /path/to/preprocessed/scannet200 --merge True --data_aug_3d True --start_save_iter 35000 --save_freq 500 --threads=4 --train_phase=trainval --val_freq=1000000"

# server-2: model C / mix3d / instance augmentation
export BATCH_SIZE=3; export ITER_SIZE=4; export MODEL=Res16UNet34C;
bash run.sh 1 final_1 "--scannet_path /path/to/preprocessed/scannet200 --merge True --sample_tail_instances True --start_save_iter 35000 --save_freq 500 --threads=4 --train_phase=train --val_freq=1000000"
```



# Dataset

## Train val split

I split training and validation data set (15%).

You can put train.txt and val.txt under the same folder of original train.txt and test.txt folder. You may want to rename your original train.txt to keep them.

# Train

```
bash run.sh 0 default "--scannet_path /path/to/preprocessed/scannet200"
```
Parameters can be modified using `export`, editing run.sh or appending other arguments in double quotes as the above example.

GPU memory usage varies for different ply files. Training without OOM in the first 1000 iterations doesn’t mean that OOM will not occur after 1000 iterations. You may want to set `batch_size` to 1(recommended, because it fuses two scenes to generate an augmented scene that will have a large number of points) or 2 and increase `iter_size` to accumulate gradient.

## Important flags

- `--merge`: Turn on Mix3d augmentation
- `--data_aug_3d`: Turn on basic random 3D augmentation(rotate, flip and scale)
- `--sample_tail_instances`: Turn on instance augmentation
  - To perform instance augmentation, first you need to preprocess data following [instance augmentation part](#Preprocess-for-tail-instance-augmentation)

- `--threads`: num_workers of dataloader. This has very big impact with mix3d augmentation (intense CPU computing)

## Some useful flags

- `start_save_iter`: after `start_save_iter`, every ckpt will be saved separately (ckpt_30, ckpt_60, ...)
- `val_freq`: you can set val_freq to very large if you don’t want to perform validation

# Resume

```
bash run.sh 0 default "--scannet_path /path/to/preprocessed/scannet200 --merge True --resume PATH_TO_CHECKPOINT_FOLDER --resume_optimizer"
```

Your config will be replaced by config file in the checkpoint folder. If you want to modify parameters, you can directly modify config.json in the checkpoint folder.

# Inference

```
python3 main.py --weights path_to_weight \
    --scannet_path /path/to/preprocessed/scannet200 \
    --save_pred_dir sample_submission \
    --is_train False \
    --test_original_pointcloud True \
    --dataset Scannet200Voxelization2cmDataset \
    --model Res16UNet34A
```

## Test time augmentation and ensemble

### TTA

```
bash scripts tta.sh start_id end_id weight_fullname pred_root_dir prefix_of_prediction_dir dataset_path model_type do_data_aug_3d do_test_elastic_distortion
```

For example, 

```
bash scripts/tta.sh 11 15 spatio_temporal/outputs/mix3d_Res16UNet34A_iter48000.pth aug_preds aug1 dataset_path Res16UNet34A True False
```

will generate 5 test time augmentation predictions under spatio_temporal/sample_submission/val folder, with 3D basic data augmentation on and test_elastic_distortion off.

### Ensemble

```
python3 postprocessing/ensemble.py --input_dir sample_submission/val --output_dir sample_submission/ensemble
```

The above command will ensemble predictions under `input_dir` via voting.

# Evaluation

(Currently not ignore invalid labels yet)

```
python3 evaluate.py \
    --dataset_dir DATASET_DIR \
    --val_txt VAL_TXT \
    --pred PRED           
```

- `DATASET_DIR`: training and testing dataset root directory to get ground truth labels
- `VAL_TXT`: full path name to validation txt file
- `PRED`: prediction folders which contains sceneXXX.txt or folders of predictions
  - if `PRED` contains several folders of predictions, this program will evaluate all folders
  - if `PRED` contains only txts, this program will run evaluation as previous version


# Info

For more details, please click [this link](https://docs.google.com/presentation/d/1Y-gwBmucYgbWLLk-u6coHi7LybFLXgA9gV8KiOiKShI/edit?usp=sharing) to view the slides of Final Project - ScanNet200. **Note that video and introduction pdf files for final project can be accessed in your NTU COOL.**

# Submission Rules

### Deadline

111/12/29 (Thur.) 23:59 (GMT+8)

# Q&A

If you have any problems related to Final Project, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under Final Project FAQ section in NTU Cool Discussion

# Reference

## Mix3d

- modify from [SpatioTemporalSegmentation-ScanNet](https://github.com/chrischoy/SpatioTemporalSegmentation-ScanNet)
- dataset from [LanguageGroundedSemseg](https://github.com/RozDavid/LanguageGroundedSemseg)
- Since original SpatioTemporal repo is based on MinkowskiEngine 0.4, I update models from LanguageGroundedSemseg as well.

## LanguageGroundedSemseg

modify from [LanguageGroundedSemseg](https://github.com/RozDavid/LanguageGroundedSemseg)
