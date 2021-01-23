# nerf_pl

### This branch is still under construction. Current progress: successfully reproduced NeRF-U locally.

Unofficial implementation of [NeRF-W](https://nerf-w.github.io/) (NeRF in the wild) using pytorch ([pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)). I tried to reproduce the results on the lego dataset (Section D). Training on real images (as the main content of the paper) cannot be realized since the authors didn't provide the data.

# :computer: Installation

## Hardware

* OS: Ubuntu 18.04
* NVIDIA GPU with **CUDA>=10.2** (tested with 1 RTX2080Ti)

## Software

* Clone this repo by `git clone https://github.com/kwea123/nerf_pl`
* Python>=3.6 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n nerf_pl python=3.6` to create a conda environment and activate it by `conda activate nerf_pl`)
* Python libraries
    * Install core requirements by `pip install -r requirements.txt`
    
# :key: Training

## Blender
   
### Data download

Download `nerf_synthetic.zip` from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

### Training model

Base:
```
python train.py \
   --dataset_name blender \
   --root_dir $BLENDER_DIR \
   --N_importance 64 --img_wh 400 400 --noise_std 0 \
   --num_epochs 20 --batch_size 1024 \
   --optimizer adam --lr 5e-4 --lr_scheduler cosine \
   --exp_name exp
```

Add `--encode_a` for appearance embedding, `--encode_t` for transient embedding.

Add `--data_perturb color occ` to perturb the dataset.

Example:
```
python train.py \
   --dataset_name blender \
   --root_dir $BLENDER_DIR \
   --N_importance 64 --img_wh 400 400 --noise_std 0 \
   --num_epochs 20 --batch_size 1024 \
   --optimizer adam --lr 5e-4 --lr_scheduler cosine \
   --exp_name exp \
   --data_perturb occ \
   --encode_t --beta_min 0.1
```

To train NeRF-U on "occluders" (Table 3).

See [opt.py](opt.py) for all configurations.

You can monitor the training process by `tensorboard --logdir logs/` and go to `localhost:6006` in your browser.

## Pretrained models and logs
Download the pretrained models and training logs in [release](https://github.com/kwea123/nerf_pl/releases).

## Comparison with the paper

# :mag_right: Testing

Example: [test_nerf-u.ipynb](test_nerf-u.ipynb) shows how NeRF-U successfully decomposes the scene into static and transient components.

Use [eval.py](eval.py) to create the whole sequence of moving views.
E.g.
```
python eval.py \
   --root_dir $BLENDER \
   --dataset_name blender --scene_name lego \
   --img_wh 400 400 --N_importance 64 --ckpt_path $CKPT_PATH
```

It will create folder `results/{dataset_name}/{scene_name}` and run inference on all test data, finally create a gif out of them.

Example of lego scene using pretrained **NeRF-U** model under **occluder** condition: (PSNR=28.60, paper=23.47)
![nerf-u](https://user-images.githubusercontent.com/11364490/105578186-a9933400-5dc1-11eb-8865-e276b581d8fd.gif)

# :warning: Notes on differences with the original repo

*  Network structure ([nerf.py](models/nerf.py)):
  *  My base MLP uses 8 layers of 256 units as the original NeRF, while NeRF-W uses **512** units each.
  *  My static head uses 1 layer as the original NeRF, while NeRF-W uses **4** layers.
  *  I use **softplus** activation for sigma (reason explained [here](https://github.com/bmild/nerf/issues/29#issuecomment-765335765)) while NeRF-W uses **relu**.

*  Training hyperparameters
  *  I find larger `beta_min` achieves better result, so my default `beta_min` is `0.1` instead of `0.03` in the paper.
  *  I add 3 to `beta_loss` (equation 13) to make it positive empirically.

*  Evalutaion
  *  The evaluation metric is computed on the **test** set, while NeRF evaluates on val and test combined.