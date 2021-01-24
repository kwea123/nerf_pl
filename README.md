# nerf_pl

### This branch is still under construction. The code is ready to train, I just need more time to run all experiments.

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

### Data perturbations

All random seeds are fixed to reproduce the same perturbations every time.
For detailed implementation, see [blender.py](datasets/blender.py).

*  Color perturbations: Uses the same parameters in the paper.

![color](https://user-images.githubusercontent.com/11364490/105580035-4ad3b780-5dcd-11eb-97cc-4cea3c9743ac.gif)

*  Occlusions: The square has size 200x200 (should be the same as the paper), the position is randomly sampled inside the central 400x400 area; the 10 colors are random.

![occ](https://user-images.githubusercontent.com/11364490/105578658-283da080-5dc5-11eb-9438-9368ee241cde.gif)

*  Combined: First perturb the color then add square.

![combined](https://user-images.githubusercontent.com/11364490/105580018-31cb0680-5dcd-11eb-82bf-eca3133f2586.gif)

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

To train NeRF-U on occluders (Table 3 bottom left).

See [opt.py](opt.py) for all configurations.

You can monitor the training process by `tensorboard --logdir logs/` and go to `localhost:6006` in your browser.

Example training loss evolution (NeRF-U on occluders):

![log](https://user-images.githubusercontent.com/11364490/105621776-a72aeb80-5e4e-11eb-9d12-c8b6f2336d25.png)

## Pretrained models and logs
Download the pretrained models and training logs in [release](https://github.com/kwea123/nerf_pl/releases).

# :mag_right: Testing

Use [eval.py](eval.py) to create the whole sequence of moving views.
E.g.
```
python eval.py \
   --root_dir $BLENDER \
   --dataset_name blender --scene_name lego --split test \
   --img_wh 400 400 --N_importance 64 --ckpt_path $CKPT_PATH
```

It will create folder `results/{dataset_name}/{scene_name}` and run inference on all test data, finally create a gif out of them.

Examples: 

1.  [test_nerfu_occ](test_nerfu_occ.ipynb) shows that NeRF-U is able to decompose the scene into static and transient components when the scene has random occluders. Using [pretrained](https://github.com/kwea123/nerf_pl/releases/tag/nerfu_occ) **NeRF-U** model under **occluder** condition: (PSNR=28.60, paper=23.47)

![nerf-u](https://user-images.githubusercontent.com/11364490/105578186-a9933400-5dc1-11eb-8865-e276b581d8fd.gif)

2.  [test_nerfa_color](test_nerfa_color.ipynb) shows that NeRF-A is able to capture image-dependent color variations. Using [pretrained](https://github.com/kwea123/nerf_pl/releases/tag/nerfa_color) **NeRF-A** model under **color perturbation** condition: (PSNR=28.20, paper=30.66)

![nerfa_color](https://user-images.githubusercontent.com/11364490/105626088-0a2d7a00-5e71-11eb-926d-2f7d18816462.gif)

3.  [test_nerfw_all](test_nerfw_all.ipynb) shows that NeRF-W is able to both handle color variation and decompose the scene into static and transient components.

# :warning: Notes on differences with the paper

*  Network structure ([nerf.py](models/nerf.py)):
    *  My base MLP uses 8 layers of 256 units as the original NeRF, while NeRF-W uses **512** units each.
    *  The static rgb head uses **1** layer as the original NeRF, while NeRF-W uses **4** layers. Empirically I found more layers to overfit when there is data perturbation, as it tends to explain the color change by the view change as well.
    *  I use **softplus** activation for sigma (reason explained [here](https://github.com/bmild/nerf/issues/29#issuecomment-765335765)) while NeRF-W uses **relu**.

*  Training hyperparameters
    *  I find larger (but not too large) `beta_min` achieves better result, so my default `beta_min` is `0.1` instead of `0.03` in the paper.
    *  I add 3 to `beta_loss` (equation 13) to make it positive empirically.
    *  When there is no transient head (NeRF-A), the loss is the average MSE error of coarse and fine models (not specified in the paper).
    *  Other hyperparameters differ quite a lot from the paper (although many are not specified, they say that they use grid search to find the best). Please check each pretrained models in the release.