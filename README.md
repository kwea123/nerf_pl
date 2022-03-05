# nerf_pl

Unofficial implementation of [NeRF-W](https://nerf-w.github.io/) (NeRF in the wild) using pytorch ([pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)). I try to reproduce (some of) the results on the lego dataset (Section D). Training on [Phototourism real images](https://github.com/ubc-vision/image-matching-benchmark) (as the main content of the paper) has also passed. Please read the following sections for the results.

The code is largely based on NeRF implementation (see master or dev branch), the main difference is the model structure and the rendering process, which can be found in the two files under `models/`.

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

#### Update: There is a [difference](https://github.com/kwea123/nerf_pl/issues/130) between the paper: I didn't add the appearance embedding in the coarse model while it should. Please change [this line](https://github.com/kwea123/nerf_pl/blob/nerfw/models/nerf.py#L65) to `self.encode_appearance = encode_appearance` to align with the paper.

## Blender

<details>
  <summary>Steps</summary>
   
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

</details>

## Phototourism dataset

<details>
  <summary>Steps</summary>

### Data download

Download the scenes you want from [here](https://www.cs.ubc.ca/~kmyi/imw2020/data.html) (train/test splits are only provided for "Brandenburg Gate", "Sacre Coeur" and "Trevi Fountain", if you want to train on other scenes, you need to clean the data (Section C) and split the data by yourself)

Download the train/test split from the "Additional links" [here](https://nerf-w.github.io/) and put under each scene's folder (the **same level** as the "dense" folder)

(Optional but **highly** recommended) Run `python prepare_phototourism.py --root_dir $ROOT_DIR --img_downscale {an integer, e.g. 2 means half the image sizes}` to prepare the training data and save to disk first, if you want to run multiple experiments or run on multiple gpus. This will **largely** reduce the data preparation step before training.

### Data visualization (Optional)

Take a look at [phototourism_visualization.ipynb](https://nbviewer.jupyter.org/github/kwea123/nerf_pl/blob/nerfw/phototourism_visualization.ipynb), a quick visualization of the data: scene geometry, camera poses, rays and bounds, to assure you that my data convertion works correctly.

### Training model

Run (example)

```
python train.py \
  --root_dir /home/ubuntu/data/IMC-PT/brandenburg_gate/ --dataset_name phototourism \
  --img_downscale 8 --use_cache --N_importance 64 --N_samples 64 \
  --encode_a --encode_t --beta_min 0.03 --N_vocab 1500 \
  --num_epochs 20 --batch_size 1024 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name brandenburg_scale8_nerfw
```

`--encode_a` and `--encode_t` options are both required to maximize NeRF-W performance.

`--N_vocab` should be set to an integer larger than the number of images (dependent on different scenes). For example, "brandenburg_gate" has in total 1363 images (under `dense/images/`), so any number larger than 1363 works (no need to set to exactly the same number). **Attention!** If you forget to set this number, or it is set smaller than the number of images, the program will yield `RuntimeError: CUDA error: device-side assert triggered` (which comes from `torch.nn.Embedding`).

</details>

## Pretrained models and logs
Download the pretrained models and training logs in [release](https://github.com/kwea123/nerf_pl/releases).

# :mag_right: Testing

Use [eval.py](eval.py) to create the whole sequence of moving views.
It will create folder `results/{dataset_name}/{scene_name}` and run inference on all test data, finally create a gif out of them.

## Lego from Blender

All my experiments are done with image size 200x200, so theoretically PSNR is expected to be lower.

1.  [test_nerfa_color](test_nerfa_color.ipynb) shows that NeRF-A is able to capture image-dependent color variations.

<p align="center">
  <img src="https://user-images.githubusercontent.com/11364490/105712423-1db60f00-5f5d-11eb-9135-d9df602fa249.gif">
  <img src="https://user-images.githubusercontent.com/11364490/105626088-0a2d7a00-5e71-11eb-926d-2f7d18816462.gif">
  <br>
  Left: NeRF, PSNR=23.17 (paper=23.38). Right: <a href=https://github.com/kwea123/nerf_pl/releases/tag/nerfa_color>pretrained</a> <b>NeRF-A</b>, PSNR=<b>28.20</b> (paper=30.66).
</p>

2.  [test_nerfu_occ](test_nerfu_occ.ipynb) shows that NeRF-U is able to decompose the scene into static and transient components when the scene has random occluders.

<p align="center">
  <img src="https://user-images.githubusercontent.com/11364490/105696553-d4f35b80-5f46-11eb-84f6-2ab0c4f73501.gif">
  <img src="https://user-images.githubusercontent.com/11364490/105578186-a9933400-5dc1-11eb-8865-e276b581d8fd.gif">
  <br>
  Left: NeRF, PSNR=21.94 (paper=19.35). Right: <a href=https://github.com/kwea123/nerf_pl/releases/tag/nerfu_occ>pretrained</a> <b>NeRF-U</b>, PSNR=<b>28.60</b> (paper=23.47).
</p>

3.  [test_nerfw_all](test_nerfw_all.ipynb) shows that NeRF-W is able to both handle color variation and decompose the scene into static and transient components (color variation is not that well learnt though, maybe adding more layers in the static rgb head will help).

<p align="center">
  <img src="https://user-images.githubusercontent.com/11364490/105775080-8d51eb80-5fa9-11eb-9e89-7147c6377453.gif">
  <img src="https://user-images.githubusercontent.com/11364490/105630746-43c0ae00-5e8e-11eb-856a-e6ce7ac8c16f.gif">
  <br>
  Left: NeRF, PSNR=18.83 (paper=15.73). Right: <a href=https://github.com/kwea123/nerf_pl/releases/tag/nerfw_all>pretrained</a> <b>NeRF-W</b>, PSNR=<b>24.86</b> (paper=22.19).
</p>

4. Reference: Original NeRF (without `--encode_a` and `--encode_t`) trained on unperturbed data.

<p align="center">
   <img src="https://user-images.githubusercontent.com/11364490/105649082-0e4dac00-5ef2-11eb-9d56-946e2ac068c4.gif">
   <br>
   PSNR=30.93 (paper=32.89)
</p>

## Brandenburg Gate from Phototourism dataset

See [test_phototourism.ipynb](https://nbviewer.jupyter.org/github/kwea123/nerf_pl/blob/nerfw/test_phototourism.ipynb) for some paper results' reproduction.

Use [eval.py](eval.py) ([example](https://github.com/kwea123/nerf_pl/releases/tag/nerfw_branden)) to create a flythrough video. You might need to design a camera path to make it look more cool!

![brandenburg_test](https://user-images.githubusercontent.com/11364490/107109627-54f1bd80-6885-11eb-9ab1-74a9d66d8942.gif)

# :warning: Notes on differences with the paper

*  Network structure ([nerf.py](models/nerf.py)):
    *  My base MLP uses 8 layers of 256 units as the original NeRF, while NeRF-W uses **512** units each.
    *  The static rgb head uses **1** layer as the original NeRF, while NeRF-W uses **4** layers. Empirically I found more layers to overfit when there is data perturbation, as it tends to explain the color change by the view change as well.
    *  I use **softplus** activation for sigma (reason explained [here](https://github.com/bmild/nerf/issues/29#issuecomment-765335765)) while NeRF-W uses **relu**.
    *  I apply `+beta_min` all the way at the end of compositing all raw betas (see `results['beta']` in [rendering.py](models/rendering.py)). The paper adds `beta_min` to raw betas first then composite them. I think my implementation is the correct way because initially the network outputs low sigmas, in which case the composited beta (if `beta_min` is added first) will be low too. Therefore not only values lower than `beta_min` will be output, but sometimes the composited beta will be *zero* if all sigmas are zeros, which causes problem in loss computation (division by zero). I'm not totally sure about this part, if anyone finds a better implementation please tell me.

*  Training hyperparameters
    *  I find larger (but not too large) `beta_min` achieves better result, so my default `beta_min` is `0.1` instead of `0.03` in the paper.
    *  I add 3 to `beta_loss` (equation 13) to make it positive empirically.
    *  When there is no transient head (NeRF-A), the loss is the average MSE error of coarse and fine models (not specified in the paper).
    *  Other hyperparameters differ quite a lot from the paper (although many are not specified, they say that they use grid search to find the best). Please check each pretrained models in the release.

*  Phototourism evaluation
    *  To evaluate the results on the testing set, they train on the left half of the image and evaluate on the right half (to train the embedding of the test images). I didn't perform this additional training, I only evaluated on the training images. It should be easy to implement this.
