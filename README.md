# jax-fid-parallel
Frechet inception distance (FID) evaluation in JAX

This repo is a standalone helper file for doing FID evaluation when training generative models in the JAX ecosystem. 

## Installation
Copy the `fid.py` file into your project directory. Inception-v3 weights will be downloaded automatically on first run.

## Usage
The `fid.py` file implements two core functions, `get_fid_network()` and `fid_from_stats()`. Use the first function to calculate the Inception activations of a batch of images (provided in numpy/jax arrays). The images should be normalized between `[-1, 1]`. You should calculate the activations over a decently sized set, at least 10000. Then, calculate the mean and covariance with `np.mean(a, axis=0)` and `np.cov(a, rowvar=False)`. You can then use `fid_from_stats` to calculate FID between two sets of mean+covariance.

## Sanity Checking
We provide some test scripts that calculates FID from images provided at https://github.com/openai/guided-diffusion/tree/main/evaluations.
First, you will want to download some reference images.
```
wget -P data/ https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
wget -P data/ https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/admnet_imagenet256.npz
wget -P data/ https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/admnet_guided_imagenet256.npz
```
Then, you can run `test_imagenet.py` to generate mean+covariance across the entire dataset. The code requires setting up Imagenet with `tfds` (https://www.tensorflow.org/datasets/catalog/imagenet2012) or you can load Imagenet in a different way.
Afterwards, calculate FID of the generated images using:
```
python test_fid.py --images data/VIRTUAL_imagenet256_labeled.npz
python test_fid.py --images data/admnet_guided_imagenet256.npz
python test_fid.py --images data/admnet_imagenet256.npz
```
You should get the following numbers:
| Data              | FID (ours) | FID (ADM paper) |
| :---------------- | :------: | ----: |
| Imagenet256 Training (10K)        |   7.6874113   | N/A |
| ADM (50K)    |  False   | 19.99 |
| ADM-G (50K)           |   True   | 4.59 |


## History
This repo is an adaption of https://github.com/matthias-wright/jax-fid, which traces its origins from a Pytorch port (https://github.com/mseitzer/pytorch-fid) of the original Tensorflow implementation (https://github.com/bioinf-jku/TTUR)
