import jax
import jax.numpy as jnp
import numpy as np
import argparse
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
tf.config.set_visible_devices([], 'TPU')


from localutils.debugger import enable_debug
enable_debug()

from fid import get_fid_network, fid_from_stats
get_activations = get_fid_network()

parser = argparse.ArgumentParser()
parser.add_argument('--images', type=str, default='data/admnet_guided_imagenet256.npz')
args = parser.parse_args()

dataset = np.load(args.images)['arr_0']
truth_stats = np.load('data/VIRTUAL_imagenet256_labeled.npz')
mu_truth = truth_stats['mu']
sigma_truth = truth_stats['sigma']

batch_size = 256
activations = []
for i in range(0, len(dataset) // batch_size):
    batch = dataset[i*batch_size:i*batch_size+batch_size]

    print('{:f}'.format((len(activations)*batch_size) / dataset.shape[0]))
    batch = jnp.array(batch)
    batch /= 255.0
    batch = jax.image.resize(batch, (batch_size, 299, 299, 3), method='bilinear', antialias=False)

    batch = 2 * batch - 1 # Normalize to [-1, 1]
    batch = batch.reshape((len(jax.local_devices()), -1, *batch.shape[1:])) # [devices, batch//devices, etc..]
    preds = get_activations(batch)
    preds = preds.reshape((batch_size, -1))
    preds = np.array(preds)
    activations.append(preds)
activations = np.concatenate(activations, axis=0)
mu1 = np.mean(activations, axis=0)
sigma1 = np.cov(activations, rowvar=False)

fid = fid_from_stats(mu1, sigma1, mu_truth, sigma_truth)
print("FID is: ", fid)