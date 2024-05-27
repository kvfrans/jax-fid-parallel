import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from fid import get_fid_network, fid_from_stats
get_activations = get_fid_network()

def deserialization_fn(data):
    image = data['image']
    min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, (299, 299), method='bilinear', antialias=False)
    return image

batch_size = 2048
split = tfds.split_for_jax_process('train', drop_remainder=True)
dataset = tfds.load('imagenet2012', split=split)
dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=True)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
dataset = tfds.as_numpy(dataset)
dataset = iter(dataset)

activations = []
for batch in dataset:
    print('{:f}'.format((len(activations)*batch_size) / 1_200_000.0))

    # For the last batch, we need to pad with zeros
    if batch.shape[0] < batch_size:
        zeros_added = batch_size - batch.shape[0]
        batch = np.concatenate([batch, np.zeros((batch_size - batch.shape[0], 299, 299, 3))], axis=0)
    else:
        zeros_added = 0

    batch = jnp.array(batch)
    batch = 2 * batch - 1 # Normalize to [-1, 1]
    batch = batch.reshape((len(jax.local_devices()), -1, *batch.shape[1:])) # [devices, batch//devices, etc..]
    preds = get_activations(batch)
    preds = preds.reshape((batch_size, -1))
    preds = np.array(preds)
    if zeros_added > 0:
        preds = preds[:-zeros_added]
    activations.append(preds)
activations = np.concatenate(activations, axis=0)
mu1 = np.mean(activations, axis=0)
sigma1 = np.cov(activations, rowvar=False)
np.savez('data/imagenet256_fidstats_ours.npz', mu=mu1, sigma=sigma1)