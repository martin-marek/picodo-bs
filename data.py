import os
import jax
import numpy as np
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh, NamedSharding


def load_ds(key, ds_path, seq_len, bs_train, bs_valid, n_tokens_valid, n_tokens_train=None, shard=False, mesh=None):

    # get dataset size
    print('getting dataset size...')
    ds_path = os.path.expanduser(ds_path)
    data = np.memmap(ds_path, dtype=np.uint16, mode='r')
    n_tokens_dataset = len(data)
    n_seq_dataset = n_tokens_dataset // seq_len

    # if n_tokens_train is None, use full dataset
    if n_tokens_train is not None: assert n_tokens_train + n_tokens_valid <= n_tokens_dataset
    if n_tokens_train is None: n_tokens_train = n_tokens_dataset - n_tokens_valid

    # get num. of train and valid sequences
    n_batch_train = n_tokens_train // (bs_train * seq_len)
    n_batch_valid = n_tokens_valid // (bs_valid * seq_len)
    n_seq_train = n_batch_train * bs_train
    n_seq_valid = n_batch_valid * bs_valid
    n_token_read = (n_seq_train+n_seq_valid) * seq_len

    # read data
    print('reading data...')
    data = np.memmap(ds_path, dtype=np.uint16, mode='r')
    data = jnp.array(data[:n_token_read])

    # shuffle data
    print('shuffling data...')
    data = data.reshape([n_seq_train+n_seq_valid, seq_len])
    data = jax.random.permutation(key, data, axis=0)

    # split data
    print('splitting data...')
    data_train = data[:n_seq_train].reshape([n_batch_train, bs_train, seq_len])
    data_valid = data[n_seq_train:].reshape([n_batch_valid, bs_train, seq_len])

    # optionally shard dataset across devices
    if shard:
        assert mesh is not None
        with mesh:
            sharding = NamedSharding(mesh, P(None, 'data', None)) # [N, B, L]
            data_train = jax.device_put(data_train, sharding) # [N, B, L]
            data_valid = jax.device_put(data_valid, sharding) # [N, B, L]
    
    return data_train, data_valid


def pad_mask(batch, eos_token_id=1):
    B, L = batch.shape

    # get idx of last EOS token
    # if there is no EOS token, equals L-1
    idx_last_eos_token = (L - 1) - jnp.argmax(batch[:, ::-1] == eos_token_id, axis=1)
    
    # only use tokens before the last EOS token
    mask = jnp.arange(L)[None, :] <= idx_last_eos_token[:, None]

    return mask # [B, L]
