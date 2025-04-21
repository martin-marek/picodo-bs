import os
import jax
import numpy as np
import jax.numpy as jnp


def load_ds(ds_path, seq_len, bs_train, bs_valid, n_tokens_valid, n_tokens_train=None, seed=0):

    # read dataset
    ds_path = os.path.expanduser(ds_path)
    data = np.memmap(ds_path, dtype=np.uint16, mode='r')
    n_tokens_dataset = len(data)
    n_seq_dataset = n_tokens_dataset // (seq_len+1)

    # if n_tokens_train is None, use full dataset
    if n_tokens_train is not None: assert n_tokens_train + n_tokens_valid <= n_tokens_dataset
    if n_tokens_train is None: n_tokens_train = n_tokens_dataset - n_tokens_valid

    # get num. of train and valid sequences
    n_batch_train = n_tokens_train // (bs_train * seq_len)
    n_batch_valid = n_tokens_valid // (bs_valid * seq_len)
    n_seq_train = n_batch_train * bs_train
    n_seq_valid = n_batch_valid * bs_valid

    # sample sequence order
    rng = np.random.default_rng(int(seed))
    seq_idx = rng.choice(n_seq_dataset, size=(n_seq_train+n_seq_valid), replace=False)
    seq_idx_train = seq_idx[:n_seq_train].reshape([n_batch_train, bs_train])
    seq_idx_valid = seq_idx[n_seq_train:].reshape([n_batch_valid, bs_valid])

    # define sequence loader
    # using np.memmap for each batch to avoid memory leak
    def get_batch(seq_idxs): # [B]
        data = np.memmap(ds_path, dtype=np.uint16, shape=[n_seq_dataset, seq_len+1], mode='r')
        batch = data[seq_idxs] # [B, L+1]
        return batch

    return get_batch, seq_idx_train, seq_idx_valid


def pad_mask(batch, eos_token_id=1):
    B, L = batch.shape

    # get idx of last EOS token
    # if there is no EOS token, equals L-1
    idx_last_eos_token = (L - 1) - jnp.argmax(batch[:, ::-1] == eos_token_id, axis=1)
    
    # only use tokens before the last EOS token
    mask = jnp.arange(L)[None, :] <= idx_last_eos_token[:, None]

    return mask # [B, L]
