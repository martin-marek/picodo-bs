import jax
import numpy as np
import jax.numpy as jnp


def load_ds(ds_path, seq_len, bs_train, bs_valid, n_tokens_valid, n_tokens_train=None, seed=0):

    # read dataset
    data = np.memmap(ds_path, dtype=np.uint16, mode='r')
    n_tokens_dataset = len(data)

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
    n_seq_dataset = n_tokens_dataset // seq_len
    seq_idx = rng.choice(n_seq_dataset, size=(n_seq_train+n_seq_valid), replace=False)
    seq_idx_train = seq_idx[:n_seq_train].reshape([n_batch_train, bs_train])
    seq_idx_valid = seq_idx[:n_seq_valid].reshape([n_batch_valid, bs_valid])

    # define sequence loader
    def get_batch(seq_idxs): # [B]

        # read dataset
        # using np.memmap for each batch to avoid memory leak
        data = np.memmap(ds_path, dtype=np.uint16, mode='r')

        # get batch
        token_idxs = seq_len*seq_idxs[:, None] + np.arange(seq_len)[None, :] # [B, L]
        batch = data[token_idxs] # [B, L]

        return batch

    return get_batch, seq_idx_train, seq_idx_valid


def get_in_out(batch: jax.Array, pad_id: int = 0):
  """Returns input, output, and weights for a batch of examples."""
  # Assumes input of the form <BOS> <IDs> <EOS> for eval.
  x = batch # [B, L]
  y = jnp.pad(x[:, 1:], ((0, 0), (0, 1)), constant_values=pad_id) # shift x by 1 along L axis
  weights = jnp.where(y != pad_id, 1, 0).astype(jnp.float32)
  return x, y, weights
