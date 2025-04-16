import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from omegaconf.dictconfig import DictConfig
from rope import apply_rope


class TransformerDecoder(nnx.Module):
    def __init__(self, c: DictConfig, rngs: nnx.Rngs):
        embed_init = fsdp_init('embedding', c.fsdp_enabled)
        self.in_embed = nnx.Embed(num_embeddings=c.V, features=c.D, embedding_init=embed_init, rngs=rngs)
        self.out_embed = nnx.Embed(num_embeddings=c.V, features=c.D, embedding_init=embed_init, rngs=rngs)
        self.blocks = [TransformerBlock(c, rngs) for _ in range(c.N)]
        self.out_ln = nnx.LayerNorm(c.D, use_bias=False, dtype=c.dtype, rngs=rngs)
        
    def __call__(self, x): # [B, S]
        # token embedding
        h = self.in_embed(x) # [B, L, D]
        
        # transformer blocks
        for block in self.blocks:
            h = block(h)
            
        # project back to vocabulary
        h = self.out_ln(h)
        logits = self.out_embed.attend(h.astype(jnp.float32)) # [B, L, V]
        return logits


class TransformerBlock(nnx.Module):
    def __init__(self, c: DictConfig, rngs: nnx.Rngs):
        self.ln1 = nnx.LayerNorm(c.D, use_bias=False, dtype=c.dtype, rngs=rngs)
        self.attn = MultiHeadAttention(c, rngs=rngs)
        self.ln2 = nnx.LayerNorm(c.D, use_bias=False, dtype=c.dtype, rngs=rngs)
        self.mlp = Mlp(c, rngs)
        
    def __call__(self, x): # [B, L, D]
        x = x + self.attn(self.ln1(x)) # attention block
        return x + self.mlp(self.ln2(x)) # MLP block


class MultiHeadAttention(nnx.Module):
  """Causal attention layer."""
  def __init__(self, c: DictConfig, rngs: nnx.Rngs):
    qkv_proj_init = fsdp_init('attn_qkv_proj', c.fsdp_enabled)
    out_proj_init = fsdp_init('attn_out_proj', c.fsdp_enabled)
    self.head_dim = c.D // c.H
    self.qkv_proj = nnx.LinearGeneral(axis=(-1), in_features=(c.D), out_features=(3, c.H, c.D//c.H), kernel_init=qkv_proj_init, use_bias=False, dtype=c.dtype, rngs=rngs)
    self.out_proj = nnx.LinearGeneral(axis=(-2, -1), in_features=(c.H, c.D//c.H), out_features=(c.D), kernel_init=out_proj_init, use_bias=False, dtype=c.dtype, rngs=rngs)
    self.query_norm = nnx.RMSNorm(self.head_dim, rngs=rngs)
    self.key_norm = nnx.RMSNorm(self.head_dim, rngs=rngs)
    self.query_scaling = (c.D/c.H)**-0.5
    self.dtype = c.dtype

  def __call__(self, x): # [B, L, D]
    B, L, D = x.shape

    # input projection
    qkv = self.qkv_proj(x) # [B, L, 3, H, D/H]
    q, k, v = jnp.moveaxis(qkv, 2, 0) # [B, L, H, D/H]

    # qk-norm
    q = self.query_norm(q)
    k = self.key_norm(k)

    # position embedding
    position = jnp.arange(L)
    q = apply_rope(q, position[None], self.head_dim)
    k = apply_rope(k, position[None], self.head_dim)
    q *= self.query_scaling

    # attention logits
    att = jnp.einsum('...qhd,...khd->...hqk', q, k).astype(jnp.float32) # [B, H, L, L]

    # causal mask
    mask = jnp.tril(jnp.ones((1, 1, L, L), dtype=jnp.bool_))
    _NEG_INF = jnp.finfo(att.dtype).min
    att = jnp.where(mask, att, _NEG_INF)

    # attended values
    att = jax.nn.softmax(att, axis=-1).astype(self.dtype)
    out = jnp.einsum('...hqk,...khd->...qhd', att, v) # [B, L, H, D/H]

    # output projection followed by contraction back to original dims
    out = self.out_proj(out) # [B, L, D]
    return out


class Mlp(nnx.Module):
    """Multilayer perceptron."""
    def __init__(self, c: DictConfig, rngs: nnx.Rngs):
        kernel_init = fsdp_init('mlp_kernel', c.fsdp_enabled)
        self.fc1 = nnx.Linear(in_features=c.D, out_features=c.F, kernel_init=kernel_init, use_bias=False, dtype=c.dtype, rngs=rngs)
        self.fc2 = nnx.Linear(in_features=c.F, out_features=c.D, kernel_init=kernel_init, use_bias=False, dtype=c.dtype, rngs=rngs)
        
    def __call__(self, x): # [B, L, D]
        h = jax.nn.gelu(self.fc1(x)) # [B, L, F]
        return self.fc2(h) # [B, L, D]


def fsdp_init(layer_type: str, fsdp_enabled: bool):
    """Initialize weights with optional FSDP partitioning."""
    partition_fn = nnx.with_partitioning if fsdp_enabled else lambda x, _: x
    kernel_init = jax.nn.initializers.xavier_uniform()
    embed_init = jax.nn.initializers.variance_scaling(1.0, 'fan_in', 'normal', out_axis=0)
    match layer_type:
        case 'embedding': # [V, D]
            return partition_fn(embed_init, (None, 'data'))
        case 'attn_qkv_proj': # [D, 3, H, D/H]
            return partition_fn(kernel_init, ('data', None, None, None))
        case 'attn_out_proj': # [H, D/H, D]
            return partition_fn(kernel_init, (None, None, "data"))
        case 'mlp_kernel': # [D, F]
            return partition_fn(kernel_init, ('data', None))
        case _:
            raise ValueError(f'unrecognized layer type: {layer_type}')


def create_sharded_model(c: DictConfig, mesh: Mesh, seed: int):
    """
    initialize sharded model without putting it on a single device
    https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html
    TODO: add rng key
    """

    @nnx.jit
    def initialize_sharded_model():
        model = TransformerDecoder(c, rngs=nnx.Rngs(seed)) # unsharded at this moment
        state = nnx.state(model) # the model's state, a pure pytree
        pspecs = nnx.get_partition_spec(state) # get annotations from state
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(model, sharded_state) # the model is sharded now
        return model

    with mesh:
        model = initialize_sharded_model()

    return model
