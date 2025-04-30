"""
Utils for positional embeddings (including RoPE).
https://github.com/google/flax/blob/main/examples/gemma/positional_embeddings.py
"""

import jax
import jax.numpy as jnp

_MAX_WAVELENGTH = 10_000


# Assume _MAX_WAVELENGTH is defined, e.g.:
_MAX_WAVELENGTH = 10000

def apply_rope(
    inputs: jax.Array,  # [B, H, L, Dh] or [B, L, Dh] - Handles both now
    positions: jax.Array,  # Broadcastable to [B, L] e.g., [1, L]
    head_dim: int,
    max_wavelength: int = _MAX_WAVELENGTH,
    scale_factor: float = 1.0,
) -> jax.Array:
  """Applies RoPE accommodating different input ranks (incl. heads dim)."""
  # Input shape could be (B, L, Dh) or (B, H, L, Dh) etc.
  # Positions shape is usually (B, L) or (1, L)

  # These calculations only depend on L and Dh // 2
  fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
  timescale = max_wavelength**fraction # Shape: (Dh // 2,)

  # Ensure positions has at least 2 dims (B, L) for broadcasting below
  if positions.ndim == 1:
      positions = positions[jnp.newaxis, :] # Convert (L,) to (1, L)

  # Calculate sinusoid input:
  # positions[..., jnp.newaxis] -> (B, L, 1) or (1, L, 1)
  # timescale[jnp.newaxis, jnp.newaxis, :] -> (1, 1, Dh // 2)
  # Result shape: (B, L, Dh // 2) or (1, L, Dh // 2) via broadcasting
  sinusoid_inp = (
      positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
  )

  if scale_factor < 1.0:
    raise ValueError(f'scale_factor must be >= 1.0, got {scale_factor}')
  sinusoid_inp /= scale_factor # Shape remains (B or 1, L, Dh // 2)

  # Reshape sin/cos for broadcasting with inputs like (B, H, L, Dh // 2)
  # Add dimensions for H (if present) and potential B mismatch.
  # Target shape: (B or 1, 1, L, Dh // 2) -> allows broadcasting with (B, H, L, Dh // 2)
  num_input_dims = inputs.ndim
  num_pos_dims = sinusoid_inp.ndim # Should be 3 (B or 1, L, Dh//2)

  # Add dummy dims before L for any extra input dims (like H)
  reshape_dims = list(sinusoid_inp.shape[:num_pos_dims-2]) # B or 1
  num_extra_dims = num_input_dims - num_pos_dims - 1 # Account for Dh dim split later
  reshape_dims.extend([1] * num_extra_dims) # Add H dims if needed
  reshape_dims.extend(sinusoid_inp.shape[num_pos_dims-2:]) # L, Dh//2
  # Example: if input=(B,H,L,Dh) and pos=(1,L), sinusoid_inp=(1,L,Dh/2) -> reshape=(1, 1, L, Dh/2)
  # Example: if input=(B,L,Dh) and pos=(1,L), sinusoid_inp=(1,L,Dh/2) -> reshape=(1, L, Dh/2) (no change needed here, but general formula handles it)

  sin = jnp.sin(sinusoid_inp).reshape(reshape_dims)
  cos = jnp.cos(sinusoid_inp).reshape(reshape_dims)
  # sin/cos shapes are now broadcastable with first_half/second_half

  first_half, second_half = jnp.split(inputs, 2, axis=-1)
  # Broadcasting works:
  # (B, H, L, Dh // 2) * (1, 1, L, Dh // 2) -> (B, H, L, Dh // 2)
  # or (B, L, Dh // 2) * (1, L, Dh // 2) -> (B, L, Dh // 2)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin
  out = jnp.concatenate([first_part, second_part], axis=-1)
  return out.astype(inputs.dtype)
