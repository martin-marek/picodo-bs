import jax
import jax.numpy as jnp


def to_bf16(key, x):
    """
    Stochastically round fp32 input to bf16.
    simplified implementation of https://github.com/nestordemeure/jochastic with a major sampling bias fixed
    """
    finfo = jnp.finfo(jnp.bfloat16)

    # round x (assumed to be in fp32) to two closest bf16 values
    # one of these values will be smaller than x, the other large
    # one of these values will be closer to x (default rounding), the other will be farther away
    x_closer = x.astype(jnp.bfloat16)
    error = x - x_closer.astype(jnp.float32)
    direction = jnp.where(error > 0, finfo.max, finfo.min)
    x_farther = jnp.nextafter(x_closer, direction)

    # we randomly round to either the closer or farther value s.t. we get the true value in expectation
    ulp = jnp.abs(x_farther.astype(jnp.float32) - x_closer.astype(jnp.float32))
    rand_unif = jax.random.uniform(key=key, shape=x.shape)
    use_closer = rand_unif * ulp > jnp.abs(error)
    x_stoch = jnp.where(use_closer, x_closer, x_farther)
    
    return x_stoch
