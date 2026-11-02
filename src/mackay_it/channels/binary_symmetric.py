import jax
import jax.numpy as jnp


def binary_symmetric_channel(
    key: jax.Array,
    bits: jax.Array,
    p: float,
) -> jax.Array:
    """
    Transmit bits through a binary symmetric channel.

    Each bit is independently flipped with
    probability p.

    Parameters
    ----------
    key : jax.Array
        Random number generator key.
    bits : jax.Array
        Array containing binary values (0 or 1).
    p : float
        Probability of a bit flip.

    Returns
    -------
    jax.Array
        Received bits after transmission.
    """
    # Your implementation goes here.
    flipped = jax.random.bernoulli(key, p=p, shape=bits.shape)
    return bits ^ flipped