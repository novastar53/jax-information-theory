import jax
import jax.numpy as jnp

from mackay_it.information.entropy import entropy


def estimate_entropy(
    samples: jax.Array,
    num_symbols: int,
) -> jax.Array:
    """
    Estimate Shannon entropy from observed samples.

    Parameters
    ----------
    samples : jax.Array
        Observed integer symbols in the range
        [0, num_symbols - 1].
    num_symbols : int
        Number of possible distinct symbols.

    Returns
    -------
    jax.Array
        Estimated Shannon entropy in bits.
    """
    freqs = jnp.bincount(samples, length=num_symbols)
    N = samples.shape[0]
    probs = freqs / N
    return entropy(probs)
