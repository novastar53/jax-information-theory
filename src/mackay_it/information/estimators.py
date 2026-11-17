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


def estimate_entropy_mm(
    samples: jax.Array,
    num_symbols: int,
) -> jax.Array:
    """
    Estimate Shannon entropy using the Miller–Madow
    bias correction.
    """
    mle_entropy = estimate_entropy(samples, num_symbols)

    freqs = jnp.bincount(samples, length=num_symbols)

    K = jnp.sum(freqs > 0)
    N = samples.shape[0]

    correction = (K - 1) / (2 * N * jnp.log(2.0))

    return mle_entropy + correction