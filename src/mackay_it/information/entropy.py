import jax
import jax.numpy as jnp


def entropy(probs: jax.Array) -> jax.Array:
    """
    Compute the Shannon entropy of a discrete
    probability distribution in bits.

    Parameters
    ----------
    probs : jax.Array
        One-dimensional array of probabilities.

    Returns
    -------
    jax.Array
        Shannon entropy in bits.
    """
    safe_probs = jnp.where(probs > 0, probs, 1.0)
    information = - jnp.log2(safe_probs)
    entropy = jnp.sum(probs * information)
    return entropy