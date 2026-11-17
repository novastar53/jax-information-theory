import jax
import jax.numpy as jnp


def sequence_information(
    sequence: jax.Array,
    probs: jax.Array,
) -> jax.Array:
    """
    Compute the information content of an IID sequence
    in bits.

    Parameters
    ----------
    sequence : jax.Array
        One-dimensional array of integer symbols.
    probs : jax.Array
        Probability distribution over the symbols.

    Returns
    -------
    jax.Array
        Total information content of the sequence.
    """
    return - jnp.sum(jnp.log2(jax.vmap(lambda x: probs[x])(sequence)))