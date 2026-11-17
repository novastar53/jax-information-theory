import jax
import jax.numpy as jnp

from mackay_it.information.entropy import entropy

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



def is_typical(
    sequence: jax.Array,
    probs: jax.Array,
    epsilon: float,
) -> jax.Array:
    """
    Determine whether an IID sequence belongs
    to the weakly typical set.

    Returns a Boolean scalar.
    """
    ent = entropy(probs)
    avg_info = sequence_information(sequence=sequence, probs=probs)/sequence.shape[0]
    return jnp.bool_(jnp.abs(avg_info - ent) <= epsilon)
    