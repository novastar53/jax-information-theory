import jax
import jax.numpy as jnp

from jax.scipy.stats import binom
from jax.scipy.special import logsumexp

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



def typical_set_statistics(
    n: int,
    p: float,
    epsilon: float,
) -> tuple[jax.Array, jax.Array]:

    probs = jnp.array([1 - p, p])
    true_entropy = entropy(probs)

    k = jnp.arange(n + 1)

    avg_information = -(
        k * jnp.log2(p)
        + (n - k) * jnp.log2(1 - p)
    ) / n

    typical_mask = (
        jnp.abs(avg_information - true_entropy) <= epsilon + 1e-7
    )


    log_fraction_typical = logsumexp(
        jnp.where(
            typical_mask,
            binom.logpmf(k, n, 0.5),
            -jnp.inf,
        )
    )

    log_probability_mass = logsumexp(
        jnp.where(
            typical_mask,
            binom.logpmf(k, n, p),
            -jnp.inf,
        )
    )

    return log_fraction_typical, jnp.exp(log_probability_mass)