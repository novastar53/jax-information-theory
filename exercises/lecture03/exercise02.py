from itertools import product

import jax
import jax.numpy as jnp

from mackay_it.information.typicality import is_typical, sequence_information

p = 0.8
probs = jnp.array([1 - p, p])
epsilon = 0.1

sequence_lengths = [5, 10, 15, 20]

for l in sequence_lengths:
    sequences = jnp.array(list(product([0, 1], repeat=l)))
    typical = jax.vmap(is_typical, (0, None, None))(sequences, probs, epsilon)
    frac_typical = jnp.sum(typical)/typical.shape[0]
    seq_info = jax.vmap(sequence_information, (0, None))(sequences, probs)
    seq_probs = jax.vmap(lambda x: 2** (-x))(seq_info)
    typical_prob = jnp.sum(jnp.where(typical, seq_probs, 0.0))
    print(f"N = {l}, fraction typical = {frac_typical}, prob mass = {typical_prob}")