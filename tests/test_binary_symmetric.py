import jax
import jax.numpy as jnp

from mackay_information_theory.mackay_it.channels.binary_symmetric import binary_symmetric_channel

key = jax.random.key(42)

bits = jnp.array([0, 1, 1, 0, 1, 0, 0, 1])

received = binary_symmetric_channel(
    key, bits, p=0.2
)

print("Original:", bits)
print("Received:", received)