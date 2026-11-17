import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mackay_it.information.entropy import entropy
from mackay_it.information.typicality import sequence_information

key = jax.random.key(42)

p = 0.8
probs = jnp.array([1 - p, p])

sequence_lengths = [10, 100, 1000, 10_000]
num_experiments = 10_000

true_entropy = entropy(probs)

for l in sequence_lengths:
    key, subkey = jax.random.split(key)
    seq = jax.random.bernoulli(subkey, p=p, shape=(num_experiments, l)).astype(jnp.int32)
    total_information = jax.vmap(sequence_information, in_axes=(0, None))(seq, probs)
    avg_information = total_information/l
    mean = jnp.mean(avg_information)
    std = jnp.std(avg_information)
    print(f"N: {l:>5}, Mean: {mean:.6f}, Std: {std:.6f}")
    plt.hist(
        avg_information,
        bins=30,
        density=True,
        histtype="step",
        label=f"N = {l}",
    )

plt.axvline(true_entropy, linestyle="--", label="True entropy")

plt.xlabel("Information content per symbol (bits)")
plt.ylabel("Probability density")
plt.title("Asymptotic Equipartition Property")
plt.legend()
plt.show()



