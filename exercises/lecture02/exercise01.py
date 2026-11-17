import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mackay_it.information.entropy import entropy


def main():
    p = jnp.linspace(0, 1, 101)

    probs = jnp.stack([p, 1 - p], axis=1)
    entropies = jax.vmap(entropy)(probs)

    plt.plot(p, entropies)
    plt.xlabel("Probability of heads")
    plt.ylabel("Entropy (bits)")
    plt.title("Entropy of a Bernoulli Distribution")
    plt.show()


if __name__ == "__main__":
    main()