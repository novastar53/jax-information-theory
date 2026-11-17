import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mackay_it.information.entropy import entropy
from mackay_it.information.estimators import estimate_entropy


def main():
    key = jax.random.key(42)

    p = 0.8
    sample_sizes = [5, 10, 20, 50, 100, 500, 1000, 10_000]
    num_experiments = 1_000

    true_entropy = entropy(jnp.array([p, 1 - p]))

    mean_entropies = []
    std_entropies = []

    for s in sample_sizes:
        estimates = []
        for _ in range(num_experiments):
            key, subkey = jax.random.split(key)
            samples = jax.random.bernoulli(subkey, p=p, shape=(s,)).astype(jnp.int32)
            est = estimate_entropy(samples, 2)
            estimates.append(est)
        mean = jnp.mean(jnp.array(estimates))
        std = jnp.std(jnp.array(estimates))
        mean_entropies.append(mean)
        std_entropies.append(std)

    mean_entropies = jnp.asarray(mean_entropies)
    std_entropies = jnp.asarray(std_entropies)

    plt.plot(sample_sizes, mean_entropies, "o-", label="Estimated entropy")

    plt.fill_between(
        sample_sizes,
        mean_entropies - std_entropies,
        mean_entropies + std_entropies,
        alpha=0.2,
        label="±1 standard deviation",
    )

    plt.axhline(true_entropy, linestyle="--", label="True entropy")

    plt.xscale("log")
    plt.xlabel("Sample size")
    plt.ylabel("Entropy (bits)")
    plt.title("Entropy Estimation from Finite Samples")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()