import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mackay_it.information.entropy import entropy
from mackay_it.information.estimators import (
    estimate_entropy,
    estimate_entropy_mm,
)


def main():
    key = jax.random.key(42)

    p = 0.99
    sample_sizes = [5, 10, 20, 50, 100, 500, 1000, 10_000]
    num_experiments = 10_000

    true_entropy = entropy(jnp.array([p, 1 - p]))

    mle_biases = []
    mm_biases = []

    for s in sample_sizes:
        key, subkey = jax.random.split(key)

        samples = jax.random.bernoulli(
            subkey, p=p, shape=(num_experiments, s)
        ).astype(jnp.int32)

        mle_estimates = jax.vmap(
            estimate_entropy, in_axes=(0, None)
        )(samples, 2)

        mm_estimates = jax.vmap(
            estimate_entropy_mm, in_axes=(0, None)
        )(samples, 2)

        mle_bias = jnp.mean(mle_estimates) - true_entropy
        mm_bias = jnp.mean(mm_estimates) - true_entropy

        mle_biases.append(mle_bias)
        mm_biases.append(mm_bias)

        print(
            f"N: {s:>5}, "
            f"MLE bias: {mle_bias:.6f}, "
            f"MM bias: {mm_bias:.6f}"
        )

    plt.plot(sample_sizes, mle_biases, "o-", label="MLE")
    plt.plot(sample_sizes, mm_biases, "o-", label="Miller–Madow")

    plt.axhline(0, linestyle="--", label="Zero bias")

    plt.xscale("log")
    plt.xlabel("Sample size")
    plt.ylabel("Bias (bits)")
    plt.title("Comparison of Entropy Estimator Bias")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()