import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mackay_it.information.typicality import typical_set_statistics

jax.config.update("jax_enable_x64", True)


def main():
    p = 0.8
    epsilon = 0.1

    sequence_lengths = [
        5, 10, 15, 20, 50, 100,
        200, 500, 1000, 5000, 10_000
    ]

    log_fractions = []
    probability_masses = []

    for n in sequence_lengths:
        log_fraction, mass = typical_set_statistics(n, p, epsilon)

        log_fractions.append(log_fraction)
        probability_masses.append(mass)

    log_fractions = jnp.asarray(log_fractions)
    probability_masses = jnp.asarray(probability_masses)

    # Fraction of typical sequences
    plt.figure()
    plt.plot(sequence_lengths, log_fractions / jnp.log(10), "o-")
    plt.xlabel("Sequence length")
    plt.ylabel("Log10 of fraction of typical sequences")
    plt.title("Fraction of Typical Sequences")
    plt.tight_layout()

    # Probability mass
    plt.figure()
    plt.plot(sequence_lengths, probability_masses, "o-")
    plt.axhline(1, linestyle="--")
    plt.xscale("log")
    plt.xlabel("Sequence length")
    plt.ylabel("Probability mass")
    plt.title("Probability Mass of the Typical Set")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()