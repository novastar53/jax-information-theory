import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from scipy.stats import binom

from mackay_it.channels.binary_symmetric import binary_symmetric_channel


def main():
    key = jax.random.key(42)

    num_bits = 10_000
    num_experiments = 100
    p = 0.1
    repeat_counts = [1, 3, 5, 7, 9, 11]

    experimental_errors = []
    theoretical_errors = []

    for n_repeat in repeat_counts:
        flip_counts = []

        for _ in range(num_experiments):
            key, signal_key, channel_key = jax.random.split(key, 3)

            signal = jax.random.bernoulli(
                signal_key, p=0.5, shape=(num_bits,)
            )

            signal_repeated = jnp.repeat(signal, n_repeat)

            received = binary_symmetric_channel(
                channel_key, signal_repeated, p
            )

            received = received.reshape(num_bits, n_repeat)
            received = jnp.sum(received, axis=1) > n_repeat // 2

            n_flipped = jnp.sum(signal != received)
            flip_counts.append(n_flipped)

        mean_flips = jnp.mean(jnp.asarray(flip_counts))
        error_rate = mean_flips / num_bits

        experimental_errors.append(float(error_rate))

        # Probability of more than half the repeated bits being flipped
        theoretical_error = binom.sf(n_repeat // 2, n_repeat, p)
        theoretical_errors.append(theoretical_error)

        print(
            f"Repetitions: {n_repeat}, "
            f"Experimental error: {error_rate:.6f}, "
            f"Theoretical error: {theoretical_error:.6f}"
        )

    plt.plot(
        repeat_counts,
        experimental_errors,
        "o-",
        label="Experimental",
    )

    plt.plot(
        repeat_counts,
        theoretical_errors,
        "s--",
        label="Theoretical",
    )

    plt.yscale("log")
    plt.xlabel("Number of repetitions")
    plt.ylabel("Bit-error rate")
    plt.title("Reliability of Repetition Codes")
    plt.xticks(repeat_counts)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()