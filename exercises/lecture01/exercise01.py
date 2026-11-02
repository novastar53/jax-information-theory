import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mackay_it.channels.binary_symmetric import binary_symmetric_channel


def main():
    key = jax.random.key(42)

    num_bits = 10_000
    num_experiments = 1_000
    p = 0.1

    flip_counts = []
    for _ in range(num_experiments):
        signal = jax.random.bernoulli(key, p=0.5, shape=(num_bits,))
        key, subkey = jax.random.split(key)
        received = binary_symmetric_channel(subkey, signal, p)
        n_flipped = jnp.sum(jnp.bool(signal != received))
        flip_counts.append(n_flipped)


    mean_flips = jnp.mean(jnp.asarray(flip_counts))
    std_flips = jnp.std(jnp.asarray(flip_counts))
    print(f"Mean: {mean_flips}, Std: {std_flips}")
    plt.hist(flip_counts)
    plt.show()



if __name__ == "__main__":
    main()