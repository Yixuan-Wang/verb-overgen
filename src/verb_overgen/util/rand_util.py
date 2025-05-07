from collections.abc import Generator
import numpy as np
import numpy._typing as np_typing

def get_u32_rand_gen(seed: int | np_typing._ArrayLikeInt_co | None = None) -> Generator[np.uint32, None, None]: # type: ignore
    """
    Create a generator that yields random 32-bit unsigned integers.
    The generator is seeded with the given seed, or a random seed if none is provided.

    Args:
        seed (int | None): The seed for the random number generator. If None, a random seed is used.
    """
    rng = np.random.default_rng(seed)
    while True:
        yield rng.integers(0, 2**32 - 1, dtype=np.uint32)
