from pathlib import Path
import pickle
from typing import NamedTuple

import jax
from jax import numpy as jnp
from jaxtyping import Array, Scalar


class ModelWeights(NamedTuple):
    w1: Array
    b1: Array
    w2: Array
    b2: Array


def save_weights(weights: ModelWeights, path: Path):
    with path.open("wb") as f:
        pickle.dump(weights, f)


def load_weights(path: Path) -> ModelWeights:
    with path.open("rb") as f:
        return pickle.load(f)


def model_forward(weights: ModelWeights, inputs: Array) -> Array:
    hidden1 = jax.nn.relu(jnp.matmul(inputs, weights.w1) + weights.b1)
    hidden2 = jnp.matmul(hidden1, weights.w2) + weights.b2
    # return hidden2
    return jax.nn.softmax(hidden2)