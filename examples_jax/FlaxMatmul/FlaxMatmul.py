# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flax Dense layer (matmul + bias) — simplest Flax model."""

import jax
import jax.numpy as jnp
import flax.linen as nn


class Model(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.features)(x)


def get_model():
    return Model(features=4096)


batch_size = 1024
input_dim = 4096


def get_inputs():
    return [jnp.ones((batch_size, input_dim))]
