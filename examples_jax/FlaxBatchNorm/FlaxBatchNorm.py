# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flax BatchNorm — tests batch_stats buffer handling."""

import jax
import jax.numpy as jnp
import flax.linen as nn


class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.BatchNorm(use_running_average=True)(x)


def get_model():
    return Model()


batch_size = 32
features = 256
height = 56
width = 56


def get_inputs():
    return [jnp.ones((batch_size, height, width, features))]
