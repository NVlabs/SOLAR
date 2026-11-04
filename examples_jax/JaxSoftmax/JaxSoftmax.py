# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure JAX softmax — no Flax module, tests raw JAX path."""

import jax
import jax.numpy as jnp


def get_model():
    return None


def forward(x):
    return jax.nn.softmax(x, axis=-1)


batch_size = 1024
num_classes = 32768


def get_inputs():
    return [jnp.ones((batch_size, num_classes))]
