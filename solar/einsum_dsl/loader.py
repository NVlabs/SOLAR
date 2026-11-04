# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load the *_einsum_lib.yaml produced by toDSL_model."""

from pathlib import Path
import yaml


def load_dsl_yaml(yaml_path: Path) -> dict:
    """Load and return the raw YAML dict from *_einsum_lib.yaml.

    Expected top-level keys:
      tensors: {name: {dims: {...}, dtype: str, role?: str}}
      graph:   {kernel_id: {type, function, inputs: [{tensor, from}], output, call_site_line}}
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if "tensors" not in data or "graph" not in data:
        raise ValueError(f"DSL YAML missing required keys 'tensors' or 'graph': {yaml_path}")

    return data
