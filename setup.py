#!/usr/bin/env python3
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

"""Setup script for the Solar package.

Solar is a comprehensive toolkit for analyzing PyTorch model graphs and 
generating einsum representations for performance analysis.
"""

import setuptools
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="solar",
    version="1.1.0",
    author="Solar Team",
    description="PyTorch model graph analysis and einsum optimization toolkit with LLM support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NVlabs/SOLAR",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchview>=0.2.6",
        "networkx>=3.0",
        "pyyaml>=6.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "llm": [
            "openai>=0.27.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "pylint>=2.17.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "solar-process=solar.cli.process:main",
            "solar-process-model=solar.cli.process_model:main",
            "solar-toeinsum=solar.cli.toeinsum:main",
            "solar-toeinsum-model=solar.cli.toeinsum_model:main",
            "solar-analyze-model=solar.cli.analyze_model:main",
            "solar-predict-perf-model=solar.cli.predict_perf_model:main",
        ],
    },
)