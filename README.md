# SOLAR <a href="https://github.com/NVlabs/SOLAR#"><img src="docs/logo/solar_icon.png" alt="SOLAR icon" width="96" align="right" /></a>

**PyTorch Model Analysis Toolkit** — graph extraction, einsum conversion, and hardware-aware SOL performance analysis.

[![Docs](https://img.shields.io/badge/docs-guides-brightgreen.svg)](#documentation) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-%3E%3D3.8-blue.svg)](https://www.python.org/) [![GitHub stars](https://img.shields.io/github/stars/NVlabs/SOLAR?style=social)](https://github.com/NVlabs/SOLAR/stargazers)

**SOLAR** = **Speed of Light Analysis for Runtime**

SOLAR is a toolkit for analyzing PyTorch model graphs, converting them to einsum representations, and performing SOL performance analysis.

It is used by [SOL-ExecBench](https://github.com/nvidia/sol-execbench) for deriving SOL performance metrics that serve as ground-truth references for evaluating LLM-generated GPU kernels.

**Quick links:** [Quickstart](#quickstart) · [Documentation](#documentation) · [License](#license)

**Related:** [SOL-ExecBench GitHub](https://github.com/nvidia/sol-execbench) · [Dataset (HuggingFace)](https://huggingface.co/datasets/nvidia/SOL-ExecBench) · [Website](https://research.nvidia.com/benchmarks/sol-execbench)

<br clear="right" />

## Quickstart

```bash
# From repo root (installs SOLAR + patched torchview)
bash install.sh  # or: bash install_uv.sh && source .venv/bin/activate

cd examples/Attention
bash run_solar.sh
```

Note: SOLAR depends on a **patched `torchview`** for parameter-tensor support. The install scripts apply `patches/torchview-parameter-tensors.patch` by default.

For PDF graph rendering, install Graphviz (`dot`).

## Documentation

- [`docs/USAGE.md`](docs/USAGE.md): End-to-end pipeline + CLI + Python API overview
- [`docs/SOL_GUIDE.md`](docs/SOL_GUIDE.md): SOL (Speed of Light Analysis for Runtime) methodology + metrics
- [`docs/EINSUM_GUIDE.md`](docs/EINSUM_GUIDE.md): Einsum conversion details and conventions
- [`docs/EINSUM_GRAPH_CHECKER_GUIDE.md`](docs/EINSUM_GRAPH_CHECKER_GUIDE.md): Einsum graph checker usage
- [`docs/TESTING_GUIDE.md`](docs/TESTING_GUIDE.md): Testing guide
- [`scripts/README.md`](scripts/README.md): Benchmark runner notes

## Contributing

See [`CONTRIBUTING`](CONTRIBUTING).

## License

Apache 2.0 License
