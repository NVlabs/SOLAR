# SOL (Speed-of-Light) Performance Models Guide

This guide explains the three roofline-based performance models used in Solar for predicting DNN execution time.

## Overview

Solar computes three SOL (Speed-of-Light) performance estimates based on the roofline model. Each model makes different assumptions about memory access patterns:

| Model | Memory Accesses | Roofline Application | Use Case |
|-------|-----------------|---------------------|----------|
| **Unfused** | All tensors (Input + Weight + Output) per op | Whole-graph roofline on summed totals | Baseline / worst case |
| **Fused** | Weights + model I/O only (intermediates excluded) | Whole-graph roofline on summed totals | Operator fusion |
| **Fused+Prefetched** | Weights + model I/O only (intermediates excluded) | Whole-graph roofline on summed totals | Best case / perfect overlap |

> **Implementation note**: All three models apply a single whole-graph roofline: `max(total_compute_cycles, total_memory_cycles)`. They differ only in how memory bytes are totaled. In the current code, `fused` and `fused_prefetched` produce identical memory totals and therefore identical runtimes. See [Section 6](#6-known-implementation-gaps) for details.

## Roofline Model Basics

The roofline model predicts performance based on two hardware limits:
- **Compute bound**: Limited by peak compute throughput (MACs/second)
- **Memory bound**: Limited by memory bandwidth (bytes/second)

```
runtime_cycles = max(compute_cycles, memory_cycles)
```

Where:
- `compute_cycles = max(total_macs / MAC_per_cycle_tc, total_other_ops / MAC_per_cycle_sm)`
- `memory_cycles = total_memory_bytes / DRAM_byte_per_cycle`
- **Arithmetic Intensity** = total_macs / total_memory_bytes

---

## 1. Unfused SOL

### Description
All tensor accesses (inputs, weights, outputs) for every operation are assumed to come from DRAM. This produces the highest memory traffic estimate.

### Memory Calculation ([`graph_analyzer.py` L338](../solar/analysis/graph_analyzer.py#L338))
```
Per layer:  unfused_elements_i = input_elems_i + output_elems_i
Total:      unfused_elements   = Σ_i unfused_elements_i
            unfused_bytes      = unfused_elements × bytes_per_element
```

### Roofline Application ([`perf_model.py` L217](../solar/perf/perf_model.py#L217))
```
unfused_total_cycles = max(compute_cycles, unfused_bytes / DRAM_byte_per_cycle)
runtime_ms           = unfused_total_cycles / (freq_GHz × 1e6)
```

A single whole-graph roofline is applied to graph-level totals. Intermediate tensors are counted as DRAM traffic (read by consumer, written by producer).

### When to Use
- Baseline performance estimate
- No kernel fusion
- Memory-bound workloads with poor data reuse
- Debugging / understanding memory bottlenecks

### Example
For a simple `Linear → ReLU → Linear` network:
```
Layer 1 (Linear): Read Input + Weight, Write Output → DRAM
Layer 2 (ReLU):   Read Input (from DRAM), Write Output → DRAM
Layer 3 (Linear): Read Input (from DRAM) + Weight, Write Output → DRAM
```

---

## 2. Fused SOL

### Description
Intermediate tensor accesses are **excluded** from memory cost. Only weights and model-boundary I/O (global inputs/outputs) are counted.

### Memory Calculation ([`graph_analyzer.py` L340–383](../solar/analysis/graph_analyzer.py#L340-L383))
```
Per layer:
  external_input_elems_i = weight_elems + non-intermediate activation elems
  model_output_elems_i   = output_elems if no downstream consumer, else 0
  model_io_elems_i       = external_input_elems_i + model_output_elems_i
  fused_elements_i       = model_io_elems_i

Total:    fused_elements = Σ_i fused_elements_i
          fused_bytes    = fused_elements × bytes_per_element
```

### Roofline Application ([`perf_model.py` L218](../solar/perf/perf_model.py#L218))
```
fused_total_cycles = max(compute_cycles, fused_bytes / DRAM_byte_per_cycle)
runtime_ms         = fused_total_cycles / (freq_GHz × 1e6)
```

A single whole-graph roofline is applied. Intermediate tensors are assumed to stay in cache/registers.

### When to Use
- Operator fusion scenarios (e.g., cuDNN fused kernels)
- When intermediate tensors fit in L2 cache
- Realistic estimate for modern GPU execution

### Example
For a simple `Linear → ReLU → Linear` network:
```
Layer 1 (Linear): Read Model_Input + Weight, intermediate stays in cache
Layer 2 (ReLU):   No DRAM access (intermediate in cache)
Layer 3 (Linear): Read Weight, Write Model_Output
```

---

## 3. Fused+Prefetched SOL

### Description
A **single roofline** is applied to the entire graph. Total compute and total memory accesses (weights + model I/O) are aggregated, assuming perfect overlap between compute and memory operations.

### Memory Calculation ([`graph_analyzer.py` L423–431](../solar/analysis/graph_analyzer.py#L423-L431))
```
fused_prefetched_elements = Σ_i model_io_elems_i
fused_prefetched_bytes    = fused_prefetched_elements × bytes_per_element
```

> **Note**: In the current implementation, `fused_prefetched_elements` is computed identically to `fused_elements` (both sum `model_io_elems` per layer). They produce the same result. See [Section 6](#6-known-implementation-gaps).

### Roofline Application ([`perf_model.py` L219](../solar/perf/perf_model.py#L219))
```
fused_prefetched_total_cycles = max(compute_cycles, fused_prefetched_bytes / DRAM_byte_per_cycle)
runtime_ms                    = fused_prefetched_total_cycles / (freq_GHz × 1e6)
```

### When to Use
- Best-case performance estimate
- Highly optimized implementations with prefetching
- Compute-bound workloads
- Upper bound on achievable performance

### Example
For a simple `Linear → ReLU → Linear` network:
```
Total FLOPs  = FLOPs(Linear1) + FLOPs(ReLU) + FLOPs(Linear2)
Total Memory = Model_Input + Weight1 + Weight2 + Model_Output
Single roofline applied to (Total FLOPs, Total Memory)
```

---

## 4. Comparison Summary

| Aspect | Unfused | Fused | Fused+Prefetched |
|--------|---------|-------|------------------|
| Intermediate tensors | Counted | Excluded | Excluded |
| Roofline granularity | Whole graph | Whole graph | Whole graph |
| Memory assumption | All from DRAM | Intermediates cached | Intermediates cached |
| Typical speedup | 1.0x (baseline) | 1.5-3x | Same as fused (*) |
| Realism | Conservative | Realistic | Optimistic (intended) |

(*) In the current implementation, fused and fused_prefetched produce identical results.

## 5. Output Fields

### analysis.yaml
```yaml
total:
  macs: 1000000                    # Total multiply-accumulate operations
  flops: 2000000                   # Total floating-point operations (2 × MACs)
  other_ops: 50000                 # Total elementwise/reduction ops (CUDA-core work)
  unfused_elements: 25000000       # Unfused memory elements (all tensor I/O)
  fused_elements: 10000000         # Fused memory elements (intermediates excluded)
  fused_prefetched_elements: 10000000  # Fused+prefetched elements (same as fused)
  model_io_elements: 2500000       # Model input/output elements
  intermediate_elements: 15000000  # Intermediate tensor elements
  weight_elements: 7500000         # Weight tensor elements
```

### perf_<arch>.yaml
```yaml
unfused:
  memory_bytes: 50000000
  compute_cycles: 2645
  memory_cycles: 49049
  total_cycles: 49049
  runtime_ms: 0.025
  arithmetic_intensity: 0.04
  bottleneck: memory

fused:
  memory_bytes: 20000000
  compute_cycles: 2645
  memory_cycles: 19619
  total_cycles: 19619
  runtime_ms: 0.010
  arithmetic_intensity: 0.1
  bottleneck: memory

fused_prefetched:
  memory_bytes: 20000000
  compute_cycles: 2645
  memory_cycles: 19619
  total_cycles: 19619
  runtime_ms: 0.010
  arithmetic_intensity: 0.1
  bottleneck: memory

speedup:
  fused_vs_unfused: 2.5
  fused_prefetched_vs_unfused: 2.5
  fused_prefetched_vs_fused: 1.0
```

## 6. Known Implementation Gaps

The current implementation has two discrepancies from the original design intent:

### Gap 1: All three models use whole-graph roofline

The original design intended unfused and fused to use **per-op roofline sums**:
```
Intended unfused:          Σ_i max(compute_i, memory_i)   (per-op, summed)
Intended fused:            Σ_i max(compute_i, fused_memory_i)
Intended fused_prefetched: max(Σ compute, Σ fused_memory)  (whole-graph)
```

The actual implementation uses whole-graph roofline for all three:
```
Actual unfused:            max(Σ compute, Σ unfused_memory)
Actual fused:              max(Σ compute, Σ fused_memory)
Actual fused_prefetched:   max(Σ compute, Σ fused_memory)
```

Per-op-sum is always >= whole-graph roofline (`Σ max(c_i, m_i) >= max(Σ c_i, Σ m_i)`), so the current unfused/fused estimates are more optimistic than intended.

### Gap 2: Fused and fused_prefetched are identical

Both compute `Σ model_io_elems` per layer:
- `fused_elements`: accumulated via per-layer sum ([line 420](../solar/analysis/graph_analyzer.py#L420))
- `fused_prefetched_elements`: accumulated via separate sum over the same field ([lines 427–431](../solar/analysis/graph_analyzer.py#L427-L431))

These always produce the same total, so fused and fused_prefetched runtime/cycles are identical.

For discussion of multi-stream concurrency and PDL modeling implications, see [SOL_CONCURRENCY_AND_PDL.md](./SOL_CONCURRENCY_AND_PDL.md).

## Practical Guidance

1. **Use Unfused** when:
   - Evaluating baseline performance
   - No fusion optimizations available
   - Debugging memory bottlenecks

2. **Use Fused** when:
   - Estimating performance with standard fusion (cuDNN, etc.)
   - Intermediate tensors fit in L2 cache
   - Most realistic estimate for modern GPUs

3. **Use Fused+Prefetched** when:
   - Estimating best-case performance
   - Highly optimized custom kernels
   - Setting performance targets

## References

- Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: An insightful visual performance model for multicore architectures.
- NVIDIA cuDNN documentation on operator fusion
- PyTorch compile / TorchInductor fusion strategies
