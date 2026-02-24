# SOL (Speed-of-Light) Performance Models Guide

This guide explains the three roofline-based performance models used in Solar for predicting DNN execution time.

## Overview

Solar computes three SOL (Speed-of-Light) performance estimates based on the roofline model. Each model makes different assumptions about memory access patterns and operation fusion:

| Model | Memory Accesses | Roofline Application | Use Case |
|-------|-----------------|---------------------|----------|
| **Unfused** | All tensors (Input + Weight + Output) per op | Per-operation, summed | Baseline / worst case |
| **Fused** | Weights + model I/O only (per op) | Per-operation, summed | Operator fusion |
| **Fused+Prefetched** | Weights + model I/O (total) | Single roofline for entire graph | Best case / perfect overlap |

## Roofline Model Basics

The roofline model predicts performance based on two hardware limits:
- **Compute bound**: Limited by peak compute throughput (MACs/second)
- **Memory bound**: Limited by memory bandwidth (bytes/second)

```
Actual Performance = min(Peak Compute, Arithmetic Intensity × Memory Bandwidth)
```

Where **Arithmetic Intensity** = FLOPs / Memory Bytes

---

## 1. Unfused SOL

### Description
Each operation runs in **isolation**. All tensor accesses (inputs, weights, outputs) are assumed to come from DRAM for every operation.

### Memory Calculation
```
orojenesis_bytes = Σ (Input_bytes + Weight_bytes + Output_bytes) per layer
```

### Roofline Application
- Roofline model applied **per operation**
- Execution times are **summed** across all operations
- Intermediate tensors are read/written to DRAM between operations

### When to Use
- Baseline performance estimate
- No kernel fusion
- Memory-bound workloads with poor data reuse
- Debugging / understanding per-layer bottlenecks

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
Intermediate tensor accesses are **excluded** from memory cost. Only weights and model I/O (global inputs/outputs) are counted. However, the roofline is still applied **per operation**.

### Memory Calculation
```
fused_bytes = Σ (Weight_bytes + Model_IO_bytes) per layer

Where Model_IO_bytes includes:
- Input bytes if layer has no predecessors in graph (model input)
- Output bytes if layer has no successors in graph (model output)
```

### Roofline Application
- Roofline model applied **per operation**
- Execution times are **summed** across all operations
- Intermediate tensors assumed to stay in cache/registers

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
A **single roofline** is applied to the entire graph. Total FLOPs and total memory accesses (weights + model I/O) are aggregated, assuming perfect overlap between compute and memory operations.

### Memory Calculation
```
fused_prefetched_bytes = Total_Weight_bytes + Total_Model_IO_bytes

Where:
- Total_Weight_bytes = Σ weight bytes across all layers
- Total_Model_IO_bytes = Σ model input/output bytes (graph boundaries only)
```

### Roofline Application
- **Single roofline** for entire graph
- Assumes perfect pipelining/prefetching
- Memory latency completely hidden by compute

### When to Use
- Best-case performance estimate
- Highly optimized implementations with prefetching
- Compute-bound workloads
- Upper bound on achievable performance

### Example
For a simple `Linear → ReLU → Linear` network:
```
Total FLOPs = FLOPs(Linear1) + FLOPs(ReLU) + FLOPs(Linear2)
Total Memory = Model_Input + Weight1 + Weight2 + Model_Output
Single roofline applied to (Total FLOPs, Total Memory)
```

---

## Comparison Summary

| Aspect | Unfused | Fused | Fused+Prefetched |
|--------|---------|-------|------------------|
| Intermediate tensors | Counted | Excluded | Excluded |
| Roofline granularity | Per-op | Per-op | Whole graph |
| Memory assumption | All from DRAM | Intermediates cached | Perfect prefetch |
| Typical speedup | 1.0x (baseline) | 1.5-3x | 2-10x |
| Realism | Conservative | Realistic | Optimistic |

## Output Fields

### analysis.yaml
```yaml
total:
  macs: 1000000           # Total multiply-accumulate operations
  flops: 2000000          # Total floating-point operations (2 × MACs)
  orojenesis_bytes: 50000 # Unfused memory bytes
  fused_bytes: 20000      # Fused memory bytes (per-op sum)
  fused_prefetched_bytes: 15000  # Fused+prefetched memory bytes
  weight_bytes: 10000     # Total weight bytes
  model_io_bytes: 5000    # Total model input/output bytes
```

### perf_<arch>.yaml
```yaml
unfused:
  memory_bytes: 50000
  runtime_ms: 0.5
  arithmetic_intensity: 40.0
  bottleneck: memory

fused:
  memory_bytes: 20000
  runtime_ms: 0.2
  arithmetic_intensity: 100.0
  bottleneck: compute

fused_prefetched:
  memory_bytes: 15000
  runtime_ms: 0.15
  arithmetic_intensity: 133.3
  bottleneck: compute

speedup:
  fused_vs_unfused: 2.5
  fused_prefetched_vs_unfused: 3.3
  fused_prefetched_vs_fused: 1.3
```

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
