# Explicit schedule estimate (experimental)

`solar/perf/schedule_estimate.py` computes a critical path and a deterministic
schedule for a caller supplied task DAG. It is separate from the existing
`unfused`, `fused`, and `fused_prefetched` roofline predictions; those outputs
are unchanged. No operator graph is silently treated as a CUDA kernel graph.

## Intended use and relation to SOL

This tool evaluates a caller's explicit dependency, stream-order, and exclusive
resource constraints with fixed task durations. It can answer whether a proposed
ordering serializes otherwise independent tasks, where a dependency chain limits
overlap, and when an exclusive resource leaves a ready task waiting. Its scope
is one supplied plan; it does not search for an optimal schedule.

In `solar/perf/perf_model.py`, each existing roofline variant computes total
cycles as the maximum of aggregate tensor-core compute cycles and that variant's
aggregate memory cycles. The variants use different memory-traffic estimates.
With fixed work and fixed traffic, changing task order does not change these SOL
values. An explicit schedule can still take longer because of dependencies or
resource serialization.

For a toy workload with normalized compute cost 3 ns and memory cost 5 ns, the
aggregate roofline is 5 ns for every ordering. Supplying two synthetic tasks of
those durations gives:

| Supplied constraints | Aggregate roofline (ns) | Critical path (ns) | Schedule estimate (ns) |
|---|---:|---:|---:|
| Separate streams and resources | 5 | 5 | 5 |
| One stream, explicit serial order | 5 | 8 | 8 |
| Separate streams, one exclusive resource | 5 | 5 | 8 |

These numbers illustrate the additional constraints; they are not GPU timing
measurements. The supplied durations have no automatic connection to SOLAR's
einsum costs, and the resource model does not establish whether real GPU kernels
can overlap. The schedule estimate is not a tighter hardware SOL bound.

A calibrated duration model, realistic concurrent-resource behavior, and a
schedule search would be separate extensions. They are not prerequisites assumed
by this standalone synthetic-plan evaluator.

Run the included end-to-end example:

```bash
python solar/perf/schedule_estimate.py docs/examples/schedule_plan.json
```

The JSON input requires `schema_version: 1`, `unit: "ns"`,
`node_kind: "synthetic_task"`, `duration_source: "synthetic"`, and
`stream_semantics: "explicit_order_only"`. Each node has a unique `id`, a
nonnegative integer `duration_ns`, one `stream`, and explicit `deps`. Every
node appears once in `stream_order` under its stream. `resources` declares
optional exclusive resource names; a node's `resource` may name one of them.
A missing cost, invalid edge, cycle, or unknown resource is an error.

The critical path uses dependency and stream-order edges. The schedule starts
ready nodes in sorted ID order, allowing overlap only when their declared
exclusive resources differ. Durations are inputs, not measurements or
predictions of GPU contention. PDL, capacity fractions, occupancy, implicit
CUDA synchronization, and mixed-precision cost derivation are unsupported.

## Synthetic timing comparison

| Fixture | Serial sum (ns) | Schedule estimate (ns) | Ratio |
|---|---:|---:|---:|
| Serial 2→3→5 | 10 | 10 | 1.00× |
| Independent 3 and 5, separate resources | 8 | 5 | 1.60× |
| Independent 3 and 5, shared exclusive resource | 8 | 8 | 1.00× |
| Fork-join 2→(3,5)→1 | 11 | 8 | 1.38× |

These are hand-checkable simulated durations. They are not measured GPU
speedups. CPU fixture tests: `python -m pytest tests/test_schedule_estimate.py`.
