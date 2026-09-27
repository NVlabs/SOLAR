# Explicit schedule estimate (experimental)

`solar/perf/schedule_estimate.py` computes a critical path and a deterministic
schedule for a caller supplied task DAG. It is separate from the existing
`unfused`, `fused`, and `fused_prefetched` roofline predictions; those outputs
are unchanged. No operator graph is silently treated as a CUDA kernel graph.

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
