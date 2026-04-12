# Springsteel Benchmarks

Local benchmark scripts for detecting performance regressions before major
changes. These are **not** run in CI — CI runner variance makes automated
perf regression detection noisy and flaky. Run manually, compare CSVs.

## Scripts

| Script | Covers | Typical runtime |
|--------|--------|-----------------|
| `bench_grids.jl` | `spectralTransform!` / `gridTransform!` at nc ∈ {10, 50} for R, RR, RZ, RL, RLZ | ~30 s |
| `bench_interpolation.jl` | `evaluate_unstructured` and `relocate_grid` for RL/RLZ | ~2 min |

## Usage

```bash
# Run grid transform benchmark, write CSV to bench/baseline/grids_YYYY-MM-DD.csv
julia --project bench/bench_grids.jl

# Run interpolation/relocation benchmark
julia --project bench/bench_interpolation.jl

# Specify custom output path
julia --project bench/bench_grids.jl /tmp/my_run.csv
```

## Workflow before major changes

1. **Run bench against the current main branch:**
   ```bash
   git checkout main
   julia --project bench/bench_grids.jl bench/baseline/grids_pre.csv
   julia --project bench/bench_interpolation.jl bench/baseline/interp_pre.csv
   ```

2. **Make your changes and run again:**
   ```bash
   git checkout my-feature
   julia --project bench/bench_grids.jl bench/baseline/grids_post.csv
   julia --project bench/bench_interpolation.jl bench/baseline/interp_post.csv
   ```

3. **Diff the CSVs** to check for regressions. A >20% slowdown on any row
   or any non-zero allocation regression is a red flag — investigate before
   merging.

## Historical anchors

Prior baseline CSVs are stored in `agent_files/` (gitignored) as reference
points. Current references:

- `agent_files/scaling_post_g8_2026-04-11.csv` — post-G8 (basis template cache) grids
- `agent_files/interp_baseline_2026-04-11.csv` — pre-P2/P3 interpolation (for reference)
