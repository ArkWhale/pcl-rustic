# RFC-0015: Open3D Comparison Benchmark Charts

- **Status:** Implemented (external evidence open)
- **Date:** 2026-05-08
- **Author:** Codex
- **Related:** RFC-0001, RFC-0005, RFC-0006, RFC-0007, RFC-0009, RFC-0011

## 1. Summary

Add a pytest-benchmark based comparison suite that measures pcl-rustic and
Open3D behavior across increasing point-cloud scales, then renders interactive
Plotly charts from the recorded benchmark artifacts.

The measurement layer and visualization layer are separate:

- pytest-benchmark owns benchmark execution, timing, calibration, warmup, and
  machine-readable result output.
- Plotly owns chart rendering after the run. It must not run benchmarks or
  invent measurements.

This RFC extends RFC-0009. RFC-0009 measures pcl-rustic large-scale batch
workloads; RFC-0015 adds Open3D comparison coverage and visual reporting for
migration decisions.

## 2. Motivation

pcl-rustic is positioned as an Open3D replacement for production point-cloud
workloads. Raw benchmark rows are necessary, but they are not enough for users
to understand scaling behavior. Users need to see:

1. How runtime changes as point counts increase.
2. Whether throughput is stable, improves, or collapses at larger scales.
3. Where pcl-rustic is faster or slower than Open3D.
4. Which functions are not directly comparable because the APIs or data models
   differ.

The suite should make these tradeoffs visible without weakening RFC-0011:
charts are evidence only when they cite recorded pytest-benchmark artifacts,
hardware details, dataset parameters, git SHA, dependency versions, and the
command used to generate them.

## 3. Detailed Design

### 3.1 Scope

The comparison matrix covers public operations that have a defensible Open3D
equivalent:

| Family | pcl-rustic operation | Open3D comparison |
|---|---|---|
| Construction | `PointCloud.from_numpy(...)` | `open3d.geometry.PointCloud` from NumPy vectors |
| Transform | `transform`, `translate`, `scale`, `rotate` | Open3D point-cloud transform wrappers |
| Downsample | `voxel_downsample(...)` | `voxel_down_sample(...)` |
| Neighbors | `knn`, `radius_search` | `KDTreeFlann.search_knn_vector_3d`, `search_radius_vector_3d` |
| Normals | `estimate_normals(...)` | `estimate_normals(...)` |
| Outliers | `remove_statistical_outlier`, `remove_radius_outlier` | Open3D SOR/ROR methods |
| Registration | point-to-point and point-to-plane ICP | Open3D ICP with matching estimators |

Operations without a stable one-to-one Open3D equivalent are excluded from
comparative speedup charts and listed in the report as `pcl_rustic_only` or
`not_comparable`. Examples include typed LAS ExtraBytes preservation,
point-format-10 waveform policy, strict/union/intersection typed-attribute
concat, and GPU device-residency behavior.

### 3.2 Scale Matrix

The default comparison suite uses deterministic synthetic point clouds and runs
in three modes:

| Mode | Point counts | Purpose |
|---|---|---|
| smoke | 1k, 10k | PR-safe validation that benchmarks and charts execute |
| standard | 10k, 100k, 1M | routine comparison on developer or CI benchmark machines |
| full | 10k, 100k, 1M, 10M | release comparison on recorded benchmark hardware |

Each case records the point count, random seed, spatial distribution, attribute
schema, operation parameters, and whether Open3D received equivalent data or an
XYZ-only projection.

### 3.3 Correctness And Comparability Contract

Speedup charts are valid only for benchmark pairs that pass a pre-benchmark
comparability check. Each operation family defines the parameters, output
equivalence rule, and tolerance used before timing results are admitted into the
comparison report.

Required equivalence checks:

| Family | Equivalence rule |
|---|---|
| Construction | Point count and XYZ arrays match the deterministic input within `1e-6` after Open3D conversion. |
| Transform | Output XYZ arrays match within `1e-5` for the same transform matrix or wrapper parameters. |
| Downsample | Output point counts and bounding boxes are recorded; exact XYZ equality is not required because voxel representatives differ by implementation. |
| Neighbors | kNN and radius results match brute-force NumPy or have documented tie-handling differences for equal distances. |
| Normals | Normal vectors are unit length and align with the known synthetic plane/shape normals up to sign. |
| Outliers | Kept/removed mask counts and known injected outlier classifications match the synthetic oracle. |
| Registration | Final transform, `fitness`, and `inlier_rmse` are compared against the known synthetic transform or an Open3D fixture tolerance. |

If an operation cannot satisfy a fair equivalence rule, the renderer labels it
`not_comparable` and excludes it from speedup-ratio charts. It may still appear
in a pcl-rustic-only behavior chart when useful.

### 3.4 Benchmark Execution

The benchmark tests live in `tests/test_open3d_benchmark.py` and use
pytest-benchmark fixtures directly. They must not use the RFC-0009 custom
`measure(...)` helper for wall-clock timing.

Required command shape:

```bash
uv run pytest tests/test_open3d_benchmark.py -v --run-slow \
  --benchmark-mode=smoke \
  --benchmark-json=reports/benchmarks/open3d-comparison-smoke.json \
  --no-cov
```

`just` recipes provide the supported entry points:

```bash
just benchmark-compare-smoke
just benchmark-compare-standard
just benchmark-compare-full
just benchmark-compare-charts
```

The benchmark implementation records additional metadata beside
pytest-benchmark's native fields:

- `library`: `pcl_rustic` or `open3d`
- `operation`
- `case_id`
- `point_count`
- `query_count`
- `voxel_size`
- `neighbors`
- `radius`
- `estimator`
- `output_points`
- `git_sha`
- `python_version`
- `pcl_rustic_version`
- `open3d_version`
- `numpy_version`
- `cpu`
- `gpu`
- `os`

Each benchmark must isolate setup from the timed callable:

- Inputs are generated or copied outside the measured section unless the
  operation being measured is construction.
- Mutating Open3D operations receive a fresh point cloud per measured
  invocation, or the benchmark explicitly measures in-place mutation for both
  libraries.
- pcl-rustic and Open3D neighbor benchmarks record whether index construction is
  included. Cold-index and warm-index timings are separate operations, not mixed
  in one chart series.
- pytest-benchmark groups and names encode `library`, `operation`, mode, scale,
  and cache policy so the JSON artifact is self-describing.

The test module must remain collection-safe when optional dependencies are not
installed. Open3D and pytest-benchmark imports use pytest importorskip or marker
guards, so ordinary `just test` does not fail just because comparison benchmark
dependencies are absent.

### 3.5 Plotly Reporting

Plotly rendering is implemented as a separate tool, for example
`tools/render_open3d_benchmark_charts.py`. The renderer reads one or more
pytest-benchmark JSON files and writes:

- `reports/benchmarks/open3d-comparison.html`
- `reports/benchmarks/open3d-comparison-summary.csv`
- optionally, static image exports when an image backend such as Kaleido is
  installed

Required chart groups:

- Runtime vs point count, faceted by operation.
- Throughput vs point count where throughput is meaningful.
- pcl-rustic/Open3D speedup ratio vs point count.
- Output point count vs input point count for downsampling and outlier removal.
- Memory/RSS chart only if memory metadata is collected by a documented helper.

Plotly charts must include enough hover metadata to trace each point back to a
pytest-benchmark result: benchmark name, mode, git SHA, Open3D version,
pcl-rustic version, hardware label, and JSON artifact path.

Reports under `reports/benchmarks/` are local/generated artifacts. Published
evidence is recorded by either copying curated report outputs into
`docs/performance/` or linking durable CI/release artifacts from
`docs/performance/benchmarks.md` and the RFC-0011 external evidence register.
No chart is a release performance claim until that publication path records the
artifact, command, hardware, dependencies, dataset parameters, date, and git SHA.

### 3.6 Dependency Policy

Dependencies are optional benchmark dependencies, not runtime dependencies:

- `pytest-benchmark` is required for the comparison benchmark group.
- `open3d` is required only when running Open3D comparison benchmarks.
- `plotly` is required only for chart rendering.
- Static chart export dependencies remain optional; HTML output is the required
  baseline.

Normal `uv run pytest tests/test_point_cloud.py` and library imports must not
require Open3D, Plotly, or pytest-benchmark.

### 3.7 Implementation Phases

RFC-0015 should be implemented in phases so the first useful comparison does not
block on the full matrix:

1. Phase 1: dependency group, just recipes, pytest-benchmark JSON generation,
   parser tests, and Plotly HTML rendering for construction, transform, and
   voxel downsample in smoke mode.
2. Phase 2: neighbors, normals, and outlier comparisons with correctness
   oracles and cold/warm cache separation.
3. Phase 3: point-to-point and point-to-plane ICP comparisons with fixture
   tolerances and chart publication docs.
4. Phase 4: standard/full modes and durable artifact publication under
   RFC-0011 evidence rules.

## 4. Acceptance Criteria

- [x] `pyproject.toml` defines benchmark-only dependencies for
      `pytest-benchmark`, `open3d`, and `plotly` without adding them to runtime
      dependencies.
- [x] `tests/test_open3d_benchmark.py` uses pytest-benchmark for timing and
      supports `smoke`, `standard`, and `full` modes.
- [x] Benchmark tests define operation-specific equivalence checks and exclude
      non-equivalent pairs from speedup-ratio charts.
- [x] Timed callables isolate setup, mutation, and cold/warm cache behavior so
      pcl-rustic and Open3D measure comparable work.
- [x] Benchmark tests are collection-safe when optional Open3D, Plotly, or
      pytest-benchmark dependencies are not installed.
- [x] Every comparable operation in §3.1 has both a pcl-rustic benchmark and an
      Open3D benchmark, or a documented `not_comparable` exclusion with a
      technical reason.
- [x] The suite writes pytest-benchmark JSON artifacts under
      `reports/benchmarks/`.
- [x] Plotly chart generation reads pytest-benchmark JSON artifacts and writes
      an interactive HTML report without rerunning benchmarks.
- [x] Charts include runtime scaling and pcl-rustic/Open3D ratio views for every
      comparable operation.
- [x] The report includes dependency versions, hardware metadata, command line,
      benchmark mode, and git SHA.
- [x] `just benchmark-compare-smoke`,
      `just benchmark-compare-standard`, `just benchmark-compare-full`, and
      `just benchmark-compare-charts` exist.
- [x] Smoke comparison benchmarks and chart generation are safe for ordinary CI
      when optional benchmark dependencies are installed.
- [x] Documentation explains how to run the comparison suite and how to
      interpret charts without making unrecorded performance claims.
- [x] Published comparison evidence is linked from `docs/performance/` and the
      RFC-0011 external evidence register, or explicitly remains local-only.

## 5. Risks And Mitigations

| Risk | Mitigation |
|---|---|
| Open3D APIs do not preserve pcl-rustic typed attributes | Compare XYZ-equivalent behavior only and label typed-attribute cases as not comparable. |
| Benchmarks compare unlike work | Require equivalence checks, operation-specific parameters, and setup/cache policy metadata before rendering speedup ratios. |
| pytest-benchmark JSON schema changes | Keep a small parser adapter with tests and fail with a clear unsupported-schema error. |
| Chart code accidentally becomes a benchmark runner | Renderer accepts artifact paths only and never imports pcl-rustic or Open3D for measurements. |
| Full comparison mode is too slow or memory-heavy | Gate full mode behind explicit just recipe and `--run-slow`; run smoke mode in CI. |
| Speedup charts overclaim performance | Apply RFC-0011 evidence rules and display hardware, dependency versions, commands, and artifact paths. |
| Open3D dependency is unavailable on some platforms | Skip comparison benchmarks with an explicit reason when Open3D cannot be imported. |

## 6. Out Of Scope

- Replacing the RFC-0009 large-scale pcl-rustic benchmark suite.
- Fabricating or hand-editing benchmark results.
- Making Open3D, Plotly, or pytest-benchmark runtime dependencies.
- Requiring static image export; interactive HTML is the required chart format.
- Comparing features that do not have an honest Open3D equivalent.

## 7. Open Questions

- Whether memory/RSS should be included in the first implementation or deferred
  until a cross-platform measurement helper is selected.
- Whether full mode should include 50M+ point cases after the initial 10M
  release comparison is stable.
- Whether CI should run only chart parser tests or the full smoke comparison
  when optional Open3D wheels are available.
