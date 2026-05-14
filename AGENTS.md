# AGENTS.md

Guidance for AI coding agents working in this repository. Format follows the
[agents.md](https://agents.md/) convention. Human-facing documentation lives in
`README.md` and `docs/`; this file collects the bits an agent needs to act
safely and effectively without re-deriving them from scratch.

## Project overview

`pcl-rustic` is a high-performance Python point-cloud library built in Rust
with [PyO3](https://pyo3.rs/) bindings, exposed as the `pcl_rustic` package
(native module: `pcl_rustic._core`). Tensor work runs on the
[Burn](https://github.com/tracel-ai/burn) framework so the same code targets
CPU and (optionally) WGPU/CUDA. Public surface area is intentionally
batch-only: there is no single-point API.

- Languages: Rust 1.70+ (edition 2021), Python 3.10–3.13 (plus 3.14t free-threaded).
- Build tooling: [maturin](https://github.com/PyO3/maturin) via
  [uv](https://docs.astral.sh/uv/), orchestrated by `just` recipes.
- License: MIT.

## Repository map

```
src/
  lib.rs                  # PyO3 module entry; wires Rust -> Python types
  registration.rs         # ICP / GICP API
  point_cloud/            # core, typed attributes, selection, transforms,
                          #   downsample, outlier removal
  neighbors/              # KD-tree, octree, normals, covariance
  io/                     # LAS/LAZ + tabular (CSV/Parquet) I/O
  interop/                # NumPy <-> Rust boundary
  utils/                  # error types, Burn tensor helpers
  pcl_rustic/             # Python facade: __init__.py, _core.pyi, py.typed
tests/                    # pytest suite (unit + benchmark harnesses)
tools/                    # benchmark renderers and reporting helpers
docs/                     # MkDocs Material site
docs/plans/               # design RFCs (rfc-NNNN-YYYY-MM-DD-*.md)
examples/                 # runnable Python examples
vendor/cubecl-runtime/    # vendored crate referenced via [patch.crates-io]
reports/                  # benchmark + coverage outputs (generated)
```

Before adding new public Rust modules or Python re-exports, check
`src/pcl_rustic/__init__.py` and `src/pcl_rustic/_core.pyi` — the type stub is
hand-maintained and must stay in sync with the PyO3 bindings.

## Setup

```bash
just install          # uv sync --all-groups + pre-commit install
# or, manually:
uv venv
uv sync --dev
pre-commit install
```

Building the native extension is required before Python tests will import the
library:

```bash
just dev              # debug build (maturin develop)
just build            # release build (maturin develop --release)
```

If `import pcl_rustic` fails with `No module named 'pcl_rustic._core'`, rerun
`just dev` (or `just build`).

## Build, test, and lint commands

Always prefer `just` recipes over invoking the underlying tools directly — they
encode the right flags (e.g. coverage config, benchmark JSON output paths).

| Task                | Command                              |
| ------------------- | ------------------------------------ |
| Format (Rust + Py)  | `just fmt`                           |
| Lint (Rust + Py)    | `just lint`                          |
| Pre-commit (all)    | `just pre-commit`                    |
| Python tests (fast) | `just test`                          |
| Python tests (slow) | `just test-slow`                     |
| Rust tests          | `just test-rust`                     |
| Local CI            | `just ci`                            |
| Build wheel         | `just wheel`                         |
| Release-prep gauntlet | `just release` (fmt + lint + test + build + wheel) |
| Clean               | `just clean`                         |

`pytest.ini` enables coverage and writes `reports/pytest.xml` plus
`reports/coverage/`. Tests marked `slow` are skipped unless `--run-slow` is
passed (handled for you by `just test-slow`).

## Benchmarks

Benchmarks are first-class — public performance numbers may only come from
recorded benchmark artifacts. Use the unified suite:

```bash
just benchmark mode=fast compare=false   # smoke run, pcl-rustic only
just benchmark mode=slow compare=true    # standard run, includes Open3D
just benchmark-visualize                 # render last-benchmark.json -> HTML/CSV
```

All modes write to `reports/benchmarks/last-benchmark.json`; the visualizer
reads the same path. When citing benchmark results, record hardware, dataset,
command, and commit alongside the numbers (see RFC-0011 evidence gates).

## Code style

### Rust
- `cargo fmt` is authoritative; CI runs `cargo clippy -- -D warnings` so treat
  clippy lints as errors locally too.
- Prefer Burn tensor ops over hand-rolled loops on hot paths; fall back to
  `rayon` for CPU-only host-side work.
- Errors flow through `thiserror` types in `src/utils/`. New PyO3-facing errors
  must convert into the existing `PyErr` mapping rather than panicking across
  the FFI boundary.

### Python
- `ruff` (config in `ruff.toml`) handles both lint and format. Line length 88,
  double-quoted strings, Google-style docstrings.
- `__init__.py` is the only place where unused imports (`F401`) are allowed.
- Public Python APIs must have matching entries in `src/pcl_rustic/_core.pyi`.

### General
- Do not add comments that merely restate code, and do not leave `// TODO:`
  notes referencing the current task or PR.
- Do not introduce backwards-compatibility shims for code paths the project
  does not expose; this is a pre-1.0 library and removals are fine when the
  surface is internal.

## Data and API contracts

These are load-bearing invariants — breaking them silently will fail tests
and/or benchmarks downstream:

- **Inputs are `dtype=float32` NumPy arrays.** XYZ is `(N, 3)` float32; typed
  attributes are `(N,)` and preserve their original dtype on the host (see
  RFC-0010).
- **No single-point access.** All operations are batched.
- **GPU device residency contract** (RFC-0004 / RFC-0012): selection,
  concatenation, voxel downsample, and coordinate transforms must return XYZ
  tensors on the *input* device. `typed attributes` stay on the host
  regardless of XYZ device.
- `PointCloud.to("cpu")` / `PointCloud.to("gpu")` are the only sanctioned
  device-move entry points; check `has_wgpu_device()` before assuming GPU
  availability in tests or examples.

## Testing instructions

- Add Python tests next to existing modules in `tests/test_point_cloud.py`
  unless the surface is genuinely new (then create a new `test_<area>.py`).
- Mark long-running cases with `@pytest.mark.slow`; CI's default test job runs
  with `-k "not slow"`.
- For Rust-only logic, add `#[cfg(test)]` modules and run `just test-rust`.
- When changing GPU code paths, exercise both CPU and (where available) WGPU
  via `pc.to("cpu")` / `pc.to("gpu")` in the same test to catch device-leak
  regressions.
- Benchmark-style tests live under `tests/test_*benchmark*.py` and run via the
  `just benchmark` recipe rather than plain `pytest`.

## RFC and planning workflow

Substantive design changes are tracked as RFCs in `docs/plans/` using the
filename pattern `rfc-{NNNN}-{YYYY-MM-DD}-{kebab-title}.md`. The current index
is `docs/plans/README.md`. Before reshaping a public API, the GPU dispatch
layer, the benchmark suite, or the I/O contract, check whether an existing RFC
already covers the area; if not, draft one (the authoring guide lives in the
external Multica knowledge repo, referenced from `docs/plans/README.md`).

Recent RFCs worth knowing about when touching adjacent code:
- RFC-0010: typed attributes stay on host.
- RFC-0011: completion evidence gates (no perf claims without artifacts).
- RFC-0012: device-residency scope for the GPU hot path.
- RFC-0016 / RFC-0018: Burn dispatch backend selection rules.
- RFC-0020 / RFC-0021: neighbor hot-path acceleration and Open3D parity.

## Commit and PR conventions

- Commit subject format: `<type>(<scope>): <subject>` — types are
  `feat | fix | docs | style | refactor | test | chore`. Subjects are written
  in **Chinese**, body and footer optional, no line over 72 chars. Full
  template: `.copilot-commit-message-instructions.md`.
- Run `just fmt && just lint && just test && just pre-commit` before opening a
  PR. The release gauntlet (`just release`) is the strictest local gate.
- CI runs lint, multi-OS / multi-Python tests, and wheel builds on every PR;
  benchmarks and PyPI publish only fire on `v*.*.*` tags.

## Things to avoid

- **Do not commit `_core.abi3.so`, build artifacts, or large benchmark
  payloads.** `dist/`, `target/`, and per-run benchmark JSONs are local-only.
- **Do not edit `vendor/cubecl-runtime/`** unless you are deliberately updating
  the patched crate referenced from `Cargo.toml`'s `[patch.crates-io]`.
- **Do not skip pre-commit hooks** (`--no-verify`) or `cargo clippy` warnings;
  CI re-enforces both.
- **Do not publish performance numbers** sourced from ad-hoc scripts. Use the
  recorded benchmark suite (see RFC-0011).
- **Do not add a single-point API or non-float32 NumPy entry point** without an
  RFC amending the data contract.

## Useful references

- README: `README.md` — user-facing overview, install, quickstart.
- Type stubs: `src/pcl_rustic/_core.pyi` — canonical Python signature surface.
- Online docs: <https://ArkWhale.github.io/pcl-rustic>.
- CI workflows: `.github/workflows/test.yml`, `release.yml`.
