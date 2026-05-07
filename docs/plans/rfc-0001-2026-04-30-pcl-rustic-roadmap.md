# RFC-0001: pcl-rustic Roadmap & Open3D Replacement Vision

- **Status:** Active roadmap
- **Date:** 2026-04-30
- **Author:** Master PM (agent)
- **Tracking issue:** LEO-36
- **Supersedes / related:** baseline repository README, `ai_doc/PROJECT_SUMMARY.md`
- **Downstream RFCs:** RFC-0002 through RFC-0007

## 1. Summary

pcl-rustic aims to become a high-performance, GPU-accelerated Python point-cloud library that replaces Open3D for our core workflows: LAS/LAZ ingest, coordinate ops, feature-based selection, voxel downsampling, outlier removal, and ICP-family registration. Today the project ships a working CPU voxel downsampler plus LAS/LAZ/Parquet/CSV I/O on top of a Burn tensor backend that is wired up but barely exploited. This RFC establishes the six-milestone roadmap that drives pcl-rustic from its current narrow baseline to Open3D-parity on the prioritized workflows, while keeping LAS/LAZ as a first-class differentiator.

## 2. Motivation

Open3D covers our use cases but:

1. Open3D does **not** natively support LAS/LAZ — we maintain brittle side converters today.
2. Its Python legacy API is CPU-only; the tensor API has CUDA but no AMD/Intel GPU path and no clean Rust extension story.
3. We need a home where Rust-first batch ops and LAS semantics live together, without C++ build overhead on Linux/macOS/Windows wheels.

pcl-rustic already has the right foundation (PyO3 + Burn `Router<(Wgpu, NdArray)>` + `maturin` wheel CI across Python 3.10–3.14t and all major OS/arch combos). What is missing is feature breadth and honest GPU utilization on the hot paths.

## 3. Scope

### 3.1 In-scope (drives the roadmap)

The nine priorities from LEO-36, restated:

1. GPU acceleration (real, benchmarked, on the hot paths — not just `transform()`)
2. Python data I/O (NumPy zero-copy)
3. Coordinate ops: transform, concatenate, region-level selection (**spatial and feature-based** — feature-based added per LEO-36 follow-up)
4. Single-point editing — ⛔ intentionally out of scope
5. LAS/LAZ I/O (with full attribute fidelity: classification, return number, GPS time, intensity)
6. Downsampling: true random, voxel
7. Outlier removal: SOR, ROR
8. Registration: ICP (point-to-point, point-to-plane), GICP
9. End-to-end example: split → per-slice downsample → concatenate, in both spatial-grid and classification-aware variants

### 3.2 Explicitly deferred

FPS / Poisson-disk sampling, colored ICP, RANSAC/FGR global registration, DBSCAN, plane segmentation, hidden-point removal, convex hull, TSDF / VoxelBlockGrid, visualization / rendering. These are Open3D "nice-to-haves" that block no user today; revisit only when a concrete user need surfaces.

## 4. Milestone map

| RFC | Milestone | Deliverable | Depends on | Est. |
|---|---|---|---|---|
| RFC-0002 | M1 — API reset & typed attributes | `AttributeValue` enum; dtype-preserving NumPy getters; seeded `RANDOM` voxel strategy; repo hygiene | — | 1 sprint |
| RFC-0003 | M2 — Coordinate ops & selection | `concatenate`, `select(mask)`, `select_where`, `select_by_classification`, `crop_aabb/obb`, `translate/rotate/scale` | M1 | 1 sprint |
| RFC-0004 | M3 — GPU hot-path rewrite | Burn-native XYZ-heavy voxel binning, device-aware XYZ gather/scatter, published GPU-vs-CPU benchmarks | M1, M2 | 1.5 sprints |
| RFC-0005 | M4 — kNN infra & normals | KD-tree (CPU first), `knn`, `radius_search`, `estimate_normals` | M1 | 1 sprint |
| RFC-0006 | M5 — Outlier removal | `remove_statistical_outlier`, `remove_radius_outlier` with kept-index mask | M4 | 0.5 sprint |
| RFC-0007 | M6 — ICP + GICP | `registration::icp` pipeline; Point-to-Point, Point-to-Plane, GICP estimators; `RegistrationResult` | M4 | 2 sprints |

Critical path: M1 → M2 → M3 for GPU readiness, and M1 → M4 → M5/M6 for registration readiness. M4 can start in parallel with M3 once M1 lands.

## 5. Cross-cutting architecture choices

### 5.1 Environment & package management (per LEO-36 follow-up)

- **Rust** remains the implementation language (Burn-based, PyO3-bound).
- **Cargo** owns Rust dependencies, build, and publishing of the static library.
- **uv** owns Python environment, dev tooling, and wheel install (`uv sync --dev`, `uv build`).
- **maturin** is the bridge (configured in `pyproject.toml` with `tool.maturin`).

This is **already** the project's setup; RFC-0001 codifies it as a non-negotiable. Any future proposal that introduces pip-tools, poetry, Hatch, setuptools, or cargo-less build paths must be rejected or argued explicitly against this RFC.

### 5.2 Tensor backend

Burn `Router<(Wgpu, NdArray)>` stays. WGPU is our GPU target (portable across NVIDIA/AMD/Intel/Apple Silicon) and ndarray is the CPU fallback. No CUDA-only paths. Backend selection is runtime-auto with explicit `gpu_device()` / `cpu_device()` escape hatches (see `src/utils/tensor.rs`).

### 5.3 Attribute typing

Attributes become typed (RFC-0002). LAS semantics (u8 classification, u8 return number, f64 GPS time, u16 intensity) round-trip losslessly. Coordinates stay f32 unless a downstream RFC proves the need for f64 — user-supplied f64 coords are autocast with a debug log.

### 5.4 Python API shape

Single `PointCloud` class, NumPy-facing, dtype-strict per attribute. No legacy `list[list[float]]` paths (the current code still accepts them — RFC-0002 removes that).

### 5.5 No back-compat constraint

Per LEO-36: forward/backward compatibility is not required. Every RFC may break the public API freely until the first tagged `0.1.x` user surfaces.

## 6. Cross-cutting risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Burn 0.20 → 0.21+ API churn | High | Medium | Pin minor in `Cargo.toml`; RFC-0004 re-evaluates |
| WGPU portability (aarch64 Linux, older Windows drivers) | Medium | Medium | CI matrix already covers; GPU benchmarks in CI, auto-skip on driverless runners |
| Typed-attribute migration breaks existing LAS users | Medium | Low | No user base yet; acceptable per LEO-36 |
| KD-tree GPU implementation complexity delays M5/M6 | Medium | High | M4 ships CPU KD-tree first (`kiddo`); GPU NN is an M6 stretch |
| Voxel downsample GPU accuracy regressions | Low | Medium | RFC-0004 includes golden-test corpus vs. CPU implementation |

## 7. Success criteria

The project is "done" relative to this RFC when:

- All nine LEO-36 priorities have a tested implementation and a section in the user docs.
- A single benchmark run on the reference machine shows the full `load → select_by_classification → transform → voxel_downsample → registration::icp → save` pipeline ≥ 3× faster than the Open3D equivalent on a ≥ 50M-point LAS file.
- End-to-end examples (spatial-grid split and classification-split variants of priority #9) ship in `examples/` and appear in the mkdocs site.
- README's "GPU acceleration" claim is backed by a GPU row in the benchmark table, not just the CPU row.

## 8. Open questions

- **Unit of downsample voxel size.** Meters today; should we expose a strategy for relative sizing (fraction of AABB diagonal)? Defer to RFC-0003.
- **Normal orientation.** RFC-0005 decides whether to ship `orient_normals_*` helpers or leave unoriented.
- **GICP covariance storage.** Persist per-point covariance as a typed attribute bundle, or recompute each call? RFC-0007 decides.

## 9. References

- Open3D PointCloud tutorial: <https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html>
- Open3D ICP tutorial: <https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html>
- Burn book (0.20): <https://burn.dev/books/burn/>
- LAS 1.4 specification (ASPRS): point data record formats 0–10
- LEO-36 issue comments (audit + gap analysis)
