# RFC-0004: GPU Hot-Path Rewrite (M3)

- **Status:** Implemented (external evidence open, amended by RFC-0010/RFC-0012/RFC-0016)
- **Date:** 2026-04-30
- **Author:** Master PM (agent)
- **Tracking issue:** LEO-36 (parent), per-milestone LEO issue TBD
- **Related:** RFC-0002 (prerequisite: typed attrs), RFC-0003 (prerequisite: `select(mask)` primitive), RFC-0010 (host typed attribute storage amendment), RFC-0012 (device residency scope amendment), RFC-0016 (Burn Dispatch backend amendment)

## 1. Summary

Close LEO-36 priority #1 by making Burn-backed GPU tensors carry real load.
RFC-0016 amends the backend adapter from `Router<(Wgpu, NdArray)>` to Dispatch.
Today only `transform()` runs on the tensor device; voxel binning and XYZ
gather/concatenate still round-trip through host data. M3 rewrites XYZ-heavy
voxel downsample as tensor-native work where practical and publishes a
GPU-vs-CPU benchmark matrix that replaces the current CPU-only README table.
RFC-0010 keeps typed attributes host-side, so this RFC must not require all
attribute gather/cat paths to remain on device.

## 2. Motivation

The LEO-36 audit showed the benchmark table in README (1.3–1.5 M pts/s) is CPU — the GPU promise is unredeemed. Priority #1 is explicitly the top priority. RFC-0010 narrows the device-residency contract to XYZ tensors for now; typed attribute masks/gather remain host-side until a future RFC proves exact multi-dtype device storage.

## 3. Detailed design

### 3.1 Voxel binning as a tensor kernel

Current CPU algorithm (`src/point_cloud/voxel.rs`): `HashMap<String, Vec<usize>>` keyed on `format!("{x}_{y}_{z}")` voxel coords. Unfit for GPU.

Replacement (pure tensor ops):

```text
1. voxel_idx = (xyz / voxel_size).floor().to_i64()          // Tensor2<i64, [N,3]>
2. key       = hash3(voxel_idx)                              // Tensor1<i64, [N]>  (64-bit cantor pair or Morton)
3. order     = argsort(key)                                  // Tensor1<i64, [N]>
4. key_sorted = gather(key, order)
5. run_starts = where(key_sorted[1..] != key_sorted[..-1])   // boundary mask → segment offsets
6. per strategy:
   - NEAREST_TO_CENTROID: segment_mean(xyz_sorted) → per-voxel centroid;
                           within-segment argmin‖p - centroid‖ → one index per segment
   - AVERAGE:             segment_mean(xyz_sorted); segment_mean(attrs_sorted) for f32/f64 attrs,
                           segment_mode for integer attrs; emit synthesized points (no original index)
   - RANDOM_SEEDED:       per-segment pick index from a ChaCha stream keyed on (seed, segment_id)
```

All steps are `burn::tensor` ops supported on both backends. The only CPU exception is `segment_mode` (integer attributes), which uses a small host-side pass — negligible since the cardinality after voxelization is ≪ N.

### 3.2 Hash function choice

64-bit Morton encoding over 21-bit-per-axis voxel indices (covers ±10⁶ voxels on each axis at any size). Morton gives spatial locality in the sorted order, which helps the later `segment_*` reductions on GPU (coalesced memory access).

Alternative considered: 3-tuple Cantor pairing. Rejected — not locality-preserving.

### 3.3 `select(mask)` and `concatenate` device scope

**Amendment:** RFC-0010 supersedes the typed-attribute device-residency
requirement in this section. GPU hot-path work may optimize XYZ selection,
concatenation, and voxel binning while preserving host typed attribute
propagation.

RFC-0003 defines these at the API level. M3 makes them tensor-native:

- `select(mask)` may use Burn `nonzero` + `select_dim(0, indices)` for XYZ.
- `concatenate(&[&Self], policy)` may use Burn `cat(dim=0)` for XYZ.
- Typed attributes follow RFC-0010 host gather/cat semantics and must preserve
  dtype and point order.
- The result's XYZ tensor stays on the source device.

### 3.4 LAS reader — still CPU

`src/io/las_laz.rs` iterates `Reader::points()` point-by-point. Batching is a `las` crate limitation (its zstd decode pipeline is streaming). Leave as-is for M3; revisit only if I/O shows up as the bottleneck post-M3 (unlikely — decode is disk-bound for LAZ).

The reader *does* gain a post-decode push to device: the intermediate `Vec<f32>` is uploaded once via `Tensor2::from_data`, not per-point.

### 3.5 Benchmark harness

Rewrite `tests/test_benchmark.py` as a matrix:

| Axis | Values |
|---|---|
| Backend | `wgpu-vulkan`, `wgpu-metal`, `wgpu-dx12`, `ndarray-cpu` |
| Point count | 1M, 10M, 50M, 200M |
| Voxel size | 0.05, 0.15, 0.50 |
| Strategy | NEAREST_TO_CENTROID, AVERAGE, RANDOM_SEEDED |

Output: a CSV with throughput (pts/s) and wall time per cell, plus a generated Markdown table that replaces the README's current performance table. CI runs a reduced matrix (10M only, two sizes) on every PR; the full matrix runs on release tags and writes to `docs/performance/benchmarks.md`.

### 3.6 Device pinning API

Expose device selection so users can force CPU for reproducibility or force GPU for throughput:

```python
from pcl_rustic import PointCloud, Device

pc = PointCloud.from_las("input.laz", device=Device.GPU)
pc = pc.to(Device.CPU)  # explicit transfer, returns a new cloud on the target device
```

Rust side: add `pub fn to_device(&self, device: BackendDevice) -> Self` that Burn's tensor API already supports.

## 4. Acceptance criteria

- [x] Voxel downsample keeps result XYZ tensors on the source Burn device per RFC-0012; host-side voxel planning and attribute propagation are explicit and dtype-preserving.
- [x] `select(mask)`, `select_indices(indices)`, `concatenate` (from RFC-0003) verified to keep result XYZ tensors on the input/common device via CPU residency tests and a WGPU-gated full pipeline test.
- [ ] Benchmark matrix (§3.5) runs in CI against at least the `wgpu-vulkan` path on Linux and `ndarray-cpu` everywhere.
- [x] README performance claim replaced with benchmark-artifact guidance; generated measured rows remain unpublished until benchmark CSV artifacts exist.
- [ ] The full `load → select_classifications → voxel_downsample → transform → to_las` pipeline on a 50M-point LAZ fixture shows ≥ 3× speedup on GPU vs. CPU on the reference machine.
- [x] Golden tests: CPU and WGPU-gated paths preserve point count, nearest-centroid coordinates within 1e-5 f32 tolerance, dtype-preserving attribute propagation, and same-backend `RANDOM_SEEDED` determinism.
- [x] Documentation: `docs/performance/optimization.md` updated with device-selection guidance and external evidence limits.

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Burn 0.20 `segment_sum` / `segment_mean` missing on WGPU | Spike in first week; if missing, implement via `scatter_add` with a boundary-mask kernel. Document the implementation in this RFC under a §7 update. |
| WGPU driver bugs on Windows DX12 or aarch64 Linux | Keep `ndarray-cpu` fallback first-class; CI auto-skips GPU matrix on runners without a usable adapter. |
| Morton encoding overflow at very large voxel indices | 21-bit axis = 2M voxels per axis, which covers any realistic scan. Guard with a runtime check and fall back to a hash key if a user hits it. |
| GPU memory pressure on 200M-point clouds | Add streaming mode: chunked decode → chunked voxelize with reduce-on-device, concatenate results. Ship with M3 only if a user actually hits OOM; otherwise document as future work. |
| Burn API churn (0.20 → 0.21) during M3 | Pin to 0.20.x in `Cargo.toml`; RFC-0004 amendment required before bumping. |

## 6. Out of scope

- CUDA-specific code paths — WGPU only.
- Multi-GPU or distributed execution.
- Custom kernels via `wgpu_core` — only what Burn exposes.
- GPU neighbor search — moved to RFC-0005 (stretch goal) and RFC-0007 (ICP correspondence).

## 7. Open questions

- Whether to ship both `NEAREST_TO_CENTROID` and `AVERAGE` in the initial M3 or defer `AVERAGE` as a polish item. Proposal: ship both; the segment-mean kernel is shared infrastructure that RFC-0005 / RFC-0007 will reuse, and building it once is cheaper than a second pass later.
- Whether device pinning belongs on `PointCloud` or on a module-level context (à la `torch.device`). Proposal: both — per-cloud `.to(device)` for fine control, module-level `pcl_rustic.set_default_device()` for convenience. Decision lands when the Python sugar is wired.
