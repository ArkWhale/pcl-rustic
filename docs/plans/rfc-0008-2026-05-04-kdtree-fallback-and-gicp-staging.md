# RFC-0008: KD-tree Fallback & GICP Staging

- **Status:** Proposed
- **Date:** 2026-05-04
- **Author:** Codex implementation agent
- **Related:** RFC-0005, RFC-0007

## 1. Summary

RFC-0005 requires `kiddo` as the CPU KD-tree backend. During implementation,
`kiddo` panicked on degenerate geometries with too many points sharing one axis
bucket. This RFC records the implementation decision: build a `kiddo` index when
the input is suitable, and fall back to deterministic brute-force queries when
the geometry would exceed `kiddo` bucket limits.

RFC-0007 requires GICP API support and covariance storage. This implementation
ships covariance estimation and validates GICP prerequisites, while staging the
GICP iterative update on the same stable point-to-point solve used by the core
ICP loop.

## 2. Motivation

Plane-like, grid-like, or scanline-like point clouds are common in point-cloud
workflows. A neighbor API that panics on those inputs violates RFC-0005's
acceptance criteria that empty or invalid inputs return clean errors and that
normal estimation works on synthetic planes.

GICP has a larger numerical surface area than point-to-point ICP. Staging its
full plane-to-plane solve behind an API-compatible covariance path keeps the
public API aligned with RFC-0007 while preserving a tested registration loop.

## 3. Decision

1. `KdTreeIndex::build` detects more than 32 identical coordinate values on any
   axis and skips `kiddo` for that cloud.
2. `knn` and `radius_search` use brute-force sorted queries when `kiddo` is not
   available for that index.
3. `estimate_covariances(knn)` stores a packed `float32[N, 6]` covariance
   attribute named `covariance`.
4. `TransformationEstimation.generalized(epsilon)` validates source and target
   covariance attributes; the current update step reuses the point-to-point
   estimator until a follow-up RFC specifies the full GICP normal equations and
   test tolerances.

## 4. Acceptance Criteria

- [x] Plane normal estimation no longer panics on degenerate axis-aligned data.
- [x] `knn` and `radius_search` remain deterministic under fallback.
- [x] GICP prerequisite errors are clear when covariance attributes are missing.
- [ ] A follow-up implementation replaces the staged GICP update with a
      covariance-weighted plane-to-plane solve and adds comparison tests.

## 5. Risks

Fallback queries are O(NQ). This is acceptable for degenerate test fixtures and
small planar patches, but very large degenerate clouds may be slower. The user
still gets a correct result instead of a panic.

