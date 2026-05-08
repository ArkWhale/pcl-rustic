# pcl-rustic RFCs

This directory holds the project's design RFCs. File naming is mechanical:

```
rfc-{id}-{YYYY-MM-DD}-{kebab-case-title}.md
```

- `id` is a zero-padded four-digit sequence, issued in order (`0001`, `0002`, …). Never reuse.
- `YYYY-MM-DD` is the draft date (the day the RFC was first written, not the day it was accepted).
- `kebab-case-title` is a short slug — prefer ≤ 6 words.

## Current RFCs

| ID | Title | Status | Related milestone |
|---|---|---|---|
| [RFC-0001](rfc-0001-2026-04-30-pcl-rustic-roadmap.md) | pcl-rustic Roadmap & Open3D Replacement Vision | Active roadmap | — (umbrella) |
| [RFC-0002](rfc-0002-2026-04-30-api-reset-and-typed-attributes.md) | API Reset & Typed Attributes | Partial (external evidence open) | M1 |
| [RFC-0003](rfc-0003-2026-04-30-coord-ops-and-selection.md) | Coordinate Ops & Selection | Implemented | M2 |
| [RFC-0004](rfc-0004-2026-04-30-gpu-hot-path.md) | GPU Hot-Path Rewrite | Implemented (external evidence open) | M3 |
| [RFC-0005](rfc-0005-2026-04-30-knn-and-normals.md) | Neighborhood Infra & Normal Estimation | Implemented (external evidence open) | M4 |
| [RFC-0006](rfc-0006-2026-04-30-outlier-removal.md) | Outlier Removal (SOR & ROR) | Implemented (external evidence open) | M5 |
| [RFC-0007](rfc-0007-2026-04-30-icp-gicp-registration.md) | ICP & GICP Registration | Implemented (external evidence open) | M6 |
| [RFC-0008](rfc-0008-2026-05-04-kdtree-fallback-and-gicp-staging.md) | KD-tree Fallback & GICP Staging | Implemented | M4/M6 implementation detail |
| [RFC-0009](rfc-0009-2026-05-04-large-scale-benchmark-suite.md) | Large-Scale Benchmark Suite | Implemented (external evidence open) | Benchmarking / RFC-0004 |
| [RFC-0010](rfc-0010-2026-05-07-host-typed-attribute-storage.md) | Host Typed Attribute Storage Amendment | Accepted | RFC-0002 amendment |
| [RFC-0011](rfc-0011-2026-05-07-completion-evidence-gates.md) | Completion Evidence Gates | Accepted | RFC tracking process |
| [RFC-0012](rfc-0012-2026-05-07-rfc0004-device-residency-scope.md) | RFC-0004 Device Residency Scope | Accepted | RFC-0004 amendment |
| [RFC-0013](rfc-0013-2026-05-07-las-point-format-10.md) | LAS Point Format 10 Precision Support | Implemented | LAS/LAZ I/O |
| [RFC-0014](rfc-0014-2026-05-08-las-point-format-10-api-policy.md) | LAS Point Format 10 Export API And Payload Policy | Implemented | RFC-0013 policy |
| [RFC-0015](rfc-0015-2026-05-08-open3d-comparison-charts.md) | Open3D Comparison Benchmark Charts | Implemented (external evidence open) | Benchmarking / migration evidence |

## Process

The authoring guide lives in the Multica shared knowledge repo at
`ArkWhale/multica-home:templates/prompts/rfc-authoring.md`. When drafting a new
RFC for pcl-rustic, read that guide first — it pins the section structure, the
acceptance-criteria format, and what belongs in "open questions" vs. "risks".
