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
| [RFC-0001](rfc-0001-2026-04-30-pcl-rustic-roadmap.md) | pcl-rustic Roadmap & Open3D Replacement Vision | Proposed | — (umbrella) |
| [RFC-0002](rfc-0002-2026-04-30-api-reset-and-typed-attributes.md) | API Reset & Typed Attributes | Proposed | M1 |
| [RFC-0003](rfc-0003-2026-04-30-coord-ops-and-selection.md) | Coordinate Ops & Selection | Proposed | M2 |
| [RFC-0004](rfc-0004-2026-04-30-gpu-hot-path.md) | GPU Hot-Path Rewrite | Proposed | M3 |
| [RFC-0005](rfc-0005-2026-04-30-knn-and-normals.md) | Neighborhood Infra & Normal Estimation | Proposed | M4 |
| [RFC-0006](rfc-0006-2026-04-30-outlier-removal.md) | Outlier Removal (SOR & ROR) | Proposed | M5 |
| [RFC-0007](rfc-0007-2026-04-30-icp-gicp-registration.md) | ICP & GICP Registration | Proposed | M6 |
| [RFC-0008](rfc-0008-2026-05-04-kdtree-fallback-and-gicp-staging.md) | KD-tree Fallback & GICP Staging | Proposed | M4/M6 implementation detail |

## Process

The authoring guide lives in the Multica shared knowledge repo at
`ArkWhale/multica-home:templates/prompts/rfc-authoring.md`. When drafting a new
RFC for pcl-rustic, read that guide first — it pins the section structure, the
acceptance-criteria format, and what belongs in "open questions" vs. "risks".
