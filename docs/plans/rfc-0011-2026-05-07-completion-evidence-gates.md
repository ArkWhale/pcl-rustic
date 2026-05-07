# RFC-0011: Completion Evidence Gates

- **Status:** Accepted
- **Date:** 2026-05-07
- **Author:** Codex
- **Tracking issue:** RFC backlog follow-up
- **Related:** RFC-0001 through RFC-0010

## 1. Summary

Split RFC completion into two explicit tracks:

1. **Repo-local implementation completion** means the required code, API,
   tests, examples, and docs that can be verified in this repository are done.
2. **External evidence completion** means hardware-bound benchmarks,
   cross-platform CI evidence, external workspace documents, or reference
   datasets have been produced and recorded.

An RFC may be marked `Implemented (external evidence open)` when repo-local
implementation criteria are met but external evidence criteria remain open. Use
plain `Implemented` only when no acceptance criterion remains open. Performance
claims remain forbidden until measured CSVs or comparable recorded artifacts
exist.

## 2. Motivation

The current RFC backlog mixes ordinary implementation work with criteria that
depend on resources outside this repository:

- 10M/50M/500k-scale benchmarks on named reference hardware.
- GPU-vs-CPU speedup evidence on a machine with the required WGPU stack.
- Standard/full benchmark runs that require high-memory benchmark hardware.
- Cross-platform CI evidence after hosted runners execute the workflow.
- External Multica workspace documentation outside this repository.

Keeping those items as unchecked RFC blockers causes two problems. First,
implemented code remains marked partial even when no repo-local action remains.
Second, future agents are tempted to mark benchmark or hardware claims complete
without evidence. This RFC makes the distinction explicit.

## 3. Detailed Design

### 3.1 Status Terms

RFC status values may use these suffixes:

- `Implemented` only when all repo-local and external-evidence criteria are
  complete.
- `Implemented (external evidence open)` when repo-local implementation is
  complete and only external evidence remains.
- `Partial` when repo-local implementation or test/doc work remains.
- `Proposed` or `Accepted` for design-only RFCs that have not yet produced an
  implementation target.

### 3.2 Acceptance Criteria Handling

Each RFC acceptance criterion must be classified by where it can be verified:

- **Repo-local:** source changes, unit/integration tests, examples, generated
  docs, and benchmark harness behavior that can run in this checkout.
- **External evidence:** high-memory or GPU benchmark results, cross-platform
  hosted CI completion, external workspace documents, unavailable external
  datasets, and non-reproducible third-party systems.

Dependency installation alone does not make a criterion external. If the test
can be run from committed fixtures plus declared dependencies, it remains
repo-local.

Repo-local criteria should be checked only after fresh verification. External
evidence criteria should remain unchecked until the artifact is recorded, but
they should not prevent marking repo-local implementation as implemented.

### 3.3 Benchmark Claims

Benchmark docs may describe how to run a benchmark suite before recorded
results exist. They must not publish throughput, speedup, or pass/fail claims
for benchmark thresholds until the repository contains or links to a recorded
artifact with enough data to recompute the claim. For RFC-0009 CSV-backed
results, keep the full RFC-0009 CSV row fields. For other benchmark claims,
record at least:

- benchmark mode and case id,
- operation,
- input and output point counts,
- dataset or fixture identity,
- voxel size / strategy / estimator when applicable,
- hardware/backend label,
- git SHA,
- wall time and derived throughput when the claim uses throughput,
- threshold being claimed,
- comparator or baseline row for speedup claims,
- date of run.

`docs/performance/benchmarks.md` remains the canonical place for published
benchmark results generated from CSV output.

### 3.4 External Workspace Documents

RFC criteria that target files outside this repository, such as
`multica-home/knowledge/projects/pcl-rustic.md`, are tracked as external
evidence. The RFC may reference them, but this repository must not claim they
exist unless they are present in the current workspace or otherwise recorded.

### 3.5 Example Fixtures

Existing RFC criteria that explicitly require LAS/LAZ fixtures, `tests/data`,
or LAS round-trip behavior remain required. Synthetic fixtures may satisfy only
new or amended criteria that do not explicitly name LAS/LAZ fixture behavior.
When synthetic fixture-backed example tests are accepted, they should assert
point-count reduction and dtype preservation.

### 3.6 External Evidence Tracking Schema

Each RFC with open external evidence must include or link to an evidence record
with these fields:

| Field | Meaning |
|---|---|
| Criterion | The exact RFC acceptance item. |
| Artifact | Path or URL for the recorded evidence, or `unrecorded`. |
| Date | Date the evidence was produced. |
| Git SHA | Source revision used for the evidence. |
| Hardware / Dataset | Machine/backend and dataset or fixture identity. |
| Status | `open`, `recorded`, or `superseded`. |
| Notes | Constraints, threshold, comparator, or reason it remains open. |

## 4. Required Repository Updates

- Update `docs/plans/README.md` to list RFC-0011 as `Accepted` after review
  changes are incorporated.
- Mark RFC-0010 acceptance criteria complete because current RFC wording,
  `docs/api/pointcloud.md`, `docs/memory/implementation-progress.md`,
  `tests/test_point_cloud.py::TestPointCloudProperties::test_typed_attributes_round_trip`,
  LAS propagation tests, concat tests, outlier tests, covariance tests, and
  `cargo test --lib` cover them.
- Keep RFC-0002 as `Partial (external evidence open)` because the external
  Multica document, cross-platform CI evidence, and XYZ getter benchmark
  evidence are not recorded here.
- Keep RFC-0003 as `Partial` because LAS fixture-backed selector/example
  coverage remains a repo-local gap.
- Keep RFC-0004 as `Proposed` because GPU hot-path implementation is still a
  repo-local gap.
- Keep RFC-0005 as `Partial` because octree pruning remains a repo-local gap.
- Keep RFC-0006 as `Partial` because LAS ExtraBytes propagation remains a
  repo-local gap.
- Keep RFC-0007 as `Partial` because real point-to-plane/GICP solvers remain
  repo-local gaps.
- Change RFC-0009 to `Implemented (external evidence open)` because the
  benchmark harness is present but standard/full benchmark artifacts are not
  recorded.
- Update `docs/memory/implementation-progress.md` to use the two-track
  terminology.

## 5. Acceptance Criteria

- [x] This RFC is reviewed by two independent subagents for ambiguity,
  accidental scope weakening, and consistency with existing RFC wording.
- [x] Review feedback is incorporated: status suffixes are mandatory when
  external criteria remain, LAS fixture wording no longer weakens RFC-0003,
  benchmark artifact fields are stronger, third-party comparisons are narrowed,
  and external evidence records have a schema.
- [x] `docs/plans/README.md` lists this RFC.
- [x] RFC status/checklist updates distinguish repo-local implementation gaps
  from external evidence gaps.
- [x] `docs/memory/implementation-progress.md` records the decision and keeps
  remaining benchmark/hardware evidence unclaimed.

## 6. Risks And Mitigations

| Risk | Mitigation |
|---|---|
| This weakens acceptance criteria | External evidence remains unchecked and visible; only repo-local implementation status is separated. |
| Agents overuse `external evidence open` to avoid hard work | RFCs with code, test, or docs gaps must stay `Partial`; §4 names known repo-local gaps that cannot be reclassified. |
| Benchmark docs publish unverified numbers | §3.3 forbids measured claims without recorded artifacts. |

## 7. Out Of Scope

- Changing the RFC-0004 GPU hot-path design.
- Accepting staged GICP as complete.
- Fabricating benchmark CSVs or performance numbers.
- Writing files outside this repository.

## 8. Open Questions

None. This RFC defines tracking semantics only; it does not change feature
requirements.
