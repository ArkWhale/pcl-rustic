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
| [RFC-0002](rfc-0002-2026-04-30-api-reset-and-typed-attributes.md) | API Reset & Typed Attributes | Accepted | M1 |
| [RFC-0003](rfc-0003-2026-04-30-coord-ops-and-selection.md) | Coordinate Ops & Selection | Accepted | M2 |
| [RFC-0004](rfc-0004-2026-04-30-gpu-hot-path.md) | GPU Hot-Path Rewrite | Accepted | M3 |
| [RFC-0005](rfc-0005-2026-04-30-knn-and-normals.md) | Neighborhood Infra & Normal Estimation | Accepted | M4 |
| [RFC-0006](rfc-0006-2026-04-30-outlier-removal.md) | Outlier Removal (SOR & ROR) | Accepted | M5 |
| [RFC-0007](rfc-0007-2026-04-30-icp-gicp-registration.md) | ICP & GICP Registration | Accepted | M6 |
| [RFC-0008](rfc-0008-2026-05-03-attribute-value-storage.md) | AttributeValue Storage Design (Burn 0.20) | Accepted | M1 |

## Process

### Authoring a new RFC

1. **Create the file** using the naming convention above in `docs/plans/`.
2. **Add a row** to the table above with Status `Proposed`.
3. **Require two sub-agent reviews**: before the RFC can move to `Accepted`,
   two independent review agents must be invoked and their feedback addressed.
   Record the review outcome in the RFC's **§ Reviews** section.
4. **Add bidirectional links**: if your RFC depends on or is depended on by
   another RFC, add cross-links in both documents under the `Related:` header
   and the `## References` section.
5. **Update status** to `Accepted` once reviews pass and any open questions
   are resolved (or explicitly deferred to a follow-up RFC).

### RFC section template

```markdown
# RFC-NNNN: <Title>

- **Status:** Proposed | Accepted | Superseded
- **Date:** YYYY-MM-DD
- **Author:** <author>
- **Related:** RFC-XXXX (<role>), …

## 1. Summary
## 2. Motivation
## 3. Detailed design
## 4. Acceptance criteria
## 5. Risks & mitigations
## 6. Out of scope
## 7. Open questions
## 8. Reviews
```

### Bidirectional link rule

Every RFC that mentions another must add a `Related:` entry in its front
matter, and the referenced RFC must be updated to reference back.  Format:

```
- **Related:** RFC-0002 (prerequisite), RFC-0004 (downstream)
```

### What belongs where

| Content | Section |
|---|---|
| What the RFC does | §1 Summary |
| Why we need it | §2 Motivation |
| How it works (API, data model, algorithm) | §3 Detailed design |
| Pass/fail gates | §4 Acceptance criteria |
| What can go wrong and how to mitigate | §5 Risks |
| Scope exclusions | §6 Out of scope |
| Unresolved choices | §7 Open questions |
| Sub-agent review outcomes | §8 Reviews |
