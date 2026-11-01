# Independent reviews

## Pre-review (openai/gpt-6.1-sol, completed 2026-11-01)

The first detached broad review hit its 20-tool-call budget and is not counted.
A fresh, synchronous, read-only consultation narrowed the scope to the full
PROTOCOL.md, grouped.rs:981–1029, and tcontract-exec-0.9.0 batch.rs:540–568.

Completed report:
- Prototype PASS: baseline Auto grants serial items; faer_auto prioritizes inner
  width. The max-job query/original-outer fallback matches the diagnostic intent;
  whole-group limitations are explicit.
- Experimental design PASS: held-Team/route checks, interleaved observations,
  separate CPU window, analytical validation support this bounded comparison.
- Claims PASS: C tests the declared limitation; equal MACs are not workload
  equivalence; 64T is reference-only; bandwidth attribution stays qualified.
- No BLOCKER/IMPORTANT/MINOR findings. Delete list: none.
- Not checked: harness/patch implementation, actual routing, measurements, issue
  discussion, broader source. Results inform #46, not establish its end-to-end
  resolution.

## Post-review (openai/gpt-6.1-sol, completed 2026-11-01)

A fresh synchronous read-only consultation covered the complete 667-line
implementation diff, protocol, results narrative, generated tables, current
production scheduler/callers, resolved faer/batch rule, build identities and
finish metadata, and selected environment/evidence ranges. The reviewer did
not exhaust the raw JSONL, evidence extract, validation logs, or issue history;
parent checks independently verified all 875 observations / 125 medians and
binary/source identities.

Completed findings and verdicts:
- MINOR: RESULTS.md's "all widths have 2 lanes" included 1T, whose recorded
  route is (1,1). Fixed the text to 8–64T and named the 1T exception. No code or
  measurement change and no rerun needed.
- No BLOCKER/IMPORTANT findings. Execution/identity, exact statistics, failure
  and noise handling, attribution/non-regression qualifications, and provenance
  passed the requested review scope.
- Acceptable diagnostic deliverable, not justification for production adoption.
  Existing dependency-documentation drift is accurately disclosed.
- Delete list: none; harness, recorder, reversible patch and evidence serve the
  bounded request. No external kernel code import or new dependency found.
- The reviewer ran no builds/writes; local gate evidence is the parent's.
