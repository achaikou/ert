# Design notes: `test_ert_storage_diff.yml`

This file records the *intent* behind design decisions in the workflow that
aren't obvious from the YAML alone. Update it whenever the workflow's
approach changes meaningfully, not for every bug fix.

## Purpose

Run every example under `test-data/ert/` on both the PR branch and `main`,
then:

1. Diff the resulting storage directories to catch unintended storage format
   changes.
2. Check that storage produced by `main` can still be opened/continued
   (`es_mda --prior-ensemble-id`) using PR code, to catch backward
   compatibility breaks.

The workflow is informational only — see "Never blocks merging" below.

## Jobs

- `run-examples`: discovers `(example, config)` pairs to test.
- `run-example`: runs `ert ensemble_smoother` once per `(ref, example,
  config)`, normalizes non-deterministic content, uploads storage as an
  artifact. Also uploads the raw (non-normalized) `main` storage separately,
  for use by `check-storage-backward-compatibility`.
- `compare`: downloads both `pr` and `main` storage artifacts, diffs them
  file by file, tolerant of known noise (see below), and uploads a
  pass/fail status + Markdown report per example.
- `check-storage-backward-compatibility`: downloads raw `main` storage,
  checks out PR code, and re-runs `es_mda --prior-ensemble-id` against it to
  confirm PR code can still open/continue an old storage.
- `report`: combines both jobs' per-example results into one unified
  `GITHUB_STEP_SUMMARY`.

## Key design decisions

### Never blocks merging

Every job (`run-example`, `compare`, `check-storage-backward-compatibility`)
has `continue-on-error: true` at the **job level**, and `report` has
`if: always()`. This is intentional: the workflow's outcome must never
prevent merging. The `report` job's summary is the actual signal reviewers
should look at, not the workflow's pass/fail status.

Because of this, do not "fix" a failing job by making the workflow red —
that defeats the purpose. Failures should surface only inside the report.

### Normalization before comparison

Raw storage contains inherently non-deterministic values that would cause
false-positive diffs:

- **GUIDs** (experiment id, ensemble ids, blob names): replaced with
  constant placeholders (`aaaaaaaa-...` for the experiment, per-ensemble
  synthetic UUIDs derived from creation order, similarly for blobs). This
  requires listing `storage/ensembles/*` sorted by directory mtime and
  substituting.
  - Caveat: directory mtime ordering has been observed to lose sub-second
    precision after artifact upload/download (tar/zip roundtrip), which can
    misidentify which ensemble is "iteration 0" and which is "iteration 1".
    Prefer reading `index.json`'s `"iteration"` field directly wherever
    possible instead of relying on mtime order (see
    `check-storage-backward-compatibility`'s "Determine prior ensemble id"
    step for the safer pattern).
- **Timestamps**: any ISO8601-looking timestamp is rewritten to a constant
  value via regex.
- **`NUM_REALIZATIONS`/`RANDOM_SEED`**: pinned to fixed values (`2` /
  `1234`) before running, so different default realization counts don't
  cause the storage shape to differ, and stochastic parameter sampling is
  reproducible.
- **`ENSPATH`/`RUNPATH`**: stripped from the config so storage lands in the
  default location the rest of the workflow expects.

### Comparison tolerance (`compare` job)

A plain byte-for-byte `diff` is too strict for two reasons discovered
during testing:

1. **JSON key ordering** is not guaranteed stable across runs/versions, but
   is semantically meaningless (JSON objects are unordered maps).
2. **Floating-point noise**: identical runs (branch vs itself) have been
   observed to produce trailing-digit differences (e.g.
   `0.5750322970816912` vs `...913`), most likely from binary/native calls,
   not from Python-level determinism.

`.github/scripts/compare_storage_file.py` handles both by:

- Comparing JSON files structurally after rounding all float leaves to 8
  decimal places (order-independent).
- Falling back to text comparison with the same float-rounding for
  non-JSON text.
- Falling back to `polars.testing.assert_frame_equal(..., atol=1e-8)` for
  parquet/binary files.

A file is only reported as "failed" if none of these tolerant comparisons
succeed. `.github/scripts/describe_storage_diff.py` is a best-effort,
job-log-only (not report-facing) summary of *why* two parquet files differ
(shape/columns/max-abs-diff), since a full column diff would bloat the
report.

### Excluded storage paths

`EXCLUDED_STORAGE_PATHS` currently excludes `**/*.log`, `**/logs/**`, and
`**/workflow_events.jsonl`. The latter contains a `run_id` UUID with no
current point of reference to normalize against (unlike experiment/ensemble
ids, which are discoverable via directory names), so the whole file is
treated as log-equivalent and excluded rather than normalized. If `run_id`
is ever found elsewhere in storage in a place that *does* need comparing,
this decision should be revisited — ideally by adding `run_id` to the same
GUID-mapping mechanism instead of excluding files wholesale.

### Backward-compatibility check specifics

- Uses `es_mda --prior-ensemble-id <oldest-by-iteration-0> --weights 1`
  rather than `ensemble_smoother`, so PR code is forced to *load and
  continue* old storage rather than create a fresh experiment — this is the
  actual scenario being tested (does old storage still open under new
  code).
- The prior ensemble id is found by reading each
  `storage/ensembles/*/index.json`'s `"iteration"` field and picking the
  one with `iteration == 0` — **not** by mtime, for the reason noted above.
- This check is expected to fail for configs whose forward-model / parameter
  code paths only get exercised at `iteration > 0` (e.g. a known
  `GenKwConfig.write_to_runpath` gap that raises `NotImplementedError` for
  `forward_init` parameters when `iteration != 0` — that failure is
  unrelated to storage compatibility and is a separate, real code gap).

### Report format

Per-example section order: emoji header (✅/⚠️/❓) → backward-compatibility
line (✅/❌/❓) → collapsed `<details>` diff report. Overall header is
`✅ No storage changes` or `⚠️ Storage changes observed` (with follow-up
guidance to revert or add a migration), computed by OR-ing every example's
compat/compare status. `GITHUB_STEP_SUMMARY` has a 1024 KiB cap; both
per-file diffs (head/tail 50 lines) and the final assembled summary have
truncation safeguards for this.

## Known limitations / possible future work

- `run_id` in `workflow_events.jsonl` is excluded, not normalized (no
  reference point found yet — see above).
- mtime-based ordering is fragile after artifact archiving; prefer
  `index.json` parsing wherever an ordering decision is needed.
- The workflow intentionally never fails the check itself — if this policy
  ever changes (e.g. wanting it to block merging), all the
  `continue-on-error: true` / `if: always()` job-level settings would need
  to be reconsidered together, not just the `report` job.