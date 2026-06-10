# TEMPLATE.md - per-assignment spec format

> Superseded 2026-06 and kept for historical reference. The two-file split this describes is gone:
> `ASSIGNMENT.md` was removed and its contract folded into a single student-facing `README.md`. For
> the current standard see `claude_notes/lecture_note_template.md` and the `lecture-notes` skill.

Every assignment directory contains an `ASSIGNMENT.md` written in this format.
It is the machine-readable contract the builder uses to generate the top-level
module files (the holed code the student edits), `solution/`, `tests/`, and
`README.md`. Copy this template, fill every field.
Do not leave a field as "TODO" - if something is genuinely N/A, write "N/A" and
why.

```yaml
id: aXX_name
title: Human-readable title
module: <0 | 1 | 2 | 3 | 3.5 | 4>
type: <Core | Survey | Mixed>
estimated_learner_hours: <number>           # focused work, excluding training time
depends_on: [a00_harness, ...]              # assignment ids whose code is imported
builds_into_shared_lib:                     # public symbols this adds to nanovision/
  - nanovision.module.Symbol
forbidden_imports:                          # enforced by a grep test on the top-level files + solution/
  - nn.MultiheadAttention
fits_12gb: <true | false>                   # true if a real train run fits; if false, explain
external_data: <none | "nuScenes v1.0-mini (~4GB, license)" | ...>
```

> ASSIGNMENT.md is the concise builder contract. The comprehensive, lecture-note
> version of motivation and background lives in the generated README.md (see
> ARCHITECTURE.md §5 for its depth requirements: historical landscape, why the
> paper mattered at the time, the technical core, inline paper links, and the
> forward connections). Keep the fields below brief here and do not duplicate the
> README's prose.

## motivation
2-4 sentences: the one-line historical placement and why the mechanism matters.
The full treatment (with paper links and significance) goes in the README.

## background
The concise math. Key equations only (use LaTeX-in-Markdown). No textbook
padding. State the core formula(s) the learner will implement and the shapes of
everything.

## what_you_implement
Bullet list of the specific mechanisms (not boilerplate) the learner writes.

## tasks
Numbered list. Each task:
- **Task N - <name>** (file: `<module>.py`, symbol: `<fn/class>`):
  one-paragraph contract - inputs, outputs, shapes, the formula/algorithm,
  and the single conceptual point it teaches. Each task maps 1:1 to a
  `NotImplementedError` hole in the top-level `<module>.py` and 1:1 to a test in
  tests/.

## tests
For each task, the test that proves it:
- `tests/test_*.py::test_name` - what it asserts and how (shape | gradcheck |
  reference-value | overfit-one-batch | end-to-end). Specify the order they
  should be run in the README.

## provided_boilerplate
What the builder gives the learner for free (data loaders, config, training-loop
wiring, plotting). The learner should never have to write these.

## compute_notes
What fits in 12GB, default image/model/batch sizes, expected wall-clock for the
real run (or "overfit-only; no real run" with the reason), and what a healthy
loss curve looks like (so a flat/slow curve isn't misread as a bug).

## stretch_goals
Optional extensions, ordered easy→hard.

## further_reading
3–6 papers with one-line annotations. Original sources preferred.

## solution_notes
Builder-only: known numerical-stability gotchas, the seed that makes the
overfit test reliable, any reference values hardcoded in tests and how they were
obtained.
```
