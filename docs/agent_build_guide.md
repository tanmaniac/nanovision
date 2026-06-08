# Build-agent guide

You are an ephemeral build agent. You are building ONE nanovision assignment from a plan that
has already been written and expert-reviewed. Your job is to produce the code, tests, viz, and
README, get both test modes green, and report back. You start from a cold context, so read the
authoritative sources below before writing anything; do not guess the conventions.

## Read these first (they are the ground truth, not this summary)

1. The assignment plan you were given: `docs/<assignment>_plan.md`. It is self-contained and
   carries the expert-reviewed design, every hole, every test with its pass condition, and the
   measured thresholds. Build exactly what it specifies. Corrections from the expert review are
   already folded in - do not re-litigate them.
2. `ARCHITECTURE.md` sections 2-5: the repo layout, the harness, and section 5 the README
   depth spec.
3. `TEMPLATE.md`: the `ASSIGNMENT.md` builder-contract format.
4. `~/.claude/CLAUDE.md` (Writing style + Claims and reasoning): the prose style rules. Read
   the file; do not assume you already know them.
5. The lecture-notes skill at `.claude/skills/lecture-notes/SKILL.md`: the README standard and
   the mandatory context-less style-review step.
6. AN EXEMPLAR ASSIGNMENT, named in your spawn prompt (e.g. `assignments/a06_0_flow_matching`).
   Treat its file structure as the template to mirror exactly: `conftest.py`, the top-level
   holed files, `solution/`, `tests/`, `config.py`, `viz.py`, `README.md`, `ASSIGNMENT.md`,
   the `__init__.py` files. Copy its conventions; change only what the plan changes.

## The layout, in one paragraph (verify against the exemplar)

The student edits the TOP-LEVEL files in the assignment dir (these carry the
`raise NotImplementedError` holes). `solution/<file>.py` is the filled answer key for each
holed file. A holed file's provided helpers and classes appear IDENTICALLY in the top-level and
solution copies; only the hole bodies differ. `config.py` and `viz.py` live only at the top
level (provided, no solution copy). A shared symbol that other assignments import lives in a
thin `nanovision/<mod>.py` shim that does `load("<assignment_dir>", "<module>")` via
`nanovision/_student.py`; the owning file is the assignment's top-level/solution `<module>.py`.
Import rule: a shared OWNED file is imported ONLY via `nanovision.*` (never bare, by anyone,
including the assignment's own sibling files and tests); an assignment-LOCAL file is imported
ONLY bare (never via `nanovision`). `conftest.py` puts the assignment dir then the impl dir
(solution/ under `NANOVISION_IMPL=solution`, else top-level) on `sys.path`.

## Rules that are written nowhere else

- VERIFY BOTH MODES. `NANOVISION_IMPL=solution python -m pytest assignments/<dir>/tests` must be
  fully green. Default mode (`NANOVISION_IMPL` unset) must FAIL CLEANLY at the holes
  (`NotImplementedError`), NOT with import/collection errors - except `test_forbidden_imports`,
  which passes in both modes (it is a static scan). Run pytest with the env's python:
  `/home/tanmay/miniconda3/envs/nanovision/bin/python`.
- DO NOT THRASH ON TESTS. The plan's thresholds are pre-measured. If a test floors above its
  threshold, run it once or twice, then REPORT THE MEASURED NUMBER and stop - do not hand-tune
  hyperparameters in an endless loop trying to force it under. A genuine floor (e.g. an
  irreducible loss from data structure) is information to report, not a number to beat. (A past
  background agent burned 45 minutes thrashing a single impossible threshold; do not repeat it.)
- TESTS MUST BE FAST AND CHEAP. Each test runs on CPU in seconds; overfit tests are bounded step
  counts (a few hundred to ~3000) that reach a known-reachable pass condition. Prefer
  training-free exact checks (an analytic oracle whose output the sampler/loss must reproduce
  exactly) over tests that depend on a training run converging.
- ONE FORBIDDEN-IMPORTS TEST per assignment, scanning the top-level files + solution + the
  shim, stripping comments/strings via tokenize. Mirror the exemplar's `test_forbidden_imports.py`.
- README is comprehensive lecture notes per the skill: motivation (historical landscape, why the
  paper mattered, the technical core, named forward connections), real-LaTeX math (`$...$` /
  `$$...$$`, never inline-code or ASCII for equations; inline code only for identifiers and
  file names; mermaid node labels stay plain text), inline arXiv links you have VERIFIED by
  fetching `https://arxiv.org/abs/<id>` and confirming the title. After drafting the README, run
  the mandatory context-less style-review (spawn a general-purpose subagent given only the README
  path + `~/.claude/CLAUDE.md`, per the skill) and apply its edits.
- ASSIGNMENT.md is the concise builder contract in `TEMPLATE.md` format; do not echo the README's
  prose. No stray trailing code fence.
- Style on everything (code comments, README, ASSIGNMENT): match `~/.claude/CLAUDE.md` - plain
  American English, sentence-case headings, single-hyphen not em dash, straight quotes, no filler
  ("delve/leverage/robust/crucial/seamless/fundamentally/spine/..."), active voice (no "X is what
  does Y" clefts), refer to prior concepts by name not assignment number, define jargon at first
  use.

## What to return (keep it under ~400 words)

- The files you created (paths), and which are holes vs provided.
- Both-mode results: the solution-mode test count (all green?), and confirmation default mode
  fails only at the holes.
- The measured numbers the plan asked for (final losses, perplexity, straightness, etc.), so the
  orchestrator can sanity-check against expectations.
- ANY deviation from the plan and why (a threshold you had to set from measurement, a shape you
  changed, an API that differed from what the plan assumed).
- Anything that looked wrong in the plan that you did NOT change (flag it for the orchestrator).

Do NOT commit, push, or edit `BUILD_CHECKLIST.md`/memory - the orchestrator does that after
verifying your output on disk.
