# Lecture-note README structure (single merged file per assignment)

Agreed 2026-06-09, revised after the a05 pilot. Each assignment has ONE student-facing file:
`assignments/<id>/README.md`. `ASSIGNMENT.md` is removed; the brief "which files, how to run" part
moves into the README. The professor chooses the structure that teaches best - there is no mandated
section list. What the note MUST satisfy:

## Hard requirements

1. **A clear boundary between theory and assignment.** It must be obvious where the literature
   (what the field established) ends and the assignment (what the student builds) begins. This is
   the one non-negotiable. Blurring the two is the exact problem this rewrite exists to fix.

2. **A short orientation up front, in plain prose** (no "(a)/(b)" labels): the new concept the
   assignment teaches, and what the student builds to learn it. A paragraph or two.

3. **Required reading up front:** the 1-3 core papers to read before starting, with arXiv links.
   Broader or optional reading can go at the end.

4. **Lecture notes that are pure literature:** motivation, intuition, the math, why the method
   works, history. Constraints:
   - No assignment framing here (no files, holes, `make` targets, tests, or "what you implement").
   - No toy, solution, or measured numbers here (those go in the assignment/validation part).
   - Impersonal voice (describe the method, not "you build/measure").
   - Refer to previously-taught concepts by name, not by assignment number. Define jargon at first
     use. Diagrams (mermaid) are welcome as teaching aids.

5. **Lecture notes must be SUFFICIENT to solve the assignment.** Validate this explicitly: every
   concept a student needs to fill the holes is either explained in the notes or covered by a
   linked paper that explains it. This does NOT mean handing over the solution - it means no
   required concept is left unexplained and unlinked. Read the holes and the solution, list the
   concepts they depend on, and confirm each is taught or cited. Fill any gap with an explanation
   or a paper link, not with the answer.

6. **A lean assignment part, in the imperative.** Instructions use the imperative mood -
   "Implement the loss", "Write `q_sample`", "Run `make verify`" - NOT second-person "You implement
   / You write / You run". This holds for the orientation paragraph too: "Build a diffusion model
   from scratch", not "You build a diffusion model". Names which files to modify and how to run and
   validate (the `make`
   targets - `make verify A=<id>`, `make test A=<id>`, `make viz A=<id>` - and the
   `NANOVISION_IMPL=solution` vs default modes). State the test/verify distinction concretely, it
   confuses students: `make test A=<id>` runs the suite against the student's top-level files (red
   until the holes are filled, green when correct); `make verify A=<id>` runs the SAME suite against
   the reference `solution/` (it sets `NANOVISION_IMPL=solution`), so it is green from the start and
   shows the target. `make viz A=<id>` renders figures from the solution and writes PNGs to `out/`
   (matplotlib's headless Agg backend - no window, works over SSH/WSL/CI, viewable in VSCode);
   `make viz-mine A=<id>` renders the same figures from the student's code (for checking a finished
   implementation), and `SHOW=1` on either opens interactive windows in addition to the PNGs. Do
   NOT restate the per-function contracts,
   signatures, or shapes that already live in the code's docstrings and comments; the student reads
   those in the files. Point to the files and say at a high level what each one is for and which
   concept from the notes it implements. Toy/measured results and "what you should see when you run
   this" belong here, with honest toy-vs-scale caveats.

7. **Further reference material** at the end: where the method goes next, optional deeper reading,
   the full reference list with links.

## Rules that apply throughout

- READ `/home/tanmay/.claude/CLAUDE.md` IN FULL BEFORE WRITING and follow its "Writing style" and
  "Claims and reasoning" sections verbatim - that file is the authority, not any paraphrase of it.
  Every professor doing a rollout must open and read it first (subagents do not inherit it
  automatically). It bans, among much else: AI-slop and inflated-significance vocabulary,
  "load-bearing", the connective tics ("Additionally", "Moreover", "It's worth noting"), copula
  avoidance ("serves as"/"represents"), the rule of three, negative parallelism, "X is what does Y"
  clefts, em-dash overuse and double hyphens, smart quotes, and title-case headings.
- Math is real LaTeX (`$...$`, `$$...$$`). Mind the GitHub underscore rendering bug: never put a
  code-identifier underscore inside `\text{}`/`\operatorname{}` (even as `\_`); use a subscript
  (`\text{n}_{\text{cls}}`) or a literal hyphen (`\text{cross-entropy}`). See the
  github-latex-underscore-gotcha memory.
- This is a restructure, clarity, and sufficiency pass, not a license to invent. Preserve the
  correct technical content; the six domain reviewers already verified the current notes are sound.
  Keep the README consistent with the solution code and the test harness.

## Rollout

Pilot a05_diffusion first for sign-off, then roll the agreed structure across the other 21 via the
six domain-grouped professors. When rolling out: delete each `ASSIGNMENT.md`, fix the few
cross-references to it (a06_0/a10_5 READMEs, the a08 and a10_5 test-file comments), and update the
two-file standard in `.claude/skills/lecture-notes/SKILL.md` and `claude_notes/ARCHITECTURE.md` §5 /
`claude_notes/TEMPLATE.md` to the single-file structure.
