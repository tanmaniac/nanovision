---
name: lecture-notes
description: Generate or revise a nanovision assignment's single-file README.md (lecture notes plus a lean assignment section) to the agreed structure. Use when writing or improving an assignment handout, when the theory and the assignment are blurred together, or when adding paper context/motivation. Delegates the writing to an Opus subagent to conserve the main context.
---

# Lecture notes for a nanovision assignment

Each assignment has ONE student-facing file: `assignments/<id>/README.md`. It holds both the
literature (lecture notes) and a lean assignment section, with a clear boundary between them. There
is no separate ASSIGNMENT.md (that two-file split was removed in 2026-06; the builder contract is
now the assignment section of the README).

`claude_notes/lecture_note_template.md` is the authoritative structure contract - read it first and
follow it. `assignments/a05_diffusion/README.md` is the approved exemplar of the target structure,
voice, and depth. In short, the README must have: a plain-prose orientation (concept taught + what
gets built) and required reading up front; pure-literature lecture notes (motivation, math, history,
impersonal voice, NO file/make/test mentions and NO toy/measured numbers, diagrams welcome); a lean
assignment section in the IMPERATIVE mood (which files to modify mapped to the concepts they
implement, not a restatement of the code's docstrings, plus how to run and validate); and references
at the end. Toy/measured numbers live only in the assignment's "what you should see when you run
this", with toy-vs-scale caveats.

## Why a subagent

Writing good lecture notes is long-form and reads several files plus the web. Do it in an Opus
subagent so the main session does not burn context. One subagent per assignment (or per small domain
group); each touches only its `README.md`, so several can run in parallel on disjoint files.

## How to run it

Launch an Opus subagent (general-purpose, web search available) with a prompt that:

1. Points it at `claude_notes/lecture_note_template.md` (the structure contract) and
   `assignments/a05_diffusion/README.md` (the exemplar), and requires it to read
   `~/.claude/CLAUDE.md` in full (the writing-style authority - subagents do not inherit it).
2. Names the assignment dir and tells it to read the current `README.md`, the top-level hole files,
   the `solution/` code, the `tests/`, and the `Makefile`, so the notes match the actual code,
   shapes, and run commands.
3. Tasks it to rewrite `README.md` to the template: clear theory/assignment boundary; impersonal
   lecture notes with no assignment framing or toy numbers; an imperative assignment section; the
   concrete `make test` / `make verify` / `make viz` / `make viz-mine` (+`SHOW=1`) distinction. It
   must touch ONLY that README - no code, tests, conftest, or viz.
4. Requires a SUFFICIENCY CHECK: list the concepts the holes require, and confirm each is explained
   in the lecture notes or covered by a linked paper (without putting the solution in the notes).
   Close any gap with an explanation or a paper link.
5. Requires it to verify every arXiv id by fetching `https://arxiv.org/abs/<id>` and confirming the
   title matches BEFORE citing it. No unverified citations.
6. States the writing style: plain American English, sentence-case headings, single hyphen never
   `--`, straight quotes, no AI-slop/banned vocabulary (no "load-bearing", "robust", "crucial",
   "delve", "leverage", "seamless", inflated-significance words, copula avoidance, negative
   parallelism, the rule of three). Impersonal voice in the lecture notes; imperative in the
   assignment section ("Implement X", "Run make verify"), never "You implement / You write". Refer
   to previously-taught concepts by name, not by assignment number. Define jargon at first use.
7. States the LaTeX rules: real LaTeX, inline `$...$` and display `$$...$$`, never inline-code spans
   or ASCII for equations. GitHub rendering bug: never put a code-identifier underscore inside
   `\text{}`/`\operatorname{}` (even as `\_`); use a subscript (`\text{n}_{\text{cls}}`) or a literal
   hyphen (`\text{cross-entropy}`); `log_softmax` -> `\log\operatorname{softmax}`. See the
   github-latex-underscore-gotcha memory.
8. Requires diagrams where they carry the explanation, preferring inline Mermaid fenced blocks
   (```mermaid) for architecture, data flow, and tensor-shape diagrams (GitHub renders these
   natively and they diff cleanly). A link to a specific figure in the originating paper is the
   second choice; a committed PNG under `assets/` only for quantitative plots with a reproducible
   script. Do not invent or hotlink figures you have not confirmed exist.

## Style review (mandatory, context-less subagent)

After the README is drafted, run a SEPARATE style-review subagent before considering the notes done.
The writer cannot see its own filler, so the reviewer starts from a clean context: a general-purpose
subagent given ONLY the README path and the writing-style rules. Prompt it: "Read ~/.claude/CLAUDE.md
(Writing style + Claims and reasoning) and the README at <path>. You have no other context and do
not need any. Edit ONLY that README to remove filler, meta-commentary, and banned vocabulary per
those rules, and rewrite flagged sentences to state the content directly. Do not add content, do not
change technical claims, shapes, numbers, or citations. Return a numbered list of every edit." It
must also catch: second-person "You build/write" voice (should be imperative), references to prior
concepts by assignment number instead of name, jargon used before it is defined, math written as
inline-code/ASCII instead of LaTeX, and any stray tool/markup tags left in the file.

## Verify after

Read the rendered `README.md` yourself for accuracy (the subagent can still get a claim wrong) and
spot-check a couple of cited arXiv links. Grep the file for leftover problems: stray tags
(`</content>`, `antml:`), `\_` inside `\text{}`, smart quotes, prose `--`, and `\bYou [a-z]`. The
style review is part of generation, not optional polish: a README is not done until it has passed
the context-less style pass.
