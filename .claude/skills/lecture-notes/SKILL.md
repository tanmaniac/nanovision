---
name: lecture-notes
description: Generate or revise a nanovision assignment's README.md as comprehensive lecture notes (and trim its ASSIGNMENT.md to a concise builder contract). Use when writing/improving an assignment handout, when the README is too thin or duplicates ASSIGNMENT.md, or when adding paper context/motivation. Delegates the writing to an Opus 4.8 subagent to conserve the main context.
---

# Lecture notes for a nanovision assignment

The README.md of each assignment is comprehensive lecture notes, not a terse
handout. ARCHITECTURE.md §5 is the authoritative depth spec. ASSIGNMENT.md is a
separate, concise builder contract and must not re-narrate the README's prose.

## Why a subagent

Writing good lecture notes is long-form and reads several files plus the web. Do
it in an Opus 4.8 subagent so the main session does not burn context. One subagent
per assignment; they touch only `README.md` and `ASSIGNMENT.md` for their
assignment, so several can run in parallel on disjoint files.

## How to run it

Launch an Opus subagent (general-purpose, web search available, run in background)
with a prompt that:

1. Points it at `ARCHITECTURE.md` §5 (the depth spec) and `TEMPLATE.md`.
2. Names the assignment dir and tells it to read the current `README.md`,
   `ASSIGNMENT.md`, the `starter/` holes, and the canonical `nanovision/` modules
   so the notes match the actual code and shapes.
3. Tasks it to rewrite `README.md` as comprehensive lecture notes and trim
   `ASSIGNMENT.md` to the builder contract (yaml, what_you_implement, per-task
   contracts, tests mapping, provided_boilerplate, terse compute_notes,
   solution_notes). It must touch ONLY those two files - no code, tests, conftest,
   or viz.
4. Requires it to verify every arXiv id by fetching `https://arxiv.org/abs/<id>`
   and confirming the title matches BEFORE citing it. No unverified citations.
5. States the writing style: plain American English, sentence-case headings, no em
   dashes (single hyphen), straight quotes, no filler ("delve", "leverage",
   "robust", "crucial", "seamless", etc.), and never the phrase "this is the
   substrate". Motivation must explain the pre-existing landscape and its limit,
   what the originating paper(s) changed and why that mattered at the time, the
   technical core in plain terms, and the concrete forward connections (name the
   later assignments and what they reuse), with paper links inline.

## Verify after

Read the rendered `README.md` yourself for accuracy (the subagent can still get a
claim wrong) and spot-check a couple of the cited arXiv links resolve to the
stated titles.
