---
name: lecture-notes
description: Generate or revise a nanovision assignment's README.md as comprehensive lecture notes (and trim its ASSIGNMENT.md to a concise builder contract). Use when writing/improving an assignment handout, when the README is too thin or duplicates ASSIGNMENT.md, or when adding paper context/motivation. Delegates the writing to an Opus 4.8 subagent to conserve the main context.
---

# Lecture notes for a nanovision assignment

The README.md of each assignment is comprehensive lecture notes, not a terse
handout. claude_notes/ARCHITECTURE.md §5 is the authoritative depth spec. ASSIGNMENT.md is a
separate, concise builder contract and must not re-narrate the README's prose.

## Why a subagent

Writing good lecture notes is long-form and reads several files plus the web. Do
it in an Opus 4.8 subagent so the main session does not burn context. One subagent
per assignment; they touch only `README.md` and `ASSIGNMENT.md` for their
assignment, so several can run in parallel on disjoint files.

## How to run it

Launch an Opus subagent (general-purpose, web search available, run in background)
with a prompt that:

1. Points it at `claude_notes/ARCHITECTURE.md` §5 (the depth spec) and `claude_notes/TEMPLATE.md`.
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
   substrate". Math is real LaTeX: inline `$...$`, display `$$...$$`, with proper
   symbols ($\sqrt{\,}$, $\bar\alpha_t$, $\varepsilon$), never inline-code spans or
   ASCII like `sqrt(1-abar)` for equations (GitHub renders LaTeX natively). Inline
   code is for identifiers and file names only. Motivation must explain the
   pre-existing landscape and its limit,
   what the originating paper(s) changed and why that mattered at the time, the
   technical core in plain terms, and the concrete forward connections (name the
   later assignments and what they reuse), with paper links inline.
6. Requires figures/diagrams where they carry the explanation. Use, in order of
   preference: (a) inline Mermaid fenced blocks (```mermaid) for architecture,
   data flow, and tensor-shape diagrams - GitHub renders these natively, they
   diff cleanly, and they need no binary assets; (b) a link to a specific figure
   in the originating paper when a published figure is the clearest reference
   (link the ar5iv HTML, e.g. `https://ar5iv.org/abs/<id>`, or the arXiv page,
   and verify it resolves); (c) a generated PNG committed under the assignment's
   `assets/` and referenced by relative path, only for quantitative plots that
   need real numbers (noise schedules, attention maps, loss curves), and only if
   a small reproducible script produces it. Prefer (a) for anything structural.
   Do not invent or hotlink figures you have not confirmed exist.

## Style review (mandatory, context-less subagent)

After the README is drafted, run a SEPARATE style-review subagent before considering
the notes done. The writer (and the main session) cannot see their own filler, so the
reviewer must start from a clean context: spawn a general-purpose subagent (foreground)
that is given ONLY the README path and the writing-style rules, NOT the build context
or this conversation. The user's CLAUDE.md "Writing style" and "Claims and reasoning"
sections are the authority; the reviewer reads `~/.claude/CLAUDE.md` itself.

Its single job: find and remove flowery, meaningless, or meta-commentary prose and
rewrite it to be direct, editing ONLY the README. Specifically it must strip:
- Empty meta-commentary that asserts a sentence matters instead of stating the content
  ("the question this assignment answers is the smallest one that matters", "the honest
  answer is", "the whole point is", "what this means is", "the key insight", "the
  takeaway", "it's worth noting/restating").
- Banned vocabulary and tells from the CLAUDE.md list ("honest", "genuine", "robust",
  "leverage", "crucial", "delve", "fundamentally", "seamless", inflated-significance
  words, copula avoidance, negative parallelism, the rule of three as default rhythm,
  "Label: detail" headings, em dashes).
- Sentences that can be deleted with no loss of technical content.
- References to previously-taught concepts by assignment NUMBER instead of by name:
  "A2's patch embedding" -> "ViT's patch embedding", "the A1 encoder" -> "the
  transformer encoder", "A3's MAE" -> "the masked autoencoder (MAE)". A bare number is
  not readable on its own. (Forward pointers that name the upcoming topic are fine.)
- New jargon used before it is defined: if a term (tubelet, register token, EOS
  pooling, ...) is used before a clause explains what it is, add a short gloss at first
  use or flag it. The reader should never hit a word they cannot understand in place.
- Math written as inline-code spans or ASCII instead of LaTeX: convert `` `x_t` ``,
  `sqrt(1-abar)`, `alpha_bar_t`, and similar to real LaTeX ($x_t$, $\sqrt{1-\bar\alpha_t}$,
  $\bar\alpha_t$), inline as `$...$` and display equations as `$$...$$`. Leave genuine
  code identifiers and file names in inline code.
It must NOT add claims, soften technical precision, or touch code/shapes/citations. It
returns the list of edits it made so the change is auditable.

Prompt the reviewer to: "Read ~/.claude/CLAUDE.md (Writing style + Claims and reasoning)
and the README at <path>. You have no other context and do not need any. Edit ONLY that
README to remove filler, meta-commentary, and banned vocabulary per those rules, and
rewrite flagged sentences to state the content directly. Do not add content, do not
change technical claims, shapes, numbers, or citations. Return a numbered list of every
edit."

## Verify after

Read the rendered `README.md` yourself for accuracy (the subagent can still get a
claim wrong) and spot-check a couple of the cited arXiv links resolve to the
stated titles. The style review is part of generation, not optional polish: a README is
not done until it has passed the context-less style pass.
