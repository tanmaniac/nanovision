# nanovision - learner-facing commands.
#
# Activate the env first:  conda activate nanovision
# (or override PYTHON, e.g.  make verify A=a00_harness PYTHON=/path/to/python)
#
# Usage:
#   make test    A=a00_harness   # run YOUR code (the top-level assignment files) against the tests (red until filled)
#   make verify  A=a00_harness   # run the reference in solution/ (must be green)
#   make viz     A=a00_harness   # render the result to assignments/<A>/out/
#   make test-all / make verify-all
#
# There is no install step. The repo runs from its root: pytest gets the root via
# `pythonpath = ["."]` (pyproject.toml) and scripts run as modules (`python -m`).
# The student edits the top-level files in each assignment dir; solution/ is the
# read-only reference. NANOVISION_IMPL=solution switches both the shared-library
# shims and the assignment-local imports to solution/; unset means the student's
# top-level code.
#
# Aggregate targets (test-all / verify-all) run each assignment in its OWN pytest
# process. That is deliberate: assignments reuse module basenames (test_shapes.py,
# vit.py, ...) and conftest imports the local files by bare name, so a single
# shared process would cross-contaminate sys.modules. One process per assignment
# is also how a learner actually works.

PYTHON ?= python
PYTEST := $(PYTHON) -m pytest
A ?=

.PHONY: test verify viz test-all verify-all

test:
	@test -n "$(A)" || (echo "set A=<assignment id>, e.g. make test A=a00_harness" && exit 1)
	$(PYTEST) assignments/$(A)/tests -v

verify:
	@test -n "$(A)" || (echo "set A=<assignment id>, e.g. make verify A=a00_harness" && exit 1)
	NANOVISION_IMPL=solution $(PYTEST) assignments/$(A)/tests -v

viz:
	@test -n "$(A)" || (echo "set A=<assignment id>, e.g. make viz A=a00_harness" && exit 1)
	NANOVISION_IMPL=solution $(if $(SHOW),NANOVISION_VIZ_SHOW=1 )$(PYTHON) -m assignments.$(A).viz

# Render the figures from YOUR top-level code (run once the holes are filled). Add SHOW=1 to
# open interactive windows in addition to writing the PNGs, e.g. make viz-mine A=a05_diffusion SHOW=1
viz-mine:
	@test -n "$(A)" || (echo "set A=<assignment id>, e.g. make viz-mine A=a00_harness" && exit 1)
	$(if $(SHOW),NANOVISION_VIZ_SHOW=1 )$(PYTHON) -m assignments.$(A).viz

# Student top-level code across all assignments. Does NOT stop on failure: unfilled
# assignments are expected to fail, so this just shows each assignment's status.
test-all:
	@for d in assignments/*/; do \
		[ -d "$$d/tests" ] || continue; \
		echo "== $$(basename $$d) =="; \
		$(PYTEST) "$$d/tests" -q --no-header || true; \
	done

# Reference solutions across all assignments. Stops at the first regression (the CI green bar).
verify-all:
	@set -e; for d in assignments/*/; do \
		[ -d "$$d/tests" ] || continue; \
		echo "== $$(basename $$d) =="; \
		NANOVISION_IMPL=solution $(PYTEST) "$$d/tests" -q --no-header; \
	done
