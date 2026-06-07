# nanovision — learner-facing commands.
#
# Activate the env first:  conda activate nanovision
# (or override PYTHON, e.g.  make verify A=a00_harness PYTHON=/path/to/python)
#
# Usage:
#   make test    A=a00_harness   # run YOUR starter against the tests (red until filled)
#   make verify  A=a00_harness   # run the reference solution (must be green)
#   make viz     A=a00_harness   # render the result to assignments/<A>/out/
#   make test-all / make verify-all
#
# The NANOVISION_IMPL env var selects which implementation the tests import;
# each assignment's conftest.py puts starter/ or solution/ on sys.path.
#
# Aggregate targets (test-all / verify-all) run each assignment in its OWN pytest
# process. That is deliberate: assignments reuse module basenames (test_shapes.py,
# primitives.py, ...) and the impl-switch imports them by bare name, so a single
# shared process would cross-contaminate sys.modules. One process per assignment
# is also how a learner actually works.

PYTHON ?= python
PYTEST := $(PYTHON) -m pytest
A ?=

.PHONY: test verify viz test-all verify-all install

install:
	$(PYTHON) -m pip install -e .

test:
	@test -n "$(A)" || (echo "set A=<assignment id>, e.g. make test A=a00_harness" && exit 1)
	NANOVISION_IMPL=starter $(PYTEST) assignments/$(A)/tests -v

verify:
	@test -n "$(A)" || (echo "set A=<assignment id>, e.g. make verify A=a00_harness" && exit 1)
	NANOVISION_IMPL=solution $(PYTEST) assignments/$(A)/tests -v

viz:
	@test -n "$(A)" || (echo "set A=<assignment id>, e.g. make viz A=a00_harness" && exit 1)
	NANOVISION_IMPL=solution $(PYTHON) assignments/$(A)/viz.py

# Starter across all assignments. Does NOT stop on failure: unfilled assignments
# are expected to fail, so this just shows each assignment's current status.
test-all:
	@for d in assignments/*/; do \
		[ -d "$$d/tests" ] || continue; \
		echo "== $$(basename $$d) =="; \
		NANOVISION_IMPL=starter $(PYTEST) "$$d/tests" -q --no-header || true; \
	done

# Solutions across all assignments. Stops at the first regression (the CI green bar).
verify-all:
	@set -e; for d in assignments/*/; do \
		[ -d "$$d/tests" ] || continue; \
		echo "== $$(basename $$d) =="; \
		NANOVISION_IMPL=solution $(PYTEST) "$$d/tests" -q --no-header; \
	done
