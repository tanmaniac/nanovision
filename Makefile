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

PYTHON ?= python
A ?=

.PHONY: test verify viz test-all verify-all install

install:
	$(PYTHON) -m pip install -e .

test:
	@test -n "$(A)" || (echo "set A=<assignment id>, e.g. make test A=a00_harness" && exit 1)
	NANOVISION_IMPL=starter $(PYTHON) -m pytest assignments/$(A)/tests -v

verify:
	@test -n "$(A)" || (echo "set A=<assignment id>, e.g. make verify A=a00_harness" && exit 1)
	NANOVISION_IMPL=solution $(PYTHON) -m pytest assignments/$(A)/tests -v

viz:
	@test -n "$(A)" || (echo "set A=<assignment id>, e.g. make viz A=a00_harness" && exit 1)
	NANOVISION_IMPL=solution $(PYTHON) assignments/$(A)/viz.py

test-all:
	NANOVISION_IMPL=starter $(PYTHON) -m pytest assignments -v

verify-all:
	NANOVISION_IMPL=solution $(PYTHON) -m pytest assignments -v
