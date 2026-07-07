"""Repo-wide pytest helper.

Adds --skip-holes: turn a still-unimplemented assignment hole into a skipped test instead
of a failure. That lets you run the whole suite while iterating and see pass/fail only for
the functions you have actually written; the holes you haven't reached yet show up as skips
instead of pages of tracebacks. A genuine bug in a function you HAVE implemented still fails
normally - only an unimplemented hole (anywhere in the error's cause/context chain) is
converted to a skip.

A hole is recognized two ways: the Python assignments raise NotImplementedError, and the
C++ assignments (a14*) throw std::logic_error("NOT_IMPLEMENTED: <fn>"), which pybind11
surfaces as a RuntimeError whose message begins with NOT_IMPLEMENTED.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skip-holes",
        action="store_true",
        default=False,
        help="Skip tests whose failure is an unimplemented hole (NotImplementedError) "
        "instead of reporting them as failures.",
    )


def _raised_by_hole(exc: BaseException) -> bool:
    """True if an unimplemented hole sits anywhere in this exception's cause/context chain.

    Python holes are NotImplementedError; C++ (a14*) holes are a RuntimeError whose message
    starts with the "NOT_IMPLEMENTED:" sentinel thrown from the C++ stub.
    """
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, NotImplementedError):
            return True
        if str(exc).startswith("NOT_IMPLEMENTED"):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except Exception as exc:
        if item.config.getoption("--skip-holes") and _raised_by_hole(exc):
            pytest.skip(f"hole not implemented yet: {exc}")
        raise
