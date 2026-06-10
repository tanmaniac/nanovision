"""nanovision - shared library for the implement-the-mechanism CV course.

This package holds no mechanism code of its own. Each shared module
(``nanovision.primitives``, ``nanovision.attention``, ``nanovision.transformer``,
``nanovision.trainer``, ``nanovision.geometry``) is a thin shim that loads the
symbol from the assignment where the student builds it, via
``nanovision/_student.py`` (keyed on NANOVISION_IMPL: the student's top-level code
by default, or solution/ for the reference). The few genuinely provided modules
(``gradcheck``, ``determinism``, ``data``, ``viz``) live here directly.

Kept intentionally light: submodules are imported explicitly
(``from nanovision.attention import MultiHeadAttention``) so an assignment pulls in
only the parts it needs. See claude_notes/ARCHITECTURE.md section 3 for the import-path contract.
"""

__version__ = "0.1.0"
