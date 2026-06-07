"""nanovision — shared library for the implement-the-mechanism CV course.

Kept intentionally light: submodules are imported explicitly
(``from nanovision.attention import MultiHeadAttention``) so that an assignment
can import the parts it needs without pulling in modules built by later
assignments. See ARCHITECTURE.md §3 for the stable import-path contract.
"""

__version__ = "0.1.0"
