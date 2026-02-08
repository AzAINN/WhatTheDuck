"""
Lightweight stub for mpax so OpenPhiSolve imports succeed without the real package.
If PDQP refinement is actually invoked, an ImportError is raised to prompt installation.
"""

def _missing(*_args, **_kwargs):
    raise ImportError("mpax not installed; install real mpax for PDQP refinement.")

def create_qp(*args, **kwargs):  # pragma: no cover
    return _missing()

class _DummySolver:
    def optimize(self, *args, **kwargs):
        return _missing()

def raPDHG(*args, **kwargs):  # pragma: no cover
    return _DummySolver()
