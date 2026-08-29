"""Re-installs the offline guard in every child process the suite spawns.

Python imports `sitecustomize` automatically at interpreter start if it is importable, so
putting this directory on the child's `PYTHONPATH` is what makes the guard survive
`subprocess.run([sys.executable, ...])`. Without it, patching the parent said nothing about
its children -- and this suite shells out to `sys.executable` in seven modules, one of which
drives a module with an HTTP path in it.
"""

from __future__ import annotations

import os

if os.environ.get("NBC_OFFLINE_GUARD") == "1":
    try:
        import offline_guard
    except ImportError:  # pragma: no cover - the child cannot see the test tree
        pass
    else:
        offline_guard.install()
