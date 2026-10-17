import inspect
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

PY312 = sys.version_info[:2] >= (3, 12)


@contextmanager
def max_call_depth(n) -> Iterator[Any]:
    cur_depth = len(inspect.stack(0))
    orig = sys.getrecursionlimit()
    try:
        for i in range(64):
            try:
                sys.setrecursionlimit(cur_depth + i + n)
                break
            except RecursionError:
                pass
        else:
            raise ValueError("Failed to set low recursion limit, something is wrong here")
        yield
    finally:
        sys.setrecursionlimit(orig)
