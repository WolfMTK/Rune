import inspect
import sys
import textwrap
import types
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

PY312: bool = sys.version_info[:2] >= (3, 12)
PY311: bool = sys.version_info[:2] >= (3, 11)

py315_or_later_only = pytest.mark.skipif(
    sys.version_info < (3, 15), reason="frozendict was added in 3.15"
)

py312_plus = pytest.mark.skipif(not PY312, reason="3.12+ only")


@contextmanager
def max_call_depth(n: int) -> Iterator[None]:
    cur_depth: int = len(inspect.stack(0))
    orig: int = sys.getrecursionlimit()
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


@contextmanager
def temp_module(code: str) -> Iterator[types.ModuleType]:
    """Mutually recursive struct types defined inside functions don't work (and
    probably never will). To avoid populating a bunch of test structs in the
    top level of this module, we instead create a temporary module per test to
    exec whatever is needed for that test"""
    code = textwrap.dedent(code)
    name: str = f"temp_{uuid.uuid4().hex}"
    mod: types.ModuleType = types.ModuleType(name)
    sys.modules[name] = mod
    try:
        exec(code, mod.__dict__)
        yield mod
    finally:
        sys.modules.pop(name, None)
