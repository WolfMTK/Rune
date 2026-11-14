import builtins
from collections.abc import Callable, Iterable
from typing import Any, TypeVar, overload

_T = TypeVar("_T")

# exceptions.rs
class DecodeError:
    pass

class ValidationError:
    pass

# common/raw.rs
class Raw:
    def __init__(self, data: str | bytes | bytearray | memoryview = ..., /) -> None: ...
    def __bytes__(self) -> bytes: ...
    def __len__(self) -> int: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __reduce__(self) -> tuple[Any, ...]: ...
    def copy(self) -> Raw: ...

# lib.rs
_struct_lookup_cache: dict[frozenset[type], dict[str | int, type]]

# common/struct_runtime.rs
class _StructMixin: ...

def _struct_finalize(
    cls: type,
    fields: tuple[str, ...],
    encode_fields: tuple[str, ...],
    defaults: tuple[Any, ...],
    keyword_only: int,
    config: Any,
) -> None: ...
def _struct_values(object: Any) -> tuple[Any, ...]: ...
def _struct_copy(object: _T) -> _T: ...
def _struct_replace(object: _T, changes: dict[str, Any]) -> _T: ...
def force_setattr(object: Any, name: str, value: Any) -> None: ...

# common/convert.rs
@overload
def convert(
    obj: Any,
    type: builtins.type[_T],
    *,
    strict: bool = True,
    from_attributes: bool = False,
    dec_hook: Callable[[Any, Any], Any] | None = None,
    builtin_types: Iterable[builtins.type] | None = None,
    str_keys: bool = False,
) -> _T: ...
@overload
def convert(
    obj: Any,
    type: Any,
    *,
    strict: bool = True,
    from_attributes: bool = False,
    dec_hook: Callable[[Any, Any], Any] | None = None,
    builtin_types: Iterable[builtins.type] | None = None,
    str_keys: bool = False,
) -> Any: ...
