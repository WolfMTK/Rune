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
    def __init__(self, data: str | bytes | bytearray | memoryview, /) -> None: ...
    def __bytes__(self) -> bytes: ...
    def __len__(self) -> int: ...

# lib.rs
_struct_lookup_cache: dict[frozenset[type], dict[str | int, type]]

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
