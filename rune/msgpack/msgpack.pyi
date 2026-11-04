from collections.abc import Callable
from typing import Any, Generic, Literal, TypeVar, overload

T = TypeVar("T")

def encode(
    obj: Any,
    /,
    *,
    enc_hook: Callable[[Any], Any] | None = None,
    order: Literal["deterministic", "sorted"] | None = None,
) -> bytes: ...
def decode(
    buf: bytes,
    /,
    *,
    type: Any = ...,
    dec_hook: Callable[[Any, Any], Any] | None = None,
    strict: bool = True,
) -> Any: ...

class Decoder(Generic[T]):
    dec_hook: Callable[[Any, Any], Any] | None

    @overload
    def __init__(
        self: Decoder[Any],
        type: Any = ...,
        *,
        dec_hook: Callable[[Any, Any], Any] | None = None,
        strict: bool = True,
    ) -> None: ...
    @overload
    def __init__(
        self,
        type: type[T] | Any,
        *,
        dec_hook: Callable[[Any, Any], Any] | None = None,
        strict: bool = True,
    ) -> None: ...
    @property
    def strict(self) -> bool: ...
    def __class_getitem__(cls, item: Any) -> Any: ...
    def decode(self, buf: bytes, /) -> T: ...

class Encoder:
    def __init__(
        self,
        *,
        enc_hook: Callable[[Any], Any] | None = None,
        order: Literal["deterministic", "sorted"] | None = None,
        uuid_format: Literal["canonical", "hex", "bytes"] | None = None,
    ) -> None: ...
    @property
    def enc_hook(self) -> Callable[[Any], Any] | None: ...
    @property
    def order(self) -> Literal["deterministic", "sorted"] | None: ...
    @property
    def uuid_format(self) -> Literal["canonical", "hex", "bytes"]: ...
    def encode(self, obj: Any) -> bytes: ...

class Ext:
    def __init__(self, code: int, data: bytes | bytearray | memoryview, /) -> None: ...
    @property
    def code(self) -> int: ...
    @property
    def data(self) -> bytes: ...
