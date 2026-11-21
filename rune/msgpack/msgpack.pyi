from collections.abc import Callable, Iterable
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
    ext_hook: Callable[[int, memoryview], Any] | None = None,
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
        ext_hook: Callable[[int, memoryview], Any] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        type: type[T] | Any,
        *,
        dec_hook: Callable[[Any, Any], Any] | None = None,
        strict: bool = True,
        ext_hook: Callable[[int, memoryview], Any] | None = None,
    ) -> None: ...
    @property
    def type(self) -> Any: ...
    @property
    def ext_hook(self) -> Callable[[int, memoryview], Any] | None: ...
    @property
    def strict(self) -> bool: ...
    def __class_getitem__(cls, item: Any) -> Any: ...
    def decode(self, buf: bytes, /) -> T: ...
    def decode_lines(self, buf: bytes, /) -> list[T]: ...

class Encoder:
    def __init__(
        self,
        *,
        enc_hook: Callable[[Any], Any] | None = None,
        decimal_format: Literal["string", "number"] | Callable[[Any], Any] | None = None,
        order: Literal["deterministic", "sorted"] | None = None,
        uuid_format: Literal["canonical", "hex", "bytes"] | None = None,
    ) -> None: ...
    @property
    def enc_hook(self) -> Callable[[Any], Any] | None: ...
    @property
    def decimal_format(self) -> Literal["string", "number"] | Callable[[Any], Any]: ...
    @property
    def order(self) -> Literal["deterministic", "sorted"] | None: ...
    @property
    def uuid_format(self) -> Literal["canonical", "hex", "bytes"]: ...
    def encode(self, obj: Any, /) -> bytes: ...
    def encode_lines(self, items: Iterable[Any]) -> bytes: ...
    def encode_into(self, obj: Any, buffer: bytearray, offset: int = 0, /) -> None: ...

class Ext:
    def __init__(self, code: int, data: bytes | bytearray | memoryview, /) -> None: ...
    @property
    def code(self) -> int: ...
    @property
    def data(self) -> bytes | bytearray | memoryview: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __reduce__(self) -> tuple[Any, ...]: ...
