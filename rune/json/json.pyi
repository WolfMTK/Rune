from collections.abc import Callable
from typing import Any, Generic, TypeVar, overload

T = TypeVar("T")

def encode(obj: Any, /) -> bytes: ...
def decode(
    buf: bytes,
    /,
    *,
    type: Any = None,
    dec_hook: Callable[[Any, Any], Any] | None = None,
) -> Any: ...

class Decoder(Generic[T]):
    dec_hook: Callable[[Any, Any], Any] | None

    @overload
    def __init__(
        self: Decoder[Any],
        type: None = None,
        *,
        dec_hook: Callable[[Any, Any], Any] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        type: type[T] | Any,
        *,
        dec_hook: Callable[[Any, Any], Any] | None = None,
    ) -> None: ...
    def __class_getitem__(cls, item: Any) -> Any: ...
    def decode(self, buf: bytes, /) -> T: ...

class Encoder:
    def __init__(self, *, enc_hook: Callable[[Any], Any] | None = None) -> None: ...
    @property
    def enc_hook(self) -> Callable[[Any], Any] | None: ...
    def encode(self, obj: Any) -> bytes: ...
