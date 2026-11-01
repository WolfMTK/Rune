import re
from functools import partial
from typing import Any

Number = int | float

_NUMERIC_FIELDS = ("gt", "ge", "lt", "le", "multiple_of")
_LENGTH_FIELDS = ("min_length", "max_length")
_FIELDS = (*_NUMERIC_FIELDS, "pattern", *_LENGTH_FIELDS, "tz")


class Meta:
    __slots__ = _FIELDS

    gt: Number | None
    ge: Number | None
    lt: Number | None
    le: Number | None
    multiple_of: Number | None
    pattern: str | None
    min_length: int | None
    max_length: int | None
    tz: bool | None

    def __init__(
        self,
        *,
        gt: Number | None = None,
        ge: Number | None = None,
        lt: Number | None = None,
        le: Number | None = None,
        multiple_of: Number | None = None,
        pattern: str | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        tz: bool | None = None,
    ) -> None:
        values = (gt, ge, lt, le, multiple_of, pattern, min_length, max_length, tz)
        for name, value in zip(_FIELDS, values, strict=True):
            object.__setattr__(self, name, value)
        self._validate()

    def _validate(self) -> None:
        for name in _NUMERIC_FIELDS:
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Number)):
                raise TypeError(f"`{name}` must be an int or float, got {type(value).__name__}")
        if self.gt is not None and self.ge is not None:
            raise TypeError("Cannot set both `gt` and `ge`")
        if self.lt is not None and self.le is not None:
            raise TypeError("Cannot set both `lt` and `le`")
        if self.multiple_of is not None and self.multiple_of <= 0:
            raise ValueError("`multiple_of` must be > 0")
        if self.pattern is not None:
            if not isinstance(self.pattern, str):
                raise TypeError(f"`pattern` must be a str, got {type(self.pattern).__name__}")
            re.compile(self.pattern)
        for name in _LENGTH_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"`{name}` must be an int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"`{name}` must be >= 0")
        if self.tz is not None and not isinstance(self.tz, bool):
            raise TypeError(f"`tz` must be a bool, got {type(self.tz).__name__}")

    def _specified_fields(self) -> dict[str, Any]:
        values = ((name, getattr(self, name)) for name in _FIELDS)
        return {name: value for name, value in values if value is not None}

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"cannot assign to field '{name}'")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"cannot delete field '{name}'")

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._specified_fields() == other._specified_fields()

    def __hash__(self) -> int:
        return hash(tuple(self._specified_fields().items()))

    def __repr__(self) -> str:
        fields = ", ".join(f"{name}={value!r}" for name, value in self._specified_fields().items())
        return f"Meta({fields})"

    def __reduce__(self) -> tuple[Any, ...]:
        return partial(Meta, **self._specified_fields()), ()
