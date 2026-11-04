import math
import re
import sys
from functools import partial
from typing import Any

Number = int | float

_NUMERIC_FIELDS = ("gt", "ge", "lt", "le", "multiple_of")
_LENGTH_FIELDS = ("min_length", "max_length")
_NON_NUMERIC_CONSTRAINT_FIELDS = ("pattern", *_LENGTH_FIELDS)
_FIELDS = (
    *_NUMERIC_FIELDS,
    *_NON_NUMERIC_CONSTRAINT_FIELDS,
    "tz",
    "title",
    "description",
    "examples",
    "extra_json_schema",
    "extra",
)
_TYPED_FIELDS: dict[str, type] = {
    "pattern": str,
    "tz": bool,
    "title": str,
    "description": str,
    "examples": list,
    "extra_json_schema": dict,
    "extra": dict,
}
_UNHASHABLE_FIELDS = frozenset({"examples", "extra_json_schema", "extra"})
_SSIZE_MIN, _SSIZE_MAX = -sys.maxsize - 1, sys.maxsize


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
    title: str | None
    description: str | None
    examples: list[Any] | None
    extra_json_schema: dict[str, Any] | None
    extra: dict[str, Any] | None

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
        title: str | None = None,
        description: str | None = None,
        examples: list[Any] | None = None,
        extra_json_schema: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        values = (
            gt,
            ge,
            lt,
            le,
            multiple_of,
            pattern,
            min_length,
            max_length,
            tz,
            title,
            description,
            examples,
            extra_json_schema,
            extra,
        )
        for name, value in zip(_FIELDS, values, strict=True):
            object.__setattr__(self, name, value)
        self._validate()

    def _validate(self) -> None:
        for name in _NUMERIC_FIELDS:
            _check_finite_number(name, getattr(self, name))
        for name in _LENGTH_FIELDS:
            _check_length(name, getattr(self, name))
        for name, expected in _TYPED_FIELDS.items():
            value = getattr(self, name)
            if value is not None and not isinstance(value, expected):
                raise TypeError(
                    f"`{name}` must be a {expected.__name__}, got {type(value).__name__}"
                )
        if self.multiple_of is not None and self.multiple_of <= 0:
            raise ValueError("`multiple_of` must be > 0")
        if self.pattern is not None:
            re.compile(self.pattern)
        if self.gt is not None and self.ge is not None:
            raise ValueError("Cannot specify both `gt` and `ge`")
        if self.lt is not None and self.le is not None:
            raise ValueError("Cannot specify both `lt` and `le`")
        if self._any_set(_NUMERIC_FIELDS) and self._any_set(_NON_NUMERIC_CONSTRAINT_FIELDS):
            raise ValueError(
                "Cannot mix numeric constraints (gt, lt, ge, le, multiple_of) "
                "with non-numeric constraints (pattern, min_length, max_length)"
            )

    def _any_set(self, names: tuple[str, ...]) -> bool:
        return any(getattr(self, name) is not None for name in names)

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
        fields = self._specified_fields().items()
        return hash(tuple(item for item in fields if item[0] not in _UNHASHABLE_FIELDS))

    def __repr__(self) -> str:
        fields = ", ".join(f"{name}={value!r}" for name, value in self._specified_fields().items())
        return f"rune.Meta({fields})"

    def __rich_repr__(self) -> list[tuple[str, Any]]:
        return list(self._specified_fields().items())

    def __reduce__(self) -> tuple[Any, ...]:
        return partial(Meta, **self._specified_fields()), ()


def _check_finite_number(name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, Number):
        raise TypeError(f"`{name}` must be an int or float, got {type(value).__name__}")
    try:
        as_float = float(value)
    except OverflowError:
        raise ValueError(f"`{name}` is out of range") from None
    if not math.isfinite(as_float):
        raise ValueError(f"`{name}` must be finite")


def _check_length(name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an int, got {type(value).__name__}")
    if not _SSIZE_MIN <= value <= _SSIZE_MAX:
        raise ValueError(f"`{name}` is out of range")
    if value < 0:
        raise ValueError(f"`{name}` must be >= 0, got {value}")
