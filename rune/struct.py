import inspect
from typing import Any, ClassVar


class Struct:
    __struct_fields__: ClassVar[tuple[tuple[str, Any], ...]] = ()
    __struct_array_like__: ClassVar[bool] = False
    __struct_tag__: ClassVar[str | int | None] = None
    __struct_tag_field__: ClassVar[str] = "type"

    def __init_subclass__(
        cls,
        tag: str | int | bool | None = None,
        tag_field: str = "type",
        array_like: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        hints = inspect.get_annotations(cls, eval_str=True)
        cls.__struct_fields__ = tuple(hints.items())
        cls.__struct_array_like__ = array_like
        cls.__struct_tag_field__ = tag_field
        if tag is True:
            cls.__struct_tag__ = cls.__name__
        elif tag is None or tag is False:
            cls.__struct_tag__ = None
        elif isinstance(tag, (str, int)):
            cls.__struct_tag__ = tag
        else:
            raise TypeError(f"Struct tag must be a str, an int, True or False, got {tag!r}")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        fields = self.__struct_fields__
        if len(args) > len(fields):
            raise TypeError(
                f"{type(self).__name__}() takes at most {len(fields)} positional "
                f"argument(s) but {len(args)} were given"
            )

        for (name, _), value in zip(fields, args, strict=False):
            if name in kwargs:
                raise TypeError(f"Field `{name}` given both positionally and by name")
            kwargs[name] = value

        for name, _ in fields:
            if name in kwargs:
                value = kwargs.pop(name)
            elif hasattr(type(self), name):
                value = getattr(type(self), name)
            else:
                raise TypeError(f"Missing required field `{name}`")
            setattr(self, name, value)

        if kwargs:
            unknown = ", ".join(f"`{k}`" for k in kwargs)
            raise TypeError(f"Unexpected field(s): {unknown}")

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return all(
            getattr(self, name) == getattr(other, name) for name, _ in self.__struct_fields__
        )

    def __repr__(self) -> str:
        fields = ", ".join(f"{name}={getattr(self, name)!r}" for name, _ in self.__struct_fields__)
        return f"{type(self).__name__}({fields})"
