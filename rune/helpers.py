import dataclasses
from typing import get_type_hints

from rune.struct import Fields


def plain_class_fields(cls: type) -> Fields:
    if dataclasses.is_dataclass(cls):
        declared = tuple((field.name, field.type) for field in dataclasses.fields(cls))
    else:
        declared = tuple((attribute.name, attribute.type) for attribute in cls.__attrs_attrs__)
    try:
        hints = get_type_hints(cls, include_extras=True)
    except NameError:
        return declared
    return tuple((name, hints.get(name, tp)) for name, tp in declared)
