from __future__ import annotations

import datetime
import enum

import rune


class Permissions(enum.Enum):
    READ = "READ"
    WRITE = "WRITE"
    READ_WRITE = "READ_WRITE"


class File(rune.Struct, kw_only=True, omit_defaults=True, tag="file"):
    name: str
    created_by: str
    created_at: datetime.datetime
    updated_by: str | None = None
    updated_at: datetime.datetime | None = None
    nbytes: int
    permissions: Permissions


class Directory(rune.Struct, kw_only=True, omit_defaults=True, tag="directory"):
    name: str
    created_by: str
    created_at: datetime.datetime
    updated_by: str | None = None
    updated_at: datetime.datetime | None = None
    contents: list[File | Directory]


enc = rune.json.Encoder()
dec = rune.json.Decoder(Directory)

label = "rune"
encode = enc.encode
decode = dec.decode
