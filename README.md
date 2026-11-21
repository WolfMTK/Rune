# About the Project

Rune is an experimental project that implements the full functionality of the [msgspec](https://github.com/msgspec/msgspec) library and is written in Rust.

```
WARNING: This project is not currently intended for use in production environments,
as some algorithms need to be reviewed and the code needs to be refactored.
```

# How to Install?

```To install the project, you currently need to clone it, as the library is not available on PyPI.```

1. Build and install the project: `uv run maturin develop --release`

2. Install the dependencies: `uv sync`

# Usage Example (Taken from msgspec)

**Define** your message schemas using standard Python type annotations.

```
>>> import rune

>>> class User(rune.Struct):
...     """A new type describing a User"""
...     name: str
...     groups: set[str] = set()
...     email: str | None = None
```

**Encode** messages as JSON or one of the many other supported protocols.

```
>>> alice = User("alice", groups={"admin", "engineering"})

>>> alice
User(name='alice', groups={"admin", "engineering"}, email=None)

>>> msg = rune.json.encode(alice)

>>> msg
b'{"name":"alice","groups":["admin","engineering"],"email":null}'
```

**Decode** messages back into Python objects with optional schema validation.

```
>>> rune.json.decode(msg, type=User)
User(name='alice', groups={"admin", "engineering"}, email=None)

>>> rune.json.decode(b'{"name":"bob","groups":[123]}', type=User)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
rune.exceptions.ValidationError: Expected `str`, got `int` - at `$.groups[0]`
```

# License

MIT. See the [License file](LICENSE)
