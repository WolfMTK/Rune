"""
This benchmark compares the installed library size between Rune, msgspec and
pydantic in a Python 3.10 x86 environment.
"""

import importlib
import importlib.metadata
import io
import zipfile
from pathlib import Path

import requests


def get_latest_noarch_wheel_size(library):
    """Get the total uncompressed size of the latest noarch wheel"""
    resp = requests.get(f"https://pypi.org/pypi/{library}/json").json()
    version = resp["info"]["version"]
    files = {}
    for file_info in resp["releases"][version]:
        name = file_info["filename"]
        url = file_info["url"]
        if name.endswith(".whl"):
            files[name] = url
    if len(files) != 1:
        raise ValueError(f"Expected to find only 1 matching file for {library}, got {list(files)}")

    url = next(iter(files.values()))

    resp = requests.get(url)
    fil = io.BytesIO(resp.content)
    zfil = zipfile.ZipFile(fil)
    size = sum(f.file_size for f in zfil.filelist)
    return version, size


def get_latest_manylinux_wheel_size(library):
    """Get the total uncompressed size of the latest Python 3.10 manylinux
    x86_64 wheel for the library"""
    resp = requests.get(f"https://pypi.org/pypi/{library}/json").json()
    version = resp["info"]["version"]
    files = {}
    for file_info in resp["releases"][version]:
        name = file_info["filename"]
        url = file_info["url"]
        if "310" in name and "manylinux_2_17_x86_64" in name and "pp73" not in name:
            files[name] = url
    if len(files) != 1:
        raise ValueError(f"Expected to find only 1 matching file for {library}, got {list(files)}")

    url = next(iter(files.values()))

    resp = requests.get(url)
    fil = io.BytesIO(resp.content)
    zfil = zipfile.ZipFile(fil)
    size = sum(f.file_size for f in zfil.filelist)
    return version, size


def get_installed_library_size(library):
    """Get the total size of an installed library"""
    dist = importlib.metadata.distribution(library)
    package_path = Path(next(iter(importlib.import_module(library).__path__)))
    package_size = sum(
        path.stat().st_size
        for path in package_path.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    metadata_size = sum(dist.locate_file(path).stat().st_size for path in dist.files)
    return dist.version, package_size + metadata_size


def main():
    rune_version, rune_size = get_installed_library_size("rune")
    msgspec_version, msgspec_size = get_latest_manylinux_wheel_size("msgspec")
    pydantic_version, pydantic_size = get_latest_noarch_wheel_size("pydantic")
    _, pydantic_core_size = get_latest_manylinux_wheel_size("pydantic-core")
    _, typing_extensions_size = get_latest_noarch_wheel_size("typing-extensions")
    _, annotated_types_size = get_latest_noarch_wheel_size("annotated-types")

    data = [
        ("rune", rune_version, rune_size),
        ("msgspec", msgspec_version, msgspec_size),
        (
            "pydantic",
            pydantic_version,
            pydantic_size + pydantic_core_size + typing_extensions_size + annotated_types_size,
        ),
    ]
    data.sort(key=lambda x: x[2])
    msgspec_size = next(s for l, _, s in data if l == "msgspec")  # noqa

    columns = ("", "version", "size (MiB)", "vs. msgspec")
    rows = [
        (
            f"**{lib}**",
            version,
            f"{size / (1024 * 1024):.2f}",
            f"{size / msgspec_size:.2f}x",
        )
        for lib, version, size in data
    ]

    widths = tuple(
        max(max(map(len, x)), len(c))
        for x, c in zip(zip(*rows, strict=False), columns, strict=False)
    )
    row_template = ("|" + (" %%-%ds |" * len(columns))) % widths
    header = row_template % tuple(columns)
    bar_underline = "+{}+".format("+".join("=" * (w + 2) for w in widths))
    bar = "+{}+".format("+".join("-" * (w + 2) for w in widths))
    parts = [bar, header, bar_underline]
    for r in rows:
        parts.append(row_template % r)
        parts.append(bar)
    print("\n".join(parts))


if __name__ == "__main__":
    main()
