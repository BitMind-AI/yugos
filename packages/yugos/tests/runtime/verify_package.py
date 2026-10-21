"""Verify that a staging test imports exactly the supplied installed wheel."""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import yugos

assert sys.flags.isolated
wheel_path = Path(sys.argv[1])
installed = Path(yugos.__file__).parent
assert "site-packages" in installed.parts
with zipfile.ZipFile(wheel_path) as wheel:
    expected = {
        name.removeprefix("yugos/"): wheel.read(name)
        for name in wheel.namelist()
        if name.startswith("yugos/") and not name.endswith("/")
    }
actual = {
    path.relative_to(installed).as_posix(): path.read_bytes()
    for path in installed.rglob("*")
    if path.is_file() and "__pycache__" not in path.parts
}
assert expected == actual
assert not any(name.split(".")[0] in {"mindr", "brr"} for name in sys.modules)
Path("evidence/package.json").write_text(
    json.dumps(
        {
            "package_path": str(installed),
            "python": sys.version,
            "wheel_sha256": hashlib.sha256(wheel_path.read_bytes()).hexdigest(),
            "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in actual.items()},
            "mindr_or_brr_imported": False,
        },
        indent=2,
    )
)
