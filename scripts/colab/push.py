"""Copy what a fresh Colab runtime needs to /content/ntl_boot on it: this repo's working tree
(repo.tgz), the Kaggle credential from ~/.kaggle, and boot.sh / ensure.sh.

    python scripts/colab/push.py <runtime url> <runtime proxy token>

The url and token are the runtime's ``connectionInfo`` (Colab API ``get_runtime``). Standard
library only. The credential goes straight from the file to the runtime; it is never printed.
"""

import base64
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SKIP = ("notebooks/", "reports/", "docs/", "references/")  # not needed to train
DEST = "content/ntl_boot"


def put(url: str, token: str, path: str, body: dict) -> None:
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/contents/{path}",
        data=json.dumps(body).encode(),
        method="PUT",
        headers={"X-Colab-Runtime-Proxy-Token": token, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
        assert r.status in (200, 201), r.status


def put_file(url: str, token: str, name: str, data: bytes) -> None:
    body = {"type": "file", "format": "base64", "content": base64.b64encode(data).decode()}
    put(url, token, f"{DEST}/{name}", body)
    print(f"sent {name} ({len(data)} bytes)")


def repo_tar() -> bytes:
    """The tracked and new files of the working tree, as they are now (not the last commit)."""
    names = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],  # noqa: S607
        cwd=ROOT, capture_output=True, check=True,
    ).stdout.decode().split("\0")  # fmt: skip
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in names:
            f = ROOT / name
            if not name or name.startswith(SKIP) or not f.is_file():
                continue
            data = f.read_bytes()
            if f.suffix in (".py", ".sh", ".toml", ".lock", ".md", ".json"):
                data = data.replace(b"\r\n", b"\n")
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def main(url: str, token: str) -> None:
    put(url, token, DEST, {"type": "directory"})
    creds = [f for f in (Path.home() / ".kaggle").glob("*") if f.name in ("kaggle.json", "access_token")]
    assert creds, "no kaggle.json or access_token in ~/.kaggle"
    for f in creds:
        put_file(url, token, f.name, f.read_bytes())
    put_file(url, token, "repo.tgz", repo_tar())
    for name in ("boot.sh", "ensure.sh"):
        put_file(url, token, name, (ROOT / "scripts/colab" / name).read_bytes().replace(b"\r\n", b"\n"))
    print("pushed")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
