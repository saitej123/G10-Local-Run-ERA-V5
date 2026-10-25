"""Download one python shard of bigcode/starcoderdata into G10-Local-Run-ERA-V5/g10-local/data.

The Hugging Face token is read from the project .env and is not printed.
The file stays outside era-v5-lab.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_env import hf_token

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "python-train-00000-of-00059.parquet"
RESOLVE = (
    "https://huggingface.co/datasets/bigcode/starcoderdata/resolve/main/"
    "python/train-00000-of-00059.parquet"
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _signed_url() -> str:
    opener = urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(
        RESOLVE,
        headers={"Authorization": f"Bearer {hf_token()}", "User-Agent": "era-v5-g10-local"},
    )
    try:
        opener.open(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code in {301, 302, 303, 307, 308}:
            return exc.headers["Location"]
        detail = exc.read()[:300].decode("utf-8", "replace")
        raise SystemExit(f"resolve failed: {exc.code} {detail}") from exc
    raise SystemExit("resolve did not redirect")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.stat().st_size > 1_000_000:
        print(f"already have {OUT} ({OUT.stat().st_size} bytes)")
        return
    signed = _signed_url()
    request = urllib.request.Request(signed, headers={"User-Agent": "era-v5-g10-local"})
    with urllib.request.urlopen(request, timeout=120) as response, OUT.open("wb") as handle:
        total = 0
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            total += len(chunk)
            print(f"downloaded {total // (1024 * 1024)} MiB", flush=True)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
