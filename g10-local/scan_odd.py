"""Scan a starcoderdata python shard for odd items and write redacted evidence.

Evidence stays in G10-Local-Run-ERA-V5/g10-local/evidence. Secret and email values are not stored.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
SHARD = ROOT / "data" / "python-train-00000-of-00059.parquet"
EVIDENCE = ROOT / "evidence"
MAX_DOCS = 4000
MAX_PER_CODE = 3

INVISIBLE = {
    "\u200b": "ZWSP",
    "\u200c": "ZWNJ",
    "\u200d": "ZWJ",
    "\ufeff": "BOM",
    "\u00ad": "SHY",
    "\u202a": "LRE",
    "\u202b": "RLE",
    "\u202c": "PDF",
    "\u202d": "LRO",
    "\u202e": "RLO",
    "\u2066": "LRI",
    "\u2067": "RLI",
    "\u2068": "FSI",
    "\u2069": "PDI",
    "\ufffd": "U+FFFD",
}
GHOSTS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "[INST]",
    "[/INST]",
    "<s>",
    "</s>",
    "### Assistant:",
    "<think>",
    "</think>",
)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SECRET = (
    ("aws key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("api key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("github token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
)
BLOB = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{300,}={0,2}")


def _snippet(text: str, offset: int, redact: tuple[int, int] | None) -> str:
    start = max(0, offset - 80)
    end = min(len(text), offset + 160)
    chunk = text[start:end]
    if redact is not None:
        left, right = redact
        left = max(left, start) - start
        right = min(right, end) - start
        if right > left >= 0:
            chunk = chunk[:left] + "⟦redacted⟧" + chunk[right:]
    chunk = chunk.replace("\r", "\\r")
    return chunk


def scan_text(text: str) -> list[dict]:
    found: list[dict] = []
    for ch, name in INVISIBLE.items():
        offset = text.find(ch)
        if offset >= 0:
            code = "A4" if name == "U+FFFD" else "A1"
            found.append(
                {
                    "code": code,
                    "evidence": f"{name} at offset {offset}",
                    "offset": offset,
                    "snippet": _snippet(text, offset, None),
                }
            )
            break
    for index, ch in enumerate(text):
        o = ord(ch)
        if ch in "\t\n\r":
            continue
        if o < 32 or 127 <= o <= 159:
            found.append(
                {
                    "code": "A2",
                    "evidence": f"control U+{o:04X} at offset {index}",
                    "offset": index,
                    "snippet": _snippet(text, index, None),
                }
            )
            break
    for ghost in GHOSTS:
        offset = text.find(ghost)
        if offset >= 0:
            found.append(
                {
                    "code": "B3",
                    "evidence": f"ghost marker {ghost} at offset {offset}",
                    "offset": offset,
                    "snippet": _snippet(text, offset, None),
                }
            )
            break
    lines = text.splitlines()
    if lines and max(len(line) for line in lines) >= 2000:
        longest = max(lines, key=len)
        offset = text.find(longest)
        found.append(
            {
                "code": "E1",
                "evidence": f"line of {len(longest)} chars at offset {offset}",
                "offset": offset,
                "snippet": _snippet(text, offset, None),
            }
        )
    blob = BLOB.search(text)
    if blob:
        found.append(
            {
                "code": "E2",
                "evidence": f"base64-like run of {blob.end() - blob.start()} chars at offset {blob.start()}",
                "offset": blob.start(),
                "snippet": _snippet(text, blob.start(), None),
            }
        )
    for kind, pattern in SECRET:
        match = pattern.search(text)
        if match:
            found.append(
                {
                    "code": "E3",
                    "evidence": f"{kind} at offset {match.start()}",
                    "offset": match.start(),
                    "snippet": _snippet(text, match.start(), (match.start(), match.end())),
                }
            )
            break
    mail = EMAIL.search(text)
    if mail and not mail.group(0).endswith("@users.noreply.github.com"):
        found.append(
            {
                "code": "F1",
                "evidence": f"email at offset {mail.start()}",
                "offset": mail.start(),
                "snippet": _snippet(text, mail.start(), (mail.start(), mail.end())),
            }
        )
    return found


def _field(row: dict, *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value:
            return str(value)
    return ""


def main() -> None:
    if not SHARD.is_file():
        raise SystemExit(f"missing shard {SHARD}")
    parquet = pq.ParquetFile(SHARD)
    print("columns", parquet.schema_arrow.names)
    kept: dict[str, list[dict]] = {}
    seen = 0
    for batch in parquet.iter_batches(batch_size=64):
        rows = batch.to_pylist()
        for row in rows:
            if seen >= MAX_DOCS:
                break
            text = _field(row, "content", "text")
            if not text:
                seen += 1
                continue
            hits = scan_text(text)
            doc_id = _field(row, "id", "hexsha", "doc_id") or f"row-{seen}"
            repo = _field(row, "max_stars_repo_name", "repo_name")
            path = _field(row, "max_stars_repo_path", "path")
            for hit in hits:
                bucket = kept.setdefault(hit["code"], [])
                if len(bucket) >= MAX_PER_CODE:
                    continue
                bucket.append(
                    {
                        "doc_id": doc_id,
                        "repo": repo,
                        "path": path,
                        "chars": len(text),
                        **hit,
                    }
                )
            seen += 1
        if seen >= MAX_DOCS:
            break
        if len(kept) >= 7 and all(len(items) >= MAX_PER_CODE for items in kept.values()):
            break

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    report = {
        "source": "bigcode/starcoderdata",
        "shard": SHARD.name,
        "docs_scanned": seen,
        "note": "Probe of the python shard. This is not the lab sample pack. E3 and F1 values are redacted.",
        "items": [item for items in kept.values() for item in items],
    }
    (EVIDENCE / "odd-items.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    viewer_rows = []
    for item in report["items"]:
        viewer_rows.append(
            {
                "doc_id": item["doc_id"],
                "source": "bigcode/starcoderdata",
                "lang_label": "Python",
                "url": item["path"],
                "text": (
                    f"{item['code']}: {item['evidence']}\n"
                    f"repo: {item['repo']}\n"
                    f"path: {item['path']}\n\n"
                    f"{item['snippet']}"
                ),
            }
        )
    if viewer_rows:
        pq.write_table(pa.Table.from_pylist(viewer_rows), EVIDENCE / "odd-sample.parquet")
    counts = {code: len(items) for code, items in sorted(kept.items())}
    print(json.dumps({"docs_scanned": seen, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
