"""Build the 200-document clean Python gold set and run the harm test for every medicine.

Usage:
    .venv/bin/python G10-Local-Run-ERA-V5/g10-local/run_harm_test.py

Outputs:
    G10-Local-Run-ERA-V5/g10-local/evidence/gold-code-python-200.parquet
    G10-Local-Run-ERA-V5/g10-local/evidence/harm-test-report.json
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

LOCAL_DIR = Path(__file__).resolve().parent
WORKSPACE = LOCAL_DIR.parents[1]
_FL_PATH = LOCAL_DIR / "fill_labels.py"
_fl_spec = importlib.util.spec_from_file_location("fill_labels", _FL_PATH)
assert _fl_spec is not None and _fl_spec.loader is not None
fl = importlib.util.module_from_spec(_fl_spec)
_fl_spec.loader.exec_module(fl)

SHARD = LOCAL_DIR / "data" / "python-train-00000-of-00059.parquet"
GOLD_OUT = LOCAL_DIR / "evidence" / "gold-code-python-200.parquet"
REPORT_OUT = LOCAL_DIR / "evidence" / "harm-test-report.json"
GOLD_TARGET = 200


def build_gold_set() -> list[dict]:
    pf = pq.ParquetFile(SHARD)
    gold: list[dict] = []
    for batch in pf.iter_batches(batch_size=64, columns=["id", "max_stars_repo_name", "max_stars_repo_path", "content"]):
        for row in batch.to_pylist():
            text = (row.get("content") or "").replace("\r\n", "\n").replace("\r", "\n")
            if not text:
                continue
            hits = [h for h in fl.find_hits(text) if h["code"] != "E4"]
            if not hits:
                gold.append(
                    {
                        "doc_id": str(row["id"]),
                        "source": "bigcode/starcoderdata",
                        "lang_label": "Python",
                        "url": f"{row.get('max_stars_repo_name', '')} {row.get('max_stars_repo_path', '')}".strip(),
                        "text": text,
                    }
                )
                if len(gold) >= GOLD_TARGET:
                    return gold
    return gold


def main() -> None:
    gold = build_gold_set()
    GOLD_OUT.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(gold), GOLD_OUT)

    texts = [g["text"] for g in gold]
    raw_counts = fl.harm_counts(texts)
    total = len(texts)
    results = {}
    for code, (changed, dropped) in raw_counts.items():
        pct = ((changed + dropped) / total * 100.0) if total else 0.0
        results[code] = {
            "gold_docs": total,
            "changed": changed,
            "dropped": dropped,
            "harm_pct": round(pct, 3),
            "passes_1pct_gate": pct <= 1.0,
        }

    report = {
        "pack": "code-python",
        "source": "bigcode/starcoderdata",
        "gold_set_path": str(GOLD_OUT.relative_to(WORKSPACE)),
        "gold_docs": total,
        "threshold_pct": 1.0,
        "all_pass": all(r["passes_1pct_gate"] for r in results.values()),
        "medicines": results,
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
