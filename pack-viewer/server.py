"""Local viewer for a diagnosis pack. This file is outside era-v5-lab.

Usage:
    python G10-Local-Run-ERA-V5/pack-viewer/server.py packs/code-python.parquet
    python G10-Local-Run-ERA-V5/pack-viewer/server.py G10-Local-Run-ERA-V5/g10-local/evidence/labeled-view.parquet --port 8765

The Parquet file stays out of the lab repository. The page only reads it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pyarrow.parquet as pq

_VP_PATH = Path(__file__).resolve().parents[2] / "era-v5-lab" / "diagnosis" / "tools" / "view_pack.py"
_vp_spec = importlib.util.spec_from_file_location("view_pack", _VP_PATH)
assert _vp_spec is not None and _vp_spec.loader is not None
view_pack = importlib.util.module_from_spec(_vp_spec)
_vp_spec.loader.exec_module(view_pack)

VISIBLE: dict[str, str] = dict(view_pack.VISIBLE)
show = view_pack.show


CONTROL_NAMES = {
    0x00: "NULL", 0x01: "START OF HEADING", 0x02: "START OF TEXT", 0x03: "END OF TEXT",
    0x04: "END OF TRANSMISSION", 0x05: "ENQUIRY", 0x06: "ACKNOWLEDGE", 0x07: "BELL",
    0x08: "BACKSPACE", 0x09: "CHARACTER TABULATION", 0x0A: "LINE FEED", 0x0B: "LINE TABULATION",
    0x0C: "FORM FEED", 0x0D: "CARRIAGE RETURN", 0x0E: "SHIFT OUT", 0x0F: "SHIFT IN",
    0x10: "DATA LINK ESCAPE", 0x11: "DEVICE CONTROL ONE", 0x12: "DEVICE CONTROL TWO",
    0x13: "DEVICE CONTROL THREE", 0x14: "DEVICE CONTROL FOUR", 0x15: "NEGATIVE ACKNOWLEDGE",
    0x16: "SYNCHRONOUS IDLE", 0x17: "END OF TRANSMISSION BLOCK", 0x18: "CANCEL",
    0x19: "END OF MEDIUM", 0x1A: "SUBSTITUTE", 0x1B: "ESCAPE",
    0x1C: "INFORMATION SEPARATOR FOUR", 0x1D: "INFORMATION SEPARATOR THREE",
    0x1E: "INFORMATION SEPARATOR TWO", 0x1F: "INFORMATION SEPARATOR ONE", 0x7F: "DELETE",
}


def hexdump(text: str, limit: int = 2000) -> list[dict]:
    """Same rule as view_pack.hexdump, with common control names per TSAI Lab format."""
    rows = []
    for i, ch in enumerate(text[:limit]):
        if ord(ch) > 127 or unicodedata.category(ch)[0] in "CZ":
            o = ord(ch)
            name = CONTROL_NAMES.get(o) or unicodedata.name(ch, f"CONTROL U+{o:04X}")
            rows.append(
                {
                    "offset": i,
                    "cp": f"U+{o:04X}",
                    "name": name,
                }
            )
    return rows


_TAG_LINE = re.compile(r"^([A-H][0-9]):\s*(.*)$")


def marked_issues(text: str) -> list[dict]:
    """Same characters view_pack.py makes visible, listed as diagnosis tags."""
    found = []
    for index, ch in enumerate(text):
        if ch in VISIBLE and ch != "\t":
            code = "A4" if ch == "\ufffd" else "A1"
            title = "Replacement characters" if code == "A4" else "Invisible junk characters"
            reason = (
                "Unicode replacement character U+FFFD (⟦U+FFFD⟧ in view_pack.py); original bytes were lost."
                if code == "A4"
                else "Zero-width / BOM / bidi / soft-hyphen mark highlighted by view_pack.py VISIBLE."
            )
            action = "drop — bytes are unrecoverable" if code == "A4" else "fix — strip non-tab VISIBLE characters"
            found.append(
                {
                    "code": code,
                    "title": title,
                    "evidence": f"{VISIBLE[ch]} at offset {index}",
                    "reason": reason,
                    "action": action,
                    "verdict": "drop" if code == "A4" else "fix",
                    "offset": index,
                }
            )
        elif ch not in "\n\t" and unicodedata.category(ch) == "Cc":
            found.append(
                {
                    "code": "A2",
                    "title": "Control characters",
                    "evidence": f"⟦U+{ord(ch):04X}⟧ at offset {index}",
                    "reason": "C0/C1 control character (unicodedata.category == 'Cc') shown as ⟦U+XXXX⟧ by view_pack.py.",
                    "action": "fix — normalize CRLF/CR and delete remaining Cc controls",
                    "verdict": "fix",
                    "offset": index,
                }
            )
        if len(found) >= 30:
            break
    return found


def split_tagged(text: str) -> tuple[str, list[dict]]:
    lines = text.splitlines()
    match = _TAG_LINE.match(lines[0]) if lines else None
    if match is None:
        return text, marked_issues(text)
    issues = [{"code": match.group(1), "evidence": match.group(2).strip(), "offset": None}]
    index = 1
    while index < len(lines) and (lines[index].startswith("repo:") or lines[index].startswith("path:")):
        index += 1
    if index < len(lines) and lines[index] == "":
        index += 1
    return "\n".join(lines[index:]), issues


def load_rows(path: Path) -> list[dict]:
    table = pq.read_table(path)
    names = set(table.column_names)
    required = {"doc_id", "source", "lang_label", "text"}
    missing = required - names
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(sorted(missing))}")
    rows = []
    for record in table.to_pylist():
        raw = record.get("text") or ""
        saved = record.get("issues_json")
        if saved:
            text, issues = raw, json.loads(saved)
        else:
            text, issues = split_tagged(raw)
        rows.append(
            {
                "doc_id": "" if record.get("doc_id") is None else str(record["doc_id"]),
                "source": "" if record.get("source") is None else str(record["source"]),
                "lang_label": "" if record.get("lang_label") is None else str(record["lang_label"]),
                "url": "" if record.get("url") is None else str(record.get("url") or ""),
                "text": text,
                "issues": issues,
            }
        )
    return rows


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pack viewer</title>
<style>
  :root { color-scheme: light; --bg: #f4f1ea; --ink: #1c1915; --line: #d9d2c5; --card: #fffdf8; --mark: #8a3d12; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.45 "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
  header { padding: 16px 20px 12px; border-bottom: 1px solid var(--line); background: var(--card); }
  h1 { margin: 0 0 4px; font-size: 20px; }
  header p { margin: 0; color: #5c564c; }
  .layout { display: grid; grid-template-columns: 280px 1fr; min-height: calc(100vh - 78px); }
  aside { border-right: 1px solid var(--line); background: var(--card); display: flex; flex-direction: column; }
  #q { margin: 12px; padding: 8px 10px; border: 1px solid var(--line); border-radius: 8px; }
  #list { overflow: auto; }
  button.row { display: block; width: 100%; text-align: left; border: 0; border-bottom: 1px solid var(--line); background: transparent; padding: 8px 12px; cursor: pointer; }
  button.row.on { background: #efe4d2; }
  button.row small { display: block; color: #6b645a; }
  main { padding: 16px 20px 32px; }
  .bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 12px; }
  button, input[type="number"] { font: inherit; }
  button.act { border: 1px solid var(--line); background: var(--card); border-radius: 8px; padding: 6px 10px; cursor: pointer; }
  input[type="number"] { width: 88px; padding: 6px 8px; border: 1px solid var(--line); border-radius: 8px; }
  .meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; margin-bottom: 12px; }
  .meta div { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
  .meta span { display: block; color: #6b645a; font-size: 12px; }
  pre { white-space: pre-wrap; word-break: break-word; background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px; margin: 0; }
  #hex { margin-top: 12px; font-family: ui-monospace, monospace; font-size: 13px; }
  #hex:empty { display: none; }
  code { color: var(--mark); }
  h2 { margin: 0 12px 8px; font-size: 13px; letter-spacing: 0.04em; text-transform: uppercase; color: #6b645a; }
  #issues { margin: 0 0 12px; background: #fff6ee; border: 1px solid #e4cbb6; border-radius: 10px; padding: 12px 14px; }
  #issues h2 { margin: 0 0 8px; }
  #issues ul { margin: 0; padding-left: 18px; }
  #issues li { margin: 8px 0; line-height: 1.4; }
  #issues .why { display: block; color: #4a3f34; font-size: 13px; margin-top: 2px; }
  #issues .act-tag { display: inline-block; background: #efe4d2; color: #5b2508; border-radius: 4px; padding: 1px 6px; font-size: 12px; margin-left: 6px; }
  #issues .empty { color: #6b645a; }
</style>
</head>
<body>
<header>
  <h1>Pack viewer</h1>
  <p id="file"></p>
</header>
<div class="layout">
  <aside>
    <input id="q" type="search" placeholder="Filter doc id, source, language" autocomplete="off">
    <h2>Tagged issues</h2>
    <div id="issue-list"></div>
    <h2>Documents</h2>
    <div id="list"></div>
  </aside>
  <main>
    <div class="bar">
      <button class="act" id="prev" type="button">Previous</button>
      <button class="act" id="next" type="button">Next</button>
      <label>Index <input id="jump" type="number" min="1" value="1"></label>
      <button class="act" id="go" type="button">Go</button>
      <label><input id="hex-toggle" type="checkbox"> Hex of non-ASCII and controls</label>
    </div>
    <div class="meta" id="meta"></div>
    <section id="issues">
      <h2>Tagged issues</h2>
      <ul id="issue-items"></ul>
    </section>
    <pre id="text"></pre>
    <pre id="hex"></pre>
  </main>
</div>
<script>
const state = { docs: [], shown: [], i: 0 };
const list = document.getElementById("list");
const issueList = document.getElementById("issue-list");
const issueItems = document.getElementById("issue-items");
const text = document.getElementById("text");
const hex = document.getElementById("hex");
const meta = document.getElementById("meta");

function filtered() {
  const q = document.getElementById("q").value.trim().toLowerCase();
  if (!q) return state.docs.map((_, i) => i);
  return state.docs.map((d, i) => i).filter(i => {
    const d = state.docs[i];
    return (d.doc_id + " " + d.source + " " + d.lang_label).toLowerCase().includes(q);
  });
}

function rowButton(i) {
  const d = state.docs[i];
  const b = document.createElement("button");
  b.type = "button";
  b.className = "row" + (i === state.i ? " on" : "");
  const tags = (d.issues || []).map(item => item.code).join(" ");
  const titles = (d.issues || []).map(item => item.title || "").filter(Boolean).join(", ");
  b.innerHTML = `<strong>${i + 1}. ${d.doc_id}</strong><small>${tags ? tags + (titles ? " (" + titles + ")" : "") + " · " : ""}${d.source} · ${d.lang_label} · ${d.n_chars} chars</small>`;
  b.addEventListener("click", () => showDoc(i));
  return b;
}

function drawList() {
  const ids = filtered();
  list.innerHTML = "";
  issueList.innerHTML = "";
  for (const i of ids) {
    const tagged = (state.docs[i].issues || []).length > 0;
    (tagged ? issueList : list).appendChild(rowButton(i));
  }
}

function field(label, value) {
  const div = document.createElement("div");
  const span = document.createElement("span");
  span.textContent = label;
  div.appendChild(span);
  div.appendChild(document.createTextNode(value || "—"));
  return div;
}

async function showDoc(i) {
  state.i = i;
  document.getElementById("jump").value = String(i + 1);
  const res = await fetch("/api/doc?i=" + i);
  const doc = await res.json();
  meta.innerHTML = "";
  meta.append(
    field("Document", `${i + 1} / ${state.docs.length}`),
    field("doc_id", doc.doc_id),
    field("source", doc.source),
    field("language", doc.lang_label),
    field("characters", String(doc.n_chars)),
    field("url", doc.url)
  );
  issueItems.innerHTML = "";
  const issues = doc.issues || [];
  if (!issues.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "None in this document.";
    issueItems.appendChild(empty);
  } else {
    for (const item of issues) {
      const li = document.createElement("li");
      const title = item.title ? ` (${item.title})` : "";
      const verdict = item.verdict ? `<span class="act-tag">verdict: ${item.verdict}</span>` : "";
      const action = item.action ? `<span class="act-tag">${item.action}</span>` : "";
      const why = item.reason ? `<span class="why"><strong>Why tagged:</strong> ${item.reason}</span>` : "";
      li.innerHTML = `<strong>${item.code}${title}:</strong> ${item.evidence} ${verdict} ${action}${why}`;
      issueItems.appendChild(li);
    }
  }
  text.textContent = doc.shown;
  const on = document.getElementById("hex-toggle").checked;
  hex.textContent = on ? (doc.hex.length ? doc.hex.map(r => `${String(r.offset).padStart(6)}  ${r.cp}  ${r.name}`).join("\n") : "(no non-ASCII or control characters in the scanned prefix)") : "";
  drawList();
  const onRow = list.querySelector(".on");
  if (onRow) onRow.scrollIntoView({ block: "nearest" });
}

document.getElementById("prev").onclick = () => showDoc(Math.max(0, state.i - 1));
document.getElementById("next").onclick = () => showDoc(Math.min(state.docs.length - 1, state.i + 1));
document.getElementById("go").onclick = () => {
  const n = Number(document.getElementById("jump").value);
  if (n >= 1 && n <= state.docs.length) showDoc(n - 1);
};
document.getElementById("q").oninput = drawList;
document.getElementById("hex-toggle").onchange = () => showDoc(state.i);
document.addEventListener("keydown", (ev) => {
  if (ev.target.id === "q" || ev.target.id === "jump") return;
  if (ev.key === "ArrowRight") showDoc(Math.min(state.docs.length - 1, state.i + 1));
  if (ev.key === "ArrowLeft") showDoc(Math.max(0, state.i - 1));
  if (ev.key === "h") {
    const box = document.getElementById("hex-toggle");
    box.checked = !box.checked;
    showDoc(state.i);
  }
});

fetch("/api/index").then(r => r.json()).then(data => {
  document.getElementById("file").textContent = data.path + " · " + data.n + " documents. Arrow keys move. h toggles hex. Hidden characters are written out, for example ZWSP.";
  state.docs = data.docs;
  if (!data.n) {
    text.textContent = "This pack has no rows.";
    return;
  }
  const params = new URLSearchParams(location.search);
  const start = Number(params.get("i") || "0");
  showDoc(Number.isFinite(start) ? start : 0);
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    rows: list[dict] = []
    pack_path: str = ""

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/index":
            docs = [
                {
                    "doc_id": row["doc_id"],
                    "source": row["source"],
                    "lang_label": row["lang_label"],
                    "n_chars": len(row["text"]),
                    "issues": row["issues"],
                }
                for row in self.rows
            ]
            payload = {"path": self.pack_path, "n": len(docs), "docs": docs}
            self._send(200, json.dumps(payload).encode("utf-8"), "application/json")
            return
        if parsed.path == "/api/doc":
            query = parse_qs(parsed.query)
            try:
                index = int(query.get("i", ["0"])[0])
            except ValueError:
                self._send(400, b'{"error":"bad index"}', "application/json")
                return
            if index < 0 or index >= len(self.rows):
                self._send(404, b'{"error":"no such document"}', "application/json")
                return
            row = self.rows[index]
            text = row["text"]
            payload = {
                "i": index,
                "doc_id": row["doc_id"],
                "source": row["source"],
                "lang_label": row["lang_label"],
                "url": row["url"],
                "n_chars": len(text),
                "shown": show(text),
                "hex": hexdump(text),
                "issues": row["issues"],
            }
            self._send(200, json.dumps(payload).encode("utf-8"), "application/json")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="View a diagnosis pack in the browser.")
    parser.add_argument("pack", type=Path, help="Path to the Parquet pack. Keep it outside era-v5-lab.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    pack = args.pack.expanduser().resolve()
    if not pack.is_file():
        raise SystemExit(f"No file at {pack}")
    Handler.rows = load_rows(pack)
    Handler.pack_path = str(pack)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Pack viewer: http://127.0.0.1:{args.port}  ({len(Handler.rows)} documents)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
