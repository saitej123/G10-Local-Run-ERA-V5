"""Tests for the G10 special-token list and chat format v0."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parents[1] / "era-v5-lab" / "teams" / "10-post-training"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tokens = _load("era_special_tokens", ROOT / "special-tokens" / "tokens.py")
render_mod = _load("era_render", ROOT / "chat-format" / "render.py")

GHOSTS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<think>",
    "</think>",
    "[INST]",
    "[/INST]",
    "<s>",
    "</s>",
    "### Assistant:",
)


def test_block_has_256_unique_tokens():
    rows = tokens.all_tokens()
    assert len(rows) == 256
    indexes = [row[0] for row in rows]
    surfaces = [row[2] for row in rows]
    assert indexes == list(range(256))
    assert len(set(surfaces)) == 256
    assert all(surface.startswith("<|era_") and surface.endswith("|>") for surface in surfaces)
    assert all("\n" not in surface and " " not in surface for surface in surfaces)


def test_effort_budgets_and_tokens():
    assert tokens.EFFORT_BUDGETS == {"1k": 1024, "4k": 4096, "16k": 16384, "64k": 65536}
    names = tokens.by_name()
    for effort in tokens.EFFORT_BUDGETS:
        assert names[f"effort_{effort}"] == f"<|era_effort_{effort}|>"
        assert render_mod.effort_limit(effort) == tokens.EFFORT_BUDGETS[effort]


def test_assigned_tokens_are_not_pretraining_ghost_markers():
    surfaces = {row[2] for row in tokens.all_tokens()}
    assert surfaces.isdisjoint(GHOSTS)


def test_render_parse_roundtrip_for_each_effort():
    messages = [
        {"role": "system", "content": "You edit code and you report only what a tool returned."},
        {"role": "user", "content": "test_add fails in calc.py. Fix add."},
        {
            "role": "assistant",
            "think": "The return adds one too many.",
            "tool_calls": [{"name": "read_file", "arguments": {"path": "calc.py"}}],
            "content": "I am reading calc.py before editing it.",
        },
        {"role": "tool", "content": "def add(a, b):\n    return a + b + 1\n"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "apply_patch",
                    "arguments": {"path": "calc.py", "patch": "return a + b"},
                }
            ],
            "content": "Removed the extra increment.",
        },
    ]
    for effort in ("1k", "4k", "16k", "64k"):
        text = render_mod.render(messages, effort=effort)
        assert f"<|era_effort_{effort}|>" in text
        for other in ("1k", "4k", "16k", "64k"):
            if other != effort:
                assert f"<|era_effort_{other}|>" not in text
        parsed = render_mod.parse(text)
        assert parsed["effort"] == effort
        assert parsed["messages"] == [
            messages[0],
            messages[1],
            messages[2],
            messages[3],
            {
                "role": "assistant",
                "tool_calls": messages[4]["tool_calls"],
                "content": messages[4]["content"],
            },
        ]


def test_render_rejects_bad_effort_role_and_embedded_token():
    ok = [{"role": "user", "content": "hello"}]
    for effort in ("2k", "8k", "", "64K"):
        try:
            render_mod.render(ok, effort=effort)
        except render_mod.RenderError:
            pass
        else:
            raise AssertionError(f"effort {effort!r} should fail")

    try:
        render_mod.render([{"role": "developer", "content": "no"}])
    except render_mod.RenderError:
        pass
    else:
        raise AssertionError("unknown role should fail")

    leaked = [{"role": "user", "content": "see <|era_bos|> here"}]
    try:
        render_mod.render(leaked)
    except render_mod.RenderError:
        pass
    else:
        raise AssertionError("embedded special token should fail")


def test_empty_optional_fields_are_omitted():
    text = render_mod.render(
        [{"role": "assistant", "content": "", "think": "", "tool_calls": []}],
        effort="1k",
    )
    parsed = render_mod.parse(text)
    assert parsed == {"effort": "1k", "messages": [{"role": "assistant"}]}
    assert "<|era_think|>" not in text
    assert "<|era_response|>" not in text


def test_docs_match_the_plan_sheet_set():
    chat = (LAB / "chat-format" / "chat_format_v0.md").read_text(encoding="utf-8")
    token_doc = (LAB / "special-tokens" / "special_tokens_v0.md").read_text(encoding="utf-8")
    licence = (LAB / "licences" / "licence_table.md").read_text(encoding="utf-8")
    for budget in ("1k", "4k", "16k", "64k"):
        assert budget in chat
        assert f"effort_{budget}" in token_doc
    for _, name, surface, status in tokens.all_tokens():
        if status == "assigned":
            assert surface in token_doc
            assert name in token_doc
    assert "256" in token_doc
    assert "TSAI reserves" in token_doc
    assert "bigcode/starcoderdata" in licence
    assert "code-python" in licence
    assert "TSAI reserves" in licence
    assert "Sai Teja" not in licence
    assert "saitej" not in licence.lower()

    decisions = []
    for line in licence.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or cells[0] in {"Set", ""} or set(cells[0]) <= set("-: "):
            continue
        decisions.append(cells[-1])
    assert decisions == ["TSAI reserves"]
