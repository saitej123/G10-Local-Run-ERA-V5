"""256 special tokens for the ERA V5 agentic coding model.

G3 adds every surface form as an atomic token before tokenizer training
(1 Nov). G4 and G9 use this same list. Do not rename an assigned token in v1.
A reserved index may be promoted. It may not be renumbered.
"""

from __future__ import annotations

COUNT = 256

# Token budget declared by the effort tokens. This is the think + tool-trace
# budget, not the final answer.
EFFORT_BUDGETS = {
    "1k": 1024,
    "4k": 4096,
    "16k": 16384,
    "64k": 65536,
}

# index -> name. Missing indexes are reserved_NNN.
ASSIGNED: dict[int, str] = {
    0: "bos",
    1: "eos",
    2: "pad",
    3: "unk",
    4: "end",
    8: "system",
    9: "user",
    10: "assistant",
    11: "tool",
    16: "effort_1k",
    17: "effort_4k",
    18: "effort_16k",
    19: "effort_64k",
    24: "think",
    25: "think_end",
    26: "tool_call",
    27: "tool_call_end",
    28: "tool_result",
    29: "tool_result_end",
    30: "response",
    32: "file",
    33: "patch",
    34: "test",
    35: "observe",
    36: "finish",
    40: "canary_start",
    41: "canary_end",
}


def surface(name: str) -> str:
    return f"<|era_{name}|>"


def reserved_name(index: int) -> str:
    return f"reserved_{index:03d}"


def all_tokens() -> list[tuple[int, str, str, str]]:
    """Return (index, name, surface, status) for indexes 0..255."""
    rows: list[tuple[int, str, str, str]] = []
    for index in range(COUNT):
        if index in ASSIGNED:
            name = ASSIGNED[index]
            rows.append((index, name, surface(name), "assigned"))
        else:
            name = reserved_name(index)
            rows.append((index, name, surface(name), "reserved"))
    return rows


def by_name() -> dict[str, str]:
    return {name: token for _, name, token, _ in all_tokens()}


def main() -> None:
    for index, name, token, status in all_tokens():
        print(f"{index:03d}\t{status}\t{name}\t{token}")


if __name__ == "__main__":
    main()
