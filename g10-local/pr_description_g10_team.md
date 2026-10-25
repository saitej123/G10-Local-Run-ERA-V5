### G10 Post-training and Agentic — Week 1 Deliverables

- **Licence table & strategy v0 (`teams/10-post-training/licences/licence_table.md`):** Documents `code-python` (`bigcode/starcoderdata`, `Code`) with decision `TSAI reserves` and commercial-use strategy rules.
- **Special tokens v0 & v1 (`teams/10-post-training/special-tokens/special_tokens_v0.md`, `special_tokens_v1.md`, `tokens.py`):** 256 reserved atomic `<|era_*|>` tokens (`0..255`), including role markers, tool call/result delimiters, canary wrappers for Team 9, and the four effort budgets (`1k`, `4k`, `16k`, `64k`).
- **Chat format v0 & renderer (`teams/10-post-training/chat-format/chat_format_v0.md`, `render.py`):** Turn structure, ordering of `<|era_think|>`, `<|era_tool_call|>`, `<|era_response|>`, deterministic `render()`/`parse()`, and effort-budget enforcement for Team 3, Team 4, Team 9, and Team 12.
- **End-to-end Docker SWE task (`teams/10-post-training/results/swe_e2e.md`):** Reproducible fail-to-pass run in `python:3.12-slim`.
- **Tests (`teams/10-post-training/tests/test_render.py`, `test_specs.py`):** `pytest teams/10-post-training/tests -q` passes (`10 passed`, `0` failures).
