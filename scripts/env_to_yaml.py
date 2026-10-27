#!/usr/bin/env python3
"""Convert an allowlisted subset of a .env file to a Cloud Run --env-vars-file (YAML).

Only runtime settings the service reads are copied. Paths (CHROMA_PERSIST_DIR, SQLITE_PATH) come from the
image, and GCP_* values are deploy-time only. The output file is created with mode 0600 and must not exist.
Values are never printed; only key names are.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REQUIRED = {"OPENAI_API_KEY"}
# Used by later phases (router, Tavily MCP); warned about, not required, in Phase 3.
OPTIONAL_SECRETS = {"OPENROUTER_API_KEY", "TAVILY_API_KEY"}
SETTINGS = {
    "OPENAI_MODEL", "OPENROUTER_BASE_URL", "GEMINI_MODEL", "DEFAULT_PROVIDER", "OPENAI_EMBEDDING_MODEL",
    "CHROMA_COLLECTION", "RAG_TOP_K", "MAX_SUPERVISOR_STEPS", "MAX_TOOL_ROUNDS", "WORKER_TIMEOUT_S",
    "REQUEST_BUDGET_S", "LLM_TIMEOUT_S",
}
ALLOWED = REQUIRED | OPTIONAL_SECRETS | SETTINGS
PLACEHOLDER = "REPLACE_ME"


def parse_value(raw: str, line_number: int) -> str:
    value = raw.strip()
    if value[:1] in ("'", '"'):
        quote = value[0]
        end = value.find(quote, 1)
        if end == -1:
            raise ValueError(f"unclosed quote on line {line_number}")
        trailing = value[end + 1 :].strip()
        if trailing and not trailing.startswith("#"):
            raise ValueError(f"unexpected text after quote on line {line_number}")
        return value[1:end]
    for i, ch in enumerate(value):  # inline comment: " #" or a leading "#"
        if ch == "#" and (i == 0 or value[i - 1].isspace()):
            return value[:i].rstrip()
    return value


def read_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(encoding="utf-8-sig") as src:
        for number, raw in enumerate(src, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            name, sep, rest = line.partition("=")
            name = name.strip()
            if name not in ALLOWED:
                continue
            if not sep:
                raise ValueError(f"missing '=' on line {number}")
            value = parse_value(rest, number)
            values.pop(name, None)  # last assignment wins, including an empty one
            if value and PLACEHOLDER not in value:
                values[name] = value
    missing = REQUIRED - values.keys()
    if missing:
        raise ValueError("missing, empty or placeholder: " + ", ".join(sorted(missing)))
    return values


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path(".env"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    try:
        values = read_values(args.input)
        payload = "".join(f"{k}: {json.dumps(values[k])}\n" for k in sorted(values))
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write(payload)
    except ValueError as e:
        print(f"env_to_yaml: {e}", file=sys.stderr)
        return 1
    except OSError:
        print("env_to_yaml: cannot read input or create output (output must not already exist)", file=sys.stderr)
        return 1
    absent = OPTIONAL_SECRETS - values.keys()
    if absent:
        print("env_to_yaml: warning, not set (needed in later phases): " + ", ".join(sorted(absent)), file=sys.stderr)
    print(f"env_to_yaml: wrote {len(values)} variables: " + ", ".join(sorted(values)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
