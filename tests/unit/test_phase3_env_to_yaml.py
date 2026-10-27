"""Tester Phase 3: scripts/env_to_yaml.py behaviour on FAKE env files only (no real credentials are read).

Checks the allowlist, placeholder rejection, quoting/comment parsing, YAML type safety, file mode,
O_EXCL refusal to overwrite, and that values never reach stdout/stderr.
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "env_to_yaml.py"
FAKE_KEY = "sk-FAKE-tester-canary-0123456789abcdef"
FAKE_OR = "sk-or-FAKE-tester-canary-abcdef0123456789"
FAKE_TV = "tvly-FAKE-tester-canary-9876543210"


def run(tmp_path: Path, text: str, out_name: str = "env.yaml"):
    src = tmp_path / "fake.env"
    src.write_text(text, encoding="utf-8")
    out = tmp_path / out_name
    p = subprocess.run([sys.executable, "-I", str(SCRIPT), "--input", str(src), "--output", str(out)],
                       capture_output=True, text=True, timeout=30)
    return p, out


def no_leak(p, *values):
    for v in values:
        assert v not in p.stdout and v not in p.stderr


def test_allowlist_types_mode_and_no_value_output(tmp_path):
    p, out = run(tmp_path, "\n".join([
        "# comment",
        f"OPENAI_API_KEY={FAKE_KEY}",
        f"export OPENROUTER_API_KEY='{FAKE_OR}'",
        f'TAVILY_API_KEY="{FAKE_TV}"  # trailing comment',
        "OPENAI_MODEL = gpt-6.1-sol",
        "RAG_TOP_K=6",
        "MAX_TOOL_ROUNDS=yes",            # YAML 1.1 bool trap must stay a string
        "GEMINI_MODEL=google/gemini-3.8-flash # inline",
        "CHROMA_PERSIST_DIR=./data/chroma_index",   # path: must come from the image, not .env
        "SQLITE_PATH=./data/papers.db",
        "GCP_PROJECT_ID=some-project",
        "UNRELATED_SECRET=should-not-copy",
    ]) + "\n")
    assert p.returncode == 0, p.stderr
    no_leak(p, FAKE_KEY, FAKE_OR, FAKE_TV, "should-not-copy")
    data = yaml.safe_load(out.read_text())
    assert data == {
        "OPENAI_API_KEY": FAKE_KEY, "OPENROUTER_API_KEY": FAKE_OR, "TAVILY_API_KEY": FAKE_TV,
        "OPENAI_MODEL": "gpt-6.1-sol", "RAG_TOP_K": "6", "MAX_TOOL_ROUNDS": "yes",
        "GEMINI_MODEL": "google/gemini-3.8-flash",
    }
    assert all(isinstance(v, str) for v in data.values())  # Cloud Run env values must be strings
    assert stat.S_IMODE(os.stat(out).st_mode) == 0o600


def test_example_placeholders_rejected(tmp_path):
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    p, out = run(tmp_path, example)
    assert p.returncode == 1
    assert "OPENAI_API_KEY" in p.stderr and not out.exists()


def test_placeholder_optional_secret_dropped_with_warning(tmp_path):
    p, out = run(tmp_path, f"OPENAI_API_KEY={FAKE_KEY}\nTAVILY_API_KEY=tvly-REPLACE_ME\n")
    assert p.returncode == 0
    assert "TAVILY_API_KEY" in p.stderr and "warning" in p.stderr.lower()
    assert "TAVILY_API_KEY" not in yaml.safe_load(out.read_text())
    no_leak(p, FAKE_KEY)


@pytest.mark.parametrize("text", [
    "OPENAI_MODEL=x\n",                                   # missing
    "OPENAI_API_KEY=\n",                                  # empty
    f"OPENAI_API_KEY={FAKE_KEY}\nOPENAI_API_KEY=\n",      # last (empty) assignment wins
    "OPENAI_API_KEY='unterminated\n",                    # malformed quote
])
def test_required_missing_or_malformed_fails_without_output(tmp_path, text):
    p, out = run(tmp_path, text)
    assert p.returncode == 1 and not out.exists()
    no_leak(p, FAKE_KEY, "unterminated")


def test_refuses_to_overwrite_existing_output(tmp_path):
    out = tmp_path / "env.yaml"
    out.write_text("keep: me\n")
    p, _ = run(tmp_path, f"OPENAI_API_KEY={FAKE_KEY}\n")
    assert p.returncode == 1 and out.read_text() == "keep: me\n"
    no_leak(p, FAKE_KEY)


def test_yaml_special_characters_round_trip(tmp_path):
    tricky = 'a: b #x "q" \\ é ✓ {[*&!%@'
    p, out = run(tmp_path, f"OPENAI_API_KEY='{tricky}'\n")
    assert p.returncode == 0, p.stderr
    assert yaml.safe_load(out.read_text())["OPENAI_API_KEY"] == tricky
