"""T0-PIN: every dependency is pinned with == and requirements.in agrees with the lockfile."""
import re

REQ_LINE = re.compile(r"^([A-Za-z0-9_.\-]+)(\[[^\]]+\])?==([^\s;#]+)\s*$")


def _spec_lines(path):
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        yield line


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def test_lockfile_only_exact_pins(root):
    bad = [l for l in _spec_lines(root / "requirements.txt") if not REQ_LINE.match(l)]
    assert bad == [], f"non-exact or unparseable lines: {bad}"


def test_no_range_operators_anywhere(root):
    for name in ("requirements.txt", "requirements.in"):
        for l in _spec_lines(root / name):
            assert not re.search(r">=|<=|~=|!=|(?<![=<>!~])[<>](?!=)", l), (name, l)


def test_lockfile_has_no_duplicate_packages(root):
    names = [_norm(REQ_LINE.match(l).group(1)) for l in _spec_lines(root / "requirements.txt")]
    assert len(names) == len(set(names))


def test_top_level_pins_match_lockfile(root):
    lock = {}
    for l in _spec_lines(root / "requirements.txt"):
        m = REQ_LINE.match(l)
        lock[_norm(m.group(1))] = m.group(3)
    for l in _spec_lines(root / "requirements.in"):
        m = REQ_LINE.match(l)
        assert m, f"requirements.in line not exact-pinned: {l}"
        assert lock.get(_norm(m.group(1))) == m.group(3), l


def test_env_example_has_placeholders_and_model_ids(root):
    env = dict(
        l.split("=", 1)
        for l in _spec_lines(root / ".env.example")
        if "=" in l
    )
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "TAVILY_API_KEY"):
        assert "REPLACE_ME" in env[key], f"{key} must be a placeholder"
    for key in ("OPENAI_MODEL", "GEMINI_MODEL", "OPENAI_EMBEDDING_MODEL"):
        assert env.get(key), f"{key} missing from .env.example"
