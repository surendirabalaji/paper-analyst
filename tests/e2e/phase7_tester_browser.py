"""Tester Phase 7 browser run: reuses the developer's CDP driver (phase7_browser_fault.main) with the tester's
own fault spec and question. Optional --wrong-token replaces the token with a wrong one (expects a visible 403).
Usage: python tests/e2e/phase7_tester_browser.py BASE OUT PROFILE "SPEC" "QUERY" [--wrong-token]
REAL provider calls (except --wrong-token). The token is never printed or saved (the driver asserts this)."""
import asyncio, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import phase7_browser_fault as drv  # noqa: E402

base, out, profile, spec, query = sys.argv[1:6]
drv.SPEC, drv.QUERY = spec, query
if "--wrong-token" in sys.argv:
    drv.dotenv_values = lambda p: {"FAULT_INJECT_TOKEN": "W" * 43}
sys.argv = [sys.argv[0], base, out, profile]
asyncio.run(drv.main())
