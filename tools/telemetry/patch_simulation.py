#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
import time
from pathlib import Path


def backup(p: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    b = p.with_suffix(p.suffix + f".bak_{ts}")
    b.write_text(p.read_text(encoding="utf-8", errors="replace"), encoding="utf-8", newline="\n")
    return b


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if target is None:
        for cand in ("simulation.py", "simulate.py", "sim.py"):
            if Path(cand).exists():
                target = Path(cand)
                break
    if target is None or not target.exists():
        print("Usage: python3 tools/telemetry/patch_simulation.py path/to/simulate.py")
        sys.exit(2)

    s = target.read_text(encoding="utf-8", errors="replace")
    if "TelemetryLogger" in s:
        print(f"{target}: already patched (TelemetryLogger found).")
        return

    b = backup(target)
    print(f"backup: {target} -> {b}")

    # Ensure imports exist
    if "import os" not in s:
        s = "import os\n" + s
    if "from pathlib import Path" not in s:
        s = "from pathlib import Path\n" + s

    inject_import = "from tools.telemetry.logger import TelemetryLogger\n"
    if inject_import not in s:
        imports = list(re.finditer(r"^(import .*|from .* import .*)\s*$", s, flags=re.MULTILINE))
        if imports:
            last = imports[-1].end()
            s = s[:last] + "\n" + inject_import + s[last:]
        else:
            s = inject_import + "\n" + s

    # Anchor: robot assignment line; preserve indentation
    m_robot = re.search(r"^(?P<indent>\s*)robot\s*=\s*.*$", s, flags=re.MULTILINE)
    if not m_robot:
        print(f"{target}: couldn't find a `robot = ...` line to anchor telemetry init.")
        target.write_text(s, encoding="utf-8", newline="\n")
        return

    indent = m_robot.group("indent")
    init_block = (
        f"\n{indent}# telemetry (optional)\n"
        f"{indent}_telemetry_on = os.getenv('TELEMETRY','').lower() in ('1','true','yes','on')\n"
        f"{indent}_telemetry_every = int(os.getenv('TELEMETRY_EVERY','10'))\n"
        f"{indent}_variant_id = os.getenv('TELEMETRY_VARIANT_ID','manual')\n"
        f"{indent}_run_id = os.getenv('TELEMETRY_RUN_ID','run0')\n"
        f"{indent}_out_dir = Path(os.getenv('TELEMETRY_OUT','artifacts/telemetry')) / _variant_id / _run_id\n"
        f"{indent}telemetry = TelemetryLogger(robot.robotId, _out_dir, every=_telemetry_every, variant_id=_variant_id, run_id=_run_id, enabled=_telemetry_on)\n"
    )

    insert_at = m_robot.end()
    s = s[:insert_at] + init_block + s[insert_at:]

    # Instrument after stepSimulation; preserve indentation
    m_step = re.search(r"^(?P<indent>\s*)p\.stepSimulation\(\)\s*$", s, flags=re.MULTILINE)
    if not m_step:
        print(f"{target}: couldn't find `p.stepSimulation()` to instrument.")
        target.write_text(s, encoding="utf-8", newline="\n")
        return

    s = s[:m_step.end()] + f"\n{m_step.group('indent')}telemetry.log_step(t)\n" + s[m_step.end():]

    # Finalize before disconnect if present
    m_disc = re.search(r"^(?P<indent>\s*)p\.disconnect\(\)\s*$", s, flags=re.MULTILINE)
    if m_disc:
        s = s[:m_disc.start()] + f"{m_disc.group('indent')}telemetry.finalize()\n" + s[m_disc.start():]
    else:
        s += "\ntelemetry.finalize()\n"

    target.write_text(s, encoding="utf-8", newline="\n")
    print(f"patched: {target}")


if __name__ == "__main__":
    main()
