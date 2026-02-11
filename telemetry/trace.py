"""telemetry.trace

Helpers for reading and working with trace/telemetry artifacts produced by simulations.
Typical uses: loading a trace, extracting time series, and printing quick summaries.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

@dataclass
class TraceWriter:
    path: Path
    run_meta: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("w", encoding="utf-8")
        if self.run_meta is not None:
            self.write({"type": "meta", **self.run_meta})

    def write(self, event: Dict[str, Any]) -> None:
        self.f.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass
