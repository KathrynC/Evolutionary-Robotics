"""
twine/server.py

FastAPI bridge between the SugarCube interactive story and the PyBullet simulation.

Endpoints:
    GET  /          — serves experiment.html
    POST /simulate  — accepts gait parameters, runs headless simulation, returns results

Usage:
    python3 twine/server.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

# Project root: Evolutionary-Robotics/
PROJECT_DIR = Path(__file__).resolve().parent.parent
GAITS_DIR = PROJECT_DIR / "gaits"
GAITS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Twine Experiment Designer")


class GaitParams(BaseModel):
    amplitude: float = Field(..., ge=0.3, le=1.8, description="Joint amplitude (rad)")
    frequency: float = Field(..., ge=0.6, le=5.0, description="Oscillation frequency (Hz)")
    back_offset: float = Field(..., ge=-0.7, le=0.7, description="Back leg DC offset (rad)")
    front_offset: float = Field(..., ge=-0.7, le=0.7, description="Front leg DC offset (rad)")
    max_force: float = Field(..., ge=80, le=650, description="Max motor force (N)")
    sim_steps: int = Field(1000, ge=200, le=5000, description="Simulation steps")


@app.get("/", response_class=HTMLResponse)
async def serve_story():
    """Serve the SugarCube experiment story."""
    html_path = Path(__file__).resolve().parent / "experiment.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="experiment.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/simulate")
async def run_simulation(params: GaitParams):
    """Run a headless PyBullet simulation with the given gait parameters."""
    variant_id = f"twine_{uuid.uuid4().hex[:8]}"
    run_id = f"web_{time.strftime('%Y%m%d_%H%M%S')}"

    # Build gait variant JSON (canonical keys from gaits/sine_crawl_001.json)
    variant = {
        "name": f"Twine Experiment {variant_id}",
        "variant_id": variant_id,
        "type": "sine_2joint",
        "A": params.amplitude,
        "f": params.frequency,
        "O_back": params.back_offset,
        "O_front": params.front_offset,
        "phi_back": 0.0,          # anti-phase walking defaults
        "phi_front": math.pi,
        "MAX_FORCE": params.max_force,
        "SIM_STEPS": params.sim_steps,
        "ROBOT_FRICTION": 2.5,
        "PLANE_FRICTION": 2.0,
    }

    # Write variant to temp file in gaits/
    variant_path = GAITS_DIR / f"variant_{variant_id}.json"
    variant_path.write_text(json.dumps(variant, indent=2), encoding="utf-8")

    # Telemetry output directory
    telemetry_root = PROJECT_DIR / "artifacts" / "telemetry"
    out_dir = telemetry_root / variant_id / run_id

    # Build environment following tools/zoo/run_zoo.py pattern
    env = os.environ.copy()
    env["GAIT_VARIANT_PATH"] = str(variant_path)
    env["HEADLESS"] = "1"
    env["TELEMETRY"] = "1"
    env["TELEMETRY_EVERY"] = "10"
    env["TELEMETRY_OUT"] = str(telemetry_root)
    env["TELEMETRY_VARIANT_ID"] = variant_id
    env["TELEMETRY_RUN_ID"] = run_id
    env["SIM_STEPS"] = str(params.sim_steps)
    env["MAX_FORCE"] = str(params.max_force)
    env["SLEEP_TIME"] = "0"

    try:
        result = subprocess.run(
            [sys.executable, "simulate.py"],
            env=env,
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        _cleanup(variant_path)
        raise HTTPException(status_code=504, detail="Simulation timed out (120s)")
    except Exception as e:
        _cleanup(variant_path)
        raise HTTPException(status_code=500, detail=f"Simulation failed to start: {e}")

    # Read summary.json from telemetry output
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        # Fallback: parse stdout for DX/MAX_Z lines
        summary = _parse_stdout_fallback(result.stdout)

    # Merge input params into response
    response = {
        "status": "ok" if result.returncode == 0 else "error",
        "returncode": result.returncode,
        "params": variant,
        "summary": summary,
    }

    if result.returncode != 0:
        response["stderr"] = result.stderr[-500:] if result.stderr else ""

    _cleanup(variant_path)
    return response


def _parse_stdout_fallback(stdout: str) -> dict:
    """Extract displacement metrics from simulate.py stdout as fallback."""
    summary = {}
    for line in (stdout or "").splitlines():
        if line.startswith("DX "):
            parts = line.split()
            try:
                summary["dx"] = float(parts[1])
                summary["max_dx"] = float(parts[3])
            except (IndexError, ValueError):
                pass
        if line.startswith("MAX_Z "):
            parts = line.split()
            try:
                summary["max_z"] = float(parts[1])
            except (IndexError, ValueError):
                pass
    return summary


def _cleanup(variant_path: Path):
    """Remove temporary variant file."""
    try:
        variant_path.unlink(missing_ok=True)
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
