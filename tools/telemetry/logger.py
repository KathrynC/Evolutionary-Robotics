from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pybullet as p


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _normalize_step(t, fallback: int) -> int:
    # Accept numpy scalars/arrays, floats, etc.
    try:
        if hasattr(t, "item"):
            t = t.item()
    except Exception:
        pass
    try:
        return int(t)
    except Exception:
        return int(fallback)


@dataclass
class TelemetryLogger:
    robot_id: int
    out_dir: Path
    every: int = 10
    variant_id: str = "unknown"
    run_id: str = "run0"
    enabled: bool = True

    _fp: Optional[Any] = field(default=None, init=False, repr=False)
    _t0_pos: Optional[Tuple[float, float, float]] = field(default=None, init=False)
    _max_abs_roll: float = field(default=0.0, init=False)
    _max_abs_pitch: float = field(default=0.0, init=False)
    _steps: int = field(default=0, init=False)
    _tip_steps: int = field(default=0, init=False)

    # cache last known pose so finalize() is safe even after disconnect/crash
    _last_pos: Optional[Tuple[float, float, float]] = field(default=None, init=False)
    _last_rpy: Optional[Tuple[float, float, float]] = field(default=None, init=False)

    def __post_init__(self):
        if not self.enabled:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.out_dir / "telemetry.jsonl", "w", encoding="utf-8")

    def log_step(self, t):
        if not self.enabled:
            return

        t = _normalize_step(t, fallback=self._steps)
        if self.every > 1 and (t % self.every) != 0:
            return

        # Base pose
        try:
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        except Exception:
            return

        # Cache last pose for safe finalize
        self._last_pos = tuple(pos)
        self._last_rpy = (float(roll), float(pitch), float(yaw))

        if self._t0_pos is None:
            self._t0_pos = self._last_pos

        self._max_abs_roll = max(self._max_abs_roll, abs(float(roll)))
        self._max_abs_pitch = max(self._max_abs_pitch, abs(float(pitch)))

        # crude tipping heuristic: big roll or pitch
        if abs(float(roll)) > 1.0 or abs(float(pitch)) > 1.0:
            self._tip_steps += 1
        self._steps += 1

        # Joint telemetry
        joints = []
        try:
            n = p.getNumJoints(self.robot_id)
            for j in range(n):
                js = p.getJointState(self.robot_id, j)
                joints.append({
                    "j": j,
                    "pos": _safe_float(js[0]),
                    "vel": _safe_float(js[1]),
                    "tau": _safe_float(js[3]),
                })
        except Exception:
            pass

        # Contact count
        contact_n = 0
        try:
            contact_n = len(p.getContactPoints(bodyA=self.robot_id))
        except Exception:
            pass

        rec: Dict[str, Any] = {
            "t": t,
            "base": {"x": pos[0], "y": pos[1], "z": pos[2]},
            "rpy": {"r": float(roll), "p": float(pitch), "y": float(yaw)},
            "contacts": contact_n,
            "joints": joints,
        }
        self._fp.write(json.dumps(rec) + "\n")

    def finalize(self):
        if not self.enabled:
            return

        # Prefer cached pose; do NOT depend on pybullet being alive.
        pos = self._last_pos
        rpy = self._last_rpy

        # If we never logged a step, try once, but keep it safe.
        if pos is None or rpy is None:
            try:
                pos2, orn = p.getBasePositionAndOrientation(self.robot_id)
                rpy2 = p.getEulerFromQuaternion(orn)
                pos = tuple(pos2)
                rpy = (float(rpy2[0]), float(rpy2[1]), float(rpy2[2]))
            except Exception:
                pos = (None, None, None)
                rpy = (None, None, None)

        roll, pitch, yaw = rpy

        dx = dy = dz = None
        if self._t0_pos is not None and pos[0] is not None:
            dx = pos[0] - self._t0_pos[0]
            dy = pos[1] - self._t0_pos[1]
            dz = pos[2] - self._t0_pos[2]

        upright_frac = None
        if self._steps > 0:
            upright_frac = 1.0 - (self._tip_steps / float(self._steps))

        summary = {
            "variant_id": self.variant_id,
            "run_id": self.run_id,
            "steps_logged": self._steps,
            "max_abs_roll": self._max_abs_roll,
            "max_abs_pitch": self._max_abs_pitch,
            "upright_fraction": upright_frac,
            "delta": {"dx": dx, "dy": dy, "dz": dz},
            "final_rpy": {"r": roll, "p": pitch, "y": yaw},
        }
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        try:
            if self._fp:
                self._fp.close()
        except Exception:
            pass
