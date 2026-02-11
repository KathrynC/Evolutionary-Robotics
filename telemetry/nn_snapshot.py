from __future__ import annotations

from typing import Any, Dict, Optional


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def snapshot_nn(nn: Any) -> Dict[str, Any]:
    """Best-effort snapshot of a pyrosim-style neural network."""
    if nn is None:
        return {"neurons": {}, "synapses": []}

    snap: Dict[str, Any] = {"neurons": {}, "synapses": []}

    neurons = getattr(nn, "neurons", None)
    if isinstance(neurons, dict):
        for nid, n in neurons.items():
            v = getattr(n, "value", None)
            if v is None and hasattr(n, "Get_Value"):
                v = _safe(lambda: n.Get_Value(), default=None)

            ntype = getattr(n, "type", None)
            if ntype is None and hasattr(n, "Get_Type"):
                ntype = _safe(lambda: n.Get_Type(), default=None)

            snap["neurons"][str(nid)] = {
                "type": ntype,
                "value": float(v) if v is not None else None,
            }

    synapses = getattr(nn, "synapses", None)
    if isinstance(synapses, dict):
        for (src, tgt), s in synapses.items():
            w = getattr(s, "weight", None)
            if w is None and hasattr(s, "Get_Weight"):
                w = _safe(lambda: s.Get_Weight(), default=None)
            snap["synapses"].append(
                {"src": str(src), "tgt": str(tgt), "w": float(w) if w is not None else None}
            )

    return snap


def safe_snapshot_nn(obj: Any) -> Optional[Dict[str, Any]]:
    """Accepts SIMULATION/ROBOT/NN and tries to find the NN."""
    if obj is None:
        return None
    if hasattr(obj, "robot"):
        obj = getattr(obj, "robot", obj)
    if hasattr(obj, "nn"):
        obj = getattr(obj, "nn", obj)
    return snapshot_nn(obj)
