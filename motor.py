import os
import numpy as np
import pybullet as p
import pyrosim.pyrosim as pyrosim
import constants as c


VARIANT_RULE_GAIT_V1 = True
import os as _os_vrg
import json as _json_vrg
from pathlib import Path as _Path_vrg

_variant_cache_vrg = {"path": None, "mtime": None, "data": None}

def _load_variant_vrg():
    vp = _os_vrg.getenv("GAIT_VARIANT_PATH")
    if not vp:
        return None
    try:
        mtime = _Path_vrg(vp).stat().st_mtime
    except Exception:
        return None
    c = _variant_cache_vrg
    if c["path"] == vp and c["mtime"] == mtime and c["data"] is not None:
        return c["data"]
    try:
        d = _json_vrg.loads(_Path_vrg(vp).read_text(encoding="utf-8"))
    except Exception:
        return None
    c["path"] = vp
    c["mtime"] = mtime
    c["data"] = d
    return d

def _find_value_for_key_vrg(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            got = _find_value_for_key_vrg(v, key)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = _find_value_for_key_vrg(v, key)
            if got is not None:
                return got
    return None

def _seq_from_rules_vrg(rules_obj):
    if rules_obj is None:
        return None
    if isinstance(rules_obj, list):
        if all(isinstance(x, (int,float)) for x in rules_obj):
            return [float(x) for x in rules_obj]
        out = []
        for x in rules_obj:
            if isinstance(x, (int,float)):
                out.append(float(x))
            elif isinstance(x, dict):
                for k in ("angle","target","pos","value","desired","a"):
                    if k in x and isinstance(x[k], (int,float)):
                        out.append(float(x[k]))
                        break
        return out if out else None
    if isinstance(rules_obj, dict):
        items = []
        for k,v in rules_obj.items():
            try:
                ki = int(k)
            except Exception:
                continue
            if isinstance(v, (int,float)):
                items.append((ki, float(v)))
            elif isinstance(v, dict):
                for kk in ("angle","target","pos","value","desired","a"):
                    if kk in v and isinstance(v[kk], (int,float)):
                        items.append((ki, float(v[kk])))
                        break
        if items:
            items.sort(key=lambda x: x[0])
            return [v for _,v in items]
    return None

def _get_period_steps_vrg(variant):
    if not isinstance(variant, dict):
        return 240
    bins = variant.get("bins")
    meta = variant.get("meta")
    for obj in (bins, meta, variant):
        if isinstance(obj, dict):
            for k in ("period_steps","cycle_steps","steps_per_cycle","steps"):
                v = obj.get(k)
                if isinstance(v, (int,float)) and v > 0:
                    return int(v)
    return 240

def _variant_target_vrg(joint_name, t):
    v = _load_variant_vrg()
    if not isinstance(v, dict):
        return None
    rules = v.get("rules")
    if rules is None:
        return None
    name = joint_name.decode() if isinstance(joint_name, (bytes, bytearray)) else str(joint_name)
    per = _get_period_steps_vrg(v)
    found = None
    if isinstance(rules, dict):
        if name in rules:
            found = rules[name]
        else:
            found = _find_value_for_key_vrg(rules, name)
    else:
        found = _find_value_for_key_vrg(rules, name)
    seq = _seq_from_rules_vrg(found)
    if not seq:
        return None
    if t is None:
        return None
    try:
        tt = int(t)
    except Exception:
        return None
    n = len(seq)
    if n <= 0:
        return None
    idx = int((tt % per) * n / per)
    if idx < 0:
        idx = 0
    if idx >= n:
        idx = n - 1
    return float(seq[idx])

def _set_motor_vrg(joint_name, angle, max_force):
    try:
        pyrosim.Set_Motor_For_Joint(jointName=joint_name, desiredAngle=angle, maxForce=max_force)
        return True
    except Exception:
        pass
    try:
        pyrosim.Set_Motor_For_Joint(joint_name, angle, max_force)
        return True
    except Exception:
        pass
    try:
        pyrosim.Set_Motor_For_Joint(joint_name, angle)
        return True
    except Exception:
        return False






VARIANT_RULEBOOK_V4 = True
import os as _os_rb
import json as _json_rb
from pathlib import Path as _Path_rb
from itertools import permutations as _perms_rb

try:
    import pyrosim.pyrosim as pyrosim
except Exception:
    pyrosim = None

try:
    import constants as _c_rb
except Exception:
    _c_rb = None

_rb_cache = {"path": None, "mtime": None, "data": None}
_rb_dbg = _os_rb.getenv("DEBUG_RULEBOOK","").lower() in ("1","true","yes","on")
_rb_dbg_seen = 0

def _rb_load():
    vp = _os_rb.getenv("GAIT_VARIANT_PATH")
    if not vp:
        return None
    try:
        mtime = _Path_rb(vp).stat().st_mtime
    except Exception:
        return None
    c = _rb_cache
    if c["path"] == vp and c["mtime"] == mtime and c["data"] is not None:
        return c["data"]
    try:
        d = _json_rb.loads(_Path_rb(vp).read_text(encoding="utf-8"))
    except Exception:
        return None
    c["path"] = vp
    c["mtime"] = mtime
    c["data"] = d
    return d

def _rb_touch(link_name):
    if pyrosim is None:
        return 0
    try:
        v = pyrosim.Get_Touch_Sensor_Value_For_Link(link_name)
        return 1 if float(v) > 0.5 else 0
    except Exception:
        return 0

def _rb_bins(variant):
    b = variant.get("bins") if isinstance(variant, dict) else None
    if isinstance(b,(int,float)) and b > 0:
        return int(b)
    return 64

def _rb_period_steps():
    v = _os_rb.getenv("RULEBOOK_PERIOD_STEPS")
    if v:
        try:
            return int(float(v))
        except Exception:
            pass
    if _c_rb is not None and hasattr(_c_rb, "SIM_STEPS"):
        try:
            return int(getattr(_c_rb, "SIM_STEPS"))
        except Exception:
            pass
    return 4000

def _rb_motor_index(variant, joint_name):
    if isinstance(joint_name, (bytes, bytearray)):
        try:
            joint_name = joint_name.decode('utf-8', errors='ignore')
        except Exception:
            joint_name = str(joint_name)
    motors = variant.get("motors") if isinstance(variant, dict) else None
    if not isinstance(motors, list):
        return None
    jn = str(joint_name)
    for i, m in enumerate(motors):
        if isinstance(m, str) and m == jn:
            return i
    return None

def _collect_floats(obj, out):
    if isinstance(obj,(int,float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for k in sorted(obj.keys(), key=lambda z: str(z)):
            _collect_floats(obj[k], out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_floats(v, out)

def _extract_angle(action, motor_i, joint_name=None):
    if isinstance(action,(int,float)):
        return float(action)
    if isinstance(action, dict):
        for k in (str(motor_i), motor_i):
            if k in action and isinstance(action[k], (int,float)):
                return float(action[k])
        for k in ("angles","targets","motor_angles","joint_angles"):
            if k in action:
                v = action[k]
                if isinstance(v, list) and motor_i is not None and motor_i < len(v) and isinstance(v[motor_i], (int,float)):
                    return float(v[motor_i])
                if isinstance(v, dict):
                    for kk in (str(motor_i), motor_i):
                        if kk in v and isinstance(v[kk], (int,float)):
                            return float(v[kk])
        for k in ("angle","desiredAngle","target","pos","value","a"):
            if k in action and isinstance(action[k], (int,float)):
                return float(action[k])
    vals=[]
    _collect_floats(action, vals)
    if not vals:
        return None
    if motor_i is not None and motor_i < len(vals):
        return vals[motor_i]
    return vals[0]

def _set_motor(robot_id, joint_name, angle, max_force):
    try:
        import pybullet as p
    except Exception:
        return False
    if isinstance(joint_name, (bytes, bytearray)):
        jn_b = bytes(joint_name)
        try:
            jn_s = joint_name.decode('utf-8', errors='ignore')
        except Exception:
            jn_s = str(joint_name)
    else:
        jn_s = str(joint_name)
        try:
            jn_b = jn_s.encode('utf-8')
        except Exception:
            jn_b = None
    try:
        cache = getattr(_rb_cache, 'joint_map', None)
    except Exception:
        cache = None
    if cache is None:
        cache = {}
        try:
            _rb_cache['joint_map'] = cache
        except Exception:
            pass
    jm = cache.get(int(robot_id))
    if jm is None:
        jm = {}
        try:
            nj = p.getNumJoints(robot_id)
            for j in range(nj):
                name_b = p.getJointInfo(robot_id, j)[1]
                jm[name_b] = j
                try:
                    jm[name_b.decode('utf-8', errors='ignore')] = j
                except Exception:
                    pass
        except Exception:
            return False
        cache[int(robot_id)] = jm
    jidx = None
    if jn_b is not None and jn_b in jm:
        jidx = jm[jn_b]
    elif jn_s in jm:
        jidx = jm[jn_s]
    if jidx is None:
        return False
    try:
        p.setJointMotorControl2(robot_id, jidx, p.POSITION_CONTROL, targetPosition=float(angle), force=float(max_force))
        return True
    except Exception:
        return False


_sensors = ["Torso","BackLeg","FrontLeg"]
_orders = list(_perms_rb(_sensors, 3))

def _pick_inner(outer):
    for order in _orders:
        bits = "".join(str(_rb_touch(s)) for s in order)
        if bits in outer:
            return outer[bits], bits, order
    if "000" in outer:
        return outer["000"], "000", tuple(_sensors)
    return None, None, None

def _rb_rule_angle(robot_id, joint_name, t, max_force):
    _stretch = int(os.getenv('RB_STRETCH','1') or '1')
    if _stretch > 1:
        t = int(t // _stretch)
    _gain = float(os.getenv('RB_GAIN','1.0') or '1.0')
    if isinstance(joint_name, (bytes, bytearray)):
        try:
            joint_name = joint_name.decode('utf-8', errors='ignore')
        except Exception:
            joint_name = str(joint_name)
    global _rb_dbg_seen
    v = _rb_load()
    if not isinstance(v, dict):
        return False
    rules = v.get("rules")
    if not isinstance(rules, dict):
        return False

    bins = _rb_bins(v)
    per = _rb_period_steps()
    try:
        tt = int(t)
    except Exception:
        return False
    if per <= 0:
        per = 4000
    bi = int((tt % per) * bins / per)
    if bi < 0: bi = 0
    if bi >= bins: bi = bins - 1

    outer = rules.get(str(bi)) or rules.get(bi)
    if not isinstance(outer, dict):
        return False

    inner, bits, order = _pick_inner(outer)
    if inner is None:
        return False

    mi = _rb_motor_index(v, str(joint_name))
    ang = _extract_angle(inner, mi, joint_name=str(joint_name))
    if ang is None:
        return False

    if _rb_dbg and _rb_dbg_seen < 6:
        _rb_dbg_seen += 1
        print("[RB]", _os_rb.getenv("GAIT_VARIANT_PATH",""), "t", tt, "bin", bi, "bits", bits, "order", order, "joint", joint_name, "mi", mi, "ang", ang)

    return _set_motor(robot_id, joint_name, ang, max_force)

class MOTOR:
    def __init__(self, jointName):
        # Keep original key type for pyrosim dict lookups (bytes or str)
        self.jointName = jointName
        self.jointNameStr = jointName.decode() if isinstance(jointName, (bytes, bytearray)) else str(jointName)

        # Allow quick overrides from env so demos are easy to tune
        base_f = float(os.getenv("GAIT_FREQ_HZ", str(getattr(c, "GAIT_FREQ_HZ", 1.0))))
        amp    = float(os.getenv("GAIT_AMPLITUDE", str(getattr(c, "GAIT_AMPLITUDE", 0.7))))

        demo_pure = os.getenv("DEMO_PURE_SINE", "0") == "1"
        half_demo = os.getenv("HALF_FREQ_DEMO", "0") == "1"

        # Defaults (locomotion-oriented)
        if "BackLeg" in self.jointNameStr:
            offset = float(getattr(c, "BACK_OFFSET", -0.25))
            phase  = float(getattr(c, "BACK_PHASE", 0.0))
            freq   = base_f
            sign   = +1.0
        elif "FrontLeg" in self.jointNameStr:
            offset = float(getattr(c, "FRONT_OFFSET", 0.20))
            phase  = float(getattr(c, "FRONT_PHASE", 3.1415926535))
            freq   = base_f
            sign   = -1.0
        else:
            offset, phase, freq, sign = 0.0, 0.0, base_f, 1.0

        # DEMO: make it visually countable (pure sine, no offsets/phases)
        if demo_pure:
            offset = 0.0
            phase  = 0.0
            sign   = +1.0

        # Assignment demo: one leg half frequency of the other
        # (FrontLeg slower by 2×)
        if half_demo and "FrontLeg" in self.jointNameStr:
            freq *= 0.5

        self.freq_hz = freq * (0.5 if "back" in str(self.jointName).lower() else 1.0)  # for debugging


        dt = float(getattr(c, "DT", 1/240))
        t = np.arange(c.SIM_STEPS) * dt
        self.motorValues = offset + sign * amp * np.sin(2.0 * np.pi * freq * t + phase)

        # Keep angles reasonable
        self.motorValues = np.clip(self.motorValues, -1.3, 1.3)

    def Set_Value(self, robot, t: int, max_force: float):
        import os, math
        try:
            jn = getattr(self, 'jointName', None)
            if jn is None:
                return
            if os.getenv('GAIT_VARIANT_PATH',''):
                if '_rb_rule_angle' in globals():
                    if _rb_rule_angle(jn, t, max_force):
                        return
        except Exception:
            pass
        c = __import__('constants')
        jn = getattr(self, 'jointName', None)
        if jn is None:
            return
        js = jn.decode('utf-8','replace') if isinstance(jn,(bytes,bytearray)) else str(jn)
        amp = float(getattr(c, 'GAIT_AMPLITUDE', 0.5))
        freq = float(getattr(c, 'GAIT_FREQ_HZ', 1.0))
        dt = float(getattr(c, 'DT', 1/240))
        if ('BackLeg' in js) or ('Back' in js):
            offset = float(getattr(c, 'BACK_OFFSET', 0.0))
            phase = float(getattr(c, 'BACK_PHASE', 0.0))
        else:
            offset = float(getattr(c, 'FRONT_OFFSET', 0.0))
            phase = float(getattr(c, 'FRONT_PHASE', 0.0))
        ang = offset + amp * math.sin(2.0 * math.pi * freq * (float(t) * dt) + phase)
        if ang > 1.3:
            ang = 1.3
        if ang < -1.3:
            ang = -1.3
        _set_motor(robot.robotId, jn, ang, max_force)
        return
