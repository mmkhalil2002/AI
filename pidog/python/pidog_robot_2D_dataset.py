#!/usr/bin/env python3
"""
dog_pose_math_to_motor.py
=====================================================================
YOU ASKED FOR:
  ✅ EXACT set of pose functions (calculation-only):
      calculate_left_knee_angle(...)
      calculate_right_knee_angle(...)
      calculate_hip_center(...)
      calculate_shoulder_center(...)
      calculate_torso_tilt_angle(...)
      calculate_front_paw_width(...)
      calculate_hind_paw_width(...)
      calculate_lift_score(...)

  ✅ Each function has:
      - Theoretical calculation comments
      - Simple ASCII diagram

  ✅ After executing each function, the return value is translated
     into a motor action using motor-specific functions (separate layer).

  ✅ Loop over CSV rows and execute motor actions for each pose entry.

IMPORTANT SAFETY:
  - This script is hardware-agnostic (no PiDog library).
  - DRY_RUN=True prints motor commands only (safe).
  - To actually move motors, implement MotorDriver.send_servo().
=====================================================================

RUN:
  Windows:  py dog_pose_math_to_motor.py
  Linux:    python3 dog_pose_math_to_motor.py
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
import math
import time
from typing import Dict, Tuple


# =====================================================================
# USER SETTINGS
# =====================================================================
CSV_PATH = Path("deeplabcut") / "sample_pose_dataset" / "dog_pose_2d_aligned.csv"

DRY_RUN = True
PRINT_FEATURES = True
SECONDS_PER_ROW = 1.0


# =====================================================================
# DATA TYPES
# =====================================================================
@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class PoseFeatureValues:
    """All calculated values (outputs of your exact function set)."""
    pose_label: str

    left_knee_angle_deg: float
    right_knee_angle_deg: float

    hip_center: Point2D
    shoulder_center: Point2D

    torso_tilt_angle_deg: float

    front_paw_width: float
    hind_paw_width: float

    lift_score: float


# =====================================================================
# BASIC GEOMETRY HELPERS (INTERNAL)
# =====================================================================
def _vec(a: Point2D, b: Point2D) -> Point2D:
    return Point2D(b.x - a.x, b.y - a.y)

def _dot(u: Point2D, v: Point2D) -> float:
    return u.x * v.x + u.y * v.y

def _norm(u: Point2D) -> float:
    return math.sqrt(u.x*u.x + u.y*u.y)

def _dist(a: Point2D, b: Point2D) -> float:
    return _norm(_vec(a, b))

def _midpoint(a: Point2D, b: Point2D) -> Point2D:
    return Point2D((a.x + b.x)/2.0, (a.y + b.y)/2.0)

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _map_range(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Linear map x from [in_min..in_max] to [out_min..out_max]."""
    if abs(in_max - in_min) < 1e-9:
        return (out_min + out_max) / 2.0
    t = (x - in_min) / (in_max - in_min)
    return out_min + t * (out_max - out_min)


# =====================================================================
# 1) YOUR EXACT POSE FUNCTIONS (CALCULATION-ONLY)
# =====================================================================

def calculate_left_knee_angle(left_hip: Point2D, left_knee: Point2D, left_hind_paw: Point2D) -> float:
    """
    PURPOSE (DOG MEANING):
      Measures how "bent" the LEFT hind leg is at the knee.

    SIMPLE DIAGRAM (hind leg):
        left_hip (H) ●
                      \
                       \
                        ● left_knee (K)   <-- angle θ at K
                         \
                          \
                           ● left_hind_paw (P)

    THEORY:
      We want the angle at the knee K formed by points (H-K-P).
      Define:
        u = H - K   (vector from knee to hip)
        v = P - K   (vector from knee to paw)

      Then:
        cos(θ) = (u·v) / (|u||v|)
        θ = arccos( cos(θ) )

    INTERPRETATION (typical):
      - Larger θ  -> leg more extended (standing)
      - Smaller θ -> leg more flexed (sitting / crouching / jump prep)
    """
    u = _vec(left_knee, left_hip)       # H - K
    v = _vec(left_knee, left_hind_paw)  # P - K
    nu = _norm(u)
    nv = _norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return float("nan")
    c = _dot(u, v) / (nu * nv)
    c = max(-1.0, min(1.0, c))  # numeric safety
    return math.degrees(math.acos(c))


def calculate_right_knee_angle(right_hip: Point2D, right_knee: Point2D, right_hind_paw: Point2D) -> float:
    """
    PURPOSE:
      Same as calculate_left_knee_angle, but for RIGHT hind leg.

    DIAGRAM:
        right_hip ●
                   \
                    ● right_knee  <-- θ
                     \
                      ● right_hind_paw

    THEORY:
      θ = angle( right_hip, right_knee, right_hind_paw )
    """
    u = _vec(right_knee, right_hip)
    v = _vec(right_knee, right_hind_paw)
    nu = _norm(u)
    nv = _norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return float("nan")
    c = _dot(u, v) / (nu * nv)
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def calculate_hip_center(left_hip: Point2D, right_hip: Point2D) -> Point2D:
    """
    PURPOSE:
      Creates a stable "body reference point" from the two hip points.

    DIAGRAM:
      left_hip ●         ● right_hip
              \         /
               \       /
                ● hip_center

    THEORY:
      hip_center = midpoint(left_hip, right_hip)
               = ((xL+xR)/2, (yL+yR)/2)

    INTERPRETATION:
      - Useful as the robot "body center" reference.
      - Helps reduce noise vs using one hip only.
    """
    return _midpoint(left_hip, right_hip)


def calculate_shoulder_center(left_shoulder: Point2D, right_shoulder: Point2D) -> Point2D:
    """
    PURPOSE:
      Creates a stable "upper-body reference point" from both shoulders.

    DIAGRAM:
      left_shoulder ●         ● right_shoulder
                   \         /
                    \       /
                     ● shoulder_center

    THEORY:
      shoulder_center = midpoint(left_shoulder, right_shoulder)
    """
    return _midpoint(left_shoulder, right_shoulder)


def calculate_torso_tilt_angle(shoulder_center: Point2D, hip_center: Point2D) -> float:
    """
    PURPOSE:
      Measures torso tilt angle (proxy for body lean in 2D).

    DIAGRAM:
      shoulder_center ● -----------> ● hip_center
                          tilt φ

    THEORY:
      torso vector T = hip_center - shoulder_center
      φ = atan2(Ty, Tx)

    NOTE (image coordinates):
      - If y increases downward (usual images),
        positive/negative meaning of 'up/down' depends on your camera.
      - Still consistent for relative changes.

    INTERPRETATION:
      - Larger |φ| can indicate leaning / crouching / jump posture.
    """
    T = _vec(shoulder_center, hip_center)
    return math.degrees(math.atan2(T.y, T.x))


def calculate_front_paw_width(left_front_paw: Point2D, right_front_paw: Point2D) -> float:
    """
    PURPOSE:
      Measures how wide the FRONT stance is.

    DIAGRAM:
      left_front_paw ● <------ width ------> ● right_front_paw

    THEORY:
      width = distance(LF, RF)
            = sqrt((xR-xL)^2 + (yR-yL)^2)

    INTERPRETATION:
      - Wide stance can be stability / standing.
      - Narrow stance can happen in crouch / jump / tight posture.
    """
    return _dist(left_front_paw, right_front_paw)


def calculate_hind_paw_width(left_hind_paw: Point2D, right_hind_paw: Point2D) -> float:
    """
    PURPOSE:
      Measures how wide the HIND stance is.

    DIAGRAM:
      left_hind_paw ● <------ width ------> ● right_hind_paw

    THEORY:
      width = distance(LH, RH)

    INTERPRETATION:
      - Sitting often changes hind stance width.
      - Jump prep can reduce width (legs tuck).
    """
    return _dist(left_hind_paw, right_hind_paw)


def calculate_lift_score(
    hip_center: Point2D,
    left_front_paw: Point2D,
    right_front_paw: Point2D,
    left_hind_paw: Point2D,
    right_hind_paw: Point2D
) -> float:
    """
    PURPOSE:
      Approximate "body lift" relative to paws in 2D.

    DIAGRAM (vertical idea):
          hip_center ●
      paws ●   ●   ●   ●

    THEORY:
      avg_paws_y = mean(y of all 4 paws)
      lift_score = hip_center_y - avg_paws_y

    IMAGE COORDINATE NOTE:
      Usually y increases downward:
        - If hip goes UP (smaller y), hip_center_y decreases
        - So lift_score becomes more NEGATIVE (or changes significantly)

    INTERPRETATION:
      - Large change in lift_score can indicate jumping / crouching transitions.
    """
    avg_paws_y = (left_front_paw.y + right_front_paw.y + left_hind_paw.y + right_hind_paw.y) / 4.0
    return hip_center.y - avg_paws_y


# =====================================================================
# 2) MOTOR LAYER (motor-specific functions)
# =====================================================================

@dataclass(frozen=True)
class JointLimit:
    min_deg: float
    max_deg: float
    neutral_deg: float


class MotorSpec:
    """
    Defines motor ranges and channel mapping.
    Change this class when your motors change.
    Pose functions remain unchanged.
    """
    def __init__(self):
        # Example logical joints (edit as needed)
        self.limits: Dict[str, JointLimit] = {
            "FL_HIP": JointLimit(10, 90, 50),
            "FR_HIP": JointLimit(10, 90, 50),
            "HL_HIP": JointLimit(10, 90, 50),
            "HR_HIP": JointLimit(10, 90, 50),

            "HL_KNEE": JointLimit(10, 90, 50),
            "HR_KNEE": JointLimit(10, 90, 50),

            "HEAD_TILT": JointLimit(20, 110, 70),
        }

        # Example channels (edit to match your wiring)
        self.channel: Dict[str, int] = {
            "FL_HIP": 0,
            "FR_HIP": 1,
            "HL_HIP": 2,
            "HR_HIP": 3,
            "HL_KNEE": 4,
            "HR_KNEE": 5,
            "HEAD_TILT": 6,
        }


class MotorDriver:
    """
    Hardware output. Replace send_servo() with your low-level PWM/I2C code.
    """
    def __init__(self, spec: MotorSpec, dry_run: bool = True):
        self.spec = spec
        self.dry_run = dry_run

    def send_servo(self, channel: int, angle_deg: float) -> None:
        # TODO: Implement for your servo controller (PCA9685/GPIO/serial)
        print(f"    servo[{channel}] = {angle_deg:6.1f}°")

    def execute_targets(self, targets: Dict[str, float]) -> None:
        for joint, deg in targets.items():
            if joint not in self.spec.channel or joint not in self.spec.limits:
                continue
            ch = self.spec.channel[joint]
            lim = self.spec.limits[joint]
            deg = _clamp(deg, lim.min_deg, lim.max_deg)
            if self.dry_run:
                print(f"[DRY] {joint:9s} -> {deg:6.1f}° (ch {ch})")
            else:
                self.send_servo(ch, deg)


# ---- Motor-specific translation functions (one per feature) ----

def motor_action_from_left_knee_angle(spec: MotorSpec, left_knee_angle_deg: float) -> Dict[str, float]:
    """
    FEATURE -> MOTOR:
      left_knee_angle_deg -> HL_KNEE servo target

    Mapping concept:
      pose knee angle: 70(bent) -> 160(straight)
      servo angle:     min(bent) -> max(straight)
    """
    lim = spec.limits["HL_KNEE"]
    target = _map_range(left_knee_angle_deg, 70, 160, lim.min_deg, lim.max_deg)
    return {"HL_KNEE": _clamp(target, lim.min_deg, lim.max_deg)}


def motor_action_from_right_knee_angle(spec: MotorSpec, right_knee_angle_deg: float) -> Dict[str, float]:
    lim = spec.limits["HR_KNEE"]
    target = _map_range(right_knee_angle_deg, 70, 160, lim.min_deg, lim.max_deg)
    return {"HR_KNEE": _clamp(target, lim.min_deg, lim.max_deg)}


def motor_action_from_torso_tilt_angle(spec: MotorSpec, torso_tilt_deg: float) -> Dict[str, float]:
    """
    FEATURE -> MOTOR:
      torso tilt -> hip compensation (balance)

    Simple stabilizer:
      delta = clamp(torso_tilt/30, -1..1) * 8deg
      front hips = neutral + delta
      hind hips  = neutral - delta
    """
    delta = _clamp(torso_tilt_deg / 30.0, -1.0, 1.0) * 8.0

    out = {}
    for j in ["FL_HIP", "FR_HIP"]:
        lim = spec.limits[j]
        out[j] = _clamp(lim.neutral_deg + delta, lim.min_deg, lim.max_deg)
    for j in ["HL_HIP", "HR_HIP"]:
        lim = spec.limits[j]
        out[j] = _clamp(lim.neutral_deg - delta, lim.min_deg, lim.max_deg)
    return out


def motor_action_from_front_paw_width(spec: MotorSpec, front_width: float) -> Dict[str, float]:
    """
    OPTIONAL FEATURE -> MOTOR:
      front_paw_width can be used to widen/narrow stance,
      but many simple quadrupeds don't have "sideways" DOF.

    For safety, we don't force anything here by default.
    """
    _ = spec
    _ = front_width
    return {}


def motor_action_from_hind_paw_width(spec: MotorSpec, hind_width: float) -> Dict[str, float]:
    """
    OPTIONAL FEATURE -> MOTOR:
      hind_paw_width may also control stance width if you have abduction servos.

    Default: no-op.
    """
    _ = spec
    _ = hind_width
    return {}


def motor_action_from_lift_score(spec: MotorSpec, lift_score: float) -> Dict[str, float]:
    """
    OPTIONAL FEATURE -> MOTOR:
      lift_score can influence crouch/extend behavior.

    Example idea:
      If lift_score indicates body is "higher" (more negative), slightly extend knees.
      If body is "lower" (less negative / positive), slightly bend knees.

    We'll do a tiny safe adjustment around knee neutral.
    """
    hl = spec.limits["HL_KNEE"]
    hr = spec.limits["HR_KNEE"]

    # Map lift_score into [-1..1] range using a conservative window:
    # (You can tune these numbers based on your dataset)
    t = _map_range(lift_score, -60, 20, -1.0, 1.0)
    t = _clamp(t, -1.0, 1.0)

    # small adjustment (±5 degrees)
    adj = t * 5.0

    return {
        "HL_KNEE": _clamp(hl.neutral_deg + adj, hl.min_deg, hl.max_deg),
        "HR_KNEE": _clamp(hr.neutral_deg + adj, hr.min_deg, hr.max_deg),
    }


def merge_motor_targets(*cmds: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for c in cmds:
        out.update(c)
    return out


# =====================================================================
# CSV IO + MAIN LOOP
# =====================================================================
REQUIRED_COLS = [
    "pose",
    "nose_x","nose_y",
    "left_shoulder_x","left_shoulder_y",
    "right_shoulder_x","right_shoulder_y",
    "left_hip_x","left_hip_y",
    "right_hip_x","right_hip_y",
    "left_knee_x","left_knee_y",
    "right_knee_x","right_knee_y",
    "left_front_paw_x","left_front_paw_y",
    "right_front_paw_x","right_front_paw_y",
    "left_hind_paw_x","left_hind_paw_y",
    "right_hind_paw_x","right_hind_paw_y",
]

def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path.resolve()}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for c in REQUIRED_COLS:
            if c not in (rdr.fieldnames or []):
                raise ValueError(f"Missing column '{c}'. Found: {rdr.fieldnames}")
        return list(rdr)

def pt(row: dict, xk: str, yk: str) -> Point2D:
    return Point2D(float(row[xk]), float(row[yk]))


def main() -> None:
    rows = read_rows(CSV_PATH)
    spec = MotorSpec()
    driver = MotorDriver(spec, dry_run=DRY_RUN)

    print(f"Loaded {len(rows)} rows from: {CSV_PATH.resolve()}")
    print(f"DRY_RUN={DRY_RUN}, SECONDS_PER_ROW={SECONDS_PER_ROW}\n")

    for i, row in enumerate(rows, start=1):
        pose_label = row["pose"]

        # --- read keypoints ---
        nose = pt(row, "nose_x", "nose_y")
        ls = pt(row, "left_shoulder_x", "left_shoulder_y")
        rs = pt(row, "right_shoulder_x", "right_shoulder_y")
        lh = pt(row, "left_hip_x", "left_hip_y")
        rh = pt(row, "right_hip_x", "right_hip_y")
        lk = pt(row, "left_knee_x", "left_knee_y")
        rk = pt(row, "right_knee_x", "right_knee_y")
        lfp = pt(row, "left_front_paw_x", "left_front_paw_y")
        rfp = pt(row, "right_front_paw_x", "right_front_paw_y")
        lhp = pt(row, "left_hind_paw_x", "left_hind_paw_y")
        rhp = pt(row, "right_hind_paw_x", "right_hind_paw_y")

        # =========================================================
        # (A) EXECUTE YOUR EXACT CALCULATION FUNCTIONS
        # =========================================================
        left_knee_angle = calculate_left_knee_angle(lh, lk, lhp)
        right_knee_angle = calculate_right_knee_angle(rh, rk, rhp)

        hip_center = calculate_hip_center(lh, rh)
        shoulder_center = calculate_shoulder_center(ls, rs)

        torso_tilt = calculate_torso_tilt_angle(shoulder_center, hip_center)

        front_width = calculate_front_paw_width(lfp, rfp)
        hind_width = calculate_hind_paw_width(lhp, rhp)

        lift_score = calculate_lift_score(hip_center, lfp, rfp, lhp, rhp)

        if PRINT_FEATURES:
            print("--------------------------------------------------")
            print(f"Row {i}/{len(rows)} pose='{pose_label}'")
            print(f"  calculate_left_knee_angle(...)   = {left_knee_angle:7.2f} deg")
            print(f"  calculate_right_knee_angle(...)  = {right_knee_angle:7.2f} deg")
            print(f"  calculate_hip_center(...)        = ({hip_center.x:.2f},{hip_center.y:.2f})")
            print(f"  calculate_shoulder_center(...)   = ({shoulder_center.x:.2f},{shoulder_center.y:.2f})")
            print(f"  calculate_torso_tilt_angle(...)  = {torso_tilt:7.2f} deg")
            print(f"  calculate_front_paw_width(...)   = {front_width:7.2f}")
            print(f"  calculate_hind_paw_width(...)    = {hind_width:7.2f}")
            print(f"  calculate_lift_score(...)        = {lift_score:7.2f}")

        # =========================================================
        # (B) TRANSLATE EACH RETURN VALUE INTO MOTOR ACTIONS
        #     using motor-specific functions (separate layer)
        # =========================================================
        motor_targets = merge_motor_targets(
            motor_action_from_left_knee_angle(spec, left_knee_angle),
            motor_action_from_right_knee_angle(spec, right_knee_angle),
            motor_action_from_torso_tilt_angle(spec, torso_tilt),
            motor_action_from_front_paw_width(spec, front_width),
            motor_action_from_hind_paw_width(spec, hind_width),
            motor_action_from_lift_score(spec, lift_score),
        )

        # Optional: nose->head tilt (not in your requested function list, but useful)
        nose_minus_hip_y = nose.y - hip_center.y
        ht = spec.limits["HEAD_TILT"]
        head_target = _map_range(nose_minus_hip_y, -80, 20, ht.max_deg, ht.min_deg)
        motor_targets["HEAD_TILT"] = _clamp(head_target, ht.min_deg, ht.max_deg)

        # =========================================================
        # (C) EXECUTE MOTOR TARGETS (hardware layer)
        # =========================================================
        print("Motor targets:")
        driver.execute_targets(motor_targets)

        time.sleep(SECONDS_PER_ROW)

    print("\n✅ Done. Processed all CSV rows.")


if __name__ == "__main__":
    main()