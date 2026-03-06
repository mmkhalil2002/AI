#!/usr/bin/env python3
"""
dog_pose_math_to_motor_newformat_commented.py
=====================================================================
UPDATED FOR YOUR NEW CSV FORMAT:
  /mnt/data/dog2d_keypoints_clean.csv

YOU REQUESTED:
  ✅ Add full mathematical foundation comments for EACH function
  ✅ Include ASCII diagram for EACH function (where meaningful)
  ✅ Keep calculation layer separate from motor layer
  ✅ Keep robust handling of missing points (0,0) => NaN => skip motor

NEW FILE COLUMNS (confirmed):
  motion_type, seq, split,
  FL_PW_x/y, FL_KN_x/y, FL_EL_x/y,
  FR_PW_x/y, FR_KN_x/y, FR_EL_x/y,
  RL_PW_x/y, RL_KN_x/y, RL_EL_x/y,
  RR_PW_x/y, RR_KN_x/y, RR_EL_x/y

MAPPING USED IN THIS SCRIPT:
  - Shoulder center uses (FL_EL, FR_EL)
  - Hip center uses (RL_EL, RR_EL)
  - Left knee angle uses (RL_EL as hip, RL_KN as knee, RL_PW as paw)
  - Right knee angle uses (RR_EL as hip, RR_KN as knee, RR_PW as paw)
  - Front paw width uses (FL_PW, FR_PW)
  - Hind paw width uses (RL_PW, RR_PW)
  - Lift score uses hip_center vs avg paws y

IMPORTANT SAFETY:
  - Many rows contain missing points as (0,0).
  - If any needed point is missing, the function returns NaN for that feature.
  - Motor actions that depend on NaN are skipped.

RUN:
  Windows:  py dog_pose_math_to_motor_newformat_commented.py
  Linux:    python3 dog_pose_math_to_motor_newformat_commented.py
=====================================================================
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
import math
import time
from typing import Dict


# =====================================================================
# USER SETTINGS
# =====================================================================
CSV_PATH = Path("/mnt/data/dog2d_keypoints_clean.csv")

DRY_RUN = True          # True = prints commands only (safe)
PRINT_FEATURES = True   # True = prints computed features each row
SECONDS_PER_ROW = 0.2   # playback speed


# =====================================================================
# DATA TYPES
# =====================================================================
@dataclass(frozen=True)
class Point2D:
    """
    Mathematical object: A point in the Euclidean plane ℝ².

    We represent a keypoint measured from an image frame as:
        P = (x, y)

    Note:
      In most images:
        x increases to the right
        y increases downward
      (This affects interpretation of "up/down" but not the math itself.)
    """
    x: float
    y: float


# =====================================================================
# BASIC GEOMETRY HELPERS (INTERNAL)
# =====================================================================

def _vec(a: Point2D, b: Point2D) -> Point2D:
    """
    VECTOR CONSTRUCTION
    -------------------
    Creates a vector from point 'a' to point 'b':

        v = b - a
        v = (b.x - a.x, b.y - a.y)

    ASCII:
        a ● -----> ● b
              v

    Why used?
      - Angles are computed from vectors.
      - Distances are computed from vectors.
    """
    return Point2D(b.x - a.x, b.y - a.y)


def _dot(u: Point2D, v: Point2D) -> float:
    """
    DOT PRODUCT
    -----------
    In ℝ², dot product is:
        u·v = u.x*v.x + u.y*v.y

    Geometric meaning:
      u·v = |u| |v| cos(θ)

    ASCII:
        u ↗
         \ θ
          ↘ v

    Uses:
      - Computing angle between vectors using cos(θ)
      - Similarity/projection
    """
    return u.x * v.x + u.y * v.y


def _norm(u: Point2D) -> float:
    """
    VECTOR NORM (L2 magnitude)
    --------------------------
    |u| = sqrt(u.x² + u.y²)

    ASCII:
        u = (ux, uy)
        |u| is the length of the arrow

    Uses:
      - Normalize vectors
      - Distance between points
      - Angle calculation denominator
    """
    return math.sqrt(u.x*u.x + u.y*u.y)


def _dist(a: Point2D, b: Point2D) -> float:
    """
    EUCLIDEAN DISTANCE BETWEEN TWO POINTS
    ------------------------------------
    distance(a,b) = ||b - a||

    Formula:
      d = sqrt((bx-ax)² + (by-ay)²)

    ASCII:
        a ● -------- d -------- ● b

    Uses:
      - Stance width
      - Step length proxy
      - Scaling signals
    """
    return _norm(_vec(a, b))


def _midpoint(a: Point2D, b: Point2D) -> Point2D:
    """
    MIDPOINT OF TWO POINTS
    ----------------------
    midpoint(a,b) = ((ax+bx)/2 , (ay+by)/2)

    ASCII:
        a ● ---- m ---- ● b
              ^
              midpoint

    Uses:
      - Stable body reference points (hip_center, shoulder_center)
    """
    return Point2D((a.x + b.x)/2.0, (a.y + b.y)/2.0)


def _clamp(x: float, lo: float, hi: float) -> float:
    """
    CLAMP OPERATOR
    --------------
    Restrict x to interval [lo, hi].

        clamp(x) =
          lo if x < lo
          hi if x > hi
          x otherwise

    Uses:
      - Servo safety: never command beyond physical limits
      - Normalization safety
    """
    return max(lo, min(hi, x))


def _is_valid(p: Point2D) -> bool:
    """
    MISSING DATA CHECK
    ------------------
    Your dataset uses (0,0) to represent missing keypoints.
    We treat that as invalid.

    If point is invalid, calculations return NaN so the motor layer skips it.
    """
    return not (abs(p.x) < 1e-12 and abs(p.y) < 1e-12)


def _map_range(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """
    LINEAR RANGE MAPPING
    --------------------
    Map x from interval [in_min, in_max] into [out_min, out_max].

    Formula:
        t = (x - in_min) / (in_max - in_min)
        y = out_min + t * (out_max - out_min)

    ASCII:
      in:   in_min ----- x ----- in_max
      out:  out_min ---- y ---- out_max

    Used to translate:
      - measured angles/distances -> servo degrees

    Safety:
      - If x is NaN or in_min==in_max, returns NaN or midpoint.
    """
    if not math.isfinite(x):
        return float("nan")
    if abs(in_max - in_min) < 1e-9:
        return (out_min + out_max) / 2.0
    t = (x - in_min) / (in_max - in_min)
    return out_min + t * (out_max - out_min)


# =====================================================================
# 1) POSE FUNCTIONS (CALCULATION-ONLY)
# =====================================================================

def calculate_left_knee_angle(left_hip: Point2D, left_knee: Point2D, left_hind_paw: Point2D) -> float:
    """
    LEFT KNEE ANGLE (H-K-P)
    ======================
    PURPOSE:
      Measures how bent the LEFT hind leg is at the knee.

    DIAGRAM (hind leg):
        Hip (H) ●
                 \
                  \
                   ● Knee (K)  <-- θ is measured here
                    \
                     \
                      ● Paw (P)

    MATHEMATICAL FOUNDATION:
      We want the angle θ at point K for triangle (H, K, P).

      Define vectors originating at K:
        u = H - K   (vector from K to hip)
        v = P - K   (vector from K to paw)

      Angle between u and v:
        cos(θ) = (u·v) / (|u| |v|)
        θ = arccos( clamp(cos(θ), -1, 1) )
      Then convert to degrees.

    NOTES:
      - If any point missing => NaN
      - If vector length is near zero => NaN
    """
    if not (_is_valid(left_hip) and _is_valid(left_knee) and _is_valid(left_hind_paw)):
        return float("nan")

    u = _vec(left_knee, left_hip)       # H - K
    v = _vec(left_knee, left_hind_paw)  # P - K

    nu = _norm(u)
    nv = _norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return float("nan")

    c = _dot(u, v) / (nu * nv)          # cos(theta)
    c = max(-1.0, min(1.0, c))          # numeric safety
    return math.degrees(math.acos(c))


def calculate_right_knee_angle(right_hip: Point2D, right_knee: Point2D, right_hind_paw: Point2D) -> float:
    """
    RIGHT KNEE ANGLE (H-K-P)
    =======================
    Same math as left knee, applied to the RIGHT hind leg.

    DIAGRAM:
        Hip (H) ●
                 \
                  ● Knee (K)  <-- θ
                   \
                    ● Paw (P)

    FOUNDATION:
      u = H - K
      v = P - K
      θ = arccos( (u·v) / (|u||v|) )
    """
    if not (_is_valid(right_hip) and _is_valid(right_knee) and _is_valid(right_hind_paw)):
        return float("nan")

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
    HIP CENTER (MIDPOINT)
    =====================
    PURPOSE:
      Construct a stable reference for the pelvis/torso base.

    DIAGRAM:
      left_hip ● ----- ● right_hip
                 \   /
                  \ /
                  ● hip_center

    FOUNDATION:
      hip_center = (left_hip + right_hip) / 2
                 = midpoint(left_hip, right_hip)

    NOTES:
      If either hip missing => returns (NaN, NaN)
    """
    if not (_is_valid(left_hip) and _is_valid(right_hip)):
        return Point2D(float("nan"), float("nan"))
    return _midpoint(left_hip, right_hip)


def calculate_shoulder_center(left_shoulder: Point2D, right_shoulder: Point2D) -> Point2D:
    """
    SHOULDER CENTER (MIDPOINT)
    ==========================
    PURPOSE:
      Construct stable reference point for upper torso.

    DIAGRAM:
      left_shoulder ● ----- ● right_shoulder
                    \     /
                     \   /
                      ● shoulder_center

    FOUNDATION:
      shoulder_center = midpoint(left_shoulder, right_shoulder)
    """
    if not (_is_valid(left_shoulder) and _is_valid(right_shoulder)):
        return Point2D(float("nan"), float("nan"))
    return _midpoint(left_shoulder, right_shoulder)


def calculate_torso_tilt_angle(shoulder_center: Point2D, hip_center: Point2D) -> float:
    """
    TORSO TILT ANGLE
    ================
    PURPOSE:
      Estimate torso lean direction using two reference points.

    DIAGRAM:
        shoulder_center (S) ●
                             \
                              \  vector T = H - S
                               \
                                ● hip_center (H)

      We measure angle φ of vector T relative to +x axis.

    FOUNDATION:
      T = hip_center - shoulder_center
      φ = atan2(T.y, T.x)

    INTERPRETATION:
      - φ changes when dog leans forward/backward in the 2D plane
      - In image coordinates, y direction is camera-dependent
    """
    if not (_is_valid(shoulder_center) and _is_valid(hip_center)):
        return float("nan")
    T = _vec(shoulder_center, hip_center)
    return math.degrees(math.atan2(T.y, T.x))


def calculate_front_paw_width(left_front_paw: Point2D, right_front_paw: Point2D) -> float:
    """
    FRONT PAW WIDTH (STANCE WIDTH)
    ==============================
    PURPOSE:
      Measures front stance width.

    DIAGRAM:
      LF ● <------ width ------> ● RF

    FOUNDATION:
      width = ||RF - LF|| = sqrt((dx)^2 + (dy)^2)
    """
    if not (_is_valid(left_front_paw) and _is_valid(right_front_paw)):
        return float("nan")
    return _dist(left_front_paw, right_front_paw)


def calculate_hind_paw_width(left_hind_paw: Point2D, right_hind_paw: Point2D) -> float:
    """
    HIND PAW WIDTH (STANCE WIDTH)
    =============================
    PURPOSE:
      Measures hind stance width.

    DIAGRAM:
      LH ● <------ width ------> ● RH

    FOUNDATION:
      width = ||RH - LH||
    """
    if not (_is_valid(left_hind_paw) and _is_valid(right_hind_paw)):
        return float("nan")
    return _dist(left_hind_paw, right_hind_paw)


def calculate_lift_score(
    hip_center: Point2D,
    left_front_paw: Point2D,
    right_front_paw: Point2D,
    left_hind_paw: Point2D,
    right_hind_paw: Point2D
) -> float:
    """
    LIFT SCORE (2D BODY HEIGHT PROXY)
    =================================
    PURPOSE:
      Approximate body "lift" relative to paws.

    DIAGRAM (concept):
          hip_center ●
      paw ●      paw ●
      paw ●      paw ●

    FOUNDATION:
      avg_paws_y = mean( y of all valid paws )
      lift_score = hip_center.y - avg_paws_y

    IMPORTANT IMAGE NOTE:
      If y increases DOWN (most images):
        - If body goes UP, hip_center.y decreases
        - So lift_score becomes more NEGATIVE

    NOTES:
      - Needs hip_center valid
      - Needs at least 2 valid paws
    """
    if not _is_valid(hip_center):
        return float("nan")

    paws = [left_front_paw, right_front_paw, left_hind_paw, right_hind_paw]
    valid_paws = [p for p in paws if _is_valid(p)]
    if len(valid_paws) < 2:
        return float("nan")

    avg_paws_y = sum(p.y for p in valid_paws) / float(len(valid_paws))
    return hip_center.y - avg_paws_y


# =====================================================================
# 2) MOTOR LAYER (SERVO SPECS + FEATURE->MOTOR MAPPING)
# =====================================================================
@dataclass(frozen=True)
class JointLimit:
    """
    Servo mechanical limits:
      min_deg: minimum safe servo angle
      max_deg: maximum safe servo angle
      neutral_deg: comfortable default
    """
    min_deg: float
    max_deg: float
    neutral_deg: float


class MotorSpec:
    """
    MOTOR SPECIFICATION (hardware configuration)
    ============================================
    This is the ONLY place you edit when your wiring or servo limits change.

    We represent each logical joint by:
      - A servo channel index (PCA9685 channel for example)
      - A safe angle range (min..max) and a neutral position

    NOTE:
      The pose math layer produces angle/width scores in "human units".
      This class defines how we convert those scores to servo degrees.
    """
    def __init__(self):
        self.limits: Dict[str, JointLimit] = {
            "FL_HIP": JointLimit(10, 90, 50),
            "FR_HIP": JointLimit(10, 90, 50),
            "HL_HIP": JointLimit(10, 90, 50),
            "HR_HIP": JointLimit(10, 90, 50),
            "HL_KNEE": JointLimit(10, 90, 50),
            "HR_KNEE": JointLimit(10, 90, 50),
            "HEAD_TILT": JointLimit(20, 110, 70),
        }

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
    HARDWARE DRIVER ABSTRACTION
    ===========================
    This layer sends commands to hardware.

    In DRY_RUN mode:
      - We only print intended commands (safe testing).

    In REAL mode:
      - Implement send_servo(channel, angle_deg) for:
          PCA9685 I2C board
          Pigpio
          Serial bus servos
          etc.
    """
    def __init__(self, spec: MotorSpec, dry_run: bool = True):
        self.spec = spec
        self.dry_run = dry_run

    def send_servo(self, channel: int, angle_deg: float) -> None:
        """
        LOW LEVEL OUTPUT (TO IMPLEMENT)
        ------------------------------
        This is where you'd talk to actual hardware.

        Example for PCA9685:
          set_pwm(channel, pulse_for_angle(angle_deg))

        Currently prints to console for safety.
        """
        print(f"    servo[{channel}] = {angle_deg:6.1f}°")

    def execute_targets(self, targets: Dict[str, float]) -> None:
        """
        MULTI-JOINT EXECUTION
        ---------------------
        Takes a dict of joint->target_deg and sends each one.
        Applies:
          - skip unknown joints
          - skip NaN
          - clamp to safe servo limits
        """
        for joint, deg in targets.items():
            if joint not in self.spec.channel or joint not in self.spec.limits:
                continue
            if not math.isfinite(deg):
                continue

            ch = self.spec.channel[joint]
            lim = self.spec.limits[joint]
            deg = _clamp(deg, lim.min_deg, lim.max_deg)

            if self.dry_run:
                print(f"[DRY] {joint:9s} -> {deg:6.1f}° (ch {ch})")
            else:
                self.send_servo(ch, deg)


# ---- Feature -> motor translation functions ----

def motor_action_from_left_knee_angle(spec: MotorSpec, left_knee_angle_deg: float) -> Dict[str, float]:
    """
    LEFT KNEE ANGLE -> HL_KNEE SERVO
    ===============================
    Goal:
      Convert measured knee angle (from pose) to a servo command.

    DIAGRAM (concept):
      pose_knee_angle:   70° (bent)  -------->  160° (straight)
      servo_target_deg:  min_deg     -------->  max_deg

    FOUNDATION:
      servo = map_range(pose_angle, 70..160, servo_min..servo_max)

    NOTE:
      The pose angle range [70..160] must be tuned from your dataset.
    """
    if not math.isfinite(left_knee_angle_deg):
        return {}
    lim = spec.limits["HL_KNEE"]
    target = _map_range(left_knee_angle_deg, 70, 160, lim.min_deg, lim.max_deg)
    return {"HL_KNEE": _clamp(target, lim.min_deg, lim.max_deg)}


def motor_action_from_right_knee_angle(spec: MotorSpec, right_knee_angle_deg: float) -> Dict[str, float]:
    """
    RIGHT KNEE ANGLE -> HR_KNEE SERVO
    Same mapping as left knee.
    """
    if not math.isfinite(right_knee_angle_deg):
        return {}
    lim = spec.limits["HR_KNEE"]
    target = _map_range(right_knee_angle_deg, 70, 160, lim.min_deg, lim.max_deg)
    return {"HR_KNEE": _clamp(target, lim.min_deg, lim.max_deg)}


def motor_action_from_torso_tilt_angle(spec: MotorSpec, torso_tilt_deg: float) -> Dict[str, float]:
    """
    TORSO TILT -> HIP COMPENSATION (BALANCE)
    =======================================
    PURPOSE:
      If torso leans, shift hip servos slightly to compensate.

    DIAGRAM (concept):
          lean forward (+phi)
             S ●
                \
                 \
                  ● H

      We convert tilt into a small delta for hip joints.

    FOUNDATION:
      delta = clamp(phi / 30, -1..1) * 8 degrees

      Front hips  = neutral + delta
      Hind hips   = neutral - delta

    This is a simple proportional controller:
      u = K * error
    where:
      error = torso_tilt
      K = 8/30 (degrees per degree) with clamp limits.
    """
    if not math.isfinite(torso_tilt_deg):
        return {}

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
    FRONT PAW WIDTH -> (OPTIONAL) STANCE CONTROL
    ===========================================
    Many robots do NOT have a sideways joint (abduction/adduction).
    If your robot has it, you can map width -> side stance motors.

    DIAGRAM:
      wider stance => abduct joints outward
      narrow stance => bring inward

    CURRENT DEFAULT:
      No-op for safety.
    """
    _ = spec
    _ = front_width
    return {}


def motor_action_from_hind_paw_width(spec: MotorSpec, hind_width: float) -> Dict[str, float]:
    """
    HIND PAW WIDTH -> (OPTIONAL) STANCE CONTROL
    No-op by default.
    """
    _ = spec
    _ = hind_width
    return {}


def motor_action_from_lift_score(spec: MotorSpec, lift_score: float) -> Dict[str, float]:
    """
    LIFT SCORE -> SMALL KNEE ADJUSTMENT (CROUCH/EXTEND)
    ==================================================
    PURPOSE:
      Convert lift_score into a mild knee bend/extend suggestion.

    DIAGRAM:
        hip higher => extend knees slightly
        hip lower  => bend knees slightly

    FOUNDATION:
      t = map_range(lift_score, -60..20, -1..1)
      adj = t * 5 degrees
      knee_target = neutral + adj

    This is a tiny proportional mapping.
    Tune [-60..20] window using your dataset stats.
    """
    if not math.isfinite(lift_score):
        return {}

    hl = spec.limits["HL_KNEE"]
    hr = spec.limits["HR_KNEE"]

    t = _map_range(lift_score, -60, 20, -1.0, 1.0)
    t = _clamp(t, -1.0, 1.0)
    adj = t * 5.0

    return {
        "HL_KNEE": _clamp(hl.neutral_deg + adj, hl.min_deg, hl.max_deg),
        "HR_KNEE": _clamp(hr.neutral_deg + adj, hr.min_deg, hr.max_deg),
    }


def merge_motor_targets(*cmds: Dict[str, float]) -> Dict[str, float]:
    """
    MERGE MULTIPLE MOTOR COMMAND DICTS
    ==================================
    If multiple features command the same joint, later values override earlier ones.

    Example:
      knee angle sets HL_KNEE
      lift_score also sets HL_KNEE
    If lift_score comes later, it overwrites.

    You can change policy to:
      average, weighted sum, saturating sum, etc.
    """
    out: Dict[str, float] = {}
    for c in cmds:
        out.update(c)
    return out


# =====================================================================
# CSV IO + MAIN LOOP
# =====================================================================
REQUIRED_COLS = [
    "motion_type", "seq", "split",
    "FL_PW_x", "FL_PW_y", "FL_KN_x", "FL_KN_y", "FL_EL_x", "FL_EL_y",
    "FR_PW_x", "FR_PW_y", "FR_KN_x", "FR_KN_y", "FR_EL_x", "FR_EL_y",
    "RL_PW_x", "RL_PW_y", "RL_KN_x", "RL_KN_y", "RL_EL_x", "RL_EL_y",
    "RR_PW_x", "RR_PW_y", "RR_KN_x", "RR_KN_y", "RR_EL_x", "RR_EL_y",
]


def read_rows(path: Path) -> list[dict]:
    """
    CSV READER (DICT)
    =================
    Reads full CSV into list[dict] for simple sequential playback.

    Validation:
      - Confirms expected columns exist
      - Raises error if missing (so you catch format issues immediately)
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path.resolve()}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for c in REQUIRED_COLS:
            if c not in (rdr.fieldnames or []):
                raise ValueError(f"Missing column '{c}'. Found: {rdr.fieldnames}")
        return list(rdr)


def pt(row: dict, xk: str, yk: str) -> Point2D:
    """
    PARSE POINT FROM ROW
    ====================
    Converts row values to floats and returns Point2D.

    If the CSV contains blanks, this will throw ValueError — which is good
    because blanks mean your preprocessing failed and you want to fix it.
    """
    return Point2D(float(row[xk]), float(row[yk]))


def main() -> None:
    rows = read_rows(CSV_PATH)
    spec = MotorSpec()
    driver = MotorDriver(spec, dry_run=DRY_RUN)

    print(f"Loaded {len(rows)} rows from: {CSV_PATH.resolve()}")
    print(f"DRY_RUN={DRY_RUN}, SECONDS_PER_ROW={SECONDS_PER_ROW}\n")

    for i, row in enumerate(rows, start=1):
        motion_type = row["motion_type"]
        seq = row["seq"]
        split = row["split"]

        # ---- read keypoints (new format) ----
        FL_PW = pt(row, "FL_PW_x", "FL_PW_y")
        FL_KN = pt(row, "FL_KN_x", "FL_KN_y")
        FL_EL = pt(row, "FL_EL_x", "FL_EL_y")

        FR_PW = pt(row, "FR_PW_x", "FR_PW_y")
        FR_KN = pt(row, "FR_KN_x", "FR_KN_y")
        FR_EL = pt(row, "FR_EL_x", "FR_EL_y")

        RL_PW = pt(row, "RL_PW_x", "RL_PW_y")
        RL_KN = pt(row, "RL_KN_x", "RL_KN_y")
        RL_EL = pt(row, "RL_EL_x", "RL_EL_y")

        RR_PW = pt(row, "RR_PW_x", "RR_PW_y")
        RR_KN = pt(row, "RR_KN_x", "RR_KN_y")
        RR_EL = pt(row, "RR_EL_x", "RR_EL_y")

        # =========================================================
        # (A) EXECUTE CALCULATION FUNCTIONS (using new mapping)
        # =========================================================
        # Rear EL points act like "hips" in our simplified model.
        left_knee_angle = calculate_left_knee_angle(RL_EL, RL_KN, RL_PW)
        right_knee_angle = calculate_right_knee_angle(RR_EL, RR_KN, RR_PW)

        # Hip center uses rear "EL"
        hip_center = calculate_hip_center(RL_EL, RR_EL)

        # Shoulder center uses front "EL"
        shoulder_center = calculate_shoulder_center(FL_EL, FR_EL)

        torso_tilt = calculate_torso_tilt_angle(shoulder_center, hip_center)

        front_width = calculate_front_paw_width(FL_PW, FR_PW)
        hind_width = calculate_hind_paw_width(RL_PW, RR_PW)

        lift_score = calculate_lift_score(hip_center, FL_PW, FR_PW, RL_PW, RR_PW)

        if PRINT_FEATURES:
            print("--------------------------------------------------")
            print(f"Row {i}/{len(rows)} motion_type='{motion_type}' seq='{seq}' split='{split}'")
            print(f"  left_knee_angle_deg   = {left_knee_angle:7.2f}")
            print(f"  right_knee_angle_deg  = {right_knee_angle:7.2f}")
            print(f"  hip_center            = ({hip_center.x:.2f},{hip_center.y:.2f})")
            print(f"  shoulder_center       = ({shoulder_center.x:.2f},{shoulder_center.y:.2f})")
            print(f"  torso_tilt_deg        = {torso_tilt:7.2f}")
            print(f"  front_paw_width       = {front_width:7.2f}")
            print(f"  hind_paw_width        = {hind_width:7.2f}")
            print(f"  lift_score            = {lift_score:7.2f}")

        # =========================================================
        # (B) TRANSLATE FEATURES -> MOTOR TARGETS
        # =========================================================
        motor_targets = merge_motor_targets(
            motor_action_from_left_knee_angle(spec, left_knee_angle),
            motor_action_from_right_knee_angle(spec, right_knee_angle),
            motor_action_from_torso_tilt_angle(spec, torso_tilt),
            motor_action_from_front_paw_width(spec, front_width),
            motor_action_from_hind_paw_width(spec, hind_width),
            motor_action_from_lift_score(spec, lift_score),
        )

        # NOTE: New format does not include "nose", so HEAD_TILT omitted.

        # =========================================================
        # (C) EXECUTE MOTOR TARGETS
        # =========================================================
        print("Motor targets:")
        driver.execute_targets(motor_targets)

        time.sleep(SECONDS_PER_ROW)

    print("\n✅ Done. Processed all CSV rows.")


if __name__ == "__main__":
    main()