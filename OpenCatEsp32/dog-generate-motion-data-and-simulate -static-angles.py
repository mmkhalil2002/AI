# ============================================================
# PYBULLET DOG MOTION DESIGNER + ESP32 C++ GENERATOR
# ============================================================
#
# PURPOSE:
# ------------------------------------------------------------
# This script lets you:
#
#   1. Select a dog motion from a menu
#   2. Simulate the selected motion in PyBullet
#   3. Store the selected motion table
#   4. Repeat for multiple motions
#   5. Press q to generate an ESP32/OpenCat C++ program
#
# The generated C++ file contains ONLY the motions you selected.
#
# OUTPUT FILES:
# ------------------------------------------------------------
#   ESP32/OpenCat C++ file:
#
#        generated_esp32_dog_motion_player.cpp
#
# RUN:
# ------------------------------------------------------------
#   python dog_motion_designer_generate_cpp.py
#
# MOTIONS:
# ------------------------------------------------------------
#   1 - Walk
#   2 - Trot / Run
#   3 - Sit
#   4 - Stand
#   5 - Jump
#   6 - Turn Left
#   7 - Turn Right
#   8 - Walk Backward
#   9 - Crawl
#   a - Bow
#   b - Stretch
#   c - Side Step Left
#   d - Side Step Right
#   e - Wave Left Paw
#   f - Wave Right Paw
#   g - Shake / Wiggle
#   h - Dance
#   i - Push-Up
#   j - Backflip Style
#   k - Frontflip Style
#   l - Roll Left Style
#   m - Roll Right Style
#   n - Sleep / Rest
#   o - Wake Up
#   q - Generate C++ file and exit
#
# IMPORTANT:
# ------------------------------------------------------------
# This script generates ESP32-friendly servo angles.
#
# Angle convention:
#
#   90 degrees  = neutral
#   120 degrees = forward / lift
#   60 degrees  = backward / push
#
# Legs:
#
#   FL = front left
#   FR = front right
#   RL = rear left
#   RR = rear right
#
# Table columns:
#
#   FL_hip, FL_knee,
#   FR_hip, FR_knee,
#   RL_hip, RL_knee,
#   RR_hip, RR_knee
#
# ============================================================

import csv
import math
import time
import os
from pathlib import Path

import pybullet as p
import pybullet_data


# ============================================================
# USER SETTINGS
# ============================================================

STEP_DELAY_SEC = 0.25
REPEAT_GAIT_IN_GUI = 2

CPP_OUTPUT_FILE = str(
    Path(__file__).resolve().parent /
    "generated_esp32_dog_motion_player.cpp"
)

NEUTRAL_HIP = 90
FORWARD_HIP = 120
BACKWARD_HIP = 60

SUPPORT_KNEE = 90
LIFT_KNEE = 120
CROUCH_KNEE = 135
JUMP_EXTEND_KNEE = 70

SIT_REAR_HIP = 60
SIT_REAR_KNEE = 140


# ============================================================
# SINGLE-KEY MENU INPUT
# ============================================================

def get_single_key(prompt="Select option: "):
    """
    Reads one key without requiring Enter on Windows.
    Falls back to normal input() on non-Windows systems.
    """

    print(prompt, end="", flush=True)

    if os.name == "nt":
        import msvcrt

        key = msvcrt.getch()

        try:
            choice = key.decode("utf-8")
        except UnicodeDecodeError:
            choice = ""

        print(choice)
        return choice.strip()

    return input().strip()


# ============================================================
# ROW HELPERS
# ============================================================

def make_row(
    rows,
    fl_hip, fl_knee,
    fr_hip, fr_knee,
    rl_hip, rl_knee,
    rr_hip, rr_knee
):
    rows.append([
        fl_hip, fl_knee,
        fr_hip, fr_knee,
        rl_hip, rl_knee,
        rr_hip, rr_knee
    ])


def neutral_row(rows):
    make_row(
        rows,
        NEUTRAL_HIP, SUPPORT_KNEE,
        NEUTRAL_HIP, SUPPORT_KNEE,
        NEUTRAL_HIP, SUPPORT_KNEE,
        NEUTRAL_HIP, SUPPORT_KNEE
    )


# ============================================================
# MOTION GENERATORS
# ============================================================

def generate_walk_table():
    rows = []
    neutral_row(rows)

    make_row(rows, 120, 120, 90, 90, 90, 90, 90, 90)
    make_row(rows, 120, 90, 90, 90, 90, 90, 90, 90)
    make_row(rows, 60, 90, 90, 90, 60, 90, 90, 90)

    make_row(rows, 90, 90, 120, 120, 90, 90, 90, 90)
    make_row(rows, 90, 90, 120, 90, 90, 90, 90, 90)
    make_row(rows, 90, 90, 60, 90, 90, 90, 60, 90)

    make_row(rows, 90, 90, 90, 90, 120, 120, 90, 90)
    make_row(rows, 90, 90, 90, 90, 120, 90, 90, 90)
    make_row(rows, 60, 90, 90, 90, 60, 90, 90, 90)

    make_row(rows, 90, 90, 90, 90, 90, 90, 120, 120)
    make_row(rows, 90, 90, 90, 90, 90, 90, 120, 90)
    make_row(rows, 90, 90, 60, 90, 90, 90, 60, 90)

    neutral_row(rows)
    return rows


def generate_trot_run_table():
    rows = []
    neutral_row(rows)

    make_row(rows, 120, 120, 60, 90, 60, 90, 120, 120)
    make_row(rows, 120, 90, 60, 90, 60, 90, 120, 90)

    make_row(rows, 60, 90, 120, 120, 120, 120, 60, 90)
    make_row(rows, 60, 90, 120, 90, 120, 90, 60, 90)

    neutral_row(rows)
    return rows


def generate_sit_table():
    rows = []
    neutral_row(rows)

    make_row(rows, 90, 90, 90, 90, 80, 115, 80, 115)
    make_row(rows, 90, 90, 90, 90, 70, 130, 70, 130)
    make_row(rows, 90, 90, 90, 90, SIT_REAR_HIP, SIT_REAR_KNEE, SIT_REAR_HIP, SIT_REAR_KNEE)

    return rows


def generate_stand_table():
    rows = []
    neutral_row(rows)
    return rows


def generate_jump_table():
    rows = []
    neutral_row(rows)

    make_row(rows, 90, 120, 90, 120, 90, 120, 90, 120)
    make_row(rows, 90, CROUCH_KNEE, 90, CROUCH_KNEE, 90, CROUCH_KNEE, 90, CROUCH_KNEE)
    make_row(rows, 90, 80, 90, 80, 90, 80, 90, 80)
    make_row(rows, 90, JUMP_EXTEND_KNEE, 90, JUMP_EXTEND_KNEE, 90, JUMP_EXTEND_KNEE, 90, JUMP_EXTEND_KNEE)
    make_row(rows, 90, 120, 90, 120, 90, 120, 90, 120)

    neutral_row(rows)
    return rows


def generate_turn_left_table():
    rows = []
    neutral_row(rows)

    make_row(rows, 60, 90, 120, 120, 60, 90, 120, 120)
    make_row(rows, 60, 90, 120, 90, 60, 90, 120, 90)
    make_row(rows, 120, 120, 60, 90, 120, 120, 60, 90)
    make_row(rows, 120, 90, 60, 90, 120, 90, 60, 90)

    neutral_row(rows)
    return rows


def generate_turn_right_table():
    rows = []
    neutral_row(rows)

    make_row(rows, 120, 120, 60, 90, 120, 120, 60, 90)
    make_row(rows, 120, 90, 60, 90, 120, 90, 60, 90)
    make_row(rows, 60, 90, 120, 120, 60, 90, 120, 120)
    make_row(rows, 60, 90, 120, 90, 60, 90, 120, 90)

    neutral_row(rows)
    return rows



def generate_backflip_table():
    rows=[]; neutral_row(rows)
    make_row(rows,90,125,90,125,90,135,90,135)
    make_row(rows,80,140,80,140,70,150,70,150)
    make_row(rows,110,80,110,80,130,60,130,60)
    make_row(rows,130,70,130,70,150,55,150,55)
    make_row(rows,70,140,70,140,70,140,70,140)
    make_row(rows,90,120,90,120,90,120,90,120)
    neutral_row(rows); return rows

def generate_frontflip_table():
    rows=[]; neutral_row(rows)
    make_row(rows,90,130,90,130,90,125,90,125)
    make_row(rows,110,140,110,140,100,140,100,140)
    make_row(rows,150,60,150,60,120,80,120,80)
    make_row(rows,140,70,140,70,70,140,70,140)
    make_row(rows,90,120,90,120,90,120,90,120)
    neutral_row(rows); return rows

def generate_bow_table():
    rows=[]; neutral_row(rows)
    make_row(rows,100,115,100,115,90,90,90,90)
    make_row(rows,110,130,110,130,85,90,85,90)
    make_row(rows,120,140,120,140,80,90,80,90)
    make_row(rows,110,120,110,120,90,90,90,90)
    neutral_row(rows); return rows

def generate_stretch_table():
    rows=[]; neutral_row(rows)
    make_row(rows,115,100,115,100,70,110,70,110)
    make_row(rows,130,95,130,95,60,120,60,120)
    make_row(rows,120,110,120,110,70,110,70,110)
    neutral_row(rows); return rows

def generate_crawl_table():
    rows=[]
    make_row(rows,90,115,90,115,90,115,90,115)
    make_row(rows,115,125,90,115,90,115,90,115)
    make_row(rows,115,115,90,115,90,115,90,115)
    make_row(rows,70,115,90,115,70,115,90,115)
    make_row(rows,90,115,115,125,90,115,90,115)
    make_row(rows,90,115,115,115,90,115,90,115)
    make_row(rows,90,115,70,115,90,115,70,115)
    neutral_row(rows); return rows

def generate_walk_backward_table():
    rows=[]; neutral_row(rows)
    make_row(rows,60,120,90,90,90,90,90,90)
    make_row(rows,60,90,90,90,90,90,90,90)
    make_row(rows,120,90,90,90,120,90,90,90)
    make_row(rows,90,90,60,120,90,90,90,90)
    make_row(rows,90,90,60,90,90,90,90,90)
    make_row(rows,90,90,120,90,90,90,120,90)
    neutral_row(rows); return rows

def generate_side_step_left_table():
    rows=[]; neutral_row(rows)
    make_row(rows,110,120,110,90,110,120,110,90)
    make_row(rows,100,90,80,90,100,90,80,90)
    make_row(rows,90,90,90,120,90,90,90,120)
    neutral_row(rows); return rows

def generate_side_step_right_table():
    rows=[]; neutral_row(rows)
    make_row(rows,110,90,110,120,110,90,110,120)
    make_row(rows,80,90,100,90,80,90,100,90)
    make_row(rows,90,120,90,90,90,120,90,90)
    neutral_row(rows); return rows

def generate_wave_left_table():
    rows=[]; neutral_row(rows)
    make_row(rows,120,140,90,90,90,90,90,90)
    make_row(rows,130,120,90,90,90,90,90,90)
    make_row(rows,100,140,90,90,90,90,90,90)
    make_row(rows,130,120,90,90,90,90,90,90)
    neutral_row(rows); return rows

def generate_wave_right_table():
    rows=[]; neutral_row(rows)
    make_row(rows,90,90,120,140,90,90,90,90)
    make_row(rows,90,90,130,120,90,90,90,90)
    make_row(rows,90,90,100,140,90,90,90,90)
    make_row(rows,90,90,130,120,90,90,90,90)
    neutral_row(rows); return rows

def generate_shake_table():
    rows=[]; neutral_row(rows)
    make_row(rows,110,90,70,90,110,90,70,90)
    make_row(rows,70,90,110,90,70,90,110,90)
    make_row(rows,110,90,70,90,110,90,70,90)
    make_row(rows,70,90,110,90,70,90,110,90)
    neutral_row(rows); return rows

def generate_dance_table():
    rows=[]; neutral_row(rows)
    make_row(rows,120,120,60,90,120,120,60,90)
    make_row(rows,60,90,120,120,60,90,120,120)
    make_row(rows,110,130,110,130,70,100,70,100)
    make_row(rows,70,100,70,100,110,130,110,130)
    neutral_row(rows); return rows

def generate_pushup_table():
    rows=[]; neutral_row(rows)
    make_row(rows,100,120,100,120,90,90,90,90)
    make_row(rows,110,140,110,140,90,90,90,90)
    make_row(rows,100,120,100,120,90,90,90,90)
    make_row(rows,110,140,110,140,90,90,90,90)
    neutral_row(rows); return rows

def generate_roll_left_table():
    rows=[]; neutral_row(rows)
    make_row(rows,120,140,120,80,120,140,120,80)
    make_row(rows,70,150,110,70,70,150,110,70)
    make_row(rows,60,120,130,90,60,120,130,90)
    neutral_row(rows); return rows

def generate_roll_right_table():
    rows=[]; neutral_row(rows)
    make_row(rows,120,80,120,140,120,80,120,140)
    make_row(rows,110,70,70,150,110,70,70,150)
    make_row(rows,130,90,60,120,130,90,60,120)
    neutral_row(rows); return rows

def generate_sleep_table():
    rows=[]; neutral_row(rows)
    make_row(rows,90,120,90,120,80,135,80,135)
    make_row(rows,90,140,90,140,70,150,70,150)
    return rows

def generate_wake_up_table():
    rows=[]
    make_row(rows,90,140,90,140,70,150,70,150)
    make_row(rows,90,120,90,120,80,135,80,135)
    neutral_row(rows); return rows

# ============================================================
# MOTION REGISTRY
# ============================================================

MOTIONS = {
    "1": {"name": "walk", "label": "Walk", "generator": generate_walk_table, "repeat_on_esp32": 3},
    "2": {"name": "trot_run", "label": "Trot / Run", "generator": generate_trot_run_table, "repeat_on_esp32": 3},
    "3": {"name": "sit", "label": "Sit", "generator": generate_sit_table, "repeat_on_esp32": 1},
    "4": {"name": "stand", "label": "Stand", "generator": generate_stand_table, "repeat_on_esp32": 1},
    "5": {"name": "jump", "label": "Jump", "generator": generate_jump_table, "repeat_on_esp32": 1},
    "6": {"name": "turn_left", "label": "Turn Left", "generator": generate_turn_left_table, "repeat_on_esp32": 3},
    "7": {"name": "turn_right", "label": "Turn Right", "generator": generate_turn_right_table, "repeat_on_esp32": 3},
    "8": {"name": "walk_backward", "label": "Walk Backward", "generator": generate_walk_backward_table, "repeat_on_esp32": 3},
    "9": {"name": "crawl", "label": "Crawl", "generator": generate_crawl_table, "repeat_on_esp32": 3},
    "a": {"name": "bow", "label": "Bow", "generator": generate_bow_table, "repeat_on_esp32": 1},
    "b": {"name": "stretch", "label": "Stretch", "generator": generate_stretch_table, "repeat_on_esp32": 1},
    "c": {"name": "side_left", "label": "Side Step Left", "generator": generate_side_step_left_table, "repeat_on_esp32": 2},
    "d": {"name": "side_right", "label": "Side Step Right", "generator": generate_side_step_right_table, "repeat_on_esp32": 2},
    "e": {"name": "wave_left", "label": "Wave Left Paw", "generator": generate_wave_left_table, "repeat_on_esp32": 2},
    "f": {"name": "wave_right", "label": "Wave Right Paw", "generator": generate_wave_right_table, "repeat_on_esp32": 2},
    "g": {"name": "shake", "label": "Shake / Wiggle", "generator": generate_shake_table, "repeat_on_esp32": 2},
    "h": {"name": "dance", "label": "Dance", "generator": generate_dance_table, "repeat_on_esp32": 2},
    "i": {"name": "pushup", "label": "Push-Up", "generator": generate_pushup_table, "repeat_on_esp32": 2},
    "j": {"name": "backflip", "label": "Backflip Style", "generator": generate_backflip_table, "repeat_on_esp32": 1},
    "k": {"name": "frontflip", "label": "Frontflip Style", "generator": generate_frontflip_table, "repeat_on_esp32": 1},
    "l": {"name": "roll_left", "label": "Roll Left Style", "generator": generate_roll_left_table, "repeat_on_esp32": 1},
    "m": {"name": "roll_right", "label": "Roll Right Style", "generator": generate_roll_right_table, "repeat_on_esp32": 1},
    "n": {"name": "sleep", "label": "Sleep / Rest", "generator": generate_sleep_table, "repeat_on_esp32": 1},
    "o": {"name": "wake_up", "label": "Wake Up", "generator": generate_wake_up_table, "repeat_on_esp32": 1},
}

# ============================================================
# PYBULLET VISUALIZATION
# ============================================================

DOG_BODY_ID = None
DOG_HEAD_ID = None
DOG_NOSE_ID = None


def setup_pybullet_scene():
    """
    Creates a fresh PyBullet scene every time a motion is selected.

    This version makes different motions visually clear:
      - walk/trot: leg movement
      - sit: rear body lowers
      - jump: body crouches then rises
      - turn left/right: body rotates left/right
    """

    global DOG_BODY_ID
    global DOG_HEAD_ID
    global DOG_NOSE_ID

    if not p.isConnected():
        p.connect(p.GUI)

    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    p.loadURDF("plane.urdf")

    # Body
    body_shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.38, 0.16, 0.08]
    )

    body_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.38, 0.16, 0.08],
        rgbaColor=[0.25, 0.25, 0.25, 1]
    )

    DOG_BODY_ID = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=body_shape,
        baseVisualShapeIndex=body_visual,
        basePosition=[0, 0, 0.48]
    )

    # Head
    head_shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.11, 0.10, 0.09]
    )

    head_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.11, 0.10, 0.09],
        rgbaColor=[0.15, 0.15, 0.15, 1]
    )

    DOG_HEAD_ID = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=head_shape,
        baseVisualShapeIndex=head_visual,
        basePosition=[0.50, 0, 0.52]
    )

    # Nose
    nose_visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.035,
        rgbaColor=[0.02, 0.02, 0.02, 1]
    )

    DOG_NOSE_ID = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=nose_visual,
        basePosition=[0.63, 0, 0.52]
    )

    p.addUserDebugText("FRONT", [0.70, 0, 0.75], textColorRGB=[1, 1, 0], textSize=1.4)
    p.addUserDebugText("FL", [0.28, 0.25, 0.45], textColorRGB=[1, 1, 1], textSize=1.2)
    p.addUserDebugText("FR", [0.28, -0.25, 0.45], textColorRGB=[1, 1, 1], textSize=1.2)
    p.addUserDebugText("RL", [-0.28, 0.25, 0.45], textColorRGB=[1, 1, 1], textSize=1.2)
    p.addUserDebugText("RR", [-0.28, -0.25, 0.45], textColorRGB=[1, 1, 1], textSize=1.2)

    p.resetDebugVisualizerCamera(
        cameraDistance=1.7,
        cameraYaw=35,
        cameraPitch=-25,
        cameraTargetPosition=[0.15, 0, 0.35]
    )


def set_dog_pose(motion_label, frame_index, total_frames):
    """
    Moves the body/head/nose so non-walking motions are visible.
    """

    if DOG_BODY_ID is None:
        return

    label = motion_label.lower()
    phase = frame_index / max(total_frames - 1, 1)

    x = 0
    y = 0
    z = 0.48
    yaw = 0

    # Sit: rear-looking body lowers and tilts slightly.
    if "sit" in label:
        z = 0.48 - 0.12 * phase
        yaw = 0

    # Jump: crouch then jump upward.
    elif "jump" in label:
        if phase < 0.35:
            z = 0.42
        elif phase < 0.65:
            z = 0.70
        else:
            z = 0.50

    # Turn: rotate body so it is obvious.
    elif "turn left" in label:
        yaw = phase * 0.9

    elif "turn right" in label:
        yaw = -phase * 0.9

    # Trot/run: faster body bounce.
    elif "trot" in label or "run" in label:
        z = 0.48 + 0.05 * math.sin(phase * math.pi * 4)

    # Crawl: lower body.
    elif "crawl" in label:
        z = 0.35 + 0.015 * math.sin(phase * math.pi * 4)

    # Bow / push-up: body moves lower and higher.
    elif "bow" in label or "push" in label:
        z = 0.43 + 0.03 * math.sin(phase * math.pi * 2)

    # Stretch / sleep: low posture.
    elif "stretch" in label or "sleep" in label:
        z = 0.38

    # Wake-up: body rises.
    elif "wake" in label:
        z = 0.36 + 0.12 * phase

    # Flip and roll styles: obvious simulator preview.
    elif "backflip" in label or "frontflip" in label:
        z = 0.48 + 0.22 * math.sin(phase * math.pi)

    elif "roll left" in label:
        yaw = phase * 1.2

    elif "roll right" in label:
        yaw = -phase * 1.2

    elif "side" in label:
        y = 0.10 * math.sin(phase * math.pi * 2)

    elif "wave" in label:
        z = 0.48 + 0.02 * math.sin(phase * math.pi * 2)

    elif "shake" in label or "dance" in label:
        yaw = 0.35 * math.sin(phase * math.pi * 6)
        z = 0.48 + 0.03 * math.sin(phase * math.pi * 6)

    # Walk: small body bounce.
    elif "walk" in label:
        z = 0.48 + 0.025 * math.sin(phase * math.pi * 4)

    quat = p.getQuaternionFromEuler([0, 0, yaw])

    p.resetBasePositionAndOrientation(DOG_BODY_ID, [x, y, z], quat)

    # Head and nose follow front direction.
    head_x = x + math.cos(yaw) * 0.50
    head_y = y + math.sin(yaw) * 0.50

    nose_x = x + math.cos(yaw) * 0.63
    nose_y = y + math.sin(yaw) * 0.63

    p.resetBasePositionAndOrientation(DOG_HEAD_ID, [head_x, head_y, z + 0.04], quat)
    p.resetBasePositionAndOrientation(DOG_NOSE_ID, [nose_x, nose_y, z + 0.04], quat)


def draw_leg(base_x, base_y, hip_angle_deg, knee_angle_deg):
    """
    Draws a two-link dog leg using temporary PyBullet debug lines.
    """

    upper_len = 0.22
    lower_len = 0.22

    hip_rad = math.radians(hip_angle_deg - 90)
    knee_rad = math.radians(knee_angle_deg - 90)

    hip_point = [base_x, base_y, 0.42]

    knee_point = [
        base_x + upper_len * math.sin(hip_rad),
        base_y,
        0.42 - upper_len * math.cos(hip_rad),
    ]

    foot_point = [
        knee_point[0] + lower_len * math.sin(hip_rad + knee_rad),
        base_y,
        knee_point[2] - lower_len * math.cos(hip_rad + knee_rad),
    ]

    p.addUserDebugLine(hip_point, knee_point, [1, 0, 0], lineWidth=5, lifeTime=STEP_DELAY_SEC)
    p.addUserDebugLine(knee_point, foot_point, [0, 0, 1], lineWidth=5, lifeTime=STEP_DELAY_SEC)

    foot_size = 0.025
    p.addUserDebugLine(
        [foot_point[0] - foot_size, foot_point[1], foot_point[2]],
        [foot_point[0] + foot_size, foot_point[1], foot_point[2]],
        [0, 1, 0],
        lineWidth=3,
        lifeTime=STEP_DELAY_SEC
    )
    p.addUserDebugLine(
        [foot_point[0], foot_point[1] - foot_size, foot_point[2]],
        [foot_point[0], foot_point[1] + foot_size, foot_point[2]],
        [0, 1, 0],
        lineWidth=3,
        lifeTime=STEP_DELAY_SEC
    )


def simulate_motion(rows, motion_label):
    """
    Shows the selected motion in PyBullet GUI.

    The scene resets for each motion and body animation is added
    so sit/jump/turn/run are visually different from walk.
    """

    setup_pybullet_scene()

    leg_bases = {
        "FL": [0.27, 0.17],
        "FR": [0.27, -0.17],
        "RL": [-0.27, 0.17],
        "RR": [-0.27, -0.17],
    }

    print("\nSimulating motion:", motion_label)
    print("View note: dog head/nose is in the FRONT direction.")

    total_frames = len(rows) * REPEAT_GAIT_IN_GUI
    frame_index = 0

    # Trot/run should be visibly faster.
    local_delay = STEP_DELAY_SEC
    if "trot" in motion_label.lower() or "run" in motion_label.lower():
        local_delay = STEP_DELAY_SEC * 0.55

    for repeat in range(REPEAT_GAIT_IN_GUI):
        for row in rows:
            set_dog_pose(motion_label, frame_index, total_frames)

            draw_leg(leg_bases["FL"][0], leg_bases["FL"][1], row[0], row[1])
            draw_leg(leg_bases["FR"][0], leg_bases["FR"][1], row[2], row[3])
            draw_leg(leg_bases["RL"][0], leg_bases["RL"][1], row[4], row[5])
            draw_leg(leg_bases["RR"][0], leg_bases["RR"][1], row[6], row[7])

            for _ in range(10):
                p.stepSimulation()
                time.sleep(local_delay / 10)

            frame_index += 1

    print("Simulation preview finished.")

# ============================================================
# C++ CODE GENERATION
# ============================================================

def cpp_array_name(motion_name):
    return "MOTION_" + motion_name.upper()


def generate_cpp_file(selected_motions):
    """
    Generates ESP32/OpenCat C++ code with only selected motions.
    """

    if not selected_motions:
        print("\nNo motions selected. C++ file was not generated.")
        return

    lines = []

    lines.append("// ============================================================")
    lines.append("// AUTO-GENERATED ESP32 / OPENCAT DOG MOTION TABLE PLAYER")
    lines.append("// ============================================================")
    lines.append("//")
    lines.append("// Generated by dog_motion_designer_generate_cpp.py")
    lines.append("//")
    lines.append("// This file contains ONLY the motions selected in the simulator.")
    lines.append("//")
    lines.append("// Serial commands:")
    lines.append("//   Send the listed number to play the corresponding motion.")
    lines.append("//")
    lines.append("// IMPORTANT:")
    lines.append("//   Verify servo indexes for your Bittle/OpenCat robot.")
    lines.append("//")
    lines.append("// ============================================================")
    lines.append("")
    lines.append("#include <WiFi.h>")
    lines.append("")
    lines.append("// ============================================================")
    lines.append("// WIFI COMMAND SETTINGS")
    lines.append("// ============================================================")
    lines.append("// Replace these with your Wi-Fi router credentials.")
    lines.append("const char* WIFI_SSID     = \"YOUR_WIFI_NAME\";")
    lines.append("const char* WIFI_PASSWORD = \"YOUR_WIFI_PASSWORD\";")
    lines.append("")
    lines.append("// TCP command port.")
    lines.append("// Example PC command:")
    lines.append("//   echo walk | nc <ESP32_IP_ADDRESS> 8888")
    lines.append("WiFiServer wifiServer(8888);")
    lines.append("")
    lines.append("#define FL_HIP   0")
    lines.append("#define FL_KNEE  1")
    lines.append("#define FR_HIP   2")
    lines.append("#define FR_KNEE  3")
    lines.append("#define RL_HIP   4")
    lines.append("#define RL_KNEE  5")
    lines.append("#define RR_HIP   6")
    lines.append("#define RR_KNEE  7")
    lines.append("")
    lines.append("#define STEP_DELAY_MS 250")
    lines.append("")
    lines.append("struct MotionTable {")
    lines.append("  const char* name;")
    lines.append("  const int (*rows)[8];")
    lines.append("  int rowCount;")
    lines.append("  int repeatCount;")
    lines.append("};")
    lines.append("")
    lines.append("void writeServoAngle(int servoIndex, int angle) {")
    lines.append("  // ----------------------------------------------------------")
    lines.append("  // Replace this body if your OpenCat firmware uses a different")
    lines.append("  // servo-write function.")
    lines.append("  // ----------------------------------------------------------")
    lines.append("#ifdef ESP_PWM")
    lines.append("  servo[servoIndex].write(angle);")
    lines.append("#else")
    lines.append("  pwm.writeAngle(servoIndex, angle);")
    lines.append("#endif")
    lines.append("}")
    lines.append("")
    lines.append("void applyMotionRow(const int row[8]) {")
    lines.append("  writeServoAngle(FL_HIP,  row[0]);")
    lines.append("  writeServoAngle(FL_KNEE, row[1]);")
    lines.append("  writeServoAngle(FR_HIP,  row[2]);")
    lines.append("  writeServoAngle(FR_KNEE, row[3]);")
    lines.append("  writeServoAngle(RL_HIP,  row[4]);")
    lines.append("  writeServoAngle(RL_KNEE, row[5]);")
    lines.append("  writeServoAngle(RR_HIP,  row[6]);")
    lines.append("  writeServoAngle(RR_KNEE, row[7]);")
    lines.append("}")
    lines.append("")

    for motion_name, motion_info in selected_motions.items():
        array_name = cpp_array_name(motion_name)
        rows = motion_info["rows"]

        lines.append("// ============================================================")
        lines.append(f"// MOTION - {motion_info['label']}")
        lines.append("// ============================================================")
        lines.append(f"const int {array_name}[][8] = {{")
        lines.append("  // -------------------------------------------------------------------------")
        lines.append("// FL_HIP  FL_KNEE  FR_HIP  FR_KNEE  RL_HIP  RL_KNEE  RR_HIP  RR_KNEE")
        lines.append("  // -------------------------------------------------------------------------")

        for row in rows:

            formatted_row = (
                f"  {{ "
                f"{row[0]:<9}"
                f"{row[1]:<10}"
                f"{row[2]:<9}"
                f"{row[3]:<10}"
                f"{row[4]:<9}"
                f"{row[5]:<10}"
                f"{row[6]:<9}"
                f"{row[7]:<9}"
                f"}},"
            )

            lines.append(formatted_row)

        lines.append("};")
        lines.append("")

    lines.append("// ============================================================")
    lines.append("// SELECTED MOTION REGISTRY")
    lines.append("// ============================================================")
    lines.append("MotionTable motions[] = {")

    for motion_name, motion_info in selected_motions.items():
        array_name = cpp_array_name(motion_name)
        repeat_count = motion_info["repeat_on_esp32"]
        lines.append(
            f'  {{"{motion_name}", {array_name}, sizeof({array_name}) / sizeof({array_name}[0]), {repeat_count}}},'
        )

    lines.append("};")
    lines.append("")
    lines.append("const int MOTION_COUNT = sizeof(motions) / sizeof(motions[0]);")
    lines.append("")
    lines.append("void playMotion(int motionIndex) {")
    lines.append("  if (motionIndex < 0 || motionIndex >= MOTION_COUNT) {")
    lines.append('    Serial.println("ERROR: invalid motion index");')
    lines.append("    return;")
    lines.append("  }")
    lines.append("")
    lines.append("  MotionTable motion = motions[motionIndex];")
    lines.append('  Serial.print("Playing motion: ");')
    lines.append("  Serial.println(motion.name);")
    lines.append("")
    lines.append("  for (int r = 0; r < motion.repeatCount; r++) {")
    lines.append("    for (int i = 0; i < motion.rowCount; i++) {")
    lines.append("      applyMotionRow(motion.rows[i]);")
    lines.append("      delay(STEP_DELAY_MS);")
    lines.append("    }")
    lines.append("  }")
    lines.append("}")
    lines.append("")
    lines.append("int findMotionByName(String command) {")
    lines.append("  command.trim();")
    lines.append("  command.toLowerCase();")
    lines.append("")
    lines.append("  for (int i = 0; i < MOTION_COUNT; i++) {")
    lines.append("    String motionName = String(motions[i].name);")
    lines.append("    motionName.toLowerCase();")
    lines.append("")
    lines.append("    if (command == motionName) {")
    lines.append("      return i;")
    lines.append("    }")
    lines.append("  }")
    lines.append("")
    lines.append("  return -1;")
    lines.append("}")
    lines.append("")
    lines.append("void printHelp() {")
    lines.append('  Serial.println();')
    lines.append('  Serial.println("======================================");')
    lines.append('  Serial.println(" GENERATED DOG MOTION PLAYER");')
    lines.append('  Serial.println("======================================");')
    lines.append('  Serial.println("Send one of these text commands:");')

    for motion_name, motion_info in selected_motions.items():
        lines.append(f'  Serial.println("  {motion_name}");')

    lines.append('  Serial.println("");')
    lines.append('  Serial.println("Other commands:");')
    lines.append('  Serial.println("  help");')
    lines.append('  Serial.println("======================================");')
    lines.append("}")
    lines.append("")
    lines.append("void handleCommand(String command) {")
    lines.append("  command.trim();")
    lines.append("  command.toLowerCase();")
    lines.append("")
    lines.append("  if (command.length() == 0) {")
    lines.append("    return;")
    lines.append("  }")
    lines.append("")
    lines.append('  if (command == "help" || command == "h") {')
    lines.append("    printHelp();")
    lines.append("    return;")
    lines.append("  }")
    lines.append("")
    lines.append("  int motionIndex = findMotionByName(command);")
    lines.append("")
    lines.append("  if (motionIndex >= 0) {")
    lines.append("    playMotion(motionIndex);")
    lines.append('    Serial.println("OK");')
    lines.append("  }")
    lines.append("  else {")
    lines.append('    Serial.print("ERROR: unknown command: ");')
    lines.append("    Serial.println(command);")
    lines.append("    printHelp();")
    lines.append("  }")
    lines.append("}")
    lines.append("")
    lines.append("void startWiFiCommandServer() {")
    lines.append('  Serial.println("Starting Wi-Fi command server...");')
    lines.append("")
    lines.append("  WiFi.mode(WIFI_STA);")
    lines.append("  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);")
    lines.append("")
    lines.append('  Serial.print("Connecting to Wi-Fi");')
    lines.append("")
    lines.append("  int retryCount = 0;")
    lines.append("  while (WiFi.status() != WL_CONNECTED && retryCount < 40) {")
    lines.append("    delay(500);")
    lines.append('    Serial.print(".");')
    lines.append("    retryCount++;")
    lines.append("  }")
    lines.append("")
    lines.append("  Serial.println();")
    lines.append("")
    lines.append("  if (WiFi.status() == WL_CONNECTED) {")
    lines.append('    Serial.print("ESP32 IP Address: ");')
    lines.append("    Serial.println(WiFi.localIP());")
    lines.append("    wifiServer.begin();")
    lines.append('    Serial.println("Wi-Fi TCP command server started on port 8888");')
    lines.append("  }")
    lines.append("  else {")
    lines.append('    Serial.println("Wi-Fi connection failed. USB Serial commands still work.");')
    lines.append("  }")
    lines.append("}")
    lines.append("")
    lines.append("bool handleUsbSerialCommand() {")
    lines.append("  if (!Serial.available()) {")
    lines.append("    return false;")
    lines.append("  }")
    lines.append("")
    lines.append("  String command = Serial.readStringUntil('\\n');")
    lines.append('  Serial.print("USB command received: ");')
    lines.append("  Serial.println(command);")
    lines.append("  handleCommand(command);")
    lines.append("  return true;")
    lines.append("}")
    lines.append("")
    lines.append("bool handleWiFiCommand() {")
    lines.append("  if (WiFi.status() != WL_CONNECTED) {")
    lines.append("    return false;")
    lines.append("  }")
    lines.append("")
    lines.append("  WiFiClient client = wifiServer.available();")
    lines.append("")
    lines.append("  if (!client) {")
    lines.append("    return false;")
    lines.append("  }")
    lines.append("")
    lines.append("  String command = client.readStringUntil('\\n');")
    lines.append("  command.trim();")
    lines.append("")
    lines.append('  Serial.print("Wi-Fi command received: ");')
    lines.append("  Serial.println(command);")
    lines.append("")
    lines.append("  handleCommand(command);")
    lines.append("")
    lines.append('  client.println("OK: command received");')
    lines.append("  client.stop();")
    lines.append("")
    lines.append("  return true;")
    lines.append("}")
    lines.append("")
    lines.append("void setup() {")
    lines.append("  Serial.begin(115200);")
    lines.append("  delay(1000);")
    lines.append("")
    lines.append("  printHelp();")
    lines.append("  startWiFiCommandServer();")
    lines.append("")
    lines.append("  // Start in the first selected motion once.")
    lines.append("  playMotion(0);")
    lines.append("}")
    lines.append("")
    lines.append("void loop() {")
    lines.append("  // ----------------------------------------------------------")
    lines.append("  // Priority rule:")
    lines.append("  // 1. If USB Serial command is available, use USB command.")
    lines.append("  // 2. If no USB command is available, check Wi-Fi TCP command.")
    lines.append("  // ----------------------------------------------------------")
    lines.append("")
    lines.append("  if (handleUsbSerialCommand()) {")
    lines.append("    return;")
    lines.append("  }")
    lines.append("")
    lines.append("  handleWiFiCommand();")
    lines.append("}")
    lines.append("")

    Path(CPP_OUTPUT_FILE).write_text("\n".join(lines), encoding="utf-8")

    print("\n================================================")
    print(" C++ FILE GENERATED")
    print("================================================")
    print(Path(CPP_OUTPUT_FILE).resolve())
    print("Selected motions:", ", ".join(selected_motions.keys()))
    print("================================================")


# ============================================================
# PRINT MENU
# ============================================================

def print_menu(selected_motions):
    print("\n================================================")
    print(" DOG MOTION DESIGNER + C++ GENERATOR")
    print("================================================")
    print("Select a motion to simulate and add to output:")
    print("")
    print("1 - Walk")
    print("2 - Trot / Run")
    print("3 - Sit")
    print("4 - Stand")
    print("5 - Jump")
    print("6 - Turn Left")
    print("7 - Turn Right")
    print("8 - Walk Backward")
    print("9 - Crawl")
    print("a - Bow")
    print("b - Stretch")
    print("c - Side Step Left")
    print("d - Side Step Right")
    print("e - Wave Left Paw")
    print("f - Wave Right Paw")
    print("g - Shake / Wiggle")
    print("h - Dance")
    print("i - Push-Up")
    print("j - Backflip Style")
    print("k - Frontflip Style")
    print("l - Roll Left Style")
    print("m - Roll Right Style")
    print("n - Sleep / Rest")
    print("o - Wake Up")
    print("q - Generate C++ file and exit")
    print("================================================")

    if selected_motions:
        print("Selected motions so far:")
        for motion_name, info in selected_motions.items():
            print(f"  - {info['label']} ({motion_name})")
    else:
        print("Selected motions so far: none")

    print("================================================")


# ============================================================
# MAIN
# ============================================================

def main():
    selected_motions = {}

    while True:
        print_menu(selected_motions)

        choice = get_single_key("Select option: ").lower()

        if choice == "q":
            generate_cpp_file(selected_motions)
            print("\nExiting.")
            break

        if choice not in MOTIONS:
            print("\nInvalid option.")
            continue

        motion_config = MOTIONS[choice]
        motion_name = motion_config["name"]
        motion_label = motion_config["label"]

        rows = motion_config["generator"]()

        # Store or replace the selected motion.
        selected_motions[motion_name] = {
            "label": motion_label,
            "rows": rows,
            "repeat_on_esp32": motion_config["repeat_on_esp32"],
        }

        print(f"\nSelected motion: {motion_label}")
        print(f"Rows generated : {len(rows)}")

        # Individual CSV generation removed.

        print("\n================================================")
        print(" ANGLE TABLE")
        print("================================================")
        print("Angle meaning:")
        print("  90  = neutral")
        print("  120 = forward / lifted")
        print("  60  = backward / push")
        print("================================================")

        header = (
            f"{'STEP':<6}"
            f"{'FL_HIP':<10}"
            f"{'FL_KNEE':<10}"
            f"{'FR_HIP':<10}"
            f"{'FR_KNEE':<10}"
            f"{'RL_HIP':<10}"
            f"{'RL_KNEE':<10}"
            f"{'RR_HIP':<10}"
            f"{'RR_KNEE':<10}"
        )

        meaning = (
            f"{'':<6}"
            f"{'FrontLHip':<10}"
            f"{'FrontLKnee':<10}"
            f"{'FrontRHip':<10}"
            f"{'FrontRKnee':<10}"
            f"{'RearLHip':<10}"
            f"{'RearLKnee':<10}"
            f"{'RearRHip':<10}"
            f"{'RearRKnee':<10}"
        )

        print(header)
        print(meaning)
        print("-" * len(header))

        for i, row in enumerate(rows):

            print(
                f"{i:<6}"
                f"{row[0]:<10}"
                f"{row[1]:<10}"
                f"{row[2]:<10}"
                f"{row[3]:<10}"
                f"{row[4]:<10}"
                f"{row[5]:<10}"
                f"{row[6]:<10}"
                f"{row[7]:<10}"
            )

        simulate_motion(rows, motion_label)


if __name__ == "__main__":
    main()
