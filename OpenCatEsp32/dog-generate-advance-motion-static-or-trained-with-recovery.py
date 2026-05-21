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
#   IMPORTANT UPDATED BEHAVIOR:
#   At startup, choose STATIC motions or TRAINED / OPTIMIZED motions.
#   Static mode exports fixed angle tables without training.
#   Trained mode optimizes the selected motion before export.
#
#   1 - Train Walk
#   2 - Train Trot / Run
#   3 - Train Sit
#   4 - Train Stand
#   5 - Train Jump
#   6 - Train Turn Left
#   7 - Train Turn Right
#   8 - Train Walk Backward
#   9 - Train Crawl
#   a - Train Bow
#   b - Train Stretch
#   c - Train Side Step Left
#   d - Train Side Step Right
#   e - Train Wave Left Paw
#   f - Train Wave Right Paw
#   g - Train Shake / Wiggle
#   h - Train Dance
#   i - Train Push-Up
#   j - Train Backflip Style
#   k - Train Frontflip Style
#   l - Train Roll Left Style
#   m - Train Roll Right Style
#   n - Train Sleep / Rest
#   o - Train Wake Up
#   r - Simulate Fall Scenario + Optional Recovery
#   q - Generate C++ file and exit
#
# ESP32 MANUAL RECOVERY COMMANDS GENERATED AUTOMATICALLY:
# ------------------------------------------------------------
#   When you press q, the generated ESP32 C++ file will include
#   recovery commands even if you did not select recovery from
#   the Python menu. This lets you manually push the dog to one
#   side, then send a command such as recover_left over USB Serial
#   or Wi-Fi TCP so the ESP32 executes the recovery angle table.
#
#   Manual ESP32 commands:
#
#       recover_left
#       recover_right
#       recover_front
#       recover_back
#       recover_upside_down
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
# FALL SCENARIOS + RECOVERY MOTIONS
# ============================================================
#
# PURPOSE:
# ------------------------------------------------------------
# This section adds explicit failure scenarios and recovery actions.
#
# The normal training motions try to avoid falling. However, real
# robots must also handle what happens AFTER a fall.
#
# The workflow is:
#
#   1. User selects option r from the menu
#   2. User chooses a failure scenario:
#        1 = left-side fall
#        2 = right-side fall
#        3 = front fall
#        4 = back fall
#        5 = upside-down fall
#   3. The simulator shows the fallen posture
#   4. The user is asked whether to run recovery
#   5. If yes, the matching recovery table is generated and simulated
#   6. The recovery table is stored for ESP32/C++ generation
#
# IMPORTANT:
# ------------------------------------------------------------
# This is still a simplified recovery model. The recovery tables are
# servo-angle sequences that approximate self-righting motions.
# Real robot recovery should later use IMU roll/pitch/yaw data and
# real servo feedback.
#
# Numerical example for fall detection:
#
#   roll  = 1.10 radians
#   pitch = 0.12 radians
#
#   abs(roll) = 1.10 > 0.8
#   abs(pitch)= 0.12 < 0.8
#
#   Result:
#       fallen = True
#       fall_type = left_side_fall
#
#   Recovery selected:
#       generate_recover_from_left_fall_table()
#
# ============================================================

FALL_ROLL_THRESHOLD = 0.8
FALL_PITCH_THRESHOLD = 0.8

# ============================================================
# DETAILED RECOVERY DESIGN NOTES
# ============================================================
#
# The recovery system has two layers:
#
#   LAYER 1 - SIMULATION / DESIGN TIME
#   ----------------------------------------------------------
#   In this Python tool, the user can intentionally select a fall
#   scenario from the menu using option "r". The simulator places
#   the robot body into that fallen orientation and then optionally
#   runs the matching recovery motion table.
#
#   LAYER 2 - ESP32 / REAL ROBOT PLAYBACK
#   ----------------------------------------------------------
#   When the user presses "q", the selected recovery rows are exported
#   as C++ angle tables. The ESP32 can then replay these recovery
#   motions exactly like walk, jump, crawl, or roll motions.
#
# IMPORTANT LIMITATION:
# ------------------------------------------------------------
#   The exported ESP32 code replays angle tables. It does NOT yet
#   automatically detect a real fall by itself unless an IMU sensor
#   is added later.
#
#   Future automatic recovery requires an IMU such as:
#
#       MPU6050
#       BNO055
#       ICM20948
#       BMI270
#
#   With an IMU, ESP32 could continuously measure:
#
#       roll
#       pitch
#       yaw
#
#   and automatically trigger the matching recovery table.
#
# ------------------------------------------------------------
# NUMERICAL FALL DETECTION EXAMPLE
# ------------------------------------------------------------
#
# Normal standing robot:
#
#       roll  = 0.05 radians  ≈  2.9 degrees
#       pitch = 0.03 radians  ≈  1.7 degrees
#
# Thresholds:
#
#       FALL_ROLL_THRESHOLD  = 0.8 radians ≈ 45.8 degrees
#       FALL_PITCH_THRESHOLD = 0.8 radians ≈ 45.8 degrees
#
# Check:
#
#       abs(roll)  = 0.05 < 0.8  -> OK
#       abs(pitch) = 0.03 < 0.8  -> OK
#
# Result:
#
#       not_fallen
#
# ------------------------------------------------------------
# LEFT-SIDE FALL EXAMPLE
# ------------------------------------------------------------
#
# Measured orientation:
#
#       roll  = +1.10 radians ≈ +63.0 degrees
#       pitch = +0.10 radians ≈  +5.7 degrees
#
# Check:
#
#       abs(roll)  = 1.10 > 0.8  -> FALL detected
#       abs(pitch) = 0.10 < 0.8
#
# Since roll is larger than pitch and roll is positive:
#
#       fall_type = "left_side_fall"
#
# Matching recovery:
#
#       generate_recover_from_left_fall_table()
#
# ------------------------------------------------------------
# RIGHT-SIDE FALL EXAMPLE
# ------------------------------------------------------------
#
# Measured orientation:
#
#       roll  = -1.15 radians ≈ -65.9 degrees
#       pitch = +0.06 radians ≈  +3.4 degrees
#
# Check:
#
#       abs(roll) = 1.15 > 0.8
#
# Since roll is negative:
#
#       fall_type = "right_side_fall"
#
# Matching recovery:
#
#       generate_recover_from_right_fall_table()
#
# ------------------------------------------------------------
# FRONT FALL EXAMPLE
# ------------------------------------------------------------
#
# Measured orientation:
#
#       roll  = +0.12 radians ≈  +6.9 degrees
#       pitch = +1.20 radians ≈ +68.8 degrees
#
# Check:
#
#       abs(pitch) = 1.20 > 0.8
#
# Since pitch is positive:
#
#       fall_type = "front_fall"
#
# Matching recovery:
#
#       generate_recover_from_front_fall_table()
#
# ------------------------------------------------------------
# BACK FALL EXAMPLE
# ------------------------------------------------------------
#
# Measured orientation:
#
#       roll  = +0.08 radians ≈  +4.6 degrees
#       pitch = -1.05 radians ≈ -60.2 degrees
#
# Check:
#
#       abs(pitch) = 1.05 > 0.8
#
# Since pitch is negative:
#
#       fall_type = "back_fall"
#
# Matching recovery:
#
#       generate_recover_from_back_fall_table()
#
# ------------------------------------------------------------
# UPSIDE-DOWN FALL EXAMPLE
# ------------------------------------------------------------
#
# The simplified menu can directly create an upside-down scenario.
# In a real robot, upside-down detection would usually require more
# robust IMU logic, for example:
#
#       abs(roll)  close to pi radians
#       or body Z-axis points downward
#       or gravity vector indicates inverted body
#
# Example:
#
#       roll = 3.14 radians ≈ 180 degrees
#
# Matching recovery:
#
#       generate_recover_from_upside_down_table()
#
# ============================================================


def classify_fall_from_roll_pitch(roll, pitch):
    """
    Classifies the fall direction from roll and pitch.

    Roll means side tilt.
    Pitch means front/back tilt.

    Numerical examples:

      roll =  1.10, pitch = 0.10  -> left_side_fall
      roll = -1.05, pitch = 0.05  -> right_side_fall
      roll =  0.10, pitch = 1.20  -> front_fall
      roll =  0.05, pitch = -1.10 -> back_fall

    If both roll and pitch are large, the larger absolute value wins.
    """

    if abs(roll) < FALL_ROLL_THRESHOLD and abs(pitch) < FALL_PITCH_THRESHOLD:
        return "not_fallen"

    if abs(roll) >= abs(pitch):
        if roll > 0:
            return "left_side_fall"
        return "right_side_fall"

    if pitch > 0:
        return "front_fall"

    return "back_fall"



# ============================================================
# ============================================================
# COMPLETE RECOVERY ANGLE TABLES + STEP-BY-STEP EXPLANATION
# ============================================================
#
# PURPOSE:
# ------------------------------------------------------------
# This section documents the exact recovery motion tables that
# are exported to the ESP32 C++ program.
#
# Each row has 8 values in this order:
#
#     FL_HIP, FL_KNEE,
#     FR_HIP, FR_KNEE,
#     RL_HIP, RL_KNEE,
#     RR_HIP, RR_KNEE
#
# Where:
#
#     FL = Front Left
#     FR = Front Right
#     RL = Rear Left
#     RR = Rear Right
#
# Angle convention used by this script:
#
#     90 degrees  = neutral / normal standing reference
#     140-150     = folded knee / compact leg
#     60-70       = extended push angle used to push floor
#     120-130     = aggressive hip push angle
#
# IMPORTANT:
# ------------------------------------------------------------
# These recovery motions are MANUAL recovery commands.
# That means the ESP32 will run them only when you send:
#
#     recover_left
#     recover_right
#     recover_front
#     recover_back
#     recover_upside_down
#
# The ESP32 will NOT automatically know the robot fell unless an
# IMU such as MPU6050, BNO055, BMI270, or ICM20948 is added later.
#
# Example manual test:
#
#     1. Start robot walking.
#     2. Push robot gently onto its left side.
#     3. Send command: recover_left
#     4. ESP32 executes the recover_left angle table.
#     5. Robot attempts to stand up again.
#
# ============================================================
# RECOVERY TABLE 1: recover_left
# ============================================================
#
# FALL POSITION:
# ------------------------------------------------------------
# Robot is lying on its LEFT side.
#
# Typical orientation example:
#
#     roll  = +65 degrees to +90 degrees
#     pitch = small, for example -10 degrees to +10 degrees
#
# Manual command:
#
#     recover_left
#
# FULL ANGLE TABLE:
# ------------------------------------------------------------
#
#     STEP    FL_HIP FL_KNEE  FR_HIP FR_KNEE  RL_HIP RL_KNEE  RR_HIP RR_KNEE
#     0       90     145      90     145      90     145      90     145
#     1       60     145      130    70       60     145      130    70
#     2       70     130      125    75       70     130      125    75
#     3       90     110      90     110      90     110      90     110
#     4       90     90       90     90       90     90       90     90
#
# STEP-BY-STEP MEANING:
# ------------------------------------------------------------
#
# STEP 0 - Protect and fold
#
#     FL_KNEE = 145
#     FR_KNEE = 145
#     RL_KNEE = 145
#     RR_KNEE = 145
#
#     All knees fold deeply.
#
#     Purpose:
#         Make the robot compact so the legs do not block rotation.
#         This is similar to pulling arms/legs inward before rolling.
#
# STEP 1 - Create roll-back torque to the right
#
#     Left-side legs:
#         FL_HIP = 60,  FL_KNEE = 145
#         RL_HIP = 60,  RL_KNEE = 145
#
#     Right-side legs:
#         FR_HIP = 130, FR_KNEE = 70
#         RR_HIP = 130, RR_KNEE = 70
#
#     Numerical example:
#
#         Hip difference  = 130 - 60 = 70 degrees
#         Knee difference = 145 - 70 = 75 degrees
#
#     Purpose:
#         Right side extends and pushes against the floor while the
#         left side stays folded. This creates a torque that rolls
#         the body back toward standing.
#
# STEP 2 - Intermediate push / reduce shock
#
#     Left-side legs move from 60/145 to 70/130.
#     Right-side legs move from 130/70 to 125/75.
#
#     Purpose:
#         Continue the recovery rotation but reduce the sudden snap
#         from step 1. This makes recovery smoother.
#
# STEP 3 - Soft landing posture
#
#     All hips  = 90
#     All knees = 110
#
#     Purpose:
#         Robot is nearly upright. Knees stay slightly folded to
#         absorb body weight and prevent bouncing.
#
# STEP 4 - Neutral stand
#
#     All hips  = 90
#     All knees = 90
#
#     Purpose:
#         Return to normal standing posture.
#
# ============================================================
# RECOVERY TABLE 2: recover_right
# ============================================================
#
# FALL POSITION:
# ------------------------------------------------------------
# Robot is lying on its RIGHT side.
#
# Typical orientation example:
#
#     roll  = -65 degrees to -90 degrees
#     pitch = small, for example -10 degrees to +10 degrees
#
# Manual command:
#
#     recover_right
#
# FULL ANGLE TABLE:
# ------------------------------------------------------------
#
#     STEP    FL_HIP FL_KNEE  FR_HIP FR_KNEE  RL_HIP RL_KNEE  RR_HIP RR_KNEE
#     0       90     145      90     145      90     145      90     145
#     1       130    70       60     145      130    70       60     145
#     2       125    75       70     130      125    75       70     130
#     3       90     110      90     110      90     110      90     110
#     4       90     90       90     90       90     90       90     90
#
# STEP-BY-STEP MEANING:
# ------------------------------------------------------------
#
# STEP 0 - Protect and fold
#
#     All hips  = 90
#     All knees = 145
#
#     Purpose:
#         Fold all legs and make the body compact before rolling.
#
# STEP 1 - Create roll-back torque to the left
#
#     Left-side legs:
#         FL_HIP = 130, FL_KNEE = 70
#         RL_HIP = 130, RL_KNEE = 70
#
#     Right-side legs:
#         FR_HIP = 60,  FR_KNEE = 145
#         RR_HIP = 60,  RR_KNEE = 145
#
#     Numerical example:
#
#         Hip difference  = 130 - 60 = 70 degrees
#         Knee difference = 145 - 70 = 75 degrees
#
#     Purpose:
#         Left side extends and pushes against the floor while the
#         right side stays folded. This mirrors recover_left.
#
# STEP 2 - Intermediate push / reduce shock
#
#     Left-side legs soften from 130/70 to 125/75.
#     Right-side legs soften from 60/145 to 70/130.
#
#     Purpose:
#         Continue rolling upright while smoothing the transition.
#
# STEP 3 - Soft landing posture
#
#     All hips  = 90
#     All knees = 110
#
#     Purpose:
#         Absorb landing before standing fully.
#
# STEP 4 - Neutral stand
#
#     All hips  = 90
#     All knees = 90
#
#     Purpose:
#         Finish recovery in normal standing posture.
#
# ============================================================
# RECOVERY TABLE 3: recover_front
# ============================================================
#
# FALL POSITION:
# ------------------------------------------------------------
# Robot has fallen FORWARD onto its chest/front side.
#
# Typical orientation example:
#
#     roll  = small, for example -10 degrees to +10 degrees
#     pitch = +60 degrees to +90 degrees
#
# Manual command:
#
#     recover_front
#
# FULL ANGLE TABLE:
# ------------------------------------------------------------
#
#     STEP    FL_HIP FL_KNEE  FR_HIP FR_KNEE  RL_HIP RL_KNEE  RR_HIP RR_KNEE
#     0       90     150      90     150      90     120      90     120
#     1       120    70       120    70       70     140      70     140
#     2       100    90       100    90       90     110      90     110
#     3       90     90       90     90       90     90       90     90
#
# STEP-BY-STEP MEANING:
# ------------------------------------------------------------
#
# STEP 0 - Front protection posture
#
#     Front knees:
#         FL_KNEE = 150
#         FR_KNEE = 150
#
#     Rear knees:
#         RL_KNEE = 120
#         RR_KNEE = 120
#
#     Purpose:
#         Front legs are compact because the chest/front is on the
#         ground. Rear legs prepare to help the body rotate upward.
#
# STEP 1 - Push front/body upward
#
#     Front legs:
#         FL_HIP = 120, FL_KNEE = 70
#         FR_HIP = 120, FR_KNEE = 70
#
#     Rear legs:
#         RL_HIP = 70,  RL_KNEE = 140
#         RR_HIP = 70,  RR_KNEE = 140
#
#     Numerical example:
#
#         Front knee extension = 150 - 70 = 80 degrees
#
#     Purpose:
#         Front legs extend strongly and push against the floor,
#         lifting the chest away from the ground.
#
# STEP 2 - Stabilize the body
#
#     Front legs move toward 100/90.
#     Rear legs move toward 90/110.
#
#     Purpose:
#         Prevent overshoot and bring all legs under the body.
#
# STEP 3 - Neutral stand
#
#     All hips  = 90
#     All knees = 90
#
#     Purpose:
#         Finish standing.
#
# ============================================================
# RECOVERY TABLE 4: recover_back
# ============================================================
#
# FALL POSITION:
# ------------------------------------------------------------
# Robot has fallen BACKWARD onto its back/rear side.
#
# Typical orientation example:
#
#     roll  = small, for example -10 degrees to +10 degrees
#     pitch = -60 degrees to -90 degrees
#
# Manual command:
#
#     recover_back
#
# FULL ANGLE TABLE:
# ------------------------------------------------------------
#
#     STEP    FL_HIP FL_KNEE  FR_HIP FR_KNEE  RL_HIP RL_KNEE  RR_HIP RR_KNEE
#     0       90     120      90     120      90     150      90     150
#     1       70     140      70     140      120    70       120    70
#     2       90     110      90     110      100    90       100    90
#     3       90     90       90     90       90     90       90     90
#
# STEP-BY-STEP MEANING:
# ------------------------------------------------------------
#
# STEP 0 - Rear protection posture
#
#     Front knees:
#         FL_KNEE = 120
#         FR_KNEE = 120
#
#     Rear knees:
#         RL_KNEE = 150
#         RR_KNEE = 150
#
#     Purpose:
#         Rear side is on the ground, so rear legs are folded.
#         Front legs are partially prepared for balance.
#
# STEP 1 - Push rear/body upward
#
#     Front legs:
#         FL_HIP = 70,  FL_KNEE = 140
#         FR_HIP = 70,  FR_KNEE = 140
#
#     Rear legs:
#         RL_HIP = 120, RL_KNEE = 70
#         RR_HIP = 120, RR_KNEE = 70
#
#     Numerical example:
#
#         Rear knee extension = 150 - 70 = 80 degrees
#
#     Purpose:
#         Rear legs extend and push against the floor, rotating the
#         body forward toward standing.
#
# STEP 2 - Stabilize the body
#
#     Front legs move toward 90/110.
#     Rear legs move toward 100/90.
#
#     Purpose:
#         Reduce bounce and place legs under the body.
#
# STEP 3 - Neutral stand
#
#     All hips  = 90
#     All knees = 90
#
#     Purpose:
#         Finish standing.
#
# ============================================================
# RECOVERY TABLE 5: recover_upside_down
# ============================================================
#
# FALL POSITION:
# ------------------------------------------------------------
# Robot is completely upside down or badly twisted.
#
# Typical orientation example:
#
#     roll  = around +180 or -180 degrees
#     pitch = may also be very large
#
# Manual command:
#
#     recover_upside_down
#
# FULL ANGLE TABLE:
# ------------------------------------------------------------
#
#     STEP    FL_HIP FL_KNEE  FR_HIP FR_KNEE  RL_HIP RL_KNEE  RR_HIP RR_KNEE
#     0       90     150      90     150      90     150      90     150
#     1       130    70       60     145      60     145      130    70
#     2       60     145      130    70       130    70       60     145
#     3       90     120      90     120      90     120      90     120
#     4       90     90       90     90       90     90       90     90
#
# STEP-BY-STEP MEANING:
# ------------------------------------------------------------
#
# STEP 0 - Compact curl
#
#     All hips  = 90
#     All knees = 150
#
#     Purpose:
#         Curl the robot into a compact shape so it can roll more
#         easily from an upside-down position.
#
# STEP 1 - Diagonal push A
#
#     Extended diagonal pair:
#         FL_HIP = 130, FL_KNEE = 70
#         RR_HIP = 130, RR_KNEE = 70
#
#     Folded diagonal pair:
#         FR_HIP = 60,  FR_KNEE = 145
#         RL_HIP = 60,  RL_KNEE = 145
#
#     Numerical example:
#
#         Extended diagonal knees = 70 degrees
#         Folded diagonal knees   = 145 degrees
#         Difference              = 145 - 70 = 75 degrees
#
#     Purpose:
#         Create diagonal torque to start rolling the body over.
#
# STEP 2 - Diagonal push B
#
#     Extended diagonal pair switches:
#         FR_HIP = 130, FR_KNEE = 70
#         RL_HIP = 130, RL_KNEE = 70
#
#     Folded diagonal pair switches:
#         FL_HIP = 60,  FL_KNEE = 145
#         RR_HIP = 60,  RR_KNEE = 145
#
#     Purpose:
#         Continue the roll if the first diagonal push is not enough.
#
# STEP 3 - Recovery crouch
#
#     All hips  = 90
#     All knees = 120
#
#     Purpose:
#         Absorb the landing after the robot rolls toward upright.
#
# STEP 4 - Neutral stand
#
#     All hips  = 90
#     All knees = 90
#
#     Purpose:
#         Finish standing.
#
# REAL-HARDWARE NOTE:
# ------------------------------------------------------------
# Upside-down recovery may need more than one attempt on the real
# robot. Later, with an IMU, the ESP32 can repeat recover_upside_down
# until roll and pitch return to safe values.
#
# Example retry logic for future ESP32 IMU version:
#
#     while abs(roll) > 45 or abs(pitch) > 45:
#         playMotion(recover_upside_down)
#         read_IMU_again()
#
# ============================================================

# RECOVERY PROCEDURE: LEFT-SIDE FALL
# ============================================================
#
# SCENARIO:
# ------------------------------------------------------------
# The robot is lying on its LEFT side.
#
# Example IMU reading:
#
#       roll  = +1.10 radians ≈ +63 degrees
#       pitch = +0.10 radians ≈  +6 degrees
#
# Detection:
#
#       abs(roll) = 1.10 > 0.8
#       roll is positive
#
# Result:
#
#       left_side_fall
#
# RECOVERY MECHANISM:
# ------------------------------------------------------------
# The robot must rotate back toward the right side until the body
# becomes upright again.
#
# The angle table does this in stages:
#
#   STEP 0 - Fold all knees
#       FL_KNEE = 145
#       FR_KNEE = 145
#       RL_KNEE = 145
#       RR_KNEE = 145
#
#       Purpose:
#           Curl legs inward so they do not block body rotation.
#
#   STEP 1 - Asymmetric side push
#       Left side:
#           FL_HIP = 60,  FL_KNEE = 145
#           RL_HIP = 60,  RL_KNEE = 145
#
#       Right side:
#           FR_HIP = 130, FR_KNEE = 70
#           RR_HIP = 130, RR_KNEE = 70
#
#       Numerical example:
#           Hip torque difference = 130 - 60 = 70 degrees
#           Knee extension difference = 145 - 70 = 75 degrees
#
#       Purpose:
#           One side stays folded while the other side extends,
#           creating a roll-back torque.
#
#   STEP 2 - Intermediate recovery push
#       Left side moves from 60/145 toward 70/130.
#       Right side moves from 130/70 toward 125/75.
#
#       Purpose:
#           Reduce shock and avoid snapping the body too quickly.
#
#   STEP 3 - Soft landing posture
#       All knees = 110
#
#       Purpose:
#           Absorb the return-to-ground motion.
#
#   STEP 4 - Neutral stand
#       All hips  = 90
#       All knees = 90
#
# SUCCESS EXAMPLE:
# ------------------------------------------------------------
# Before recovery:
#
#       roll  = +63 degrees
#       pitch =  +6 degrees
#
# After recovery:
#
#       roll  =  +8 degrees
#       pitch =  +3 degrees
#
# Check:
#
#       abs(roll)  = 8  < 45 degrees
#       abs(pitch) = 3  < 45 degrees
#
# Result:
#
#       recovery successful
#
# ============================================================

def generate_recover_from_left_fall_table():
    """
    Recovery sequence when the robot has fallen on its left side.

    Idea:
      - fold legs to avoid blocking rotation
      - push harder with the left-side legs
      - extend opposite side to roll body back upright
      - return to neutral stand

    Numerical example row:
      FL_HIP=60, FL_KNEE=145, FR_HIP=130, FR_KNEE=70

    Meaning:
      left/front leg folds and pushes while right/front leg extends,
      creating a roll-back torque.
    """

    rows = []
    make_row(rows, 90, 145, 90, 145, 90, 145, 90, 145)
    make_row(rows, 60, 145, 130, 70, 60, 145, 130, 70)
    make_row(rows, 70, 130, 125, 75, 70, 130, 125, 75)
    make_row(rows, 90, 110, 90, 110, 90, 110, 90, 110)
    neutral_row(rows)
    return rows



# ============================================================
# RECOVERY PROCEDURE: RIGHT-SIDE FALL
# ============================================================
#
# SCENARIO:
# ------------------------------------------------------------
# The robot is lying on its RIGHT side.
#
# Example IMU reading:
#
#       roll  = -1.15 radians ≈ -66 degrees
#       pitch = +0.06 radians ≈  +3 degrees
#
# Detection:
#
#       abs(roll) = 1.15 > 0.8
#       roll is negative
#
# Result:
#
#       right_side_fall
#
# RECOVERY MECHANISM:
# ------------------------------------------------------------
# This is the mirror image of the left-side recovery.
# The robot must rotate back toward the left side until upright.
#
# The angle table does this in stages:
#
#   STEP 0 - Fold all knees
#       all knees = 145
#
#   STEP 1 - Asymmetric side push
#       Left side:
#           FL_HIP = 130, FL_KNEE = 70
#           RL_HIP = 130, RL_KNEE = 70
#
#       Right side:
#           FR_HIP = 60,  FR_KNEE = 145
#           RR_HIP = 60,  RR_KNEE = 145
#
#       Numerical example:
#           Hip torque difference = 130 - 60 = 70 degrees
#           Knee extension difference = 145 - 70 = 75 degrees
#
#       Purpose:
#           The left side extends while the right side folds,
#           creating a roll-back torque in the opposite direction.
#
#   STEP 2 - Intermediate recovery push
#       Left side softens from 130/70 toward 125/75.
#       Right side softens from 60/145 toward 70/130.
#
#   STEP 3 - Soft landing posture
#       all knees = 110
#
#   STEP 4 - Neutral stand
#       all hips = 90, all knees = 90
#
# SUCCESS EXAMPLE:
# ------------------------------------------------------------
# Before recovery:
#
#       roll = -66 degrees
#
# After recovery:
#
#       roll = -7 degrees
#
# Check:
#
#       abs(-7) = 7 < 45 degrees
#
# Result:
#
#       recovery successful
#
# ============================================================

def generate_recover_from_right_fall_table():
    """
    Recovery sequence when the robot has fallen on its right side.

    This mirrors the left-side recovery.

    Numerical example row:
      FL_HIP=130, FL_KNEE=70, FR_HIP=60, FR_KNEE=145

    Meaning:
      right-side legs fold and push while left-side legs extend,
      creating a roll-back torque.
    """

    rows = []
    make_row(rows, 90, 145, 90, 145, 90, 145, 90, 145)
    make_row(rows, 130, 70, 60, 145, 130, 70, 60, 145)
    make_row(rows, 125, 75, 70, 130, 125, 75, 70, 130)
    make_row(rows, 90, 110, 90, 110, 90, 110, 90, 110)
    neutral_row(rows)
    return rows



# ============================================================
# RECOVERY PROCEDURE: FRONT FALL
# ============================================================
#
# SCENARIO:
# ------------------------------------------------------------
# The robot has fallen forward, meaning its front/body nose is down.
#
# Example IMU reading:
#
#       roll  = +0.12 radians ≈  +7 degrees
#       pitch = +1.20 radians ≈ +69 degrees
#
# Detection:
#
#       abs(pitch) = 1.20 > 0.8
#       pitch is positive
#
# Result:
#
#       front_fall
#
# RECOVERY MECHANISM:
# ------------------------------------------------------------
# The robot must lift the front side and bring the rear legs under
# the body.
#
# The angle table does this in stages:
#
#   STEP 0 - Front protection posture
#       Front knees:
#           FL_KNEE = 150
#           FR_KNEE = 150
#
#       Rear knees:
#           RL_KNEE = 120
#           RR_KNEE = 120
#
#       Purpose:
#           Fold/protect the front side while preparing rear support.
#
#   STEP 1 - Push front up / bring rear under body
#       Front:
#           FL_HIP = 120, FL_KNEE = 70
#           FR_HIP = 120, FR_KNEE = 70
#
#       Rear:
#           RL_HIP = 70,  RL_KNEE = 140
#           RR_HIP = 70,  RR_KNEE = 140
#
#       Numerical example:
#           Front knee extension = 150 - 70 = 80 degrees
#
#       Purpose:
#           Extending the front legs pushes against the floor and
#           helps rotate the body upward.
#
#   STEP 2 - Stabilization
#       Front hips/knees approach 100/90.
#       Rear hips/knees approach 90/110.
#
#       Purpose:
#           Avoid overshooting backward after pushing up.
#
#   STEP 3 - Neutral stand
#       all hips = 90, all knees = 90
#
# SUCCESS EXAMPLE:
# ------------------------------------------------------------
# Before recovery:
#
#       pitch = +69 degrees
#
# After recovery:
#
#       pitch = +5 degrees
#
# Check:
#
#       abs(5) < 45 degrees
#
# Result:
#
#       recovery successful
#
# ============================================================

def generate_recover_from_front_fall_table():
    """
    Recovery sequence when the robot has fallen forward.

    Idea:
      - fold front knees
      - push with front legs
      - bring rear legs under the body
      - return to neutral

    Numerical example:
      front knees  = 150
      rear knees   = 70

    This makes the front side push while the rear side extends to help
    rotate the body back to a standing posture.
    """

    rows = []
    make_row(rows, 90, 150, 90, 150, 90, 120, 90, 120)
    make_row(rows, 120, 70, 120, 70, 70, 140, 70, 140)
    make_row(rows, 100, 90, 100, 90, 90, 110, 90, 110)
    neutral_row(rows)
    return rows



# ============================================================
# RECOVERY PROCEDURE: BACK FALL
# ============================================================
#
# SCENARIO:
# ------------------------------------------------------------
# The robot has fallen backward, meaning the rear/body tail side is
# down and the front side may be high.
#
# Example IMU reading:
#
#       roll  = +0.08 radians ≈  +5 degrees
#       pitch = -1.05 radians ≈ -60 degrees
#
# Detection:
#
#       abs(pitch) = 1.05 > 0.8
#       pitch is negative
#
# Result:
#
#       back_fall
#
# RECOVERY MECHANISM:
# ------------------------------------------------------------
# This mirrors the front-fall recovery. The robot must push using
# the rear legs and bring the front legs under the body.
#
# The angle table does this in stages:
#
#   STEP 0 - Rear protection posture
#       Rear knees:
#           RL_KNEE = 150
#           RR_KNEE = 150
#
#       Front knees:
#           FL_KNEE = 120
#           FR_KNEE = 120
#
#   STEP 1 - Rear push
#       Front:
#           FL_HIP = 70,  FL_KNEE = 140
#           FR_HIP = 70,  FR_KNEE = 140
#
#       Rear:
#           RL_HIP = 120, RL_KNEE = 70
#           RR_HIP = 120, RR_KNEE = 70
#
#       Numerical example:
#           Rear knee extension = 150 - 70 = 80 degrees
#
#       Purpose:
#           Rear legs extend against the floor and rotate body
#           toward upright.
#
#   STEP 2 - Stabilization
#       Front moves toward 90/110.
#       Rear moves toward 100/90.
#
#   STEP 3 - Neutral stand
#       all hips = 90, all knees = 90
#
# SUCCESS EXAMPLE:
# ------------------------------------------------------------
# Before recovery:
#
#       pitch = -60 degrees
#
# After recovery:
#
#       pitch = -6 degrees
#
# Check:
#
#       abs(-6) = 6 < 45 degrees
#
# Result:
#
#       recovery successful
#
# ============================================================

def generate_recover_from_back_fall_table():
    """
    Recovery sequence when the robot has fallen backward.

    Idea:
      - fold rear knees
      - push with rear legs
      - bring front legs under the body
      - return to neutral

    Numerical example:
      rear knees  = 150
      front knees = 70

    This mirrors front-fall recovery.
    """

    rows = []
    make_row(rows, 90, 120, 90, 120, 90, 150, 90, 150)
    make_row(rows, 70, 140, 70, 140, 120, 70, 120, 70)
    make_row(rows, 90, 110, 90, 110, 100, 90, 100, 90)
    neutral_row(rows)
    return rows



# ============================================================
# RECOVERY PROCEDURE: UPSIDE-DOWN FALL
# ============================================================
#
# SCENARIO:
# ------------------------------------------------------------
# The robot is upside down or badly twisted. This is harder than
# side/front/back recovery because the robot may need a larger body
# roll before it can return to a normal standing posture.
#
# Example orientation:
#
#       roll = 3.14 radians ≈ 180 degrees
#
# Result:
#
#       upside_down_fall
#
# RECOVERY MECHANISM:
# ------------------------------------------------------------
# The robot creates diagonal imbalance by extending one diagonal pair
# while folding the other diagonal pair.
#
# The angle table does this in stages:
#
#   STEP 0 - Curl all legs
#       all knees = 150
#
#       Purpose:
#           Make the body compact so it can roll.
#
#   STEP 1 - Diagonal push A
#       FL = (130, 70)
#       FR = ( 60,145)
#       RL = ( 60,145)
#       RR = (130, 70)
#
#       Numerical example:
#           Extended diagonal knees = 70 degrees
#           Folded diagonal knees   = 145 degrees
#           Difference              = 75 degrees
#
#       Purpose:
#           Create rotational imbalance around a diagonal axis.
#
#   STEP 2 - Diagonal push B
#       FL = ( 60,145)
#       FR = (130, 70)
#       RL = (130, 70)
#       RR = ( 60,145)
#
#       Purpose:
#           Continue the roll if the first diagonal push is not enough.
#
#   STEP 3 - Recovery crouch
#       all knees = 120
#
#       Purpose:
#           Absorb landing and prepare to stand.
#
#   STEP 4 - Neutral stand
#       all hips = 90, all knees = 90
#
# SUCCESS EXAMPLE:
# ------------------------------------------------------------
# Before recovery:
#
#       roll = 180 degrees
#
# During recovery:
#
#       roll = 120 degrees
#       roll =  65 degrees
#       roll =  20 degrees
#
# After recovery:
#
#       roll = 8 degrees
#
# Check:
#
#       abs(8) < 45 degrees
#
# Result:
#
#       recovery successful
#
# NOTE:
# ------------------------------------------------------------
# Upside-down recovery may require multiple attempts on real hardware.
# In a future IMU-based ESP32 implementation, you can retry this table
# until roll and pitch are inside safe thresholds.
#
# ============================================================

def generate_recover_from_upside_down_table():
    """
    Recovery sequence when the robot is upside down or badly twisted.

    Idea:
      - curl all legs
      - twist hips asymmetrically
      - extend one diagonal pair
      - roll toward one side
      - return to neutral

    Numerical example:
      diagonal push row:
        FL=(130,70), FR=(60,145), RL=(60,145), RR=(130,70)

    This attempts to create enough rotational imbalance to self-right.
    """

    rows = []
    make_row(rows, 90, 150, 90, 150, 90, 150, 90, 150)
    make_row(rows, 130, 70, 60, 145, 60, 145, 130, 70)
    make_row(rows, 60, 145, 130, 70, 130, 70, 60, 145)
    make_row(rows, 90, 120, 90, 120, 90, 120, 90, 120)
    neutral_row(rows)
    return rows


RECOVERY_MOTIONS = {
    "left_side_fall": {
        "name": "recover_left",
        "label": "Recover From Left-Side Fall",
        "generator": generate_recover_from_left_fall_table,
        "repeat_on_esp32": 1,
    },
    "right_side_fall": {
        "name": "recover_right",
        "label": "Recover From Right-Side Fall",
        "generator": generate_recover_from_right_fall_table,
        "repeat_on_esp32": 1,
    },
    "front_fall": {
        "name": "recover_front",
        "label": "Recover From Front Fall",
        "generator": generate_recover_from_front_fall_table,
        "repeat_on_esp32": 1,
    },
    "back_fall": {
        "name": "recover_back",
        "label": "Recover From Back Fall",
        "generator": generate_recover_from_back_fall_table,
        "repeat_on_esp32": 1,
    },
    "upside_down_fall": {
        "name": "recover_upside_down",
        "label": "Recover From Upside-Down Fall",
        "generator": generate_recover_from_upside_down_table,
        "repeat_on_esp32": 1,
    },
}


# ============================================================
# SIMPLE TRAINING / OPTIMIZATION
# ============================================================
#
# This is a lightweight training stage.
#
# It does NOT do full reinforcement learning yet.
# It automatically searches for improved motion angles by trying
# many possible hip/knee values and keeping the best scored set.
#
# ESP32 will still replay the final generated table.
#
# IMPORTANT DESIGN CHANGE:
# ------------------------------------------------------------
# The original script trained ONLY walking.
# This version adds a training procedure for EACH motion family.
#
# The training method used here is:
#
#   1. Generate random candidate angles
#   2. Build a candidate motion table
#   3. Score the candidate using motion-specific objectives
#   4. Keep the best candidate
#   5. Return the best motion table for simulation/export
#
# This is called random-search optimization.
# It is not deep learning and not full reinforcement learning.
#
# Numerical idea:
# ------------------------------------------------------------
# Suppose a candidate walk has:
#
#   forward_hip  = 118
#   backward_hip = 62
#   lift_knee    = 124
#   support_knee = 91
#
# Then:
#
#   hip_swing = abs(118 - 62) = 56
#   knee_lift = abs(124 - 91) = 33
#
# Base score:
#
#   hip reward  = 56 * 2 = 112
#   knee reward = 33 * 2 = 66
#
# Preferred-range bonuses:
#
#   forward_hip inside 108..125  -> +30
#   backward_hip inside 55..72   -> +30
#   lift_knee inside 108..130    -> +30
#   support_knee inside 88..96   -> +30
#
# Total:
#
#   112 + 66 + 30 + 30 + 30 + 30 = 298
#
# The trainer compares this score against other random candidates.
# The highest score wins.
# ============================================================

def clamp_angle(value, low=45, high=150):
    """
    Keeps servo angles inside a safe range.
    """
    return max(low, min(high, int(value)))


def safe_angle_penalty(*angles):
    """
    Penalizes unsafe servo angles.

    Numerical example:
      angle = 160 -> unsafe because 160 > 150 -> -100
      angle = 35  -> unsafe because 35 < 45  -> -100
      angle = 120 -> safe -> 0
    """
    penalty = 0
    for angle in angles:
        if angle < 45 or angle > 150:
            penalty -= 100
    return penalty


def build_parametric_walk_table(forward_hip, backward_hip, lift_knee, support_knee):
    rows = []

    make_row(rows, 90, support_knee, 90, support_knee, 90, support_knee, 90, support_knee)

    make_row(rows, forward_hip, lift_knee, 90, support_knee, 90, support_knee, 90, support_knee)
    make_row(rows, forward_hip, support_knee, 90, support_knee, 90, support_knee, 90, support_knee)
    make_row(rows, backward_hip, support_knee, 90, support_knee, backward_hip, support_knee, 90, support_knee)

    make_row(rows, 90, support_knee, forward_hip, lift_knee, 90, support_knee, 90, support_knee)
    make_row(rows, 90, support_knee, forward_hip, support_knee, 90, support_knee, 90, support_knee)
    make_row(rows, 90, support_knee, backward_hip, support_knee, 90, support_knee, backward_hip, support_knee)

    make_row(rows, 90, support_knee, 90, support_knee, forward_hip, lift_knee, 90, support_knee)
    make_row(rows, 90, support_knee, 90, support_knee, forward_hip, support_knee, 90, support_knee)
    make_row(rows, backward_hip, support_knee, 90, support_knee, backward_hip, support_knee, 90, support_knee)

    make_row(rows, 90, support_knee, 90, support_knee, 90, support_knee, forward_hip, lift_knee)
    make_row(rows, 90, support_knee, 90, support_knee, 90, support_knee, forward_hip, support_knee)
    make_row(rows, 90, support_knee, backward_hip, support_knee, 90, support_knee, backward_hip, support_knee)

    make_row(rows, 90, support_knee, 90, support_knee, 90, support_knee, 90, support_knee)

    return rows


def score_walk_candidate(forward_hip, backward_hip, lift_knee, support_knee):
    score = 0

    hip_swing = abs(forward_hip - backward_hip)
    knee_lift = abs(lift_knee - support_knee)

    score += hip_swing * 2
    score += knee_lift * 2

    if 108 <= forward_hip <= 125:
        score += 30
    if 55 <= backward_hip <= 72:
        score += 30
    if 108 <= lift_knee <= 130:
        score += 30
    if 88 <= support_knee <= 96:
        score += 30

    score += safe_angle_penalty(forward_hip, backward_hip, lift_knee, support_knee)

    if hip_swing > 80:
        score -= 40
    if knee_lift > 55:
        score -= 30

    return score


# ============================================================
# TRAIN WALK
# ============================================================
#
# Goal:
#   Improve normal forward walking.
#
# Random parameters:
#   forward_hip, backward_hip, lift_knee, support_knee
#
# What is rewarded:
#   - large but safe hip swing
#   - useful knee lift
#   - angles inside preferred ranges
#
# Numerical example:
#   forward_hip  = 116
#   backward_hip = 65
#   lift_knee    = 120
#   support_knee = 92
#
#   hip_swing = abs(116 - 65) = 51
#   knee_lift = abs(120 - 92) = 28
#
#   base score = 51*2 + 28*2 = 102 + 56 = 158
#   bonuses    = +30 +30 +30 +30 = 120
#   total      = 278
# ============================================================

def train_optimized_walk_table(iterations=250):
    import random

    print("\n================================================")
    print(" TRAIN / OPTIMIZE WALKING MOTION")
    print("================================================")

    best_score = -999999
    best_params = None

    for i in range(iterations):
        forward_hip = random.randint(105, 130)
        backward_hip = random.randint(50, 75)
        lift_knee = random.randint(105, 135)
        support_knee = random.randint(85, 100)

        score = score_walk_candidate(forward_hip, backward_hip, lift_knee, support_knee)

        if score > best_score:
            best_score = score
            best_params = (forward_hip, backward_hip, lift_knee, support_knee)
            print("New best walk:", best_score, best_params)

    return build_parametric_walk_table(*best_params)


def generate_trained_walk_table():
    return train_optimized_walk_table(iterations=250)


# ============================================================
# TRAIN RUN / TROT
# ============================================================
#
# Goal:
#   Improve fast diagonal gait.
#
# Difference from walk:
#   Walk usually moves one leg at a time.
#   Run/trot moves diagonal legs together:
#
#      FL + RR
#      FR + RL
#
# Numerical example:
#   forward_hip  = 128
#   backward_hip = 55
#   lift_knee    = 132
#   support_knee = 86
#
#   hip_swing = abs(128 - 55) = 73
#   knee_lift = abs(132 - 86) = 46
#
#   speed reward = hip_swing*3 + knee_lift*2
#                = 73*3 + 46*2
#                = 219 + 92 = 311
#
#   If angles are safe and not too extreme, the candidate wins.
# ============================================================

def build_parametric_run_table(forward_hip, backward_hip, lift_knee, support_knee):
    rows = []
    neutral_row(rows)
    make_row(rows, forward_hip, lift_knee, backward_hip, support_knee, backward_hip, support_knee, forward_hip, lift_knee)
    make_row(rows, forward_hip, support_knee, backward_hip, support_knee, backward_hip, support_knee, forward_hip, support_knee)
    make_row(rows, backward_hip, support_knee, forward_hip, lift_knee, forward_hip, lift_knee, backward_hip, support_knee)
    make_row(rows, backward_hip, support_knee, forward_hip, support_knee, forward_hip, support_knee, backward_hip, support_knee)
    neutral_row(rows)
    return rows


def train_optimized_run_table(iterations=250):
    import random
    print("\n================================================")
    print(" TRAIN / OPTIMIZE RUN / TROT MOTION")
    print("================================================")
    best_score = -999999
    best_params = None
    for i in range(iterations):
        forward_hip = random.randint(118, 140)
        backward_hip = random.randint(45, 65)
        lift_knee = random.randint(115, 145)
        support_knee = random.randint(80, 95)
        hip_swing = abs(forward_hip - backward_hip)
        knee_lift = abs(lift_knee - support_knee)
        score = hip_swing * 3 + knee_lift * 2
        score += 40 if hip_swing >= 60 else 0
        score += 30 if knee_lift >= 30 else 0
        score += safe_angle_penalty(forward_hip, backward_hip, lift_knee, support_knee)
        if hip_swing > 95:
            score -= 50
        if score > best_score:
            best_score = score
            best_params = (forward_hip, backward_hip, lift_knee, support_knee)
            print("New best run:", best_score, best_params)
    return build_parametric_run_table(*best_params)


def generate_trained_run_table():
    return train_optimized_run_table(iterations=250)


# ============================================================
# TRAIN SIT
# ============================================================
#
# Goal:
#   Lower rear body smoothly while keeping front legs stable.
#
# Random parameters:
#   rear_hip, rear_knee, mid_hip, mid_knee
#
# Numerical example:
#   rear_hip  = 62
#   rear_knee = 142
#   mid_hip   = 75
#   mid_knee  = 128
#
#   target rear_hip  is near 60 -> error = abs(62-60)=2
#   target rear_knee is near 140 -> error = abs(142-140)=2
#
#   score = 200 - error penalties
#         = 200 - (2*3) - (2*3)
#         = 188
# ============================================================

def build_parametric_sit_table(mid_hip, mid_knee, rear_hip, rear_knee):
    rows = []
    neutral_row(rows)
    make_row(rows, 90, 90, 90, 90, mid_hip, mid_knee, mid_hip, mid_knee)
    make_row(rows, 90, 90, 90, 90, rear_hip, rear_knee, rear_hip, rear_knee)
    return rows


def train_optimized_sit_table(iterations=200):
    import random
    print("\n================================================")
    print(" TRAIN / OPTIMIZE SIT MOTION")
    print("================================================")
    best_score = -999999
    best_params = None
    for i in range(iterations):
        mid_hip = random.randint(68, 85)
        mid_knee = random.randint(115, 135)
        rear_hip = random.randint(50, 75)
        rear_knee = random.randint(130, 150)
        score = 200
        score -= abs(rear_hip - 60) * 3
        score -= abs(rear_knee - 140) * 3
        score -= abs(mid_hip - ((90 + rear_hip) // 2))
        score -= abs(mid_knee - ((90 + rear_knee) // 2))
        score += safe_angle_penalty(mid_hip, mid_knee, rear_hip, rear_knee)
        if score > best_score:
            best_score = score
            best_params = (mid_hip, mid_knee, rear_hip, rear_knee)
            print("New best sit:", best_score, best_params)
    return build_parametric_sit_table(*best_params)


def generate_trained_sit_table():
    return train_optimized_sit_table(iterations=200)


# ============================================================
# TRAIN STAND
# ============================================================
#
# Goal:
#   Return to stable neutral pose.
#
# Numerical example:
#   hip = 91, knee = 89
#   hip error  = abs(91-90)=1
#   knee error = abs(89-90)=1
#   score = 100 - 1 - 1 = 98
# ============================================================

def train_optimized_stand_table(iterations=100):
    import random
    print("\n================================================")
    print(" TRAIN / OPTIMIZE STAND MOTION")
    print("================================================")
    best_score = -999999
    best_params = None
    for i in range(iterations):
        hip = random.randint(86, 94)
        knee = random.randint(86, 94)
        score = 100 - abs(hip - 90) - abs(knee - 90)
        if score > best_score:
            best_score = score
            best_params = (hip, knee)
            print("New best stand:", best_score, best_params)
    hip, knee = best_params
    rows = []
    make_row(rows, hip, knee, hip, knee, hip, knee, hip, knee)
    return rows


def generate_trained_stand_table():
    return train_optimized_stand_table(iterations=100)


# ============================================================
# TRAIN JUMP
# ============================================================
#
# Goal:
#   Crouch, extend legs, then recover.
#
# Numerical example:
#   crouch_knee = 138
#   extend_knee = 68
#
#   extension_power = abs(138 - 68) = 70
#   target crouch close to 135 -> error = 3
#   target extend close to 70  -> error = 2
#
#   score = extension_power*3 - error penalties
#         = 70*3 - 3*2 - 2*2
#         = 210 - 6 - 4 = 200
# ============================================================

def build_parametric_jump_table(crouch_knee, extend_knee, recover_knee, hip_angle):
    rows = []
    neutral_row(rows)
    make_row(rows, hip_angle, 120, hip_angle, 120, hip_angle, 120, hip_angle, 120)
    make_row(rows, hip_angle, crouch_knee, hip_angle, crouch_knee, hip_angle, crouch_knee, hip_angle, crouch_knee)
    make_row(rows, 90, extend_knee, 90, extend_knee, 90, extend_knee, 90, extend_knee)
    make_row(rows, 90, recover_knee, 90, recover_knee, 90, recover_knee, 90, recover_knee)
    neutral_row(rows)
    return rows


def train_optimized_jump_table(iterations=250):
    import random
    print("\n================================================")
    print(" TRAIN / OPTIMIZE JUMP MOTION")
    print("================================================")
    best_score = -999999
    best_params = None
    for i in range(iterations):
        crouch_knee = random.randint(125, 150)
        extend_knee = random.randint(55, 85)
        recover_knee = random.randint(105, 130)
        hip_angle = random.randint(82, 102)
        extension_power = abs(crouch_knee - extend_knee)
        score = extension_power * 3
        score -= abs(crouch_knee - 135) * 2
        score -= abs(extend_knee - 70) * 2
        score -= abs(hip_angle - 90)
        score += safe_angle_penalty(crouch_knee, extend_knee, recover_knee, hip_angle)
        if score > best_score:
            best_score = score
            best_params = (crouch_knee, extend_knee, recover_knee, hip_angle)
            print("New best jump:", best_score, best_params)
    return build_parametric_jump_table(*best_params)


def generate_trained_jump_table():
    return train_optimized_jump_table(iterations=250)


# ============================================================
# TRAIN TURN LEFT / RIGHT
# ============================================================
#
# Goal:
#   Create opposite diagonal leg pattern to rotate the body.
#
# Numerical example:
#   inside_hip  = 58
#   outside_hip = 125
#   lift_knee   = 122
#
#   yaw_drive = abs(125 - 58) = 67
#   score = yaw_drive*3 + lift bonus
#         = 67*3 + 30 = 231
# ============================================================

def build_parametric_turn_table(direction, inside_hip, outside_hip, lift_knee, support_knee):
    rows = []
    neutral_row(rows)
    if direction == "left":
        make_row(rows, inside_hip, support_knee, outside_hip, lift_knee, inside_hip, support_knee, outside_hip, lift_knee)
        make_row(rows, outside_hip, lift_knee, inside_hip, support_knee, outside_hip, lift_knee, inside_hip, support_knee)
    else:
        make_row(rows, outside_hip, lift_knee, inside_hip, support_knee, outside_hip, lift_knee, inside_hip, support_knee)
        make_row(rows, inside_hip, support_knee, outside_hip, lift_knee, inside_hip, support_knee, outside_hip, lift_knee)
    neutral_row(rows)
    return rows


def train_optimized_turn_table(direction="left", iterations=220):
    import random
    print("\n================================================")
    print(" TRAIN / OPTIMIZE TURN", direction.upper(), "MOTION")
    print("================================================")
    best_score = -999999
    best_params = None
    for i in range(iterations):
        inside_hip = random.randint(50, 70)
        outside_hip = random.randint(112, 135)
        lift_knee = random.randint(110, 135)
        support_knee = random.randint(85, 100)
        yaw_drive = abs(outside_hip - inside_hip)
        score = yaw_drive * 3
        score += 30 if lift_knee >= 115 else 0
        score -= abs(support_knee - 90)
        score += safe_angle_penalty(inside_hip, outside_hip, lift_knee, support_knee)
        if yaw_drive > 90:
            score -= 40
        if score > best_score:
            best_score = score
            best_params = (inside_hip, outside_hip, lift_knee, support_knee)
            print("New best turn", direction + ":", best_score, best_params)
    return build_parametric_turn_table(direction, *best_params)


def generate_trained_turn_left_table():
    return train_optimized_turn_table("left", iterations=220)


def generate_trained_turn_right_table():
    return train_optimized_turn_table("right", iterations=220)


# ============================================================
# TRAIN WALK BACKWARD
# ============================================================
#
# Goal:
#   Use reverse hip pattern to move backward.
#
# Numerical example:
#   reverse_forward_hip = 62
#   reverse_back_hip    = 122
#
#   reverse swing = abs(62 - 122) = 60
#   score = reverse swing*2 + knee lift reward
# ============================================================

def build_parametric_walk_backward_table(reverse_forward_hip, reverse_back_hip, lift_knee, support_knee):
    rows = []
    neutral_row(rows)
    make_row(rows, reverse_forward_hip, lift_knee, 90, support_knee, 90, support_knee, 90, support_knee)
    make_row(rows, reverse_back_hip, support_knee, 90, support_knee, reverse_back_hip, support_knee, 90, support_knee)
    make_row(rows, 90, support_knee, reverse_forward_hip, lift_knee, 90, support_knee, 90, support_knee)
    make_row(rows, 90, support_knee, reverse_back_hip, support_knee, 90, support_knee, reverse_back_hip, support_knee)
    neutral_row(rows)
    return rows


def train_optimized_walk_backward_table(iterations=220):
    import random
    print("\n================================================")
    print(" TRAIN / OPTIMIZE WALK BACKWARD MOTION")
    print("================================================")
    best_score = -999999
    best_params = None
    for i in range(iterations):
        reverse_forward_hip = random.randint(50, 75)
        reverse_back_hip = random.randint(110, 135)
        lift_knee = random.randint(110, 135)
        support_knee = random.randint(85, 100)
        score = abs(reverse_back_hip - reverse_forward_hip) * 2
        score += abs(lift_knee - support_knee) * 2
        score += 30 if 55 <= reverse_forward_hip <= 72 else 0
        score += 30 if 112 <= reverse_back_hip <= 130 else 0
        score += safe_angle_penalty(reverse_forward_hip, reverse_back_hip, lift_knee, support_knee)
        if score > best_score:
            best_score = score
            best_params = (reverse_forward_hip, reverse_back_hip, lift_knee, support_knee)
            print("New best backward:", best_score, best_params)
    return build_parametric_walk_backward_table(*best_params)


def generate_trained_walk_backward_table():
    return train_optimized_walk_backward_table(iterations=220)


# ============================================================
# TRAIN CRAWL
# ============================================================
#
# Goal:
#   Move forward while keeping the body low.
#
# Numerical example:
#   crawl_knee = 118
#   lift_knee  = 128
#   forward_hip = 112
#   backward_hip = 72
#
#   low posture reward = abs(118-115) small -> good
#   stride = abs(112-72)=40
#   score = stride*2 + low posture reward
# ============================================================

def build_parametric_crawl_table(forward_hip, backward_hip, crawl_knee, lift_knee):
    rows = []
    make_row(rows, 90, crawl_knee, 90, crawl_knee, 90, crawl_knee, 90, crawl_knee)
    make_row(rows, forward_hip, lift_knee, 90, crawl_knee, 90, crawl_knee, 90, crawl_knee)
    make_row(rows, backward_hip, crawl_knee, 90, crawl_knee, backward_hip, crawl_knee, 90, crawl_knee)
    make_row(rows, 90, crawl_knee, forward_hip, lift_knee, 90, crawl_knee, 90, crawl_knee)
    make_row(rows, 90, crawl_knee, backward_hip, crawl_knee, 90, crawl_knee, backward_hip, crawl_knee)
    neutral_row(rows)
    return rows


def train_optimized_crawl_table(iterations=220):
    import random
    print("\n================================================")
    print(" TRAIN / OPTIMIZE CRAWL MOTION")
    print("================================================")
    best_score = -999999
    best_params = None
    for i in range(iterations):
        forward_hip = random.randint(105, 125)
        backward_hip = random.randint(65, 85)
        crawl_knee = random.randint(108, 125)
        lift_knee = random.randint(118, 138)
        stride = abs(forward_hip - backward_hip)
        low_posture_score = 60 - abs(crawl_knee - 115) * 3
        score = stride * 2 + low_posture_score + abs(lift_knee - crawl_knee)
        score += safe_angle_penalty(forward_hip, backward_hip, crawl_knee, lift_knee)
        if score > best_score:
            best_score = score
            best_params = (forward_hip, backward_hip, crawl_knee, lift_knee)
            print("New best crawl:", best_score, best_params)
    return build_parametric_crawl_table(*best_params)


def generate_trained_crawl_table():
    return train_optimized_crawl_table(iterations=220)


# ============================================================
# TRAIN BOW / STRETCH
# ============================================================
#
# Goal:
#   Bow lowers the front while keeping rear more stable.
#   Stretch extends the body with front/rear opposing posture.
#
# Numerical bow example:
#   front_hip  = 120
#   front_knee = 140
#   rear_hip   = 82
#   score rewards front lowering and rear stability.
# ============================================================

def build_parametric_bow_table(front_hip, front_knee, rear_hip, rear_knee):
    rows = []
    neutral_row(rows)
    make_row(rows, front_hip-10, front_knee-15, front_hip-10, front_knee-15, rear_hip+5, rear_knee, rear_hip+5, rear_knee)
    make_row(rows, front_hip, front_knee, front_hip, front_knee, rear_hip, rear_knee, rear_hip, rear_knee)
    neutral_row(rows)
    return rows


def train_optimized_bow_table(iterations=180):
    import random
    print("\n================================================")
    print(" TRAIN / OPTIMIZE BOW MOTION")
    print("================================================")
    best_score = -999999
    best_params = None
    for i in range(iterations):
        front_hip = random.randint(110, 130)
        front_knee = random.randint(125, 150)
        rear_hip = random.randint(75, 95)
        rear_knee = random.randint(85, 100)
        score = (front_hip - 90) + (front_knee - 90)
        score -= abs(rear_knee - 90)
        score += safe_angle_penalty(front_hip, front_knee, rear_hip, rear_knee)
        if score > best_score:
            best_score = score
            best_params = (front_hip, front_knee, rear_hip, rear_knee)
            print("New best bow:", best_score, best_params)
    return build_parametric_bow_table(*best_params)


def generate_trained_bow_table():
    return train_optimized_bow_table(iterations=180)


def build_parametric_stretch_table(front_hip, front_knee, rear_hip, rear_knee):
    rows = []
    neutral_row(rows)
    make_row(rows, front_hip, front_knee, front_hip, front_knee, rear_hip, rear_knee, rear_hip, rear_knee)
    make_row(rows, front_hip+5, front_knee, front_hip+5, front_knee, rear_hip-5, rear_knee+5, rear_hip-5, rear_knee+5)
    neutral_row(rows)
    return rows


def train_optimized_stretch_table(iterations=180):
    import random
    print("\n================================================")
    print(" TRAIN / OPTIMIZE STRETCH MOTION")
    print("================================================")
    best_score = -999999
    best_params = None
    for i in range(iterations):
        front_hip = random.randint(115, 140)
        front_knee = random.randint(90, 115)
        rear_hip = random.randint(55, 80)
        rear_knee = random.randint(105, 130)
        extension = abs(front_hip - rear_hip) + abs(rear_knee - front_knee)
        score = extension
        score += safe_angle_penalty(front_hip, front_knee, rear_hip, rear_knee)
        if score > best_score:
            best_score = score
            best_params = (front_hip, front_knee, rear_hip, rear_knee)
            print("New best stretch:", best_score, best_params)
    return build_parametric_stretch_table(*best_params)


def generate_trained_stretch_table():
    return train_optimized_stretch_table(iterations=180)


# ============================================================
# TRAIN SIDE STEP LEFT / RIGHT
# ============================================================
#
# Goal:
#   Shift body laterally by making one side lift/extend more than the other.
#
# Numerical example:
#   side_hip_a = 112
#   side_hip_b = 78
#   lift_knee = 124
#   lateral_score = abs(112-78)*2 + abs(124-90)
#                 = 68 + 34 = 102
# ============================================================

def build_parametric_side_step_table(direction, hip_a, hip_b, lift_knee, support_knee):
    rows = []
    neutral_row(rows)
    if direction == "left":
        make_row(rows, hip_a, lift_knee, hip_a, support_knee, hip_a, lift_knee, hip_a, support_knee)
        make_row(rows, 100, support_knee, hip_b, support_knee, 100, support_knee, hip_b, support_knee)
        make_row(rows, 90, support_knee, 90, lift_knee, 90, support_knee, 90, lift_knee)
    else:
        make_row(rows, hip_a, support_knee, hip_a, lift_knee, hip_a, support_knee, hip_a, lift_knee)
        make_row(rows, hip_b, support_knee, 100, support_knee, hip_b, support_knee, 100, support_knee)
        make_row(rows, 90, lift_knee, 90, support_knee, 90, lift_knee, 90, support_knee)
    neutral_row(rows)
    return rows


def train_optimized_side_step_table(direction="left", iterations=180):
    import random
    print("\n================================================")
    print(" TRAIN / OPTIMIZE SIDE STEP", direction.upper(), "MOTION")
    print("================================================")
    best_score = -999999
    best_params = None
    for i in range(iterations):
        hip_a = random.randint(105, 125)
        hip_b = random.randint(70, 90)
        lift_knee = random.randint(110, 135)
        support_knee = random.randint(85, 100)
        score = abs(hip_a - hip_b) * 2 + abs(lift_knee - support_knee)
        score -= abs(support_knee - 90)
        score += safe_angle_penalty(hip_a, hip_b, lift_knee, support_knee)
        if score > best_score:
            best_score = score
            best_params = (hip_a, hip_b, lift_knee, support_knee)
            print("New best side", direction + ":", best_score, best_params)
    return build_parametric_side_step_table(direction, *best_params)


def generate_trained_side_left_table():
    return train_optimized_side_step_table("left", iterations=180)


def generate_trained_side_right_table():
    return train_optimized_side_step_table("right", iterations=180)


# ============================================================
# TRAIN WAVE LEFT / RIGHT
# ============================================================
#
# Goal:
#   Lift one front paw and move it back/forth.
#
# Numerical example:
#   wave_high_knee = 145
#   wave_low_knee  = 118
#   wave_hip_a     = 130
#   wave_hip_b     = 100
#
#   wave amplitude = abs(130-100) + abs(145-118)
#                  = 30 + 27 = 57
# ============================================================

def build_parametric_wave_table(side, hip_a, hip_b, knee_high, knee_low):
    rows = []
    neutral_row(rows)
    if side == "left":
        make_row(rows, hip_a, knee_high, 90, 90, 90, 90, 90, 90)
        make_row(rows, hip_b, knee_low, 90, 90, 90, 90, 90, 90)
        make_row(rows, hip_a, knee_high, 90, 90, 90, 90, 90, 90)
    else:
        make_row(rows, 90, 90, hip_a, knee_high, 90, 90, 90, 90)
        make_row(rows, 90, 90, hip_b, knee_low, 90, 90, 90, 90)
        make_row(rows, 90, 90, hip_a, knee_high, 90, 90, 90, 90)
    neutral_row(rows)
    return rows


def train_optimized_wave_table(side="left", iterations=160):
    import random
    print("\n================================================")
    print(" TRAIN / OPTIMIZE WAVE", side.upper(), "MOTION")
    print("================================================")
    best_score = -999999
    best_params = None
    for i in range(iterations):
        hip_a = random.randint(120, 140)
        hip_b = random.randint(95, 115)
        knee_high = random.randint(130, 150)
        knee_low = random.randint(105, 130)
        amplitude = abs(hip_a - hip_b) + abs(knee_high - knee_low)
        score = amplitude * 2
        score += safe_angle_penalty(hip_a, hip_b, knee_high, knee_low)
        if score > best_score:
            best_score = score
            best_params = (hip_a, hip_b, knee_high, knee_low)
            print("New best wave", side + ":", best_score, best_params)
    return build_parametric_wave_table(side, *best_params)


def generate_trained_wave_left_table():
    return train_optimized_wave_table("left", iterations=160)


def generate_trained_wave_right_table():
    return train_optimized_wave_table("right", iterations=160)


# ============================================================
# TRAIN SHAKE / DANCE / PUSH-UP
# ============================================================
#
# Goal:
#   Shake and dance reward rhythmic left/right alternation.
#   Push-up rewards front-body lowering and raising.
#
# Numerical dance example:
#   hip_a = 120, hip_b = 65
#   rhythm = abs(120-65)=55
#   score = rhythm*2 = 110
# ============================================================

def train_optimized_shake_table(iterations=160):
    import random
    print("\n================================================")
    print(" TRAIN / OPTIMIZE SHAKE MOTION")
    print("================================================")
    best_score = -999999
    best_params = None
    for i in range(iterations):
        hip_a = random.randint(105, 125)
        hip_b = random.randint(55, 80)
        knee = random.randint(85, 100)
        score = abs(hip_a - hip_b) * 2 - abs(knee - 90)
        score += safe_angle_penalty(hip_a, hip_b, knee)
        if score > best_score:
            best_score = score
            best_params = (hip_a, hip_b, knee)
            print("New best shake:", best_score, best_params)
    hip_a, hip_b, knee = best_params
    rows=[]; neutral_row(rows)
    make_row(rows, hip_a, knee, hip_b, knee, hip_a, knee, hip_b, knee)
    make_row(rows, hip_b, knee, hip_a, knee, hip_b, knee, hip_a, knee)
    make_row(rows, hip_a, knee, hip_b, knee, hip_a, knee, hip_b, knee)
    make_row(rows, hip_b, knee, hip_a, knee, hip_b, knee, hip_a, knee)
    neutral_row(rows)
    return rows


def generate_trained_shake_table():
    return train_optimized_shake_table(iterations=160)


def train_optimized_dance_table(iterations=160):
    import random
    print("\n================================================")
    print(" TRAIN / OPTIMIZE DANCE MOTION")
    print("================================================")
    best_score = -999999
    best_params = None
    for i in range(iterations):
        hip_a = random.randint(108, 130)
        hip_b = random.randint(55, 80)
        knee_high = random.randint(120, 145)
        knee_low = random.randint(90, 110)
        score = abs(hip_a - hip_b) * 2 + abs(knee_high - knee_low)
        score += safe_angle_penalty(hip_a, hip_b, knee_high, knee_low)
        if score > best_score:
            best_score = score
            best_params = (hip_a, hip_b, knee_high, knee_low)
            print("New best dance:", best_score, best_params)
    hip_a, hip_b, knee_high, knee_low = best_params
    rows=[]; neutral_row(rows)
    make_row(rows, hip_a, knee_high, hip_b, knee_low, hip_a, knee_high, hip_b, knee_low)
    make_row(rows, hip_b, knee_low, hip_a, knee_high, hip_b, knee_low, hip_a, knee_high)
    make_row(rows, hip_a-10, knee_high, hip_a-10, knee_high, hip_b+10, knee_low, hip_b+10, knee_low)
    make_row(rows, hip_b+10, knee_low, hip_b+10, knee_low, hip_a-10, knee_high, hip_a-10, knee_high)
    neutral_row(rows)
    return rows


def generate_trained_dance_table():
    return train_optimized_dance_table(iterations=160)


def train_optimized_pushup_table(iterations=160):
    import random
    print("\n================================================")
    print(" TRAIN / OPTIMIZE PUSH-UP MOTION")
    print("================================================")
    best_score = -999999
    best_params = None
    for i in range(iterations):
        front_low_knee = random.randint(130, 150)
        front_high_knee = random.randint(105, 125)
        front_hip = random.randint(95, 115)
        score = abs(front_low_knee - front_high_knee) * 3
        score -= abs(front_hip - 105)
        score += safe_angle_penalty(front_low_knee, front_high_knee, front_hip)
        if score > best_score:
            best_score = score
            best_params = (front_low_knee, front_high_knee, front_hip)
            print("New best push-up:", best_score, best_params)
    front_low_knee, front_high_knee, front_hip = best_params
    rows=[]; neutral_row(rows)
    make_row(rows, front_hip, front_high_knee, front_hip, front_high_knee, 90, 90, 90, 90)
    make_row(rows, front_hip+5, front_low_knee, front_hip+5, front_low_knee, 90, 90, 90, 90)
    make_row(rows, front_hip, front_high_knee, front_hip, front_high_knee, 90, 90, 90, 90)
    make_row(rows, front_hip+5, front_low_knee, front_hip+5, front_low_knee, 90, 90, 90, 90)
    neutral_row(rows)
    return rows


def generate_trained_pushup_table():
    return train_optimized_pushup_table(iterations=160)


# ============================================================
# TRAIN BACKFLIP / FRONTFLIP / ROLL LEFT / ROLL RIGHT
# ============================================================
#
# Goal:
#   These are style motions, not real safe acrobatics.
#   The optimizer searches for strong motion contrast while keeping
#   servo angles in a safe range.
#
# Numerical roll-left example:
#   high_knee = 145
#   low_knee  = 78
#   hip_a     = 128
#   hip_b     = 70
#
#   roll amplitude = abs(145-78) + abs(128-70)
#                  = 67 + 58 = 125
# ============================================================

def train_optimized_flip_table(kind="backflip", iterations=160):
    import random
    print("\n================================================")
    print(" TRAIN / OPTIMIZE", kind.upper(), "STYLE MOTION")
    print("================================================")
    best_score = -999999
    best_params = None
    for i in range(iterations):
        crouch = random.randint(125, 150)
        extend = random.randint(55, 85)
        hip1 = random.randint(70, 110)
        hip2 = random.randint(120, 150)
        score = abs(crouch - extend) * 3 + abs(hip2 - hip1)
        score += safe_angle_penalty(crouch, extend, hip1, hip2)
        if score > best_score:
            best_score = score
            best_params = (crouch, extend, hip1, hip2)
            print("New best", kind + ":", best_score, best_params)
    crouch, extend, hip1, hip2 = best_params
    rows=[]; neutral_row(rows)
    if kind == "backflip":
        make_row(rows, hip1, crouch, hip1, crouch, hip1-10, crouch, hip1-10, crouch)
        make_row(rows, hip2, extend, hip2, extend, hip2, extend, hip2, extend)
    else:
        make_row(rows, hip2, extend, hip2, extend, hip1, crouch, hip1, crouch)
        make_row(rows, hip1, crouch, hip1, crouch, hip2, extend, hip2, extend)
    make_row(rows, 90, 120, 90, 120, 90, 120, 90, 120)
    neutral_row(rows)
    return rows


def generate_trained_backflip_table():
    return train_optimized_flip_table("backflip", iterations=160)


def generate_trained_frontflip_table():
    return train_optimized_flip_table("frontflip", iterations=160)


def train_optimized_roll_table(direction="left", iterations=160):
    import random
    print("\n================================================")
    print(" TRAIN / OPTIMIZE ROLL", direction.upper(), "STYLE MOTION")
    print("================================================")
    best_score = -999999
    best_params = None
    for i in range(iterations):
        high_knee = random.randint(130, 150)
        low_knee = random.randint(65, 90)
        hip_a = random.randint(115, 135)
        hip_b = random.randint(60, 80)
        score = abs(high_knee - low_knee) + abs(hip_a - hip_b)
        score += safe_angle_penalty(high_knee, low_knee, hip_a, hip_b)
        if score > best_score:
            best_score = score
            best_params = (high_knee, low_knee, hip_a, hip_b)
            print("New best roll", direction + ":", best_score, best_params)
    high_knee, low_knee, hip_a, hip_b = best_params
    rows=[]; neutral_row(rows)
    if direction == "left":
        make_row(rows, hip_a, high_knee, hip_a, low_knee, hip_a, high_knee, hip_a, low_knee)
        make_row(rows, hip_b, high_knee, hip_a, low_knee, hip_b, high_knee, hip_a, low_knee)
    else:
        make_row(rows, hip_a, low_knee, hip_a, high_knee, hip_a, low_knee, hip_a, high_knee)
        make_row(rows, hip_a, low_knee, hip_b, high_knee, hip_a, low_knee, hip_b, high_knee)
    neutral_row(rows)
    return rows


def generate_trained_roll_left_table():
    return train_optimized_roll_table("left", iterations=160)


def generate_trained_roll_right_table():
    return train_optimized_roll_table("right", iterations=160)


# ============================================================
# TRAIN SLEEP / WAKE UP
# ============================================================
#
# Goal:
#   Sleep lowers body into a resting posture.
#   Wake-up reverses from resting posture to neutral.
#
# Numerical example:
#   sleep_knee = 148, sleep_hip = 72
#   target knee = 150 -> error 2
#   target hip  = 70  -> error 2
#   score = 100 - 2 - 2 = 96
# ============================================================

def train_optimized_sleep_table(iterations=120):
    import random
    print("\n================================================")
    print(" TRAIN / OPTIMIZE SLEEP MOTION")
    print("================================================")
    best_score = -999999
    best_params = None
    for i in range(iterations):
        front_knee = random.randint(130, 150)
        rear_hip = random.randint(60, 80)
        rear_knee = random.randint(135, 150)
        score = 120 - abs(front_knee - 140) - abs(rear_hip - 70) - abs(rear_knee - 150)
        score += safe_angle_penalty(front_knee, rear_hip, rear_knee)
        if score > best_score:
            best_score = score
            best_params = (front_knee, rear_hip, rear_knee)
            print("New best sleep:", best_score, best_params)
    front_knee, rear_hip, rear_knee = best_params
    rows=[]; neutral_row(rows)
    make_row(rows, 90, 120, 90, 120, 80, 135, 80, 135)
    make_row(rows, 90, front_knee, 90, front_knee, rear_hip, rear_knee, rear_hip, rear_knee)
    return rows


def generate_trained_sleep_table():
    return train_optimized_sleep_table(iterations=120)


def generate_trained_wake_up_table():
    rows = train_optimized_sleep_table(iterations=80)
    rows.reverse()
    neutral_row(rows)
    return rows


# ============================================================
# TRAIN ALL MOTIONS
# ============================================================
#
# Goal:
#   Run every trainer and store every optimized motion.
#
# Numerical example:
#   If walk score = 278 and jump score = 200, both are accepted
#   independently because each motion has its own scoring goal.
#
# Important:
#   This may take longer because it calls all training functions.
# ============================================================

def train_all_motion_tables():
    trained = {}

    training_plan = [
        ("trained_walk", "Trained Walk", generate_trained_walk_table, 3),
        ("trained_run", "Trained Run / Trot", generate_trained_run_table, 3),
        ("trained_sit", "Trained Sit", generate_trained_sit_table, 1),
        ("trained_stand", "Trained Stand", generate_trained_stand_table, 1),
        ("trained_jump", "Trained Jump", generate_trained_jump_table, 1),
        ("trained_turn_left", "Trained Turn Left", generate_trained_turn_left_table, 3),
        ("trained_turn_right", "Trained Turn Right", generate_trained_turn_right_table, 3),
        ("trained_walk_backward", "Trained Walk Backward", generate_trained_walk_backward_table, 3),
        ("trained_crawl", "Trained Crawl", generate_trained_crawl_table, 3),
        ("trained_bow", "Trained Bow", generate_trained_bow_table, 1),
        ("trained_stretch", "Trained Stretch", generate_trained_stretch_table, 1),
        ("trained_side_left", "Trained Side Step Left", generate_trained_side_left_table, 2),
        ("trained_side_right", "Trained Side Step Right", generate_trained_side_right_table, 2),
        ("trained_wave_left", "Trained Wave Left Paw", generate_trained_wave_left_table, 2),
        ("trained_wave_right", "Trained Wave Right Paw", generate_trained_wave_right_table, 2),
        ("trained_shake", "Trained Shake / Wiggle", generate_trained_shake_table, 2),
        ("trained_dance", "Trained Dance", generate_trained_dance_table, 2),
        ("trained_pushup", "Trained Push-Up", generate_trained_pushup_table, 2),
        ("trained_backflip", "Trained Backflip Style", generate_trained_backflip_table, 1),
        ("trained_frontflip", "Trained Frontflip Style", generate_trained_frontflip_table, 1),
        ("trained_roll_left", "Trained Roll Left Style", generate_trained_roll_left_table, 1),
        ("trained_roll_right", "Trained Roll Right Style", generate_trained_roll_right_table, 1),
        ("trained_sleep", "Trained Sleep / Rest", generate_trained_sleep_table, 1),
        ("trained_wake_up", "Trained Wake Up", generate_trained_wake_up_table, 1),
    ]

    print("\n================================================")
    print(" TRAINING ALL MOTIONS")
    print("================================================")

    for name, label, generator, repeat_count in training_plan:
        rows = generator()
        trained[name] = {
            "label": label,
            "rows": rows,
            "repeat_on_esp32": repeat_count,
        }

    print("\n================================================")
    print(" ALL MOTIONS TRAINED")
    print("================================================")
    print("Total trained motions:", len(trained))
    print("================================================")

    return trained


def generate_trained_all_table_placeholder():
    """
    Placeholder only.
    The main loop handles option 'z' specially because train-all returns
    many motion tables, not one motion table.
    """
    return []


def generate_train_selected_motion_placeholder():
    """
    Placeholder only.
    The main loop handles option 'u' specially because it must first ask
    the user which motion should be trained.
    """
    return []


# ============================================================
# TRAINABLE MOTION SELECTION REGISTRY
# ============================================================
#
# This registry is used by option 'u'.
#
# New behavior:
#   1. User presses u
#   2. Script asks which motion to train
#   3. User enters 1, 2, 3, ..., a, b, c, etc.
#   4. Script calls the matching training generator
#
# Example:
#   User presses: u
#   Prompt asks : Which motion do you want to train?
#   User enters : 1
#   Result      : train walking motion
#
# Another example:
#   User presses: u
#   User enters : 5
#   Result      : train jump motion
#
# This is different from the older behavior where option 'u' always
# trained run/trot only.
# ============================================================

TRAINABLE_MOTIONS = {
    "1": {"name": "trained_walk", "label": "Trained / Optimized Walk", "generator": generate_trained_walk_table, "repeat_on_esp32": 3},
    "2": {"name": "trained_run", "label": "Trained / Optimized Run / Trot", "generator": generate_trained_run_table, "repeat_on_esp32": 3},
    "3": {"name": "trained_sit", "label": "Trained / Optimized Sit", "generator": generate_trained_sit_table, "repeat_on_esp32": 1},
    "4": {"name": "trained_stand", "label": "Trained / Optimized Stand", "generator": generate_trained_stand_table, "repeat_on_esp32": 1},
    "5": {"name": "trained_jump", "label": "Trained / Optimized Jump", "generator": generate_trained_jump_table, "repeat_on_esp32": 1},
    "6": {"name": "trained_turn_left", "label": "Trained / Optimized Turn Left", "generator": generate_trained_turn_left_table, "repeat_on_esp32": 3},
    "7": {"name": "trained_turn_right", "label": "Trained / Optimized Turn Right", "generator": generate_trained_turn_right_table, "repeat_on_esp32": 3},
    "8": {"name": "trained_walk_backward", "label": "Trained / Optimized Walk Backward", "generator": generate_trained_walk_backward_table, "repeat_on_esp32": 3},
    "9": {"name": "trained_crawl", "label": "Trained / Optimized Crawl", "generator": generate_trained_crawl_table, "repeat_on_esp32": 3},
    "a": {"name": "trained_bow", "label": "Trained / Optimized Bow", "generator": generate_trained_bow_table, "repeat_on_esp32": 1},
    "b": {"name": "trained_stretch", "label": "Trained / Optimized Stretch", "generator": generate_trained_stretch_table, "repeat_on_esp32": 1},
    "c": {"name": "trained_side_left", "label": "Trained / Optimized Side Step Left", "generator": generate_trained_side_left_table, "repeat_on_esp32": 2},
    "d": {"name": "trained_side_right", "label": "Trained / Optimized Side Step Right", "generator": generate_trained_side_right_table, "repeat_on_esp32": 2},
    "e": {"name": "trained_wave_left", "label": "Trained / Optimized Wave Left Paw", "generator": generate_trained_wave_left_table, "repeat_on_esp32": 2},
    "f": {"name": "trained_wave_right", "label": "Trained / Optimized Wave Right Paw", "generator": generate_trained_wave_right_table, "repeat_on_esp32": 2},
    "g": {"name": "trained_shake", "label": "Trained / Optimized Shake / Wiggle", "generator": generate_trained_shake_table, "repeat_on_esp32": 2},
    "h": {"name": "trained_dance", "label": "Trained / Optimized Dance", "generator": generate_trained_dance_table, "repeat_on_esp32": 2},
    "i": {"name": "trained_pushup", "label": "Trained / Optimized Push-Up", "generator": generate_trained_pushup_table, "repeat_on_esp32": 2},
    "j": {"name": "trained_backflip", "label": "Trained / Optimized Backflip Style", "generator": generate_trained_backflip_table, "repeat_on_esp32": 1},
    "k": {"name": "trained_frontflip", "label": "Trained / Optimized Frontflip Style", "generator": generate_trained_frontflip_table, "repeat_on_esp32": 1},
    "l": {"name": "trained_roll_left", "label": "Trained / Optimized Roll Left Style", "generator": generate_trained_roll_left_table, "repeat_on_esp32": 1},
    "m": {"name": "trained_roll_right", "label": "Trained / Optimized Roll Right Style", "generator": generate_trained_roll_right_table, "repeat_on_esp32": 1},
    "n": {"name": "trained_sleep", "label": "Trained / Optimized Sleep / Rest", "generator": generate_trained_sleep_table, "repeat_on_esp32": 1},
    "o": {"name": "trained_wake_up", "label": "Trained / Optimized Wake Up", "generator": generate_trained_wake_up_table, "repeat_on_esp32": 1},
}


def print_trainable_motion_menu():
    print("\n================================================")
    print(" TRAIN ONE SELECTED MOTION")
    print("================================================")
    print("Enter the motion you want to train:")
    print("")
    print("1 - Train Walk")
    print("2 - Train Trot / Run")
    print("3 - Train Sit")
    print("4 - Train Stand")
    print("5 - Train Jump")
    print("6 - Train Turn Left")
    print("7 - Train Turn Right")
    print("8 - Train Walk Backward")
    print("9 - Train Crawl")
    print("a - Train Bow")
    print("b - Train Stretch")
    print("c - Train Side Step Left")
    print("d - Train Side Step Right")
    print("e - Train Wave Left Paw")
    print("f - Train Wave Right Paw")
    print("g - Train Shake / Wiggle")
    print("h - Train Dance")
    print("i - Train Push-Up")
    print("j - Train Backflip Style")
    print("k - Train Frontflip Style")
    print("l - Train Roll Left Style")
    print("m - Train Roll Right Style")
    print("n - Train Sleep / Rest")
    print("o - Train Wake Up")
    print("0 - Cancel training")
    print("================================================")


def choose_motion_to_train():
    print_trainable_motion_menu()

    train_choice = get_single_key("Motion to train: ").lower()

    if train_choice == "0":
        print("\nTraining cancelled.")
        return None

    if train_choice not in TRAINABLE_MOTIONS:
        print("\nInvalid training selection.")
        return None

    return TRAINABLE_MOTIONS[train_choice]


# ============================================================
# MOTION REGISTRY
# ============================================================
#
# UPDATED BEHAVIOR:
# ------------------------------------------------------------
# Every motion menu option now calls its TRAINED / OPTIMIZED
# generator directly.
#
# That means:
#
#   1 -> trains/generates optimized walk
#   2 -> trains/generates optimized run/trot
#   5 -> trains/generates optimized jump
#   l -> trains/generates optimized roll left
#
# There are no separate t/u/z training commands anymore.
# The generated ESP32/OpenCat C++ code will contain the trained
# motion table for each motion the user selected.
# ============================================================

# ============================================================
# MOTION REGISTRY - STATIC AND TRAINED MODES
# ============================================================
#
# UPDATED BEHAVIOR:
# ------------------------------------------------------------
# At program startup, the user chooses how normal motions are built:
#
#   1 - Static motion configuration
#       Uses the fixed hand-written motion tables such as:
#           generate_walk_table()
#           generate_jump_table()
#           generate_turn_left_table()
#
#       No random-search optimization/training is executed.
#       This is the fastest and most deterministic mode.
#
#   2 - Trained / optimized motion configuration
#       Uses the training/optimization functions such as:
#           generate_trained_walk_table()
#           generate_trained_jump_table()
#           generate_trained_turn_left_table()
#
#       This mode searches for improved angles before displaying,
#       simulating, and exporting the selected motion.
#
# The menu keys stay the same in both modes:
#
#   1 = Walk
#   2 = Trot / Run
#   3 = Sit
#   ...
#   o = Wake Up
#
# Recovery motions are always available separately through option r.
# ============================================================

STATIC_MOTIONS = {
    "1": {"name": "walk", "label": "Static Walk", "generator": generate_walk_table, "repeat_on_esp32": 3},
    "2": {"name": "run", "label": "Static Trot / Run", "generator": generate_trot_run_table, "repeat_on_esp32": 3},
    "3": {"name": "sit", "label": "Static Sit", "generator": generate_sit_table, "repeat_on_esp32": 1},
    "4": {"name": "stand", "label": "Static Stand", "generator": generate_stand_table, "repeat_on_esp32": 1},
    "5": {"name": "jump", "label": "Static Jump", "generator": generate_jump_table, "repeat_on_esp32": 1},
    "6": {"name": "turn_left", "label": "Static Turn Left", "generator": generate_turn_left_table, "repeat_on_esp32": 3},
    "7": {"name": "turn_right", "label": "Static Turn Right", "generator": generate_turn_right_table, "repeat_on_esp32": 3},
    "8": {"name": "walk_backward", "label": "Static Walk Backward", "generator": generate_walk_backward_table, "repeat_on_esp32": 3},
    "9": {"name": "crawl", "label": "Static Crawl", "generator": generate_crawl_table, "repeat_on_esp32": 3},
    "a": {"name": "bow", "label": "Static Bow", "generator": generate_bow_table, "repeat_on_esp32": 1},
    "b": {"name": "stretch", "label": "Static Stretch", "generator": generate_stretch_table, "repeat_on_esp32": 1},
    "c": {"name": "side_left", "label": "Static Side Step Left", "generator": generate_side_step_left_table, "repeat_on_esp32": 2},
    "d": {"name": "side_right", "label": "Static Side Step Right", "generator": generate_side_step_right_table, "repeat_on_esp32": 2},
    "e": {"name": "wave_left", "label": "Static Wave Left Paw", "generator": generate_wave_left_table, "repeat_on_esp32": 2},
    "f": {"name": "wave_right", "label": "Static Wave Right Paw", "generator": generate_wave_right_table, "repeat_on_esp32": 2},
    "g": {"name": "shake", "label": "Static Shake / Wiggle", "generator": generate_shake_table, "repeat_on_esp32": 2},
    "h": {"name": "dance", "label": "Static Dance", "generator": generate_dance_table, "repeat_on_esp32": 2},
    "i": {"name": "pushup", "label": "Static Push-Up", "generator": generate_pushup_table, "repeat_on_esp32": 2},
    "j": {"name": "backflip", "label": "Static Backflip Style", "generator": generate_backflip_table, "repeat_on_esp32": 1},
    "k": {"name": "frontflip", "label": "Static Frontflip Style", "generator": generate_frontflip_table, "repeat_on_esp32": 1},
    "l": {"name": "roll_left", "label": "Static Roll Left Style", "generator": generate_roll_left_table, "repeat_on_esp32": 1},
    "m": {"name": "roll_right", "label": "Static Roll Right Style", "generator": generate_roll_right_table, "repeat_on_esp32": 1},
    "n": {"name": "sleep", "label": "Static Sleep / Rest", "generator": generate_sleep_table, "repeat_on_esp32": 1},
    "o": {"name": "wake_up", "label": "Static Wake Up", "generator": generate_wake_up_table, "repeat_on_esp32": 1},
}

TRAINED_MOTIONS = {
    "1": {"name": "trained_walk", "label": "Trained / Optimized Walk", "generator": generate_trained_walk_table, "repeat_on_esp32": 3},
    "2": {"name": "trained_run", "label": "Trained / Optimized Trot / Run", "generator": generate_trained_run_table, "repeat_on_esp32": 3},
    "3": {"name": "trained_sit", "label": "Trained / Optimized Sit", "generator": generate_trained_sit_table, "repeat_on_esp32": 1},
    "4": {"name": "trained_stand", "label": "Trained / Optimized Stand", "generator": generate_trained_stand_table, "repeat_on_esp32": 1},
    "5": {"name": "trained_jump", "label": "Trained / Optimized Jump", "generator": generate_trained_jump_table, "repeat_on_esp32": 1},
    "6": {"name": "trained_turn_left", "label": "Trained / Optimized Turn Left", "generator": generate_trained_turn_left_table, "repeat_on_esp32": 3},
    "7": {"name": "trained_turn_right", "label": "Trained / Optimized Turn Right", "generator": generate_trained_turn_right_table, "repeat_on_esp32": 3},
    "8": {"name": "trained_walk_backward", "label": "Trained / Optimized Walk Backward", "generator": generate_trained_walk_backward_table, "repeat_on_esp32": 3},
    "9": {"name": "trained_crawl", "label": "Trained / Optimized Crawl", "generator": generate_trained_crawl_table, "repeat_on_esp32": 3},
    "a": {"name": "trained_bow", "label": "Trained / Optimized Bow", "generator": generate_trained_bow_table, "repeat_on_esp32": 1},
    "b": {"name": "trained_stretch", "label": "Trained / Optimized Stretch", "generator": generate_trained_stretch_table, "repeat_on_esp32": 1},
    "c": {"name": "trained_side_left", "label": "Trained / Optimized Side Step Left", "generator": generate_trained_side_left_table, "repeat_on_esp32": 2},
    "d": {"name": "trained_side_right", "label": "Trained / Optimized Side Step Right", "generator": generate_trained_side_right_table, "repeat_on_esp32": 2},
    "e": {"name": "trained_wave_left", "label": "Trained / Optimized Wave Left Paw", "generator": generate_trained_wave_left_table, "repeat_on_esp32": 2},
    "f": {"name": "trained_wave_right", "label": "Trained / Optimized Wave Right Paw", "generator": generate_trained_wave_right_table, "repeat_on_esp32": 2},
    "g": {"name": "trained_shake", "label": "Trained / Optimized Shake / Wiggle", "generator": generate_trained_shake_table, "repeat_on_esp32": 2},
    "h": {"name": "trained_dance", "label": "Trained / Optimized Dance", "generator": generate_trained_dance_table, "repeat_on_esp32": 2},
    "i": {"name": "trained_pushup", "label": "Trained / Optimized Push-Up", "generator": generate_trained_pushup_table, "repeat_on_esp32": 2},
    "j": {"name": "trained_backflip", "label": "Trained / Optimized Backflip Style", "generator": generate_trained_backflip_table, "repeat_on_esp32": 1},
    "k": {"name": "trained_frontflip", "label": "Trained / Optimized Frontflip Style", "generator": generate_trained_frontflip_table, "repeat_on_esp32": 1},
    "l": {"name": "trained_roll_left", "label": "Trained / Optimized Roll Left Style", "generator": generate_trained_roll_left_table, "repeat_on_esp32": 1},
    "m": {"name": "trained_roll_right", "label": "Trained / Optimized Roll Right Style", "generator": generate_trained_roll_right_table, "repeat_on_esp32": 1},
    "n": {"name": "trained_sleep", "label": "Trained / Optimized Sleep / Rest", "generator": generate_trained_sleep_table, "repeat_on_esp32": 1},
    "o": {"name": "trained_wake_up", "label": "Trained / Optimized Wake Up", "generator": generate_trained_wake_up_table, "repeat_on_esp32": 1},
}

# Default mode is static because it is deterministic and does not train.
MOTION_GENERATION_MODE = "static"
MOTIONS = STATIC_MOTIONS


def choose_motion_generation_mode():
    """
    Ask the user whether to use static motion tables or trained motions.

    Static mode:
        Uses fixed motion tables and does NOT train.

    Trained mode:
        Uses optimization/training functions before simulation/export.
    """

    global MOTION_GENERATION_MODE
    global MOTIONS

    print("\n================================================")
    print(" SELECT MOTION GENERATION MODE")
    print("================================================")
    print("1 - Static configuration")
    print("    Use fixed motion tables without training")
    print("")
    print("2 - Trained / optimized configuration")
    print("    Train/optimize each selected motion before export")
    print("================================================")

    choice = get_single_key("Select mode [default 1 = static]: ").lower()

    if choice in ("", "1", "s"):
        MOTION_GENERATION_MODE = "static"
        MOTIONS = STATIC_MOTIONS
        print("\nSelected mode: STATIC configuration - no training will run.")
        return

    if choice in ("2", "t"):
        MOTION_GENERATION_MODE = "trained"
        MOTIONS = TRAINED_MOTIONS
        print("\nSelected mode: TRAINED / OPTIMIZED configuration.")
        return

    MOTION_GENERATION_MODE = "static"
    MOTIONS = STATIC_MOTIONS
    print("\nInvalid mode. Defaulting to STATIC configuration - no training will run.")


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


def simulate_fall_pose(fall_type):
    """
    Shows a visual fallen posture in the PyBullet GUI.

    This is a failure-scenario preview. It does not depend on the
    robot actually falling during a previous motion. The user can
    intentionally test each fall state and then choose recovery.
    """

    setup_pybullet_scene()

    pose_map = {
        "left_side_fall": (1.15, 0.0, 0.0),
        "right_side_fall": (-1.15, 0.0, 0.0),
        "front_fall": (0.0, 1.15, 0.0),
        "back_fall": (0.0, -1.15, 0.0),
        "upside_down_fall": (math.pi, 0.0, 0.0),
    }

    roll, pitch, yaw = pose_map.get(fall_type, (0.0, 0.0, 0.0))
    quat = p.getQuaternionFromEuler([roll, pitch, yaw])

    p.resetBasePositionAndOrientation(DOG_BODY_ID, [0, 0, 0.18], quat)
    p.resetBasePositionAndOrientation(DOG_HEAD_ID, [0.50, 0, 0.20], quat)
    p.resetBasePositionAndOrientation(DOG_NOSE_ID, [0.63, 0, 0.20], quat)

    p.addUserDebugText(
        "FAIL SCENARIO: " + fall_type,
        [0, 0, 0.85],
        textColorRGB=[1, 0.3, 0.3],
        textSize=1.4
    )

    print("\n================================================")
    print(" FALL SCENARIO")
    print("================================================")
    print("fall_type:", fall_type)
    print("roll     :", round(roll, 3))
    print("pitch    :", round(pitch, 3))
    print("yaw      :", round(yaw, 3))
    print("classified as:", classify_fall_from_roll_pitch(roll, pitch) if fall_type != "upside_down_fall" else "upside_down_fall")
    print("================================================")

    for _ in range(80):
        p.stepSimulation()
        time.sleep(0.02)


def handle_fall_scenario_and_recovery(selected_motions):
    """
    Interactive fall scenario handler.

    User chooses a fall scenario. The script displays it and asks
    whether to execute the matching recovery motion. If recovery is
    accepted, the recovery motion is stored and can be exported to C++.
    """

    print("\n================================================")
    print(" FALL SCENARIO + RECOVERY")
    print("================================================")
    print("1 - Left-side fall")
    print("2 - Right-side fall")
    print("3 - Front fall")
    print("4 - Back fall")
    print("5 - Upside-down fall")
    print("0 - Cancel")
    print("================================================")

    choice = get_single_key("Select fall scenario: ").lower()

    scenario_map = {
        "1": "left_side_fall",
        "2": "right_side_fall",
        "3": "front_fall",
        "4": "back_fall",
        "5": "upside_down_fall",
    }

    if choice == "0":
        print("\nFall scenario cancelled.")
        return

    if choice not in scenario_map:
        print("\nInvalid fall scenario.")
        return

    fall_type = scenario_map[choice]
    simulate_fall_pose(fall_type)

    answer = get_single_key("Run matching recovery now? (y/n): ").lower()
    if answer != "y":
        print("\nRecovery skipped. No recovery motion was stored.")
        return

    recovery_config = RECOVERY_MOTIONS[fall_type]
    recovery_rows = recovery_config["generator"]()

    selected_motions[recovery_config["name"]] = {
        "label": recovery_config["label"],
        "rows": recovery_rows,
        "repeat_on_esp32": recovery_config["repeat_on_esp32"],
    }

    print("\nRecovery motion stored:", recovery_config["label"])
    display_angle_table_and_simulate(recovery_rows, recovery_config["label"])

# ============================================================
# C++ CODE GENERATION
# ============================================================

def cpp_array_name(motion_name):
    return "MOTION_" + motion_name.upper()


def generate_cpp_file(selected_motions):
    """
    Generates ESP32/OpenCat C++ code.

    IMPORTANT UPDATED RECOVERY BEHAVIOR:
    ------------------------------------------------------------
    The generated ESP32 file always includes manual recovery commands:

        recover_left
        recover_right
        recover_front
        recover_back
        recover_upside_down

    This supports the manual test procedure:

        1. Upload the generated C++ code to ESP32.
        2. Start a normal motion such as walk/run.
        3. Manually push the dog so it falls left/right/front/back.
        4. Send the matching command, for example: recover_left.
        5. ESP32 executes the recovery angle table.

    NOTE:
    This is manual recovery triggering. The ESP32 still does not
    automatically detect falling unless IMU logic is added later.
    """

    # ------------------------------------------------------------
    # Always include recovery motions in the generated ESP32 file.
    # ------------------------------------------------------------
    # Why?
    #   You want to manually push the dog left/right/front/back and
    #   then send a command such as:
    #
    #       recover_left
    #
    #   If recovery motions are not automatically inserted here, the
    #   generated C++ file may not contain MOTION_RECOVER_LEFT unless
    #   the recovery option was selected earlier in the Python menu.
    #
    # This block guarantees that the recovery commands are always
    # available on ESP32 after pressing q.
    # ------------------------------------------------------------
    selected_motions = dict(selected_motions)

    for recovery_info in RECOVERY_MOTIONS.values():
        recovery_name = recovery_info["name"]

        if recovery_name not in selected_motions:
            selected_motions[recovery_name] = {
                "label": recovery_info["label"],
                "rows": recovery_info["generator"](),
                "repeat_on_esp32": recovery_info["repeat_on_esp32"],
            }

    lines = []

    lines.append("// ============================================================")
    lines.append("// AUTO-GENERATED ESP32 / OPENCAT DOG MOTION TABLE PLAYER")
    lines.append("// ============================================================")
    lines.append("//")
    lines.append("// Generated by dog_motion_designer_generate_cpp.py")
    lines.append("//")
    lines.append("// This file contains selected trained motions PLUS built-in manual recovery motions.")
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
    lines.append("// ============================================================")
    lines.append("// OpenCat / Bittle walking servo index mapping")
    lines.append("// ------------------------------------------------------------")
    lines.append("// IMPORTANT:")
    lines.append("//   row[0]..row[7] are motion-table ANGLES.")
    lines.append("//   FL_HIP, FR_HIP, etc. are REAL OpenCat servo indices.")
    lines.append("//")
    lines.append("// OpenCat BiBoard V0 order:")
    lines.append("//   servo 0..3  = head / shoulder-roll group")
    lines.append("//   servo 4..7  = hip / shoulder-pitch group")
    lines.append("//   servo 8..11 = knee group")
    lines.append("//")
    lines.append("// Leg order inside OpenCat groups:")
    lines.append("//   LF, RF, RB, LB")
    lines.append("//")
    lines.append("// This script table order is:")
    lines.append("//   FL_HIP, FL_KNEE, FR_HIP, FR_KNEE,")
    lines.append("//   RL_HIP, RL_KNEE, RR_HIP, RR_KNEE")
    lines.append("// ============================================================")
    lines.append("#define FL_HIP   4")
    lines.append("#define FR_HIP   5")
    lines.append("#define RR_HIP   6")
    lines.append("#define RL_HIP   7")
    lines.append("")
    lines.append("#define FL_KNEE  8")
    lines.append("#define FR_KNEE  9")
    lines.append("#define RR_KNEE  10")
    lines.append("#define RL_KNEE  11")
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

    if MOTION_GENERATION_MODE == "static":
        print("Mode: STATIC configuration - fixed motion tables, no training")
        print("Select a motion to simulate and add to output:")
    else:
        print("Mode: TRAINED / OPTIMIZED configuration")
        print("Select a motion to train, simulate, and add to output:")

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
    print("r - Simulate Fall Scenario + Optional Recovery")
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

def display_angle_table_and_simulate(rows, motion_label):
    """
    Prints the motion angle table and previews the motion in PyBullet.

    This helper keeps the same original display behavior.
    In static mode, the fixed table is displayed immediately.
    In trained mode, the selected motion trains before display.
    """

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


def main():
    choose_motion_generation_mode()

    selected_motions = {}

    while True:
        print_menu(selected_motions)

        choice = get_single_key("Select option: ").lower()

        if choice == "q":
            generate_cpp_file(selected_motions)
            print("\nExiting.")
            break

        if choice == "r":
            handle_fall_scenario_and_recovery(selected_motions)
            continue

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

        display_angle_table_and_simulate(rows, motion_label)


if __name__ == "__main__":
    main()
