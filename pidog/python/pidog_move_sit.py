import time
import smbus

# -------------------------------------------
# 📌 PCA9685 Setup (Servo Controller over I2C)
# -------------------------------------------
I2C_ADDR = 0x40       # Default I2C address of PCA9685
bus = smbus.SMBus(1)  # I2C bus 1 on Raspberry Pi

# PCA9685 Register Addresses
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06

PWM_FREQ = 50  # 50 Hz PWM frequency (for most hobby servos)
TICK_COUNT = 4096
SERVO_MIN = 205  # ≈1 ms pulse → 0°
SERVO_MAX = 410  # ≈2 ms pulse → 180°

# -------------------------------------------
# 🦿 Servo Channel Mapping
# -------------------------------------------
JOINT_CHANNELS = {
    "FL_HIP":  0,  # Front Left Hip
    "FL_KNEE": 1,  # Front Left Knee
    "FR_HIP":  2,  # Front Right Hip
    "FR_KNEE": 3,  # Front Right Knee
    "RL_HIP":  4,  # Rear Left Hip
    "RL_KNEE": 5,  # Rear Left Knee
    "RR_HIP":  6,  # Rear Right Hip
    "RR_KNEE": 7,  # Rear Right Knee
}

"""
╔════════╤═════════════════╤════════════════════════════════════════════╗
║ Angle  │ Used For        │ Why                                       ║
╟────────┼─────────────────┼────────────────────────────────────────────╢
║ 90°    │ HIP (neutral)   │ Leg is centered vertically under body     ║
║ 60°    │ KNEE (standing) │ Slight bend for stability and realism     ║
║ 120°   │ KNEE (sitting)  │ Deep bend to tuck legs in while sitting   ║
║ 110°   │ HIP (walking)   │ Leg swings forward to step                ║
╚════════╧═════════════════╧════════════════════════════════════════════╝
"""

# -------------------------------------------
# 🔢 Initial Joint Angles Matrix (Standing)
# -------------------------------------------
initial_angles = {
    "FL_HIP":  90, "FL_KNEE": 60,
    "FR_HIP":  90, "FR_KNEE": 60,
    "RL_HIP":  90, "RL_KNEE": 60,
    "RR_HIP":  90, "RR_KNEE": 60,
}

# -------------------------------------------
# ⚙️ PCA9685 Setup
# -------------------------------------------
def init_pca9685():
    bus.write_byte_data(I2C_ADDR, MODE1, 0x00)  # Wake from sleep
    set_pwm_freq(PWM_FREQ)

def set_pwm_freq(freq):
    prescale = int(25000000.0 / (4096 * freq) - 1)
    old_mode = bus.read_byte_data(I2C_ADDR, MODE1)
    bus.write_byte_data(I2C_ADDR, MODE1, old_mode | 0x10)  # Sleep
    bus.write_byte_data(I2C_ADDR, PRESCALE, prescale)
    bus.write_byte_data(I2C_ADDR, MODE1, old_mode)  # Wake
    time.sleep(0.005)
    bus.write_byte_data(I2C_ADDR, MODE1, old_mode | 0xA1)  # Restart + Auto-Increment

# -------------------------------------------
# 🎯 Servo Control
# -------------------------------------------
def angle_to_pwm(angle):
    angle = max(0, min(180, angle))  # Clamp angle within safe range
    return int(SERVO_MIN + (angle / 180.0) * (SERVO_MAX - SERVO_MIN))

def set_pwm(channel, on, off):
    reg = LED0_ON_L + 4 * channel
    bus.write_byte_data(I2C_ADDR, reg, on & 0xFF)
    bus.write_byte_data(I2C_ADDR, reg+1, on >> 8)
    bus.write_byte_data(I2C_ADDR, reg+2, off & 0xFF)
    bus.write_byte_data(I2C_ADDR, reg+3, off >> 8)

def set_servo_angle(joint, angle):
    channel = JOINT_CHANNELS[joint]
    pwm_val = angle_to_pwm(angle)
    set_pwm(channel, 0, pwm_val)

# -------------------------------------------
# 📏 Posture Routines
# -------------------------------------------
def stand():
    """
    Set each joint to its defined 'standing' position using the initial_angles matrix.
    """
    for joint, angle in initial_angles.items():
        set_servo_angle(joint, angle)

def sit():
    """
    Deeper bend in knees to simulate sitting posture.
    """
    for joint in JOINT_CHANNELS:
        if "HIP" in joint:
            set_servo_angle(joint, 90)
        elif "KNEE" in joint:
            set_servo_angle(joint, 120)

# -------------------------------------------
# 🚶 Walking (Trot Gait)
# -------------------------------------------
def trot_forward(steps=3, delay=0.3):
    """
    Simple trot gait: moves diagonal pairs (FL+RR, then FR+RL).
    """
    for i in range(steps):
        print(f"Step {i+1} — Trot gait")
        set_servo_angle("FL_HIP", 110)
        set_servo_angle("RR_HIP", 110)
        time.sleep(delay)
        set_servo_angle("FL_HIP", 90)
        set_servo_angle("RR_HIP", 90)

        set_servo_angle("FR_HIP", 110)
        set_servo_angle("RL_HIP", 110)
        time.sleep(delay)
        set_servo_angle("FR_HIP", 90)
        set_servo_angle("RL_HIP", 90)

# -------------------------------------------
# ↩️ Turning
# -------------------------------------------
def turn_left(steps=2, delay=0.3):
    for i in range(steps):
        print(f"Turning left — step {i+1}")
        set_servo_angle("FR_HIP", 110)
        set_servo_angle("RR_HIP", 110)
        time.sleep(delay)
        set_servo_angle("FR_HIP", 90)
        set_servo_angle("RR_HIP", 90)

def turn_right(steps=2, delay=0.3):
    for i in range(steps):
        print(f"Turning right — step {i+1}")
        set_servo_angle("FL_HIP", 110)
        set_servo_angle("RL_HIP", 110)
        time.sleep(delay)
        set_servo_angle("FL_HIP", 90)
        set_servo_angle("RL_HIP", 90)

# -------------------------------------------
# 🚀 Main Execution
# -------------------------------------------
def main():
    print("🔧 Init...")
    init_pca9685()

    print("🦴 Stand")
    stand()
    time.sleep(1)

    print("🚶 Walk Forward")
    trot_forward(steps=3)

    print("↩️ Turn Left")
    turn_left(steps=2)

    print("↪️ Turn Right")
    turn_right(steps=2)

    print("🪑 Sit")
    sit()
    print("✅ Done")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("❌ Interrupted")
