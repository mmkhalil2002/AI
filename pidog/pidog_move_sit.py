import time
import smbus  # Used for I2C communication with PCA9685

# -----------------------------
# PCA9685 I2C Setup
# -----------------------------
I2C_ADDR = 0x40       # Default I2C address of PCA9685
bus = smbus.SMBus(1)  # Use I2C bus 1 on Raspberry Pi

# PCA9685 register addresses
MODE1     = 0x00
PRESCALE  = 0xFE
LED0_ON_L = 0x06

# PWM timing configuration
PWM_FREQ         = 50       # 50Hz PWM frequency for hobby servos
TICK_COUNT       = 4096     # 12-bit PWM resolution
TICK_DURATION_US = 20000 / TICK_COUNT  # Duration of each tick (~4.88 µs)

# Pulse length ranges for servos
SERVO_MIN = 205   # 0° → ~1ms pulse
SERVO_MAX = 410   # 180° → ~2ms pulse

# -----------------------------
# Servo Joint-to-Channel Mapping
# -----------------------------
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
Angle Reference Table:

╔═══════╤═════════════════╤════════════════════════════════════════════╗
║ Angle    │ Used For        │ Why                                       ║
╟───────┼─────────────────┼────────────────────────────────────────────╢
║ 90°   │ HIP (neutral)   │ Leg under the body (natural standing base) ║
║       │                 │ Default center position for servos         ║
╟───────┼─────────────────┼────────────────────────────────────────────╢
║ 60°   │ KNEE (standing) │ Slight bend for upright pose               ║
╟───────┼─────────────────┼────────────────────────────────────────────╢
║ 120°  │ KNEE (sitting)  │ Deep bend to simulate crouch/sit           ║
╟───────┼─────────────────┼────────────────────────────────────────────╢
║ 110°  │ HIP (walking)   │ Swing leg forward for step                 ║
╚═══════╧═════════════════╧════════════════════════════════════════════╝
"""

# -----------------------------
# PCA9685 Initialization
# -----------------------------
def init_pca9685():
    """ Initialize the PCA9685 and set PWM frequency """
    bus.write_byte_data(I2C_ADDR, MODE1, 0x00)  # Wake up PCA9685
    set_pwm_freq(PWM_FREQ)                     # Set PWM frequency

def set_pwm_freq(freq):
    """ Set PWM frequency by configuring the prescaler """
    prescale_val = int(25000000.0 / (4096 * freq) - 1)
    old_mode = bus.read_byte_data(I2C_ADDR, MODE1)
    new_mode = (old_mode & 0x7F) | 0x10  # Sleep
    bus.write_byte_data(I2C_ADDR, MODE1, new_mode)
    bus.write_byte_data(I2C_ADDR, PRESCALE, prescale_val)
    bus.write_byte_data(I2C_ADDR, MODE1, old_mode)
    time.sleep(0.005)
    bus.write_byte_data(I2C_ADDR, MODE1, old_mode | 0xA1)  # Restart + auto-increment

# -----------------------------
# Servo Control Utilities
# -----------------------------
def angle_to_pwm(angle):
    """ Convert angle in degrees (0°-180°) to PCA9685 PWM ticks """
    angle = max(0, min(180, angle))  # Clamp to valid range
    return int(SERVO_MIN + (angle / 180.0) * (SERVO_MAX - SERVO_MIN))

def set_pwm(channel, on, off):
    """ Send PWM values to a PCA9685 channel """
    reg = LED0_ON_L + 4 * channel
    bus.write_byte_data(I2C_ADDR, reg,     on & 0xFF)
    bus.write_byte_data(I2C_ADDR, reg + 1, on >> 8)
    bus.write_byte_data(I2C_ADDR, reg + 2, off & 0xFF)
    bus.write_byte_data(I2C_ADDR, reg + 3, off >> 8)

def set_servo_angle(joint_name, angle):
    """ Move a servo to a specific angle by joint name """
    channel = JOINT_CHANNELS[joint_name]
    pwm     = angle_to_pwm(angle)
    set_pwm(channel, 0, pwm)

# -----------------------------
# Basic Robot Postures
# -----------------------------
def stand_posture():
    """ Bring all legs to a stable standing pose (HIP=90, KNEE=60) """
    angles = {
        "FL_HIP": 90, "FL_KNEE": 60,
        "FR_HIP": 90, "FR_KNEE": 60,
        "RL_HIP": 90, "RL_KNEE": 60,
        "RR_HIP": 90, "RR_KNEE": 60,
    }
    for joint, angle in angles.items():
        set_servo_angle(joint, angle)

def sit_posture():
    """ Make the robot sit by bending all knees to 120° """
    angles = {
        "FL_HIP": 90, "FL_KNEE": 120,
        "FR_HIP": 90, "FR_KNEE": 120,
        "RL_HIP": 90, "RL_KNEE": 120,
        "RR_HIP": 90, "RR_KNEE": 120,
    }
    for joint, angle in angles.items():
        set_servo_angle(joint, angle)

# -----------------------------
# Simple Gait Cycle (Walk Forward)
# -----------------------------
def walk_forward(steps=3, delay=0.3):
    """
    Perform an alternating gait:
    - Step 1: FL and RR legs swing forward
    - Step 2: FR and RL legs swing forward
    """
    for i in range(steps):
        print(f"\U0001f6b6 Walking step {i + 1}")

        # Move FL and RR forward
        set_servo_angle("FL_HIP", 110)
        set_servo_angle("RR_HIP", 110)
        time.sleep(delay)
        set_servo_angle("FL_HIP", 90)
        set_servo_angle("RR_HIP", 90)

        # Move FR and RL forward
        set_servo_angle("FR_HIP", 110)
        set_servo_angle("RL_HIP", 110)
        time.sleep(delay)
        set_servo_angle("FR_HIP", 90)
        set_servo_angle("RL_HIP", 90)

# -----------------------------
# Entry Point
# -----------------------------
def main():
    print("\U0001f527 Initializing PCA9685...")
    init_pca9685()

    print("\U0001f436 Standing...")
    stand_posture()
    time.sleep(1)

    print("\U0001f6b6 Walking...")
    walk_forward(steps=4, delay=0.3)

    print("\U0001fa91 Sitting...")
    sit_posture()

    print("\u2705 Done.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\U0001f534 Stopped by user.")
