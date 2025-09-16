import time
import smbus  # I2C control for PCA9685

I2C_ADDR = 0x40
bus = smbus.SMBus(1)
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06
PWM_FREQ = 50
SERVO_MIN = 205
SERVO_MAX = 410

JOINT_CHANNELS = {
    "FL_HIP": 0, "FL_KNEE": 1,
    "FR_HIP": 2, "FR_KNEE": 3,
    "RL_HIP": 4, "RL_KNEE": 5,
    "RR_HIP": 6, "RR_KNEE": 7,
}

"""
╔═══════╤═════════════════╤════════════════════════════════════════════╗
║ Angle  │ Used For        │ Why                                       ║
╟────────┼─────────────────┼────────────────────────────────────────────╢
║ 90°    │ HIP (neutral)   │ Leg centered under body                  ║
║ 110°   │ HIP (step fwd)  │ Swing leg forward                        ║
║ 95°    │ HIP (half-step) │ Slight curve step                        ║
╚════════╧═════════════════╧════════════════════════════════════════════╝
"""

def init_pca9685():
    bus.write_byte_data(I2C_ADDR, MODE1, 0x00)
    set_pwm_freq(PWM_FREQ)

def set_pwm_freq(freq):
    prescale = int(25000000.0 / (4096 * freq) - 1)
    old_mode = bus.read_byte_data(I2C_ADDR, MODE1)
    bus.write_byte_data(I2C_ADDR, MODE1, (old_mode & 0x7F) | 0x10)
    bus.write_byte_data(I2C_ADDR, PRESCALE, prescale)
    bus.write_byte_data(I2C_ADDR, MODE1, old_mode)
    time.sleep(0.005)
    bus.write_byte_data(I2C_ADDR, MODE1, old_mode | 0xA1)

def angle_to_pwm(angle):
    angle = max(0, min(180, angle))
    return int(SERVO_MIN + (angle / 180.0) * (SERVO_MAX - SERVO_MIN))

def set_pwm(channel, on, off):
    reg = LED0_ON_L + 4 * channel
    bus.write_byte_data(I2C_ADDR, reg, on & 0xFF)
    bus.write_byte_data(I2C_ADDR, reg+1, on >> 8)
    bus.write_byte_data(I2C_ADDR, reg+2, off & 0xFF)
    bus.write_byte_data(I2C_ADDR, reg+3, off >> 8)

def set_servo_angle(joint, angle):
    channel = JOINT_CHANNELS[joint]
    pwm = angle_to_pwm(angle)
    set_pwm(channel, 0, pwm)

def stand():
    for joint in JOINT_CHANNELS:
        angle = 90 if 'HIP' in joint else 60
        set_servo_angle(joint, angle)

def turn_left(steps=2, delay=0.3):
    for _ in range(steps):
        print("↩️ Turning Left")
        set_servo_angle("FR_HIP", 110)
        set_servo_angle("RR_HIP", 110)
        time.sleep(delay)
        set_servo_angle("FR_HIP", 90)
        set_servo_angle("RR_HIP", 90)

def turn_right(steps=2, delay=0.3):
    for _ in range(steps):
        print("↪️ Turning Right")
        set_servo_angle("FL_HIP", 110)
        set_servo_angle("RL_HIP", 110)
        time.sleep(delay)
        set_servo_angle("FL_HIP", 90)
        set_servo_angle("RL_HIP", 90)

def curve_left(steps=2, delay=0.3):
    for _ in range(steps):
        print("⬅️ Curve Left")
        set_servo_angle("FL_HIP", 95)
        set_servo_angle("RR_HIP", 110)
        time.sleep(delay)
        set_servo_angle("FL_HIP", 90)
        set_servo_angle("RR_HIP", 90)

def curve_right(steps=2, delay=0.3):
    for _ in range(steps):
        print("➡️ Curve Right")
        set_servo_angle("FR_HIP", 95)
        set_servo_angle("RL_HIP", 110)
        time.sleep(delay)
        set_servo_angle("FR_HIP", 90)
        set_servo_angle("RL_HIP", 90)

def trot_forward(steps=3, delay=0.2):
    """
    Trot gait:
    - A faster gait where diagonal leg pairs (FL+RR and FR+RL) move together.
    - More stable than gallop and faster than walk.
    - Used in quadrupeds to maintain balance at higher speeds.
    """
    for _ in range(steps):
        print("🐎 Trot Step")
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

def main():
    init_pca9685()
    stand()
    time.sleep(1)

    turn_left()
    time.sleep(1)
    turn_right()
    time.sleep(1)
    curve_left()
    time.sleep(1)
    curve_right()
    time.sleep(1)
    trot_forward()
    print("✅ Motion sequence complete.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
