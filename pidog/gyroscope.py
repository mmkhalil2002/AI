# mpu6050_read.py
# ==============================================================================
# ✅ MPU6050 (gyroscope) OVERVIEW: HOW IT WORKS
# ------------------------------------------------------------------------------
# 📦 What's inside the MPU6050?
# - 3-axis Accelerometer: Measures linear motion (g-force)
# - 3-axis Gyroscope: Measures rotational speed (degrees/sec)
# - Optional: Temperature sensor and a DMP (Digital Motion Processor)
#
# 🧠 Data Reading Workflow:
#   1. Communicate using I2C protocol (bus 1 on Raspberry Pi)
#   2. Wake the sensor up from sleep mode by writing to register 0x6B
#   3. Read 6 bytes from accelerometer (starting at 0x3B) and 6 from gyroscope (0x43)
#   4. Convert raw 16-bit integers to human-readable units using scaling:
#        • Accelerometer (±2g range):    16384 LSB = 1g → divide by 16384.0
#        • Gyroscope (±250°/s range):    131 LSB = 1°/s → divide by 131.0
#
# 📏 Why divide by 16384 for accel and 131 for gyro?
# ---------------------------------------------------
# The MPU6050 outputs **raw sensor data** in 16-bit signed integer format (−32768 to +32767).
#
# • The **accelerometer** operates in ±2g mode by default. "g" is acceleration due to gravity,
#   approximately **9.81 m/s²** on Earth. So:
#     - 1g = 9.81 m/s²
#     - 2g = 19.62 m/s²
#     - 0g  = freefall (no net force)
#
#   The sensor encodes 1g as 16384. So:
#     - +16384 → +1g → divide by 16384.0 to get "g" units.
#     - +8192  → +0.5g
#     - 0      → 0g (freefall, e.g., jump or fall)
#     - −16384 → −1g (opposite acceleration)
#
#   These g-units are **directional** and **relative to gravity**.
#   Example: If PiDog is standing upright, the Z-axis will be near +1.0g (gravity pulling down).
#
# • The **gyroscope** outputs angular velocity in degrees per second (°/s).
#   In ±250°/s mode:
#     - +131 → +1°/s
#     - +262 → +2°/s
#     - 0    → no rotation
#     - −131 → −1°/s
#   Divide raw value by 131.0 to get degrees/second.
#
# 🔭 Axis Orientation (Facing Forward):
#
#        PiDog (Top View Facing Forward)
#                ↑ Y axis (forward)
#                |
#                |
#                ● MPU6050 chip
#               / \
#     X axis ←     → X axis
#           (left)     (right)
#
#        Z axis → upward (out of the page)
#        Z axis → downward (into the page) when upright
#
# 🧭 Summary:
#   - X = side-to-side tilt (roll)
#   - Y = front-to-back tilt (pitch)
#   - Z = up/down movement or gravity
#
# 🪂 Why all g ≈ 0 during freefall or jumping?
# -------------------------------------------
# The accelerometer measures **force acting on the device** relative to free space.
# If PiDog is falling (like in a jump or dropped), there's no normal force on it—
# it's in "weightlessness", so the sensor reads 0g on all axes.
#
# 🔬 Real-world examples:
# ┌────────────────────────────┬─────────────────────────────────────────────┐
# │ Situation                  │ Expected Output                             │
# ├────────────────────────────┼─────────────────────────────────────────────┤
# │ PiDog is upright & still   │ Accel Z ≈ +1.00g, others ≈ 0, gyro ≈ 0       │
# │ PiDog falls sideways       │ Accel X or Y ≈ ±1g, Z drops from 1.00g      │
# │ PiDog is shaken            │ High gyro X/Y values (e.g., ±200°/s)        │
# │ PiDog spins in place       │ Gyro Z shows ±180°/s                        │
# │ Freefall (jump/dropped)    │ All accel axes ≈ 0g (Z, X, Y → 0)           │
# └────────────────────────────┴─────────────────────────────────────────────┘
#
# 📊 Table: Interpreting g values for PiDog posture and movement
# ┌───────────────┬──────────────┬──────────────┬─────────────────────────────┐
# │ Accel X (g)   │ Accel Y (g)  │ Accel Z (g)  │ Interpreted Motion/Posture  │
# ├───────────────┼──────────────┼──────────────┼─────────────────────────────┤
# │ ~0            │ ~0           │ +1.0         │ Standing upright             │
# │ ±1.0          │ ~0           │ ~0           │ Tipped left/right (roll)     │
# │ ~0            │ ±1.0         │ ~0           │ Tipped forward/backward      │
# │ ~0            │ ~0           │ ~0           │ Freefall / jumping           │
# │ +1.0          │ ~0           │ ~0           │ Left tilt / left acceleration│
# │ -1.0          │ ~0           │ ~0           │ Right tilt / right accel.    │
# │ ~0            │ +1.0         │ ~0           │ Forward acceleration         │
# │ ~0            │ -1.0         │ ~0           │ Backward acceleration        │
# │ ~0            │ ~0           │ +1.0         │ Downward gravity (upright)   │
# │ ~0            │ ~0           │ -1.0         │ Upside down / upward motion  │
# └───────────────┴──────────────┴──────────────┴─────────────────────────────┘
#
# 🎯 Use cases in PiDog:
#   - 📉 Bark when falling (Z < 0.5g)
#   - 🌀 Recenter when gyro > 180°/s
#   - 🐾 Adjust posture if tilted (X/Y > 0.5g)
#   - 🤖 Combine with sound sensor to react to sudden motion + noise
#
# 🧰 Physical Wiring (Raspberry Pi GPIO to MPU6050 I2C):
#   SDA (data) → GPIO2 (Pin 3)
#   SCL (clock) → GPIO3 (Pin 5)
#   VCC → 3.3V or 5V
#   GND → Ground
#
# 🔌 Bus note:
#   Raspberry Pi uses I2C bus 1 → /dev/i2c-1
#   MPU6050 default address = 0x68
# ==============================================================================

import smbus     # For I2C communication with the MPU6050
import time      # For delays between readings

# --------------------- MPU6050 REGISTER MAP ---------------------
MPU6050_ADDR    = 0x68     # MPU6050 I2C address from datasheet
PWR_MGMT_1      = 0x6B     # Power management register to wake the device
ACCEL_XOUT_H    = 0x3B     # Start of accelerometer data (X-axis high byte)
GYRO_XOUT_H     = 0x43     # Start of gyroscope data (X-axis high byte)

# --------------------- Initialize I2C ---------------------
# Use I2C bus 1 (default for Raspberry Pi 3/4/5)
bus = smbus.SMBus(1)

# 💤 Wake up the MPU6050 from sleep mode
# Writing 0 to PWR_MGMT_1 turns off sleep mode (bit 6 = 0)
bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)

# --------------------- Raw Data Reading Helper ---------------------
def read_raw_data(addr):
    """
    Reads two bytes (high and low) from the given I2C register and converts to signed 16-bit integer.

    Args:
        addr (int): The starting register address

    Returns:
        int: Converted signed integer (-32768 to +32767)
    """
    high = bus.read_byte_data(MPU6050_ADDR, addr)       # High byte (MSB)
    low  = bus.read_byte_data(MPU6050_ADDR, addr + 1)   # Low byte (LSB)
    value = (high << 8) | low                           # Combine to 16-bit

    # Convert to signed 2's complement
    if value > 32767:
        value -= 65536
    return value

# --------------------- Main Program ---------------------
def main():
    print("📡 Reading MPU6050 data continuously (Ctrl+C to stop)...\n")

    try:
        while True:
            # ----------- ACCELEROMETER READINGS (g-force) -----------
            # Raw output range: ±32768
            # Sensitivity: 16384 LSB = 1g (where g = 9.81 m/s²)
            # → Divide raw values by 16384.0 to convert to g-units
            acc_x = read_raw_data(ACCEL_XOUT_H)     / 16384.0
            acc_y = read_raw_data(ACCEL_XOUT_H + 2) / 16384.0
            acc_z = read_raw_data(ACCEL_XOUT_H + 4) / 16384.0

            # ----------- GYROSCOPE READINGS (angular velocity) -----------
            # Sensitivity: 131 LSB = 1°/sec
            # → Divide raw values by 131.0 to convert to °/s
            gyro_x = read_raw_data(GYRO_XOUT_H)     / 131.0
            gyro_y = read_raw_data(GYRO_XOUT_H + 2) / 131.0
            gyro_z = read_raw_data(GYRO_XOUT_H + 4) / 131.0

            # ----------- Display the sensor values -----------
            print(f"📈 Accel (g):  X={acc_x:+.2f}, Y={acc_y:+.2f}, Z={acc_z:+.2f}")
            print(f"🌀 Gyro (°/s): X={gyro_x:+.2f}, Y={gyro_y:+.2f}, Z={gyro_z:+.2f}")
            print("-" * 42)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("❌ Stopped by user")

# --------------------- Entry Point ---------------------
if __name__ == "__main__":
    main()
