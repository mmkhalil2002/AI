"""
Sound Direction Sensor Interface (TR16F064B) for Raspberry Pi
=============================================================

This script allows the Raspberry Pi to interface with the TR16F064B sound direction sensor
using SPI and a GPIO line for the BUSY signal.

🧠 What this sensor does:
-------------------------
- Detects the direction of incoming sound in 360 degrees.
- Resolution: 20 degrees (returns angles like 0°, 20°, ..., 340°, 355°).
- When a sound is detected, it pulls the BUSY pin LOW.
- The Pi polls the BUSY pin and reads the direction via SPI.

🧪 Example Use Case:
--------------------
A robot (like PiDog) uses this sensor to rotate and face toward the source of a loud sound.

Example main loop:
    sd = SoundDirection()
    while True:
        if sd.isdetected():
            print(f"🔊 Sound at {sd.read()}°")
        sleep(0.2)

🔌 Connections (Raspberry Pi → TR16F064B):
------------------------------------------
- SPI0:
    - MOSI  → GPIO10 (Pin 19)
    - MISO  → GPIO9  (Pin 21)
    - SCLK  → GPIO11 (Pin 23)
    - CE0   → GPIO8  (Pin 24) ← used as Chip Select
- BUSY     → GPIO6  (Pin 31) ← used as signal from sensor to indicate detection

📦 Communication Format:
------------------------
- The Pi sends **6 dummy bytes** over SPI: `[0, 0, 0, 0, 0, 0]`
    - The sensor ignores the data, but uses it to clock out its own response.
- The sensor responds with **6 bytes**. Only the **last two** are meaningful:
    - response[4] = LOW byte
    - response[5] = HIGH byte
- The returned values are in **little-endian** format:
    - Combined as: `(HIGH << 8) | LOW`

📊 Response interpretation:
---------------------------
- If `HIGH == 255 (0xFF)` → **Invalid or no detection**
    - No direction is returned
    - This might happen due to:
        - Background noise
        - Weak sound
        - Glitch during transfer
- If `HIGH != 255` → **Valid detection**
    - Sensor encodes angle with formula: `raw = 160 - angle`
    - To decode: `angle = (360 + 160 - raw) % 360`
    - Allows full 360° wrap-around using modulus

📐 Direction Range:
-------------------
- 360-degree circular detection
- Resolution: 20° per step
- Output range: 0° to 355°

🧠 How the TR16F064B Detects Sound:
-----------------------------------
The TR16F064B module contains multiple microphones (usually 6 or 8) arranged in a circular layout.

🔁 Step-by-step Detection Process:
- All microphones listen simultaneously.
- For each sound wave received, the chip measures:
    1. Time difference of arrival (TDOA) between microphones
    2. Amplitude (loudness)
- It uses beamforming algorithms to:
    - Cancel out ambient noise (to some degree)
    - Estimate the direction of arrival (DOA) of the dominant sound
- The module then:
    - Locks in a direction
    - Pulls the BUSY pin LOW
    - Sends the 16-bit direction angle to master via SPI

🎯 Real-world Sound Selection Logic:
------------------------------------
Scenario	                  → What the sensor detects
----------------------------|-------------------------------
One person claps nearby     → Detects clap direction accurately
Two people speak at once    → Reports the louder/closer speaker
Loud music & quiet speech   → Picks music direction (higher volume)
Sudden noise (e.g. knock)   → Detects knock instantly (sharp peak)
Constant background hum     → May ignore if not sudden/loud

🧠 Key Characteristics:
------------------------
- ❗ It detects **only the dominant (loudest)** sound at the moment.
- 🕒 It is **event-triggered** — BUSY goes LOW only when a new sound is detected.
- 📈 You can’t get intensity (dB) — only direction (angle).
- ⚠️ Quiet or far sounds may be ignored if background noise is strong.

Tested on: Raspberry Pi 4 running Ubuntu 22.04

"""

import spidev                     # SPI communication library
from gpiozero import InputDevice  # Library for reading GPIO pins
from time import sleep            # Delay utility


class SoundDirection:
    """
    Class for interacting with the TR16F064B sound direction sensor via SPI and BUSY pin.
    """

    # SPI Configuration Constants
    CLOCK_SPEED_HZ = 10_000_000   # SPI clock frequency (10 MHz)
    CS_DELAY_US = 500             # Delay between chip select toggles (in microseconds)

    def __init__(self, spi_bus=0, spi_device=0, busy_pin=6):
        """
        Initializes the SPI interface and the BUSY GPIO pin.

        Parameters:
            spi_bus (int): SPI bus number (usually 0)
            spi_device (int): SPI device number (0 = CE0, 1 = CE1)
            busy_pin (int): GPIO pin used for BUSY (active LOW)
        """

        # --- SPI Setup ---
        self.spi = spidev.SpiDev()               # Create SPI device object
        self.spi.open(spi_bus, spi_device)       # Open SPI interface (e.g. /dev/spidev0.0)
        self.spi.max_speed_hz = self.CLOCK_SPEED_HZ  # Set the SPI clock speed

        # --- BUSY GPIO Setup ---
        self.busy = InputDevice(busy_pin, pull_up=False)  # Set BUSY as input (active LOW)

    def read(self):
        """
        Sends a 6-byte dummy command over SPI and receives the direction angle.

        Returns:
            int: Direction angle in degrees (0–355), or -1 if invalid.
        """

        # 🛰️ Send 6 dummy bytes; the sensor uses them to clock out its response
        response = self.spi.xfer2(
            [0, 0, 0, 0, 0, 0],         # Dummy data (sensor only responds)
            self.CLOCK_SPEED_HZ,       # SPI clock rate
            self.CS_DELAY_US           # Delay between toggling CS (microseconds)
        )

        # 🧾 Extract the 5th and 6th bytes: [low_byte, high_byte]
        low_byte = response[4]
        high_byte = response[5]

        # ❌ HIGH == 255 (0xFF) means the sensor did not detect any valid sound
        if high_byte == 0xFF:
            return -1

        # ✅ If HIGH is a valid number (e.g., 0–254), proceed
        # Combine the two bytes into a single 16-bit raw value
        raw_value = (high_byte << 8) | low_byte

        # 🔁 Convert raw value to angle:
        # Sensor encodes angle as: raw = 160 - angle
        # We reverse this: angle = (360 + 160 - raw) % 360
        angle = (360 + 160 - raw_value) % 360

        return angle

    def isdetected(self):
        """
        Checks if a new sound has been detected by monitoring the BUSY pin.

        Returns:
            bool: True if BUSY is LOW (detection occurred), False otherwise.
        """
        return self.busy.value == 0  # Sensor pulls BUSY LOW when it hears a sound


# --------------------- MAIN LOOP EXAMPLE ---------------------
if __name__ == '__main__':
    # Create an instance of the sound direction sensor on GPIO6
    sd = SoundDirection(busy_pin=6)

    print("📡 Waiting for sound direction...")

    # Infinite loop to poll for sound detection
    while True:
        if sd.isdetected():  # BUSY is LOW → sound was detected
            angle = sd.read()  # Try to read the sound direction angle

            if angle >= 0:
                print(f"🔊 Sound detected at {angle}°")
            else:
                print("⚠️ Invalid or no direction data received")

        sleep(0.2)  # Slight delay between checks
