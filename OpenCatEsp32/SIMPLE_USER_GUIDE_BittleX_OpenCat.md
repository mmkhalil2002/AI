# Bittle X / OpenCat Tools — Simple User Guide

This guide explains how to use these two files:

1. `config.py`
2. `generate-motion-data-and-simulate.py`

---

# 1. config.py

## Purpose

This tool helps you:

- Install all required ESP32 tools
- Download OpenCat firmware
- Install required libraries
- Compile firmware
- Merge custom code
- Upload firmware to ESP32

---

# How To Run

Open PowerShell inside the project folder and run:

```powershell
python config.py
```

---

# Main Menu Commands

| Command | Description |
|---|---|
| `1` | Install all required tools, firmware, and libraries |
| `2` | Compile firmware only |
| `3` | Merge custom firmware with OpenCat firmware |
| `4` | Copy merged firmware into OpenCat project |
| `5` | Flash/upload firmware to ESP32 |
| `q` | Exit |

---

# ESP32 Target Selection

At startup you select the board type.

| Option | Board Type |
|---|---|
| `1` | Classic ESP32 boards |
| `2` | ESP32-S2 boards |
| `3` | ESP32-S3 boards |

---

# Example Supported Boards

## Classic ESP32

Examples:

```text
ESP32-WROOM-32
ESP32-WROVER
ESP32 DevKit V1
NodeMCU-32S
Bittle X classic ESP32 board
```

## ESP32-S2

Examples:

```text
ESP32-S2 Saola
ESP32-S2 DevKit
```

## ESP32-S3

Examples:

```text
ESP32-S3 DevKitC
ESP32-S3-WROOM
ESP32-S3-WROVER
```

---

# Typical Workflow

## First Time Setup

Run:

```text
1
```

This installs everything automatically.

---

## Compile Firmware

Run:

```text
2
```

This generates firmware files such as:

```text
.bin
.elf
.map
```

---

## Merge Custom Firmware

Run:

```text
3
```

This merges:

```text
myOpenCatEsp32.ino
```

with:

```text
OpenCatEsp32.ino
```

The result is:

```text
mergeOpenCatEsp32.ino
```

---

## Copy Merged Firmware

Run:

```text
4
```

This copies:

```text
mergeOpenCatEsp32.ino
```

into the OpenCat firmware folder.

---

## Upload Firmware

Run:

```text
5
```

This flashes the firmware to the ESP32 board over USB.

---

# 2. generate-motion-data-and-simulate.py

## Purpose

This tool lets you:

- Create robot dog motions
- Simulate motions in PyBullet
- Generate ESP32/OpenCat C++ motion code

---

# How To Run

```powershell
python generate-motion-data-and-simulate.py
```

---

# Motion Commands

| Command | Motion |
|---|---|
| `1` | Walk |
| `2` | Trot / Run |
| `3` | Sit |
| `4` | Stand |
| `5` | Jump |
| `6` | Turn Left |
| `7` | Turn Right |
| `q` | Generate C++ file and exit |

---

# What Happens

1. Select a motion
2. The simulator shows the motion
3. Select more motions if needed
4. Press `q`
5. The tool generates:

```text
generated_esp32_dog_motion_player.cpp
```

---

# Generated C++ Features

The generated C++ program:

- Stores motion tables
- Controls servos
- Accepts USB commands
- Accepts Wi-Fi commands

---

# Supported Robot Commands

Example commands:

```text
walk
trot_run
sit
stand
jump
turn_left
turn_right
help
```

---

# USB and Wi-Fi Priority

The ESP32 checks commands in this order:

```text
1. USB Serial
2. Wi-Fi TCP
```

If USB has a command, USB is used first.

If USB has no command, Wi-Fi is checked.

---

# Wi-Fi Setup

Inside generated C++ code:

```cpp
const char* WIFI_SSID     = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
```

Replace them with your real Wi-Fi settings.

---

# Wi-Fi TCP Port

The ESP32 listens on:

```text
Port 8888
```

---

# Example Wi-Fi Python Command

```python
import socket

ESP32_IP = "192.168.1.55"
PORT = 8888

s = socket.socket()
s.connect((ESP32_IP, PORT))

s.sendall(b"walk\n")

print(s.recv(1024))

s.close()
```

---

# Generated Servo Labels

The generated C++ uses:

```text
FL_HIP
FL_KNEE
FR_HIP
FR_KNEE
RL_HIP
RL_KNEE
RR_HIP
RR_KNEE
```

Meaning:

| Label | Meaning |
|---|---|
| `FL` | Front Left |
| `FR` | Front Right |
| `RL` | Rear Left |
| `RR` | Rear Right |

---

# Summary

| File | Purpose |
|---|---|
| `config.py` | Install, compile, merge, and upload firmware |
| `generate-motion-data-and-simulate.py` | Design motions and generate ESP32 motion code |

---

# Recommended Workflow

```text
Generate motion
    ↓
Simulate motion
    ↓
Generate C++ file
    ↓
Merge firmware
    ↓
Compile firmware
    ↓
Flash ESP32
    ↓
Control robot
```
// =====================================================
// Example ESP32 Servo GPIO Mapping
// =====================================================
//
// These numbers are ESP32 GPIO pins.
//
// GPIO = General Purpose Input Output
//
// The ESP32 servo library (ESP32Servo.h)
// generates PWM (Pulse Width Modulation)
// signals on these GPIO pins to control servos.
//
// Servos typically use:
//   Frequency: ~50 Hz
//   Pulse width: ~500 us to ~2500 us
//
// These GPIO pins were selected because:
//
// 1. They support PWM (LEDC hardware)
// 2. They support Digital Output mode
// 3. They are commonly available
// 4. They avoid ESP32 internal flash pins
// 5. They avoid input-only pins
//
// Avoid for servos:
//
// GPIO34-39
//   -> Input only
//   -> Cannot generate servo output
//
// GPIO6-11
//   -> Used internally by ESP32 flash
//
// GPIO1, GPIO3
//   -> Serial programming pins
//
// GPIO0, GPIO2, GPIO15
//   -> Boot configuration pins
//   -> Use carefully
//
// Mapping:
//
// Front Left Hip    -> GPIO13
// Front Left Knee   -> GPIO12
//
// Front Right Hip   -> GPIO14
// Front Right Knee  -> GPIO27
//
// Rear Left Hip     -> GPIO26
// Rear Left Knee    -> GPIO25
//
// Rear Right Hip    -> GPIO33
// Rear Right Knee   -> GPIO32
//
// =====================================================

int servoPins[8] =
{
    13, // GPIO13
        // Digital Output: YES
        // PWM (LEDC): YES
        // ADC: YES
        // Selected because PWM capable

    12, // GPIO12
        // Digital Output: YES
        // PWM (LEDC): YES
        // ADC: YES
        // Selected because PWM capable

    14, // GPIO14
        // Digital Output: YES
        // PWM (LEDC): YES
        // ADC: YES
        // Selected because PWM capable

    27, // GPIO27
        // Digital Output: YES
        // PWM (LEDC): YES
        // ADC: YES
        // Selected because PWM capable

    26, // GPIO26
        // Digital Output: YES
        // PWM (LEDC): YES
        // ADC: YES
        // DAC: YES
        // Selected because PWM capable

    25, // GPIO25
        // Digital Output: YES
        // PWM (LEDC): YES
        // ADC: YES
        // DAC: YES
        // Selected because PWM capable

    33, // GPIO33
        // Digital Output: YES
        // PWM (LEDC): YES
        // ADC: YES
        // Selected because PWM capable

    32  // GPIO32
        // Digital Output: YES
        // PWM (LEDC): YES
        // ADC: YES
        // Selected because PWM capable
};