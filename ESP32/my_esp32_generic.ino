/*
============================================================
ESP32 GENERIC DIRECT-SERVO + SMART-MOTOR FRAMEWORK
============================================================

PURPOSE
------------------------------------------------------------
This firmware is generic. It is NOT tied to one application.

You can use it for:
  - robot arm
  - gripper
  - pan/tilt camera
  - robot dog
  - wheels/tracks with continuous servos
  - smart-servo experiments
  - professional actuator experiments

This version assumes:
  - Hobby servos connect DIRECTLY to ESP32 GPIO pins.
  - No PCA9685 is required.
  - Professional/smart motors connect through UART/CAN/RS485 style buses.

============================================================
1) HOBBY SERVO DIRECT CONNECTION TO ESP32
============================================================

A hobby servo usually has 3 wires:

  Brown/Black  = GND
  Red          = +5V or +6V servo power
  Yellow/White = signal

Correct wiring:

  ESP32 GPIO4  --------> Servo signal
  External 5V  --------> Servo red wire
  External GND --------> Servo brown/black wire
  ESP32 GND    --------> External GND

IMPORTANT:
  Do NOT power servos from ESP32 3.3V.
  Do NOT power many servos from ESP32 5V pin.
  Use an external 5V/6V supply with enough current.
  Always connect ESP32 GND and servo power GND together.

Example for 4 hobby servos:

  Motor 0 signal -> GPIO4
  Motor 1 signal -> GPIO5
  Motor 2 signal -> GPIO13
  Motor 3 signal -> GPIO14

============================================================
2) PROFESSIONAL / SMART MOTOR CONNECTION TO ESP32
============================================================

Professional motors are usually NOT controlled by hobby PWM.

They often use:
  - UART / TTL serial
  - RS485
  - CAN bus

Examples:
  - Feetech STS3215 / ST3215
  - Dynamixel
  - Unitree-style actuator
  - CAN smart motors

For smart motors, the ESP32 sends digital commands:

  motor ID
  angle
  speed
  torque
  direction

Example STS3215-style UART wiring:

  ESP32 GPIO17 TX2 ----> Smart servo bus signal
  ESP32 GPIO16 RX2 <---- Smart servo bus feedback
  ESP32 GND ----------- Smart servo GND
  External battery ---- Smart servo power

Many smart servos can be daisy chained:

  ESP32 UART
      |
      +---- Smart Servo ID 1
              |
              +---- Smart Servo ID 2
                      |
                      +---- Smart Servo ID 3

Each smart motor must have a unique ID.

IMPORTANT:
  Do NOT power smart motors from ESP32.
  Use the correct external voltage for the motor model.
  Connect grounds together.
  Some buses need a half-duplex driver or RS485/CAN transceiver.

============================================================
3) HOBBY VS PROFESSIONAL CONTROL MODEL
============================================================

Hobby positional servo:
  - angle is real target angle.
  - speed is simulated by stepping slowly.
  - torque is not truly controllable.

Hobby continuous-rotation servo:
  - angle is not real position.
  - 90  = stop
  - >90 = clockwise
  - <90 = counter-clockwise
  - speed is distance from 90.

Professional motor:
  - angle, speed, torque, and direction are separate concepts.
  - feedback may be available.
  - command protocol must be added inside sendSmartMotorCommand().

============================================================
4) ESP32 GPIO APPLICATION NOTES
============================================================

Recommended for direct servo/PWM:
  GPIO4, GPIO5, GPIO13, GPIO14,
  GPIO15, GPIO16, GPIO17, GPIO18,
  GPIO19, GPIO21, GPIO22, GPIO23,
  GPIO25, GPIO26, GPIO27, GPIO32,
  GPIO33

I2C:
  GPIO21 = SDA
  GPIO22 = SCL

UART smart motors:
  GPIO17 = TX2
  GPIO16 = RX2

SPI:
  GPIO18 = CLK
  GPIO19 = MISO
  GPIO23 = MOSI
  GPIO5  = CS

ADC / sensors:
  GPIO32, GPIO33, GPIO34, GPIO35, GPIO36, GPIO39

DAC:
  GPIO25 = DAC1
  GPIO26 = DAC2

Avoid for direct motor output:
  GPIO0  = boot strap pin
  GPIO1  = UART0 TX / Serial Monitor
  GPIO2  = boot strap pin
  GPIO3  = UART0 RX / Serial Monitor
  GPIO6-11 = connected to SPI flash memory
  GPIO12 = boot strap / flash voltage risk
  GPIO34-39 = input only

============================================================
*/

#include <Arduino.h>
#include <ESP32Servo.h>
#include <WiFi.h>
#include <ESPmDNS.h>


// ============================================================
// ARDUINO AUTO-PROTOTYPE COMPATIBILITY FOR MotionFrame
// ============================================================
// Arduino IDE / Arduino CLI automatically creates function prototypes
// before compiling a .ino sketch. Because runMotionFrame() uses the custom
// type MotionFrame, the compiler must know that MotionFrame is a valid type
// name before the auto-generated prototype is created.
//
// This forward declaration does NOT create a variable and does NOT change the
// motion logic. It simply tells the compiler:
//   "A struct named MotionFrame will be fully defined later."
//
// Without this line, some Arduino builds report:
//   error: variable or field 'runMotionFrame' declared void
//   error: 'MotionFrame' was not declared in this scope
// ============================================================
struct MotionFrame;


// ============================================================
// WIFI CONNECTIVITY CONFIGURATION
// ============================================================
// This section adds Wi-Fi communication in addition to USB Serial.
//
// Normal operation:
//   1) Python tries USB Serial first.
//   2) If USB is not available, Python can connect to the ESP32
//      over Wi-Fi using the IP address and TCP port below.
//
// IMPORTANT:
//   - Replace WIFI_SSID and WIFI_PASSWORD with your Wi-Fi network.
//   - The ESP32 and the Python PC must be on the same network.
//   - The ESP32 also prints its IP address over Serial at startup.
//   - Python can request the IP any time by sending: @IP
//
// TCP server:
//   Python can connect to: ESP32_IP_ADDRESS:3333
//
// mDNS hostname:
//   If your PC supports mDNS/Bonjour, you can also try:
//      esp32motor.local
// ============================================================

const char* WIFI_SSID     = "PUT_YOUR_WIFI_NAME_HERE";
const char* WIFI_PASSWORD = "PUT_YOUR_WIFI_PASSWORD_HERE";

const char* WIFI_HOSTNAME = "esp32motor";
const uint16_t WIFI_MOTOR_PORT = 3333;

WiFiServer motorTcpServer(WIFI_MOTOR_PORT);
WiFiClient motorTcpClient;

bool wifiConnected = false;
bool motorTcpServerStarted = false;
bool periodicMotorStatusEnabled = false;
unsigned long lastPeriodicMotorStatusMs = 0;
const unsigned long PERIODIC_MOTOR_STATUS_INTERVAL_MS = 1000;

String serialCommandLine = "";
String wifiCommandLine = "";

// ============================================================
// MOTOR TYPES
// ============================================================

enum MotorType
{
    MOTOR_HOBBY_POSITIONAL_SERVO = 0,
    MOTOR_HOBBY_CONTINUOUS_SERVO = 1,
    MOTOR_SMART_PROFESSIONAL_MOTOR = 2
};

// ============================================================
// DIRECTION TYPES
// ============================================================

enum MotorDirection
{
    DIR_STOP = 0,
    DIR_CW   = 1,
    DIR_CCW  = -1
};

// ============================================================
// MOTOR COUNT AND DIRECT ESP32 GPIO MAP
// ============================================================
//
// This framework directly connects hobby servos to these ESP32 GPIOs.
//
// Motor index to GPIO:
//
//   Motor 0  -> GPIO4
//   Motor 1  -> GPIO5
//   Motor 2  -> GPIO13
//   Motor 3  -> GPIO14
//   Motor 4  -> GPIO15
//   Motor 5  -> GPIO16
//   Motor 6  -> GPIO17
//   Motor 7  -> GPIO18
//   Motor 8  -> GPIO19
//   Motor 9  -> GPIO21
//   Motor 10 -> GPIO22
//   Motor 11 -> GPIO23
//   Motor 12 -> GPIO25
//   Motor 13 -> GPIO26
//   Motor 14 -> GPIO27
//   Motor 15 -> GPIO32
//
// You can change this table to match your wiring.
//
// ============================================================

#define MOTOR_COUNT 16

const uint8_t DIRECT_ESP32_GPIO_PIN[MOTOR_COUNT] =
{
     4,  5, 13, 14,
    15, 16, 17, 18,
    19, 21, 22, 23,
    25, 26, 27, 32
};

// ============================================================
// SMART MOTOR UART SETTINGS
// ============================================================
//
// These are used only if you configure a motor as:
//   MOTOR_SMART_PROFESSIONAL_MOTOR
//
// For STS3215-style smart servos, UART2 is commonly used:
//
//   TX2 = GPIO17
//   RX2 = GPIO16
//
// ============================================================

HardwareSerial SmartMotorSerial(2);

const int SMART_UART_RX = 16;
const int SMART_UART_TX = 17;
const int SMART_UART_BAUD = 1000000;

// ============================================================
// FIRST-MOTOR TYPE AUTO-DETECTION SETTINGS
// ============================================================
//
// IMPORTANT LIMITATION:
//   A normal hobby PWM servo cannot identify itself to the ESP32.
//   It has only power, ground, and one PWM signal wire, so there is no
//   feedback path for model/type detection.
//
// What this auto-detection does:
//   1) It checks ONLY the first motor position.
//   2) It tries to ping a smart/professional servo on the smart UART bus
//      using the first motor ID.
//   3) If a smart servo replies, ALL motors are configured as
//      MOTOR_SMART_PROFESSIONAL_MOTOR.
//   4) If no smart servo replies, ALL motors are configured as the
//      fallback hobby-servo type below.
//
// This matches the requested rule:
//   "Check only the type of the first motor, and the rest will be the same."
//
// If you are using hobby servos, no smart reply will be received, so the
// program falls back to AUTO_DETECT_FALLBACK_HOBBY_TYPE.
//
// If you are using continuous-rotation hobby servos, change the fallback to:
//   MOTOR_HOBBY_CONTINUOUS_SERVO
//
// If you are using positional hobby servos, keep the fallback as:
//   MOTOR_HOBBY_POSITIONAL_SERVO
//
// Smart-servo protocol note:
//   The ping packet below is written for common Feetech/STS/SCS-style serial
//   bus servos that use 0xFF 0xFF packet headers. If your smart servo uses a
//   different protocol, such as Dynamixel Protocol 2.0, CAN, or RS485 Modbus,
//   update pingSmartMotorId() with the correct ping/read-model command.
// ============================================================

const bool AUTO_DETECT_FIRST_MOTOR_TYPE_AT_BOOT = true;
const MotorType AUTO_DETECT_FALLBACK_HOBBY_TYPE = MOTOR_HOBBY_POSITIONAL_SERVO;

// Runtime motor type selected by setup() auto-detection.
//
// Why this exists:
//   After setup() checks the first motor, this variable stores the final
//   detected type. Every high-level command then uses motors[i].type, which
//   is set from this value, so the program calls the correct related driver:
//
//     MOTOR_HOBBY_POSITIONAL_SERVO       -> driveHobbyPositionalServo()
//     MOTOR_HOBBY_CONTINUOUS_SERVO       -> driveHobbyContinuousServo()
//     MOTOR_SMART_PROFESSIONAL_MOTOR     -> driveSmartProfessionalMotor()
//
// This means detection is not just printed; it actually changes the runtime
// control path used by @SET, @CW, @CCW, @STOP, @SPEED_UP, @SPEED_DOWN,
// motion frames, and serial one-key controls.
MotorType g_detectedMotorType = AUTO_DETECT_FALLBACK_HOBBY_TYPE;

const unsigned long SMART_PING_TIMEOUT_MS = 80;

// First smart motor ID to test.
// The program checks only this first ID, then applies the detected type to
// all motors. The rest of the smart motors use IDs 1..MOTOR_COUNT.
const int AUTO_DETECT_FIRST_SMART_ID = 1;

// ============================================================
// MOTOR CONFIGURATION STRUCTURE
// ============================================================

struct MotorConfig
{
    int index;
    MotorType type;

    // For hobby servos:
    //   outputPin = ESP32 GPIO pin.
    //
    // For smart motors:
    //   outputPin is not used and can stay -1.
    int outputPin;

    // For smart motors:
    //   id = digital motor ID on the bus.
    //
    // For hobby servos:
    //   id can simply be index + 1.
    int id;

    // Generic command values:
    float angleDeg;
    float speed;
    float torque;

    MotorDirection direction;
    bool enabled;
};

// ============================================================
// GLOBAL MOTOR OBJECTS
// ============================================================

MotorConfig motors[MOTOR_COUNT];
Servo hobbyServos[MOTOR_COUNT];

// ============================================================
// LIMITS AND DEFAULTS
// ============================================================

const float MIN_ANGLE_DEG = 0.0;
const float MAX_ANGLE_DEG = 180.0;

const float MIN_SPEED = 0.0;
const float MAX_SPEED = 100.0;

const float MIN_TORQUE = 0.0;
const float MAX_TORQUE = 100.0;

const float SPEED_STEP = 10.0;
const float TORQUE_STEP = 5.0;

const int CONTINUOUS_STOP_VALUE = 90;

// ============================================================
// UTILITY FUNCTIONS
// ============================================================


// ------------------------------------------------------------
// FUNCTION: isValidMotorIndex()
// ------------------------------------------------------------
// Checks whether a motor index is inside the valid range 0..MOTOR_COUNT-1. This protects all motor functions from reading or writing outside the motors[] array.
// ------------------------------------------------------------
bool isValidMotorIndex(int motorIndex)
{
    if(motorIndex < 0 || motorIndex >= MOTOR_COUNT)
    {
        Serial.print("[ERROR] Invalid motor index: ");
        Serial.println(motorIndex);
        return false;
    }

    return true;
}


// ------------------------------------------------------------
// FUNCTION: clampFloat()
// ------------------------------------------------------------
// Limits a floating-point value to a safe minimum and maximum. Used for angle, speed, and torque so invalid values from Serial/Wi-Fi/Python cannot damage the motion logic.
// ------------------------------------------------------------
float clampFloat(float value, float minValue, float maxValue)
{
    if(value < minValue) return minValue;
    if(value > maxValue) return maxValue;
    return value;
}

// ============================================================
// INITIALIZE MOTOR CONFIGURATION
// ============================================================
//
// Default:
//   All motors are configured as hobby positional servos.
//
// To use a continuous servo, change:
//   motors[i].type = MOTOR_HOBBY_CONTINUOUS_SERVO;
//
// To use a smart/professional motor, change:
//   motors[i].type = MOTOR_SMART_PROFESSIONAL_MOTOR;
//   motors[i].outputPin = -1;
//   motors[i].id = smart motor ID;
//
// ============================================================


// ------------------------------------------------------------
// FUNCTION: initializeMotorConfig()
// ------------------------------------------------------------
// Initializes the motor table at boot. It assigns each motor an index, default type, GPIO pin, ID, default angle, speed, torque, direction, and disabled state.
// ------------------------------------------------------------
void initializeMotorConfig()
{
    for(int i = 0; i < MOTOR_COUNT; i++)
    {
        motors[i].index = i;
        motors[i].type = MOTOR_HOBBY_POSITIONAL_SERVO;
        motors[i].outputPin = DIRECT_ESP32_GPIO_PIN[i];
        motors[i].id = i + 1;

        motors[i].angleDeg = 90;
        motors[i].speed = 50;
        motors[i].torque = 0;
        motors[i].direction = DIR_STOP;
        motors[i].enabled = false;
    }

    // Example: uncomment to make motor 0 a continuous-rotation servo.
    // motors[0].type = MOTOR_HOBBY_CONTINUOUS_SERVO;

    // Example: uncomment to make motor 1 a smart/professional motor.
    // motors[1].type = MOTOR_SMART_PROFESSIONAL_MOTOR;
    // motors[1].outputPin = -1;
    // motors[1].id = 1;
}


// ============================================================
// FIRST-MOTOR TYPE AUTO-DETECTION
// ============================================================
//
// These functions keep the original manual configuration logic available,
// but add an optional boot-time and command-time detection step.
//
// The detection rule is intentionally simple:
//
//   - Test the FIRST smart motor ID only.
//   - If the first motor replies on UART, assume all motors are smart.
//   - If the first motor does not reply, assume all motors are hobby servos.
//
// This is useful when your robot/project uses one consistent motor family.
// It avoids requiring every motor entry to be edited manually.
//
// Manual override is still possible:
//   After auto-detection, you can still edit motors[i].type, outputPin, and id
//   in initializeMotorConfig() if you want a mixed system.
// ============================================================


// ------------------------------------------------------------
// FUNCTION: motorTypeName()
// ------------------------------------------------------------
// Converts the MotorType enum into readable text for Serial, Wi-Fi TCP,
// and Python status messages.
// ------------------------------------------------------------
const char* motorTypeName(MotorType type)
{
    if(type == MOTOR_HOBBY_POSITIONAL_SERVO) return "HOBBY_POSITIONAL_SERVO";
    if(type == MOTOR_HOBBY_CONTINUOUS_SERVO) return "HOBBY_CONTINUOUS_SERVO";
    if(type == MOTOR_SMART_PROFESSIONAL_MOTOR) return "SMART_PROFESSIONAL_MOTOR";
    return "UNKNOWN";
}


// ------------------------------------------------------------
// FUNCTION: flushSmartMotorInput()
// ------------------------------------------------------------
// Clears old bytes from the smart-motor UART receive buffer before sending
// a new ping. This prevents stale bytes from a previous test from being
// mistaken as a new smart-servo reply.
// ------------------------------------------------------------
void flushSmartMotorInput()
{
    while(SmartMotorSerial.available())
    {
        SmartMotorSerial.read();
    }
}


// ------------------------------------------------------------
// FUNCTION: pingSmartMotorId()
// ------------------------------------------------------------
// Attempts to detect a smart/professional servo by sending a ping to one ID.
//
// Current implementation:
//   Feetech/STS/SCS-style packet:
//     0xFF 0xFF ID LENGTH INSTRUCTION CHECKSUM
//
// Ping packet:
//     Header      = 0xFF 0xFF
//     ID          = motor ID to test
//     LENGTH      = 0x02
//     INSTRUCTION = 0x01  (PING)
//     CHECKSUM    = bitwise NOT of ID + LENGTH + INSTRUCTION
//
// Expected behavior:
//   If a smart servo with that ID exists and uses this protocol, it replies
//   with a packet beginning with:
//     0xFF 0xFF ID ...
//
// Return value:
//   true  = smart servo responded
//   false = no response before timeout
//
// If your smart motor uses another protocol, replace this function only.
// The rest of the auto-detection framework can stay the same.
// ------------------------------------------------------------
bool pingSmartMotorId(int smartId)
{
    if(smartId < 1 || smartId > 253)
    {
        return false;
    }

    flushSmartMotorInput();

    uint8_t id = (uint8_t)smartId;
    uint8_t length = 0x02;
    uint8_t instruction = 0x01;
    uint8_t checksum = (uint8_t)(~(id + length + instruction));

    SmartMotorSerial.write(0xFF);
    SmartMotorSerial.write(0xFF);
    SmartMotorSerial.write(id);
    SmartMotorSerial.write(length);
    SmartMotorSerial.write(instruction);
    SmartMotorSerial.write(checksum);
    SmartMotorSerial.flush();

    unsigned long startMs = millis();
    int state = 0;

    while((millis() - startMs) < SMART_PING_TIMEOUT_MS)
    {
        while(SmartMotorSerial.available())
        {
            uint8_t b = SmartMotorSerial.read();

            // Look for response header: FF FF ID
            if(state == 0)
            {
                if(b == 0xFF) state = 1;
            }
            else if(state == 1)
            {
                if(b == 0xFF) state = 2;
                else state = 0;
            }
            else if(state == 2)
            {
                if(b == id)
                {
                    return true;
                }
                state = 0;
            }
        }
    }

    return false;
}


// ------------------------------------------------------------
// FUNCTION: applyOneTypeToAllMotors()
// ------------------------------------------------------------
// Applies the detected type to every motor because the requested design is:
//   "Detect the first motor type and make the rest the same."
//
// For hobby motors:
//   outputPin is assigned from DIRECT_ESP32_GPIO_PIN[].
//   id is kept as index + 1 only for software tracking.
//
// For smart motors:
//   outputPin is set to -1 because smart servos share the UART bus.
//   id is assigned as index + 1 so motor 0 -> ID 1, motor 1 -> ID 2, etc.
// ------------------------------------------------------------
void applyOneTypeToAllMotors(MotorType detectedType)
{
    // Save the final detected type globally.
    // All motor commands use motors[i].type after this point, so storing
    // the detection result here guarantees that the correct related
    // hobby-servo or smart-motor function is used by the dispatcher.
    g_detectedMotorType = detectedType;

    for(int i = 0; i < MOTOR_COUNT; i++)
    {
        motors[i].type = detectedType;

        if(detectedType == MOTOR_SMART_PROFESSIONAL_MOTOR)
        {
            motors[i].outputPin = -1;
            motors[i].id = i + 1;
        }
        else
        {
            motors[i].outputPin = DIRECT_ESP32_GPIO_PIN[i];
            motors[i].id = i + 1;
        }

        motors[i].enabled = false;
        motors[i].direction = DIR_STOP;
    }
}


// ------------------------------------------------------------
// FUNCTION: autoDetectFirstMotorTypeAndApplyToAll()
// ------------------------------------------------------------
// Performs the actual first-motor detection and applies the result to all
// motors. The result is printed in a Python-friendly format.
//
// Output examples:
//   AUTO_DETECT_BEGIN
//   FIRST_MOTOR_TEST_ID=1
//   FIRST_MOTOR_DETECTED_TYPE=SMART_PROFESSIONAL_MOTOR
//   AUTO_DETECT_APPLIED_TO_ALL=YES
//   AUTO_DETECT_END
//
// or:
//
//   FIRST_MOTOR_DETECTED_TYPE=HOBBY_POSITIONAL_SERVO
//
// This function can be called:
//   - automatically during setup()
//   - manually from Python using: @AUTO_DETECT
// ------------------------------------------------------------
void autoDetectFirstMotorTypeAndApplyToAll(Print &out)
{
    out.println("AUTO_DETECT_BEGIN");
    out.print("FIRST_MOTOR_TEST_ID=");
    out.println(AUTO_DETECT_FIRST_SMART_ID);

    bool smartDetected = pingSmartMotorId(AUTO_DETECT_FIRST_SMART_ID);

    MotorType detectedType = AUTO_DETECT_FALLBACK_HOBBY_TYPE;

    if(smartDetected)
    {
        detectedType = MOTOR_SMART_PROFESSIONAL_MOTOR;
    }

    applyOneTypeToAllMotors(detectedType);

    out.print("FIRST_MOTOR_DETECTED_TYPE=");
    out.println(motorTypeName(detectedType));

    out.print("CONTROL_FUNCTION_PATH=");
    if(detectedType == MOTOR_HOBBY_POSITIONAL_SERVO)
    {
        out.println("driveHobbyPositionalServo");
    }
    else if(detectedType == MOTOR_HOBBY_CONTINUOUS_SERVO)
    {
        out.println("driveHobbyContinuousServo");
    }
    else
    {
        out.println("driveSmartProfessionalMotor");
    }

    out.print("AUTO_DETECT_APPLIED_TO_ALL=");
    out.println("YES");

    out.print("MOTOR_COUNT=");
    out.println(MOTOR_COUNT);

    out.println("AUTO_DETECT_END");
}


// ============================================================
// AUTO ENABLE MOTOR
// ============================================================
//
// There is no public enable function.
// Every control function calls autoEnableMotor() internally.
//
// For hobby servos:
//   attach() is called automatically.
//
// For smart/professional motors:
//   enabled flag is set, but real startup protocol can be added.
//
// ============================================================


// ------------------------------------------------------------
// FUNCTION: autoEnableMotor()
// ------------------------------------------------------------
// Automatically enables a motor the first time it is used. For hobby servos it attaches the ESP32Servo object to the configured GPIO pin. For smart motors it marks the motor as enabled so a real bus protocol can be added later.
// ------------------------------------------------------------
void autoEnableMotor(int motorIndex)
{
    if(!isValidMotorIndex(motorIndex)) return;
    if(motors[motorIndex].enabled) return;

    if(
        motors[motorIndex].type == MOTOR_HOBBY_POSITIONAL_SERVO ||
        motors[motorIndex].type == MOTOR_HOBBY_CONTINUOUS_SERVO
    )
    {
        hobbyServos[motorIndex].attach(motors[motorIndex].outputPin);

        motors[motorIndex].enabled = true;

        Serial.print("[AUTO ENABLE] Hobby servo motor ");
        Serial.print(motorIndex);
        Serial.print(" attached to ESP32 GPIO ");
        Serial.println(motors[motorIndex].outputPin);
    }
    else
    {
        motors[motorIndex].enabled = true;

        Serial.print("[AUTO ENABLE] Smart/professional motor ID ");
        Serial.println(motors[motorIndex].id);
    }
}

// ============================================================
// HOBBY SERVO WRITE
// ============================================================


// ------------------------------------------------------------
// FUNCTION: writeHobbyServoValue()
// ------------------------------------------------------------
// Low-level helper for hobby servos. It attaches the motor if needed, clamps the output to 0..180, and writes the PWM servo value.
// ------------------------------------------------------------
void writeHobbyServoValue(int motorIndex, int value)
{
    if(!isValidMotorIndex(motorIndex)) return;

    autoEnableMotor(motorIndex);

    value = constrain(value, 0, 180);

    hobbyServos[motorIndex].write(value);
}

// ============================================================
// LOW-LEVEL DRIVER: HOBBY POSITIONAL SERVO
// ============================================================
//
// angleDeg:
//   real target angle, 0 to 180.
//
// speed:
//   0 to 100.
//   Higher value means faster simulated motion.
//
// torque:
//   stored for software consistency.
//   Hobby PWM servos do not accept real torque commands.
//
// ============================================================


// ------------------------------------------------------------
// FUNCTION: driveHobbyPositionalServo()
// ------------------------------------------------------------
// Moves a normal 0..180 degree hobby servo toward a target angle. It simulates speed by stepping through intermediate angles with a delay based on the requested speed.
// ------------------------------------------------------------
void driveHobbyPositionalServo(
    int motorIndex,
    float angleDeg,
    float speed,
    float torque
)
{
    if(!isValidMotorIndex(motorIndex)) return;

    autoEnableMotor(motorIndex);

    angleDeg = clampFloat(angleDeg, MIN_ANGLE_DEG, MAX_ANGLE_DEG);
    speed = clampFloat(speed, MIN_SPEED, MAX_SPEED);
    torque = clampFloat(torque, MIN_TORQUE, MAX_TORQUE);

    int currentAngle = (int)motors[motorIndex].angleDeg;
    int targetAngle = (int)angleDeg;

    int stepDelayMs = map((int)speed, 0, 100, 30, 1);

    if(currentAngle < targetAngle)
    {
        for(int a = currentAngle; a <= targetAngle; a++)
        {
            hobbyServos[motorIndex].write(a);
            delay(stepDelayMs);
        }
    }
    else
    {
        for(int a = currentAngle; a >= targetAngle; a--)
        {
            hobbyServos[motorIndex].write(a);
            delay(stepDelayMs);
        }
    }

    motors[motorIndex].angleDeg = angleDeg;
    motors[motorIndex].speed = speed;
    motors[motorIndex].torque = torque;

    Serial.print("[POSITIONAL SERVO] motor=");
    Serial.print(motorIndex);
    Serial.print(" angle=");
    Serial.print(angleDeg);
    Serial.print(" speed=");
    Serial.print(speed);
    Serial.print(" torque=");
    Serial.print(torque);
    Serial.println(" torque_not_supported_for_hobby_pwm");
}

// ============================================================
// LOW-LEVEL DRIVER: HOBBY CONTINUOUS ROTATION SERVO
// ============================================================
//
// Continuous servo rule:
//
//   90  = stop
//   >90 = clockwise
//   <90 = counter-clockwise
//
// speed:
//   0 to 100.
//
// torque:
//   stored only.
//   Hobby continuous servos do not accept real torque commands.
//
// ============================================================


// ------------------------------------------------------------
// FUNCTION: driveHobbyContinuousServo()
// ------------------------------------------------------------
// Controls a continuous-rotation servo. For these servos, 90 means stop, values above 90 rotate one direction, and values below 90 rotate the opposite direction.
// ------------------------------------------------------------
void driveHobbyContinuousServo(
    int motorIndex,
    MotorDirection direction,
    float speed,
    float torque
)
{
    if(!isValidMotorIndex(motorIndex)) return;

    autoEnableMotor(motorIndex);

    speed = clampFloat(speed, MIN_SPEED, MAX_SPEED);
    torque = clampFloat(torque, MIN_TORQUE, MAX_TORQUE);

    int offset = map((int)speed, 0, 100, 0, 90);
    int pwmValue = CONTINUOUS_STOP_VALUE;

    if(direction == DIR_CW)
    {
        pwmValue = CONTINUOUS_STOP_VALUE + offset;
    }
    else if(direction == DIR_CCW)
    {
        pwmValue = CONTINUOUS_STOP_VALUE - offset;
    }
    else
    {
        pwmValue = CONTINUOUS_STOP_VALUE;
    }

    pwmValue = constrain(pwmValue, 0, 180);

    hobbyServos[motorIndex].write(pwmValue);

    motors[motorIndex].speed = speed;
    motors[motorIndex].torque = torque;
    motors[motorIndex].direction = direction;

    Serial.print("[CONTINUOUS SERVO] motor=");
    Serial.print(motorIndex);
    Serial.print(" value=");
    Serial.print(pwmValue);
    Serial.print(" speed=");
    Serial.print(speed);
    Serial.print(" direction=");
    Serial.println(direction);
}

// ============================================================
// LOW-LEVEL DRIVER: SMART / PROFESSIONAL MOTOR PLACEHOLDER
// ============================================================
//
// Replace this function with your real motor protocol.
//
// STS3215 concept:
//   send motor ID, position, speed, acceleration.
//
// Dynamixel concept:
//   set goal position, goal velocity, current limit.
//
// Unitree/CAN actuator concept:
//   send position, velocity, torque, Kp, Kd.
//
// This placeholder only prints the command.
//
// ============================================================


// ------------------------------------------------------------
// FUNCTION: sendSmartMotorCommand()
// ------------------------------------------------------------
// Placeholder/adapter for smart or professional motors. This is where UART, CAN, or RS485 commands would be sent to real industrial or smart servo drivers.
// ------------------------------------------------------------
void sendSmartMotorCommand(
    int motorId,
    float angleDeg,
    float speed,
    float torque,
    MotorDirection direction
)
{
    Serial.print("[SMART MOTOR COMMAND] id=");
    Serial.print(motorId);
    Serial.print(" angleDeg=");
    Serial.print(angleDeg);
    Serial.print(" speed=");
    Serial.print(speed);
    Serial.print(" torque=");
    Serial.print(torque);
    Serial.print(" direction=");
    Serial.println(direction);

    // TODO:
    // Add real UART, RS485, or CAN motor command here.
}


// ------------------------------------------------------------
// FUNCTION: driveSmartProfessionalMotor()
// ------------------------------------------------------------
// Stores and forwards a command for a smart/professional motor. It clamps angle, speed, and torque, updates the software status table, and calls sendSmartMotorCommand().
// ------------------------------------------------------------
void driveSmartProfessionalMotor(
    int motorIndex,
    float angleDeg,
    float speed,
    float torque,
    MotorDirection direction
)
{
    if(!isValidMotorIndex(motorIndex)) return;

    autoEnableMotor(motorIndex);

    angleDeg = clampFloat(angleDeg, MIN_ANGLE_DEG, MAX_ANGLE_DEG);
    speed = clampFloat(speed, MIN_SPEED, MAX_SPEED);
    torque = clampFloat(torque, MIN_TORQUE, MAX_TORQUE);

    motors[motorIndex].angleDeg = angleDeg;
    motors[motorIndex].speed = speed;
    motors[motorIndex].torque = torque;
    motors[motorIndex].direction = direction;

    sendSmartMotorCommand(
        motors[motorIndex].id,
        angleDeg,
        speed,
        torque,
        direction
    );
}

// ============================================================
// MAIN GENERIC MOTOR API
// ============================================================


// ------------------------------------------------------------
// FUNCTION: setMotorCommand()
// ------------------------------------------------------------
// Main motor dispatcher. It checks motors[motorIndex].type, which is set during setup() by autoDetectFirstMotorTypeAndApplyToAll(), and routes the command to the correct related low-level function: driveHobbyPositionalServo(), driveHobbyContinuousServo(), or driveSmartProfessionalMotor().
// ------------------------------------------------------------
void setMotorCommand(
    int motorIndex,
    float angleDeg,
    float speed,
    float torque,
    MotorDirection direction
)
{
    if(!isValidMotorIndex(motorIndex)) return;

    if(motors[motorIndex].type == MOTOR_HOBBY_POSITIONAL_SERVO)
    {
        driveHobbyPositionalServo(motorIndex, angleDeg, speed, torque);
    }
    else if(motors[motorIndex].type == MOTOR_HOBBY_CONTINUOUS_SERVO)
    {
        driveHobbyContinuousServo(motorIndex, direction, speed, torque);
    }
    else
    {
        driveSmartProfessionalMotor(motorIndex, angleDeg, speed, torque, direction);
    }
}


// ------------------------------------------------------------
// FUNCTION: setMotorAngleSpeedTorque()
// ------------------------------------------------------------
// Convenience wrapper used by Python commands and motion frames to set angle, speed, and torque in one call.
// ------------------------------------------------------------
void setMotorAngleSpeedTorque(
    int motorIndex,
    float angleDeg,
    float speed,
    float torque
)
{
    if(!isValidMotorIndex(motorIndex)) return;

    MotorDirection direction = DIR_STOP;

    if(angleDeg > motors[motorIndex].angleDeg) direction = DIR_CW;
    else if(angleDeg < motors[motorIndex].angleDeg) direction = DIR_CCW;

    setMotorCommand(
        motorIndex,
        angleDeg,
        speed,
        torque,
        direction
    );
}


// ------------------------------------------------------------
// FUNCTION: setMotorAngle()
// ------------------------------------------------------------
// Changes only the target angle while reusing the motor's current speed and torque values.
// ------------------------------------------------------------
void setMotorAngle(int motorIndex, float angleDeg)
{
    if(!isValidMotorIndex(motorIndex)) return;

    setMotorAngleSpeedTorque(
        motorIndex,
        angleDeg,
        motors[motorIndex].speed,
        motors[motorIndex].torque
    );
}


// ------------------------------------------------------------
// FUNCTION: moveMotorCW()
// ------------------------------------------------------------
// Commands one motor to move clockwise using the requested speed and torque. For positional servos, this maps to a high target angle; for continuous servos it maps to clockwise rotation.
// ------------------------------------------------------------
void moveMotorCW(int motorIndex, float speed, float torque)
{
    if(!isValidMotorIndex(motorIndex)) return;

    setMotorCommand(
        motorIndex,
        motors[motorIndex].angleDeg,
        speed,
        torque,
        DIR_CW
    );
}


// ------------------------------------------------------------
// FUNCTION: moveMotorCCW()
// ------------------------------------------------------------
// Commands one motor to move counter-clockwise using the requested speed and torque. For positional servos, this maps to a low target angle; for continuous servos it maps to counter-clockwise rotation.
// ------------------------------------------------------------
void moveMotorCCW(int motorIndex, float speed, float torque)
{
    if(!isValidMotorIndex(motorIndex)) return;

    setMotorCommand(
        motorIndex,
        motors[motorIndex].angleDeg,
        speed,
        torque,
        DIR_CCW
    );
}


// ------------------------------------------------------------
// FUNCTION: stopMotor()
// ------------------------------------------------------------
// Stops one motor safely. For hobby servos it writes the neutral/stop value. For smart motors it sends a zero-speed stop command through the smart motor adapter.
// ------------------------------------------------------------
void stopMotor(int motorIndex)
{
    if(!isValidMotorIndex(motorIndex)) return;

    if(motors[motorIndex].type == MOTOR_HOBBY_CONTINUOUS_SERVO)
    {
        driveHobbyContinuousServo(
            motorIndex,
            DIR_STOP,
            0,
            motors[motorIndex].torque
        );
    }
    else
    {
        setMotorCommand(
            motorIndex,
            motors[motorIndex].angleDeg,
            0,
            motors[motorIndex].torque,
            DIR_STOP
        );
    }

    Serial.print("[STOP] motor=");
    Serial.println(motorIndex);
}


// ------------------------------------------------------------
// FUNCTION: increaseMotorSpeed()
// ------------------------------------------------------------
// Increases the stored speed value of one motor by a fixed step, clamps it to the safe range, and reports the new value on Serial.
// ------------------------------------------------------------
void increaseMotorSpeed(int motorIndex)
{
    if(!isValidMotorIndex(motorIndex)) return;

    float newSpeed = clampFloat(
        motors[motorIndex].speed + SPEED_STEP,
        MIN_SPEED,
        MAX_SPEED
    );

    setMotorCommand(
        motorIndex,
        motors[motorIndex].angleDeg,
        newSpeed,
        motors[motorIndex].torque,
        motors[motorIndex].direction
    );
}


// ------------------------------------------------------------
// FUNCTION: decreaseMotorSpeed()
// ------------------------------------------------------------
// Decreases the stored speed value of one motor by a fixed step, clamps it to the safe range, and reports the new value on Serial.
// ------------------------------------------------------------
void decreaseMotorSpeed(int motorIndex)
{
    if(!isValidMotorIndex(motorIndex)) return;

    float newSpeed = clampFloat(
        motors[motorIndex].speed - SPEED_STEP,
        MIN_SPEED,
        MAX_SPEED
    );

    setMotorCommand(
        motorIndex,
        motors[motorIndex].angleDeg,
        newSpeed,
        motors[motorIndex].torque,
        motors[motorIndex].direction
    );
}


// ------------------------------------------------------------
// FUNCTION: increaseMotorTorque()
// ------------------------------------------------------------
// Increases the stored torque value of one motor by a fixed step. Hobby servos do not use real torque, but the value is kept for status and smart-motor expansion.
// ------------------------------------------------------------
void increaseMotorTorque(int motorIndex)
{
    if(!isValidMotorIndex(motorIndex)) return;

    float newTorque = clampFloat(
        motors[motorIndex].torque + TORQUE_STEP,
        MIN_TORQUE,
        MAX_TORQUE
    );

    setMotorCommand(
        motorIndex,
        motors[motorIndex].angleDeg,
        motors[motorIndex].speed,
        newTorque,
        motors[motorIndex].direction
    );
}


// ------------------------------------------------------------
// FUNCTION: decreaseMotorTorque()
// ------------------------------------------------------------
// Decreases the stored torque value of one motor by a fixed step. Hobby servos do not use real torque, but the value is kept for status and smart-motor expansion.
// ------------------------------------------------------------
void decreaseMotorTorque(int motorIndex)
{
    if(!isValidMotorIndex(motorIndex)) return;

    float newTorque = clampFloat(
        motors[motorIndex].torque - TORQUE_STEP,
        MIN_TORQUE,
        MAX_TORQUE
    );

    setMotorCommand(
        motorIndex,
        motors[motorIndex].angleDeg,
        motors[motorIndex].speed,
        newTorque,
        motors[motorIndex].direction
    );
}


// ------------------------------------------------------------
// FUNCTION: stopAllMotors()
// ------------------------------------------------------------
// Stops every configured motor by calling stopMotor() for each motor index.
// ------------------------------------------------------------
void stopAllMotors()
{
    for(int i = 0; i < MOTOR_COUNT; i++)
    {
        stopMotor(i);
    }
}

// ============================================================
// GENERIC MOTION TABLE
// ============================================================
//
// This is application-neutral.
//
// What is MotionFrame?
// ------------------------------------------------------------
// MotionFrame is one time-step in a motion sequence.
// Think of it like one row in a motion table. Each row tells the ESP32:
//   1) what angle each motor should move to,
//   2) what speed should be used for this frame,
//   3) what torque value should be stored/sent for this frame, and
//   4) how long the firmware should wait before moving to the next frame.
//
// Example:
//   Frame 1: all motors at 90 degrees, wait 500 ms
//   Frame 2: motor 0 to 110 degrees, motor 1 to 70 degrees, wait 400 ms
//   Frame 3: return all motors to 90 degrees
//
// For a robot dog, each MotionFrame can represent one pose in a walking gait.
// For a robot arm, each MotionFrame can represent one arm position.
// For a pan/tilt camera, each MotionFrame can represent one camera direction.
//
// Field description:
//   angle[MOTOR_COUNT]
//     Target angle for every motor index. The array has MOTOR_COUNT entries,
//     so angle[0] belongs to motor 0, angle[1] belongs to motor 1, and so on.
//
//   speed
//     Common motion speed for all motors in this frame. In this generic
//     firmware it is treated as a 0-100 software speed value.
//
//   torque
//     Common torque value for all motors in this frame. Hobby PWM servos do
//     not accept real torque commands, so for hobby servos this is stored for
//     status/reporting. Smart/professional motor drivers can later use it.
//
//   durationMs
//     How long to wait after applying this frame before the next frame starts.
//
// You can use it for arms, legs, grippers, pan/tilt,
// wheel mechanisms, or any repeated actuator motion.
// ============================================================

struct MotionFrame
{
    // Target angle for every motor in this frame.
    float angle[MOTOR_COUNT];

    // Common speed value used by the frame. Range is normally 0 to 100.
    float speed;

    // Common torque value used by the frame. Range is normally 0 to 100.
    float torque;

    // Delay after executing this frame, in milliseconds.
    int durationMs;
};

MotionFrame exampleMotion[] =
{
    {{90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90}, 50, 20, 500},
    {{110,70,110,70,90,90,90,90,90,90,90,90,90,90,90,90}, 60, 25, 400},
    {{90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90}, 50, 20, 300},
    {{70,110,70,110,90,90,90,90,90,90,90,90,90,90,90,90}, 60, 25, 400}
};


// ------------------------------------------------------------
// FUNCTION: runMotionFrame()
// ------------------------------------------------------------
// Executes one MotionFrame. It sends the frame angle, speed, and torque to every motor, then waits for frame.durationMs before returning.
// ------------------------------------------------------------
void runMotionFrame(MotionFrame frame)
{
    for(int i = 0; i < MOTOR_COUNT; i++)
    {
        setMotorAngleSpeedTorque(
            i,
            frame.angle[i],
            frame.speed,
            frame.torque
        );
    }

    delay(frame.durationMs);
}


// ------------------------------------------------------------
// FUNCTION: runExampleMotionOnce()
// ------------------------------------------------------------
// Plays the exampleMotion[] table one time from first frame to last frame. This demonstrates how a sequence of MotionFrame rows can produce a complete motion.
// ------------------------------------------------------------
void runExampleMotionOnce()
{
    int frameCount = sizeof(exampleMotion) / sizeof(exampleMotion[0]);

    for(int i = 0; i < frameCount; i++)
    {
        runMotionFrame(exampleMotion[i]);
    }
}


// ============================================================
// WIFI STATUS / TCP SERVER FUNCTIONS
// ============================================================


// ------------------------------------------------------------
// FUNCTION: wifiCredentialsConfigured()
// ------------------------------------------------------------
// Returns true only when WIFI_SSID was changed from the placeholder value. This prevents the ESP32 from wasting time trying to connect with dummy credentials.
// ------------------------------------------------------------
bool wifiCredentialsConfigured()
{
    return String(WIFI_SSID) != "PUT_YOUR_WIFI_NAME_HERE" && String(WIFI_SSID).length() > 0;
}


// ------------------------------------------------------------
// FUNCTION: printWifiStatusTo()
// ------------------------------------------------------------
// Prints Wi-Fi status to any Arduino Print output. The output can be Serial or a Wi-Fi TCP client, so both USB Python and Wi-Fi Python receive the same IP/status format.
// ------------------------------------------------------------
void printWifiStatusTo(Print &out)
{
    out.println("WIFI_STATUS_BEGIN");
    out.print("WIFI_CONNECTED=");
    out.println(WiFi.status() == WL_CONNECTED ? "YES" : "NO");
    out.print("WIFI_SSID=");
    out.println(WIFI_SSID);
    out.print("HOSTNAME=");
    out.print(WIFI_HOSTNAME);
    out.println(".local");
    out.print("PORT=");
    out.println(WIFI_MOTOR_PORT);

    if(WiFi.status() == WL_CONNECTED)
    {
        out.print("IP_ADDRESS=");
        out.println(WiFi.localIP());
        out.print("MAC_ADDRESS=");
        out.println(WiFi.macAddress());
        out.print("WIFI_RSSI=");
        out.println(WiFi.RSSI());
    }
    else
    {
        out.println("IP_ADDRESS=NOT_CONNECTED");
    }

    out.println("WIFI_STATUS_END");
}


// ------------------------------------------------------------
// FUNCTION: connectWifiAndStartServer()
// ------------------------------------------------------------
// Connects the ESP32 to Wi-Fi, starts the TCP server on WIFI_MOTOR_PORT, starts mDNS when available, and prints the IP information to Serial.
// ------------------------------------------------------------
void connectWifiAndStartServer()
{
    Serial.println();
    Serial.println("================================================");
    Serial.println("WIFI STARTUP");
    Serial.println("================================================");

    if(!wifiCredentialsConfigured())
    {
        Serial.println("Wi-Fi credentials are not configured.");
        Serial.println("Edit WIFI_SSID and WIFI_PASSWORD in this file.");
        Serial.println("USB Serial control will still work.");
        Serial.println("================================================");
        return;
    }

    WiFi.mode(WIFI_STA);
    WiFi.setHostname(WIFI_HOSTNAME);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    Serial.print("Connecting to Wi-Fi");

    unsigned long startMs = millis();
    while(WiFi.status() != WL_CONNECTED && (millis() - startMs) < 15000)
    {
        delay(500);
        Serial.print(".");
    }

    Serial.println();

    if(WiFi.status() == WL_CONNECTED)
    {
        wifiConnected = true;
        motorTcpServer.begin();
        motorTcpServerStarted = true;

        if(MDNS.begin(WIFI_HOSTNAME))
        {
            MDNS.addService("esp32motor", "tcp", WIFI_MOTOR_PORT);
            Serial.println("mDNS started.");
        }
        else
        {
            Serial.println("mDNS failed to start. IP address can still be used.");
        }

        Serial.println("Wi-Fi connected successfully.");
        printWifiStatusTo(Serial);
    }
    else
    {
        wifiConnected = false;
        motorTcpServerStarted = false;
        Serial.println("Wi-Fi connection failed or timed out.");
        Serial.println("USB Serial control will still work.");
    }

    Serial.println("================================================");
}


// ------------------------------------------------------------
// FUNCTION: maintainWifiClient()
// ------------------------------------------------------------
// Keeps the TCP server ready for Python. If no client is connected, it checks for a new Python TCP connection and sends a welcome/status message.
// ------------------------------------------------------------
void maintainWifiClient()
{
    if(!motorTcpServerStarted)
    {
        return;
    }

    if(!motorTcpClient || !motorTcpClient.connected())
    {
        WiFiClient newClient = motorTcpServer.available();
        if(newClient)
        {
            motorTcpClient = newClient;
            motorTcpClient.println("ESP32_MOTOR_CONTROLLER_CONNECTED");
            printWifiStatusTo(motorTcpClient);
        }
    }
}


// ------------------------------------------------------------
// FUNCTION: sendMotorStatusLineTo()
// ------------------------------------------------------------
// Sends one Python-friendly MOTOR_STATUS line for one motor. Python parses this line to update its live motor table.
// ------------------------------------------------------------
void sendMotorStatusLineTo(Print &out, int motorIndex)
{
    if(motorIndex < 0 || motorIndex >= MOTOR_COUNT)
    {
        out.println("ERROR=BAD_MOTOR_INDEX");
        return;
    }

    out.print("MOTOR_STATUS,index=");
    out.print(motorIndex);
    out.print(",type=");
    out.print(motors[motorIndex].type);
    out.print(",type_name=");
    out.print(motorTypeName(motors[motorIndex].type));
    out.print(",gpio=");
    out.print(motors[motorIndex].outputPin);
    out.print(",id=");
    out.print(motors[motorIndex].id);
    out.print(",angle=");
    out.print(motors[motorIndex].angleDeg);
    out.print(",speed=");
    out.print(motors[motorIndex].speed);
    out.print(",torque=");
    out.print(motors[motorIndex].torque);
    out.print(",direction=");
    out.print(motors[motorIndex].direction);
    out.print(",enabled=");
    out.println(motors[motorIndex].enabled ? 1 : 0);
}


// ------------------------------------------------------------
// FUNCTION: sendAllMotorStatusLinesTo()
// ------------------------------------------------------------
// Sends all motor statuses between MOTOR_STATUS_BEGIN and MOTOR_STATUS_END markers.
// ------------------------------------------------------------
void sendAllMotorStatusLinesTo(Print &out)
{
    out.println("MOTOR_STATUS_BEGIN");
    for(int i = 0; i < MOTOR_COUNT; i++)
    {
        sendMotorStatusLineTo(out, i);
    }
    out.println("MOTOR_STATUS_END");
}


// ------------------------------------------------------------
// FUNCTION: sendCommandHelpTo()
// ------------------------------------------------------------
// Sends the list of supported @ commands to Serial or the TCP client.
// ------------------------------------------------------------
void sendCommandHelpTo(Print &out)
{
    out.println("COMMAND_HELP_BEGIN");
    out.println("@IP                  : return Wi-Fi IP, hostname, port, RSSI");
    out.println("@WIFI                : same as @IP");
    out.println("@STATUS_ALL         : return all motor status lines");
    out.println("@STATUS n           : return one motor status line");
    out.println("@SPEED_UP n step    : increase motor speed by step");
    out.println("@SPEED_DOWN n step  : decrease motor speed by step");
    out.println("@STOP n             : stop one motor");
    out.println("@STOP_ALL           : stop all motors");
    out.println("@PERIODIC_ON        : send all motor status every second");
    out.println("@PERIODIC_OFF       : stop periodic status");
    out.println("@AUTO_DETECT        : check first motor type and apply same type to all motors");
    out.println("@TYPE               : report detected type and active driver function");
    out.println("@HELP               : show this help");
    out.println("COMMAND_HELP_END");
}


// ------------------------------------------------------------
// FUNCTION: handleLineCommand()
// ------------------------------------------------------------
// Parses Python-friendly text commands that start with @. The same parser is used for USB Serial and Wi-Fi TCP commands.
// ------------------------------------------------------------
void handleLineCommand(String line, Print &out)
{
    line.trim();
    if(line.length() == 0)
    {
        return;
    }

    if(line == "@IP" || line == "@WIFI")
    {
        printWifiStatusTo(out);
    }
    else if(line == "@STATUS_ALL")
    {
        sendAllMotorStatusLinesTo(out);
    }
    else if(line.startsWith("@STATUS "))
    {
        int motorIndex = line.substring(8).toInt();
        sendMotorStatusLineTo(out, motorIndex);
    }
    else if(line.startsWith("@SPEED_UP "))
    {
        int motorIndex = -1;
        float step = 5.0;
        sscanf(line.c_str(), "@SPEED_UP %d %f", &motorIndex, &step);
        if(motorIndex >= 0 && motorIndex < MOTOR_COUNT)
        {
            // Do not only update the stored speed value. Route through
            // setMotorCommand(), so the detected motor type chooses the
            // correct related function:
            //   hobby positional  -> driveHobbyPositionalServo()
            //   hobby continuous  -> driveHobbyContinuousServo()
            //   smart/professional-> driveSmartProfessionalMotor()
            float newSpeed = clampFloat(
                motors[motorIndex].speed + step,
                MIN_SPEED,
                MAX_SPEED
            );

            setMotorCommand(
                motorIndex,
                motors[motorIndex].angleDeg,
                newSpeed,
                motors[motorIndex].torque,
                motors[motorIndex].direction
            );

            out.println("ACK=SPEED_UP");
            sendMotorStatusLineTo(out, motorIndex);
        }
        else
        {
            out.println("ERROR=BAD_MOTOR_INDEX");
        }
    }
    else if(line.startsWith("@SPEED_DOWN "))
    {
        int motorIndex = -1;
        float step = 5.0;
        sscanf(line.c_str(), "@SPEED_DOWN %d %f", &motorIndex, &step);
        if(motorIndex >= 0 && motorIndex < MOTOR_COUNT)
        {
            // Route through setMotorCommand() for the same reason as
            // @SPEED_UP: the detected type must control which driver runs.
            float newSpeed = clampFloat(
                motors[motorIndex].speed - step,
                MIN_SPEED,
                MAX_SPEED
            );

            setMotorCommand(
                motorIndex,
                motors[motorIndex].angleDeg,
                newSpeed,
                motors[motorIndex].torque,
                motors[motorIndex].direction
            );

            out.println("ACK=SPEED_DOWN");
            sendMotorStatusLineTo(out, motorIndex);
        }
        else
        {
            out.println("ERROR=BAD_MOTOR_INDEX");
        }
    }
    else if(line.startsWith("@STOP "))
    {
        int motorIndex = line.substring(6).toInt();
        stopMotor(motorIndex);
        out.println("ACK=STOP");
        sendMotorStatusLineTo(out, motorIndex);
    }
    else if(line == "@STOP_ALL")
    {
        stopAllMotors();
        out.println("ACK=STOP_ALL");
        sendAllMotorStatusLinesTo(out);
    }
    else if(line == "@PERIODIC_ON")
    {
        periodicMotorStatusEnabled = true;
        out.println("ACK=PERIODIC_ON");
    }
    else if(line == "@PERIODIC_OFF")
    {
        periodicMotorStatusEnabled = false;
        out.println("ACK=PERIODIC_OFF");
    }
    else if(line == "@AUTO_DETECT")
    {
        autoDetectFirstMotorTypeAndApplyToAll(out);
        sendAllMotorStatusLinesTo(out);
    }
    else if(line == "@TYPE")
    {
        out.print("DETECTED_TYPE=");
        out.println(motorTypeName(g_detectedMotorType));

        out.print("CONTROL_FUNCTION_PATH=");
        if(g_detectedMotorType == MOTOR_HOBBY_POSITIONAL_SERVO)
        {
            out.println("driveHobbyPositionalServo");
        }
        else if(g_detectedMotorType == MOTOR_HOBBY_CONTINUOUS_SERVO)
        {
            out.println("driveHobbyContinuousServo");
        }
        else
        {
            out.println("driveSmartProfessionalMotor");
        }
    }
    else if(line == "@HELP")
    {
        sendCommandHelpTo(out);
    }
    else
    {
        out.print("ERROR=UNKNOWN_COMMAND,");
        out.println(line);
    }
}


// ------------------------------------------------------------
// FUNCTION: handleSerialInput()
// ------------------------------------------------------------
// Reads incoming USB Serial data. @ commands are collected until newline and sent to handleLineCommand(); legacy one-character commands are passed to handleSerialCommand().
// ------------------------------------------------------------
void handleSerialInput()
{
    while(Serial.available())
    {
        char cmd = Serial.read();

        // New Python-friendly commands begin with '@' and end with Enter/newline.
        // Example from Python:
        //   @IP
        //   @STATUS_ALL
        //   @SPEED_UP 0 5
        if(cmd == '@' || serialCommandLine.length() > 0)
        {
            if(cmd == '\n' || cmd == '\r')
            {
                handleLineCommand(serialCommandLine, Serial);
                serialCommandLine = "";
            }
            else
            {
                serialCommandLine += cmd;
            }
        }
        else
        {
            // Keep the original one-character serial interface unchanged.
            handleSerialCommand(cmd);
        }
    }
}


// ------------------------------------------------------------
// FUNCTION: handleWifiInput()
// ------------------------------------------------------------
// Reads incoming Wi-Fi TCP data from Python. Commands are collected until newline and then processed by handleLineCommand().
// ------------------------------------------------------------
void handleWifiInput()
{
    if(!motorTcpClient || !motorTcpClient.connected())
    {
        return;
    }

    while(motorTcpClient.available())
    {
        char cmd = motorTcpClient.read();
        if(cmd == '\n' || cmd == '\r')
        {
            handleLineCommand(wifiCommandLine, motorTcpClient);
            wifiCommandLine = "";
        }
        else
        {
            wifiCommandLine += cmd;
        }
    }
}


// ------------------------------------------------------------
// FUNCTION: sendPeriodicMotorStatusIfNeeded()
// ------------------------------------------------------------
// If periodic status is enabled, sends all motor status lines once per configured interval. This lets Python receive live updates without repeatedly polling.
// ------------------------------------------------------------
void sendPeriodicMotorStatusIfNeeded()
{
    if(!periodicMotorStatusEnabled)
    {
        return;
    }

    unsigned long nowMs = millis();
    if(nowMs - lastPeriodicMotorStatusMs >= PERIODIC_MOTOR_STATUS_INTERVAL_MS)
    {
        lastPeriodicMotorStatusMs = nowMs;
        sendAllMotorStatusLinesTo(Serial);
        if(motorTcpClient && motorTcpClient.connected())
        {
            sendAllMotorStatusLinesTo(motorTcpClient);
        }
    }
}

// ============================================================
// STATUS / DEBUG FUNCTIONS
// ============================================================


// ------------------------------------------------------------
// FUNCTION: printConnectionGuide()
// ------------------------------------------------------------
// Prints startup instructions showing how to use USB Serial and Wi-Fi TCP connections from Python or the Arduino Serial Monitor.
// ------------------------------------------------------------
void printConnectionGuide()
{
    Serial.println();
    Serial.println("================================================");
    Serial.println("CONNECTION GUIDE");
    Serial.println("================================================");
    Serial.println("Hobby servo:");
    Serial.println("  Signal -> assigned ESP32 GPIO");
    Serial.println("  Red    -> external 5V/6V");
    Serial.println("  GND    -> external GND");
    Serial.println("  ESP32 GND must connect to external GND");
    Serial.println("  Do NOT power servos from ESP32 3.3V");
    Serial.println();
    Serial.println("Smart motor:");
    Serial.println("  ESP32 GPIO17 TX2 -> smart motor bus input");
    Serial.println("  ESP32 GPIO16 RX2 -> smart motor bus feedback");
    Serial.println("  External power   -> smart motor power");
    Serial.println("  Common GND       -> ESP32 and motor supply");
    Serial.println("================================================");
}


// ------------------------------------------------------------
// FUNCTION: printPinMapping()
// ------------------------------------------------------------
// Prints the configured motor-to-GPIO mapping so the user can verify wiring.
// ------------------------------------------------------------
void printPinMapping()
{
    Serial.println();
    Serial.println("================================================");
    Serial.println("DIRECT ESP32 GPIO MOTOR MAP");
    Serial.println("================================================");

    for(int i = 0; i < MOTOR_COUNT; i++)
    {
        Serial.print("Motor ");
        Serial.print(i);
        Serial.print(" -> ESP32 GPIO ");
        Serial.println(DIRECT_ESP32_GPIO_PIN[i]);
    }

    Serial.println("================================================");
}


// ------------------------------------------------------------
// FUNCTION: printGpioRecommendations()
// ------------------------------------------------------------
// Prints GPIO safety notes and recommendations, including which ESP32 pins are safe, limited, input-only, or reserved for flash/boot functions.
// ------------------------------------------------------------
void printGpioRecommendations()
{
    Serial.println();
    Serial.println("================================================");
    Serial.println("GPIO RECOMMENDATIONS");
    Serial.println("================================================");
    Serial.println("Servo/PWM recommended:");
    Serial.println("4,5,13,14,15,16,17,18,19,21,22,23,25,26,27,32,33");
    Serial.println();
    Serial.println("I2C:");
    Serial.println("21=SDA, 22=SCL");
    Serial.println();
    Serial.println("UART smart motors:");
    Serial.println("17=TX2, 16=RX2");
    Serial.println();
    Serial.println("Avoid direct motor output:");
    Serial.println("0,1,2,3,6,7,8,9,10,11,12,34,35,36,39");
    Serial.println("================================================");
}


// ------------------------------------------------------------
// FUNCTION: printMotorStatus()
// ------------------------------------------------------------
// Prints a human-readable motor status table to Serial for debugging and manual testing.
// ------------------------------------------------------------
void printMotorStatus()
{
    Serial.println();
    Serial.println("================================================");
    Serial.println("MOTOR STATUS");
    Serial.println("================================================");

    for(int i = 0; i < MOTOR_COUNT; i++)
    {
        Serial.print("Motor ");
        Serial.print(i);
        Serial.print(" | Type=");
        Serial.print(motors[i].type);
        Serial.print(" | GPIO=");
        Serial.print(motors[i].outputPin);
        Serial.print(" | ID=");
        Serial.print(motors[i].id);
        Serial.print(" | Angle=");
        Serial.print(motors[i].angleDeg);
        Serial.print(" | Speed=");
        Serial.print(motors[i].speed);
        Serial.print(" | Torque=");
        Serial.print(motors[i].torque);
        Serial.print(" | Direction=");
        Serial.print(motors[i].direction);
        Serial.print(" | Enabled=");
        Serial.println(motors[i].enabled ? "YES" : "NO");
    }

    Serial.println("================================================");
}

// ============================================================
// SERIAL TEST INTERFACE
// ============================================================
//
// 0-9 : select motor 0..9
// A-F : select motor 10..15
//
// q   : increase speed
// a   : decrease speed
//
// e   : increase torque
// d   : decrease torque
//
// c   : move clockwise
// z   : move counter-clockwise
//
// x   : stop active motor
// t   : stop all motors
//
// m   : run example motion once
//
// p   : print motor status
// o   : print pin mapping
// g   : print GPIO recommendations
// h   : print connection guide
//
// Python-friendly line commands, sent over USB Serial or Wi-Fi TCP:
//
// @IP                 : request Wi-Fi IP address, hostname, port, and RSSI
// @WIFI               : same as @IP
// @STATUS_ALL         : request all motor status lines
// @STATUS n           : request status for motor n
// @SPEED_UP n step    : increase speed of motor n by step
// @SPEED_DOWN n step  : decrease speed of motor n by step
// @STOP n             : stop motor n
// @STOP_ALL           : stop all motors
// @PERIODIC_ON        : ESP32 sends motor status every second
// @PERIODIC_OFF       : stop automatic periodic status messages
// @HELP               : print command help
//
// ============================================================

// ------------------------------------------------------------
// ACTIVE MOTOR SELECTION FOR LEGACY SINGLE-KEY SERIAL COMMANDS
// ------------------------------------------------------------
// The original interactive Serial Monitor interface lets the user select a
// motor using one key, then apply commands like speed up/down or stop.
// activeMotor stores the selected motor index for that legacy interface.
// Python @ commands do not need this variable because they include the motor
// index directly, for example: @STOP 0 or @SPEED_UP 2 5.
// ------------------------------------------------------------
int activeMotor = 0;


// ------------------------------------------------------------
// FUNCTION: handleSerialCommand()
// ------------------------------------------------------------
// Processes the original single-character interactive command interface. This is kept for backward compatibility with the original Serial Monitor controls.
// ------------------------------------------------------------
void handleSerialCommand(char cmd)
{
    if(cmd >= '0' && cmd <= '9')
    {
        activeMotor = cmd - '0';
        Serial.print("[ACTIVE MOTOR] ");
        Serial.println(activeMotor);
    }
    else if(cmd >= 'A' && cmd <= 'F')
    {
        activeMotor = 10 + (cmd - 'A');
        Serial.print("[ACTIVE MOTOR] ");
        Serial.println(activeMotor);
    }
    else if(cmd >= 'a' && cmd <= 'f')
    {
        // Lowercase a-f conflicts with command 'a'.
        // Therefore lowercase a-f are not used for selecting 10-15.
        // Use uppercase A-F for motors 10-15.
        if(cmd == 'a')
        {
            decreaseMotorSpeed(activeMotor);
        }
    }
    else if(cmd == 'q')
    {
        increaseMotorSpeed(activeMotor);
    }
    else if(cmd == 'e')
    {
        increaseMotorTorque(activeMotor);
    }
    else if(cmd == 'd')
    {
        decreaseMotorTorque(activeMotor);
    }
    else if(cmd == 'c')
    {
        moveMotorCW(activeMotor, motors[activeMotor].speed, motors[activeMotor].torque);
    }
    else if(cmd == 'z')
    {
        moveMotorCCW(activeMotor, motors[activeMotor].speed, motors[activeMotor].torque);
    }
    else if(cmd == 'x')
    {
        stopMotor(activeMotor);
    }
    else if(cmd == 't')
    {
        stopAllMotors();
    }
    else if(cmd == 'm')
    {
        runExampleMotionOnce();
    }
    else if(cmd == 'p')
    {
        printMotorStatus();
    }
    else if(cmd == 'o')
    {
        printPinMapping();
    }
    else if(cmd == 'g')
    {
        printGpioRecommendations();
    }
    else if(cmd == 'h')
    {
        printConnectionGuide();
    }
}

// ============================================================
// SETUP
// ============================================================


// ------------------------------------------------------------
// FUNCTION: setup()
// ------------------------------------------------------------
// Arduino startup function. Initializes Serial, motor configuration, SmartMotorSerial, runs first-motor auto-detection once, applies the detected type to all motors, starts Wi-Fi/TCP server, and prints the help/status information. After this function completes, all motor control commands use the detected type through setMotorCommand().
// ------------------------------------------------------------
void setup()
{
    Serial.begin(115200);
    delay(1000);

    initializeMotorConfig();

    SmartMotorSerial.begin(
        SMART_UART_BAUD,
        SERIAL_8N1,
        SMART_UART_RX,
        SMART_UART_TX
    );

    if(AUTO_DETECT_FIRST_MOTOR_TYPE_AT_BOOT)
    {
        // Run detection once during boot. The result is stored in
        // g_detectedMotorType and copied to motors[i].type for every motor.
        // From this point forward, setMotorCommand() dispatches to the
        // related driver function for the detected type.
        autoDetectFirstMotorTypeAndApplyToAll(Serial);
    }

    Serial.println();
    Serial.println("================================================");
    Serial.println("ESP32 GENERIC DIRECT SERVO + SMART MOTOR STARTED");
    Serial.println("================================================");

    connectWifiAndStartServer();

    printConnectionGuide();
    printPinMapping();
    printMotorStatus();
}

// ============================================================
// LOOP
// ============================================================


// ------------------------------------------------------------
// FUNCTION: loop()
// ------------------------------------------------------------
// Arduino main loop. Continuously handles USB Serial input, Wi-Fi TCP input, TCP client connection maintenance, and periodic status reporting.
// ------------------------------------------------------------
void loop()
{
    // USB Serial input from Arduino Serial Monitor or Python.
    handleSerialInput();

    // Wi-Fi TCP input from Python when USB is not used.
    maintainWifiClient();
    handleWifiInput();

    // Optional automatic telemetry to Python.
    sendPeriodicMotorStatusIfNeeded();
}
