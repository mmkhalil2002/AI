// ============================================================
// myDroneESP32.ino
// ============================================================
//
// GENERIC ESP32 DRONE FIRMWARE
//
// Commands over USB Serial or Wi-Fi TCP:
//
//   arm
//   disarm
//   takeoff
//   land
//   up
//   down
//   forward
//   backward
//   left
//   right
//   yaw_left
//   yaw_right
//   hover
//   stop
//   throttle 1200
//   status
//
// SAFETY:
//   Test without propellers first.
//   This is a starter framework, not a certified flight controller.
//
// ============================================================

#include <Arduino.h>
#include <WiFi.h>
#include <ESP32Servo.h>

// ============================================================
// GPIO MAPPING
// ============================================================

#define MOTOR_FL_PIN 25
#define MOTOR_FR_PIN 26
#define MOTOR_RL_PIN 27
#define MOTOR_RR_PIN 14

// ============================================================
// ESC PWM RANGE
// ============================================================

#define ESC_MIN_US 1000
#define ESC_IDLE_US 1100
#define ESC_HOVER_US 1250
#define ESC_TAKEOFF_US 1300
#define ESC_MAX_US 1800

#define THROTTLE_STEP_US 40
#define ROLL_STEP_US 60
#define PITCH_STEP_US 60
#define YAW_STEP_US 50

// ============================================================
// WIFI SETTINGS
// ============================================================

#define WIFI_AP_SSID "ESP32_DRONE"
#define WIFI_AP_PASSWORD "12345678"
#define WIFI_TCP_PORT 4210

// ============================================================
// SYSTEM SETTINGS
// ============================================================

#define START_DISARMED 1
#define SERIAL_BAUD 115200
#define CONTROL_LOOP_PERIOD_MS 20

Servo motorFL;
Servo motorFR;
Servo motorRL;
Servo motorRR;

WiFiServer wifiServer(WIFI_TCP_PORT);
WiFiClient wifiClient;

bool droneArmed = false;
int baseThrottleUs = ESC_MIN_US;
unsigned long lastLoopMs = 0;

int safePwm(int valueUs) {
  return constrain(valueUs, ESC_MIN_US, ESC_MAX_US);
}

void writeMotorsRaw(int flUs, int frUs, int rlUs, int rrUs) {
  flUs = safePwm(flUs);
  frUs = safePwm(frUs);
  rlUs = safePwm(rlUs);
  rrUs = safePwm(rrUs);

  if (!droneArmed) {
    flUs = ESC_MIN_US;
    frUs = ESC_MIN_US;
    rlUs = ESC_MIN_US;
    rrUs = ESC_MIN_US;
  }

  motorFL.writeMicroseconds(flUs);
  motorFR.writeMicroseconds(frUs);
  motorRL.writeMicroseconds(rlUs);
  motorRR.writeMicroseconds(rrUs);
}

void writeAllMotors(int valueUs) {
  valueUs = safePwm(valueUs);
  writeMotorsRaw(valueUs, valueUs, valueUs, valueUs);
}

void stopMotors() {
  baseThrottleUs = ESC_MIN_US;
  writeAllMotors(ESC_MIN_US);
}

void armDrone() {
  droneArmed = true;
  baseThrottleUs = ESC_IDLE_US;
  writeAllMotors(baseThrottleUs);
  Serial.println("OK: armed");
}

void disarmDrone() {
  droneArmed = false;
  stopMotors();
  Serial.println("OK: disarmed");
}

void droneHover() {
  if (!droneArmed) {
    Serial.println("ERROR: arm first");
    return;
  }
  baseThrottleUs = ESC_HOVER_US;
  writeAllMotors(baseThrottleUs);
  Serial.println("OK: hover");
}

void droneTakeoff() {
  if (!droneArmed) {
    Serial.println("ERROR: arm first");
    return;
  }
  baseThrottleUs = ESC_TAKEOFF_US;
  writeAllMotors(baseThrottleUs);
  Serial.println("OK: takeoff");
}

void droneLand() {
  if (!droneArmed) {
    Serial.println("ERROR: already disarmed");
    return;
  }

  while (baseThrottleUs > ESC_IDLE_US) {
    baseThrottleUs -= 10;
    writeAllMotors(baseThrottleUs);
    delay(80);
  }

  writeAllMotors(ESC_IDLE_US);
  Serial.println("OK: land");
}

void droneUp() {
  if (!droneArmed) {
    Serial.println("ERROR: arm first");
    return;
  }

  baseThrottleUs += THROTTLE_STEP_US;
  baseThrottleUs = safePwm(baseThrottleUs);
  writeAllMotors(baseThrottleUs);
  Serial.println("OK: up");
}

void droneDown() {
  if (!droneArmed) {
    Serial.println("ERROR: arm first");
    return;
  }

  baseThrottleUs -= THROTTLE_STEP_US;
  baseThrottleUs = safePwm(baseThrottleUs);
  writeAllMotors(baseThrottleUs);
  Serial.println("OK: down");
}

void droneForward() {
  if (!droneArmed) {
    Serial.println("ERROR: arm first");
    return;
  }

  writeMotorsRaw(
    baseThrottleUs - PITCH_STEP_US,
    baseThrottleUs - PITCH_STEP_US,
    baseThrottleUs + PITCH_STEP_US,
    baseThrottleUs + PITCH_STEP_US
  );

  Serial.println("OK: forward");
}

void droneBackward() {
  if (!droneArmed) {
    Serial.println("ERROR: arm first");
    return;
  }

  writeMotorsRaw(
    baseThrottleUs + PITCH_STEP_US,
    baseThrottleUs + PITCH_STEP_US,
    baseThrottleUs - PITCH_STEP_US,
    baseThrottleUs - PITCH_STEP_US
  );

  Serial.println("OK: backward");
}

void droneLeft() {
  if (!droneArmed) {
    Serial.println("ERROR: arm first");
    return;
  }

  writeMotorsRaw(
    baseThrottleUs - ROLL_STEP_US,
    baseThrottleUs + ROLL_STEP_US,
    baseThrottleUs - ROLL_STEP_US,
    baseThrottleUs + ROLL_STEP_US
  );

  Serial.println("OK: left");
}

void droneRight() {
  if (!droneArmed) {
    Serial.println("ERROR: arm first");
    return;
  }

  writeMotorsRaw(
    baseThrottleUs + ROLL_STEP_US,
    baseThrottleUs - ROLL_STEP_US,
    baseThrottleUs + ROLL_STEP_US,
    baseThrottleUs - ROLL_STEP_US
  );

  Serial.println("OK: right");
}

void droneYawLeft() {
  if (!droneArmed) {
    Serial.println("ERROR: arm first");
    return;
  }

  writeMotorsRaw(
    baseThrottleUs - YAW_STEP_US,
    baseThrottleUs + YAW_STEP_US,
    baseThrottleUs + YAW_STEP_US,
    baseThrottleUs - YAW_STEP_US
  );

  Serial.println("OK: yaw_left");
}

void droneYawRight() {
  if (!droneArmed) {
    Serial.println("ERROR: arm first");
    return;
  }

  writeMotorsRaw(
    baseThrottleUs + YAW_STEP_US,
    baseThrottleUs - YAW_STEP_US,
    baseThrottleUs - YAW_STEP_US,
    baseThrottleUs + YAW_STEP_US
  );

  Serial.println("OK: yaw_right");
}

void setThrottle(int pwmUs) {
  if (!droneArmed) {
    Serial.println("ERROR: arm first");
    return;
  }

  baseThrottleUs = safePwm(pwmUs);
  writeAllMotors(baseThrottleUs);

  Serial.print("OK: throttle=");
  Serial.println(baseThrottleUs);
}

void printStatus() {
  Serial.println("================================================");
  Serial.println("DRONE STATUS");
  Serial.println("================================================");
  Serial.print("armed          : ");
  Serial.println(droneArmed ? "yes" : "no");
  Serial.print("baseThrottleUs : ");
  Serial.println(baseThrottleUs);
  Serial.print("WiFi SSID      : ");
  Serial.println(WIFI_AP_SSID);
  Serial.print("WiFi IP        : ");
  Serial.println(WiFi.softAPIP());
  Serial.print("TCP port       : ");
  Serial.println(WIFI_TCP_PORT);
  Serial.println("================================================");
}

void handleCommand(String cmd) {
  cmd.trim();
  cmd.toLowerCase();

  if (cmd.length() == 0) {
    return;
  }

  Serial.print("CMD: ");
  Serial.println(cmd);

  if (cmd == "arm") {
    armDrone();
  }
  else if (cmd == "disarm") {
    disarmDrone();
  }
  else if (cmd == "takeoff") {
    droneTakeoff();
  }
  else if (cmd == "land") {
    droneLand();
  }
  else if (cmd == "up") {
    droneUp();
  }
  else if (cmd == "down") {
    droneDown();
  }
  else if (cmd == "forward") {
    droneForward();
  }
  else if (cmd == "backward") {
    droneBackward();
  }
  else if (cmd == "left") {
    droneLeft();
  }
  else if (cmd == "right") {
    droneRight();
  }
  else if (cmd == "yaw_left") {
    droneYawLeft();
  }
  else if (cmd == "yaw_right") {
    droneYawRight();
  }
  else if (cmd == "hover") {
    droneHover();
  }
  else if (cmd == "stop") {
    stopMotors();
    Serial.println("OK: stop");
  }
  else if (cmd == "status") {
    printStatus();
  }
  else if (cmd.startsWith("throttle ")) {
    int valueUs = cmd.substring(9).toInt();
    setThrottle(valueUs);
  }
  else {
    Serial.println("ERROR: unknown command");
  }
}

void updateSerialCommands() {
  static String serialBuffer = "";

  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '
' || c == '') {
      if (serialBuffer.length() > 0) {
        handleCommand(serialBuffer);
        serialBuffer = "";
      }
    }
    else {
      serialBuffer += c;
    }
  }
}

void setupWiFi() {
  WiFi.mode(WIFI_AP);
  WiFi.softAP(WIFI_AP_SSID, WIFI_AP_PASSWORD);
  wifiServer.begin();

  Serial.println("Wi-Fi AP started.");
  Serial.print("SSID: ");
  Serial.println(WIFI_AP_SSID);
  Serial.print("IP: ");
  Serial.println(WiFi.softAPIP());
  Serial.print("TCP port: ");
  Serial.println(WIFI_TCP_PORT);
}

void updateWiFiCommands() {
  if (!wifiClient || !wifiClient.connected()) {
    wifiClient = wifiServer.available();
  }

  if (wifiClient && wifiClient.connected()) {
    while (wifiClient.available()) {
      String cmd = wifiClient.readStringUntil('
');
      handleCommand(cmd);
      wifiClient.println("OK command received");
    }
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(1000);

  Serial.println();
  Serial.println("================================================");
  Serial.println("myDroneESP32 generic drone firmware");
  Serial.println("================================================");

  motorFL.attach(MOTOR_FL_PIN, ESC_MIN_US, ESC_MAX_US);
  motorFR.attach(MOTOR_FR_PIN, ESC_MIN_US, ESC_MAX_US);
  motorRL.attach(MOTOR_RL_PIN, ESC_MIN_US, ESC_MAX_US);
  motorRR.attach(MOTOR_RR_PIN, ESC_MIN_US, ESC_MAX_US);

#if START_DISARMED
  disarmDrone();
#else
  armDrone();
#endif

  setupWiFi();

  Serial.println("Commands:");
  Serial.println("arm, disarm, takeoff, land, up, down");
  Serial.println("forward, backward, left, right");
  Serial.println("yaw_left, yaw_right, hover, stop");
  Serial.println("throttle 1200, status");
  Serial.println("================================================");
}

void loop() {
  unsigned long nowMs = millis();

  if (nowMs - lastLoopMs >= CONTROL_LOOP_PERIOD_MS) {
    lastLoopMs = nowMs;

    updateSerialCommands();
    updateWiFiCommands();

    // Future expansion:
    //   IMU stabilization
    //   PID control
    //   GPS waypoint navigation
    //   battery failsafe
  }
}
