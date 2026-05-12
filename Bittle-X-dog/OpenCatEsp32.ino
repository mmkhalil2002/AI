#include <WiFi.h>

// ============================================================
// OpenCat / Bittle X model and board definitions
// ============================================================

#define BITTLE

// #define BiBoard_V0_1
// #define BiBoard_V0_2
#define BiBoard_V1_0
// #define BiBoard2

// Optional modules
#define VOICE
#define ULTRASONIC
#define PIR
#define DOUBLE_TOUCH
#define DOUBLE_LIGHT
#define DOUBLE_INFRARED_DISTANCE
#define GESTURE
#define CAMERA
#define QUICK_DEMO

#include "src/OpenCat.h"


// ============================================================
// Wi-Fi settings
// ============================================================

const char* WIFI_SSID     = "TMOBILE"  //"YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "khalil02" //YOUR_WIFI_PASSWORD";

WiFiServer wifiServer(8888);

/ ============================================================
// Angle Definitions:
//  Hip:   90 = neutral, 120 = forward, 60 = backward
//  Knee:  90 = support, 120 = lift
//
// PUSH_BACK:
//  Knee stays support = 90
//  Hip moves from forward to backward: 120 -> 60
//  This pushes the robot body forward.
//
// Symbolic Walking Table:
//
// Step | FL            | FR            | RL            | RR
// ----------------------------------------------------------------
// 0    | support       | support       | support       | support
// 1    | lift+forward  | support       | support       | support
// 2    | land          | support       | support       | support
// 3    | push_back     | support       | push_back     | support
// 4    | support       | lift+forward  | support       | support
// 5    | support       | land          | support       | support
// 6    | support       | push_back     | support       | push_back
// 7    | support       | support       | lift+forward  | support
// 8    | support       | support       | land          | support
// 9    | push_back     | support       | push_back     | support
// 10   | support       | support       | support       | lift+forward
// 11   | support       | support       | support       | land
// 12   | support       | push_back     | support       | push_back
//
// ============================================================


// ============================================================
// Servo indexes
// IMPORTANT:
// Verify these indexes with your Bittle X/OpenCat servo map.
// If a wrong leg moves, change these servo index numbers.
// ============================================================


// ============================================================
// Custom symbolic walking program
// ============================================================
//
// Angle Definitions:
//  Hip:   90 = neutral, 120 = forward, 60 = backward
//  Knee:  90 = support, 120 = lift
//
// PUSH_BACK:
//  Knee stays support = 90
//  Hip moves from forward to backward: 120 -> 60
//
// Supported Wi-Fi commands:
//  mywalk   -> custom walk once
//  mywalk3  -> custom walk 3 times
//  stand    -> neutral stand
//
// IMPORTANT:
//  If setServo() does not compile in your OpenCat version,
//  replace it with the correct OpenCat servo-write routine.
// ============================================================


// ============================================================
// Servo indexes
// Verify with your Bittle X servo map
// ============================================================

#define FL_HIP   0
#define FL_KNEE  1
#define FR_HIP   2
#define FR_KNEE  3
#define RL_HIP   4
#define RL_KNEE  5
#define RR_HIP   6
#define RR_KNEE  7


// ============================================================
// Angle constants
// ============================================================

#define NEUTRAL_HIP    90
#define FORWARD_HIP    120
#define BACKWARD_HIP   60

#define SUPPORT_KNEE   90
#define LIFT_KNEE      120

#define STEP_DELAY_MS   300
#define START_DELAY_MS  500


// ============================================================
// Low-level servo write
// ============================================================

void writeServoAngle(int servoIndex, int angle) {
  setServo(servoIndex, angle);
}


// ============================================================
// Low-level leg control
// ============================================================

void setLeg(int hipServo, int kneeServo, int hipAngle, int kneeAngle) {
  writeServoAngle(hipServo, hipAngle);
  writeServoAngle(kneeServo, kneeAngle);
}


// ============================================================
// Support poses
// ============================================================

void FL_support() { setLeg(FL_HIP, FL_KNEE, NEUTRAL_HIP, SUPPORT_KNEE); }
void FR_support() { setLeg(FR_HIP, FR_KNEE, NEUTRAL_HIP, SUPPORT_KNEE); }
void RL_support() { setLeg(RL_HIP, RL_KNEE, NEUTRAL_HIP, SUPPORT_KNEE); }
void RR_support() { setLeg(RR_HIP, RR_KNEE, NEUTRAL_HIP, SUPPORT_KNEE); }


// ============================================================
// Lift + forward
// ============================================================

void FL_lift_forward() { setLeg(FL_HIP, FL_KNEE, FORWARD_HIP, LIFT_KNEE); }
void FR_lift_forward() { setLeg(FR_HIP, FR_KNEE, FORWARD_HIP, LIFT_KNEE); }
void RL_lift_forward() { setLeg(RL_HIP, RL_KNEE, FORWARD_HIP, LIFT_KNEE); }
void RR_lift_forward() { setLeg(RR_HIP, RR_KNEE, FORWARD_HIP, LIFT_KNEE); }


// ============================================================
// Land
// ============================================================

void FL_land() { setLeg(FL_HIP, FL_KNEE, FORWARD_HIP, SUPPORT_KNEE); }
void FR_land() { setLeg(FR_HIP, FR_KNEE, FORWARD_HIP, SUPPORT_KNEE); }
void RL_land() { setLeg(RL_HIP, RL_KNEE, FORWARD_HIP, SUPPORT_KNEE); }
void RR_land() { setLeg(RR_HIP, RR_KNEE, FORWARD_HIP, SUPPORT_KNEE); }


// ============================================================
// Push back
// ============================================================

void FL_push_back() { setLeg(FL_HIP, FL_KNEE, BACKWARD_HIP, SUPPORT_KNEE); }
void FR_push_back() { setLeg(FR_HIP, FR_KNEE, BACKWARD_HIP, SUPPORT_KNEE); }
void RL_push_back() { setLeg(RL_HIP, RL_KNEE, BACKWARD_HIP, SUPPORT_KNEE); }
void RR_push_back() { setLeg(RR_HIP, RR_KNEE, BACKWARD_HIP, SUPPORT_KNEE); }


// ============================================================
// Neutral standing pose
// ============================================================

void stand_neutral_custom() {
  FL_support();
  FR_support();
  RL_support();
  RR_support();
}


// ============================================================
// Custom walking sequence
// ============================================================

void my_custom_walk() {
  stand_neutral_custom();
  delay(START_DELAY_MS);

  FL_lift_forward();
  FR_support();
  RL_support();
  RR_support();
  delay(STEP_DELAY_MS);

  FL_land();
  FR_support();
  RL_support();
  RR_support();
  delay(STEP_DELAY_MS);

  FL_push_back();
  FR_support();
  RL_push_back();
  RR_support();
  delay(STEP_DELAY_MS);

  FL_support();
  FR_lift_forward();
  RL_support();
  RR_support();
  delay(STEP_DELAY_MS);

  FL_support();
  FR_land();
  RL_support();
  RR_support();
  delay(STEP_DELAY_MS);

  FL_support();
  FR_push_back();
  RL_support();
  RR_push_back();
  delay(STEP_DELAY_MS);

  FL_support();
  FR_support();
  RL_lift_forward();
  RR_support();
  delay(STEP_DELAY_MS);

  FL_support();
  FR_support();
  RL_land();
  RR_support();
  delay(STEP_DELAY_MS);

  FL_push_back();
  FR_support();
  RL_push_back();
  RR_support();
  delay(STEP_DELAY_MS);

  FL_support();
  FR_support();
  RL_support();
  RR_lift_forward();
  delay(STEP_DELAY_MS);

  FL_support();
  FR_support();
  RL_support();
  RR_land();
  delay(STEP_DELAY_MS);

  FL_support();
  FR_push_back();
  RL_support();
  RR_push_back();
  delay(STEP_DELAY_MS);

  stand_neutral_custom();
}


void my_custom_walk_repeat(int repeatCount) {
  for (int i = 0; i < repeatCount; i++) {
    my_custom_walk();
  }
}


// ============================================================
// Start Wi-Fi control
// ============================================================

void startWiFiControl() {
  Serial.println("Starting Wi-Fi control...");

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to Wi-Fi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());

  wifiServer.begin();
  Serial.println("Wi-Fi command server started on port 8888");
}


// ============================================================
// Handle Wi-Fi commands
// ============================================================

void handleWiFiCommand() {
  WiFiClient client = wifiServer.available();

  if (!client) {
    return;
  }

  String cmd = client.readStringUntil('\n');
  cmd.trim();

  Serial.print("Wi-Fi command received: ");
  Serial.println(cmd);

  if (cmd == "mywalk") {
    my_custom_walk();
    client.println("OK: mywalk");
  }
  else if (cmd == "mywalk3") {
    my_custom_walk_repeat(3);
    client.println("OK: mywalk3");
  }
  else if (cmd == "stand") {
    stand_neutral_custom();
    client.println("OK: stand");
  }
  else {
    client.println("ERROR: unknown command");
  }

  client.stop();
}


// ============================================================
// setup()
// Keep original OpenCat initialization, then start Wi-Fi
// ============================================================

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(SERIAL_TIMEOUT);

  while (Serial.available() && Serial.read())
    ;

  initRobot();

  startWiFiControl();
}


// ============================================================
// loop()
// Keep original OpenCat loop, then check Wi-Fi commands
// ============================================================

void loop() {

#ifdef VOLTAGE
  lowBattery();
#endif

  readEnvironment();
  dealWithExceptions();

  if (!tQueue->cleared()) {
    tQueue->popTask();
  } else {
    readSignal();

#ifdef QUICK_DEMO
    if (moduleList[moduleIndex] == EXTENSION_QUICK_DEMO)
      quickDemo();
#endif
  }

#ifdef NEOPIXEL_PIN
  playLight();
#endif

  reaction();

#ifdef WEB_SERVER
  WebServerLoop();
#endif

  handleWiFiCommand();
}


// ============================================================
// QUICK_DEMO original code
// ============================================================

#ifdef QUICK_DEMO

int prevReading = 0;

void quickDemo() {
  int currentReading = analogRead(ANALOG1);

  if (abs(currentReading - prevReading) > 50) {
    PT("Reading on pin ANALOG1:\t");
    PTL(currentReading);

    if (currentReading < 50) {
      tQueue->addTask(T_BEEP, "12 4 14 4 16 2");
      tQueue->addTask(T_INDEXED_SEQUENTIAL_ASC, "0 30 0 -30", 1000);
    }
    else if (abs(currentReading - prevReading) < 100) {
      if (strcmp(lastCmd, "sit"))
        tQueue->addTask(T_SKILL, "sit", 1000);
    }
    else {
      if (strcmp(lastCmd, "up"))
        tQueue->addTask(T_SKILL, "up", 1000);
    }
  }

  prevReading = currentReading;
}

#endif