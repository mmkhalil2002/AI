#include <WiFi.h>

// ============================================================
// BITTLE X CUSTOM WALKING + WIFI COMMAND CONTROL
// ============================================================
//
// ESP32 mode:
//  Bittle X uses ESP32-WROOM-32.
//  Use WIFI_STA mode so the ESP32 joins your router Wi-Fi.
//
// Wi-Fi control flow:
//  Python PC/Raspberry Pi -> Wi-Fi TCP port 8888 -> ESP32
//  ESP32 receives command string -> calls walking function
//
// Supported Wi-Fi commands:
//  mywalk   -> run custom walking sequence once
//  mywalk3  -> run custom walking sequence 3 times
//  stand    -> return to neutral stand
//
// ============================================================


// ============================================================
// WIFI SETTINGS
// ============================================================

const char* WIFI_SSID     = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

WiFiServer wifiServer(8888);


// ============================================================
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


// ============================================================
// Timing constants
// ============================================================

#define STEP_DELAY_MS   300
#define START_DELAY_MS  500


// ============================================================
// Low-level servo write function
//
// IMPORTANT:
// OpenCat ESP32 uses setServoP() instead of setServoP().
//
//    writeServoAngle(servoIndex, angle);
//
// This function wraps setServoP() so the rest of the code stays clean.
// ============================================================

void writeServoAngle(int servoIndex, int angle) {
  // ----------------------------------------------------------
  // OpenCat-compatible servo angle writer
  //
  // IMPORTANT:
  // Do NOT use:
  //
  //   writeServoAngle(servoIndex, angle);
  //
  // because OpenCat ESP32 defines setServoP() as:
  //
  //   setServoP(unsigned int p)
  //
  // and it accepts only ONE argument.
  //
  // For ESP PWM boards, OpenCat exposes:
  //
  //   servo[index].write(angle)
  //
  // For PCA9685 boards, OpenCat exposes:
  //
  //   pwm.writeAngle(index, angle)
  //
  // ----------------------------------------------------------

  if (servoIndex < 0 || servoIndex >= PWM_NUM) {
    return;
  }

#ifdef ESP_PWM
  servo[servoIndex].write(angle);
#else
  pwm.writeAngle(servoIndex, angle);
#endif
}


// ============================================================
// Low-level leg control
// ============================================================

void setLeg(int hipServo, int kneeServo, int hipAngle, int kneeAngle) {
  writeServoAngle(hipServo, hipAngle);
  writeServoAngle(kneeServo, kneeAngle);
}


// ============================================================
// support
// ============================================================

void FL_support() {
  setLeg(FL_HIP, FL_KNEE, NEUTRAL_HIP, SUPPORT_KNEE);
}

void FR_support() {
  setLeg(FR_HIP, FR_KNEE, NEUTRAL_HIP, SUPPORT_KNEE);
}

void RL_support() {
  setLeg(RL_HIP, RL_KNEE, NEUTRAL_HIP, SUPPORT_KNEE);
}

void RR_support() {
  setLeg(RR_HIP, RR_KNEE, NEUTRAL_HIP, SUPPORT_KNEE);
}


// ============================================================
// lift + forward
// ============================================================

void FL_lift_forward() {
  setLeg(FL_HIP, FL_KNEE, FORWARD_HIP, LIFT_KNEE);
}

void FR_lift_forward() {
  setLeg(FR_HIP, FR_KNEE, FORWARD_HIP, LIFT_KNEE);
}

void RL_lift_forward() {
  setLeg(RL_HIP, RL_KNEE, FORWARD_HIP, LIFT_KNEE);
}

void RR_lift_forward() {
  setLeg(RR_HIP, RR_KNEE, FORWARD_HIP, LIFT_KNEE);
}


// ============================================================
// land
// ============================================================

void FL_land() {
  setLeg(FL_HIP, FL_KNEE, FORWARD_HIP, SUPPORT_KNEE);
}

void FR_land() {
  setLeg(FR_HIP, FR_KNEE, FORWARD_HIP, SUPPORT_KNEE);
}

void RL_land() {
  setLeg(RL_HIP, RL_KNEE, FORWARD_HIP, SUPPORT_KNEE);
}

void RR_land() {
  setLeg(RR_HIP, RR_KNEE, FORWARD_HIP, SUPPORT_KNEE);
}


// ============================================================
// push_back
// ============================================================

void FL_push_back() {
  setLeg(FL_HIP, FL_KNEE, BACKWARD_HIP, SUPPORT_KNEE);
}

void FR_push_back() {
  setLeg(FR_HIP, FR_KNEE, BACKWARD_HIP, SUPPORT_KNEE);
}

void RL_push_back() {
  setLeg(RL_HIP, RL_KNEE, BACKWARD_HIP, SUPPORT_KNEE);
}

void RR_push_back() {
  setLeg(RR_HIP, RR_KNEE, BACKWARD_HIP, SUPPORT_KNEE);
}


// ============================================================
// Neutral standing pose
// ============================================================

void my_stand_neutral_custom() {
  FL_support();
  FR_support();
  RL_support();
  RR_support();
}


// ============================================================
// Full custom walking sequence
// ============================================================

void my_custom_walk() {

  my_stand_neutral_custom();
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

  my_stand_neutral_custom();
}


// ============================================================
// Repeat walking
// ============================================================

void my_custom_walk_repeat(int repeatCount) {
  for (int i = 0; i < repeatCount; i++) {
    my_custom_walk();
  }
}


// ============================================================
// Start Wi-Fi
// ============================================================
//
// PREFIX RULE:
// This function is called from setup(), so it uses:
//
//   mysetup<FunctionName>()
//
// Function name:
//
//   mysetupWiFiControl()
//
// ============================================================

void mysetupWiFiControl() {
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
// Call this continuously inside loop()
// ============================================================
//
// PREFIX RULE:
// This function is called from loop(), so it uses:
//
//   myloop<FunctionName>()
//
// Function name:
//
//   myloopWiFiCommand()
//
// ============================================================

void myloopWiFiCommand() {
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
    my_stand_neutral_custom();
    client.println("OK: stand");
  }
  else {
    client.println("ERROR: unknown command");
  }

  client.stop();
}


// ============================================================
// setup()
// Runs once when ESP32 starts
// ============================================================

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Put robot in a safe known pose at startup
  my_stand_neutral_custom();

  // Start Wi-Fi TCP command server
  //
  // Custom setup functions use:
  //
  //   mysetup<FunctionName>();
  //
  mysetupWiFiControl();
}


// ============================================================
// loop()
// Runs forever
// ============================================================

void loop() {
  // Custom loop functions use:
  //
  //   myloop<FunctionName>();
  //
  myloopWiFiCommand();

  // Keep loop light.
  // Walking happens only when Wi-Fi command is received.
}