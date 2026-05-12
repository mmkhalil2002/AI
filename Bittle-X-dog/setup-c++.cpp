// ============================================================
// setup()
// ============================================================
//
// PURPOSE:
// --------
// setup() runs ONE TIME when the ESP32 starts or resets.
//
// This is the "initialization phase" of the robot.
// It prepares everything before the robot begins normal operation.
//
// WHAT HAPPENS HERE:
// ------------------
// 1. Initialize Serial communication
//    - Used for debugging and printing information
//    - Needed to display Wi-Fi IP address
//
// 2. Initialize robot to a SAFE known state
//    - We call stand_neutral_custom()
//    - This ensures all legs start from a balanced position
//    - Prevents sudden jumps or unstable motion
//
// 3. Start Wi-Fi communication
//    - ESP32 connects to your router (WIFI_STA mode)
//    - Gets an IP address
//    - Starts TCP server on port 8888
//
// WHY THIS IS IMPORTANT:
// ----------------------
// - Without setup(), hardware is not initialized
// - Wi-Fi will not work
// - Robot may start from random servo positions
//
// EXECUTION FLOW:
// ---------------
// Power ON → setup() runs → loop() starts forever
//
// ============================================================

void setup() {

  // Start serial communication (for debug output)
  Serial.begin(115200);

  // Give hardware time to stabilize
  delay(1000);

  // ----------------------------------------------------------
  // Initialize robot pose
  // ----------------------------------------------------------
  // Put all legs into a stable neutral position
  // This prevents sudden movement or falling
  stand_neutral_custom();


  // ----------------------------------------------------------
  // Start Wi-Fi control system
  // ----------------------------------------------------------
  // Connect to router and start TCP server
  startWiFiControl();
}