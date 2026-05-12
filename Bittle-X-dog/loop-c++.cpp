// ============================================================
// loop()
// ============================================================
//
// PURPOSE:
// --------
// loop() runs FOREVER after setup() completes.
//
// This is the "runtime phase" of the robot.
// It continuously checks for commands and reacts.
//
// WHAT HAPPENS HERE:
// ------------------
// 1. Check if a Wi-Fi client connected
//    (Python script)
//
// 2. If a command is received:
//      - Read command string (e.g., "mywalk")
//      - Call corresponding function
//
// 3. If no command:
//      - Do nothing (robot stays idle)
//
// IMPORTANT DESIGN PRINCIPLE:
// ---------------------------
// loop() must be LIGHTWEIGHT and FAST.
// It should NOT block execution unnecessarily.
//
// Heavy operations (like walking) are triggered
// only when a command is received.
//
// EXECUTION FLOW:
// ---------------
// loop():
//   check Wi-Fi → receive command → execute → repeat
//
// EXAMPLE:
// --------
// Python sends "mywalk"
// → loop() detects client
// → calls my_custom_walk()
// → robot moves
//
// ============================================================

void loop() {

  // ----------------------------------------------------------
  // Check for incoming Wi-Fi commands
  // ----------------------------------------------------------
  // If Python sends a command, it will be handled here
  handleWiFiCommand();


  // ----------------------------------------------------------
  // Important note:
  // ----------------------------------------------------------
  // Do NOT add long delays here
  // Do NOT block execution unnecessarily
  //
  // The robot should always be ready to receive commands
  //
}