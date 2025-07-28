# INFOPBIP Equivalent Flask App with GPT and Google Calendar
import os
import json
import pickle
import dateparser
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
# ✅ Load environment variables from .env
load_dotenv()
from flask import Flask, request, jsonify

# ✅ Flask app configuration
app = Flask(__name__)
app.url_map.strict_slashes = False
session_data = {}


# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDENTIALS = "credentials.json"
SPEECH_INPUT_DURATION = int(os.getenv("SPEECH_INPUT_DURATION", 15))
MAX_RECORD_TIME = int(os.getenv("MAX_RECORD_TIME", 60))
MAX_NUMBER_DR_RETRY = int(os.getenv("MAX_NUMBER_DR_RETRY", 3))
MAX_APPT_RETRIEVED_FROM_CALNDER = int(os.getenv("MAX_APPT_RETRIEVED_FROM_CALENDER", 50))
INFOPBIP_USERNAME = os.getenv("INFOPBIP_USERNAME")
INFOPBIP_API_KEY = os.getenv("INFOPBIP_API_KEY")
SPEECH_INPUT_DURATION = int(os.getenv("SPEECH_INPUT_DURATION", 5))
MAX_RECORD_TIME = int(os.getenv("MAX_RECORD_TIME", 60))
MAX_NUMBER_DR_RETRY = int(os.getenv("MAX_NUMBER_DR_RETRY", 3))
MAX_APPT_RETRIEVED_FROM_CALNDER = int(os.getenv("MAX_APPT_RETRIEVED_FROM_CALENDER", 50))
# 🔧 Appointment duration in minutes (can be 15, 30, 60)
APPOINTMENT_DURATION_MINUTES = int(os.getenv("APPOINTMENT_DURATION_MINUTES", 30))
# 🌐 Global settings
MAX_TIME_SELECTION_ATTEMPTS = int(os.getenv("MAX_TIME_SELECTION_ATTEMPTS", 3))
# Define working days (0 = Monday, 6 = Sunday)
# Example: [0,1,2,3,4] for Mon–Fri in US; [0,1,2,3,5] for Sun–Thu (skip Friday)
# 0 = Monday, 1 = Tuesday, 2 = Wednesday, 3 = Thursday, 4 = Friday, 5 = Saturday, 6 = Sunday

WORKING_DAYS = [0, 1, 2, 3, 4]  # Adjust based on your local week


USE_GPT = False

#################################################
# Voice	          Description
# Polly.Joanna	Friendly US female
# Polly.Matthew	Warm US male
# Polly.Kendra	Soft, natural US female
# Polly.Ruth	Cheerful female (for casual tones)
######################################################

VOICE = "Polly.Joanna"
# Load admin numbers and doctor mapping





@app.route("/voice", methods=["POST"])
@app.route("/voice/", methods=["POST"])
def infobip_voice():
    data = request.json
    call_sid = data.get("callId", "")
    speech_result = data.get("speechText", "").strip().lower()
    stage = session_data.get(call_sid, {}).get("stage", "intro")
    actions = []

    if stage == "intro":
        session_data[call_sid] = {"stage": "intent"}
        prompt = "Greet the customer cheerfully and ask if they want to book an appointment, cancel an appointment or leave a message."
        actions.append({"action": "talk", "text": gpt_speak(prompt), "voice": "female", "language": "en"})
        actions.append({"action": "collectSpeech", "eventUrl": ["/voice"]})

    elif stage == "intent":
        if any(word in speech_result for word in ["book", "booking", "appointment", "schedule", "make", "reserve"]):
            session_data[call_id] = {"stage": "booking", "booking": {}, "retry_time": 0}
            actions.append({"action": "talk", "text": gpt_speak("list of doctors")})
            actions.append({"action": "collectSpeech", "eventUrl": ["/voice"]})

        elif "message" in speech_result or "voicemail" in speech_result:
            session_data[call_sid] = {"stage": "voicemail"}
            actions.append({"action": "talk", "text": gpt_speak("Please leave your name, phone number, and message after the beep.")})
            actions.append({"action": "record", "maxDuration": MAX_RECORD_TIME, "eventUrl": ["/transcription"]})

        elif any(word in speech_result for word in ["cancel", "reschedule", "change"]):
            session_data[call_sid] = {"stage": "cancel_appointment", "cancel": {}, "retry_booking": 0}
            prompt = "Sure, I can help you cancel your appointment. Please say the name of the doctor you had booked with."
            actions.append({"action": "talk", "text": gpt_speak(prompt)})
            actions.append({"action": "collectSpeech", "eventUrl": ["/voice"]})

        else:
            session_data[call_sid]["stage"] = "intent"
            prompt = "Sorry, I didn’t catch that. Would you like to book an appointment, cancel one, or leave a message?"
            actions.append({"action": "talk", "text": gpt_speak(prompt)})
            actions.append({"action": "collectSpeech", "eventUrl": ["/voice"]})

        return jsonify({"actions": actions})
    
    elif stage == "booking":
        # ----------------------------------------------------------------------
        # 📍 Booking flow: the caller has just been asked to name a doctor.
        # Our task here is to identify which doctor they said and, if successful,
        # proceed to ask what time they’d like to book.
        # ----------------------------------------------------------------------

        if "retry_booking" not in session_data[call_sid]:
            session_data[call_sid]["retry_booking"] = 0

        import string

        # 📻 Clean and normalize speech input
        spoken_text = speech_result.lower().strip() if speech_result else ""
        spoken_clean = spoken_text.translate(str.maketrans('', '', string.punctuation)).strip()
        print(f"📻 booking :speech_result: {spoken_clean}")

        # 🚫 Block common junk phrases
        junk_inputs = {
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening", "yo", "test",
            "1", "yes", "no", "i know", "huh", "what", "okay", "ok", "bye", "goodbye", ""
        }

        if not spoken_clean or spoken_clean in junk_inputs or len(spoken_clean) < 3:
            print(f"⏩ Skipping junk doctor input: '{spoken_clean}' — re-prompting without retry")
            doctor_list_str = ", ".join(googleid_dr_name_map.values())
            return {
                "actions": [
                    {"action": "talk", "text": gpt_speak("Please say the name of the doctor you'd like to book with.")},
                    {
                        "action": "collectSpeech",
                        "eventUrl": [f"{BASE_URL}/voice"],
                        "speechTimeout": "auto",
                        "bargeIn": True
                    }
                ]
            }

        matched_id = None

        # 🔍 1. Partial or exact substring match
        partial_matches = []
        for doc_id, friendly in googleid_dr_name_map.items():
            friendly_clean = friendly.lower().translate(str.maketrans('', '', string.punctuation)).strip()
            if spoken_clean in friendly_clean or friendly_clean in spoken_clean:
                partial_matches.append((doc_id, friendly))

        if len(partial_matches) == 1:
            matched_id = partial_matches[0][0]
            print(f"✅ Partial match with: {partial_matches[0][1]}")

        # 🤖 2. GPT fallback (only if 2+ words)
        if matched_id is None and len(spoken_clean.split()) >= 2:
            try:
                extracted = extract_doctor_name(spoken_text)
                if extracted:
                    extracted_clean = extracted.lower().translate(str.maketrans('', '', string.punctuation)).strip()
                    for doc_id, friendly in googleid_dr_name_map.items():
                        friendly_clean = friendly.lower().translate(str.maketrans('', '', string.punctuation)).strip()
                        if extracted_clean in friendly_clean or friendly_clean in extracted_clean:
                            matched_id = doc_id
                            print(f"✅ Matched via GPT fallback: {friendly}")
                            break
            except Exception as e:
                print(f"⚠️ GPT fallback failed: {e}")

        # ❌ 3. Still no match → Retry logic
        if matched_id is None:
            print(f"❌ No doctor match for: '{spoken_clean}'")
            session_data[call_sid]["retry_booking"] += 1
            retries = session_data[call_sid]["retry_booking"]

            if retries >= 3:
                session_data.pop(call_sid, None)
                return {
                    "actions": [
                        {"action": "talk", "text": gpt_speak(
                            "I'm sorry, I still couldn't match that name with any doctor in our clinic. "
                            "Please call us again when convenient. Goodbye."
                        )},
                        {"action": "hangup"}
                    ]
                }

            doctor_list_str = ", ".join(googleid_dr_name_map.values())
            retry_prompt = (
                f"I couldn't match that to a doctor. Available doctors are: {doctor_list_str}. "
                "Please say the doctor name again."
            )
            return {
                "actions": [
                    {"action": "talk", "text": gpt_speak(retry_prompt)},
                    {
                        "action": "collectSpeech",
                        "eventUrl": [f"{BASE_URL}/voice"],
                        "speechTimeout": "auto",
                        "bargeIn": True
                    }
                ]
            }

        # ✅ 4. Success → Ask for time
        session_data[call_sid]["doctor_id"] = matched_id
        session_data[call_sid]["stage"] = "ask_time_date"

        friendly_name = googleid_dr_name_map[matched_id]
        time_prompt = f"What time would you like to book with {friendly_name}?"

        return {
            "actions": [
                {"action": "talk", "text": gpt_speak(time_prompt)},
                {
                    "action": "collectSpeech",
                    "eventUrl": [f"{BASE_URL}/voice"],
                    "speechTimeout": "auto",
                    "bargeIn": True
                }
            ]
        }


    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)