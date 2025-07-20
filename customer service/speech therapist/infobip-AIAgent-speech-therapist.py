# INFOPBIP Equivalent Flask App with GPT and Google Calendar

from flask import Flask, request, jsonify
import os
import json
import pickle
import re
from datetime import datetime, timedelta
from google.oauth2 import service_account
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from openai import OpenAI, APIConnectionError, AuthenticationError, RateLimitError

app = Flask(__name__)
session_data = {}

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDENTIALS = "credentials.json"
SPEECH_INPUT_DURATION = int(os.getenv("SPEECH_INPUT_DURATION", 15))
MAX_RECORD_TIME = int(os.getenv("MAX_RECORD_TIME", 60))
MAX_NUMBER_DR_RETRY = int(os.getenv("MAX_NUMBER_DR_RETRY", 3))
MAX_APPT_RETRIEVED_FROM_CALNDER = int(os.getenv("MAX_APPT_RETRIEVED_FROM_CALENDER", 50))

# Load doctor and admin data
with open("admin_numbers.txt") as f:
    admin_numbers = [line.strip() for line in f.readlines() if line.strip()]

with open("doctors.txt") as f:
    dr_google_calendar_ids = dict(line.strip().split(":") for line in f if ":" in line)

with open("doctors_map.json") as f:
    googleid_dr_name_map = json.load(f)

# Initialize Google Calendar credentials
if not os.path.exists("token.pkl"):
    flow = InstalledAppFlow.from_client_secrets_file(
        "credentials.json",
        scopes=["https://www.googleapis.com/auth/calendar"]
    )
    creds = flow.run_local_server(port=0)
    with open("token.pkl", "wb") as token:
        pickle.dump(creds, token)
else:
    with open("token.pkl", "rb") as f:
        creds = pickle.load(f)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open("token.pkl", "wb") as token:
            pickle.dump(creds, token)

client = OpenAI(api_key=OPENAI_API_KEY)
prompt_cache = {}

def extract_doctor_name(speech_text):
    """
    Use ChatGPT (GPT-3.5) to extract the doctor's name from the caller's spoken input.

    Parameters:
        speech_text (str): The full transcribed sentence spoken by the user.

    Returns:
        str: The extracted doctor name as interpreted by the GPT model.
             If GPT is unavailable, return the original input as fallback.
    """

    # ✅ GPT prompt: ask for name only
    prompt = f"Extract the doctor name from this sentence: '{speech_text}'. Only return the name."

    try:
        # 🔗 Call OpenAI API to extract name
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract doctor names from user speech."},
                {"role": "user", "content": prompt}
            ],
            temperature=0  # 🔁 Deterministic output for consistency
        )

        # ✅ Extract and return only the name
        return response.choices[0].message.content.strip()

    except (APIConnectionError, AuthenticationError, RateLimitError) as e:
        # 🔁 Graceful fallback in case of GPT issues
        print(f"⚠️ GPT fallback in extract_doctor_name: {type(e).__name__}: {e}")
        return speech_text.strip()

    except Exception as e:
        # 🔁 Handle unexpected errors
        print(f"⚠️ Unexpected error in extract_doctor_name: {e}")
        return speech_text.strip()

import re  # ✅ Required for regular expression matching

def extract_phone_number(speech_text: str) -> str:
    """
    🔢 Extract a phone number from speech text (spoken user input).

    This function scans the input for digit patterns resembling a phone number.
    Accepts common formats like:
        - "1234567890"
        - "123 456 7890"
        - "123-456-7890"

    Parameters:
        speech_text (str): Transcribed user speech.

    Returns:
        str: A clean, digits-only phone number string.
             Returns "" if no phone number found.
    """

    # 🔍 Match 7 to 11 digits, possibly separated by space or dash
    match = re.search(r'\b(?:\d[\s\-]?){7,11}\b', speech_text)

    if match:
        # 🧼 Remove all separators and return digits only
        return match.group().replace(" ", "").replace("-", "")

    # ❌ No valid number found
    return ""

from typing import Optional
from datetime import datetime
from googleapiclient.discovery import build

def cancel_event_by_phone(
    calendar_id: str,
    phone: str,
    spoken_day: Optional[str] = None,     # e.g. "Monday" or "July 14"
    spoken_time: Optional[str] = None,    # e.g. "2 PM" or "14 00"
    creds=None
) -> bool:
    """
    Cancel (delete) a Google Calendar event that belongs to a specific phone
    number according to three fallback rules:

    1. If BOTH `spoken_day` and `spoken_time` are provided:
       • Delete the FIRST event whose phone appears in summary/description
         AND whose weekday/date matches `spoken_day`
         AND whose clock-time matches `spoken_time`.

    2. If ONLY `spoken_time` is provided:
       • Delete the FIRST event whose phone matches AND whose clock-time matches.

    3. If NEITHER day nor time is provided:
       • Delete simply the FIRST future event whose phone matches,
         regardless of date or time.

    Parameters
    ----------
    calendar_id : str
        The Google Calendar ID for the doctor.
    phone : str
        Normalized phone digits to search for (e.g. "01012345678").
    spoken_day : str | None
        Optional day or date string spoken by the caller
        ("monday", "next tuesday", "july 14", etc.)
    spoken_time : str | None
        Optional time string spoken by the caller
        ("2 pm", "14 00", etc.)
    creds : google.oauth2.credentials.Credentials | None
        Authorized credentials for Google Calendar API.

    Returns
    -------
    bool
        True  → Event found and deleted.
        False → No matching event found.
    """

    # ----- Step 1: Initialise Google Calendar service ------------------------
    calendar = build("calendar", "v3", credentials=creds)

    # ----- Step 2: Fetch up to 50 upcoming events ---------------------------
    now_iso = datetime.utcnow().isoformat() + "Z"
    events = calendar.events().list(
        calendarId=calendar_id,
        timeMin=now_iso,                               # Only future events
        maxResults=50,                                 # Reasonable window
        singleEvents=True,                             # Expand recurring events
        orderBy="startTime"                            # Chronological order
    ).execute().get("items", [])

    # ----- Helper functions to normalize spoken day/time --------------------
    def norm_day(dt: datetime) -> str:
        return dt.strftime("%A").lower()  # → 'monday', 'tuesday', etc.

    def norm_time(dt: datetime) -> str:
        return dt.strftime("%-I %p").lower()  # → '2 pm', etc.

    # Pre-normalize spoken inputs if available
    target_day  = spoken_day.lower()  if spoken_day  else None
    target_time = spoken_time.lower() if spoken_time else None

    # ----- Step 3: Iterate through events and find a match ------------------
    for event in events:
        # Event start can be dateTime or all-day date
        start_raw = event["start"].get("dateTime") or event["start"]["date"]
        start_dt  = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))

        summary     = event.get("summary", "")
        description = event.get("description", "")

        # ➊ Phone match is MANDATORY
        if phone not in summary and phone not in description:
            continue

        # ➋ Day match if provided
        if target_day and target_day not in norm_day(start_dt):
            continue

        # ➌ Time match if provided
        if target_time and target_time not in norm_time(start_dt):
            continue

        # ✅ Found the matching event — delete it
        calendar.events().delete(
            calendarId=calendar_id,
            eventId=event["id"]
        ).execute()

        return True  # Match and deletion successful

    # 🔚 No matching event found
    return False




def fallback_response(prompt):
    prompt_lower = prompt.lower()
    greeting_keywords = ["hello", "hi", "good morning", "good afternoon", "good evening", "hey", "greetings", "salaam", "marhaba"]
    booking_keywords = ["book", "make", "schedule", "appointment", "visit", "slot", "reserve"]
    cancel_keywords = ["cancel", "reschedule", "change", "remove", "delete"]
    voicemail_keywords = ["message", "voicemail", "leave", "say something", "record", "note"]

    if "list of doctors" in prompt_lower or "available doctors" in prompt_lower:
        doctor_names = ", ".join(googleid_dr_name_map.values())
        return f"The available doctors are: {doctor_names}. Please say the name of the doctor you'd like to book with."
    if any(kw in prompt_lower for kw in greeting_keywords):
        return "This is an AI Agent. Welcome to Epic Therapist Clinic! Would you like to book an appointment, cancel one, or leave a message?"
    elif any(kw in prompt_lower for kw in booking_keywords):
        return "Sure, I can help you book an appointment. Please tell me the doctor name, here is the doctors list."
    elif any(kw in prompt_lower for kw in cancel_keywords):
        return "Okay, I can help cancel your appointment. Can you please tell me your name and appointment time?"
    elif any(kw in prompt_lower for kw in voicemail_keywords):
        return "Alright, please leave your message after the beep."
    else:
        return prompt

def gpt_speak(prompt):
    print(f"📨 Prompt: {prompt}")
    print(f"🔑 Using API Key (first 8 chars): {OPENAI_API_KEY[:8] if OPENAI_API_KEY else 'Not set'}")
    if prompt in prompt_cache:
        return prompt_cache[prompt]
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful and friendly assistant for a therapy clinic."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        message = response.choices[0].message.content.strip()
        prompt_cache[prompt] = message
        return message
    except Exception as e:
        print(f"❌ GPT error: {e}")
        return fallback_response(prompt)

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

             # 🗣️ 1) Capture the caller’s speech and normalize to lowercase
            spoken_text = speech_result.lower()
            print(f"📥 Doctor name spoken: {spoken_text}")
            matched_id = None  # Will hold the Google-calendar ID once we find a match

            # ------------------------------------------------------------------
            # 🔍 2) FAST MATCH: Try simple substring matching first (cheap & quick)
            # ------------------------------------------------------------------
            for doc_id, friendly in googleid_dr_name_map.items():
                if friendly.lower() in spoken_text or spoken_text in friendly.lower():
                    matched_id = doc_id
                    break

            # ------------------------------------------------------------------
            # 🤖 3) FALLBACK: Use GPT extraction if no match
            # ------------------------------------------------------------------
            if matched_id is None:
                extracted = extract_doctor_name(speech_result)  # e.g., returns "Dr. Ahmed"
                if extracted:
                    extracted_lower = extracted.lower()
                    for doc_id, friendly in googleid_dr_name_map.items():
                        if extracted_lower in friendly.lower() or friendly.lower() in extracted_lower:
                            matched_id = doc_id
                            break

            # ------------------------------------------------------------------
            # ❌ 4) NO MATCH FOUND → retry (up to MAX_NUMBER_DR_RETRY)
            # ------------------------------------------------------------------
            if matched_id is None:
                session_data[call_sid]["retry_booking"] += 1
                retries = session_data[call_sid]["retry_booking"]

            if retries >= MAX_NUMBER_DR_RETRY:
                session_data.pop(call_sid, None)
                return jsonify({
                        "actions": [
                        {"say": {"text": gpt_speak(
                            "I'm sorry, I still couldn't match that name with any doctor in our clinic. "
                             "Please call us again when convenient. Goodbye.")}},
                        {"hangup": {}}
                    ]
                })

            # 🔁 Re-prompt the user with doctor list
            doctor_list = ", ".join(googleid_dr_name_map.values())
            retry_prompt = (
                            f"I didn't recognize that name. Available doctors are: {doctor_list}. "
                            "Please say the doctor's name again."
                            )
            return jsonify({
                "actions": [
                            {"say": {"text": gpt_speak(retry_prompt)}},
                            {
                                "collectSpeech": {
                                "timeout": SPEECH_INPUT_DURATION,
                                 "speechRecognition": {
                                 "language": "en-US"
                                 },
                        "action": {
                            "url": "/voice",
                            "method": "POST"
                            }
                        }
                    }
                ]
            })

            # ------------------------------------------------------------------
            # ✅ 5) MATCH SUCCESS → save doctor and move to "ask_time"
            # ------------------------------------------------------------------
            
            session_data[call_sid]["doctor_id"] = matched_id
            session_data[call_sid]["stage"] = "ask_time"
            friendly_name = googleid_dr_name_map[matched_id]
            time_prompt = f"What time would you like to book with {friendly_name}?"

            return jsonify({
                             "actions": [
                                         {"say": {"text": gpt_speak(time_prompt)}},
                                             {
                                                "collectSpeech": {
                                                "timeout": SPEECH_INPUT_DURATION,
                                                "speechRecognition": {
                                                "language": "en-US"
                                        },
                            "action": {
                                        "url": "/voice",
                                        "method": "POST"
                                     }
                             }
                          }
                        ]
                    })

    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)