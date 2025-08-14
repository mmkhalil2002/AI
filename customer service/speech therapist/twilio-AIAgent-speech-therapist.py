# update  07/22/25 2:45 pm
import os
import json
#import openai
import pickle
import dateparser
from flask import Flask, request
from twilio.twiml.voice_response import VoiceResponse, Gather
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime, timedelta
#import Re
from dotenv import load_dotenv
from datetime import timedelta


# ---------------- Project Structure -----------------
# speech_AI_agent/
# speech_ai_agent.py
#  generate_token.py
# .env
# credentials.json
# token.pkl          # produced by step 7
# admin_numbers.txt
# doctors.txt
# doctor_map.json
# requirements.txt
# ----------------------------------------------------

app = Flask(__name__)
app.url_map.strict_slashes = False
load_dotenv()
# Environment & API setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_NUMBER")
GOOGLE_CREDENTIALS = "credentials.json"
SPEECH_INPUT_DURATION = int(os.getenv("SPEECH_INPUT_DURATION", 12))  # how long to wait for input
PAUSE_BETWEEN_DIGITS = int(os.getenv("PAUSE_BETWEEN_DIGITS", 7))  # pause between digits
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
MAX_SILENCE_RETRIES = int(os.getenv("MAX_SILENCE_RETRIES", 3))
WORKING_DAYS = [0, 1, 2, 3, 4]  # Adjust based on your local week


USE_GPT = False
DEBUG  = True

if USE_GPT:
    import openai

    class OpenAIClient:
        def generate_text(self, prompt):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()

    client = OpenAIClient()
    print("🔁 Using OpenAI client")

else:
    from twilio.rest import Client as TwilioClient

    client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    print("📞 Using Twilio client")

## print debug

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)





# --- retry counter utils ---
def _retry_key(stage_name: str) -> str:
    return f"retry_silence__{stage_name}"

def _inc_retry(session_data, call_sid, stage_name) -> int:
    key = _retry_key(stage_name)
    session_data[call_sid][key] = session_data[call_sid].get(key, 0) + 1
    return session_data[call_sid][key]

def _reset_retry(session_data, call_sid, stage_name):
    key = _retry_key(stage_name)
    session_data[call_sid].pop(key, None)

def make_gather(prompt_text: str, hints: str = ""):
    """
    Standard gather (drop-in replacement):
    - Accepts BOTH speech and DTMF globally.
    - Caller can press '#' to finish input at any time.
    - Uses project-wide SPEECH_INPUT_DURATION (increase this to ~20 if needed).
    - speechTimeout='5' lets callers pause up to ~5s while reading digits.
    """
    g = Gather(
        input="dtmf speech",
        action="/voice",
        method="POST",
        timeout=SPEECH_INPUT_DURATION,  # consider 20 for phone collection
        speech_model="phone_call",
        speechTimeout=PAUSE_BETWEEN_DIGITS,  # ← allow longer pauses between digits
        bargeIn=True,
        finishOnKey="#",
        actionOnEmptyResult=True
    )
    if hints:
        g.hints = hints
    g.say(gpt_speak(prompt_text), VOICE)
    return g





# --- silence handler ---
def handle_silence_or_continue(
    resp, session_data, call_sid, stage_name: str, speech_result: str, reprompt_text: str, hints: str = None
):
    """
    If no speech was detected, reprompt and increment a per-stage counter.
    Returns True if we handled the response (appended a Gather or hung up).
    Returns False if normal stage logic should continue.
    """
    if speech_result and speech_result.strip():
        _reset_retry(session_data, call_sid, stage_name)
        return False

    tries = _inc_retry(session_data, call_sid, stage_name)
    debug_print(f"{stage_name}: ⚠️ No speech detected. Silence retries → {tries}/{MAX_SILENCE_RETRIES}")

    if tries >= MAX_SILENCE_RETRIES:
        debug_print(f"{stage_name}: ⛔ Max silence retries reached, ending call.")
        resp.say(gpt_speak("Sorry, I still can't hear you. Please call back later."), VOICE)
        resp.hangup()
        session_data.pop(call_sid, None)
        return True

    reprompt = f"I can't hear you. {reprompt_text}"
    gather = make_gather(reprompt, hints=hints)
    resp.append(gather)
    return True

#################################################
# Voice	          Description
# Polly.Joanna	Friendly US female
# Polly.Matthew	Warm US male
# Polly.Kendra	Soft, natural US female
# Polly.Ruth	Cheerful female (for casual tones)
######################################################

VOICE = "Polly.Joanna"
# Load admin numbers and doctor mapping
with open("admin_numbers.txt") as f:
    admin_numbers = [line.strip() for line in f.readlines() if line.strip()]

"""
Purpose: Loads a dictionary mapping Google Calendar IDs to spoken-friendly doctor names.
{
  "dr.smith@example.com": "Dr. Smith",
  "dr.jones@example.com": "Dr. Jones",
  "dr.alex@example.com": "Dr. Alex"
}
"""
with open("doctors_map.json") as f:
    googleid_dr_name_map = json.load(f)


#load_doctor_appt()  # <== Call it here on startup


# OpenAI API key
#openai.api_key = "YOUR_OPENAI_API_KEY"

# ---------------- Google Calendar Auth -----------------
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

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

#service = build("calendar", "v3", credentials=creds)

# Calendar IDs are loaded from doctors.txt

# State memory for active calls
"""
session_data = {
    "CALL_SID_1": {
        "stage": "ask_doctor",
        "doctor_id": "doctor1_calendar_id",
        "cancel": {
            "doctor": "Dr. Sarah"
        },
        ...
    },
    "CALL_SID_2": {
        "stage": "voicemail",
        ...
    },
    ...
}

As new calls arrive (via Twilio webhook), entries are added or updated like so:
session_data[call_sid] = {
    "stage": "intent"
}
Later, more keys are added dynamically:

session_data[call_sid]["doctor_id"] = matched_id
session_data[call_sid]["cancel"] = {"doctor": "Dr. Sarah"}

So the full nested structure grows as the call progresses.


"""
session_data = {}
"""
Role	    Meaning
"system"	Sets up the assistant’s behavior, tone, and expertise (e.g., polite receptionist, helpful tutor).
            Only one system message is typically needed.
            You can use the following informal structure:
            You are a [tone] [role] for a [domain]. Your job is to [main task]. Respond in a [language/style], and [behavior instruction or limitation].
                 Part	                 Description	                         Example
                [tone]	                 Personality trait or attitude	        friendly, polite, professional
                [role]    	            What the assistant should act as	    AI assistant, receptionist, customer support agent
                [domain]	            The industry or context                 therapist clinic, restaurant, medical office
                [main task]     	    Main job or expected function	        help users book appointments, take orders
                [language/style]	    Language or tone of responses           Egyptian Araic, concise English, cheerful tone
                [behavior/limitation]	Optional — any constraint or control    don’t give medical advice, wait for confirmation
"user"	    Represents input from the end user — the prompt/question you're asking.
           If you're building a voice bot (as you are), the "user" input is usually:
           request.values.get("SpeechResult", "")
           So you should assume it’s spoken language and keep the assistant ready for less formal wording, e.g.:
                 "I want to talk to the doctor."
                "Book me a session at 5."
"               I need help."

"assistant"	Represents a reply from the AI assistant — used to simulate ongoing dialogue or give memory context.
"""
def is_time_slot_available(calendar_id: str, start_time: str, end_time: str, creds) -> bool:
    """
    Check if the given time range is free on the doctor's calendar.
    Returns True if available, False if already booked.
    """
    from googleapiclient.discovery import build

    service = build("calendar", "v3", credentials=creds)
    events_result = service.events().list(
        calendarId=calendar_id,
        timeMin=start_time,
        timeMax=end_time,
        singleEvents=True,
        orderBy="startTime"
    ).execute()

    events = events_result.get("items", [])
    return len(events) == 0  # True if no conflicts


from datetime import datetime, timedelta
from typing import List, Dict
import pytz

def get_next_available_slots(calendar_id: str, creds, limit: int = 3, duration_minutes: int = 30) -> List[Dict]:
    """
    Return a list of the next `limit` available time slots for a doctor.
    Each slot is a dictionary with 'start', 'end', and 'friendly' keys.
    """
    from googleapiclient.discovery import build

    print(f"🔍 Starting search for next {limit} available slots.")
    print(f"📅 Duration per slot: {duration_minutes} minutes")

    service = build("calendar", "v3", credentials=creds)

    now = datetime.utcnow().replace(second=0, microsecond=0)
    tz = pytz.timezone("America/Chicago")
    now_local = now.astimezone(tz)
    print(f"🕒 Current local time (Chicago): {now_local}")

    # 🔁 Start from the next rounded-up duration boundary (e.g., 9:00, 9:30, etc.)
    minute = (now_local.minute // duration_minutes + 1) * duration_minutes
    rounded_start = now_local.replace(minute=0) + timedelta(minutes=minute)
    if rounded_start.minute >= 60:
        rounded_start += timedelta(hours=1)
        rounded_start = rounded_start.replace(minute=0)

    print(f"⏱️ First slot to check (rounded): {rounded_start.strftime('%Y-%m-%d %H:%M')}")

    suggestions = []
    checked_slots = 0
    MAX_LOOKAHEAD_HOURS = 72  # ⏩ Max time range to scan

    while len(suggestions) < limit and checked_slots < (MAX_LOOKAHEAD_HOURS * 60) // duration_minutes:
        end_slot = rounded_start + timedelta(minutes=duration_minutes)
        time_min = rounded_start.isoformat()
        time_max = end_slot.isoformat()

        print(f"🔎 Checking slot: {time_min} → {time_max} ...")

        try:
            events_result = service.events().list(
                calendarId=calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True
            ).execute()
        except Exception as e:
            print(f"❌ Error while querying events: {e}")
            break

        if not events_result.get("items", []):
            friendly = rounded_start.strftime("%B %-d at %-I:%M %p")
            print(f"✅ Slot available: {friendly}")
            suggestions.append({
                "start": time_min,
                "end": time_max,
                "friendly": friendly
            })
        else:
            print(f"❌ Slot is busy ({len(events_result.get('items', []))} events)")

        rounded_start += timedelta(minutes=duration_minutes)
        checked_slots += 1

    print(f"📦 Found {len(suggestions)} available slots after checking {checked_slots} candidates.")
    return suggestions




from typing import List, Tuple
from datetime import datetime, timedelta, time
import pytz

# Global working config
WORKING_DAYS = [0, 2, 3, 4]  # Mon=0, Tue=1,... Friday=4
WORKING_HOURS_START = 8  # 8:00 AM
WORKING_HOURS_END = 17   # 5:00 PM
LUNCH_BREAK_START = time(13, 0)  # 1:00 PM
LUNCH_BREAK_END = time(14, 0)    # 2:00 PM

def suggest_alternative_times(
    doctor_id: str,
    creds,
    num_options: int = 3
) -> List[Tuple[str, str]]:
    from googleapiclient.discovery import build
    service = build("calendar", "v3", credentials=creds)

    now = datetime.utcnow().replace(tzinfo=pytz.UTC)
    search_end = now + timedelta(days=7)

    suggested = []
    current_time = now.replace(hour=WORKING_HOURS_START, minute=0, second=0, microsecond=0)

    while current_time < search_end and len(suggested) < num_options:
        weekday = current_time.weekday()
        current_local = current_time.astimezone(pytz.timezone("UTC"))  # adjust as needed

        # Skip non-working days
        if weekday not in WORKING_DAYS:
            current_time += timedelta(days=1)
            current_time = current_time.replace(hour=WORKING_HOURS_START, minute=0)
            continue

        # Skip outside working hours
        if current_time.time() < time(WORKING_HOURS_START) or current_time.time() >= time(WORKING_HOURS_END):
            current_time += timedelta(days=1)
            current_time = current_time.replace(hour=WORKING_HOURS_START, minute=0)
            continue

        # Skip lunch break
        if LUNCH_BREAK_START <= current_time.time() < LUNCH_BREAK_END:
            current_time = current_time.replace(hour=LUNCH_BREAK_END.hour, minute=0)
            continue

        start_str = current_time.isoformat()
        end_time = current_time + timedelta(minutes=30)
        end_str = end_time.isoformat()

        # Check if time slot is free
        events_result = service.events().list(
            calendarId=doctor_id,
            timeMin=start_str,
            timeMax=end_str,
            singleEvents=True,
            orderBy="startTime"
        ).execute()
        events = events_result.get("items", [])

        if not events:
            suggested.append((start_str, end_str))

        current_time += timedelta(minutes=30)

    return suggested







# 🗣️ Helper to speak readable version of time: "July 3rd at 9:30 AM"
def format_time_for_speech(slot: Tuple[str, str]) -> str:
    dt = datetime.fromisoformat(slot[0])
    month = dt.strftime("%B")
    day = dt.day
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    time_str = dt.strftime("%I:%M %p").lstrip("0")
    return f"{month} {day}{suffix} at {time_str}"





from datetime import datetime, timedelta
from typing import Tuple
import re
def normalize_date_time(spoken_day: str, spoken_time: str) -> str:
    """
    Normalize input like '29th of July' to 'July 29'
    """
    import re
    # Remove ordinal suffixes (e.g., "29th" → "29")
    day = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', spoken_day.strip(), flags=re.IGNORECASE)

    # Handle formats like "29 of July"
    match = re.match(r"(\d+)\s+of\s+([A-Za-z]+)", day, flags=re.IGNORECASE)
    if match:
        day = f"{match.group(2)} {match.group(1)}"

    # Remove "of", commas, etc.
    day = day.replace(",", "").replace("of", "").strip()

    # Combine with time
    return f"{day} {spoken_time}".strip()





from datetime import datetime, timedelta, date, time as dtime
from typing import Tuple, Union
import pytz, re

def build_timeslot_range(spoken_day: Union[str, date], spoken_time: Union[str, dtime],
                         tolerance_minutes: int = 30) -> Tuple[str, str]:
    """
    Convert a spoken date/time (or date/time objects) to a UTC ISO 8601 window.
    Returns (utc_start_iso, utc_end_iso). Always uses America/Chicago → UTC.
    """
    debug_print(f"📥 build_timeslot_range: Input → Day: '{spoken_day}', Time: '{spoken_time}'")
    local_tz = pytz.timezone("America/Chicago")

    # If already date/time objects
    if isinstance(spoken_day, date) and isinstance(spoken_time, dtime):
        combined = datetime.combine(spoken_day, spoken_time)
        localized = local_tz.localize(combined)
        utc_start = localized.astimezone(pytz.utc)
        utc_end = utc_start + timedelta(minutes=tolerance_minutes)
        debug_print(f"📅 Local slot: {localized} → {localized + timedelta(minutes=tolerance_minutes)}")
        debug_print(f"🌍 UTC slot: {utc_start.isoformat()} → {utc_end.isoformat()}")
        return utc_start.isoformat(), utc_end.isoformat()

    # Else strings → clean + parse
    day_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', str(spoken_day)).replace(",", "").replace(" of ", " ").strip()
    combo = f"{day_str} {spoken_time}".strip()
    debug_print(f"🧽 Cleaned combined input: '{combo}'")

    formats = [
        "%A %B %d %I:%M %p",  # Wednesday July 8 10:00 AM
        "%B %d %I:%M %p",     # July 8 10:00 AM
        "%A %B %d %H:%M",     # Wednesday July 8 14:30
        "%B %d %H:%M",        # July 8 14:30
    ]

    parsed = None
    for fmt in formats:
        try:
            parsed = datetime.strptime(combo, fmt)
            debug_print(f"✅ Parsed datetime: {parsed} using format {fmt}")
            break
        except ValueError:
            continue

    if not parsed:
        raise ValueError(f"🛑 Could not parse datetime from '{combo}'")

    parsed = parsed.replace(year=datetime.now().year)  # infer current year
    debug_print(f"📅 Inferred year → {parsed}")

    localized = local_tz.localize(parsed)
    utc_start = localized.astimezone(pytz.utc)
    utc_end = utc_start + timedelta(minutes=tolerance_minutes)
    debug_print(f"📅 Local slot: {localized} → {localized + timedelta(minutes=tolerance_minutes)}")
    debug_print(f"🌍 UTC slot: {utc_start.isoformat()} → {utc_end.isoformat()}")
    return utc_start.isoformat(), utc_end.isoformat()








import os
from openai import OpenAI
from openai import APIConnectionError, AuthenticationError, RateLimitError


def smart_parse_time(text):
    import dateparser
    import re
    from datetime import datetime

    if not text:
        return None

    original_text = text

    # 🧽 Normalize "2 30", "2, 30" → "2:30"
    text = re.sub(r"\b(\d{1,2})[,\s]+(\d{2})\b", r"\1:\2", text)

    # 🧽 Remove ordinal suffixes like "3rd", "22nd" → "3", "22"
    text = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", text, flags=re.IGNORECASE)

    # 🧼 Trim and fix isolated time (e.g., "2:30")
    text = re.sub(r"\s+", " ", text.strip())
    if re.match(r"^\d{1,2}:\d{2}$", text):
        text = "at " + text

    print(f"🧽 Cleaned time input: {text} (original was: {original_text})")

    # 🧠 Parse using dateparser
    dt = dateparser.parse(text, settings={"PREFER_DATES_FROM": "future"})

    if not dt:
        return None

    now = datetime.now()

    # 🛠 Ensure current year if user mentioned a month name
    if re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", text, re.IGNORECASE):
        dt = dt.replace(year=now.year)

    # ✅ Return (day, time) as clean strings
    spoken_day = dt.strftime("%A, %B %-d")  # e.g., "Monday, July 3"
    spoken_time = dt.strftime("%-I:%M %p")  # e.g., "9:00 AM"

    return spoken_day, spoken_time





# ✅ OpenAI client initialization
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#client = OpenAI(api_key=OPENAI_API_KEY)




# ✅ Cache dictionary
prompt_cache = {}

def fallback_response(prompt):
    """
    Rule-based fallback if GPT is not available or quota is exceeded.
    If you want to customize this further in the future, you can re-enable
    rule-based matching (e.g., for greetings, booking, etc).
    For now, this just returns the original input prompt as-is.
    """
    return prompt

def gpt_speak(prompt):
    """
    Tries to use GPT to answer a prompt. Falls back to rule-based logic on error or quota limits.
    Caches responses to avoid duplicate API calls.
    """
    print(f"📨 Prompt: {prompt}")
    print(f"🔑 Using API Key (first 8 chars): {OPENAI_API_KEY[:8] if OPENAI_API_KEY else 'Not set'}")
    if USE_GPT == False:
        return fallback_response (prompt)
    else:
        # Use cached response if available
        if prompt in prompt_cache:
            print("🔁 Returning cached GPT response.")
            return prompt_cache[prompt]

        try:
            # Send request to OpenAI ChatGPT API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful and friendly assistant for a therapy clinic."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            # Extract and store response
            message = response.choices[0].message.content.strip()
            prompt_cache[prompt] = message
            print(f"✅ GPT response: {message}")
            return message

        except Exception as e:
            print(f"❌ GPT error: {e}")
            print("↪️ Falling back to rule-based logic.")
            return fallback_response(prompt)



# Doctor/phone extraction
"""
    If you’d like ChatGPT to do the heavy lifting of fuzzy name extraction—for example
    when the caller says a whole sentence (“I want to see Doctor Omar, please”)—you can dro the helper into the exact spot where you try to match the spoken name.

    Below is the recommended place: inside your cancel_phone stage (or ask_doctor stage for booking).
    The idea is:

    Try fast exact matching with your loop (what you already do).

    If no match, fall back to GPT extraction, then try the match again.
    Robustness: Handles phrases like
    “I’d like to cancel with Dr Ahmed next week.”
    Your simple loop might fail because "dr ahmed" is embedded in a longer sentence; GPT neatly isolates “Dr Ahmed.”

    Performance: You still keep the quick dictionary check first; GPT is only called if the easy path fails.
"""
from openai import OpenAI, APIConnectionError, AuthenticationError, RateLimitError

# Initialize the OpenAI client (using the environment variable OPENAI_API_KEY)
#client = OpenAI()

from openai import OpenAIError  # Add this import at the top

def extract_doctor_name(speech_text):
    """
    Use ChatGPT (GPT-3.5) to extract the doctor's name from the caller's spoken input.

    Parameters:
        speech_text (str): The full transcribed sentence spoken by the user.

    Returns:
        str: The extracted doctor name as interpreted by the GPT model.
             If GPT is unavailable or uncertain, return the original input as fallback.
    """

    if not speech_text.strip():
        return ""

    # 🚀 Prompt engineering: Ask GPT to extract ONLY the name
    prompt = (
        f"From this sentence: \"{speech_text}\", extract only the doctor's name "
        f"mentioned. Return only the name, without titles like Dr. or punctuation. "
        f"If no name is mentioned, return an empty string."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract doctor names from user speech."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        extracted = response.choices[0].message.content.strip()
        print(f"✅ GPT extracted doctor name: {extracted}")
        return extracted

    except (APIConnectionError, AuthenticationError, RateLimitError, OpenAIError) as e:
        print(f"⚠️ GPT fallback in extract_doctor_name: {type(e).__name__}: {e}")
        return speech_text.strip()

    except Exception as e:
        print(f"⚠️ Unexpected error in extract_doctor_name: {e}")
        return speech_text.strip()


import re


def extract_phone_number(speech_text: str) -> str:
    """
    Extracts a phone number from a transcribed voice input.

    Supports:
    - Spoken digit sequences with spaces or hyphens (e.g. "4 6 9 4 6 3 3 2 7 6")
    - Common phone number groupings (e.g. "469-463-3276", "469 463 3276")
    - Returns a compact number with digits only (e.g. "4694633276")

    Parameters:
        speech_text (str): Transcribed text from the caller.

    Returns:
        str: Cleaned phone number string (digits only), or empty string if invalid.
    """
    import re

    if not speech_text:
        print("📞 extract_phone_number: Input is empty.")
        return ""

    print(f"🗣️ Original speech input: '{speech_text}'")

    # 🧼 Step 1: Remove all characters except digits, spaces, and dashes
    cleaned = re.sub(r"[^\d\s\-]", "", speech_text)
    print(f"🔍 Cleaned speech (kept digits/spaces/dashes): '{cleaned}'")

    # 🔢 Step 2: Extract digits only
    digits = re.sub(r"[^\d]", "", cleaned)
    print(f"📞 Digits only: '{digits}'")

    # ✅ Step 3: Check length
    if 7 <= len(digits) <= 11:
        print(f"✅ Valid phone number found: {digits}")
        return digits
    else:
        print(f"❌ Invalid phone number length: {len(digits)} digits")
        return ""

import re  # Import the regular expression module
from datetime import datetime, timedelta
#from googleapiclient.discovery import build
from typing import Optional
import pytz
from googleapiclient.discovery import build

def cancel_event_by_phone(
    calendar_id: str,
    phone: str,
    spoken_day: Optional[str] = None,
    spoken_time: Optional[str] = None,
    creds=None,
    return_details: bool = False
):
    """
    Cancel a Google Calendar event based on phone number and optional day/time.

    Returns:
        - Matching event object if return_details is True
        - True if deletion was successful
        - False / None if not found
    """

    clean_phone = re.sub(r"[^\d]", "", phone)
    print(f"🔍 Searching for normalized phone: {clean_phone}")

    parsed_datetime = None
    if spoken_day and spoken_time:
        try:
            """
              spoken_day = "July 29"
              spoken_time = "8:30 AM"
              start_iso, _ = build_timeslot_range("July 29", "8:30 AM")
              print(start_iso)
              output 
                2025-07-29T13:30:00+00:00

            """
            start_iso, _ = build_timeslot_range(spoken_day, spoken_time)
            """
            start_iso = "2025-07-29T13:30:00Z"
            Z will be repaced 
            2025-07-29 13:30:00+00:00
            <class 'datetime.datetime'>

            """
            parsed_datetime = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
            print(f"🧠 Parsed target datetime (UTC): {parsed_datetime.isoformat()}")
        except Exception as e:
            print(f"⚠️ Failed to parse spoken datetime → {spoken_day}, {spoken_time}: {e}")

    """

     Parameter	                            Purpose
     --------------------------------------------------------------------------------------------------------------------
    calendarId=calendar_id	Specifies the calendar you want to query. This ID is typically a Google email or a group calendar ID.
    timeMin=now	            Filters events to only include those starting after the current time.
    maxResults=50	        Limits the number of events returned to a maximum of 50.
    singleEvents=True	    Expands recurring events into individual occurrences. So if a meeting repeats every Monday, each one appears separately.
    orderBy="startTime"	    Sorts the results chronologically by their start time.

    Finally, .execute() sends the API request and returns the response as a dictionary containing events.

    Example Result (events_result):
    
     {
        "items": [
                     {
                         "summary": "Appointment for Ali",
                         "start": { "dateTime": "2025-08-01T09:00:00-05:00" },
                         "description": "Name: Ali\nPhone: 4694633276\nAddress: 123 Main St"
                    },
                    {
                       "summary": "Appointment for Sarah",
                       "start": { "dateTime": "2025-08-02T14:30:00-05:00" },
                       "description": "Name: Sarah\nPhone: 4699991234\nAddress: ..."
                    }
                 ]
    c}

       """
    
    service = build("calendar", "v3", credentials=creds)
    now = datetime.utcnow().isoformat() + 'Z'
    events_result = service.events().list(
        calendarId=calendar_id,
        timeMin=now,
        maxResults=50,
        singleEvents=True,
        orderBy="startTime"
    ).execute()

    """
    events_result["items"]
    [
     {
        "kind": "calendar#event",
        "id": "evt-abc123",
        "status": "confirmed",
        "summary": "Appointment for Muhammad Khalil",
        "description": "Name: Muhammad Khalil\nPhone: 4694633276\nAddress: 118 Briar Oak, Murphy, TX 75094",
        "start": {
                     "dateTime": "2025-07-29T08:30:00-05:00",
                    "timeZone": "America/Chicago"
                },
        "end": {
                    "dateTime": "2025-07-29T09:00:00-05:00",
                    "timeZone": "America/Chicago"
               },
        "created": "2025-07-20T14:00:00Z",
        "updated": "2025-07-20T14:01:00Z",
        "organizer": {
                         "email": "dr.john@example.com"
                    },
         "htmlLink": "https://www.google.com/calendar/event?eid=evt-abc123"
     },
     {
         "kind": "calendar#event",
         "id": "evt-def456",
         "status": "confirmed",
         "summary": "Appointment for Ali Abdel",
         "description": "Phone: 4694633276\nAddress: 118 Brier Oak, Murphy, TX 75094",
        "start": {
                     "dateTime": "2025-08-01T13:00:00-05:00",
                     "timeZone": "America/Chicago"
                },
        "end": {
                     "dateTime": "2025-08-01T13:30:00-05:00",
                     "timeZone": "America/Chicago"
                },
        "created": "2025-07-21T11:22:00Z",
        "updated": "2025-07-21T11:22:30Z",
        "organizer": {
                         "email": "dr.john@example.com"
                     },
         "htmlLink": "https://www.google.com/calendar/event?eid=evt-def456"
     }
    ]
     if summanry is    Appointment for Muhammad Khalil 469-463-3276"
     then  summary_digits → "4694633276"

     if description "Name: Muhammad Khalil\nPhone: 469 463 3276\nAddress: 118 Briar Oak, Murphy, TX 75094"
     then description_digits → "469463327611875094"
    
    """

    events = events_result.get("items", [])
    print(f"📅 Retrieved {len(events)} upcoming events to check")

    for event in events:
        summary = event.get("summary", "").lower()
        description = event.get("description", "").lower()

        summary_digits = re.sub(r"[^\d]", "", summary)
        description_digits = re.sub(r"[^\d]", "", description)

        print("🔎 Checking event:")
        print(f"     summary: {summary}")
        print(f"     description: {description}")
        print(f"     normalized summary digits: {summary_digits}")
        print(f"     normalized description digits: {description_digits}")

        if clean_phone in summary_digits or clean_phone in description_digits:
            print("✅ Phone number matched.")
            """
            event_start (dateTime) -> "2025-07-29T08:30:00-05:00"

                2025-07-29: the date
                T08:30:00: the time (08:30 AM)
                -05:00: the timezone offset from UTC (Central Daylight Time in this case)
            """
            event_start = event.get("start", {}).get("dateTime")
            if not event_start:
                print("⚠️ Skipping all-day or malformed event.")
                continue

            try:
                event_dt = datetime.fromisoformat(event_start.replace("Z", "+00:00"))

                if parsed_datetime:
                    """
                    from datetime import datetime

                    # 📅 Event datetime from Google Calendar (in UTC)
                    event_dt = datetime.fromisoformat("2025-07-29T13:30:00+00:00")

                    # 📞 Parsed user input (converted to UTC)
                    parsed_datetime = datetime.fromisoformat("2025-07-29T13:30:00+00:00")

                    # 🔁 Compute time difference in seconds
                    delta = abs((event_dt - parsed_datetime).total_seconds())

                    print("Time difference in seconds:", delta)

                    """
                    delta = abs((event_dt - parsed_datetime).total_seconds())
                    print(f"🕐 Comparing event start {event_dt} to target {parsed_datetime}, Δ={delta}s")

                    if delta <= 10:
                        print("🗑️ Deleting matching event...")
                        """
                        calendar_id = "doctor123@clinic-calendar.com"
                        event = {
                                     "id": "ab12cd34ef56gh78ij90kl",
                                     "summary": "Appointment for Mohamed Khalil",
                                     "start": {"dateTime": "2025-07-29T13:30:00+00:00"},
                                     "description": "Name: Mohamed Khalil\nPhone: 4694633276\nAddress: 118 Briar Oak, Murphy, TX"
                                }
                         event["id"] ->  "ab12cd34ef56gh78ij90kl"
                         calendar_id -?  input to this function
                         delete based on event_id, and claender_id    

                        """
                        service.events().delete(calendarId=calendar_id, eventId=event["id"]).execute()
                        return event if return_details else True
                    else:
                        print("❌ Date/time mismatch despite phone match.")
                else:
                    print("⚠️ No valid spoken datetime to match against.")

            except Exception as e:
                print(f"⚠️ Failed to parse event datetime: {e}")
                continue

    print("🚫 No matching appointment found.")
    return None if return_details else False




import os
import json

# 📁 Directory to store appointment data files
APPOINTMENT_TABLE_DIR = "./appointment_data"

# Global dictionary: {doctor_filename: {phone: {datetime, calendar_id}}}
doctor_appointments = {}

# ------------------------
# 🧰 Utility functions
# ------------------------

def sanitize_filename(name: str) -> str:
    """Convert doctor name to a safe filename (e.g., John Wayne → john_wayne.json)"""
    filename = name.lower().replace(" ", "_") + ".json"
    print(f"[sanitize_filename] Sanitized '{name}' → '{filename}'")
    return filename

def get_doctor_filename(friendly_name: str) -> str:
    """Return full path for doctor's JSON file"""
    path = os.path.join(APPOINTMENT_TABLE_DIR, sanitize_filename(friendly_name))
    print(f"[get_doctor_filename] Full path for '{friendly_name}' → {path}")
    return path

# ------------------------
# 🔄 Load on startup
# ------------------------

def load_doctor_appointments():
    """Load all appointment mappings into memory on startup"""
    print("[load_doctor_appointments] Loading appointment tables...")
    os.makedirs(APPOINTMENT_TABLE_DIR, exist_ok=True)
    for filename in os.listdir(APPOINTMENT_TABLE_DIR):
        if filename.endswith(".json"):
            path = os.path.join(APPOINTMENT_TABLE_DIR, filename)
            try:
                with open(path, "r") as f:
                    doctor_name = filename.replace(".json", "")
                    doctor_appointments[doctor_name] = json.load(f)
                    print(f"[load_doctor_appointments] Loaded {filename}")
            except Exception as e:
                print(f"[load_doctor_appointments] ⚠️ Failed to load {filename}: {e}")
    print(f"[load_doctor_appointments] 📂 Loaded appointment tables: {list(doctor_appointments.keys())}")

# ------------------------
# ➕ Add appointment
# ------------------------
import os
import json

def confirm_appointment_by_name(
    doctor_name: str,
    phone: str,
    utc_start: str,
    calendar_id: str,
    name: str = None,
    dob: str = None,
    address: str = None,
    event_id: str = None,
    debug: bool = False,
):
    """
    Add a new appointment to the doctor's table and save to JSON file.

    Compatibility:
      - Required params remain: doctor_name, phone, utc_start, calendar_id.
      - Optional params (name, dob, address, event_id, debug) have defaults, so existing
        call sites won't break if they don't pass them.

    Behavior:
      - Normalizes phone to digits-only.
      - Ensures utc_start is UTC ISO8601 (e.g., '2025-08-07T10:00:00Z').
      - Searches existing file by (phone + dob) if dob provided; otherwise by phone only.
      - Skips exact duplicates (same phone + dob + time + calendar_id).
      - Appends record with optional name/dob/address/event_id.
      - Saves back to disk and refreshes in-memory cache doctor_appointments[filename], if defined.

    Returns:
      dict with:
        created: bool           # True if appended, False if duplicate
        record: dict            # The record (new or existing)
        reason: str | None      # 'duplicate' if not created, else None
    """
    import os
    import re
    import json
    from datetime import datetime, timezone

    def _dbg(msg: str):
        if debug:
            print(f"[confirm_appointment_by_name] {msg}")

    # -----------------------
    # Normalize phone digits
    # -----------------------
    digits_only_phone = re.sub(r"\D", "", phone or "")
    if not digits_only_phone:
        raise ValueError("Phone is required and must contain digits.")

    # -----------------------------------------
    # Normalize DOB into ISO YYYY-MM-DD (if any)
    # -----------------------------------------
    dob_iso = (dob or "").strip()
    if dob_iso:
        # Already ISO?
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", dob_iso) is None:
            # Try MM/DD/YYYY or MM-DD-YYYY
            m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", dob_iso)
            if m:
                mm, dd, yyyy = m.groups()
                dob_iso = f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}"
            else:
                # Light normalization: 2025/08/07 → 2025-08-07
                dob_iso = dob_iso.replace("/", "-")

    # --------------------------------------
    # Ensure utc_start is UTC ISO8601 string
    # --------------------------------------
    def ensure_utc_iso(ts: str) -> str:
        """
        Accepts:
          - '2025-08-07T10:00:00Z'
          - '2025-08-07T10:00:00+00:00'
          - '2025-08-07 10:00:00' (assumed UTC if naive)
        Returns: 'YYYY-MM-DDTHH:MM:SSZ'
        """
        if not ts:
            raise ValueError("utc_start is required")
        s = ts.strip().replace(" ", "T")
        try:
            # Handle trailing Z by converting to +00:00 for fromisoformat
            if s.endswith("Z"):
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            else:
                dt = datetime.fromisoformat(s)
        except Exception:
            # If naive pattern 'YYYY-MM-DDTHH:MM:SS', try that
            if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", s):
                dt = datetime.fromisoformat(s)
            else:
                raise
        # Force UTC tz-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    utc_start_iso = ensure_utc_iso(utc_start)

    # --------------------------
    # Resolve file paths/keys
    # --------------------------
    filename = sanitize_filename(doctor_name).replace(".json", "")
    full_path = get_doctor_filename(doctor_name)
    _dbg(f"🔍 File → {full_path}")

    # --------------------------
    # Load existing appointments
    # --------------------------
    appts = []
    if os.path.exists(full_path):
        try:
            with open(full_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                appts = data
                _dbg(f"✅ Loaded list with {len(appts)} appointment(s)")
            else:
                _dbg("⚠️ Root JSON was not a list; reinitializing")
        except Exception as e:
            _dbg(f"⚠️ Failed to parse JSON → {e}")
    else:
        _dbg("📂 No file found — starting new list")

    # -------------------------------------------------------
    # Search by phone (+ dob if provided) for duplicates/info
    # -------------------------------------------------------
    matches = []
    for idx, appt in enumerate(appts):
        p = re.sub(r"\D", "", appt.get("phone", ""))
        d = (appt.get("dob") or "").strip()
        if dob_iso:
            if p == digits_only_phone and d == dob_iso:
                matches.append((idx, appt))
        else:
            if p == digits_only_phone:
                matches.append((idx, appt))

    _dbg(f"🔎 Search by phone+dob → {len(matches)} match(es) "
         f"(phone={digits_only_phone}, dob={dob_iso or 'N/A'})")

    # -----------------------------------------------------------
    # Skip exact duplicate (same phone + dob + time + calendar)
    # -----------------------------------------------------------
    for _, appt in matches:
        try:
            appt_time_iso = ensure_utc_iso(appt.get("time", ""))
        except Exception:
            # If bad time in file, don't treat as duplicate
            appt_time_iso = None

        if appt_time_iso == utc_start_iso and appt.get("calendar_id") == calendar_id:
            _dbg("🔁 Exact duplicate detected — skipping append")
            # Normalize record before returning
            appt_norm = dict(appt)
            appt_norm["phone"] = re.sub(r"\D", "", appt_norm.get("phone", ""))
            appt_norm["time"] = utc_start_iso
            return {"created": False, "record": appt_norm, "reason": "duplicate"}

    # ---------------------------------
    # Append new appointment record
    # ---------------------------------
    new_record = {
        "phone": digits_only_phone,
        "time": utc_start_iso,
        "calendar_id": calendar_id,
    }
    if name:
        new_record["name"] = name
    if dob_iso:
        new_record["dob"] = dob_iso
    if address:
        new_record["address"] = address
    if event_id:
        new_record["event_id"] = event_id

    appts.append(new_record)
    _dbg(f"➕ Appended: {new_record}")

    # -----------------------------
    # Save back to disk (+ cache)
    # -----------------------------
    try:
        with open(full_path, "w") as f:
            json.dump(appts, f, indent=2)
        _dbg(f"💾 Saved to {full_path}")

        # Update in-memory cache if present
        try:
            doctor_appointments[filename] = appts
        except Exception:
            pass

        return {"created": True, "record": new_record, "reason": None}
    except Exception as e:
        _dbg(f"❌ Failed to write JSON → {e}")
        raise







def normalize_phone_digits(phone: str) -> str:
    """Digits-only normalization for matching (calendar description & JSON)."""
    return ''.join(ch for ch in (phone or "") if ch.isdigit())



from dateutil import parser
import os
import json
import os, json
from dateutil import parser as dtparser
def cancel_appointment_by_name(doctor_name: str, phone: str, utc_start: str, dob: str = None) -> bool:
    """
    Remove a doctor's appointment by exact UTC start time and phone,
    and (optionally) DOB. If DOB is provided, it must match too.

    Matching rules:
      - Always require: normalized(phone) == appt.phone AND utc_start == appt.time (both UTC ISO)
      - Additionally require: normalized(dob) == appt.dob (if dob is provided)

    All times are normalized to UTC ISO (e.g., 'YYYY-MM-DDTHH:MM:SS+00:00') before comparison.
    Keeps all other appointments intact. Returns True if ≥1 appointment removed.
    """
    import os, json
    from datetime import timezone
    try:
        import dateutil.parser as dtparser
    except Exception:
        # If your code already imports dtparser elsewhere, you may remove this.
        raise

    def normalize_phone_digits(s: str) -> str:
        return "".join(ch for ch in (s or "") if ch.isdigit())

    def normalize_dob(s: str) -> str:
        """
        Keep simple ISO-like 'YYYY-MM-DD' if present.
        - Trims whitespace
        - If a full datetime was stored, use the date portion before 'T'
        - Returns lower-cased, trimmed string (though digits/hyphens only expected)
        """
        s = (s or "").strip()
        if "T" in s:
            s = s.split("T", 1)[0].strip()
        return s

    def normalize_utc_iso(s: str) -> str:
        """
        Parse any ISO-ish string and return strict UTC ISO string with +00:00 offset.
        """
        dt = dtparser.isoparse(s)
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.isoformat()

    key = sanitize_filename(doctor_name).replace(".json", "")
    full_path = get_doctor_filename(doctor_name)
    phone_digits = normalize_phone_digits(phone)
    dob_norm = normalize_dob(dob) if dob else None

    debug_print(f"🩺 cancel_appointment_by_name → doctor={doctor_name}, phone={phone_digits}, dob={dob_norm or '∅'}, utc_start={utc_start}")

    if not os.path.exists(full_path):
        debug_print(f"⚠️ File not found: {full_path}")
        return False

    # Load the doctor's JSON list
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                debug_print(f"❌ JSON not a list for {full_path}")
                return False
    except Exception as e:
        debug_print(f"❌ Read error {full_path} → {e}")
        return False

    # Normalize the target UTC time once
    try:
        target_norm = normalize_utc_iso(utc_start)
    except Exception as e:
        debug_print(f"❌ utc_start parse error → {e}")
        return False

    kept, removed = [], 0
    for appt in data:
        ap_phone = normalize_phone_digits(appt.get("phone", ""))
        ap_time_raw = appt.get("time", "")
        ap_dob_raw = appt.get("dob", "") or appt.get("date_of_birth", "")
        ap_dob_norm = normalize_dob(ap_dob_raw)

        # Normalize appt time; skip malformed records (keep them)
        try:
            ap_time_norm = normalize_utc_iso(ap_time_raw)
        except Exception as e:
            debug_print(f"⚠️ skip invalid appt time '{ap_time_raw}' → {e}")
            kept.append(appt)
            continue

        # Match rule: phone & time must match; if caller provided DOB, that must match too
        base_match = (ap_phone == phone_digits and ap_time_norm == target_norm)
        dob_ok = (True if dob_norm is None else (ap_dob_norm == dob_norm))

        if base_match and dob_ok:
            removed += 1
            debug_print(f"🗑️ Removing appt → phone={ap_phone}, time={ap_time_norm}, dob={ap_dob_norm or '∅'}")
            # don't append → this record is deleted
        else:
            kept.append(appt)

    if removed == 0:
        msg = f"⚠️ No appointment found for phone={phone_digits} time={target_norm}"
        if dob_norm is not None:
            msg += f" dob={dob_norm}"
        debug_print(msg)
        return False

    # Write updated file + refresh in-memory cache if you keep one
    try:
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(kept, f, indent=2)
        doctor_appointments[key] = kept
        debug_print(f"✅ Deleted {removed} appt(s) from {full_path}")
        return True
    except Exception as e:
        debug_print(f"❌ Write error {full_path} → {e}")
        return False






from googleapiclient.discovery import build

def list_events_in_window_utc(calendar_id: str, creds, utc_start: str, utc_end: str, debug: bool=False):
    """
    Low-level fetch of events in a UTC window [timeMin, timeMax].
    Returns a list of event dicts (possibly empty).
    """
    try:
        if debug:
            debug_print(f"📅 list_events_in_window_utc: calendar={calendar_id}")
            debug_print(f"⏱️ timeMin={utc_start}, timeMax={utc_end}")
        service = build("calendar", "v3", credentials=creds)
        resp = service.events().list(
            calendarId=calendar_id,
            timeMin=utc_start,
            timeMax=utc_end,
            singleEvents=True,
            orderBy="startTime"
        ).execute()
        events = resp.get("items", []) or []
        if debug:
            debug_print(f"🔍 list_events_in_window_utc: found {len(events)} event(s)")
        return events
    except Exception as e:
        debug_print(f"❌ list_events_in_window_utc error → {e}")
        return []


def get_upcoming_events(calendar_id: str, phone: str, utc_start: str, utc_end: str, creds, debug: bool=False):
    """
    Search a specific Google Calendar for events within a given UTC time window
    and return the first event whose description contains the caller's phone number.

    Arguments:
    ----------
    calendar_id : str
        The Google Calendar ID where the search should be performed (e.g., "doctor@example.com").
    phone : str
        The caller's phone number in any format (will be normalized to digits only).
    utc_start : str
        The ISO 8601 formatted UTC start time of the search window (e.g., "2025-08-07T14:00:00Z").
    utc_end : str
        The ISO 8601 formatted UTC end time of the search window.
    creds :
        An authenticated Google API credentials object to authorize the Calendar API requests.
    debug : bool, optional
        If True, prints detailed debug logs for troubleshooting.

    Returns:
    --------
    dict or None
        The first matching Google Calendar event (full event dictionary) if found,
        otherwise None.
    """

    # 1️⃣ Normalize the phone number so we can reliably match it against event descriptions.
    #    This strips all non-digit characters (e.g., spaces, dashes, parentheses, etc.)
    phone_digits = normalize_phone_digits(phone)

    # 2️⃣ If debugging is enabled, print the search parameters.
    if debug:
        debug_print(f"📅 get_upcoming_events: Searching in calendar → {calendar_id}")
        debug_print(f"⏱️ Time window → {utc_start} to {utc_end}")
        debug_print(f"📞 Looking for phone digits → {phone_digits}")

    # 3️⃣ Retrieve all events from the given calendar in the specified UTC time window.
    #    This is done via a helper function that wraps the Google Calendar API request.
    events = list_events_in_window_utc(calendar_id, creds, utc_start, utc_end, debug=debug)

    # 4️⃣ Log how many events were retrieved in the time range (useful for diagnostics).
    if debug:
        debug_print(f"🔍 get_upcoming_events: Found {len(events)} event(s) in the time window")

    # 5️⃣ Loop through all events to look for one that matches our phone number.
    for ev in events:
        # Extract the event description (can contain customer details such as phone, name, address).
        # Ensure it's a string, default to empty if missing.
        desc = ev.get("description", "") or ""

        # Normalize the digits from the event description so we can compare to the caller's digits.
        desc_digits = normalize_phone_digits(desc)

        # 6️⃣ For debugging: print the event summary, start time, and extracted phone digits from the description.
        if debug:
            # Try to get the start datetime in a consistent format.
            start_dbg = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date")
            debug_print(f"📝 Event → summary={ev.get('summary')} start={start_dbg} desc_digits={desc_digits}")

        # 7️⃣ If the caller's phone digits are non-empty AND found within the event description's digits → MATCH.
        if phone_digits and phone_digits in desc_digits:
            if debug:
                debug_print("✅ Match found by phone number in event description")
            return ev  # Return the entire event dictionary.

    # 8️⃣ If no event matched the phone number, log this (if debugging) and return None.
    if debug:
        debug_print("❌ No matching event found with the provided phone number.")
    return None
##
###    DOB  parsing and processing
##

from datetime import datetime

ORDINALS = {
    "first":"1","second":"2","third":"3","fourth":"4","fifth":"5","sixth":"6","seventh":"7","eighth":"8","ninth":"9","tenth":"10",
    "eleventh":"11","twelfth":"12","thirteenth":"13","fourteenth":"14","fifteenth":"15","sixteenth":"16","seventeenth":"17",
    "eighteenth":"18","nineteenth":"19","twentieth":"20","twenty-first":"21","twentyfirst":"21","twenty-second":"22","twentysecond":"22",
    "twenty-third":"23","twentythird":"23","twenty-fourth":"24","twentyfourth":"24","twenty-fifth":"25","twentyfifth":"25",
    "twenty-sixth":"26","twentysixth":"26","twenty-seventh":"27","twentyseventh":"27","twenty-eighth":"28","twentyeighth":"28",
    "twenty-ninth":"29","twentyninth":"29","thirtieth":"30","thirty-first":"31","thirtyfirst":"31"
}

MONTHS = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12
}

def _clean_ordinals(text: str) -> str:
    # replace 'third'->'3', remove commas/periods, strip extra spaces
    t = text.lower()
    for k,v in ORDINALS.items():
        t = t.replace(k, v)
    t = t.replace(",", " ").replace(".", " ").replace("  ", " ").strip()
    return t

from typing import Optional
from datetime import datetime

def parse_dob_input(speech_text: str, dtmf_digits: str) -> Optional[datetime]:
    """
    Date of Birth parser (speech preferred):
    - First try spoken formats like 'July 3 1990' or '3 July 1990'
    - If that fails, try keypad DTMF in MMDDYYYY format
    Returns a datetime.date (as datetime) or None.
    """

    # 1) Spoken path first
    if speech_text:
        cleaned = _clean_ordinals(speech_text)  # Remove 'st', 'nd', 'rd', 'th'
        parts = cleaned.split()

        try:
            # Month Day Year
            for i, p in enumerate(parts):
                if p in MONTHS and i+2 < len(parts):
                    month = MONTHS[p]
                    day = int(parts[i+1])
                    year = int(parts[i+2])
                    return datetime(year, month, day)

            # Day Month Year
            for i, p in enumerate(parts):
                if p.isdigit() and i+2 < len(parts):
                    day = int(p)
                    mword = parts[i+1]
                    if mword in MONTHS:
                        month = MONTHS[mword]
                        year = int(parts[i+2])
                        return datetime(year, month, day)
        except Exception:
            pass

        # Forgiving natural language parse
        try:
            from dateutil import parser as dtparser
            dt = dtparser.parse(speech_text, dayfirst=False, fuzzy=True)
            return datetime(dt.year, dt.month, dt.day)
        except Exception:
            pass

    # 2) DTMF fallback
    if dtmf_digits and dtmf_digits.isdigit():
        if len(dtmf_digits) == 8:
            mm = int(dtmf_digits[0:2])
            dd = int(dtmf_digits[2:4])
            yyyy = int(dtmf_digits[4:8])
            try:
                return datetime(yyyy, mm, dd)
            except ValueError:
                return None
        return None  # Invalid length

    return None


def make_gather_dob(prompt_text: str):
    """
    DOB gather that delegates to the shared make_gather helper:
    - Reuses your standard 'can't hear you' behavior (silence re-prompt).
    - Adds month-name hints for better speech recognition.
    - Prompt explains speech OR keypad entry (MMDDYYYY + #).
    NOTE: Assumes make_gather() is configured to accept speech + DTMF.
    """
    month_hints = "january,february,march,april,may,june,july,august,september,october,november,december"
    return make_gather(
        (
            f"{prompt_text} "
            "You can say it, for example, 'July third 1990', "
            "or type two digits for month, two digits for day, and four digits for year, "
            "then press pound. For example, 07031990#."
        ),
        hints=month_hints
    )


############################################
##  Customer DB
#############################################
import os
import json
from datetime import datetime

# ----------------------------------------------------------------------
# Path under the existing appointment_data folder
# ----------------------------------------------------------------------

# =============================================================================
# Human-readable customer "DB" in ONE file (appointment_data/customers.json)
# - Each customer is exactly ONE block of 12 lines (no JSON lines).
# - Match key: (Phone, DOB)
# - New customer  -> append a block.
# - Existing      -> update that block IN PLACE (no duplicate).
# - PAN/CVV are MASKED in the file.
# - All scans/updates are simple sequential text processing.
# =============================================================================
import os, re
from datetime import datetime

# ---------- Config ----------
DB_FOLDER = "appointment_data"
DB_FILE   = os.path.join(DB_FOLDER, "customers.json")  # human-readable, not JSON

# ---------- Logging helper ----------
try:
    debug_print  # type: ignore # will raise if not defined
except NameError:  # minimal fallback so this module is self-contained
    def debug_print(*args, **kwargs):
        print(*args, **kwargs)

# ---------- Init ----------
def init_db():
    """Ensure DB folder/file exist (creates empty file if missing)."""
    os.makedirs(DB_FOLDER, exist_ok=True)
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w", encoding="utf-8"):
            pass

# ---------- Sanitizers / formatters ----------
def _oneline(s: str) -> str:
    """Collapse whitespace/newlines to single spaces; trim."""
    return re.sub(r"\s+", " ", (s or "").strip())

def _normalize_phone(s: str) -> str:
    """
    Keep digits only; if NANP 11-digit starting with '1', strip leading '1'.
    Returns 10-digit for US numbers where applicable; no validation beyond that.
    """
    d = "".join(ch for ch in (s or "") if ch.isdigit())
    return d[1:] if len(d) == 11 and d.startswith("1") else d

def _mask_pan(n: str) -> str:
    """Mask a PAN for storage/logs (keep last 4)."""
    n = (n or "").strip()
    return ("*" * max(0, len(n) - 4)) + n[-4:] if n else ""

def _mask_all(n: str) -> str:
    """Mask entire sensitive string (e.g., CVV)."""
    return "*" * len((n or "").strip())

def _block_title(new: bool) -> str:
    return "insert_customer: ✅ Added new customer" if new \
           else "insert_customer: ℹ️ Existing customer — updated last_seen_at"

def _render_block_lines(new: bool, rec: dict) -> list[str]:
    """
    Render the 12-line, human-readable block for a customer.
    PAN/CVV are MASKED here so the file never stores raw values.
    """
    return [
        _block_title(new),
        f"Phone: {rec.get('phone') or '—'}",
        f"DOB: {rec.get('dob') or '—'}",
        f"First Name: {rec.get('first_name') or '—'}",
        f"Last Name: {rec.get('last_name') or '—'}",
        f"Address: {rec.get('address') or '—'}",
        f"CC Name: {rec.get('cc_name') or '—'}",
        f"CC Number: {_mask_pan(rec.get('cc_number'))}",
        f"CC Exp: {rec.get('cc_exp') or '—'}",
        f"CC CVV: {_mask_all(rec.get('cc_cvv'))}",
        f"Created At: {rec.get('created_at') or '—'}",
        f"Last Seen At: {rec.get('last_seen_at') or '—'}",
    ]

# ---------- File parsing helpers ----------
def _iter_blocks(lines: list[str]):
    """
    Yield (start_idx, end_idx_exclusive, block_lines).
    A block starts at a line beginning with 'insert_customer:' and ends
    right before the next such line (or EOF).
    """
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith("insert_customer:"):
            if start is not None:
                yield (start, i, lines[start:i])
            start = i
    if start is not None:
        yield (start, len(lines), lines[start:])

def _get_value(block_lines: list[str], label: str) -> str | None:
    """Fetch 'Label: value' from a block."""
    prefix = f"{label}:"
    for ln in block_lines:
        if ln.startswith(prefix):
            return ln.split(":", 1)[1].strip()
    return None

def _extract_phone_dob(block_lines: list[str]) -> tuple[str | None, str | None]:
    """Get (Phone, DOB) from a block."""
    return _get_value(block_lines, "Phone"), _get_value(block_lines, "DOB")

# ---------- Public API ----------
def customer_search(phone: str, dob: str) -> bool:
    """
    Sequentially scan the human-readable file and return True if a block
    exists with matching (Phone, DOB). Simple, O(n) pass.
    """
    phone_norm = _normalize_phone(phone)
    dob_clean  = _oneline(dob)

    if not os.path.exists(DB_FILE):
        return False

    with open(DB_FILE, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    for _, _, blk in _iter_blocks(lines):
        b_phone, b_dob = _extract_phone_dob(blk)
        if _normalize_phone(b_phone) == phone_norm and (b_dob or "") == dob_clean:
            return True
    return False

def _update_existing_block_in_place(phone_norm: str, dob_clean: str, updates: dict) -> bool:
    """
    Edit the matching block in place:
      - Always bump 'Last Seen At' to now
      - If updates contain non-empty values, refresh:
          First Name, Last Name, Address, CC Name, CC Number, CC Exp, CC CVV
      - Preserve original 'Created At' and title line
    Returns True if a block was updated.
    """
    if not os.path.exists(DB_FILE):
        return False

    with open(DB_FILE, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    changed = False
    out: list[str] = []
    i = 0

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("insert_customer:"):
            start = i
            i += 1
            while i < len(lines) and not lines[i].startswith("insert_customer:"):
                i += 1
            block = lines[start:i]

            b_phone, b_dob = _extract_phone_dob(block)
            if _normalize_phone(b_phone) == phone_norm and (b_dob or "") == dob_clean:
                # Pull existing values
                cur = {
                    "title":       block[0],
                    "phone":       _get_value(block, "Phone") or "",
                    "dob":         _get_value(block, "DOB") or "",
                    "first_name":  _get_value(block, "First Name") or "",
                    "last_name":   _get_value(block, "Last Name") or "",
                    "address":     _get_value(block, "Address") or "",
                    "cc_name":     _get_value(block, "CC Name") or "",
                    "cc_number":   _get_value(block, "CC Number") or "",  # masked in file
                    "cc_exp":      _get_value(block, "CC Exp") or "",
                    "cc_cvv":      _get_value(block, "CC CVV") or "",     # masked in file
                    "created_at":  _get_value(block, "Created At") or "—",
                    "last_seen_at": now,
                }

                # Apply non-empty updates (sanitize to one line)
                def pick(new_val, old_val):
                    new_val = _oneline(new_val)
                    return new_val if new_val else old_val

                cur["first_name"] = pick(updates.get("first_name"), cur["first_name"])
                cur["last_name"]  = pick(updates.get("last_name"),  cur["last_name"])
                cur["address"]    = pick(updates.get("address"),    cur["address"])
                cur["cc_name"]    = pick(updates.get("cc_name"),    cur["cc_name"])
                if _oneline(updates.get("cc_number")):
                    cur["cc_number"] = updates["cc_number"]
                if _oneline(updates.get("cc_exp")):
                    cur["cc_exp"] = updates["cc_exp"]
                if _oneline(updates.get("cc_cvv")):
                    cur["cc_cvv"] = updates["cc_cvv"]

                # Re-render; keep original title text
                new_block = _render_block_lines(new=True, rec=cur)
                new_block[0] = cur["title"]
                out.extend(new_block)
                changed = True
            else:
                out.extend(block)
        else:
            out.append(ln)
            i += 1

    if changed:
        with open(DB_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + ("\n" if out else ""))

    return changed

def insert_customer(phone: str, dob: str, first_name: str, last_name: str, address: str,
                    cc_name: str, cc_number: str, cc_exp: str, cc_cvv: str) -> bool:
    """
    Append/Update a customer in the single human-readable file:
      • NEW customer → append one 12-line block (masked PAN/CVV).
      • EXISTING     → update that block IN PLACE (no duplicate) and bump 'Last Seen At'.
    Returns:
      True  -> new block appended
      False -> existing block updated in place
    """
    init_db()

    phone_norm = _normalize_phone(phone)
    dob_clean  = _oneline(dob)

    # Existing? Update in place (no append)
    if customer_search(phone_norm, dob_clean):
        _update_existing_block_in_place(
            phone_norm, dob_clean,
            updates={
                "first_name": first_name,
                "last_name":  last_name,
                "address":    address,
                "cc_name":    cc_name,
                "cc_number":  cc_number,
                "cc_exp":     cc_exp,
                "cc_cvv":     cc_cvv,
            }
        )
        debug_print("\n".join([
            "insert_customer: ℹ️ Existing customer — updated in place (no duplicate)",
            f"Phone: {phone_norm}",
            f"DOB: {dob_clean}",
        ]))
        return False

    # New record → append block
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rec = {
        "phone":       phone_norm,
        "dob":         dob_clean,
        "first_name":  _oneline(first_name),
        "last_name":   _oneline(last_name),
        "address":     _oneline(address),
        "cc_name":     _oneline(cc_name),
        "cc_number":   _oneline(cc_number),  # will be masked by renderer
        "cc_exp":      _oneline(cc_exp),
        "cc_cvv":      _oneline(cc_cvv),     # will be masked by renderer
        "created_at":  now,
        "last_seen_at": now
    }

    with open(DB_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(_render_block_lines(new=True, rec=rec)) + "\n")

    debug_print("\n".join([
        "insert_customer: ✅ Added new customer",
        f"Phone: {rec['phone']}",
        f"DOB: {rec['dob']}",
        f"First Name: {rec['first_name']}",
        f"Last Name: {rec['last_name']}",
        f"Address: {rec['address']}",
        f"CC Name: {rec['cc_name']}",
        f"CC Number: {_mask_pan(rec['cc_number'])}",
        f"CC Exp: {rec['cc_exp']}",
        f"CC CVV: {_mask_all(rec['cc_cvv'])}",
        f"Created At: {rec['created_at']}",
        f"Last Seen At: {rec['last_seen_at']}",
    ]))

    return True


def _normalize_mmyy(s: str) -> str:
    """Return 'MM/YY' from inputs like 'MMYY', 'M/YY', 'MM/YY'. Leave empty if unusable."""
    s = (s or "").strip()
    if not s:
        return ""
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) == 4:  # e.g., '0229'
        mm, yy = digits[:2], digits[2:]
    else:
        # Try to parse formats with slash; e.g., '2/29', '02/29'
        parts = s.split("/")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            mm = parts[0].zfill(2)
            yy = parts[1][-2:].zfill(2)
        else:
            return ""
    # Basic month guard
    try:
        m = int(mm)
        if not (1 <= m <= 12):
            return ""
    except ValueError:
        return ""
    return f"{mm}/{yy}"

def update_customer_cc(
    phone: str,
    dob: str,
    cc_name: str | None = None,
    cc_number: str | None = None,
    cc_exp: str | None = None,   # accepts 'MMYY' or 'MM/YY'
    cc_cvv: str | None = None
) -> bool:
    """
    Update ONLY the CC fields for an existing customer (matched by Phone + DOB).
    - Edits the single block IN PLACE; does not append/duplicate.
    - Always bumps 'Last Seen At'.
    - Fields left as None are NOT changed.
    Returns:
      True  -> updated successfully
      False -> no matching customer found (nothing changed)
    """
    init_db()

    phone_norm = _normalize_phone(phone)
    dob_clean  = _oneline(dob)

    if not phone_norm or not dob_clean:
        debug_print("update_customer_cc: ❌ phone or dob missing/blank")
        return False

    # Prepare updates dict (only include non-empty values so we don't overwrite)
    updates: dict[str, str] = {}

    if cc_name is not None and _oneline(cc_name):
        updates["cc_name"] = _oneline(cc_name)

    if cc_number is not None and _oneline(cc_number):
        # Do NOT mask here; _update_existing_block_in_place -> _render_block_lines will mask on write.
        updates["cc_number"] = "".join(ch for ch in cc_number if ch.isdigit())

    if cc_exp is not None and _oneline(cc_exp):
        norm_exp = _normalize_mmyy(cc_exp)
        if norm_exp:
            updates["cc_exp"] = norm_exp
        else:
            debug_print(f"update_customer_cc: ⚠️ ignoring invalid expiration '{cc_exp}'")

    if cc_cvv is not None and _oneline(cc_cvv):
        updates["cc_cvv"] = "".join(ch for ch in cc_cvv if ch.isdigit())

    if not updates:
        debug_print("update_customer_cc: ℹ️ no CC fields provided to update")
        return False

    changed = _update_existing_block_in_place(phone_norm, dob_clean, updates)
    if not changed:
        debug_print("\n".join([
            "update_customer_cc: ❌ no matching customer found",
            f"Phone: {phone_norm}",
            f"DOB: {dob_clean}",
        ]))
        return False

    # Log (masked)
    def _mask_pan(n: str) -> str:
        n = n or ""
        return ("*" * max(0, len(n) - 4)) + n[-4:] if n else ""
    def _mask_all(n: str) -> str:
        return "*" * len((n or "").strip())

    debug_print("\n".join([
        "update_customer_cc: ✅ CC info updated",
        f"Phone: {phone_norm}",
        f"DOB: {dob_clean}",
        f"CC Name: {updates.get('cc_name', '—') if 'cc_name' in updates else '—'}",
        f"CC Number: {_mask_pan(updates.get('cc_number')) if 'cc_number' in updates else '—'}",
        f"CC Exp: {updates.get('cc_exp', '—') if 'cc_exp' in updates else '—'}",
        f"CC CVV: {_mask_all(updates.get('cc_cvv')) if 'cc_cvv' in updates else '—'}",
    ]))

    return True






def find_future_events_for_caller(
    calendars,
    phone: str,
    dob: str = None,
    creds=None,
    *,
    start_utc: str = None,
    end_utc: str = None,
    horizon_days: int = 90,
    limit: int = 25,
    tz_name: str = "America/Chicago",
    dr_map: dict = None,          # fallback to global googleid_dr_name_map if None
    debug: bool = False,
):
    """
    Fetch upcoming Google Calendar events for the caller, filtered by phone (required)
    and optional DOB. Returns a list of normalized 'candidate' dicts ready for
    cancellation iteration or confirmation.

    Return schema for each candidate:
        {
          "event_id": str,       # GC event id
          "calendar_id": str,    # calendar we found it in
          "doctor_name": str,    # friendly name from map
          "start_utc": str,      # ISO UTC start
          "end_utc": str,        # ISO UTC end
          "friendly": str,       # local time phrase for TTS
          "summary": str,        # optional GC title
          "location": str,       # optional GC location
          "phone": str,          # normalized phone we extracted from event
          "dob": str,            # ISO DOB we extracted from event
          # "raw_event": dict    # only present if debug=True
        }
    """
    import re                                             # regex utils for parsing description text and DOB formats
    from datetime import datetime, timedelta, timezone    # for time windows and UTC handling
    from googleapiclient.discovery import build           # Google Calendar API client
    try:
        from dateutil import parser as dtparser           # robust ISO-ish datetime parsing
    except Exception:
        raise                                             # bubble up if dateutil is missing

    # --------- small helpers (scoped to this function) ---------

    def debug_print_safe(msg: str):
        """Call your app's debug_print if available; otherwise fall back to print."""
        try:
            debug_print(msg)                              # prefer your injected logger
        except Exception:
            print(msg)                                    # safe fallback in isolated contexts

    def _normalize_phone_digits(s: str) -> str:
        """Keep only digits; strip leading '1' from 11-digit US numbers → 10 digits."""
        d = "".join(ch for ch in (s or "") if ch.isdigit())         # remove non-digits
        return d[1:] if len(d) == 11 and d.startswith("1") else d   # normalize US style

    def _normalize_dob(s: str) -> str:
        """
        Normalize DOB to ISO 'YYYY-MM-DD' when possible.
        - Trim whitespace.
        - If a datetime-like string is passed, keep only date part before 'T'.
        - Accept 'YYYY-MM-DD', 'MM/DD/YYYY', 'MM-DD-YYYY'; otherwise return as-is.
        """
        s = (s or "").strip()                                       # clean input
        if not s:
            return ""                                                # no DOB available
        if "T" in s:                                                 # drop time portion if present
            s = s.split("T", 1)[0].strip()
        m = re.match(r"^\s*(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})\s*$", s)  # try MM/DD/YYYY or MM-DD-YYYY
        if m:
            mm, dd, yyyy = m.groups()
            try:
                return f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}"       # standardize
            except Exception:
                return s                                                    # fallback unchanged
        if re.match(r"^\d{4}\-\d{2}\-\d{2}$", s):                            # already ISO date
            return s
        return s                                                             # unknown pattern → return as-is

    def _to_utc_iso(dt_like: str) -> str:
        """Parse any ISO-ish datetime/date string and return strict UTC ISO with +00:00 offset."""
        dt = dtparser.isoparse(dt_like)                           # parse flexible ISO formats
        return dt.astimezone(timezone.utc).isoformat()            # convert to UTC and serialize

    def _event_dt_to_utc_iso(ev_when: dict, key: str) -> str:
        """
        From a Google event's 'start'/'end' dict, prefer 'dateTime' else 'date' (all-day).
        Return UTC ISO string or raise if missing.
        """
        val = (ev_when or {}).get("dateTime") or (ev_when or {}).get("date")  # pick available field
        if not val:
            raise ValueError("Missing event time")                             # invalid event payload
        return _to_utc_iso(val)                                                # normalize to UTC

    def _to_local_friendly(utc_iso: str) -> str:
        """Render a UTC ISO timestamp into a local (tz_name) speech-friendly string."""
        import pytz
        try:
            dt_utc = dtparser.isoparse(utc_iso)                                # parse ISO
            local = dt_utc.astimezone(pytz.timezone(tz_name))                   # convert to clinic TZ
            try:
                return local.strftime("%A, %B %-d at %-I:%M %p")               # GNU/Unix
            except Exception:
                return local.strftime("%A, %B %d at %I:%M %p").replace(" 0", " ")  # Windows-friendly
        except Exception:
            return utc_iso                                                      # fallback: return raw ISO

    def _extract_phone_dob_from_event(ev: dict) -> tuple[str, str]:
        """
        Pull phone and DOB from event metadata.
        Priority:
          1) extendedProperties.private.phone / .dob
          2) description: parse digits & common date patterns
        Return: (normalized_phone, normalized_dob)
        """
        phone_found, dob_found = "", ""                                         # defaults

        # Try GCal extendedProperties first (if your create routine stored these)
        extp = (ev.get("extendedProperties") or {}).get("private") or {}        # safe dict
        if isinstance(extp, dict):
            phone_found = _normalize_phone_digits(extp.get("phone") or extp.get("Phone") or "")
            dob_found   = _normalize_dob(extp.get("dob") or extp.get("DOB") or "")

        # Fallback to parsing the description body if still missing
        if (not phone_found) or (not dob_found):
            desc = ev.get("description") or ""                                  # free-text notes
            digits = "".join(ch for ch in desc if ch.isdigit())                 # all digits in description
            if not phone_found and len(digits) >= 10:
                phone_found = _normalize_phone_digits(digits[-10:])             # take last 10 as US phone
            m = re.search(r"\b(\d{4}\-\d{2}\-\d{2})\b", desc)                   # look for YYYY-MM-DD
            if not dob_found and m:
                dob_found = _normalize_dob(m.group(1))
            else:
                m2 = re.search(r"\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})\b", desc)  # or MM/DD/YYYY
                if not dob_found and m2:
                    dob_found = _normalize_dob(m2.group(0))

        return phone_found, dob_found                                           # return normalized values

    # --------- normalize/prepare inputs ---------

    # Ensure 'calendars' is a list; accept a single id, tuple, or any iterable.
    if isinstance(calendars, (str, bytes)):
        calendars = [calendars]                                                 # single → list
    elif isinstance(calendars, tuple):
        calendars = list(calendars)                                             # tuple → list
    elif not isinstance(calendars, list):
        calendars = list(calendars or [])                                       # generic iterable → list

    # Prefer provided doctor map; otherwise use global mapping if present.
    try:
        dr_map = dr_map or googleid_dr_name_map                                 # global fallback
    except NameError:
        dr_map = {}                                                             # empty if global missing

    phone_norm_input = _normalize_phone_digits(phone)                           # normalize caller phone
    dob_norm_input   = _normalize_dob(dob) if dob else ""                       # normalize DOB if provided

    # Build time window for the search:
    # - lower bound (timeMin): now (UTC) unless a start_utc is provided
    # - upper bound (timeMax): start + horizon_days unless end_utc is provided
    now_utc = datetime.now(timezone.utc)                                        # current time in UTC
    if start_utc:
        try:
            time_min = _to_utc_iso(start_utc)                                   # normalize provided lower bound
        except Exception:
            time_min = now_utc.isoformat()                                      # fallback to now if invalid
    else:
        time_min = now_utc.isoformat()                                          # default lower bound = now

    if end_utc:
        try:
            time_max = _to_utc_iso(end_utc)                                     # normalize provided upper bound
        except Exception:
            time_max = (now_utc + timedelta(days=horizon_days)).isoformat()     # fallback to horizon
    else:
        time_max = (now_utc + timedelta(days=horizon_days)).isoformat()         # default upper bound

    if debug:
        debug_print_safe(f"find_future_events_for_caller: 📞 phone={phone_norm_input} dob={dob_norm_input or '∅'}")
        debug_print_safe(f"find_future_events_for_caller: ⏱️ timeMin={time_min} timeMax={time_max}")
        debug_print_safe(f"find_future_events_for_caller: 🗓️ calendars={calendars}")

    # Create Google Calendar API service client using provided creds
    service = build("calendar", "v3", credentials=creds)

    results = []                                                                # will hold normalized candidates

    # Iterate each calendar (doctor); stop when we hit 'limit' total results
    for cal_id in calendars:
        if len(results) >= limit:
            break                                                               # respect global cap

        page_token = None                                                       # GC pagination token
        fetched = 0                                                             # count of raw events fetched for this cal

        while True:
            if len(results) >= limit:
                break                                                           # stop early if we filled results

            try:
                # Request a page of events for this calendar within [timeMin, timeMax]
                resp = service.events().list(
                    calendarId=cal_id,
                    timeMin=time_min,
                    timeMax=time_max,
                    singleEvents=True,                                          # expand recurring events
                    orderBy="startTime",                                        # chronological order
                    pageToken=page_token,
                    maxResults=min(250, max(10, limit - len(results))),         # sensible page size
                ).execute()
            except Exception as e:
                if debug:
                    debug_print_safe(f"find_future_events_for_caller: ❌ list error for {cal_id} → {e}")
                break                                                           # skip this calendar on error

            items = resp.get("items", []) or []                                 # raw events for this page
            fetched += len(items)                                               # track how many retrieved

            for ev in items:
                # Normalize event start/end to UTC; skip malformed
                try:
                    start_utc_iso = _event_dt_to_utc_iso(ev.get("start"), "start")
                    end_utc_iso   = _event_dt_to_utc_iso(ev.get("end"), "end")
                except Exception as e:
                    if debug:
                        debug_print_safe(f"find_future_events_for_caller: ⚠️ skip malformed times → {e}")
                    continue                                                    # skip bad event timestamps

                # Try to pull phone/DOB from event metadata/description
                ev_phone, ev_dob = _extract_phone_dob_from_event(ev)
                ev_phone_norm = _normalize_phone_digits(ev_phone)               # normalize for comparison
                ev_dob_norm   = _normalize_dob(ev_dob)

                # Filter: phone must match; if caller supplied DOB, that must match too
                if ev_phone_norm != phone_norm_input:
                    continue                                                    # wrong person → skip
                if dob_norm_input and ev_dob_norm != dob_norm_input:
                    continue                                                    # DOB mismatch → skip

                # Resolve friendly doctor name from mapping (fallback label if unknown)
                doctor_name = dr_map.get(cal_id, "the doctor")

                # Render a human-friendly local time (e.g., "Tuesday, August 12 at 9:00 AM")
                friendly = _to_local_friendly(start_utc_iso)

                # Build normalized candidate (what your iterator & confirm stages expect)
                cand = {
                    "event_id": ev.get("id", ""),                               # GC event id
                    "calendar_id": cal_id,                                      # which calendar
                    "doctor_name": doctor_name,                                 # friendly doctor label
                    "start_utc": start_utc_iso,                                 # normalized UTC start
                    "end_utc": end_utc_iso,                                     # normalized UTC end
                    "friendly": friendly,                                       # human-readable local time
                    "summary": ev.get("summary", "") or "",                     # optional title
                    "location": ev.get("location", "") or "",                   # optional location
                    "phone": ev_phone_norm,                                     # the matched phone
                    "dob": ev_dob_norm,                                         # the matched DOB (possibly empty)
                }
                if debug:
                    cand["raw_event"] = ev                                      # include raw payload when debugging

                results.append(cand)                                            # add to output list
                if len(results) >= limit:
                    break                                                       # hit global cap → stop inner loop

            page_token = resp.get("nextPageToken")                               # move to next page if any
            if not page_token:
                break                                                            # no more pages for this calendar

        if debug:
            debug_print_safe(
                f"find_future_events_for_caller: ✅ calendar={cal_id} fetched={fetched}, kept={len(results)} total"
            )

    # Sort results chronologically by UTC start; ignore errors gracefully
    try:
        results.sort(key=lambda r: r["start_utc"])                               # earliest first
    except Exception:
        pass                                                                     # leave unsorted if something odd

    if debug:
        debug_print_safe(f"find_future_events_for_caller: ✅ returning {len(results)} candidate(s)")

    return results                                                               # list[dict] normalized events




#app = Flask(__name__)

from flask import request
from twilio.twiml.messaging_response import MessagingResponse
import os
from datetime import datetime

@app.route("/sms", methods=["POST"])
def sms_reply():
    """
    This function handles incoming SMS messages from Twilio.
    It stores each message in a file named after the sender's phone number
    and replies with a polite message using GPT-based phrasing.
    """

    # 📩 Step 1: Extract the message content and sender's phone number
    incoming_msg = request.form.get("Body")        # SMS text content
    from_number = request.form.get("From")         # Phone number (e.g., +12025550123)

    print(f"📩 Message from {from_number}: {incoming_msg}")

    # 📂 Step 2: Prepare the file path using the sender's phone number
    sms_dir = "sms"
    os.makedirs(sms_dir, exist_ok=True)            # Ensure the directory exists

    # Create a safe filename by stripping '+' and non-digit characters
    safe_number = ''.join(c for c in from_number if c.isdigit())
    filepath = os.path.join(sms_dir, f"{safe_number}.txt")

    # 🕓 Step 3: Append the message to the sender's file with a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {incoming_msg}\n")

    # 🤖 Step 4: Generate a polite reply using GPT-powered phrasing
    reply_message = gpt_speak("Thanks for your message! We’ll get back to you shortly.")

    # 📤 Step 5: Send the reply back to Twilio as an SMS
    resp = MessagingResponse()
    resp.message(reply_message)

    return str(resp)


@app.route("/transcription", methods=["POST"])
def transcription():
        # Import the Twilio client to send SMS messages via Twilio's REST API
        from twilio.rest import Client

        # Initialize the Twilio client with your Account SID and Auth Token,
        # which are stored securely in environment variables
        client = Client(
            os.getenv("TWILIO_ACCOUNT_SID"),  # Your Twilio Account SID
            os.getenv("TWILIO_AUTH_TOKEN")    # Your Twilio Auth Token
        )

        # Extract the transcribed text from the POST request sent by Twilio after voicemail transcription
        # If the transcription failed or isn't present, it will default to an empty string
        text = request.values.get("TranscriptionText", "")

        # Extract the URL of the voicemail audio file
        # Twilio provides the base URL without the ".mp3" extension, so we append it manually
        url = request.values.get("RecordingUrl", "") + ".mp3"

        # Loop over all admin phone numbers (e.g., clinic staff or receptionists)
        # and send each of them a text message with the transcription and audio link
        for admin in admin_numbers:
            client.messages.create(
                to=admin,                     # Recipient's phone number (an admin)
                from_=TWILIO_PHONE_NUMBER,         # Your Twilio phone number used to send the SMS
                body=f"Voicemail:\n{text}\nAudio: {url}"  # Message body with both transcription and link to audio
            )

        # Return an empty HTTP response with status code 204 (No Content),
        # indicating that the webhook was successfully handled but there's no response content needed
        return "", 204

import re  # Used for name normalization

# 🔧 Helper function to normalize and clean names
def normalize(text):
    """
    Lowercase, remove punctuation, and trim extra spaces from text.
    """
    return re.sub(r"[^a-zA-Z\s]", "", text).lower().strip()
@app.route("/voice", methods=["POST"])
@app.route("/voice/", methods=["POST"])  # Accepts trailing slash
def voice():
    # Create a new TwiML VoiceResponse object to build the voice reply to the caller
    resp = VoiceResponse()
    

    # Extract the unique call ID (SID) from the request parameters to track the session
    call_sid = request.values.get("CallSid", "")

    # Retrieve the customer's speech input (transcribed by Twilio's Speech-to-Text)
    speech_result = request.values.get("SpeechResult", "").strip()
    print(f"📢 voice :speech_result: {speech_result}")
    # Determine the current interaction stage (default to "intro" if not previously set)
    stage = session_data.get(call_sid, {}).get("stage", "intro")
    """
    # What happens in this stage:
    # The caller calls the clinic.

    # Twilio sends a webhook to your /voice endpoint.

    # You respond with a greeting prompt, dynamically generated using ChatGPT.

    # You ask: “Would you like to book an appointment or leave a message?”

    # The system listens for speech and sends the result back to the same endpoint (/voice) using a POST request.

    # The session progresses from "intro" to "intent" for next steps.
    # If this is the start of the call, begin with the "intro" stage
    """
    if stage == "intro":

        # Save the current session state as moving to the next stage ("intent")
        session_data[call_sid] = {"stage": "intent"}

        # Define a friendly prompt to ask the customer what they want to do
        prompt = (
            "Thank you for calling EPIC therapist. "
            "Would you like to book an appointment, cancel an appointment, "
            "change an appointment, or leave a message?"
        )

        # Create a <Gather> TwiML block using our helper that:
        # - Speaks the prompt with GPT voice
        # - Listens for the caller’s voice input
        # - If silence / no input, re-prompts with 'I can't hear you...'
        # - Sends the speech result to /voice for further processing
        gather = make_gather(prompt)

        """
        Speaks the message inside <Say>
        Listens for the caller’s voice input for SPEECH_INPUT_DURATION seconds
        Sends the speech result to /voice for further handling

        <Response>
        <Gather ...>  <!-- created via make_gather(...) -->
            <Say>Thank you for calling EPIC therapist...</Say>
        </Gather>
        </Response>
        """

        # Append the <Gather> block to the overall TwiML response
        resp.append(gather)

        # Return the XML response as a string (TwiML) to Twilio to speak it to the caller
        return str(resp)


    elif stage == "intent":
        # ----------------------------------------------------------------------
        # 🎯 Intent detection stage: figure out if the caller wants to:
        #  1. Book an appointment
        #  2. Cancel an appointment
        #  3. Reschedule an appointment
        #  4. Leave a voicemail
        # ----------------------------------------------------------------------

        lower = speech_result.lower()
        print(f"📢 intent :speech_result: {lower.strip()}")

        # 🚫 Ignore junk or greeting phrases commonly returned by Twilio
        junk_inputs = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening", "yo", "test", "1", "yes", "no"}
        if not lower.strip() or lower.strip() in junk_inputs:
            print(f"⛔ Ignored junk input: '{lower}' — re-prompting without response")

            # ⬇️ CHANGED: use make_gather (same behavior, cleaner + consistent)
            gather = make_gather(
                "Thank you for Calling EPIC thearapist : Please tell me if you'd like to book an appointment, cancel one, reschedule, or leave a message."
            )
            resp.append(gather)
            return str(resp)

        # ✅ Rescheduling intent
        elif any(word in lower for word in ["change", "move"]):
            print("🔁 Intent to reschedule detected → will cancel then rebook")
            
            # 🧼 Initialize session
            session_data[call_sid] = {
                "stage": "cancel_appointment",
                "cancel": {},
                "retry_booking": 0,
                "reschedule_after_cancel": True
            }

            doctor_names = list(googleid_dr_name_map.values())
            print(f"📋 Available doctor names: {doctor_names}")

            if len(doctor_names) > 1:
                doctor_list = ", ".join(doctor_names[:-1]) + ", or " + doctor_names[-1]
            else:
                doctor_list = doctor_names[0]

            prompt = (
                f"Sure, let's reschedule your appointment. First, we'll cancel your current appointment. "
                f"Available doctors include: {doctor_list}. Please say the name of the doctor you had booked with."
            )

            # 🎤 Prepare re-prompt for doctor name
            # ⬇️ CHANGED: use make_gather with hints
            gather = make_gather(prompt, hints=", ".join(doctor_names))
            resp.append(gather)

            print("🎙️ Prompted user to specify doctor for cancellation as part of reschedule.")
            return str(resp)

        # ✅ Cancellation intent
        elif any(word in lower for word in ["cancel", "delete"]):
            print("❌ Intent to cancel appointment detected → entering cancellation flow")
            session_data[call_sid] = {
                "stage": "cancel_appointment",
                "cancel": {},
                "retry_booking": 0
            }
            doctor_names = list(googleid_dr_name_map.values())
            doctor_list = ", ".join(doctor_names[:-1]) + ", or " + doctor_names[-1] if len(doctor_names) > 1 else doctor_names[0]
            prompt = (
                f"Sure, I can help you cancel your appointment. "
                f"We currently have the following doctors: {doctor_list}. "
                f"Please say the name of the doctor you had booked with."
            )

            # ⬇️ CHANGED: use make_gather with hints
            gather = make_gather(prompt, hints=", ".join(doctor_names))
            resp.append(gather)
            return str(resp)

        # ✅ Booking intent (placed **after** cancel/reschedule to avoid false positives)
        elif any(word in lower for word in ["book", "booking", "schedule", "make","making", "reserve", "meet","meeting","making"]):
            print(f"📅 Intent to book recognized → advancing to 'booking' stage")

            # ✅ Fix: Use update instead of overwrite to preserve previous session info
            session_data.setdefault(call_sid, {})
            session_data[call_sid].update({
                "stage": "booking",
                "booking": {},
                "retry_booking": 0,
                "retry_time": 0
            })

            doctor_list = ", ".join(googleid_dr_name_map.values())
            prompt = (
                f"Great! Let's schedule your appointment. Here is the list of doctors: {doctor_list}. "
                "Please say the name of the doctor you want to book with."
            )

            # ⬇️ CHANGED: use make_gather with hints
            gather = make_gather(prompt, hints=", ".join(googleid_dr_name_map.values()))
            resp.append(gather)
            return str(resp)

        # ✅ Voicemail intent
        elif "message" in lower or "voicemail" in lower:
            print("📩 Intent to leave a message detected → recording voicemail")
            session_data[call_sid]["stage"] = "voicemail"
            resp.say(gpt_speak("Please leave your name, phone number, and message after the beep."), VOICE)
            resp.record(
                max_length=MAX_RECORD_TIME,
                action="/voice",
                transcribe=True,
                transcribe_callback="/transcription"
            )
            return str(resp)

        # ❓ Fallback
        else:
            print(f"❓ Unclear intent: '{lower}' → re-prompting for intent choice")

            # Initialize or increment retry counter
            if "retry_intent" not in session_data[call_sid]:
                session_data[call_sid]["retry_intent"] = 1
            else:
                session_data[call_sid]["retry_intent"] += 1

            retry_count = session_data[call_sid]["retry_intent"]

            if retry_count >= 3:
                # Too many failed attempts — exit gracefully
                print("⚠️ Too many unclear responses — ending call")
                resp.say(gpt_speak("I'm sorry, I still didn't catch that. Please call us again when convenient. Goodbye."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)  # Clear session
                return str(resp)

            # Otherwise, re-prompt for intent
            session_data[call_sid]["stage"] = "intent"

            # ⬇️ CHANGED: use make_gather
            gather = make_gather(
                "Sorry, I didn’t catch that. Would you like to book an appointment, cancel one, reschedule, or leave a message?"
            )
            resp.append(gather)
            return str(resp)


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

        # 🚫 Block common junk phrases often returned by Twilio hallucination
        junk_inputs = {
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening", "yo", "test",
            "1", "yes", "no", "i know", "huh", "what", "okay", "ok", "bye", "goodbye", ""
        }

        if not spoken_clean or spoken_clean in junk_inputs or len(spoken_clean) < 3:
            print(f"⏩ Skipping junk doctor input: '{spoken_clean}' — re-prompting without retry")
            doctor_list_str = ", ".join(googleid_dr_name_map.values())
            # ⬇️ CHANGED: use make_gather (keeps same behavior and hints)
            gather = make_gather("Please say the name of the doctor you'd like to book with.", hints=doctor_list_str)
            resp.append(gather)
            return str(resp)

        matched_id = None

        # ------------------------------------------------------------------
        # 🔍 1. Partial token-based name match
        # ------------------------------------------------------------------
        partial_matches = []
        spoken_tokens = set(spoken_clean.split())

        for doc_id, friendly in googleid_dr_name_map.items():
            friendly_clean = friendly.lower().translate(str.maketrans('', '', string.punctuation)).strip()
            friendly_tokens = set(friendly_clean.split())

            if (
                spoken_clean in friendly_clean
                or friendly_clean in spoken_clean
                or spoken_tokens & friendly_tokens  # Token overlap
            ):
                partial_matches.append((doc_id, friendly))

        if len(partial_matches) == 1:
            matched_id = partial_matches[0][0]
            print(f"✅ Partial match with: {partial_matches[0][1]}")
        elif len(partial_matches) > 1:
            print(f"🔍 Multiple potential matches found: {[name for _, name in partial_matches]}")
            matched_id = partial_matches[0][0]  # Default to first match (or you can ask user to clarify)

        # ------------------------------------------------------------------
        # 🤖 2. GPT fallback (only if 2+ words)
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # ❌ 3. Still no match → Retry logic
        # ------------------------------------------------------------------
        if matched_id is None:
            print(f"❌ No doctor match for: '{spoken_clean}'")
            session_data[call_sid]["retry_booking"] += 1
            retries = session_data[call_sid]["retry_booking"]

            if retries >= 3:
                resp.say(gpt_speak(
                    "I'm sorry, I still couldn't match that name with any doctor in our clinic. "
                    "Please call us again when convenient. Goodbye."
                ), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            doctor_list_str = ", ".join(googleid_dr_name_map.values())
            # ⬇️ CHANGED: use make_gather (keeps same behavior and hints)
            retry_prompt = (
                f"I couldn't match that to a doctor. Available doctors are: {doctor_list_str}. "
                "Please say the doctor name again."
            )
            gather = make_gather(retry_prompt, hints=doctor_list_str)
            resp.append(gather)
            return str(resp)

        # ------------------------------------------------------------------
        # ✅ 4. Success → Go collect phone FIRST
        # ------------------------------------------------------------------
        session_data[call_sid]["doctor_id"] = matched_id
        session_data[call_sid]["stage"] = "collect_phone"  # ⬅️ CHANGED

        friendly_name = googleid_dr_name_map[matched_id]
        phone_prompt = (
            f"Great, we'll book with {friendly_name}. "
            "Please say or enter your phone number including area code."
        )  # ⬅️ CHANGED

        # ⬇️ Keep using make_gather
        gather = make_gather(phone_prompt)  # ⬅️ CHANGED (prompt only)
        resp.append(gather)
        return str(resp)


    elif stage == "collect_phone":
        # ----------------------------------------------------------------------
        # ☎️ Stage: Collect Phone Number (after DOB)
        # Purpose:
        #   - Prefer SPEECH input first (e.g., "four six nine ...").
        #   - If speech is missing/invalid, FALL BACK to KEYPAD (DTMF) digits.
        #   - Normalize to 10-digit US number (strip leading '1').
        #   - On failure, re-prompt using make_gather with longer wait and hints.
        # ----------------------------------------------------------------------
        import re

        # ✅ Ensure session buckets exist (prevents KeyError: 'customer')
        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("customer", {})

        # Retry counter
        session_data[call_sid]["retry_phone"] = session_data[call_sid].get("retry_phone", 0)

        def _validate_normalize_us_phone(raw_digits: str) -> str:
            d = re.sub(r"\D", "", raw_digits or "")
            if len(d) == 11 and d.startswith("1"):
                d = d[1:]
            return d if len(d) == 10 else ""

        # 🎙️ 1) Try SPEECH first
        speech_text = (speech_result or "").strip()
        debug_print(f"📱 collect_phone (speech raw): '{speech_text}'")
        speech_digits = re.sub(r"\D", "", speech_text)
        debug_print(f"📞 From speech → digits_only: '{speech_digits}', length={len(speech_digits)}")
        normalized = _validate_normalize_us_phone(speech_digits)

        # 🔢 2) If speech didn’t yield a valid 10-digit number, try DTMF fallback
        if not normalized:
            try:
                dtmf_digits = (request.values.get("Digits") or "").strip()
            except Exception:
                dtmf_digits = ""
            debug_print(f"🎛️ collect_phone (DTMF raw): '{dtmf_digits}'")
            dtmf_only = re.sub(r"\D", "", dtmf_digits)
            normalized = _validate_normalize_us_phone(dtmf_only)
            if normalized:
                debug_print(f"📟 Using DTMF fallback → normalized='{normalized}', length=10")

        # ❌ Still invalid → Re-prompt and stay in collect_phone
        if not normalized:
            # Soft retry: don't increment if caller started but hasn't finished (some digits heard)
            heard_some = len(speech_digits) > 0 or len(dtmf_digits if 'dtmf_digits' in locals() else "") > 0
            if not heard_some:
                session_data[call_sid]["retry_phone"] += 1

            debug_print(f"❌ Phone invalid/missing. heard_some={heard_some} retries={session_data[call_sid]['retry_phone']}")

            if session_data[call_sid]["retry_phone"] >= 5:
                debug_print("⛔ collect_phone: max retries reached.")
                resp.say(gpt_speak("Sorry, I couldn’t capture a valid phone number. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Re-prompt with longer wait + digit hints; tell them they can press '#'
            gather = make_gather(
                "Please provide your ten digit phone number including area code. "
                "You can say it clearly with short pauses, or type the digits and press pound.",
                hints="zero one two three four five six seven eight nine double triple"
            )
            resp.append(gather)
            # Post-gather fallback so the call never dead-ends on silence
            resp.say(gpt_speak("I didn't get the phone number."), VOICE)
            resp.redirect("/voice")
            return str(resp)

        # ✅ 3) Save validated 10-digit number and proceed
        debug_print(f"✅ Valid phone number accepted (10-digit): {normalized}")
        session_data[call_sid]["customer"]["phone"] = normalized

        # 📅 Go to DOB collection next
        session_data[call_sid]["stage"] = "collect_dob"

        # 🎂 Prompt for DOB (using make_gather with clearer keypad option)
        gather = make_gather(
            "Thank you. Please say your date of birth, for example, January fifteenth nineteen eighty five. "
            "Or enter two digits for month, two for day, and four for year, then press pound."
        )
        resp.append(gather)
        return str(resp)



    
    elif stage == "collect_dob":
        # ----------------------------------------------------------------------
        # 🎂 Stage: Collect Date of Birth (DOB)
        # ----------------------------------------------------------------------
        debug_print("collect_dob: 📍 Stage entered")

        # Pull DTMF if present, otherwise use speech
        try:
            dtmf_digits = (request.values.get("Digits") or "").strip()
        except Exception:
            dtmf_digits = ""

        speech_text = (speech_result or "").strip()
        debug_print(f"collect_dob: 🎙️ speech_text='{speech_text}', 🔢 dtmf_digits='{dtmf_digits}'")

        # Parse DOB input
        dt = parse_dob_input(speech_text, dtmf_digits)
        if not dt:
            session_data[call_sid]["retry_dob"] = session_data[call_sid].get("retry_dob", 0) + 1
            r = session_data[call_sid]["retry_dob"]
            debug_print(f"collect_dob: ❌ Parse failed. Retry={r}")

            if r >= 3:
                resp.say(gpt_speak("Sorry, I couldn’t understand your date of birth. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            prompt_text = (
                "Please say your date of birth, for example July third nineteen ninety, "
                "or type two digits for month, two digits for day, and four digits for year, then press pound. "
                "For example, 07031990#."
            )
            gather = make_gather_dob(prompt_text)
            resp.append(gather)
            return str(resp)

        # Validate DOB range
        try:
            from datetime import date
            today = date.today()
            min_date = date(1900, 1, 1)
            dob_date = dt.date()
            if not (min_date <= dob_date <= today):
                debug_print(f"collect_dob: ⚠️ DOB out of range → {dob_date.isoformat()}")
                session_data[call_sid]["retry_dob"] = session_data[call_sid].get("retry_dob", 0) + 1
                prompt_text = (
                    "That doesn't sound like a valid birth date. Please say it again, "
                    "or type two digits for month, two digits for day, and four digits for year, then press pound. "
                    "For example, 07031990#."
                )
                gather = make_gather_dob(prompt_text)
                resp.append(gather)
                return str(resp)
        except Exception as e:
            debug_print(f"collect_dob: ⚠️ Validation error → {e}")

        # Store ISO DOB
        iso_dob = dt.strftime("%Y-%m-%d")
        session_data[call_sid].setdefault("customer", {})
        session_data[call_sid]["customer"]["dob"] = iso_dob
        debug_print(f"collect_dob: ✅ Stored DOB → {iso_dob}")

        # ✅ Always move to ask_time_date next
        session_data[call_sid]["stage"] = "ask_time_date"
        debug_print("collect_dob: ➡️ Next stage → ask_time_date")

        # Prompt for appointment time/date
        gather = make_gather(
            "Thanks. What time and date would you like to book your appointment?"
        )
        resp.append(gather)
        return str(resp)


    elif stage == "ask_time_date":
        # ----------------------------------------------------------------------
        # 📍 Stage: ask_time_date
        # ----------------------------------------------------------------------
        debug_print(f"🗣️ Received spoken time: {speech_result}")
        time_info = smart_parse_time(speech_result)

        if not time_info or not isinstance(time_info, tuple) or len(time_info) != 2:
            session_data[call_sid]["retry_time"] = session_data[call_sid].get("retry_time", 0) + 1
            retry_count = session_data[call_sid]["retry_time"]
            debug_print(f"⚠️ Time parsing failed. Retry count: {retry_count}")

            if retry_count >= 3:
                debug_print("❌ Max retries reached. Ending call.")
                resp.say(gpt_speak("Sorry, I still couldn't understand the time. Please try again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            gather = make_gather("Please say the date and time again, for example, July 3rd at 9 AM.")
            resp.append(gather)
            # fallback to avoid dead-end
            resp.say(gpt_speak("I didn't get the date and time."), VOICE)
            resp.redirect("/voice")
            return str(resp)

        # ✅ Valid date and time extracted
        spoken_day, spoken_time = time_info
        debug_print(f"📆 Extracted → Day: {spoken_day}, Time: {spoken_time}")
        session_data[call_sid]["spoken_day"] = spoken_day
        session_data[call_sid]["spoken_time"] = spoken_time

        try:
            appointment_start, appointment_end = build_timeslot_range(spoken_day, spoken_time)
            session_data[call_sid]["appointment_time"] = {"start": appointment_start, "end": appointment_end}
            debug_print(f"📆 Appointment requested → Start: {appointment_start}, End: {appointment_end}")
        except Exception as e:
            debug_print(f"❌ Failed to build appointment time range from '{spoken_day}' and '{spoken_time}': {e}")
            session_data[call_sid]["retry_time"] = session_data[call_sid].get("retry_time", 0) + 1

            if session_data[call_sid]["retry_time"] >= 3:
                debug_print("❌ Max retries reached during build_timeslot_range.")
                resp.say(gpt_speak("Sorry, I couldn’t understand the time you mentioned. Please try again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            gather = make_gather("I didn't catch that clearly. Please repeat the date and time, like July 3rd at 9 AM.")
            resp.append(gather)
            resp.say(gpt_speak("I still didn't get the date and time."), VOICE)
            resp.redirect("/voice")
            return str(resp)

        # 🔎 Check availability
        doctor_id = session_data.get(call_sid, {}).get("doctor_id")
        if not doctor_id:
            debug_print("⚠️ No doctor_id in session. Returning to booking stage.")
            session_data[call_sid]["stage"] = "booking"
            gather = make_gather("Please say the name of the doctor you'd like to book with.")
            resp.append(gather)
            resp.say(gpt_speak("I didn't hear the doctor name."), VOICE)
            resp.redirect("/voice")
            return str(resp)

        calendar_id = doctor_id
        debug_print(f"👨‍⚕️ Checking calendar ID: {calendar_id}")

        if not is_time_slot_available(calendar_id, appointment_start, appointment_end, creds):
            debug_print("❌ Requested time slot is not available")

            alts = get_next_available_slots(calendar_id, creds, limit=3, duration_minutes=APPOINTMENT_DURATION_MINUTES)

            if alts:
                options = " or ".join([slot["friendly"] for slot in alts])
                prompt = f"That time is not available. Would you like to book on {options}?"
                debug_print(f"💡 Offering alternatives: {options}")
            else:
                prompt = "That time is not available, and I couldn't find any open slots soon. Please try again later."
                debug_print("⚠️ No alternative slots found.")

            gather = make_gather(prompt)
            resp.append(gather)
            resp.say(gpt_speak("I didn't get your choice."), VOICE)
            resp.redirect("/voice")
            return str(resp)

        # ✅ Slot is free → Check if customer exists
        debug_print("✅ Slot is available. Checking customer existence...")
        customer = session_data[call_sid].get("customer", {})
        # normalize phone the same way your DB/search expects
        phone_norm = "".join(ch for ch in (customer.get("phone", "") or "") if ch.isdigit())
        if len(phone_norm) == 11 and phone_norm.startswith("1"):
            phone_norm = phone_norm[1:]
        dob_val = customer.get("dob", "")

        try:
            if phone_norm and dob_val and customer_search(phone_norm, dob_val):
                debug_print("📋 Customer exists — skipping info collection, going to book_appt_confirm.")
                session_data[call_sid]["stage"] = "book_appt_confirm"
                # auto-run confirm now (no yes/no that could contradict)
                try:
                    from flask import url_for
                    resp.redirect(url_for("voice"))
                except Exception:
                    resp.redirect(request.path)
                return str(resp)
            else:
                debug_print("🆕 Customer not found — proceeding to first name collection.")
                session_data[call_sid]["stage"] = "collect_first_name"
                gather = make_gather("Thanks. What is your first name?")
                resp.append(gather)
                resp.say(gpt_speak("I didn't catch the first name."), VOICE)
                resp.redirect("/voice")
                return str(resp)
        except Exception as e:
            debug_print(f"⚠️ Error during customer_search: {e}")
            session_data[call_sid]["stage"] = "collect_first_name"
            gather = make_gather("Thanks. What is your first name?")
            resp.append(gather)
            resp.say(gpt_speak("I didn't catch the first name."), VOICE)
            resp.redirect("/voice")
            return str(resp)







    elif stage == "collect_first_name":
        # ----------------------------------------------------------------------
        # 🧍 Stage: Collect First Name
        #  - Accept speech, normalize gently (letters, spaces, hyphen, apostrophe)
        #  - Do NOT wipe other customer fields
        #  - Re-prompt on empty/unclear input
        # ----------------------------------------------------------------------
        import re

        # Buckets
        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("customer", {})

        # Raw speech
        first_name_raw = (speech_result or "").strip()
        debug_print(f"📛 collect_first_name: raw='{first_name_raw}'")

        # Heuristic: if they say "my name is ...", keep the tail
        text = first_name_raw
        lowered = first_name_raw.lower()
        for cue in ("my name is", "this is", "i am", "i'm"):
            if cue in lowered:
                # take substring after the cue
                idx = lowered.rfind(cue)
                text = first_name_raw[idx + len(cue):].strip()
                break

        # Normalize: keep letters (basic latin + accents), space, hyphen, apostrophe
        # Then collapse spaces
        text = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ' -]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        debug_print(f"📛 collect_first_name: normalized='{text}'")

        # Validate
        if not text or len(text) > 40 or len(text.split()) > 2:
            gather = make_gather("I didn't catch that clearly. Please say just your first name.")
            resp.append(gather)
            resp.say(gpt_speak("I didn't get the first name."), VOICE)
            resp.redirect("/voice")
            return str(resp)

        # Save without wiping other fields
        session_data[call_sid]["customer"]["first_name"] = text
        debug_print(f"collect_first_name: ✅ saved first_name='{text}'")

        # Next
        session_data[call_sid]["stage"] = "collect_last_name"
        gather = make_gather("Thank you. Now, what is your last name?")
        resp.append(gather)
        # safety line if gather times out
        resp.say(gpt_speak("I didn't get the last name."), VOICE)
        resp.redirect("/voice")
        return str(resp)





    elif stage == "collect_last_name":
        # ----------------------------------------------------------------------
        # 👤 Stage: Collect Last Name
        #  - Accept speech, normalize gently (letters, spaces, hyphen, apostrophe)
        #  - Do NOT wipe other customer fields
        #  - Re-prompt on empty/unclear input
        #  - After success: if a waiting stage is set (return_stage), go there;
        #    otherwise proceed to collect_address.
        # ----------------------------------------------------------------------
        import re

        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("customer", {})

        last_raw = (speech_result or "").strip()
        debug_print(f"👤 collect_last_name: raw='{last_raw}'")

        # Heuristic strip like first name (handle “last name is …” just in case)
        text = last_raw
        lowered = last_raw.lower()
        for cue in ("last name is", "my last name is"):
            if cue in lowered:
                idx = lowered.rfind(cue)
                text = last_raw[idx + len(cue):].strip()
                break

        # Normalize
        text = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ' -]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        debug_print(f"👤 collect_last_name: normalized='{text}'")

        if not text or len(text) > 60 or len(text.split()) > 3:
            gather = make_gather("Sorry, please say just your last name.")
            resp.append(gather)
            resp.say(gpt_speak("I didn't get the last name."), VOICE)
            resp.redirect("/voice")
            return str(resp)

        # Save without wiping other fields
        session_data[call_sid]["customer"]["last_name"] = text
        debug_print(f"collect_last_name: ✅ saved last_name='{text}'")

        # Optional handoff: if some stage is waiting for a name, jump back there
        return_stage = session_data[call_sid].pop("return_stage", None)
        if return_stage:
            session_data[call_sid]["stage"] = return_stage
            debug_print(f"collect_last_name: 🔁 Returning to waiting stage → {return_stage}")
            resp.redirect("/voice")
            return str(resp)

        # Default flow: proceed to address
        session_data[call_sid]["stage"] = "collect_address"
        gather = make_gather("Got it. What is your full address, please?")
        resp.append(gather)
        # safety line if gather times out
        resp.say(gpt_speak("I didn't hear the address."), VOICE)
        resp.redirect("/voice")
        return str(resp)




   


    elif stage == "collect_address":
        # ----------------------------------------------------------------------
        # 🏠 Stage: Collect Customer Address (INFO ONLY)
        # ----------------------------------------------------------------------
        import re

        address_raw = (speech_result or "").strip()
        debug_print(f"collect_address: 📬 Collected address (raw): {address_raw}")

        address = re.sub(r"\s+", " ", address_raw).strip()
        address = re.sub(r"[.,;\-–—\s]+$", "", address)
        debug_print(f"collect_address: 📬 Normalized address: {address}")

        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("customer", {})

        if not address or len(address) < 5:
            gather = make_gather("Sorry, I didn't catch your full address. Please say your street, city, state, and zip.")
            resp.append(gather)
            resp.say(gpt_speak("I didn't hear the address."), VOICE)
            resp.redirect("/voice")
            return str(resp)

        session_data[call_sid]["customer"]["address"] = address

        # Require phone + DOB before CC
        def _normalize_10(d):
            d = "".join(ch for ch in (d or "") if ch.isdigit())
            return d[1:] if len(d) == 11 and d.startswith("1") else d

        phone_norm = _normalize_10(session_data[call_sid]["customer"].get("phone", ""))
        if len(phone_norm) != 10:
            debug_print("collect_address: ❌ Phone missing/invalid → collect_phone")
            session_data[call_sid]["stage"] = "collect_phone"
            gather = make_gather(
                "Thanks. Before we continue, please provide your ten digit phone number including area code. "
                "You can say it or type the digits and press pound."
            )
            resp.append(gather)
            resp.say(gpt_speak("I didn't get the phone number."), VOICE)
            resp.redirect("/voice")
            return str(resp)

        if not session_data[call_sid]["customer"].get("dob"):
            debug_print("collect_address: ❌ DOB missing → collect_dob")
            session_data[call_sid]["stage"] = "collect_dob"
            gather = make_gather(
                "Thanks. Before we continue, please provide your date of birth. "
                "You can say it, or enter two digits for month, two for day, and four for year, then press pound."
            )
            resp.append(gather)
            resp.say(gpt_speak("I didn't get the date of birth."), VOICE)
            resp.redirect("/voice")
            return str(resp)

        session_data[call_sid]["stage"] = "collect_cc"
        debug_print("collect_address: 🔁 Redirecting to /voice to run collect_cc")
        resp.redirect("/voice")
        return str(resp)



    
    
    elif stage == "collect_cc":
        # ----------------------------------------------------------------------
        # 💳 Stage: collect_cc
        # Purpose:
        #   - Collect credit card info in three mini-steps:
        #       (1) Card number (13–19 digits, Luhn-checked)
        #       (2) Expiration (MMYY or MMYYYY) → stored as 'MM/YY' and must be current/future
        #       (3) CVV (3–4 digits)
        #   - Stores under session_data[call_sid]["customer"]:
        #       cc_number, cc_exp, cc_cvv, cc_name
        #   - Auto-advances to book_appt_confirm upon success (no extra prompt)
        # Notes:
        #   - Uses make_gather() (speech + DTMF, finishOnKey="#").
        #   - Speech digits supported via spoken→digit normalization; DTMF preferred.
        #   - Guards: requires phone (10-digit) and DOB before collecting CC.
        #   - Logs: PAN/CVV masked to avoid leaking sensitive data.
        #   - New: voice-friendly partial capture (e.g., 15-of-16 digits) and
        #          escalation to “please type it” after repeated speech failures.
        # ----------------------------------------------------------------------
        import re
        from datetime import datetime as _dt

        # --- Luhn mod-10 -------------------------------------------------------
        def luhn_check(number: str) -> bool:
            s = 0
            alt = False
            for ch in number[::-1]:
                if not ch.isdigit():
                    return False
                d = ord(ch) - 48
                if alt:
                    d *= 2
                    if d > 9:
                        d -= 9
                s += d
                alt = not alt
            return (s % 10) == 0

        # --- Voice → digits (supports "double"/"triple", common homophones) ----
        def normalize_spoken_digits(raw: str) -> str:
            if not raw:
                return ""
            words = raw.lower().strip().split()
            m = {
                "zero":"0","oh":"0","o":"0",
                "one":"1","two":"2","to":"2","too":"2",
                "three":"3","four":"4","for":"4",
                "five":"5","six":"6","seven":"7",
                "eight":"8","ate":"8","nine":"9"
            }
            out, i = [], 0
            while i < len(words):
                w = words[i].strip(".,;:-")
                if w in ("double","triple") and i+1 < len(words):
                    nxt = words[i+1].strip(".,;:-")
                    if nxt in m:
                        out.extend([m[nxt]] * (2 if w == "double" else 3))
                        i += 2
                        continue
                if w in m:
                    out.append(m[w])
                else:
                    out.extend([c for c in w if c.isdigit()])
                i += 1
            return "".join(out)

        # --- Maskers for logging ------------------------------------------------
        def _mask_pan(n: str) -> str:
            n = n or ""
            return ("*" * max(0, len(n) - 4)) + n[-4:] if n else ""
        def _mask_all(n: str) -> str:
            return "*" * len(n or "")

        # --- Ensure session buckets exist --------------------------------------
        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("customer", {})
        customer = session_data[call_sid]["customer"]

        # 🔒 Require phone + DOB before CC
        def _normalize_10(d):
            d = "".join(ch for ch in (d or "") if ch.isdigit())
            return d[1:] if len(d) == 11 and d.startswith("1") else d
        phone_guard = _normalize_10(customer.get("phone"))
        if len(phone_guard) != 10 or not customer.get("dob"):
            debug_print("collect_cc: ❌ Missing phone/DOB → redirecting")
            session_data[call_sid]["stage"] = "collect_phone" if len(phone_guard) != 10 else "collect_dob"
            gather = make_gather(
                "Before payment details, please provide your ten digit phone number including area code."
                if len(phone_guard) != 10 else
                "Before payment details, please provide your date of birth. You can say it, or enter MMDDYYYY then press pound.",
                hints="zero one two three four five six seven eight nine"
            )
            resp.append(gather)
            resp.say(gpt_speak("Let's get that first."), VOICE)
            resp.redirect("/voice")
            return str(resp)

        # Mini-step tracker: 1=number, 2=exp, 3=cvv
        cc_step = session_data[call_sid].get("cc_step", 1)

        # Retries + speech assistance + partial buffer for PAN
        session_data[call_sid]["retry_cc"] = session_data[call_sid].get("retry_cc", 0)
        session_data[call_sid]["cc_speech_tries"] = session_data[call_sid].get("cc_speech_tries", 0)
        cc_partial = session_data[call_sid].get("cc_partial", "")

        # Pull inputs
        try:
            dtmf_digits = (request.values.get("Digits") or "").strip()
        except Exception:
            dtmf_digits = ""
        speech_text = (speech_result or "").strip()

        # Mask logs for step 1 (PAN)
        masked_speech = "***" if cc_step == 1 and speech_text else speech_text
        masked_dtmf = "***" if cc_step == 1 and dtmf_digits else dtmf_digits
        debug_print(f"collect_cc: 📍 step={cc_step}, DTMF='{masked_dtmf}', speech='{masked_speech}'")

        # Prefer DTMF; if none, extract digits from speech (words → digits)
        def get_digits() -> str:
            if dtmf_digits:
                return re.sub(r"\D", "", dtmf_digits)
            return re.sub(r"\D", "", normalize_spoken_digits(speech_text))

        # Helper: re-prompt with retry cap
        def _reprompt(prompt: str, hints: str):
            session_data[call_sid]["retry_cc"] += 1
            if session_data[call_sid]["retry_cc"] >= 5:
                debug_print("collect_cc: ⛔ max CC retries. Ending.")
                resp.say(gpt_speak("Sorry, we’re having trouble collecting your details. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return True  # handled
            gather = make_gather(prompt, hints=hints)
            resp.append(gather)
            resp.say(gpt_speak("I didn't get that."), VOICE)
            resp.redirect("/voice")
            return True

        # -------------------------------
        # Step 1: Card Number (13–19)
        # -------------------------------
        if cc_step == 1:
            new_digits = get_digits()
            # Support partial carry-over for speech (e.g., caller pauses)
            digits = (cc_partial + new_digits) if (cc_partial or new_digits) else ""
            # Avoid runaway length (keep max 19)
            if digits and len(digits) > 19:
                digits = digits[:19]

            if not digits:
                if _reprompt(
                    "Please enter your card number now, then press pound.",
                    hints="zero one two three four five six seven eight nine double triple"
                ): return str(resp)

            # If we heard 15 digits via speech, ask for the last single digit
            if len(digits) == 15 and not dtmf_digits:
                session_data[call_sid]["cc_partial"] = digits
                debug_print(f"collect_cc: 🧩 Heard 15 digits {_mask_pan(digits)}; asking for the last digit")
                if _reprompt(
                    "I heard fifteen digits. Please say or type the last single digit now, then press pound.",
                    hints="zero one two three four five six seven eight nine"
                ): return str(resp)

            # Luhn failure → if speech, escalate to “please type it” after 2 tries
            if not (13 <= len(digits) <= 19) or not luhn_check(digits):
                session_data[call_sid]["cc_speech_tries"] += (0 if dtmf_digits else 1)
                escalate = session_data[call_sid]["cc_speech_tries"] >= 2 and not dtmf_digits
                debug_print(f"collect_cc: ❌ Invalid card number: '{_mask_pan(digits)}' (len={len(digits)}), escalate={escalate}")
                if escalate:
                    if _reprompt(
                        "That number didn’t sound clear. Please TYPE the full card number now, then press pound.",
                        hints="zero one two three four five six seven eight nine"
                    ): return str(resp)
                else:
                    if _reprompt(
                        "That card number doesn't look right. Please re-enter the full card number, then press pound.",
                        hints="zero one two three four five six seven eight nine double triple"
                    ): return str(resp)

            # Save and advance
            customer["cc_number"] = digits
            session_data[call_sid]["cc_step"] = 2
            # clear helpers for next steps
            session_data[call_sid]["cc_partial"] = ""
            session_data[call_sid]["cc_speech_tries"] = 0
            debug_print(f"collect_cc: ✅ Saved card number {_mask_pan(digits)}")

            gather = make_gather(
                "Thank you. Now enter the expiration as two digits for month and two digits for year. "
                "For example, 0527. Then press pound.",
                hints="zero one two three four five six seven eight nine"
            )
            resp.append(gather)
            resp.say(gpt_speak("I didn't hear the expiration."), VOICE)
            resp.redirect("/voice")
            return str(resp)

        # -------------------------------
        # Step 2: Expiration (MMYY or MMYYYY, must be current/future)
        # -------------------------------
        if cc_step == 2:
            digits = get_digits()
            if len(digits) not in (4, 6):
                debug_print(f"collect_cc: ❌ Exp bad length: '{digits}'")
                if _reprompt(
                    "Please enter the expiration as four digits MMYY, then press pound.",
                    hints="zero one two three four five six seven eight nine"
                ): return str(resp)

            mm = int(digits[:2]) if digits[:2].isdigit() else 0
            yy = digits[2:]
            if not (1 <= mm <= 12):
                debug_print(f"collect_cc: ❌ Invalid month: '{digits}'")
                if _reprompt(
                    "The month must be 01 through 12. Please re-enter expiration MMYY, then press pound.",
                    hints="zero one two three four five six seven eight nine"
                ): return str(resp)

            # Normalize year to 2-digit (handle MMYYYY too)
            if len(yy) == 4:
                yy = yy[-2:]

            # Reject past month
            now = _dt.now()
            exp_year = 2000 + int(yy)
            exp_cmp  = exp_year * 100 + mm
            now_cmp  = now.year * 100 + now.month
            if exp_cmp < now_cmp:
                debug_print(f"collect_cc: ❌ Expired card: {mm:02d}/{yy}")
                if _reprompt(
                    "That card appears expired. Please enter a valid expiration date as MMYY, then press pound.",
                    hints="zero one two three four five six seven eight nine"
                ): return str(resp)

            customer["cc_exp"] = f"{mm:02d}/{yy}"
            session_data[call_sid]["cc_step"] = 3
            debug_print(f"collect_cc: ✅ Saved expiration {customer['cc_exp']}")

            gather = make_gather(
                "Great. Finally, enter the three or four digit security code, then press pound.",
                hints="zero one two three four five six seven eight nine"
            )
            resp.append(gather)
            resp.say(gpt_speak("I didn't hear the security code."), VOICE)
            resp.redirect("/voice")
            return str(resp)

        # -------------------------------
        # Step 3: CVV (3–4 digits)
        # -------------------------------
        if cc_step == 3:
            digits = get_digits()
            if not (3 <= len(digits) <= 4) or not digits.isdigit():
                debug_print(f"collect_cc: ❌ Invalid CVV length={len(digits)}")
                if _reprompt(
                    "That security code doesn't look right. Please re-enter the three or four digit code, then press pound.",
                    hints="zero one two three four five six seven eight nine"
                ): return str(resp)

            customer["cc_cvv"] = digits

            # Default cardholder name from collected name, if not set
            if not customer.get("cc_name"):
                name = customer.get("name") or " ".join(
                    p for p in [customer.get("first_name"), customer.get("last_name")] if p
                )
                customer["cc_name"] = name.strip() if name else None

            debug_print(f"collect_cc: ✅ Saved CVV (len={len(digits)}); cc_name='{customer.get('cc_name')}'")

            # Clear step tracker and jump to confirmation
            session_data[call_sid].pop("cc_step", None)
            session_data[call_sid]["stage"] = "book_appt_confirm"
            debug_print("collect_cc: ➡️ Auto-advancing to book_appt_confirm")

            # Immediately re-enter main handler so book_appt_confirm runs now
            try:
                from flask import url_for
                resp.redirect(url_for("voice"))  # adjust endpoint name if different
            except Exception:
                resp.redirect(request.path)      # fallback to current path
            return str(resp)



    

     # -----------------------------------------------------------------
    # ✅ Stage: book_appt_confirm
    # Purpose:
    #   - Finalize the appointment using data gathered in the session.
    #   - Guards:
    #       • must have a doctor_id (calendar id resolved to a friendly name)
    #       • must have a computed appointment_time start (UTC ISO)
    #       • must have customer with a valid 10-digit phone + DOB
    #   - Effects:
    #       • writes/updates the human-readable customer DB (insert_customer)
    #       • creates/saves the calendar event (confirm_appointment_by_name)
    #       • speaks a confirmation + sends an SMS (if save succeeded)
    #   - Privacy:
    #       • credit card values are always masked in logs
    #   - Flow resilience:
    #       • if phone/DOB missing/invalid, we set `return_stage="book_appt_confirm"`
    #         then jump to the proper collector; when that stage completes, we jump back here.
# -----------------------------------------------------------------
    elif stage == "book_appt_confirm":
        debug_print("book_appt_confirm: 📍 Stage entered")

        # -------------------------------
        # 1) Doctor resolution
        # -------------------------------
        doctor_id = session_data[call_sid].get("doctor_id")
        debug_print(f"book_appt_confirm: 🧩 Raw doctor_id from session → {doctor_id}")

        if not doctor_id:
            # If somehow doctor_id was lost, send user back to doctor selection flow
            debug_print("book_appt_confirm: ❌ doctor_id missing → redirecting to booking (doctor selection)")
            session_data[call_sid]["stage"] = "booking"
            gather = make_gather("I lost the doctor selection. Please say the name of the doctor you'd like to book with.")
            resp.append(gather)
            return str(resp)

        doctor_name = googleid_dr_name_map.get(doctor_id, "the doctor")
        debug_print(f"book_appt_confirm: 👨‍⚕️ Resolved doctor_name → {doctor_name}")

        # -------------------------------
        # 2) Appointment time (UTC ISO)
        # -------------------------------
        appointment_time = session_data[call_sid].get("appointment_time", {}).get("start")
        debug_print(f"book_appt_confirm: 🕓 Raw appointment_time UTC → {appointment_time}")
        if not appointment_time:
            debug_print("book_appt_confirm: ❌ No appointment time found in session data")
            resp.say(gpt_speak("Appointment time is missing. Let's try again later. Goodbye!"), VOICE)
            resp.hangup()
            return str(resp)

        # For voice/SMS we show a local-friendly string; keep UTC for writing
        formatted_time = ""
        try:
            from datetime import datetime
            import pytz
            # Normalize Z suffix for fromisoformat
            dt_utc = datetime.fromisoformat(appointment_time.replace("Z", "+00:00"))
            tz = pytz.timezone("America/Chicago")
            dt_local = dt_utc.astimezone(tz)
            # Linux supports %-d / %-I; if not, fall back to portable format
            try:
                formatted_time = dt_local.strftime("%A, %B %-d at %-I:%M %p")
            except Exception:
                formatted_time = dt_local.strftime("%A, %B %d at %I:%M %p").lstrip("0").replace(" 0", " ")
            debug_print(f"book_appt_confirm: 📆 Formatted appointment time (local) → {formatted_time}")
        except Exception as e:
            debug_print(f"book_appt_confirm: ⚠️ Failed to parse/format appointment time → {e}")
            resp.say(gpt_speak("Sorry, we couldn't confirm the appointment time."), VOICE)
            resp.hangup()
            return str(resp)

        # -------------------------------
        # 3) Customer info + masking utils
        # -------------------------------
        customer = session_data[call_sid].get("customer", {}) or {}
        debug_print(f"book_appt_confirm: 🧾 Raw customer object → {customer}")

        customer_name    = customer.get("name", "")  # optional combined name
        customer_phone   = customer.get("phone", "")
        customer_dob     = customer.get("dob", "")
        customer_address = customer.get("address", "")

        cc_name   = customer.get("cc_name", "")
        cc_number = customer.get("cc_number", "")
        cc_exp    = customer.get("cc_exp", "")
        cc_cvv    = customer.get("cc_cvv", "")

        def _mask_tail(s: str, keep_last: int = 4) -> str:
            s = s or ""
            return ("*" * max(0, len(s) - keep_last)) + s[-keep_last:] if s else ""

        def _mask_all(s: str) -> str:
            return "*" * len((s or ""))

        debug_print(f"book_appt_confirm: 👤 Name → {customer_name}")
        debug_print(f"book_appt_confirm: 📞 Phone → {customer_phone}")
        debug_print(f"book_appt_confirm: 🎂 DOB → {customer_dob}")
        debug_print(f"book_appt_confirm: 🏠 Address → {customer_address}")
        debug_print(f"book_appt_confirm: 💳 CC Name → {cc_name}")
        debug_print(f"book_appt_confirm: 💳 CC Number → { _mask_tail(cc_number) }")
        debug_print(f"book_appt_confirm: 💳 CC Exp → {cc_exp}")
        debug_print(f"book_appt_confirm: 💳 CC CVV → { _mask_all(cc_cvv) }")

        # -------------------------------
        # 4) Guard: require valid 10-digit phone + DOB
        #    If missing, bounce to collector and auto-return here
        # -------------------------------
        def _normalize_10(d: str) -> str:
            d = "".join(ch for ch in (d or "") if ch.isdigit())
            return d[1:] if len(d) == 11 and d.startswith("1") else d

        digits_phone = _normalize_10(customer_phone)

        if len(digits_phone) != 10:
            debug_print("book_appt_confirm: ❌ Missing/invalid phone → redirecting to collect_phone")
            session_data[call_sid]["return_stage"] = "book_appt_confirm"  # come back here after phone is captured
            session_data[call_sid]["stage"] = "collect_phone"
            gather = make_gather(
                "Before we confirm your appointment, I need your ten digit phone number including area code. "
                "You can say it or type the digits, then press pound.",
                hints="zero one two three four five six seven eight nine"
            )
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        if not customer_dob:
            debug_print("book_appt_confirm: ❌ Missing DOB → redirecting to collect_dob")
            session_data[call_sid]["return_stage"] = "book_appt_confirm"  # come back here after DOB is captured
            session_data[call_sid]["stage"] = "collect_dob"
            gather = make_gather(
                "Before we confirm, please provide your date of birth. "
                "You can say it, or enter two digits for month, two for day, and four for year, then press pound."
            )
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # -------------------------------
        # 5) Upsert customer record in the single-file DB
        #    • insert_customer() now writes/updates human-readable blocks (no duplicates)
        #    • phone is normalized; CC data may be empty (will be masked when written)
        # -------------------------------
        try:
            first_name = (customer.get("first_name") or "").strip()
            last_name  = (customer.get("last_name") or "").strip()

            # Derive names from combined 'name' if needed
            if not first_name and customer_name:
                parts = customer_name.strip().split()
                first_name = parts[0]
                last_name  = " ".join(parts[1:]) if len(parts) > 1 else ""

            init_db()  # ensure file exists

            inserted = insert_customer(
                phone=digits_phone,          # normalized 10-digit
                dob=customer_dob,
                first_name=first_name,
                last_name=last_name,
                address=customer_address,
                cc_name=cc_name,
                cc_number=cc_number,
                cc_exp=cc_exp,
                cc_cvv=cc_cvv
            )
            debug_print(f"book_appt_confirm: 🗃️ insert_customer result → {'inserted' if inserted else 'updated existing'}")
        except Exception as e:
            debug_print(f"book_appt_confirm: ⚠️ insert_customer failed → {e}")

        # -------------------------------
        # 6) Save the appointment in Calendar
        #    If this fails, keep the caller in flow (ask for a new time).
        # -------------------------------
        appointment_saved = False
        try:
            effective_name = (customer_name or " ".join(n for n in [first_name, last_name] if n)).strip()
            confirm_appointment_by_name(
                doctor_name=doctor_name,
                phone=digits_phone,      # normalized digits
                dob=customer_dob,
                name=effective_name,
                address=customer_address,
                utc_start=appointment_time,
                calendar_id=doctor_id
            )
            appointment_saved = True
            debug_print(f"book_appt_confirm: ✅ Appointment saved for {doctor_name} (Calendar ID: {doctor_id})")
        except Exception as e:
            debug_print(f"book_appt_confirm: ⚠️ Failed to save appointment → {e}")

        # -------------------------------
        # 7) Output to caller + SMS (only if we saved successfully)
        # -------------------------------
        if appointment_saved:
            confirmation_message = (
                f"Your appointment with {doctor_name} has been successfully booked."
                f"{' on ' + formatted_time if formatted_time else ''} "
                "We look forward to seeing you. Goodbye!"
            )
            debug_print(f"book_appt_confirm: 🗣️ Speaking confirmation message → {confirmation_message}")
            resp.say(gpt_speak(confirmation_message), VOICE)

            # SMS is sent only if we actually saved the event
            try:
                e164_phone = f"+1{digits_phone}"
                sms_text = f"Hi {(effective_name or 'there')}, your appointment with {doctor_name} is confirmed"
                if formatted_time:
                    sms_text += f" on {formatted_time}"
                sms_text += ". Thank you for choosing Epic Therapist Clinic."
                message = client.messages.create(
                    body=sms_text,
                    from_=TWILIO_PHONE_NUMBER,
                    to=e164_phone
                )
                debug_print(f"book_appt_confirm: 📤 SMS sent to {e164_phone}, SID: {message.sid}")
            except Exception as e:
                debug_print(f"book_appt_confirm: ❌ SMS send failed → {e}")

            # End call and clean session only after success path
            resp.hangup()
            session_data.pop(call_sid, None)
            debug_print("book_appt_confirm: 🧼 Session data cleared")
            return str(resp)

        # If we got here, appointment save failed; keep caller in flow
        debug_print("book_appt_confirm: ❌ Save failed — reprompting for a new time")
        session_data[call_sid]["stage"] = "ask_time_date"
        gather = make_gather("Sorry, I couldn't confirm that slot. Please say a new date and time, for example, August 14th at 10 AM.")
        resp.append(gather)
        return str(resp)







    
    elif stage == "cancel_appointment":
        # ----------------------------------------------------------------------
        # 🔄 Stage: Cancel Appointment — after the caller says the doctor’s name
        # This stage:
        #  1️⃣ Tries to match the spoken name to a doctor in our list.
        #  2️⃣ If no match is found, it retries with GPT extraction.
        #  3️⃣ If still no match, it re-prompts (with retry limits).
        #  4️⃣ Once matched, moves to the next stage to get the phone number.
        # ----------------------------------------------------------------------

        import string
        selected_text = (speech_result or "").strip()

        # 🆕 Check if nothing was heard → immediate re-prompt
        if not selected_text:
            print("⚠️ cancel_appointment: No speech detected — re-prompting user to say the doctor's name.")
            doctor_list = ", ".join(googleid_dr_name_map.values())
            retry_prompt = (
                f"I can't hear you. Available doctors are: {doctor_list}. "
                "Please say the name of the doctor whose appointment you want to cancel."
            )
            resp.append(make_gather(
                prompt_text=retry_prompt,
                next_stage="cancel_appointment"
            ))
            return str(resp)

        selected_clean = selected_text.lower().translate(str.maketrans('', '', string.punctuation)).strip()
        print(f"🗣️ Received doctor name: {selected_clean}")
        matched_id = None

        # 🔍 Step 1: Try partial substring match
        partial_matches = []
        for doc_id, friendly_name in googleid_dr_name_map.items():
            friendly_clean = friendly_name.lower().translate(str.maketrans('', '', string.punctuation)).strip()
            if selected_clean in friendly_clean or friendly_clean in selected_clean:
                partial_matches.append((doc_id, friendly_name))

        if len(partial_matches) == 1:
            matched_id = partial_matches[0][0]
            print(f"✅ Partial match found: {partial_matches[0][1]}")

        # 🤖 Step 2: GPT fallback
        if not matched_id:
            try:
                extracted_name = extract_doctor_name(speech_result)
                print(f"🤖 GPT extracted name: {extracted_name}")
                if extracted_name:
                    extracted_clean = extracted_name.lower().translate(str.maketrans('', '', string.punctuation)).strip()
                    for doc_id, friendly_name in googleid_dr_name_map.items():
                        friendly_clean = friendly_name.lower().translate(str.maketrans('', '', string.punctuation)).strip()
                        if extracted_clean in friendly_clean or friendly_clean in extracted_clean:
                            matched_id = doc_id
                            print(f"✅ GPT matched: {friendly_name}")
                            break
            except Exception as e:
                print(f"⚠️ GPT fallback in extract_doctor_name: {e}")

        # ❌ Step 3: Still no match → retry
        if not matched_id:
            retries = session_data[call_sid].get("retry_booking", 0)
            session_data[call_sid]["retry_booking"] = retries + 1

            if retries >= MAX_NUMBER_DR_RETRY:
                resp.say(gpt_speak(
                    "I'm sorry, I still couldn't match that name with any doctor in our clinic. Please try again later. Goodbye."
                ), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            doctor_list = ", ".join(googleid_dr_name_map.values())
            retry_prompt = (
                f"I didn't recognize that name. Available doctors are: {doctor_list}. "
                "Please say the name again."
            )
            resp.append(make_gather(
                prompt_text=retry_prompt,
                next_stage="cancel_appointment"
            ))
            return str(resp)

        # ✅ Step 4: Proceed with matched doctor
        session_data[call_sid]["cancel"]["doctor"] = googleid_dr_name_map[matched_id]
        session_data[call_sid]["doctor_id"] = matched_id
        session_data[call_sid]["stage"] = "cancel_appt_by_phone_number"

        resp.append(make_gather(
            prompt_text="Thanks. What phone number did you use when booking the appointment?",
            next_stage="cancel_appt_by_phone_number"
        ))
        return str(resp)



    elif stage == "cancel_appt_by_phone_number":
        # ----------------------------------------------------------------------
        # 📌 Purpose:
        # This stage collects the **phone number** associated with the
        # appointment the caller wants to cancel.
        # 
        # Flow:
        # 1. Extract the phone number from the caller's speech.
        # 2. Validate the phone number (minimum length check).
        # 3. If invalid → prompt the user again using make_gather().
        # 4. If valid → store the number in session data and move on to
        #    ask for the date/time of the appointment to cancel.
        # 
        # Notes:
        # - This ensures the cancellation request is linked to the correct
        #   appointment record in the system.
        # ----------------------------------------------------------------------

        # 📞 Step 1: Extract phone number
        phone = extract_phone_number(speech_result)
        print(f"📱 Extracted phone → {phone}")

        # ❌ Step 2: If invalid phone, re-prompt
        if not phone or len(phone) < 7:
            gather = make_gather("I didn’t catch your phone number. Please say it again clearly.")
            resp.append(gather)
            return str(resp)

        # ✅ Step 3: Store phone and move to next stage
        session_data[call_sid]["cancel"]["phone"] = phone
        session_data[call_sid]["stage"] = "cancel_appt_get_date_time"

        # 🗣️ Step 4: Ask for date/time of appointment
        gather = make_gather(
            "Thanks. Now, please tell me the date and time of the appointment you want to cancel. "
            "For example, say July 3rd at 9 AM."
        )
        resp.append(gather)
        return str(resp)




    elif stage == "cancel_appt_get_dob":
        # ----------------------------------------------------------------------
        # ❌ Stage: cancel_appt_get_dob
        # Purpose:
        #   - Collect and validate the caller's Date of Birth for appointment
        #     cancellation lookup/verification.
        #   - Accepts speech (e.g., "July third nineteen fifty six") OR DTMF
        #     as MMDDYYYY (e.g., 07031956#).
        #   - Stores DOB as ISO (YYYY-MM-DD) under session_data[call_sid]["customer"]["dob"].
        #   - Requires a 10-digit phone on file before proceeding; if missing,
        #     redirects to collect_phone and returns here afterward.
        # Flow after success:
        #   - Sets stage -> "cancel_appt_confirm" (you should implement that to
        #     locate & cancel the appointment using phone+dob [+ optional doctor_id]).
        # Resilience:
        #   - 3 retries on parse errors with polite reprompts.
        #   - Consistent "can't hear you" behavior via make_gather_dob().
        # ----------------------------------------------------------------------
        debug_print("cancel_appt_get_dob: 📍 Stage entered")

        # Ensure buckets exist
        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("customer", {})

        # --- Guard: require 10-digit phone first -------------------------------
        def _normalize_10(d: str) -> str:
            d = "".join(ch for ch in (d or "") if ch.isdigit())
            return d[1:] if len(d) == 11 and d.startswith("1") else d

        phone_norm = _normalize_10(session_data[call_sid]["customer"].get("phone"))
        if len(phone_norm) != 10:
            debug_print("cancel_appt_get_dob: ❌ phone missing/invalid → redirecting to collect_phone")
            # Set return path so collect_phone bounces back here once done
            session_data[call_sid]["return_stage"] = "cancel_appt_get_dob"
            session_data[call_sid]["stage"] = "collect_phone"
            gather = make_gather(
                "To cancel your appointment, please provide your ten digit phone number including area code. "
                "You can say it, or type the digits and press pound.",
                hints="zero one two three four five six seven eight nine"
            )
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # --- Pull inputs (DTMF preferred if provided by Twilio) ----------------
        try:
            dtmf_digits = (request.values.get("Digits") or "").strip()
        except Exception:
            dtmf_digits = ""
        speech_text = (speech_result or "").strip()
        debug_print(f"cancel_appt_get_dob: 🎙️ speech_text='{speech_text}', 🔢 dtmf_digits='{dtmf_digits}'")

        # --- Attempt to parse DOB (speech or keypad) ---------------------------
        dt = parse_dob_input(speech_text, dtmf_digits)
        if not dt:
            session_data[call_sid]["retry_cancel_dob"] = session_data[call_sid].get("retry_cancel_dob", 0) + 1
            r = session_data[call_sid]["retry_cancel_dob"]
            debug_print(f"cancel_appt_get_dob: ❌ Parse failed. Retry={r}")

            if r >= 3:
                resp.say(gpt_speak("Sorry, I couldn’t understand your date of birth. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            prompt_text = (
                "Please say your date of birth to locate your appointment, for example July third nineteen fifty six, "
                "or type two digits for month, two for day, and four for year, then press pound. "
                "For example, 07031956#."
            )
            # Use your DOB gather helper if available; otherwise fall back to make_gather
            try:
                gather = make_gather_dob(prompt_text)
            except Exception:
                gather = make_gather(prompt_text)
            resp.append(gather)
            return str(resp)

        # --- Validate reasonable DOB range -------------------------------------
        try:
            from datetime import date
            today = date.today()
            min_date = date(1900, 1, 1)
            dob_date = dt.date()
            if not (min_date <= dob_date <= today):
                debug_print(f"cancel_appt_get_dob: ⚠️ DOB out of range → {dob_date.isoformat()}")
                session_data[call_sid]["retry_cancel_dob"] = session_data[call_sid].get("retry_cancel_dob", 0) + 1
                prompt_text = (
                    "That doesn't sound like a valid birth date. Please say it again, "
                    "or type two digits for month, two for day, and four for year, then press pound. "
                    "For example, 07031956#."
                )
                try:
                    gather = make_gather_dob(prompt_text)
                except Exception:
                    gather = make_gather(prompt_text)
                resp.append(gather)
                return str(resp)
        except Exception as e:
            debug_print(f"cancel_appt_get_dob: ⚠️ Validation error → {e}")

        # --- Store ISO DOB and advance -----------------------------------------
        iso_dob = dt.strftime("%Y-%m-%d")
        session_data[call_sid]["customer"]["dob"] = iso_dob
        debug_print(f"cancel_appt_get_dob: ✅ Stored DOB → {iso_dob}")

        # Advance to the confirmation/lookup stage of your cancel flow
        session_data[call_sid]["stage"] = "cancel_appt_get_date_time"
        debug_print("cancel_appt_get_dob: ➡️ Next stage → cancel_appt_confirm")

        # If you already know the appointment time, you can mention it; otherwise generic prompt
        next_prompt = (
            "Thanks. I found your details. Do you want me to cancel your appointment now?"
        )
        gather = make_gather(next_prompt)
        resp.append(gather)
        return str(resp)





    elif stage == "cancel_appt_get_date_time":
        # ----------------------------------------------------------------------
        # ❌ Stage: cancel_appt_get_date_time
        # Goal:
        #   - If caller provides a date/time, parse it and validate against calendar(s).
        #       • If a matching appointment is found → go to cancel_appt_confirm
        #       • If not found / invalid → go to cancel_appt_iterate
        #   - If caller does NOT provide date/time → go to cancel_appt_iterate
        #
        # IMPORTANT: No DOB verification here. Any DOB checks happen in
        #            cancel_appt_get_dob. We only rely on phone (if available).
        # ----------------------------------------------------------------------
        debug_print("cancel_appt_get_date_time: 📍 Stage entered")

        # Ensure buckets
        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("cancel", {})
        cancel_ctx = session_data[call_sid]["cancel"]

        # --- Require caller phone (10-digit) for event search; DO NOT check DOB here ---
        def _normalize_10(d: str) -> str:
            d = "".join(ch for ch in (d or "") if ch.isdigit())
            return d[1:] if len(d) == 11 and d.startswith("1") else d

        phone_norm = _normalize_10(
            cancel_ctx.get("phone") or session_data[call_sid].get("customer", {}).get("phone")
        )
        if len(phone_norm) != 10:
            debug_print("cancel_appt_get_date_time: ❌ phone missing/invalid → redirecting to collect_phone")
            session_data[call_sid]["return_stage"] = "cancel_appt_get_date_time"
            session_data[call_sid]["stage"] = "collect_phone"
            gather = make_gather(
                "To locate your appointment, please provide your ten digit phone number including area code. "
                "You can say it, or type the digits and press pound.",
                hints="zero one two three four five six seven eight nine"
            )
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # Persist normalized phone for downstream use
        cancel_ctx["phone"] = phone_norm

        # --- Pull the user's utterance for date/time ----------------------------
        utter = (speech_result or "").strip()
        debug_print(f"cancel_appt_get_date_time: 🗣️ Raw speech input → '{utter}'")

        # If no input given → immediately fall back to iterator flow
        if not utter:
            debug_print("cancel_appt_get_date_time: 🚫 No date/time provided → falling back to cancel_appt_iterate")
            cancel_ctx["iter_index"] = 0
            session_data[call_sid]["stage"] = "cancel_appt_iterate"
            resp.redirect("/voice")
            return str(resp)

        # --- Try to parse (no retry loop here) ----------------------------------
        time_info = smart_parse_time(utter)
        debug_print(f"cancel_appt_get_date_time: 🧠 smart_parse_time() returned → {time_info}")

        if not time_info or not isinstance(time_info, tuple) or len(time_info) != 2:
            debug_print("cancel_appt_get_date_time: ❌ Could not parse date/time → going to iterator")
            cancel_ctx["iter_index"] = 0
            session_data[call_sid]["stage"] = "cancel_appt_iterate"
            resp.redirect("/voice")
            return str(resp)

        spoken_day, spoken_time = time_info
        debug_print(f"cancel_appt_get_date_time: 📆 Parsed → Day: {spoken_day}, Time: {spoken_time}")

        # --- Build UTC search window --------------------------------------------
        try:
            utc_start, utc_end = build_timeslot_range(spoken_day, spoken_time)
            debug_print(f"cancel_appt_get_date_time: ✅ UTC range → Start: {utc_start}, End: {utc_end}")
        except Exception as e:
            debug_print(f"cancel_appt_get_date_time: ❌ build_timeslot_range failed → {e} → iterator")
            cancel_ctx["iter_index"] = 0
            session_data[call_sid]["stage"] = "cancel_appt_iterate"
            resp.redirect("/voice")
            return str(resp)

        # Save spoken + window in session (useful for confirm/logging)
        cancel_ctx["day"]       = spoken_day
        cancel_ctx["time"]      = spoken_time
        cancel_ctx["utc_start"] = utc_start
        cancel_ctx["utc_end"]   = utc_end

        # --- Resolve/search calendars ------------------------------------------
        calendar_id = cancel_ctx.get("calendar_id")
        if calendar_id:
            calendars_to_check = [(calendar_id, googleid_dr_name_map.get(calendar_id, "the doctor"))]
        else:
            calendars_to_check = list(googleid_dr_name_map.items())

        matching_event = None
        matched_calendar = None

        for cal_id, friendly_name in calendars_to_check:
            try:
                # Your helper may return None/dict/list; normalize here
                ev = get_upcoming_events(cal_id, phone_norm, utc_start, utc_end, creds, debug=True)
                if isinstance(ev, list) and ev:
                    candidate = ev[0]
                else:
                    candidate = ev if ev else None
            except Exception as e:
                debug_print(f"cancel_appt_get_date_time: ⚠️ get_upcoming_events error for {friendly_name} → {e}")
                candidate = None

            if candidate:
                matching_event = candidate
                matched_calendar = cal_id
                debug_print(f"cancel_appt_get_date_time: ✅ Found matching event in {friendly_name} ({cal_id})")
                break

        # --- Branch: found vs not found -----------------------------------------
        if matching_event and matched_calendar:
            cancel_ctx["calendar_id"]    = matched_calendar
            cancel_ctx["matching_event"] = matching_event
            debug_print("cancel_appt_get_date_time: ▶️ Match confirmed → proceeding to cancel_appt_confirm")
            session_data[call_sid]["stage"] = "cancel_appt_confirm"
            resp.redirect("/voice")
            return str(resp)

        # No exact match at that date/time → iterate through appointments
        debug_print("cancel_appt_get_date_time: 🚫 No matching event at that date/time → falling back to cancel_appt_iterate")
        cancel_ctx["iter_index"] = 0
        session_data[call_sid]["stage"] = "cancel_appt_iterate"
        resp.redirect("/voice")
        return str(resp)



    elif stage == "cancel_appt_iterate":
        # ----------------------------------------------------------------------
        # 🔁 Stage: cancel_appt_iterate
        # Purpose:
        #   - Iterate over the caller’s upcoming appointments and ask,
        #     one-by-one: “Should I cancel this one?”
        #   - Match scope:
        #       * If cancel["calendar_id"] present → search only that doctor’s calendar.
        #       * Else → search across all doctors (googleid_dr_name_map).
        #   - Matching criteria: by phone (10-digit) and DOB (ISO), consistent with
        #     how you embed caller info in event description/extendedProperties.
        #
        # Flow:
        #   1) On first entry, fetch candidate appointments → save in cancel["candidates"].
        #   2) Read the current appointment (at cancel["iter_index"]) aloud and ask
        #      “Cancel this one?” (Yes = 1 / No = 2 / Repeat = 3).
        #   3) YES → set cancel["matching_event"] and jump to cancel_appt_confirm.
        #   4) NO  → iter_index += 1; if exhausted → apologize+hangup OR reschedule→booking.
        #
        # Resilience:
        #   - If phone/DOB missing → bounce to collectors with return path back here.
        #   - Robust speech/DTMF handling (yes/no/3=repeat). Unknown → re-prompt.
        #
        # Side paths:
        #   - If cancel["reschedule"] is True and no match found, go to booking flow.
        # ----------------------------------------------------------------------
        debug_print("cancel_appt_iterate: 📍 Stage entered")

        # --- Ensure buckets exist ------------------------------------------------
        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("cancel", {})
        cancel_ctx = session_data[call_sid]["cancel"]

        # --- Guards: need phone + DOB (normalized) -------------------------------
        def _normalize_10(d: str) -> str:
            d = "".join(ch for ch in (d or "") if ch.isdigit())
            return d[1:] if len(d) == 11 and d.startswith("1") else d

        phone_norm = _normalize_10(cancel_ctx.get("phone") or session_data[call_sid].get("customer", {}).get("phone"))
        dob_val    = (cancel_ctx.get("dob") or session_data[call_sid].get("customer", {}).get("dob") or "").strip()

        if len(phone_norm) != 10:
            debug_print("cancel_appt_iterate: ❌ phone missing/invalid → redirecting to collect_phone")
            session_data[call_sid]["return_stage"] = "cancel_appt_iterate"
            session_data[call_sid]["stage"] = "collect_phone"
            gather = make_gather(
                "To find your appointment, please provide your ten digit phone number including area code. "
                "You can say it, or type the digits and press pound.",
                hints="zero one two three four five six seven eight nine"
            )
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        if not dob_val:
            debug_print("cancel_appt_iterate: ❌ DOB missing → redirecting to cancel_appt_get_dob")
            session_data[call_sid]["return_stage"] = "cancel_appt_iterate"
            session_data[call_sid]["stage"] = "cancel_appt_get_dob"
            gather = make_gather(
                "Please say your date of birth, or enter two digits for month, two for day, and four for year, then press pound."
            )
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        cancel_ctx["phone"] = phone_norm
        cancel_ctx["dob"]   = dob_val

        # --- Helper: friendly local time for speaking/SMS ------------------------
        def _fmt_local_friendly(utc_iso: str) -> str:
            if not utc_iso:
                return ""
            try:
                from datetime import datetime
                import pytz
                dt_utc = datetime.fromisoformat(utc_iso.replace("Z", "+00:00"))
                tz = pytz.timezone("America/Chicago")
                dt_local = dt_utc.astimezone(tz)
                try:
                    return dt_local.strftime("%A, %B %-d at %-I:%M %p")
                except Exception:
                    return dt_local.strftime("%A, %B %d at %I:%M %p").lstrip("0").replace(" 0", " ")
            except Exception as e:
                debug_print(f"cancel_appt_iterate: ⚠️ time format failed → {e}")
                return utc_iso

        # --- Helper: fetch candidate appts for this caller -----------------------
        def _fetch_candidates() -> list:
            """
            Returns a list of dicts:
                {
                "event_id": "...",
                "calendar_id": "...",
                "doctor_name": "Dr. X",
                "start_utc": "YYYY-MM-DDTHH:MM:SS+00:00",
                "end_utc":   "YYYY-MM-DDTHH:MM:SS+00:00",
                "friendly":  "Tuesday, August 12 at 9:00 AM",
                "summary":   "...",    # optional
                "location":  "..."     # optional
                }
            """
            horizon_days = 90
            limit = 25
            results = []

            # Scope: single calendar if set; else all mapped doctors
            target_cals = []
            if cancel_ctx.get("calendar_id"):
                target_cals = [(cancel_ctx["calendar_id"], googleid_dr_name_map.get(cancel_ctx["calendar_id"], "the doctor"))]
            else:
                # Iterate all doctors
                target_cals = list(googleid_dr_name_map.items())

            for cal_id, friendly_name in target_cals:
                try:
                    # 🔎 Replace with your real helper that filters by phone/DOB:
                    #     find_future_events_for_caller(calendar_id, phone, dob, creds, horizon_days, limit, debug)
                    events = find_future_events_for_caller(
                        calendar_id=cal_id,
                        phone=phone_norm,
                        dob=dob_val,
                        creds=creds,
                        horizon_days=horizon_days,
                        limit=limit,
                        debug=True
                    )
                except NameError:
                    debug_print("cancel_appt_iterate: ⚠️ Missing helper find_future_events_for_caller(...) → returning no events for this calendar")
                    events = []
                except Exception as e:
                    debug_print(f"cancel_appt_iterate: ❌ fetch failed for {friendly_name} ({cal_id}) → {e}")
                    events = []

                # Normalize each event into our shape
                for ev in events or []:
                    start = ev.get("start_utc") or ev.get("start")
                    end   = ev.get("end_utc")   or ev.get("end")
                    eid   = ev.get("id") or ev.get("event_id")
                    if not (start and end and eid):
                        continue
                    results.append({
                        "event_id": eid,
                        "calendar_id": cal_id,
                        "doctor_name": friendly_name,
                        "start_utc": start,
                        "end_utc": end,
                        "friendly": _fmt_local_friendly(start),
                        "summary": ev.get("summary", ""),
                        "location": ev.get("location", "")
                    })

            # Sort by start time ascending if timestamps exist
            try:
                results.sort(key=lambda r: r.get("start_utc", ""))
            except Exception:
                pass

            return results

        # --- Load or create candidate list + iterator index ----------------------
        candidates = cancel_ctx.get("candidates")
        if candidates is None:
            debug_print("cancel_appt_iterate: 🔍 Fetching candidate appointments for caller")
            candidates = _fetch_candidates()
            cancel_ctx["candidates"] = candidates
            cancel_ctx["iter_index"] = 0

        idx = int(cancel_ctx.get("iter_index", 0))

        # --- If no candidates, decide end path (apology vs reschedule→booking) ---
        if not candidates:
            if cancel_ctx.get("reschedule") is True:
                debug_print("cancel_appt_iterate: 🚫 No appointments found → reschedule path → booking")
                session_data[call_sid]["stage"] = "booking"
                gather = make_gather(
                    "I couldn't find any upcoming appointments to cancel. "
                    "Would you like to book a new appointment? Please say the doctor's name to begin."
                )
                resp.append(gather)
                return str(resp)
            else:
                debug_print("cancel_appt_iterate: 🚫 No appointments found → apologizing and hanging up")
                resp.say(gpt_speak("I couldn't find any upcoming appointments on file. Please call us if you need more help. Goodbye."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

        # --- Pull current candidate ---------------------------------------------
        if idx >= len(candidates):
            # exhausted list
            if cancel_ctx.get("reschedule") is True:
                debug_print("cancel_appt_iterate: 📦 All appointments checked → reschedule path → booking")
                session_data[call_sid]["stage"] = "booking"
                gather = make_gather(
                    "We’ve checked all your upcoming appointments. Would you like to book a new time now? "
                    "Please say the doctor's name to begin."
                )
                resp.append(gather)
                return str(resp)
            else:
                debug_print("cancel_appt_iterate: 📦 All appointments checked → no match → apologize + hangup")
                resp.say(gpt_speak("I couldn't find an appointment to cancel. Please call us if you need more help. Goodbye."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

        cand = candidates[idx]
        doc_name   = cand.get("doctor_name", "the doctor")
        when_friendly = cand.get("friendly", "")
        summary    = cand.get("summary", "")
        location   = cand.get("location", "")

        # --- Read out the candidate and ask for confirmation ---------------------
        try:
            dtmf_digits = (request.values.get("Digits") or "").strip()
        except Exception:
            dtmf_digits = ""
        speech_text = (speech_result or "").strip().lower()

        # Interpret user response (if any on this pass)
        yes_tokens = {"yes", "yeah", "yep", "yup", "correct", "sure", "affirmative"}
        no_tokens  = {"no", "nope", "nah", "negative"}
        repeat_tokens = {"repeat", "again", "one more time"}

        answered_yes = (dtmf_digits == "1") or any(tok in speech_text for tok in yes_tokens)
        answered_no  = (dtmf_digits == "2") or any(tok in speech_text for tok in no_tokens)
        asked_repeat = (dtmf_digits == "3") or any(tok in speech_text for tok in repeat_tokens)

        # If user already answered on this same turn, act on it
        if answered_yes:
            debug_print(f"cancel_appt_iterate: ✅ User chose YES for {when_friendly} with {doc_name}")
            cancel_ctx["matching_event"] = cand
            session_data[call_sid]["stage"] = "cancel_appt_confirm"
            resp.redirect("/voice")
            return str(resp)

        if answered_no:
            cancel_ctx["iter_index"] = idx + 1
            debug_print(f"cancel_appt_iterate: ↘️ User chose NO → advancing to index {cancel_ctx['iter_index']}")
            resp.redirect("/voice")
            return str(resp)

        if asked_repeat:
            debug_print("cancel_appt_iterate: 🔁 User asked to repeat the current appointment details")
            # fall through to read the same candidate again

        # Speak current candidate details and prompt
        prompt_lines = [
            f"I found an appointment on {when_friendly} with {doc_name}."
        ]
        if summary:
            prompt_lines.append(f"The subject is {summary}.")
        if location:
            prompt_lines.append(f"Location: {location}.")
        prompt_lines.append("Should I cancel this one? Say yes or no. You can press 1 for yes, 2 for no, or 3 to hear it again.")
        prompt = " ".join(prompt_lines)

        debug_print(f"cancel_appt_iterate: 🔊 Prompting candidate idx {idx}/{len(candidates)-1} → {prompt}")
        gather = make_gather(prompt, hints="yes no repeat")
        resp.append(gather)
        return str(resp)






    elif stage == "cancel_appt_confirm":
    # ----------------------------------------------------------------------
    # 📌 Stage: cancel_appt_confirm
    #
    # Finalize the cancellation:
    #   1) Read details from session: phone, doctor, spoken day/time, UTC window, calendar_id, dob.
    #   2) Use a pre-fetched matching event if present; otherwise query for an event in the window.
    #   3) If found:
    #        - Delete from Google Calendar (by eventId).
    #        - Delete from local doctor JSON via cancel_appointment_by_name(doctor, phone, utc_start, dob).
    #        - Speak a friendly confirmation time back to caller.
    #   4) If not found:
    #        - Inform the caller no matching appt could be located.
    #   5) If reschedule flag is set, jump to booking; otherwise clear session and hang up.
    # ----------------------------------------------------------------------
    debug_print("📍 Stage: cancel_appt_confirm")

    cancel_ctx  = session_data[call_sid].get("cancel", {})
    phone       = cancel_ctx.get("phone")
    doctor      = cancel_ctx.get("doctor")
    spoken_day  = cancel_ctx.get("day")
    spoken_time = cancel_ctx.get("time")
    utc_start   = cancel_ctx.get("utc_start")
    utc_end     = cancel_ctx.get("utc_end")
    calendar_id = cancel_ctx.get("calendar_id")
    dob         = cancel_ctx.get("dob") or session_data[call_sid].get("customer", {}).get("dob")

    debug_print(f"📱 Phone: {phone}")
    debug_print(f"👨‍⚕️ Doctor: {doctor}")
    debug_print(f"🗓️ Day/Time: {spoken_day}, {spoken_time}")
    debug_print(f"🌍 UTC window: {utc_start} → {utc_end}")
    debug_print(f"📅 Calendar ID: {calendar_id}")
    debug_print(f"🎂 DOB (for cancellation match): {dob or '∅'}")

    # ❌ If no calendar ID → cannot proceed with Google Calendar deletion
    if not calendar_id:
        resp.say(gpt_speak("Sorry, I couldn't find the doctor's calendar. Please try again later."), VOICE)
        resp.hangup()
        session_data.pop(call_sid, None)
        return str(resp)

    # Use pre-fetched match or search now (normalize possible list return)
    event_to_cancel = cancel_ctx.get("matching_event")
    if not event_to_cancel:
        try:
            fetched = get_upcoming_events(calendar_id, phone, utc_start, utc_end, creds, debug=True)
            if isinstance(fetched, list):
                event_to_cancel = fetched[0] if fetched else None
            else:
                event_to_cancel = fetched
        except Exception as e:
            debug_print(f"❌ Error fetching event for cancel → {e}")
            event_to_cancel = None

    if event_to_cancel:
        event_id = event_to_cancel.get("id")
        try:
            # 🗑️ Delete from Google Calendar
            service = build("calendar", "v3", credentials=creds)
            service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
            debug_print(f"🗑️ Deleted calendar event id={event_id}")

            # 🎙️ Friendly local time to speak back
            from dateutil import parser as dtparser
            try:
                start_str = (
                    (event_to_cancel.get("start") or {}).get("dateTime", "")
                    or (event_to_cancel.get("start") or {}).get("date", "")
                )
                dt = dtparser.parse(start_str)
                import pytz
                friendly = dt.astimezone(pytz.timezone("America/Chicago")).strftime("%B %-d at %-I:%M %p")
            except Exception:
                friendly = f"{spoken_day} at {spoken_time}"

            # 🧹 Remove from local JSON mapping (now with DOB match)
            try:
                removed = cancel_appointment_by_name(doctor_name=doctor, phone=phone, utc_start=utc_start, dob=dob)
                if removed:
                    debug_print(f"🧹 Local mapping removed for {doctor} using phone+dob+time")
                else:
                    debug_print(f"⚠️ Local JSON had no matching record (phone+dob+time) to remove.")
            except Exception as e:
                debug_print(f"⚠️ Local JSON remove error → {e}")

            # ✅ Confirm to caller
            resp.say(gpt_speak(f"Your appointment with {doctor} on {friendly} has been cancelled. Thank you!"), VOICE)

        except Exception as e:
            debug_print(f"❌ Calendar delete failed → {e}")
            resp.say(gpt_speak("Sorry, something went wrong while cancelling your appointment."), VOICE)
    else:
        debug_print("🚫 No matching appointment found to cancel.")
        resp.say(gpt_speak("I'm sorry, I couldn't find an appointment under that phone number and time."), VOICE)

    # 🔁 If rescheduling after cancel
    if session_data[call_sid].get("reschedule_after_cancel"):
        debug_print("🔁 Reschedule requested; transitioning to booking.")
        session_data[call_sid]["stage"] = "booking"
        session_data[call_sid].pop("cancel", None)

        doctor_list_str = ", ".join(googleid_dr_name_map.values())
        gather = make_gather("Now, please tell me which doctor you'd like to reschedule with.", hints=doctor_list_str)
        resp.append(gather)
        return str(resp)

    # 🧼 End cancellation flow
    session_data.pop(call_sid, None)
    debug_print("🧼 Session data cleared after cancellation.")
    return str(resp)




   



if __name__ == "__main__":
    
    # ------------------------
    # 🔁 Call this once at startup
    # ------------------------

    load_doctor_appointments()


    app.run(host="0.0.0.0", port=5000)
