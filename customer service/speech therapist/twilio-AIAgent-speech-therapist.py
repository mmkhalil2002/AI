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
SPEECH_INPUT_DURATION = int(os.getenv("SPEECH_INPUT_DURATION", 10))  # how long to wait for input
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
from googleapiclient.discovery import build
import pytz
from typing import List, Dict

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




from datetime import datetime, timedelta, date, time
from typing import Tuple, Union
import pytz
import re

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
def confirm_appointment_by_name(doctor_name: str, phone: str, utc_start: str, calendar_id: str):
    """Add a new appointment to the doctor's table and save to JSON file."""

    filename = sanitize_filename(doctor_name).replace(".json", "")
    full_path = get_doctor_filename(doctor_name)

    print(f"[confirm_appointment_by_name] 🔍 Loading file: {full_path}")

    # 📥 Load or initialize doctor_appointments[filename] as a list
    if os.path.exists(full_path):
        try:
            with open(full_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    doctor_appointments[filename] = data
                    print(f"[confirm_appointment_by_name] ✅ Loaded existing list with {len(data)} appointments")
                else:
                    print(f"[confirm_appointment_by_name] ⚠️ JSON was a dict instead of a list. Resetting.")
                    doctor_appointments[filename] = []
        except Exception as e:
            print(f"[confirm_appointment_by_name] ⚠️ Failed to parse JSON file → {e}")
            doctor_appointments[filename] = []
    else:
        print(f"[confirm_appointment_by_name] 📂 No file found — starting new list")
        doctor_appointments[filename] = []

    # 🆕 Append the new appointment
    new_appt = {
        "phone": phone,
        "time": utc_start,
        "calendar_id": calendar_id
    }
    doctor_appointments[filename].append(new_appt)
    print(f"[confirm_appointment_by_name] ➕ Appended: {new_appt}")

    # 💾 Save updated list back to file
    try:
        with open(full_path, "w") as f:
            json.dump(doctor_appointments[filename], f, indent=2)
        print(f"[confirm_appointment_by_name] 💾 Saved to {full_path}")
    except Exception as e:
        print(f"[confirm_appointment_by_name] ❌ Failed to write JSON → {e}")



def normalize_phone_digits(phone: str) -> str:
    """Digits-only normalization for matching (calendar description & JSON)."""
    return ''.join(ch for ch in (phone or "") if ch.isdigit())



from dateutil import parser
import os
import json
import os, json
from dateutil import parser as dtparser

def cancel_appointment_by_name(doctor_name: str, phone: str, utc_start: str) -> bool:
    """
    Remove a doctor's appointment by exact UTC time and phone.
    All times normalized to ISO UTC before comparison.
    Keeps other appointments intact.
    """
    key = sanitize_filename(doctor_name).replace(".json", "")
    full_path = get_doctor_filename(doctor_name)
    phone_digits = normalize_phone_digits(phone)

    debug_print(f"🩺 cancel_appointment_by_name → doctor={doctor_name}, phone={phone_digits}, utc_start={utc_start}")
    if not os.path.exists(full_path):
        debug_print(f"⚠️ File not found: {full_path}")
        return False

    try:
        with open(full_path, "r") as f:
            data = json.load(f)
            if not isinstance(data, list):
                debug_print(f"❌ JSON not a list for {full_path}")
                return False
    except Exception as e:
        debug_print(f"❌ Read error {full_path} → {e}")
        return False

    # normalize target UTC
    try:
        target_norm = dtparser.isoparse(utc_start).astimezone().astimezone(tz=None).isoformat()
    except Exception as e:
        debug_print(f"❌ utc_start parse error → {e}")
        return False

    kept, removed = [], 0
    for appt in data:
        ap_phone = normalize_phone_digits(appt.get("phone", ""))
        ap_time_raw = appt.get("time", "")
        try:
            ap_time_norm = dtparser.isoparse(ap_time_raw).astimezone().astimezone(tz=None).isoformat()
        except Exception as e:
            debug_print(f"⚠️ skip invalid appt time '{ap_time_raw}' → {e}")
            kept.append(appt)
            continue

        if ap_phone == phone_digits and ap_time_norm == target_norm:
            removed += 1
            debug_print(f"🗑️ Removing appt → phone={ap_phone}, time={ap_time_norm}")
        else:
            kept.append(appt)

    if removed == 0:
        debug_print(f"⚠️ No appointment found for phone={phone_digits} time={target_norm}")
        return False

    try:
        with open(full_path, "w") as f:
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

       # Create a <Gather> TwiML block to collect voice input from the caller
       gather = Gather(
                         input="speech",         # Expect spoken input
                         action="/voice",        # Send the result to the /voice route for further processing
                         method="POST",          # Use POST method for the follow-up request
                         speech_model="phone_call",  # 🎯 Optimized for voice calls
                         bargeIn=True,  
                        timeout= SPEECH_INPUT_DURATION  # Wait up to 5 seconds for a response before timing out
                      )

       # Define a friendly prompt to ask the customer what they want to do
       prompt = "    Thank you for calling EPIC therapist: would you like  to book an appointment, cancel an appointment, change an appointment  or leave a message."

       # Use GPT to generate a dynamic and friendly greeting based on the prompt
       gather.say(gpt_speak(prompt),VOICE)  # This adds spoken text to the <Gather> block
       """
       Speaks the message inside <Say>

        Listens for the caller’s voice input for 5 seconds

        Sends the speech result to /voice for further handling

        <Response>
        <Gather input="speech" action="/voice" method="POST" timeout="5">
            <Say>Hello! Welcome to Epic Therapist Clinic. Would you like to book an appointment or leave a message?</Say>
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
            gather = Gather(
                             input="speech",
                             action="/voice",
                             method="POST",
                             speech_model="phone_call",
                             bargeIn=True,
                             timeout=SPEECH_INPUT_DURATION
                         )
            gather.say(gpt_speak("Thank you for Calling EPIC thearapist : Please tell me if you'd like to book an appointment, cancel one, reschedule, or leave a message."))
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
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=SPEECH_INPUT_DURATION,
                speech_model="phone_call",
                bargeIn=True,
                hints=", ".join(doctor_names)
            )

            gather.say(gpt_speak(prompt), VOICE)
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
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=SPEECH_INPUT_DURATION,
                speech_model="phone_call",
                bargeIn=True,
                hints=", ".join(doctor_names)
            )
            gather.say(gpt_speak(prompt), VOICE)
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
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=SPEECH_INPUT_DURATION,
                speech_model="phone_call",
                bargeIn=True,
                hints=", ".join(googleid_dr_name_map.values())
            )
            gather.say(gpt_speak(prompt), VOICE)
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
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                speech_model="phone_call",
                bargeIn=True,
                timeout=SPEECH_INPUT_DURATION
            )
            gather.say(gpt_speak(
                "Sorry, I didn’t catch that. Would you like to book an appointment, cancel one, reschedule, or leave a message?"
            ), VOICE)
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
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                speech_model="phone_call",
                bargeIn=True,
                timeout=SPEECH_INPUT_DURATION,
                hints=doctor_list_str
            )
            gather.say(gpt_speak("Please say the name of the doctor you'd like to book with."), VOICE)
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
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                speech_model="phone_call",
                bargeIn=True,
                timeout=SPEECH_INPUT_DURATION,
                hints=doctor_list_str
            )
            retry_prompt = (
                f"I couldn't match that to a doctor. Available doctors are: {doctor_list_str}. "
                "Please say the doctor name again."
            )
            gather.say(gpt_speak(retry_prompt), VOICE)
            resp.append(gather)
            return str(resp)

        # ------------------------------------------------------------------
        # ✅ 4. Success → Ask for time
        # ------------------------------------------------------------------
        session_data[call_sid]["doctor_id"] = matched_id
        session_data[call_sid]["stage"] = "ask_time_date"

        friendly_name = googleid_dr_name_map[matched_id]
        time_prompt = f"What time would you like to book with {friendly_name}?"

        gather = Gather(
            input="speech",
            action="/voice",
            method="POST",
            speech_model="phone_call",
            bargeIn=True,
            timeout=SPEECH_INPUT_DURATION
        )
        gather.say(gpt_speak(time_prompt), VOICE)
        resp.append(gather)
        return str(resp)


    # ----------------------------------------------------------------------
    # 📍 Stage: ask_time_date
    # Triggered after doctor is selected. This routine:
    #  1. Parses user's spoken time (e.g. "July 3rd 12 30")
    #  2. Checks Google Calendar for availability
    #  3. If available, confirms and moves to collect name/phone/address
    # ----------------------------------------------------------------------
    elif stage == "ask_time_date":
        # ----------------------------------------------------------------------
        # 🧠 Step 1: Parse date and time from spoken input
        # ----------------------------------------------------------------------
        print(f"🗣️ Received spoken time: {speech_result}")
        time_info = smart_parse_time(speech_result)

        if not time_info or not isinstance(time_info, tuple) or len(time_info) != 2:
            session_data[call_sid]["retry_time"] = session_data[call_sid].get("retry_time", 0) + 1
            retry_count = session_data[call_sid]["retry_time"]
            print(f"⚠️ Time parsing failed. Retry count: {retry_count}")

            if retry_count >= 3:
                print("❌ Max retries reached. Ending call.")
                resp.say(gpt_speak("Sorry, I still couldn't understand the time. Please try again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=SPEECH_INPUT_DURATION,
                speech_model="phone_call",
                bargeIn=True
            )
            gather.say(gpt_speak("Please say the date and time again, for example, July 3rd at 9 AM."), VOICE)
            resp.append(gather)
            return str(resp)

        # ✅ Valid date and time extracted
        spoken_day, spoken_time = time_info
        print(f"📆 Extracted → Day: {spoken_day}, Time: {spoken_time}")
        session_data[call_sid]["spoken_day"] = spoken_day
        session_data[call_sid]["spoken_time"] = spoken_time

        try:
            appointment_start, appointment_end = build_timeslot_range(spoken_day, spoken_time)
            session_data[call_sid]["appointment_time"] = {
                "start": appointment_start,
                "end": appointment_end
            }
            print(f"📆 Appointment requested → Start: {appointment_start}, End: {appointment_end}")
        except Exception as e:
            print(f"❌ Failed to build appointment time range from '{spoken_day}' and '{spoken_time}': {e}")
            session_data[call_sid]["retry_time"] = session_data[call_sid].get("retry_time", 0) + 1

            if session_data[call_sid]["retry_time"] >= 3:
                print("❌ Max retries reached during build_timeslot_range.")
                resp.say(gpt_speak("Sorry, I couldn’t understand the time you mentioned. Please try again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=SPEECH_INPUT_DURATION,
                speech_model="phone_call",
                bargeIn=True
            )
            gather.say(gpt_speak("I didn't catch that clearly. Please repeat the date and time, like July 3rd at 9 AM."), VOICE)
            resp.append(gather)
            return str(resp)

        # 🔎 Check availability
        doctor_id = session_data[call_sid]["doctor_id"]
        calendar_id = doctor_id
        print(f"👨‍⚕️ Checking calendar ID: {calendar_id}")

        if not is_time_slot_available(calendar_id, appointment_start, appointment_end, creds):
            print("❌ Requested time slot is not available")

            # 📅 Fetch alternative slots
            alts = get_next_available_slots(
                calendar_id,
                creds,
                limit=3,
                duration_minutes=APPOINTMENT_DURATION_MINUTES
            )

            if alts:
                options = " or ".join([slot["friendly"] for slot in alts])
                prompt = f"That time is not available. Would you like to book on {options}?"
                print(f"💡 Offering alternatives: {options}")
            else:
                prompt = "That time is not available, and I couldn't find any open slots soon. Please try again later."
                print("⚠️ No alternative slots found.")

            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=SPEECH_INPUT_DURATION,
                speech_model="phone_call",
                bargeIn=True
            )
            gather.say(gpt_speak(prompt), VOICE)
            resp.append(gather)
            return str(resp)

        # ✅ Slot is free → proceed to collect first name
        print("✅ Slot is available. Moving to name collection.")
        session_data[call_sid]["stage"] = "collect_first_name"

        gather = Gather(
            input="speech",
            action="/voice",
            method="POST",
            timeout=SPEECH_INPUT_DURATION,
            speech_model="phone_call",
            bargeIn=True
        )
        gather.say(gpt_speak("Thanks. What is your first name?"), VOICE)
        resp.append(gather)
        return str(resp)




    elif stage == "collect_first_name":
        # ----------------------------------------------------------------------
        # 🧍 Stage: Collect First Name
        # ----------------------------------------------------------------------
        first_name = speech_result.strip()
        print(f"📛 collect_first_name : Collected first name: {first_name}")

        if not first_name or len(first_name.split()) > 2:
            # ⚠️ If unclear or too long, ask again
            gather = Gather(
                                input="speech",
                                action="/voice",
                                method="POST",
                                speech_model="phone_call",
                                bargeIn=True,
                                timeout=SPEECH_INPUT_DURATION
                            )
            gather.say(gpt_speak("I didn't catch that clearly. Please say your first name again."), VOICE)
            resp.append(gather)
            return str(resp)

        # ✅ Save and move to last name
        session_data[call_sid]["customer"] = {"first_name": first_name}
        session_data[call_sid]["stage"] = "collect_last_name"

        gather = Gather(
                        input="speech",
                        action="/voice",
                        method="POST",
                        speech_model="phone_call",
                        bargeIn=True,
                        timeout=SPEECH_INPUT_DURATION
                       )
        gather.say(gpt_speak("Thank you. Now, what is your last name?"), VOICE)
        resp.append(gather)
        return str(resp)

    elif stage == "collect_last_name":
        try:
            last = speech_result.strip()
            print(f"👤 collect_last_name: {last}")

            if not last:
                gather = Gather(
                                  input="speech",
                                  action="/voice",
                                  method="POST",
                                  speech_model="phone_call",
                                  bargeIn=True,
                                  timeout=SPEECH_INPUT_DURATION
                                )
                gather.say(gpt_speak("Sorry, I didn't catch your last name. Please repeat it."), VOICE)
                resp.append(gather)
                return str(resp)

            # ✅ Save and move to next stage
            session_data[call_sid]["customer"]["last_name"] = last
            session_data[call_sid]["stage"] = "collect_phone"
            print(f"✅ Stored last name: {last}")

            gather = Gather(
                            input="speech",
                            action="/voice",
                            method="POST",
                            speech_model="phone_call",
                            bargeIn=True,
                            timeout=SPEECH_INPUT_DURATION
                            )
            gather.say(gpt_speak("Got it. What is your phone number, please?"), VOICE)
            resp.append(gather)
            return str(resp)

        except Exception as e:
            print(f"❌ Exception in collect_last_name stage: {e}")
            resp.say(gpt_speak("Sorry, there was an error. Let's try again later."), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)

    elif stage == "collect_phone":
        # ----------------------------------------------------------------------
        # ☎️ Stage: Collect Phone Number
        # ----------------------------------------------------------------------
        import re

        raw_phone = speech_result.strip()
        print(f"📱 collect_phone (raw): '{raw_phone}'")

        # 🔧 Normalize and compact the phone number (remove all non-digit characters)
        digits_only = re.sub(r"\D", "", raw_phone)
        print(f"📞 Cleaned phone number (digits only): '{digits_only}'")
        print(f"🔢 Phone number length: {len(digits_only)} digits")

        if len(digits_only) < 7:
            print("❌ Phone number too short. Re-prompting user.")
            # Not enough digits → re-prompt
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                speech_model="phone_call",
                bargeIn=True,
                timeout=SPEECH_INPUT_DURATION
            )
            gather.say(
                gpt_speak("Sorry, I didn't catch your phone number clearly. Please say it again, digit by digit."),
                VOICE
            )
            resp.append(gather)
            return str(resp)

        # ✅ Save cleaned number
        print(f"✅ Valid phone number accepted: {digits_only}")
        session_data[call_sid]["customer"]["phone"] = digits_only
        session_data[call_sid]["stage"] = "collect_address"

        # 🏠 Prompt for address
        gather = Gather(
            input="speech",
            action="/voice",
            method="POST",
            speech_model="phone_call",
            bargeIn=True,
            timeout=SPEECH_INPUT_DURATION
        )
        gather.say(gpt_speak("Thank you. What is your full address, please?"), VOICE)
        resp.append(gather)
        return str(resp)


    elif stage == "collect_address":
        # ----------------------------------------------------------------------
        # 🏠 Stage: Collect Customer Address and finalize appointment booking
        # ----------------------------------------------------------------------
        address = speech_result.strip()
        print(f"collect_address: 📬 Collected address: {address}")

        session_data[call_sid]["customer"]["address"] = address
        session_data[call_sid]["stage"] = "book_appt_confirm"  # ✅ next stage

        customer = session_data[call_sid]["customer"]
        appointment = session_data[call_sid]["appointment_time"]
        doctor_id = session_data[call_sid]["doctor_id"]

        # 🧩 Assemble full name safely
        full_name = f"{customer.get('first_name', '')} {customer.get('last_name', '')}".strip()
        session_data[call_sid]["customer"]["name"] = full_name

        # 📞 Normalize phone number
        import re
        raw_phone = customer.get("phone", "")
        normalized_phone = re.sub(r"[^\d]", "", raw_phone)
        session_data[call_sid]["customer"]["phone"] = normalized_phone
        print(f"collect_address: 📞 Final stored phone: {normalized_phone}")

        # 📅 Handle time conversion: appointment time is in UTC
        try:
            from datetime import datetime
            import pytz

            local_tz = pytz.timezone("America/Chicago")
            start_utc = datetime.fromisoformat(appointment["start"]).astimezone(pytz.utc)
            end_utc = datetime.fromisoformat(appointment["end"]).astimezone(pytz.utc)

            print(f"collect_address: 📅 Local time slot → Start: {start_utc.isoformat()}, End: {end_utc.isoformat()}")
        except Exception as e:
            print(f"collect_address: ❌ Error converting to UTC: {e}")
            resp.say(gpt_speak("Sorry, I had trouble confirming your appointment. Please try again later."), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)

        # 📅 Build Google Calendar event
        try:
            calendar = build("calendar", "v3", credentials=creds)
            event = {
                "summary": f"Appointment for {full_name}",
                "description": f"Name: {full_name}\nPhone: {normalized_phone}\nAddress: {address}",
                "start": {"dateTime": start_utc.isoformat(), "timeZone": "UTC"},
                "end": {"dateTime": end_utc.isoformat(), "timeZone": "UTC"},
            }

            calendar.events().insert(calendarId=doctor_id, body=event).execute()
            print("collect_address: ✅ Google Calendar event created")
        except Exception as e:
            print(f"collect_address: ❌ Failed to create calendar event: {e}")
            resp.say(gpt_speak("Sorry, something went wrong while saving your appointment. Please try again later."), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)

        # 🔁 Forward to /voice to trigger book_appt_confirm
        print("collect_address: 🔁 Redirecting to /voice to confirm booking")
        resp.redirect("/voice")
        return str(resp)
    


    

# -# -----------------------------------------------------------------
    elif stage == "book_appt_confirm":
        print("book_appt_confirm: 📍 Stage entered")

        # 🆔 Doctor info
        doctor_id = session_data[call_sid].get("doctor_id")
        print(f"book_appt_confirm: 🧩 Raw doctor_id from session → {doctor_id}")
        doctor_name = googleid_dr_name_map.get(doctor_id, "the doctor")
        print(f"book_appt_confirm: 👨‍⚕️ Resolved doctor_name → {doctor_name}")

        # 🕐 Appointment info
        appointment_time = session_data[call_sid].get("appointment_time", {}).get("start")
        print(f"book_appt_confirm: 🕓 Raw appointment_time UTC → {appointment_time}")
        formatted_time = ""
        if appointment_time:
            from datetime import datetime
            import pytz
            try:
                dt_utc = datetime.fromisoformat(appointment_time.replace("Z", "+00:00"))
                tz = pytz.timezone("America/Chicago")
                dt_local = dt_utc.astimezone(tz)
                formatted_time = dt_local.strftime("%A, %B %-d at %-I:%M %p")
                print(f"book_appt_confirm: 📆 Formatted appointment time (local) → {formatted_time}")
            except Exception as e:
                print(f"book_appt_confirm: ⚠️ Failed to parse appointment time → {e}")
                resp.say(gpt_speak("Sorry, we couldn't confirm the appointment time."), VOICE)
                resp.hangup()
                return str(resp)
        else:
            print("book_appt_confirm: ❌ No appointment time found in session data")
            resp.say(gpt_speak("Appointment time missing. Goodbye!"), VOICE)
            resp.hangup()
            return str(resp)

        # 🧍 Customer info
        customer = session_data[call_sid].get("customer", {})
        print(f"book_appt_confirm: 🧾 Raw customer object → {customer}")
        customer_name = customer.get("name", "")
        customer_phone = customer.get("phone")
        print(f"book_appt_confirm: 👤 Extracted customer name → {customer_name}")
        print(f"book_appt_confirm: 📞 Extracted customer phone → {customer_phone}")

        # 📥 Save confirmation in doctor table with calendar_id
        try:
            confirm_appointment_by_name(
                doctor_name=doctor_name,
                phone=customer_phone,
                utc_start=appointment_time,
                calendar_id=doctor_id
            )
            print(f"book_appt_confirm: ✅ Mapping saved → {doctor_name}.json: {customer_phone} → {appointment_time} (Calendar ID: {doctor_id})")
        except Exception as e:
            print(f"book_appt_confirm: ⚠️ Failed to save appointment in doctor table → {e}")

        # 🗣️ Voice confirmation message
        confirmation_message = (
            f"Your appointment with {doctor_name} has been successfully booked. "
            "We look forward to seeing you. Goodbye!"
        )
        print(f"book_appt_confirm: 🗣️ Speaking confirmation message → {confirmation_message}")
        resp.say(gpt_speak(confirmation_message), VOICE)

        # 📩 Send SMS confirmation
        if customer_phone:
            try:
                print(f"book_appt_confirm: 📦 Preparing to send SMS to → {customer_phone}")

                # ✅ Normalize phone number (E.164 format, US default)
                digits_only = ''.join(filter(str.isdigit, customer_phone))
                print(f"book_appt_confirm: 🔢 Digits-only phone → {digits_only}")
                if not digits_only.startswith("1"):
                    digits_only = "1" + digits_only
                customer_phone = f"+{digits_only}"
                print(f"book_appt_confirm: ☎️ Normalized E.164 phone → {customer_phone}")

                sms_text = f"Hi {customer_name}, your appointment with {doctor_name} is confirmed"
                if formatted_time:
                    sms_text += f" on {formatted_time}"
                sms_text += ". Thank you for choosing Epic Therapist Clinic."

                print(f"book_appt_confirm: 📝 Final SMS text → {sms_text}")

                message = client.messages.create(
                    body=sms_text,
                    from_=TWILIO_PHONE_NUMBER,
                    to=customer_phone
                )

                print(f"book_appt_confirm: 📤 SMS sent to {customer_phone}")
                print(f"book_appt_confirm: 🧾 SMS SID: {message.sid}, Status: {message.status}")

            except Exception as e:
                print(f"book_appt_confirm: ❌ SMS send failed → {e}")
        else:
            print("book_appt_confirm: ⚠️ No phone number provided — skipping SMS.")

        # 📞 End the call
        print("book_appt_confirm: 📞 Hanging up the call")
        resp.hangup()

        # 🧹 Clear session
        session_data.pop(call_sid, None)
        print("book_appt_confirm: 🧼 Session data cleared")

        return str(resp)






    
   
    elif stage == "cancel_appointment":
        # ----------------------------------------------------------------------
        # 🔄 This stage handles when a user wants to cancel an appointment
        # and just spoke the doctor’s name (e.g., "Dr. Omar", or "cancel with Dr. Alex")
        # ----------------------------------------------------------------------

        import string
        selected_text = speech_result or ""
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
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                speech_model="phone_call",
                bargeIn=True,
                timeout=SPEECH_INPUT_DURATION
            )

            retry_prompt = (
                f"I didn't recognize that name. Available doctors are: {doctor_list}. "
                "Please say the name again."
            )
            try:
                gather.say(gpt_speak(retry_prompt), VOICE)
            except Exception as e:
                print(f"⚠️ GPT error fallback: {e}")
                gather.say(retry_prompt, VOICE)

            resp.append(gather)
            return str(resp)

        # ✅ Step 4: Proceed with matched doctor
        session_data[call_sid]["cancel"]["doctor"] = googleid_dr_name_map[matched_id]
        session_data[call_sid]["doctor_id"] = matched_id
        session_data[call_sid]["stage"] = "cancel_appt_by_phone_number"

        gather = Gather(
            input="speech",
            action="/voice",
            method="POST",
            speech_model="phone_call",
            bargeIn=True,
            timeout=SPEECH_INPUT_DURATION
        )
        gather.say(gpt_speak("Thanks. What phone number did you use when booking the appointment?"), VOICE)
        resp.append(gather)
        return str(resp)


    elif stage == "cancel_appt_by_phone_number":
            # 📞 Step 1: Extract phone number
            phone = extract_phone_number(speech_result)
            print(f"📱 Extracted phone → {phone}")

            if not phone or len(phone) < 7:
                # ❗ Not valid → ask again
                gather = Gather(
                    input="speech",
                    action="/voice",
                    method="POST",
                    timeout=SPEECH_INPUT_DURATION,
                    speech_model="phone_call",
                    bargeIn=True
                )
                gather.say(gpt_speak("I didn’t catch your phone number. Please say it again clearly."), VOICE)
                resp.append(gather)
                return str(resp)

            session_data[call_sid]["cancel"]["phone"] = phone
            session_data[call_sid]["stage"] = "cancel_appt_get_date_time"

            # 🗣️ Ask for date and time
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=SPEECH_INPUT_DURATION,
                speech_model="phone_call",
                bargeIn=True
            )
            gather.say(gpt_speak("Thanks. Now, please tell me the date and time of the appointment you want to cancel. For example, say July 3rd at 9 AM."), VOICE)
            resp.append(gather)
            return str(resp)






    elif stage == "cancel_appt_get_date_time":
        debug_print("cancel_appt_get_date_time: 📍 Stage entered")

        # 🧠 Step 1: Parse speech
        debug_print(f"cancel_appt_get_date_time: 🗣️ Raw speech input → '{speech_result}'")
        time_info = smart_parse_time(speech_result)
        debug_print(f"cancel_appt_get_date_time: 🧠 smart_parse_time() returned → {time_info}")

        if not time_info or not isinstance(time_info, tuple) or len(time_info) != 2:
            session_data[call_sid]["retry_time"] = session_data[call_sid].get("retry_time", 0) + 1
            debug_print(f"cancel_appt_get_date_time: ❌ Failed to parse time. Retry count → {session_data[call_sid]['retry_time']}")
            if session_data[call_sid]["retry_time"] >= 3:
                debug_print("cancel_appt_get_date_time: ⛔ Max retries reached. Ending call.")
                resp.say(gpt_speak("Sorry, I couldn't understand the time. Please try again later. Goodbye."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=SPEECH_INPUT_DURATION,
                speech_model="phone_call",
                bargeIn=True
            )
            gather.say(gpt_speak("I didn’t catch that. Please say the date and time of the appointment you want to cancel, like July 29th at 8:30 AM."), VOICE)
            resp.append(gather)
            return str(resp)

        # ✅ Parsed
        spoken_day, spoken_time = time_info
        debug_print(f"cancel_appt_get_date_time: 📆 Parsed → Day: {spoken_day}, Time: {spoken_time}")

        try:
            # 🌍 Convert to UTC window
            utc_start, utc_end = build_timeslot_range(spoken_day, spoken_time)
            debug_print(f"cancel_appt_get_date_time: ✅ UTC range → Start: {utc_start}, End: {utc_end}")

            # Save for next stage
            session_data[call_sid]["cancel"]["day"] = spoken_day
            session_data[call_sid]["cancel"]["time"] = spoken_time
            session_data[call_sid]["cancel"]["utc_start"] = utc_start
            session_data[call_sid]["cancel"]["utc_end"] = utc_end

            # 📅 Resolve doctor calendar ID
            calendar_id = session_data[call_sid]["cancel"].get("calendar_id")
            if not calendar_id:
                doctor = session_data[call_sid]["cancel"].get("doctor", "")
                for doc_id, friendly in googleid_dr_name_map.items():
                    if friendly.lower() == doctor.lower():
                        calendar_id = doc_id
                        session_data[call_sid]["cancel"]["calendar_id"] = calendar_id
                        break

            if not calendar_id:
                debug_print("cancel_appt_get_date_time: ❌ Doctor calendar ID not found.")
                resp.say(gpt_speak("Sorry, I couldn't find the doctor's calendar. Please try again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            debug_print(f"cancel_appt_get_date_time: 📅 Doctor calendar ID → {calendar_id}")

            # 🔎 Optional prefetch: try to find the event now (won't block flow if None)
            phone = session_data[call_sid]["cancel"].get("phone")
            try:
                matching_event = get_upcoming_events(calendar_id, phone, utc_start, utc_end, creds, debug=True)
            except Exception as e:
                debug_print(f"cancel_appt_get_date_time: ❌ get_upcoming_events raised → {e}")
                matching_event = None

            session_data[call_sid]["cancel"]["matching_event"] = matching_event
            if matching_event:
                debug_print("cancel_appt_get_date_time: ✅ Matching event found and stored")
            else:
                debug_print("cancel_appt_get_date_time: 🚫 No matching event found at this step")

            # ▶️ Next stage
            session_data[call_sid]["stage"] = "cancel_appt_confirm"
            debug_print("cancel_appt_get_date_time: ✅ Session updated, moving to cancel_appt_confirm")
            return voice()

        except Exception as e:
            debug_print(f"cancel_appt_get_date_time: ❌ Failed to convert time or search calendar → {e}")
            resp.say(gpt_speak("Sorry, I couldn't process the date and time correctly."), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)











    elif stage == "cancel_appt_confirm":
        debug_print("📍 Stage: cancel_appt_confirm")

        phone       = session_data[call_sid]["cancel"].get("phone")
        doctor      = session_data[call_sid]["cancel"].get("doctor")
        spoken_day  = session_data[call_sid]["cancel"].get("day")
        spoken_time = session_data[call_sid]["cancel"].get("time")
        utc_start   = session_data[call_sid]["cancel"].get("utc_start")
        utc_end     = session_data[call_sid]["cancel"].get("utc_end")
        calendar_id = session_data[call_sid]["cancel"].get("calendar_id")

        debug_print(f"📱 Phone: {phone}")
        debug_print(f"👨‍⚕️ Doctor: {doctor}")
        debug_print(f"🗓️ Day/Time: {spoken_day}, {spoken_time}")
        debug_print(f"🌍 UTC window: {utc_start} → {utc_end}")
        debug_print(f"📅 Calendar ID: {calendar_id}")

        if not calendar_id:
            resp.say(gpt_speak("Sorry, I couldn't find the doctor's calendar. Please try again later."), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)

        # Use pre-fetched match or search now
        event_to_cancel = session_data[call_sid]["cancel"].get("matching_event")
        if not event_to_cancel:
            event_to_cancel = get_upcoming_events(calendar_id, phone, utc_start, utc_end, creds, debug=True)

        if event_to_cancel:
            event_id = event_to_cancel.get("id")
            try:
                service = build("calendar", "v3", credentials=creds)
                service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
                debug_print(f"🗑️ Deleted calendar event id={event_id}")

                # Friendly time for TTS
                from dateutil import parser as dtparser
                try:
                    start_str = event_to_cancel.get("start", {}).get("dateTime", "") or event_to_cancel.get("start", {}).get("date", "")
                    dt = dtparser.parse(start_str)
                    import pytz
                    friendly = dt.astimezone(pytz.timezone("America/Chicago")).strftime("%B %-d at %-I:%M %p")
                except Exception:
                    friendly = f"{spoken_day} at {spoken_time}"

                # Remove from JSON mapping
                try:
                    removed = cancel_appointment_by_name(doctor, phone, utc_start)
                    if removed:
                        debug_print(f"🧹 Removed mapping from {doctor}.json")
                    else:
                        debug_print(f"⚠️ No JSON mapping found to remove for the exact UTC time.")
                except Exception as e:
                    debug_print(f"⚠️ JSON remove error → {e}")

                resp.say(gpt_speak(f"Your appointment with {doctor} on {friendly} has been cancelled. Thank you!"), VOICE)

            except Exception as e:
                debug_print(f"❌ Calendar delete failed → {e}")
                resp.say(gpt_speak("Sorry, something went wrong while cancelling your appointment."), VOICE)
        else:
            debug_print("🚫 No matching appointment found to cancel.")
            resp.say(gpt_speak("I'm sorry, I couldn't find an appointment under that phone number and time."), VOICE)

        # End or reschedule
        if session_data[call_sid].get("reschedule_after_cancel"):
            debug_print("🔁 Reschedule requested; transitioning to booking.")
            session_data[call_sid]["stage"] = "booking"
            session_data[call_sid].pop("cancel", None)
            doctor_list_str = ", ".join(googleid_dr_name_map.values())
            gather = Gather(input="speech", action="/voice", method="POST",
                            speech_model="phone_call", bargeIn=True,
                            timeout=SPEECH_INPUT_DURATION, hints=doctor_list_str)
            gather.say(gpt_speak("Now, please tell me which doctor you'd like to reschedule with."), VOICE)
            resp.append(gather)
            return str(resp)

        session_data.pop(call_sid, None)
        debug_print("🧼 Session data cleared after cancellation.")
        return str(resp)





      
    


if __name__ == "__main__":
    
    # ------------------------
    # 🔁 Call this once at startup
    # ------------------------

    load_doctor_appointments()


    app.run(host="0.0.0.0", port=5000)
