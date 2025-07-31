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





from datetime import datetime, timedelta, timezone
from typing import Tuple

from datetime import datetime, timedelta
from typing import Tuple
import re

def build_timeslot_range(spoken_day: str, spoken_time: str) -> Tuple[str, str]:
    """
    Given a spoken day and time, return a UTC ISO 8601 start/end range.
    Ensures correct conversion from local (America/Chicago) to UTC.
    """
    from dateutil import parser
    import pytz

    combined = normalize_date_time(spoken_day, spoken_time)
    formats = [
        "%B %d %I:%M %p",     # July 3 8:30 AM
        "%B %d %H:%M",        # July 3 08:30
        "%A %B %d %I:%M %p",  # Thursday July 3 8:30 AM
        "%A %B %d %H:%M",     # Thursday July 3 08:30
    ]

    dt = None
    for fmt in formats:
        try:
            dt = datetime.strptime(combined, fmt)
            print(f"✅ Parsed datetime: {dt} using format {fmt}")
            break
        except ValueError:
            continue

    if not dt:
        raise ValueError(f"🛑 Could not parse datetime from: '{combined}'")

    # Get current year to avoid 1900 issue
    now = datetime.now()
    dt = dt.replace(year=now.year)
    print(f"📅 Inferred year → Updated datetime: {dt}")

    tz = pytz.timezone("America/Chicago")

    # Convert to timezone-aware datetime safely
    if dt.tzinfo is None:
        dt_local = tz.localize(dt)
    else:
        dt_local = dt.astimezone(tz)

    dt_utc = dt_local.astimezone(pytz.UTC)
    start = dt_utc.isoformat()
    end = (dt_utc + timedelta(minutes=30)).isoformat()

    print(f"📅 Local time slot → Start: {dt_local}, End: {dt_local + timedelta(minutes=30)}")
    print(f"🌍 UTC time slot → Start: {start}, End: {end}")

    return start, end









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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)




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
client = OpenAI()

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
def cancel_event_by_phone(
    calendar_id: str,
    phone: str,
    spoken_day: Optional[str] = None,
    spoken_time: Optional[str] = None,
    creds=None,
    return_details: bool = False
):
    """
    Cancel (delete) a Google Calendar event by matching phone number and optional spoken date/time.

    Returns:
        - The matching event (dict) if return_details is True
        - True on success
        - False / None if not found
    """
    from googleapiclient.discovery import build
    from datetime import datetime
    import pytz
    import re

    # 📞 Normalize phone number (remove non-digit characters)
    clean_phone = re.sub(r"[^\d]", "", phone)
    print(f"🔍 Searching for normalized phone: {clean_phone}")

    # 🗓️ Parse expected spoken datetime to full datetime object
    parsed_datetime = None
    if spoken_day and spoken_time:
        try:
            #from utils.time_tools import build_timeslot_range
            start_iso, _ = build_timeslot_range(spoken_day, spoken_time)
            parsed_datetime = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
            print(f"🧠 Parsed target datetime: {parsed_datetime.isoformat()}")
        except Exception as e:
            print(f"⚠️ Failed to parse spoken datetime → {spoken_day}, {spoken_time}: {e}")

    # 🔧 Setup Google Calendar API
    service = build("calendar", "v3", credentials=creds)
    now = datetime.utcnow().isoformat() + 'Z'

    # 🔍 Search 25 upcoming events
    events_result = service.events().list(
        calendarId=calendar_id,
        timeMin=now,
        maxResults=25,
        singleEvents=True,
        orderBy="startTime"
    ).execute()

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

            event_start = event.get("start", {}).get("dateTime")
            if not event_start:
                print("⚠️ Skipping all-day or malformed event.")
                continue

            try:
                event_dt = datetime.fromisoformat(event_start.replace("Z", "+00:00"))

                if parsed_datetime:
                    # Compare with ±1 minute tolerance
                    delta = abs((event_dt - parsed_datetime).total_seconds())
                    print(f"🕐 Comparing event start {event_dt} to target {parsed_datetime}, Δ={delta}s")

                    if delta <= 90:
                        print("🗑️ Found matching event. Deleting...")
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
       prompt = "would you like  to book an appointment, cancel an appointment, reschedulle an appointment  or leave a message."

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
            gather.say(gpt_speak("Please tell me if you'd like to book an appointment, cancel one, reschedule, or leave a message."))
            resp.append(gather)
            return str(resp)

        # ✅ Rescheduling intent
        elif any(word in lower for word in ["reschedule", "change", "move"]):
            print("🔁 Intent to reschedule detected → will cancel then rebook")
            session_data[call_sid] = {
                "stage": "cancel_appointment",
                "cancel": {},
                "retry_booking": 0,
                "reschedule_after_cancel": True
            }
            doctor_names = list(googleid_dr_name_map.values())
            doctor_list = ", ".join(doctor_names[:-1]) + ", or " + doctor_names[-1] if len(doctor_names) > 1 else doctor_names[0]
            prompt = (
                f"Sure, let's reschedule your appointment. First, we'll cancel your current appointment. "
                f"Available doctors include: {doctor_list}. Please say the name of the doctor you had booked with."
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
        elif any(word in lower for word in ["book", "booking", "schedule", "make", "reserve", "meet"]):
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
                            speech_model="phone_call",  # 🎯 Optimized for voice calls
                            bargeIn=True, 
                            timeout=SPEECH_INPUT_DURATION,
                            hints=doctor_list_str
                        )
            gather.say(gpt_speak("Please say the name of the doctor you'd like to book with."),VOICE)
            resp.append(gather)
            return str(resp)

        matched_id = None

        # ------------------------------------------------------------------
        # 🔍 1. Partial or exact substring match
        # ------------------------------------------------------------------
        partial_matches = []
        for doc_id, friendly in googleid_dr_name_map.items():
            friendly_clean = friendly.lower().translate(str.maketrans('', '', string.punctuation)).strip()
            if spoken_clean in friendly_clean or friendly_clean in spoken_clean:
                partial_matches.append((doc_id, friendly))

        if len(partial_matches) == 1:
            matched_id = partial_matches[0][0]
            print(f"✅ Partial match with: {partial_matches[0][1]}")

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
                ),VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            doctor_list_str = ", ".join(googleid_dr_name_map.values())
            gather = Gather(
                                input="speech",
                                action="/voice",
                                method="POST",
                                speech_model="phone_call",  # 🎯 Optimized for voice calls
                                bargeIn=True, 
                                timeout=SPEECH_INPUT_DURATION,
                                hints=doctor_list_str
                        )
            retry_prompt = (
                f"I couldn't match that to a doctor. Available doctors are: {doctor_list_str}. "
                "Please say the doctor name again."
            )
            gather.say(gpt_speak(retry_prompt),VOICE)
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
                         speech_model="phone_call",  # 🎯 Optimized for voice calls
                         bargeIn=True, 
                         timeout=SPEECH_INPUT_DURATION
                      )
        gather.say(gpt_speak(time_prompt),VOICE)
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
        print(f"📬 collect_address: Collected address: {address}")

        session_data[call_sid]["customer"]["address"] = address
        session_data[call_sid]["stage"] = "confirmed"  # ✅ next stage

        customer = session_data[call_sid]["customer"]
        appointment = session_data[call_sid]["appointment_time"]
        doctor_id = session_data[call_sid]["doctor_id"]

        # 🧩 Assemble full name safely
        full_name = f"{customer.get('first_name', '')} {customer.get('last_name', '')}".strip()
        session_data[call_sid]["customer"]["name"] = full_name  # for use in SMS & confirmation

        # 📞 Normalize phone number
        import re
        raw_phone = customer.get("phone", "")
        normalized_phone = re.sub(r"[^\d]", "", raw_phone)
        session_data[call_sid]["customer"]["phone"] = normalized_phone
        print(f"📞 Final stored phone: {normalized_phone} (digits only expected)")

        # 📅 Handle time conversion: appointment time is in UTC
        try:
            from datetime import datetime
            import pytz

            local_tz = pytz.timezone("America/Chicago")
            start_utc = datetime.fromisoformat(appointment["start"]).astimezone(pytz.utc)
            end_utc = datetime.fromisoformat(appointment["end"]).astimezone(pytz.utc)

            print(f"📅 Local time slot → Start: {start_utc.isoformat()}, End: {end_utc.isoformat()}")
        except Exception as e:
            print(f"❌ Error converting to UTC: {e}")
            resp.say(gpt_speak("Sorry, I had trouble confirming your appointment. Please try again later."), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)

        # 📅 Build Google Calendar API service and create the appointment
        try:
            calendar = build("calendar", "v3", credentials=creds)
            event = {
                "summary": f"Appointment for {full_name}",
                "description": f"Name: {full_name}\nPhone: {normalized_phone}\nAddress: {address}",
                "start": {"dateTime": start_utc.isoformat(), "timeZone": "UTC"},
                "end": {"dateTime": end_utc.isoformat(), "timeZone": "UTC"},
            }

            calendar.events().insert(calendarId=doctor_id, body=event).execute()
            print("✅ Google Calendar event created")
        except Exception as e:
            print(f"❌ Failed to create calendar event: {e}")
            resp.say(gpt_speak("Sorry, something went wrong while saving your appointment. Please try again later."), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)

        # 📣 Success — proceed to confirmation
        resp.say(gpt_speak(f"Thanks {full_name}. Your appointment has been confirmed. Goodbye!"), VOICE)
        return str(resp)


    
    elif stage == "confirmed":
        # ------------------------------------------------------------
        # 📍 Final confirmation stage after booking is complete
        # ------------------------------------------------------------
        print("📍 Stage: confirmed")

        # 🆔 Doctor info
        doctor_id = session_data[call_sid].get("doctor_id")
        doctor_name = googleid_dr_name_map.get(doctor_id, "the doctor")
        print(f"👨‍⚕️ Doctor: {doctor_name} (ID: {doctor_id})")

        # 🕐 Appointment info
        appointment_time = session_data[call_sid].get("appointment_time", {}).get("start")
        formatted_time = ""
        if appointment_time:
            from datetime import datetime
            import pytz
            try:
                # Convert UTC → Central Time (America/Chicago)
                dt_utc = datetime.fromisoformat(appointment_time.replace("Z", "+00:00"))
                tz = pytz.timezone("America/Chicago")
                dt_local = dt_utc.astimezone(tz)
                formatted_time = dt_local.strftime("%A, %B %-d at %-I:%M %p")
                print(f"📆 Appointment (local): {formatted_time}")
            except Exception as e:
                print(f"⚠️ Failed to parse appointment time: {e}")

        # 🧍 Customer info
        customer = session_data[call_sid].get("customer", {})
        customer_name = customer.get("name", "")
        customer_phone = customer.get("phone")
        print(f"👤 Customer: {customer_name}, Phone: {customer_phone}")

        # 🗣️ Voice confirmation message
        confirmation_message = (
            f"Your appointment with {doctor_name} has been successfully booked."
            " We look forward to seeing you. Goodbye!"
        )
        resp.say(gpt_speak(confirmation_message), VOICE)

        # ------------------------------------------------------------
        # 📩 Send SMS confirmation
        # ------------------------------------------------------------
        if customer_phone:
            sms_text = f"Hi {customer_name}, your appointment with {doctor_name} is confirmed"
            if formatted_time:
                sms_text += f" on {formatted_time}"
            sms_text += ". Thank you for choosing Epic Therapist Clinic."

            try:
                message = client.messages.create(
                    body=sms_text,
                    from_=TWILIO_PHONE_NUMBER,
                    to=customer_phone
                )
                print("📤 SMS sent to:", customer_phone)
                print("🧾 SID:", message.sid, "| Status:", message.status)
            except Exception as e:
                print(f"❌ SMS send failed: {e}")
        else:
            print("⚠️ No phone number available — skipping SMS.")

        # 📞 End the call
        resp.hangup()

        # 🧹 Clear session
        session_data.pop(call_sid, None)
        print("🧼 Session data cleared")

        return str(resp)






    
    elif stage == "cancel_appointment":
            # ----------------------------------------------------------------------
        # 🔄 This stage handles when a user wants to cancel an appointment
        # and just spoke the doctor’s name (e.g., "Dr. Omar", or "cancel with Dr. Alex")
        # ----------------------------------------------------------------------

        selected_name = speech_result.lower()  # 🎤 Raw speech input from caller (e.g., "cancel with Dr. Omar")
        matched_id = None                      # 🧠 Variable to store matching calendar ID

        # 🔍 Step 1: Try basic match — loop through known doctors and check if the spoken input contains a known name
        for doc_id, friendly_name in googleid_dr_name_map.items():
            if friendly_name.lower() in selected_name:
                matched_id = doc_id
                break

        # 🤖 Step 2: If no match found, call GPT to extract doctor name intelligently
        if not matched_id:
            extracted_name = extract_doctor_name(speech_result)  # 🧠 GPT will try to extract just the doctor name
            if extracted_name:
                for doc_id, friendly_name in googleid_dr_name_map.items():
                    if friendly_name.lower() == extracted_name.lower():
                        matched_id = doc_id
                        break

        # ❌ Step 3: Still no match → handle retries
        if not matched_id:
            # Safely retrieve or initialize retry counter
            retries = session_data[call_sid].get("retry_booking", 0)
            session_data[call_sid]["retry_booking"] = retries + 1

            if retries >= MAX_NUMBER_DR_RETRY:
                # 🛑 Too many failed attempts → end call
                resp.say(gpt_speak(
                    "I'm sorry, I still couldn't match that name with any doctor in our clinic. Please try again later. Goodbye."
                ), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)  # 🧹 Clean up session data
                return str(resp)

            # 🔁 Less than 3 attempts → re-ask with full doctor list
            gather = Gather(
                              input="speech", 
                              action="/voice",
                              method="POST",
                              speech_model="phone_call",
                              bargeIn=True, 
                              timeout=SPEECH_INPUT_DURATION
                            )

            # 🧾 Convert the list of available doctors to a comma-separated string
            doctor_list = ", ".join(googleid_dr_name_map.values())
            retry_prompt = (
                f"I didn't recognize that name. Available doctors are: {doctor_list}. "
                "Please say the name again."
            )

            try:
                gather.say(gpt_speak(retry_prompt), VOICE)
            except Exception as e:
                print(f"⚠️ GPT error in retry_prompt fallback: {e}")
                gather.say(retry_prompt, VOICE)

            resp.append(gather)
            return str(resp)

        # ✅ Step 4: Doctor matched — store info and ask for phone number
        session_data[call_sid]["cancel"]["doctor"] = googleid_dr_name_map[matched_id]  # Save friendly name
        session_data[call_sid]["doctor_id"] = matched_id                                # Save Google Calendar ID
        session_data[call_sid]["stage"] = "cancel_appt_by_phone_number"                 # Advance to phone entry

        # 📞 Prompt user for the phone number they used when booking
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
            session_data[call_sid]["stage"] = "cancel_appt_get_date"

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


    elif stage == "cancel_appt_get_date":
        # 🧠 Step 1: Try extracting spoken date and time
        print(f"🗣️ Received for cancellation date/time: {speech_result}")
        time_info = smart_parse_time(speech_result)

        if not time_info or not isinstance(time_info, tuple) or len(time_info) != 2:
            session_data[call_sid]["retry_time"] = session_data[call_sid].get("retry_time", 0) + 1
            if session_data[call_sid]["retry_time"] >= 3:
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

        # ✅ Successfully parsed
        spoken_day, spoken_time = time_info
        print(f"📆 Extracted → Day: {spoken_day}, Time: {spoken_time}")
        session_data[call_sid]["cancel"]["day"] = spoken_day
        session_data[call_sid]["cancel"]["time"] = spoken_time

        phone = session_data[call_sid]["cancel"].get("phone")
        doctor = session_data[call_sid]["cancel"].get("doctor")
        print(f"📱 Using phone → {phone}")
        print(f"👨‍⚕️ Using doctor → {doctor}")

        # 🔍 Match doctor to calendar ID
        calendar_id = None
        for doc_id, friendly in googleid_dr_name_map.items():
            if friendly.lower() == doctor.lower():
                calendar_id = doc_id
                break

        if not calendar_id:
            resp.say(gpt_speak("Sorry, I couldn't find the doctor in our system. Please try again later."), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)

        # 🔁 Attempt to cancel the event
        canceled_event = cancel_event_by_phone(
            calendar_id=calendar_id,
            phone=phone,
            spoken_day=spoken_day,
            spoken_time=spoken_time,
            creds=creds,
            return_details=True
        )

        if canceled_event:
            from dateutil import parser
            try:
                start_str = canceled_event.get("start", {}).get("dateTime", "") or canceled_event.get("start", {}).get("date", "")
                dt = parser.parse(start_str)
                spoken_time_str = dt.strftime("%B %-d at %-I:%M %p")
            except Exception as e:
                print(f"⚠️ Failed to format date for confirmation: {e}")
                spoken_time_str = f"{spoken_day} at {spoken_time}"

            msg = f"Your appointment with {doctor} on {spoken_time_str} has been cancelled. Thank you!"
            resp.say(gpt_speak(msg), VOICE)
        else:
            resp.say(gpt_speak("I'm sorry, I couldn't find an appointment under that phone number and time."), VOICE)

        session_data.pop(call_sid, None)
        return str(resp)


      
    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
