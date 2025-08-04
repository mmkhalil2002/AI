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

from datetime import datetime, timedelta
from typing import List, Dict

def get_next_available_slots(calendar_id: str, creds, limit: int = 3, duration_minutes: int = 30) -> List[Dict]:
    """
    Return a list of the next `limit` available time slots for a doctor.
    Each slot is a dictionary with 'start', 'end', and 'friendly' keys.
    """
    from googleapiclient.discovery import build
    import pytz

    service = build("calendar", "v3", credentials=creds)
    now = datetime.utcnow().replace(second=0, microsecond=0)
    tz = pytz.timezone("America/Chicago")
    now_local = now.astimezone(tz)

    # 🔁 Start from the next rounded-up 30-minute boundary
    minute = (now_local.minute // duration_minutes + 1) * duration_minutes
    rounded_start = now_local.replace(minute=0) + timedelta(minutes=minute)
    if rounded_start.minute >= 60:
        rounded_start += timedelta(hours=1)
        rounded_start = rounded_start.replace(minute=0)

    suggestions = []
    checked_slots = 0
    MAX_LOOKAHEAD_HOURS = 72  # how far to search into the future

    while len(suggestions) < limit and checked_slots < (MAX_LOOKAHEAD_HOURS * 60) // duration_minutes:
        end_slot = rounded_start + timedelta(minutes=duration_minutes)
        time_min = rounded_start.isoformat()
        time_max = end_slot.isoformat()

        events = service.events().list(
            calendarId=calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True
        ).execute()

        if not events["items"]:
            suggestions.append({
                "start": time_min,
                "end": time_max,
                "friendly": rounded_start.strftime("%B %-d at %-I:%M %p")
            })

        rounded_start += timedelta(minutes=duration_minutes)
        checked_slots += 1

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

import re

def normalize_date_time(spoken_day: str, spoken_time: str) -> str:
    """
    Normalize spoken input like "3rd of July" → "July 3"
    Cleans leading noise (e.g., "Is"), removes ordinal suffixes,
    and rearranges day/month if needed.
    """

    # 🧹 Remove filler/junk at the start of the input (e.g., "is", "it's", "on", etc.)
    spoken_day = re.sub(r"^(is|it's|on|at|for)\s+", "", spoken_day.strip(), flags=re.IGNORECASE)
    spoken_time = re.sub(r"^(is|it's|on|at|for)\s+", "", spoken_time.strip(), flags=re.IGNORECASE)

    # 🔢 Remove ordinal suffixes (e.g., 1st → 1, 2nd → 2, 3rd → 3)
    day = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', spoken_day, flags=re.IGNORECASE)

    # 🔁 Convert "3 of July" → "July 3"
    match = re.match(r"(\d+)\s+of\s+([A-Za-z]+)", day, flags=re.IGNORECASE)
    if match:
        day = f"{match.group(2)} {match.group(1)}"

    # 🧽 Remove commas and remaining 'of'
    day = day.replace(",", "").replace("of", "").strip()

    # ✅ Combine normalized day and time
    return f"{day} {spoken_time}".strip()



from datetime import datetime, timedelta
from typing import Tuple

from datetime import datetime, timedelta, timezone
from typing import Tuple

def build_timeslot_range(spoken_day: str, spoken_time: str) -> Tuple[str, str]:
    """
    Converts spoken day/time into ISO 8601 datetime range (30 minutes).
    Accepts various formats like:
        - "July 3", "8:30 AM"
        - "3rd of July", "8:30"
        - "Thursday, July 3", "8:30"
    """
    combined = normalize_date_time(spoken_day, spoken_time)

    formats = [
        "%B %d %I:%M %p",
        "%B %d %H:%M",
        "%A %B %d %I:%M %p",
        "%A %B %d %H:%M",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(combined, fmt)
            dt = dt.replace(year=datetime.now().year)  # Ensure year is set
            dt = dt.replace(tzinfo=timezone.utc)       # 👈 Add UTC timezone
            break
        except ValueError:
            dt = None

    if not dt:
        raise ValueError(f"Failed to parse time from: '{combined}'")

    start = dt.isoformat()          # RFC3339 format with timezone
    end = (dt + timedelta(minutes=30)).isoformat()

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

def extract_phone_number(speech_text):
    """
    Extracts a phone number from the given speech text.
    
    This function is useful when a user speaks their phone number during a voice call.
    It searches for digit sequences that resemble phone numbers (7 to 11 digits),
    optionally separated by spaces or dashes.

    Parameters:
        speech_text (str): The transcribed text spoken by the user.

    Returns:
        str: The extracted phone number as a clean string of digits.
             Returns an empty string if no valid phone number is found.
    """

    # 🔍 Use regular expression to find a pattern of 7 to 11 digits that may be separated by spaces or hyphens.
    # Example matches: "123 456 7890", "123-456-7890", "1234567890"
    match = re.search(r'\b(?:\d[\s\-]?){7,11}\b', speech_text)

    # ✅ If a match is found:
    if match:
        # 🧼 Remove all spaces and hyphens to return a clean digit-only phone number
        return match.group().replace(" ", "").replace("-", "")

    # ❌ If no match found, return an empty string
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
    return_details: bool = False  # ✅ Whether to return full event details
):
    """
    Cancel (delete) a Google Calendar event that matches a given phone number,
    and optionally a specific spoken day and/or time.

    Parameters:
    - calendar_id (str): ID of the Google Calendar to query.
    - phone (str): Phone number to match in event summary or description.
    - spoken_day (Optional[str]): Natural language string like "Monday" or "July 3".
    - spoken_time (Optional[str]): Time string like "9:00 AM".
    - creds: Google OAuth2 credentials.
    - return_details (bool): If True, return full event object instead of just success/failure.

    Returns:
    - dict: Matching event object if return_details is True and a match was found.
    - True: If deletion was successful (and return_details is False).
    - False/None: If no match was found or deletion failed.
    """

    from googleapiclient.discovery import build
    from datetime import datetime
    import pytz
    import re

    # Build Google Calendar service client
    service = build("calendar", "v3", credentials=creds)

    # Start from current time
    now = datetime.utcnow().isoformat() + 'Z'

    # Fetch upcoming events (limit 25 for performance)
    events_result = service.events().list(
        calendarId=calendar_id,
        timeMin=now,
        maxResults=25,
        singleEvents=True,
        orderBy="startTime"
    ).execute()

    events = events_result.get("items", [])

    for event in events:
        # Extract phone number from summary or description
        summary = event.get("summary", "").lower()
        description = event.get("description", "").lower()

        if phone in summary or phone in description:
            # ✅ Found an event with matching phone number

            event_start = event.get("start", {}).get("dateTime")
            if not event_start:
                continue  # Skip all-day or malformed events

            try:
                # Parse the event datetime
                dt = datetime.fromisoformat(event_start.replace("Z", "+00:00"))

                # Normalize and check day match
                dt_day_str = dt.strftime("%A, %B %-d").lower()  # e.g., "Thursday, July 3"
                spoken_day_clean = (spoken_day or "").lower().strip()
                day_match = not spoken_day or spoken_day_clean in dt_day_str

                # Normalize and check time match
                dt_time_str = dt.strftime("%-I:%M %p").lower()  # e.g., "9:00 am"
                spoken_time_clean = (spoken_time or "").lower().strip()
                time_match = not spoken_time or spoken_time_clean in dt_time_str

                if day_match and time_match:
                    # 🎯 Matching event → Delete it
                    service.events().delete(calendarId=calendar_id, eventId=event["id"]).execute()

                    return event if return_details else True

            except Exception as e:
                print(f"⚠️ Error parsing event datetime: {e}")
                continue

    # ❌ No match found
    return None if return_details else False



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