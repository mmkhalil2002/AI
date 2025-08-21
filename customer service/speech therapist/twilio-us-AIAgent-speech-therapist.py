# update  07/22/25 2:45 pm
import os
import json
#import openai
import pickle
import dateparser
import  calendar
import  re
import openai
import pytz
import os
import json
import re as re_mod
import pytz as _pytz
import string  # <-- needed for string.punctuation


from dateutil.parser import isoparse  # for parsing RFC3339/ISO datetimes to extract dates
from datetime import datetime, date

from datetime import datetime, timedelta
from dotenv import load_dotenv
from datetime import timedelta
from datetime import datetime as _dt

from datetime import datetime, timedelta
from dateutil.tz import gettz
from datetime import datetime, timedelta, time
from datetime import datetime, timedelta
from typing import Tuple
from datetime import datetime, timedelta, date, time as dtime
from typing import Tuple, Union
from typing import Optional
from typing import Any, Dict
from dateutil import parser as dtparser
from datetime import datetime
from datetime import timedelta
from datetime import datetime
from datetime import timedelta, timezone
from datetime import datetime as _dt
from uuid import uuid4  # only used if CallSid is missing







# BEFORE:
# def _render_block_lines(new: bool, rec: dict) -> list[str]:
# AFTER (3.8-safe):
from typing import Any, Optional, List, Dict, Tuple, Iterator, Iterable

from googleapiclient.discovery import build
from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client


from openai import OpenAI, APIConnectionError, AuthenticationError, RateLimitError
from openai import OpenAIError  # Add this import at the top

from flask import request
from flask import url_for
from flask import Flask, request
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

MAX_GET_PHONE_RETRIES = int(os.getenv("MAX_GET_PHONE_RETRIES", 3))
WORKING_DAYS = [0, 1, 2, 3, 4]  # Adjust based on your local week

DB_FOLDER = "appointment_data"
DB_FILE   = os.path.join(DB_FOLDER, "customers.json")  # human-readable, not JSON
# Global working config
WORKING_DAYS = [0, 2, 3, 4]  # Mon=0, Tue=1,... Friday=4
WORKING_HOURS_START = 8  # 8:00 AM
WORKING_HOURS_END = 17   # 5:00 PM
LUNCH_BREAK_START = time(13, 0)  # 1:00 PM
LUNCH_BREAK_END = time(14, 0)    # 2:00 PM

USE_GPT = False
DEBUG  = True

if USE_GPT:
    

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
   

    client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    print("📞 Using Twilio client")

## print debug


def debug_print(msg: str) -> None:
    """Your app already defines this; keep here for completeness if not imported."""
    try:
        print(msg)
    except Exception:
        pass



# --- retry counter utils ---


def make_gather(
    prompt: str,
    *,
    next_stage: str = None,        # ← NEW (optional)
    hints: str = None,
    input: str = "speech dtmf",
    num_digits: int = None,
    timeout: int = 6,
    speech_timeout: str = "auto",
    finish_on_key: str = "#",
    barge_in: bool = True,
    language: str = "en-US",
    action: str = None
):
    """
    Build a Twilio <Gather> with sane defaults.
    If next_stage is provided, it will be appended to the action URL as
    '?next_stage=...' so the /voice handler can switch stages on the next POST.
    """
    try:
        from flask import request, url_for
        base_action = action or request.path or "/voice"
        if next_stage:
            sep = "&" if "?" in base_action else "?"
            base_action = f"{base_action}{sep}next_stage={next_stage}"
            try:
                debug_print(f"make_gather: ↪️ action with next_stage → {base_action}")
            except Exception:
                pass
    except Exception:
        base_action = action or "/voice"

    g = Gather(
        input=input,
        action=base_action,
        method="POST",
        timeout=timeout,
        speech_timeout=speech_timeout,
        hints=hints,
        language=language,
        finish_on_key=finish_on_key,
        barge_in=barge_in,
        num_digits=num_digits,
    )
    # Keep your TTS wrapper and voice selection
    try:
        g.say(gpt_speak(prompt), VOICE)
    except Exception:
        g.say(prompt)
    return g







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


def is_time_slot_available(calendar_id: str, start_time: str, end_time: str, creds) -> bool:
    """
    Check if the given time range is free on the doctor's calendar.
    Returns True if available, False if already booked.

    Logs (via debug_print):
      - the calendar and window being checked
      - how many events were returned
      - for each event: (one line) input_start, event_start, input_date, event_date
      - final verdict (FREE/BUSY)
    """
    

    debug_print(
        f"is_time_slot_available: 🔎 cal='{calendar_id}' "
        f"window={start_time}→{end_time}"
    )

    # Pre-parse input window once so we can also log the date-only view
    try:
        _in_start_dt = isoparse(start_time)
        _in_start_date = _in_start_dt.date().isoformat()
    except Exception:
        _in_start_dt = None
        _in_start_date = "?"

    try:
        _in_end_dt = isoparse(end_time)
        _in_end_date = _in_end_dt.date().isoformat()
    except Exception:
        _in_end_dt = None
        _in_end_date = "?"

    try:
        service = build("calendar", "v3", credentials=creds)
        events_result = service.events().list(
            calendarId=calendar_id,
            timeMin=start_time,          # RFC3339 / ISO 8601
            timeMax=end_time,            # RFC3339 / ISO 8601
            singleEvents=True,
            showDeleted=False,
            orderBy="startTime",
            maxResults=250,
        ).execute()

        events = events_result.get("items", [])
        debug_print(f"is_time_slot_available: ℹ️ events count={len(events)}")

        # Print (one line) input_start, event_start, input_date, event_date for each event
        for ev in events:
            event_start_raw = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date")
            try:
                _ev_start_dt = isoparse(event_start_raw) if event_start_raw else None
                _ev_start_date = _ev_start_dt.date().isoformat() if _ev_start_dt else "?"
            except Exception:
                _ev_start_dt = None
                _ev_start_date = "?"

            debug_print(
                "is_time_slot_available: • "
                f"input_start={start_time} | "
                f"event_start={event_start_raw} | "
                f"input_date={_in_start_date} | "
                f"event_date={_ev_start_date}"
            )

        available = (len(events) == 0)
        debug_print(f"is_time_slot_available: {'✅ FREE' if available else '❌ BUSY'}")
        return available

    except Exception as e:
        debug_print(f"is_time_slot_available: ❗ error during events.list → {e}")
        # Safer to fail closed to avoid double-booking
        return False





def get_next_available_slots(
    calendar_id: str,
    creds,
    *,
    from_start_iso: str,
    duration_minutes: int = 30,
    limit: int = 3,
    tz_name: str = "America/Chicago",
    # Working hours per day (local time) — tuples of (start_hour, end_hour) in 24h.
    # You can make this a dict keyed by weekday if you need different hours per day.
    work_hours=( (8, 17), ),  # 8:00–17:00 local; multiple windows supported, e.g., ((8,12),(13,17))
    slot_step_minutes: int = 30,   # align suggestions to a 30-minute grid
    search_days: int = 14          # scan forward up to 14 days
) -> list:
    """
    Return a list of up to `limit` free slots after `from_start_iso` *for this doctor*.
    Each element: {"start": "<UTC-iso>", "end": "<UTC-iso>", "friendly": "Friday, August 15 at 8:30 AM"}

    Strategy:
      - Work in the doctor's LOCAL TZ for grid/working-hours alignment; convert to UTC for FreeBusy.
      - For each day in the search window:
          - Query FreeBusy once for the full working window(s).
          - Subtract busy intervals and scan the free timeline on a fixed grid (slot_step_minutes).
          - Return the first `limit` non-overlapping slots of `duration_minutes`.
    """
    service = build("calendar", "v3", credentials=creds)
    tz_local = gettz(tz_name)

    # ---------- helpers ----------
    def _round_up(dt: datetime, minutes: int) -> datetime:
        """Round dt up to the next multiple of `minutes`."""
        q = (dt.minute // minutes) * minutes
        base = dt.replace(minute=q, second=0, microsecond=0)
        if base < dt:
            base += timedelta(minutes=minutes)
        return base

    def _merge_intervals(intervals):
        """Merge overlapping (start_dt, end_dt) intervals (both timezone-aware)."""
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            ms, me = merged[-1]
            if s <= me:
                merged[-1] = (ms, max(me, e))
            else:
                merged.append((s, e))
        return merged

    def _friendly(local_start: datetime) -> str:
        """Readable label like 'Friday, August 15 at 8:30 AM' (use local tz)."""
        return local_start.strftime("%A, %B %#d at %-I:%M %p") if hasattr(local_start, "strftime") else \
               local_start.strftime("%A, %B %d at %I:%M %p")
        # Note: %#d / %-I flags vary by OS; the fallback still works cross-platform.

    # ---------- seed / context ----------
    start_utc = isoparse(from_start_iso)                # timezone-aware UTC
    start_local = start_utc.astimezone(tz_local)        # align to local grid/hours
    results = []

    # Scan forward day by day until we find `limit` slots or exhaust `search_days`
    for day_offset in range(search_days):
        day_local = (start_local if day_offset == 0 else
                     (start_local + timedelta(days=day_offset)).replace(hour=0, minute=0, second=0, microsecond=0))

        # Handle each working window (you can pass multiple windows, e.g., morning & afternoon)
        for wh_start_hour, wh_end_hour in work_hours:
            # Local working-window bounds for that day
            window_start_local = day_local.replace(hour=wh_start_hour, minute=0, second=0, microsecond=0)
            window_end_local   = day_local.replace(hour=wh_end_hour,   minute=0, second=0, microsecond=0)

            # If it's the first day and we're already past start of window, start from NOW (rounded)
            if day_offset == 0 and start_local > window_start_local:
                window_start_local = _round_up(start_local, slot_step_minutes)

            # Convert to UTC for FreeBusy query
            window_start_utc = window_start_local.astimezone(gettz("UTC"))
            window_end_utc   = window_end_local.astimezone(gettz("UTC"))
            if window_start_utc >= window_end_utc:
                continue  # skip invalid windows

            # ---- Query FreeBusy once for the whole window ----
            fb = service.freebusy().query(body={
                "timeMin": window_start_utc.isoformat(),
                "timeMax": window_end_utc.isoformat(),
                "items": [{"id": calendar_id}],
                "timeZone": "UTC",
            }).execute()

            busy_list = (fb.get("calendars", {}).get(calendar_id, {}) or {}).get("busy", []) or []

            # Convert busy intervals to LOCAL TZ for easier local grid checks
            busy_intervals_local = []
            for b in busy_list:
                try:
                    bs = isoparse(b["start"]).astimezone(tz_local)
                    be = isoparse(b["end"]).astimezone(tz_local)
                    # Keep only overlaps with the working window
                    if be > window_start_local and bs < window_end_local:
                        busy_intervals_local.append( (max(bs, window_start_local), min(be, window_end_local)) )
                except Exception:
                    continue
            busy_intervals_local = _merge_intervals(busy_intervals_local)

            # ---- Scan the local working window on a fixed grid ----
            cur = _round_up(window_start_local, slot_step_minutes)
            slot_delta = timedelta(minutes=duration_minutes)
            while cur + slot_delta <= window_end_local:
                # Check overlap against merged busy intervals (in LOCAL tz)
                overlap = False
                for bs, be in busy_intervals_local:
                    if cur < be and (cur + slot_delta) > bs:
                        overlap = True
                        # Jump ahead to the end of this busy block aligned to grid to speed scanning
                        cur = _round_up(be, slot_step_minutes)
                        break
                if overlap:
                    continue

                # Candidate is free → record it (convert start/end to UTC ISO; keep friendly local label)
                start_local_slot = cur
                end_local_slot   = cur + slot_delta
                start_utc_slot   = start_local_slot.astimezone(gettz("UTC"))
                end_utc_slot     = end_local_slot.astimezone(gettz("UTC"))
                results.append({
                    "start": start_utc_slot.isoformat(),
                    "end":   end_utc_slot.isoformat(),
                    "friendly": _friendly(start_local_slot),
                })
                if len(results) >= limit:
                    return results

                # Move to next grid tick
                cur += timedelta(minutes=slot_step_minutes)

    return results  # may be empty if no free time found in the window







def suggest_alternative_times(
    doctor_id: str,
    creds,
    num_options: int = 3
    ) -> List[Tuple[str, str]]:
    
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





def normalize_date_time(spoken_day: str, spoken_time: str) -> str:
    """
    Normalize input like '29th of July' to 'July 29'
    """
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





###########   smart parser ##########


# =============================================================================
# Noisy-ASR tolerant fallback + unified smart_parse_time wrapper
# - Safe to paste alongside your existing helpers.
# - Does NOT require global imports; uses function-local imports to avoid scope bugs.
# - Uses debug_print() if present; otherwise silently continues.
# =============================================================================

def parse_time_fallback_noisy(raw: str, *, tz_name: str = "America/Chicago",
                              default_meridiem: str = "AM"):
    """
    Robust fallback parser for casual, noisy speech recognition outputs.
    Examples it handles:
      - "August.  15 5:30 a.m."     → ("Friday, August 15", "5:30 AM")
      - "August 1 5. at 5:30 a.m."  → ("Friday, August 15", "5:30 AM")  # joins "1 5" → 15
      - "augest 15 530"             → ("Friday, August 15", "5:30 AM")  # month typo + no AM/PM
      - "8/15 at 17:30"             → ("Friday, August 15", "5:30 PM")  # 24h → 12h PM
      - "August 15 at 5:30"         → ("Friday, August 15", "5:30 AM")  # AM/PM inferred

    Returns:
      (spoken_day, spoken_time) where:
        - spoken_day  = "Friday, August 15"    (year = current system year for weekday calc)
        - spoken_time = "h:mm AM/PM"           (always 12-hour with meridiem)
      or None if the input is too ambiguous to parse confidently.

    Notes:
      - Requires at least a recognizable date (month name/abbrev OR numeric M/D).
      - If AM/PM missing, uses `default_meridiem` (configurable; default "AM").
      - Converts 24-hour inputs (e.g., 17:30, 1730) to 12-hour + PM automatically.
    """
    

    def _dbg(msg: str):
        try:
            debug_print(msg)
        except Exception:
            pass

    def _infer_meridiem(hh: int, mer: str) -> str:
        """
        If a meridiem (am/pm) was spoken, honor it. Otherwise, infer using `default_meridiem`.
        You can replace this with a smarter rule (e.g., clinic hours) if desired.
        """
        if mer:
            return mer.upper()
        _dbg(f"parse_time_fallback_noisy: ℹ️ inferring meridiem='{default_meridiem}' for hour={hh}")
        return (default_meridiem or "AM").upper()

    if not raw:
        return None

    s = raw.lower()

    # -------------------------------------------------------------------------
    # 1) Normalize punctuation/spacing and AM/PM spellings (keep colons for 5:30)
    # -------------------------------------------------------------------------
    s = (s.replace("a.m.", "am").replace("p.m.", "pm")
           .replace("a. m.", "am").replace("p. m.", "pm")
           .replace("a. m", "am").replace("p. m", "pm")
           .replace("a m", "am").replace("p m", "pm"))
    # Replace dots/commas/dashes with a single space; collapse multiple spaces.
    s = re.sub(r"[,\.\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # -------------------------------------------------------------------------
    # 2) Locate a date (prefer month names, else numeric M/D)
    # -------------------------------------------------------------------------
    MONTHS = {
        "january":1,"jan":1,
        "february":2,"feb":2,
        "march":3,"mar":3,
        "april":4,"apr":4,
        "may":5,
        "june":6,"jun":6,
        "july":7,"jul":7,
        "august":8,"aug":8,"augest":8,"augt":8,  # tolerate common typos
        "september":9,"sep":9,"sept":9,
        "october":10,"oct":10,
        "november":11,"nov":11,
        "december":12,"dec":12,
    }

    month = None
    day = None
    tokens = s.split()
    mi = -1           # index of month token if matched by name/abbr/typo
    day_index = None  # index of the day token we end up using

    # 2A) Try named month first
    for i, t in enumerate(tokens):
        if t in MONTHS:
            month = MONTHS[t]
            mi = i
            break

    # 2B) If no named month, try numeric M/D or M-D anywhere in the string.
    #     We keep this *secondary* to avoid false positives on random numbers.
    if month is None:
        mnum = re.search(r"\b(\d{1,2})[\/\-](\d{1,2})\b", s)
        if mnum:
            mval, dval = int(mnum.group(1)), int(mnum.group(2))
            if 1 <= mval <= 12 and 1 <= dval <= 31:
                month = mval
                day = dval
                # fabricate indices to guide time extraction window below
                mi = 0
                day_index = 1

    # Require a month at minimum (either named or numeric)
    if month is None:
        _dbg("parse_time_fallback_noisy: ❌ no recognizable month (named or numeric)")
        return None

    # 2C) If named month found but day still unknown, pick the first reasonable integer after it.
    if day is None:
        for j in range(mi + 1, min(mi + 4, len(tokens))):
            tj = re.sub(r"\D", "", tokens[j])
            if tj.isdigit():
                val = int(tj)
                if 1 <= val <= 31:
                    day = val
                    day_index = j
                    break
        # If not found, try joining split digits like "1 5" → 15
        if day is None and mi + 2 < len(tokens):
            a = re.sub(r"\D", "", tokens[mi + 1])
            b = re.sub(r"\D", "", tokens[mi + 2])
            if len(a) == 1 and len(b) == 1 and a.isdigit() and b.isdigit():
                val = int(a + b)
                if 1 <= val <= 31:
                    day = val
                    day_index = mi + 2

    if day is None:
        _dbg("parse_time_fallback_noisy: ❌ could not find day")
        return None

    # -------------------------------------------------------------------------
    # 3) Extract a time AFTER the day token (or after month if numeric date used)
    #    Accepts: "h:mm", "h mm", "hmm"/"hhmm", or just "h"; am/pm optional.
    # -------------------------------------------------------------------------
    # Choose a window that starts after the day token if known; else after month.
    start_idx = (day_index + 1) if (day_index is not None) else (mi + 1)
    rest = " ".join(tokens[start_idx:]) if start_idx < len(tokens) else ""

    # Try patterns in decreasing specificity:
    # (a) h:mm (with optional am/pm)  → "5:30 am", "5:30"
    # (b) h mm (with optional am/pm)  → "5 30 am", "5 30"
    # (c) hmm/hhmm (optional am/pm)   → "530 am", "0530", "1730"
    # (d) h (optional am/pm)          → "5 am", "5"
    spoken_time = None

    # (a)
    m = re.search(r"\b(\d{1,2})\s*:\s*(\d{1,2})(?:\s*(am|pm))?\b", rest)
    if m:
        hh, mm, mer = int(m.group(1)), int(m.group(2)), (m.group(3) or "").upper()
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            _dbg("parse_time_fallback_noisy: ❌ invalid h:mm bounds")
            return None
        # 24h → convert to 12h
        if hh == 0:
            hh, mer = 12, "AM"
        elif 1 <= hh <= 12:
            mer = _infer_meridiem(hh, mer)
        elif 13 <= hh <= 23:
            mer = "PM"; hh -= 12
        spoken_time = f"{hh}:{mm:02d} {mer}"

    # (b)
    if spoken_time is None:
        m2 = re.search(r"\b(\d{1,2})\s+(\d{2})(?:\s*(am|pm))?\b", rest)
        if m2:
            hh, mm, mer = int(m2.group(1)), int(m2.group(2)), (m2.group(3) or "").upper()
            if not (0 <= hh <= 23 and 0 <= mm <= 59):
                _dbg("parse_time_fallback_noisy: ❌ invalid 'h mm' bounds")
                return None
            if hh == 0:
                hh, mer = 12, "AM"
            elif 1 <= hh <= 12:
                mer = _infer_meridiem(hh, mer)
            elif 13 <= hh <= 23:
                mer = "PM"; hh -= 12
            spoken_time = f"{hh}:{mm:02d} {mer}"

    # (c)
    if spoken_time is None:
        m3 = re.search(r"\b(\d{3,4})(?:\s*(am|pm))?\b", rest)
        if m3:
            digits, mer = m3.group(1), (m3.group(2) or "").upper()
            if len(digits) == 3:  # HMM
                hh, mm = int(digits[0]), int(digits[1:])
            else:                 # HHMM
                hh, mm = int(digits[:2]), int(digits[2:])
            if not (0 <= hh <= 23 and 0 <= mm <= 59):
                _dbg("parse_time_fallback_noisy: ❌ invalid 'hmm/hhmm' bounds")
                return None
            if hh == 0:
                hh, mer = 12, "AM"
            elif 1 <= hh <= 12:
                mer = _infer_meridiem(hh, mer)
            elif 13 <= hh <= 23:
                mer = "PM"; hh -= 12
            spoken_time = f"{hh}:{mm:02d} {mer}"

    # (d)
    if spoken_time is None:
        m4 = re.search(r"\b(\d{1,2})(?:\s*(am|pm))?\b", rest)
        if m4:
            hh, mer = int(m4.group(1)), (m4.group(2) or "").upper()
            if hh == 0:
                hh, mer = 12, "AM"
            if not (1 <= hh <= 12):
                _dbg("parse_time_fallback_noisy: ❌ bare hour out of 1..12")
                return None
            mer = _infer_meridiem(hh, mer)
            spoken_time = f"{hh}:00 {mer}"

    if spoken_time is None:
        _dbg("parse_time_fallback_noisy: ❌ no recognizable time pattern")
        return None

    # -------------------------------------------------------------------------
    # 4) Build "Weekday, Month Day" for friendliness using the current year.
    # -------------------------------------------------------------------------
    try:
        year = datetime.now().year
        dt = date(year, month, day)
        weekday = dt.strftime("%A")  # e.g., "Friday"
    except Exception:
        weekday = ""
    month_name = calendar.month_name[month]
    spoken_day = f"{weekday + ', ' if weekday else ''}{month_name} {day}"

    _dbg(f"parse_time_fallback_noisy: ✅ parsed → day='{spoken_day}' time='{spoken_time}'")
    return (spoken_day, spoken_time)


# -----------------------------------------------------------------------------
# Unified smart_parse_time wrapper
# - Tries legacy parser FIRST (if it exists in your codebase)
# - Falls back to the tolerant parser above if legacy returns unusable.
# -----------------------------------------------------------------------------
try:
    # If you already defined a smart_parse_time elsewhere, capture it.
    _smart_parse_time_prev = smart_parse_time  # type: ignore[name-defined]
except Exception:
    _smart_parse_time_prev = None

def smart_parse_time(raw: str, *, tz_name: str = "America/Chicago"):
    """
    Unified parser:
      1) Try legacy smart_parse_time (if present).
         - Accepts only a (day, time) tuple with both parts non-empty.
      2) If unusable, try tolerant fallback:
         - Handles month typos, "1 5"→15, missing AM/PM, '530', '17:30', etc.
      3) Return (spoken_day, spoken_time) or None (to trigger your retries).
    """
    # 1) Legacy parser path
    if _smart_parse_time_prev:
        try:
            v = _smart_parse_time_prev(raw)
            if isinstance(v, tuple) and len(v) == 2 and all(v):
                try:
                    debug_print("smart_parse_time: ✅ legacy parser")
                except Exception:
                    pass
                return v
            else:
                try:
                    debug_print("smart_parse_time: ℹ️ legacy unusable → trying fallback")
                except Exception:
                    pass
        except Exception as e:
            try:
                debug_print(f"smart_parse_time: ℹ️ legacy error → {e} ; trying fallback")
            except Exception:
                pass

    # 2) Fallback path (tolerant)
    v = parse_time_fallback_noisy(raw, tz_name=tz_name, default_meridiem="AM")
    if v:
        try:
            debug_print(f"smart_parse_time: ✅ fallback parsed → day='{v[0]}' time='{v[1]}'")
        except Exception:
            pass
        return v

    # 3) Both failed
    try:
        debug_print("smart_parse_time: ❌ both parsers failed")
    except Exception:
        pass
    return None



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
           "user"    Represents input from the end user — the prompt/question you're asking.
                      If you're building a voice bot (as you are), the "user" input is usually:
                    request.values.get("SpeechResult", "")
                    So you should assume it’s spoken language and keep the assistant ready for less formal wording, e.g.:
                    "I want to talk to the doctor."
                     "Book me a session at 5."
                     "               I need help."

            "assistant"	Represents a reply from the AI assistant — used to simulate ongoing dialogue or give memory context.
        """
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


# Initialize the OpenAI client (using the environment variable OPENAI_API_KEY)
#client = OpenAI()



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
    debug_print(f"[get_doctor_filename] Full path for '{friendly_name}' → {path}")
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
    debug_print(f"🔍 File → {full_path}")

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
                debug_print(f"✅ Loaded list with {len(appts)} appointment(s)")
            else:
                debug_print("⚠️ Root JSON was not a list; reinitializing")
        except Exception as e:
            debug_print(f"⚠️ Failed to parse JSON → {e}")
    else:
        debug_print("📂 No file found — starting new list")

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

    debug_print(f"🔎 Search by phone+dob → {len(matches)} match(es) "
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
            debug_print("🔁 Exact duplicate detected — skipping append")
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
    debug_print(f"➕ Appended: {new_record}")

    # -----------------------------
    # Save back to disk (+ cache)
    # -----------------------------
    try:
        with open(full_path, "w") as f:
            json.dump(appts, f, indent=2)
        debug_print(f"💾 Saved to {full_path}")

        # Update in-memory cache if present
        try:
            doctor_appointments[filename] = appts
        except Exception:
            pass

        return {"created": True, "record": new_record, "reason": None}
    except Exception as e:
        debug_print(f"❌ Failed to write JSON → {e}")
        raise







def normalize_phone_digits(phone: str) -> str:
    """Digits-only normalization for matching (calendar description & JSON)."""
    return ''.join(ch for ch in (phone or "") if ch.isdigit())


# ===== local doctor JSON cancellation (by doctor+phone+dob+utc_start) =====



def cancel_appointment_by_name(doctor_name: str, phone: str, dob: str, utc_start: str) -> bool:
    """
    Remove a single appointment from appointment_data/doctors/<doctor>.json
    matching ALL of:
      • phone (10-digit normalized)
      • dob (exact string match; expected ISO YYYY-MM-DD)
      • time (exact UTC ISO match)
    Returns True if a record was removed, else False.
    """
    def normalize_phone_digits(s: str) -> str:
        d = "".join(ch for ch in (s or "") if ch.isdigit())
        return d[1:] if len(d) == 11 and d.startswith("1") else d

   
    
    full_path = get_doctor_filename(doctor_name)
    phone10 = normalize_phone_digits(phone)
    dob_str = (dob or "").strip()

    debug_print(f"cancel_appointment_by_name: doctor='{doctor_name}' phone='{phone10}' dob='{dob_str}' utc='{utc_start}'")

    if not (os.path.exists(full_path) and phone10 and dob_str and utc_start):
        return False

    # load list
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return False
    except Exception as e:
        debug_print(f"cancel_appointment_by_name: read error → {e}")
        return False

    # normalize target UTC
    try:
        target_norm = dtparser.isoparse(utc_start).astimezone().astimezone(tz=None).isoformat()
    except Exception as e:
        debug_print(f"cancel_appointment_by_name: utc parse error → {e}")
        return False

    kept = []
    removed = 0
    for appt in data:
        if not isinstance(appt, dict):
            kept.append(appt)
            continue
        ap_phone = normalize_phone_digits(appt.get("phone", ""))
        ap_dob   = (appt.get("dob", "") or "").strip()
        ap_time_raw = (appt.get("time") or appt.get("start") or "").strip()
        try:
            ap_time_norm = dtparser.isoparse(ap_time_raw).astimezone().astimezone(tz=None).isoformat()
        except Exception:
            kept.append(appt)
            continue

        if ap_phone == phone10 and ap_dob == dob_str and ap_time_norm == target_norm:
            removed += 1
        else:
            kept.append(appt)

    if removed == 0:
        debug_print("cancel_appointment_by_name: no matching record found")
        return False

    try:
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(kept, f, indent=2, ensure_ascii=False)
        debug_print(f"cancel_appointment_by_name: ✅ deleted {removed} appt(s)")
        return True
    except Exception as e:
        debug_print(f"cancel_appointment_by_name: write error → {e}")
        return False








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

# ---------- Config ----------


# ---------- Logging helper ----------
try:
    debug_print  # type: ignore # will raise if not defined
except NameError:  # minimal fallback so this module is self-contained
    def debug_print(*args, **kwargs):
        print(*args, **kwargs)

# ---------- Init ----------
def init_db() -> None:
    """
    Ensure appointment_data folder exists and customers.json is a dict file.
    Creates an empty {} if missing or invalid.
    """
    os.makedirs(DB_FOLDER, exist_ok=True)
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)
        return

    # Validate existing file is a JSON object; if not, reset to {}
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("customers.json must be a JSON object")
    except Exception:
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)



# ---------- Sanitizers / formatters ----------
def _oneline(s: str) -> str:
    """Compact whitespace/newlines to a single line."""
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



def _render_block_lines(new: bool, rec: Dict[str, Any]) -> List[str]:
    """
    Build a fixed-length, human-readable *12-line* block that summarizes a single
    customer record. This block is intended for logs or plaintext exports only.
    Your canonical, machine-readable store remains `customers.json`.

    SECURITY:
      - PAN (credit card number) and CVV are **masked** here so that raw values
        are never written to a human-readable file. Masking functions are:
          • _mask_pan(...)  → shows last 4 digits (e.g., ************7026)
          • _mask_all(...)  → masks entire string (e.g., ***)
      - Never mutate `rec`; this function is read-only.

    OUTPUT CONTRACT:
      - Returns a list of **exactly 12 strings**, in a strict, known order:
          0: Title line (varies with `new`)
          1: Phone
          2: DOB
          3: First Name
          4: Last Name
          5: Address
          6: CC Name
          7: CC Number (masked)
          8: CC Exp
          9: CC CVV (masked)
         10: Created At
         11: Last Seen At
      - Each line follows the form "Label: Value".
      - Missing/empty values are rendered as an em dash '—' to keep layout stable.

    PARAMETERS:
      new : bool
          If True, the title line should reflect a "new customer" (e.g., something
          like "insert_customer: ✅ Added new customer"). If False, use a generic
          on-file title. The exact wording comes from `_block_title(new)`.
      rec : Dict[str, Any]
          The customer record dictionary (already normalized); expected keys:
            'phone', 'dob', 'first_name', 'last_name', 'address',
            'cc_name', 'cc_number', 'cc_exp', 'cc_cvv',
            'created_at', 'last_seen_at'
          Any key may be missing/empty; we display '—' in that case.

    RETURNS:
      List[str] : The 12 lines described above, suitable for joining with '\n'
                  and writing to a log file.

    NOTE:
      - This renderer is intentionally dumb and side-effect free: it does not perform
        validation or formatting beyond masking and fallback to '—'.
      - Keep the labels and order stable; other utility functions may rely on
        parsing these lines by position/label (e.g., _iter_blocks / _get_value).
    """

    # Pull values from the record. We deliberately don’t mutate or normalize here;
    # we only render whatever the caller provided, swapping empty/None for '—'.
    phone        = rec.get("phone") or "—"
    dob          = rec.get("dob") or "—"
    first_name   = rec.get("first_name") or "—"
    last_name    = rec.get("last_name") or "—"
    address      = rec.get("address") or "—"
    cc_name      = rec.get("cc_name") or "—"
    cc_number    = _mask_pan(rec.get("cc_number"))  # show only last 4 digits
    cc_exp       = rec.get("cc_exp") or "—"
    cc_cvv       = _mask_all(rec.get("cc_cvv"))     # mask entire CVV
    created_at   = rec.get("created_at") or "—"
    last_seen_at = rec.get("last_seen_at") or "—"

    # Assemble the 12-line block in a consistent order.
    # Line 0 is a dynamic title provided by `_block_title(new)`.
    lines: List[str] = [
        _block_title(new),             # 0  → e.g., "insert_customer: ✅ Added new customer" or "Customer on file"
        f"Phone: {phone}",             # 1
        f"DOB: {dob}",                 # 2
        f"First Name: {first_name}",   # 3
        f"Last Name: {last_name}",     # 4
        f"Address: {address}",         # 5
        f"CC Name: {cc_name}",         # 6
        f"CC Number: {cc_number}",     # 7 (masked)
        f"CC Exp: {cc_exp}",           # 8
        f"CC CVV: {cc_cvv}",           # 9 (masked)
        f"Created At: {created_at}",   # 10
        f"Last Seen At: {last_seen_at}"# 11
    ]

    # Optional defensive check (keep as comment to avoid runtime overhead):
    # assert len(lines) == 12, "Rendered block must contain exactly 12 lines"

    return lines


# ---------- File parsing helpers ----------


# BEFORE:
# def _iter_blocks(lines: list[str]):
# AFTER (3.8-safe):
def _iter_blocks(lines: List[str]) -> Iterator[List[str]]:
    # ... keep your existing body ...
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



# change this:
# def _get_value(block_lines: list[str], label: str) -> str | None:

# to this:
# BEFORE:
# def _get_value(block_lines: list[str], label: str) -> str | None:
# AFTER (3.8-safe):
def _get_value(block_lines: List[str], label: str) -> Optional[str]:
    # ... keep your existing body ...
    """Fetch 'Label: value' from a block."""
    prefix = f"{label}:"
    for ln in block_lines:
        if ln.startswith(prefix):
            return ln.split(":", 1)[1].strip()
    return None

# BEFORE:
# def _extract_phone_dob(block_lines: list[str]) -> tuple[str | None, str | None]:
# AFTER (3.8-safe):
def _extract_phone_dob(block_lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
    # ... keep your existing body ...
    """Get (Phone, DOB) from a block."""
    return _get_value(block_lines, "Phone"), _get_value(block_lines, "DOB")

# ---------- Public API ----------

def _normalize_phone10(phone: str) -> str:
    """
    Keep digits only, drop leading US '1' if present, and return 10-digit phone or ''.
    """
    d = "".join(ch for ch in (phone or "") if ch.isdigit())
    if len(d) == 11 and d.startswith("1"):
        d = d[1:]
    return d if len(d) == 10 else ""

def _load_customers() -> Dict[str, Dict[str, Any]]:
    """Read the customers map from disk (already ensured by init_db)."""
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        debug_print(f"customers.json read error → {e}")
    return {}


def _key(phone10: str, dob_iso: str) -> str:
    """Stable map key to prevent duplicates."""
    return f"{phone10}|{dob_iso or ''}"

def customer_search(phone: str, dob: str) -> bool:
    """
    Return True if a customer (phone|dob) exists in customers.json, else False.
    """
    init_db()
    phone10 = _normalize_phone10(phone)
    dob_iso = (dob or "").strip()
    if not phone10:
        return False
    data = _load_customers()
    exists = _key(phone10, dob_iso) in data
    debug_print(f"customer_search: phone={phone10} dob={dob_iso or '∅'} → {exists}")
    return exists


def _save_customers(data: Dict[str, Dict[str, Any]]) -> None:
    """Write the customers map to disk in readable (pretty) form."""
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



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


def insert_customer(
    phone: str,
    dob: str,
    first_name: str,
    last_name: str,
    address: str,
    cc_name: str,
    cc_number: str,
    cc_exp: str,
    cc_cvv: str,
) -> bool:
    """
    Insert or update a customer in customers.json (single pretty JSON dict):
      • If (phone|dob) exists: update 'last_seen_at' only; return False.
      • If new: create record with 'created_at' and 'last_seen_at'; return True.
    Never duplicates because the map key is unique.
    All values are stored on one logical line each (pretty JSON with indent=2).
    """
    init_db()
    phone10 = _normalize_phone10(phone)
    dob_iso = (dob or "").strip()
    if not phone10:
        raise ValueError("insert_customer: invalid phone (must normalize to 10 digits)")

    data = _load_customers()
    key = _key(phone10, dob_iso)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if key in data:
        # existing → just refresh last_seen_at
        data[key]["last_seen_at"] = now
        _save_customers(data)
        debug_print(f"insert_customer: ℹ️ exists; updated last_seen_at for {key}")
        return False

    # new record
    rec: Dict[str, Any] = {
        "phone": phone10,
        "dob": dob_iso,
        "first_name": _oneline(first_name),
        "last_name": _oneline(last_name),
        "address": _oneline(address),
        # store CC fields if captured (can be empty strings)
        "cc_name": _oneline(cc_name),
        "cc_number": _oneline(cc_number),
        "cc_exp": _oneline(cc_exp),   # MM/YY
        "cc_cvv": _oneline(cc_cvv),
        "created_at": now,
        "last_seen_at": now,
    }
    data[key] = rec
    _save_customers(data)

    # mask PAN/CVV in logs
    pan = rec.get("cc_number", "")
    masked_pan = ("*" * max(0, len(pan) - 4)) + pan[-4:] if pan else ""
    cvv = rec.get("cc_cvv", "")
    masked_cvv = "*" * len(cvv) if cvv else ""

    debug_print(
        "insert_customer: ✅ Added new customer\n"
        f"Phone: {rec['phone']}\n"
        f"DOB: {rec['dob'] or '∅'}\n"
        f"First Name: {rec['first_name']}\n"
        f"Last Name: {rec['last_name']}\n"
        f"Address: {rec['address']}\n"
        f"CC Name: {rec.get('cc_name','')}\n"
        f"CC Number: {masked_pan}\n"
        f"CC Exp: {rec.get('cc_exp','')}\n"
        f"CC CVV: {masked_cvv}\n"
        f"Created At: {rec['created_at']}\n"
        f"Last Seen At: {rec['last_seen_at']}"
    )
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

def update_cc_info(
    phone: str,
    dob: str,
    *,
    cc_name: Optional[str] = None,
    cc_number: Optional[str] = None,
    cc_exp: Optional[str] = None,
    cc_cvv: Optional[str] = None,
) -> bool:
    """
    Update the customer's CC fields in customers.json (by phone|dob).
    Returns True if updated, False if no such customer.
    """
    init_db()
    phone10 = _normalize_phone10(phone)
    dob_iso = (dob or "").strip()
    if not phone10:
        return False

    data = _load_customers()
    key = _key(phone10, dob_iso)
    if key not in data:
        return False

    rec = data[key]
    if cc_name is not None:
        rec["cc_name"] = _oneline(cc_name)
    if cc_number is not None:
        rec["cc_number"] = _oneline(cc_number)
    if cc_exp is not None:
        rec["cc_exp"] = _oneline(cc_exp)
    if cc_cvv is not None:
        rec["cc_cvv"] = _oneline(cc_cvv)

    rec["last_seen_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_customers(data)

    pan = rec.get("cc_number", "")
    masked_pan = ("*" * max(0, len(pan) - 4)) + pan[-4:] if pan else ""
    masked_cvv = "*" * len(rec.get("cc_cvv", ""))

    debug_print(
        "update_cc_info: ✅ updated\n"
        f"Phone: {phone10}\n"
        f"DOB: {dob_iso or '∅'}\n"
        f"CC Name: {rec.get('cc_name','')}\n"
        f"CC Number: {masked_pan}\n"
        f"CC Exp: {rec.get('cc_exp','')}\n"
        f"CC CVV: {masked_cvv}\n"
        f"Last Seen At: {rec['last_seen_at']}"
    )
    return True




def get_doctor_appts_for(doctor_name: str, phone: str, dob: str = None) -> list:
    """
    Read appointment_data/doctors/<doctor_name>.json and return all
    appointment dicts that match the given caller:
      - phone is REQUIRED (normalized to 10 digits; strips leading 1)
      - dob is OPTIONAL (normalized to YYYY-MM-DD if possible)

    Returned list is sorted chronologically by start time if present.
    Uses debug_print for logging (falls back to print if unavailable).
    """

    

    # ---------- local helpers (self-contained) ----------

    

    def _normalize_phone_digits(s: str) -> str:
        """Keep only digits; if 11-digit US starting with '1', strip to 10 digits."""
        d = "".join(ch for ch in (s or "") if ch.isdigit())
        return d[1:] if len(d) == 11 and d.startswith("1") else d

    def _normalize_dob_iso(s: str) -> str:
        """
        Normalize to 'YYYY-MM-DD' when possible.
        Accepts:
          - 'YYYY-MM-DD' (kept)
          - 'MM/DD/YYYY' or 'MM-DD-YYYY' (converted)
          - empty/unknown → ''
        """
        s = (s or "").strip()
        if not s:
            return ""
        if "T" in s:
            s = s.split("T", 1)[0].strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            return s
        m = re.match(r"^\s*(\d{1,2})[\/-](\d{1,2})[\/-](\d{4})\s*$", s)
        if m:
            mm, dd, yyyy = m.groups()
            try:
                return f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}"
            except Exception:
                return s
        return s


    def _extract_start_iso(appt: dict) -> str:
        """Prefer 'start' then 'time' field; may be empty if not present."""
        return (appt.get("start") or appt.get("time") or "").strip()

    # ---------- normalize inputs ----------
    phone10 = _normalize_phone_digits(phone)
    dob_iso = _normalize_dob_iso(dob) if dob else ""

    if len(phone10) != 10:
        debug_print(f"get_doctor_appts_for: ❌ invalid phone '{phone}' → normalized '{phone10}'")
        return []

    path = get_doctor_filename(doctor_name)
    if not os.path.exists(path):
        debug_print(f"get_doctor_appts_for: ⚠️ file not found → {path}")
        return []

    # ---------- load and filter ----------
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        debug_print(f"get_doctor_appts_for: ❌ read/parse error for {path} → {e}")
        return []

    if not isinstance(data, list):
        debug_print(f"get_doctor_appts_for: ❌ JSON not a list in {path}")
        return []

    matches = []
    for appt in data:
        if not isinstance(appt, dict):
            continue
        ap_phone = _normalize_phone_digits(appt.get("phone", ""))
        if ap_phone != phone10:
            continue
        if dob_iso:
            ap_dob = _normalize_dob_iso(appt.get("dob", ""))
            if ap_dob != dob_iso:
                continue
        matches.append(appt)

    # ---------- sort by start time if available ----------
    def _sort_key(a: dict):
        raw = _extract_start_iso(a)
        try:
            return dtparser.isoparse(raw)
        except Exception:
            return None

    try:
        matches.sort(key=lambda a: (_sort_key(a) is None, _sort_key(a)))
    except Exception:
        pass

    debug_print(f"get_doctor_appts_for: ✅ doctor='{doctor_name}' phone='{phone10}' dob='{dob_iso or '∅'}' → {len(matches)} appt(s)")
    return matches


# ----------------------------------------------------------------------
# backward-compat alias (typo): some code may call get_docotor_appt_for
# ----------------------------------------------------------------------
def get_docotor_appt_for(doctor_name: str, phone: str, dob: str = None) -> list:
    """
    Alias for get_doctor_appts_for (typo compatibility).
    """
    return get_doctor_appts_for(doctor_name, phone, dob)



# =============================================================================
# Helper: Per-doctor availability using Google Calendar FreeBusy
# - Purpose-built for availability. More reliable than events().list overlap.
# - Adds ±1s boundary "fuzz" to avoid edge inclusivity issues.
# - Logs any blocking busy windows for debugging.
# =============================================================================
def is_time_slot_available(calendar_id: str, start_iso: str, end_iso: str, creds) -> bool:
    """
    Return True if NO overlapping event exists on this doctor's calendar.
    - Primary check: Google Calendar FreeBusy (robust for availability).
    - Fallback: events().list with explicit overlap test.
    - Adds ±60s guard to catch edge-inclusive events.
    - Emits debug lines showing what blocked the slot when busy.
    """
    

    service = build("calendar", "v3", credentials=creds)

    start_dt = isoparse(start_iso)
    end_dt   = isoparse(end_iso)
    # ±60s guard for edge cases where an event starts/ends exactly at boundaries
    tmin = (start_dt - timedelta(seconds=60)).isoformat()
    tmax = (end_dt   + timedelta(seconds=60)).isoformat()

    debug_print(
        "is_time_slot_available: 🔎 Checking "
        f"calendar='{calendar_id}' window={start_iso}→{end_iso} "
        f"(padded {tmin}→{tmax})"
    )

    # ---- 1) FreeBusy (preferred) ----
    try:
        fb = service.freebusy().query(body={
            "timeMin": tmin,
            "timeMax": tmax,
            "items": [{"id": calendar_id}],
            "timeZone": "UTC",
        }).execute()

        busy = (fb.get("calendars", {}).get(calendar_id, {}) or {}).get("busy", []) or []
        if busy:
            debug_print(f"is_time_slot_available: 🚫 FreeBusy shows {len(busy)} busy block(s)")
            for b in busy:
                debug_print(f"  BUSY (freebusy) {b.get('start')} → {b.get('end')}")
            debug_print("is_time_slot_available: ❌ Slot NOT available (FreeBusy)")
            return False
        else:
            debug_print("is_time_slot_available: ✅ FreeBusy shows NO busy blocks in padded window")
    except Exception as e:
        debug_print(f"is_time_slot_available: ⚠️ FreeBusy error → {e} (will check events().list)")

    # ---- 2) Fallback/double-check: events().list with explicit overlap test ----
    try:
        items = service.events().list(
            calendarId=calendar_id,
            timeMin=tmin,
            timeMax=tmax,
            singleEvents=True,
            showDeleted=False,
            orderBy="startTime",
            maxResults=250,
        ).execute().get("items", [])

        debug_print(f"is_time_slot_available: ℹ️ events().list returned {len(items)} item(s) in padded window")

        for ev in items:
            if ev.get("status") == "cancelled":
                continue
            # 'transparent' events shouldn't block availability; 'opaque' (default) does.
            if ev.get("transparency") == "transparent":
                continue

            ev_start_raw = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date")
            ev_end_raw   = ev.get("end",   {}).get("dateTime") or ev.get("end",   {}).get("date")
            if not ev_start_raw or not ev_end_raw:
                continue

            es = isoparse(ev_start_raw)
            ee = isoparse(ev_end_raw)

            # Overlap rule: (start < ev_end) AND (end > ev_start)
            if start_dt < ee and end_dt > es:
                title = ev.get("summary", "")
                debug_print(
                    "is_time_slot_available: 🚫 BUSY (events overlap) "
                    f"{es.isoformat()} → {ee.isoformat()} title='{title}'"
                )
                debug_print("is_time_slot_available: ❌ Slot NOT available (events overlap)")
                return False

        debug_print("is_time_slot_available: ✅ No overlapping events found via events().list")
    except Exception as e:
        # Safer to fail closed than double-book if both checks error out.
        debug_print(f"is_time_slot_available: ❗ events.list error → {e}; treating as NOT available (fail-closed)")
        return False

    debug_print("is_time_slot_available: ✅ Slot FREE (final)")
    return True






#app = Flask(__name__)



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
                            debug_print(f"✅ Matched via GPT fallback: {friendly}")
                            break
            except Exception as e:
                debug_print(f"⚠️ GPT fallback failed: {e}")

        # ------------------------------------------------------------------
        # ❌ 3. Still no match → Retry logic
        # ------------------------------------------------------------------
        if matched_id is None:
            debug_print(f"❌ No doctor match for: '{spoken_clean}'")
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


    # ===== collect_phone (stage) =====
    elif stage == "collect_phone":
        # Collect a 10-digit US phone via speech or DTMF, stay here until valid.

        session_data.setdefault(call_sid, {}).setdefault("customer", {})
        session_data[call_sid]["retry_phone"] = session_data[call_sid].get("retry_phone", 0)
        """
        re.sub(pattern, repl, string, count=0)
            pattern → regex pattern to search for.

            repl → what to replace it with.

            string → the original text.

            count (optional) → how many occurrences to replace (default = all).
            d = re.sub(r"\D", "", raw_digits or "")
                    raw_digits or "" → if raw_digits is None, use "" instead.

                    \D = regex for “any character that is NOT a digit”.

                    re.sub(r"\D", "", ...) → replace all non-digits with "" (delete them).
                    Example
                    "(312) 555-0199" → "3125550199"
                    "+1-800-123-4567" → "18001234567"
            if len(d) == 11 and d.startswith("1"):
                    d = d[1:]
                            If the number is 11 digits long and starts with "1",
                            → Strip off the leading "1".

                This is because "1" is the US country code.
                ✅ Example:
                      "18001234567" → "8001234567"
                return d if len(d) == 10 else ""
                    If the result is exactly 10 digits → return it (valid US number).
                    Otherwise → return empty string "" (invalid phone).
                    ✅ Example:
                    "3125550199" → valid, returned as "3125550199".
                    "5550199" → too short, returns "".
                    "441234567890" → UK number, not 10 digits, returns "".
        """

        def _validate_normalize_us_phone(raw_digits: str) -> str:
            d = re.sub(r"\D", "", raw_digits or "")
            if len(d) == 11 and d.startswith("1"):
                d = d[1:]
            return d if len(d) == 10 else ""

        # Try speech first
        speech_text = (speech_result or "").strip()
        debug_print(f"collect_phone: speech='{speech_text}'")
        speech_digits = re.sub(r"\D", "", speech_text)
        normalized = _validate_normalize_us_phone(speech_digits)

        # Fallback to DTMF if speech invalid
        if not normalized:
            try:
                dtmf_digits = (request.values.get("Digits") or "").strip()
            except Exception:
                dtmf_digits = ""
            debug_print(f"collect_phone: dtmf='{dtmf_digits}'")
            normalized = _validate_normalize_us_phone(dtmf_digits)

        if not normalized:
            # soft retry (don’t bump counter if caller started input)
            heard_some = bool(speech_digits or (locals().get("dtmf_digits", "")))
            if not heard_some:
                session_data[call_sid]["retry_phone"] += 1

            if session_data[call_sid]["retry_phone"] >= MAX_GET_PHONE_RETRIES:
                debug_print("collect_phone: max retries reached")
                resp.say(gpt_speak("Sorry, I couldn’t capture a valid phone number. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            gather = make_gather(
                "Please provide your ten digit phone number including area code. "
                "You can say it clearly with short pauses, or type the digits and press pound.",
                hints="zero one two three four five six seven eight nine double triple"
            )
            resp.append(gather)
            resp.say(gpt_speak("I didn't get the phone number."), VOICE)
            resp.redirect("/voice")
            return str(resp)

        # valid → store and move on
        session_data[call_sid]["customer"]["phone"] = normalized
        debug_print(f"collect_phone: ✅ accepted={normalized}")
        session_data[call_sid]["stage"] = "collect_dob"

        gather = make_gather(
            "Thank you. Please say your date of birth, for example, January fifteenth nineteen eighty five. "
            "Or enter two digits for month, two for day, and four for year, then press pound."
        )
        resp.append(gather)
        return str(resp)










    # ----------------------------------------------------------------------
    # 🎂 Stage: collect_dob
    # Purpose:
    #   - Accept DOB via speech (e.g., “July third 1990”) or keypad (MMDDYYYY#).
    #   - Parse and validate reasonable date range.
    #   - Store DOB as ISO (YYYY-MM-DD) in session.
    #   - On failure, re-prompt with the SHORT prompt (DOB_PROMPT_SHORT).
    # Integration points:
    #   - Uses: parse_dob_input(), make_gather_dob(), debug_print, session_data, call_sid
    #   - Next stage: ask_time_date (always, after successful DOB store)
    # ----------------------------------------------------------------------
    elif stage == "collect_dob":
            # ----------------------------------------------------------------------
        # 🔊 Short, centralized prompts
        #   Put these near your other constants so every stage uses the same text.
        # ----------------------------------------------------------------------
        DOB_PROMPT_SHORT = (
            "Please say your birth date, for example 'July third 1990'. "
            "Or type MMDDYYYY then press pound."
        )
        TIME_PROMPT_SHORT = "Please say the date and time, for example 'August 12 at 5 PM'."

        debug_print("collect_dob: 📍 Stage entered")

        # 1) Pull DTMF if present (Twilio sends digits on the same webhook), otherwise use speech.
        try:
            dtmf_digits = (request.values.get("Digits") or "").strip()
        except Exception:
            dtmf_digits = ""

        speech_text = (speech_result or "").strip()
        debug_print(f"collect_dob: 🎙️ speech_text='{speech_text}', 🔢 dtmf_digits='{dtmf_digits}'")

        # 2) Parse DOB input (helper handles speech and/or MMDDYYYY)
        dt = parse_dob_input(speech_text, dtmf_digits)
        if not dt:
            # Retry counter (so we don’t loop forever)
            session_data[call_sid]["retry_dob"] = session_data[call_sid].get("retry_dob", 0) + 1
            r = session_data[call_sid]["retry_dob"]
            debug_print(f"collect_dob: ❌ Parse failed. Retry={r}")

            if r >= 3:
                # Fail out cleanly if user can’t provide a DOB we can parse
                resp.say(gpt_speak("Sorry, I couldn’t understand your date of birth. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Re-prompt using short, consistent copy
            gather = make_gather_dob(DOB_PROMPT_SHORT)
            resp.append(gather)
            return str(resp)

        # 3) Validate DOB sanity window (e.g., 1900..today)
        try:
            today = date.today()
            min_date = date(1900, 1, 1)
            dob_date = dt.date()
            if not (min_date <= dob_date <= today):
                debug_print(f"collect_dob: ⚠️ DOB out of range → {dob_date.isoformat()}")

                session_data[call_sid]["retry_dob"] = session_data[call_sid].get("retry_dob", 0) + 1
                gather = make_gather_dob(DOB_PROMPT_SHORT)
                resp.append(gather)
                return str(resp)
        except Exception as e:
            # Do not fail the call; just log and continue to store parsed value
            debug_print(f"collect_dob: ⚠️ Validation error → {e}")

        # 4) Store ISO DOB in session
        iso_dob = dt.strftime("%Y-%m-%d")
        session_data[call_sid].setdefault("customer", {})
        session_data[call_sid]["customer"]["dob"] = iso_dob
        debug_print(f"collect_dob: ✅ Stored DOB → {iso_dob}")

        # 5) Always move to ask_time_date next (your booking flow expects this)
        session_data[call_sid]["stage"] = "ask_time_date"
        debug_print("collect_dob: ➡️ Next stage → ask_time_date")

        # 6) Prompt for appointment time/date using the short prompt
        gather = make_gather("Thanks. " + TIME_PROMPT_SHORT)
        resp.append(gather)
        return str(resp)




# ----------------------------------------------------------------------
# 📅 Stage: ask_time_date
# Purpose:
#   - Parse spoken date/time (e.g., “August 12 at 5 PM”).
#   - Compute a UTC timeslot window for the appointment.
#   - Check provider availability; if busy, offer next available options.
#   - If free:
#       * If (phone + dob) exists in DB → skip name collection and go to confirm.
#       * Else → collect first name.
# Prompts:
#   - Uses TIME_PROMPT_SHORT when re-prompting.
# Integration points:
#   - Uses: smart_parse_time(), build_timeslot_range(), is_time_slot_available(),
#           get_next_available_slots(), customer_search(), make_gather()
#   - Globals referenced: APPOINTMENT_DURATION_MINUTES, googleid_dr_name_map, creds
# ----------------------------------------------------------------------
    elif stage == "ask_time_date":
        debug_print(f"ask_time_date: 🗣️ Received speech: {speech_result}")

        # ----------------------------------------------------------------------
        # Prompt constants
        # Keep these short and consistent so callers learn the pattern quickly.
        # ----------------------------------------------------------------------
        TIME_PROMPT_SHORT = (
            "That doesn't sound like a valid date or time. "
            "Please say the appointment time again, for example, "
            "'August 15th at 5 AM'."
        )
        PROMPT_NEED_BOTH = (
            "I couldn't hear the date or the time. "
            "Please say both, for example, 'August 15th at 5 AM'."
        )
        PROMPT_NEED_DATE = (
            "I couldn't hear the date. "
            "Please include the date and time, for example, 'August 15th at 5 AM'."
        )
        PROMPT_NEED_TIME = (
            "I couldn't hear the time. "
            "Please include the time as well, for example, 'August 15th at 5 AM'."
        )

        # --- tiny helpers for readability ---
        def _is_blank(x) -> bool:
            return (x is None) or (str(x).strip() == "")

        def _has_time_token(raw: str) -> bool:
            """Heuristic: if parse fully failed, check if caller likely said a time."""
            s = (raw or "").lower()
            return ("am" in s) or ("pm" in s) or (":" in s) or ("o'clock" in s) or ("oclock" in s) or re.search(r"\b\d{3,4}\b", s) is not None

        def _has_date_token(raw: str) -> bool:
            """Heuristic: if parse fully failed, check for month/weekday/date words."""
            s = (raw or "").lower()
            months = ("january","february","march","april","may","june","july",
                    "august","september","october","november","december",
                    "jan","feb","mar","apr","jun","jul","aug","sep","sept","oct","nov","dec")
            weekdays = ("monday","tuesday","wednesday","thursday","friday","saturday","sunday",
                        "mon","tue","tues","wed","thu","thur","thurs","fri","sat","sun")
            keywords = ("today","tomorrow","tmrw","next","this","on","at","the")
            if any(m in s for m in months): return True
            if any(w in s for w in weekdays): return True
            if any(k in s for k in keywords): return True
            if "/" in s or "-" in s: return True  # dates like 8/15 or 08-15
            return False

        # Ensure we have 're' available even if not globally imported
        try:
            import re
        except Exception:
            pass

        # --- Minimal pre-clean for AM/PM variants & trailing punctuation ---
        try:
            _raw = (speech_result or "").strip()
            _raw = re.sub(r"\b(a\s*\.?\s*m\.?)\b", "am", _raw, flags=re.IGNORECASE)
            _raw = re.sub(r"\b(p\s*\.?\s*m\.?)\b", "pm", _raw, flags=re.IGNORECASE)
            _raw = re.sub(r"[.!?]\s*$", "", _raw)
        except Exception:
            _raw = (speech_result or "")

        # 0) Guard: doctor must be chosen (per-doctor calendar)
        doctor_id = session_data.get(call_sid, {}).get("doctor_id")
        if not doctor_id:
            debug_print("ask_time_date: ❌ no doctor selected → sending user to pick a doctor")
            session_data.setdefault(call_sid, {})["stage"] = "choose_doctor"
            gather = make_gather("Which doctor would you like to see?")
            resp.append(gather)
            return str(resp)

        calendar_id = doctor_id

        # 1) Parse (day, time) from the caller’s phrase
        #    Expect a tuple like: ("Friday, August 15", "5:00 AM") or similar.
        time_info = smart_parse_time(_raw)

        # --- Branch A: parser returned nothing useful (None / wrong type / wrong length) ---
        if not time_info or not isinstance(time_info, tuple) or len(time_info) != 2:
            # Before generic retry: try to be *specific* about what’s missing using heuristics.
            need_date = not _has_date_token(_raw)
            need_time = not _has_time_token(_raw)

            if need_date and need_time:
                prompt = PROMPT_NEED_BOTH
            elif need_date:
                prompt = PROMPT_NEED_DATE
            elif need_time:
                prompt = PROMPT_NEED_TIME
            else:
                prompt = TIME_PROMPT_SHORT

            # Retry parsing up to 3 times
            session_data.setdefault(call_sid, {})
            session_data[call_sid]["retry_time"] = session_data[call_sid].get("retry_time", 0) + 1
            retry_count = session_data[call_sid]["retry_time"]
            debug_print(f"ask_time_date: ⚠️ Time parse failed (no tuple). Retry={retry_count} — prompt='{prompt}'")

            if retry_count >= 3:
                debug_print("ask_time_date: ⛔ Max retries reached.")
                resp.say(gpt_speak("Sorry, I still couldn't understand the date and time. Please try again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            gather = make_gather(prompt)
            resp.append(gather)
            return str(resp)

        # --- Branch B: parser returned (day, time) — validate each part explicitly ---
        spoken_day, spoken_time = time_info
        debug_print(f"ask_time_date: 📆 Extracted → Day: {spoken_day}, Time: {spoken_time}")

        missing_date = _is_blank(spoken_day)
        missing_time = _is_blank(spoken_time)

        if missing_date or missing_time:
            if missing_date and missing_time:
                prompt = PROMPT_NEED_BOTH
            elif missing_date:
                prompt = PROMPT_NEED_DATE
            else:
                prompt = PROMPT_NEED_TIME

            session_data[call_sid]["retry_time"] = session_data[call_sid].get("retry_time", 0) + 1
            retry_count = session_data[call_sid]["retry_time"]
            debug_print(f"ask_time_date: ⛔ Missing component(s). date={missing_date} time={missing_time} Retry={retry_count}")

            if retry_count >= 3:
                debug_print("ask_time_date: ⛔ Max retries reached (component missing).")
                resp.say(gpt_speak("Sorry, I still couldn't get the full date and time. Please try again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            gather = make_gather(prompt)
            resp.append(gather)
            return str(resp)

        # Persist what the caller actually said (useful for debugging / confirmation later).
        session_data[call_sid]["spoken_day"] = spoken_day
        session_data[call_sid]["spoken_time"] = spoken_time

        # 2) Convert to concrete UTC timeslot (start/end)
        try:
            appointment_start, appointment_end = build_timeslot_range(spoken_day, spoken_time)
            session_data[call_sid]["appointment_time"] = {"start": appointment_start, "end": appointment_end}
            # Reset retry counter after a successful parse/build
            session_data[call_sid]["retry_time"] = 0
            debug_print(f"ask_time_date: ⏰ Built slot → Start: {appointment_start}, End: {appointment_end}")
        except Exception as e:
            debug_print(f"ask_time_date: ❌ build_timeslot_range failed → {e}")
            session_data[call_sid]["retry_time"] = session_data[call_sid].get("retry_time", 0) + 1

            if session_data[call_sid]["retry_time"] >= 3:
                debug_print("ask_time_date: ⛔ Max retries reached during slot build.")
                resp.say(gpt_speak("Sorry, I couldn’t understand the time you mentioned. Please try again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            gather = make_gather(TIME_PROMPT_SHORT)
            resp.append(gather)
            return str(resp)

        # 3) Check the doctor’s calendar availability for this slot (per-doctor only)
        debug_print(f"ask_time_date: 👨‍⚕️ Checking calendar → {calendar_id}")

        slot_available = False
        try:
            # ✅ Use the stricter helper (FreeBusy + events fallback + padding + debug)
            slot_available = is_time_slot_available(calendar_id, appointment_start, appointment_end, creds)
            if slot_available:
                debug_print("ask_time_date: ✅ Slot free (first check) → proceed to customer lookup/confirmation")
        except Exception as e:
            debug_print(f"ask_time_date: ⚠️ Availability check error → {e}")
            slot_available = False

        if not slot_available:
            debug_print("ask_time_date: ❌ Slot not available")

            # Find nearby alternatives (friendly strings already included by helper).
            alts = []
            try:
                dur_minutes = int(globals().get("APPOINTMENT_DURATION_MINUTES", 30))
                alts = get_next_available_slots(
                    calendar_id,
                    creds,
                    from_start_iso=appointment_start,              # start searching from requested time
                    duration_minutes=dur_minutes,
                    limit=3,
                    tz_name="America/Chicago",                     # clinic tz
                    work_hours=((8,12),(13,17)),                   # adjust to your schedule
                    slot_step_minutes=30,
                    search_days=14
                ) or []
            except Exception as e:
                debug_print(f"ask_time_date: ⚠️ get_next_available_slots error → {e}")
                alts = []

            if alts:
                try:
                    options = " or ".join([slot.get("friendly") for slot in alts if slot.get("friendly")])
                except Exception as e:
                    debug_print(f"ask_time_date: ⚠️ options build error → {e}")
                    options = ""

                if options:
                    prompt = f"That time is not available. Would you like {options}?"
                    debug_print(f"ask_time_date: 💡 Offering alternatives → {options}")
                else:
                    prompt = "That time is not available. Please say another time, for example, 'today at 3:30 PM'."
                    debug_print("ask_time_date: ⚠️ Alternatives missing friendly labels")
            else:
                prompt = "That time is not available, and I couldn't find open slots soon. Please say another date and time."
                debug_print("ask_time_date: ⚠️ No alternatives found")

            gather = make_gather(prompt)
            resp.append(gather)
            return str(resp)

        # 4) Slot is available → decide whether to confirm or collect name details
        debug_print("ask_time_date: ✅ Slot free (per strict check) → proceed to customer lookup/confirmation")

        customer = session_data[call_sid].get("customer", {})
        customer_phone = (customer.get("phone") or "").strip()
        customer_dob   = (customer.get("dob") or "").strip()

        try:
            # If we have both phone & DOB and the customer exists → go straight to booking stage
            if customer_phone and customer_dob and customer_search(customer_phone, customer_dob):
                debug_print("ask_time_date: 📋 Customer on file — skip name collection")
                # 🚩 IMPORTANT: actually transition to book_appt_confirm so it can reserve Google Calendar.
                session_data[call_sid]["stage"] = "book_appt_confirm"
                session_data[call_sid]["auto_confirm"] = True  # optional: let confirm stage know it can auto-book
                debug_print("ask_time_date: ➡️ Redirecting to book_appt_confirm (auto_confirm=True)")
                resp.redirect("/voice")  # immediately re-enter handler; book_appt_confirm will execute now
                return str(resp)
            else:
                # Otherwise collect name (first → last → address → cc → confirm)
                debug_print("ask_time_date: 🆕 Customer not found → collecting first name")
                session_data[call_sid]["stage"] = "collect_first_name"
                gather = make_gather("Thanks. What is your first name?")
                resp.append(gather)
                return str(resp)
        except Exception as e:
            # If lookup errors out, fall back to collecting the name
            debug_print(f"ask_time_date: ⚠️ customer_search error → {e}")
            session_data[call_sid]["stage"] = "collect_first_name"
            gather = make_gather("Thanks. What is your first name?")
            resp.append(gather)
            return str(resp)






# ===== collect_first_name (stage) =====
    elif stage == "collect_first_name":
        # Capture first name and merge into existing customer bucket.
        first_name = (speech_result or "").strip()
        debug_print(f"collect_first_name: raw='{first_name}'")

        if not first_name or len(first_name.split()) > 2:
            gather = make_gather("I didn't catch that clearly. Please say your first name again.")
            resp.append(gather)
            return str(resp)

        session_data.setdefault(call_sid, {}).setdefault("customer", {})
        session_data[call_sid]["customer"]["first_name"] = first_name
        session_data[call_sid]["stage"] = "collect_last_name"

        gather = make_gather("Thank you. Now, what is your last name?")
        resp.append(gather)
        return str(resp)


# ===== collect_last_name (stage) =====
    elif stage == "collect_last_name":
        try:
            last = (speech_result or "").strip()
            debug_print(f"collect_last_name: raw='{last}'")

            if not last:
                gather = make_gather("Sorry, I didn't catch your last name. Please repeat it.")
                resp.append(gather)
                return str(resp)

            session_data.setdefault(call_sid, {}).setdefault("customer", {})
            session_data[call_sid]["customer"]["last_name"] = last
            session_data[call_sid]["stage"] = "collect_address"

            gather = make_gather("Got it. What is your full address, please?")
            resp.append(gather)
            return str(resp)

        except Exception as e:
            debug_print(f"collect_last_name: exception → {e}")
            resp.say(gpt_speak("Sorry, there was an error. Let's try again later."), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)



   


    elif stage == "collect_address":
        # ----------------------------------------------------------------------
        # 🏠 Stage: Collect Customer Address (INFO ONLY)
        # ----------------------------------------------------------------------

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
        #       (2) Expiration (MMYY or MMYYYY) → saved as 'MM/YY' (must be current/future)
        #       (3) CVV (3–4 digits)
        #   - Stores under session_data[call_sid]["customer"]:
        #       cc_number, cc_exp, cc_cvv, cc_name
        #   - Auto-advances to book_appt_confirm upon success.
        # Notes:
        #   - Uses make_gather() (speech + DTMF). DTMF preferred; speech digits supported.
        #   - Requires phone (10-digit) and DOB before collecting CC.
        #   - Logging is UNMASKED here per your request (not recommended for prod).
        # ----------------------------------------------------------------------

        # --- Luhn mod-10 -------------------------------------------------------
        def luhn_check(number: str) -> bool:
            s, alt = 0, False
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

        # --- Voice → digits (supports "double"/"triple", homophones) ----------
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

        # Ensure session buckets exist
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

        debug_print(f"collect_cc: 📍 step={cc_step}, DTMF='{dtmf_digits}', speech='{speech_text}'")

        # Prefer DTMF; otherwise convert spoken words to digits, then strip non-digits.
        # Use the module-level alias `re_mod` to avoid any closure/scoping issues.
        def get_digits() -> str:
            if dtmf_digits:
                return re_mod.sub(r"\D", "", dtmf_digits)
            return re_mod.sub(r"\D", "", normalize_spoken_digits(speech_text))

        # Re-prompt helper with retry cap
        def _reprompt(prompt: str, hints: str):
            session_data[call_sid]["retry_cc"] += 1
            if session_data[call_sid]["retry_cc"] >= 5:
                debug_print("collect_cc: ⛔ max CC retries. Ending.")
                resp.say(gpt_speak("Sorry, we’re having trouble collecting your details. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return True
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
            digits = (cc_partial + new_digits) if (cc_partial or new_digits) else ""

            # If nothing heard yet, reprompt right away
            if not digits:
                debug_print("collect_cc: ℹ️ no digits heard → reprompt")
                if _reprompt(
                    "Please enter your card number now, then press pound.",
                    hints="zero one two three four five six seven eight nine double triple"
                ): return str(resp)

            # Keep max length sane
            if len(digits) > 19:
                digits = digits[:19]

            # If we heard 15 digits via speech, ask for the last single digit
            if len(digits) == 15 and not dtmf_digits:
                session_data[call_sid]["cc_partial"] = digits
                debug_print(f"collect_cc: 🧩 Heard 15 digits '{digits}'; asking for the last digit")
                if _reprompt(
                    "I heard fifteen digits. Please say or type the last single digit now, then press pound.",
                    hints="zero one two three four five six seven eight nine"
                ): return str(resp)

            # Luhn validation
            if not (13 <= len(digits) <= 19) or not luhn_check(digits):
                session_data[call_sid]["cc_speech_tries"] += (0 if dtmf_digits else 1)
                escalate = session_data[call_sid]["cc_speech_tries"] >= 2 and not dtmf_digits
                debug_print(f"collect_cc: ❌ Invalid card number: '{digits}' (len={len(digits)}), escalate={escalate}")
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
            session_data[call_sid]["cc_partial"] = ""
            session_data[call_sid]["cc_speech_tries"] = 0
            debug_print(f"collect_cc: ✅ Saved card number '{digits}'")

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
        # Step 2: Expiration (MMYY/MMYYYY, must be current/future)
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
                debug_print(f"collect_cc: ❌ Invalid CVV '{digits}' length={len(digits)}")
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

            debug_print(f"collect_cc: ✅ Saved CVV '{digits}'; cc_name='{customer.get('cc_name')}'")

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





    

    elif stage == "book_appt_confirm":
        debug_print("book_appt_confirm: 📍 Stage entered")

        # ----------------------------------------------------------------------
        # Doctor info
        # ----------------------------------------------------------------------
        doctor_id = session_data[call_sid].get("doctor_id")
        doctor_name = googleid_dr_name_map.get(doctor_id, "the doctor")
        debug_print(f"book_appt_confirm: doctor_id={doctor_id} name={doctor_name}")

        # ----------------------------------------------------------------------
        # Appointment time (need start; compute end if missing)
        # ----------------------------------------------------------------------
        appt_payload = session_data[call_sid].get("appointment_time", {}) or {}
        appointment_start = appt_payload.get("start")
        appointment_end   = appt_payload.get("end")
        debug_print(f"book_appt_confirm: utc_start={appointment_start} utc_end={appointment_end}")

        if not appointment_start:
            debug_print("book_appt_confirm: ❌ missing appointment_start")
            resp.say(gpt_speak("Appointment time missing. Goodbye!"), VOICE)
            resp.hangup()
            return str(resp)

        # Human-friendly local time for voice/SMS (America/Chicago)
        formatted_time = ""
        try:
            
            dt_utc = datetime.fromisoformat(appointment_start.replace("Z", "+00:00"))
            dt_local = dt_utc.astimezone(pytz.timezone("America/Chicago"))
            try:
                formatted_time = dt_local.strftime("%A, %B %-d at %-I:%M %p")
            except Exception:
                # Windows-compatible fallback (remove leading zero from day/hour)
                formatted_time = dt_local.strftime("%A, %B %d at %I:%M %p").replace(" 0", " ")
        except Exception as e:
            debug_print(f"book_appt_confirm: time parse/format error → {e}")
            resp.say(gpt_speak("Sorry, we couldn't confirm the appointment time."), VOICE)
            resp.hangup()
            return str(resp)

        # Compute end if missing (default duration = 30m)
        if not appointment_end:
            try:
                dur_min = int(APPOINTMENT_DURATION_MINUTES) if 'APPOINTMENT_DURATION_MINUTES' in globals() else 30
                end_dt  = dt_utc + timedelta(minutes=dur_min)
                
                appointment_end = end_dt.replace(tzinfo=_pytz.UTC).isoformat()
                debug_print(f"book_appt_confirm: computed utc_end={appointment_end} (duration={dur_min}m)")
            except Exception as e:
                debug_print(f"book_appt_confirm: ❌ failed computing end time → {e}")
                resp.say(gpt_speak("Sorry, we couldn't confirm the appointment time."), VOICE)
                resp.hangup()
                return str(resp)

        # ----------------------------------------------------------------------
        # Customer info (basic validation)
        # ----------------------------------------------------------------------
        customer = session_data[call_sid].get("customer", {}) or {}
        customer_name    = (customer.get("name") or "").strip()
        customer_phone   = (customer.get("phone") or "").strip()
        customer_dob     = (customer.get("dob") or "").strip()
        customer_address = (customer.get("address") or "").strip()

        # Derive name parts if only full name present
        first_name = (customer.get("first_name") or "").strip()
        last_name  = (customer.get("last_name")  or "").strip()
        if not first_name and customer_name:
            parts = customer_name.split()
            first_name = parts[0]
            last_name  = " ".join(parts[1:]) if len(parts) > 1 else ""
        effective_name = customer_name or " ".join([n for n in [first_name, last_name] if n]).strip()

        # Normalize phone to 10-digit (required)
        def _normalize_phone10(d):
            d = "".join(ch for ch in (d or "") if ch.isdigit())
            return d[1:] if len(d) == 11 and d.startswith("1") else d
        phone10 = _normalize_phone10(customer_phone)
        if len(phone10) != 10:
            debug_print("book_appt_confirm: ❌ invalid/missing phone; redirecting to collect_phone")
            session_data[call_sid]["stage"] = "collect_phone"
            gather = make_gather("Before we confirm your appointment, I need your ten digit phone number including area code. You can say it or type the digits and press pound.")
            resp.append(gather)
            return str(resp)
        if not customer_dob:
            debug_print("book_appt_confirm: ❌ missing DOB; redirecting to collect_dob")
            session_data[call_sid]["stage"] = "collect_dob"
            gather = make_gather("Before we confirm, please provide your date of birth. You can say it, or enter two digits for month, two for day, and four for year, then press pound.")
            resp.append(gather)
            return str(resp)

        # ----------------------------------------------------------------------
        # Upsert customer (FIX: pass cc_* fields to satisfy function signature)
        # ----------------------------------------------------------------------
        try:
            init_db()
            cc_name   = (customer.get("cc_name") or effective_name or "")
            cc_number = (customer.get("cc_number") or "")
            cc_exp    = (customer.get("cc_exp") or "")
            cc_cvv    = (customer.get("cc_cvv") or "")
            inserted = insert_customer(
                phone=phone10,
                dob=customer_dob,
                first_name=first_name,
                last_name=last_name,
                address=customer_address,
                cc_name=cc_name,
                cc_number=cc_number,
                cc_exp=cc_exp,
                cc_cvv=cc_cvv
            )
            debug_print(f"book_appt_confirm: customers DB → {'inserted' if inserted else 'seen/updated'}")
        except Exception as e:
            debug_print(f"book_appt_confirm: insert_customer failed → {e}")

        # ----------------------------------------------------------------------
        # Single availability check (simple)
        # ----------------------------------------------------------------------
        calendar_id = doctor_id
        debug_print(f"book_appt_confirm: 🔎 availability check → cal={calendar_id} {appointment_start}→{appointment_end}")
        try:
            slot_free = is_time_slot_available(calendar_id, appointment_start, appointment_end, creds)
        except Exception as e:
            debug_print(f"book_appt_confirm: ⚠️ availability check error → {e}")
            slot_free = False  # be safe

        if not slot_free:
            debug_print("book_appt_confirm: ❌ Slot not free → offering alternatives or reprompting")
            try:
                # Keep it simple: offer up to 3 suggestions if available
                alts = get_next_available_slots(calendar_id, creds, limit=3) or []
                if alts:
                    options = " or ".join([a.get("friendly","") for a in alts if a.get("friendly")])
                    session_data[call_sid]["stage"] = "ask_time_date"
                    gather = make_gather(f"That time is not available. Would you like {options}?")
                    resp.append(gather)
                    return str(resp)
            except Exception as e:
                debug_print(f"book_appt_confirm: ⚠️ get_next_available_slots error → {e}")

            session_data[call_sid]["stage"] = "ask_time_date"
            gather = make_gather("That time is not available. Please say another date and time, for example, 'tomorrow at 3:30 PM'.")
            resp.append(gather)
            return str(resp)

        # ----------------------------------------------------------------------
        # Save appointment in your system (simple, no rollback)
        # ----------------------------------------------------------------------
        appointment_saved_internal = False
        try:
            confirm_appointment_by_name(
                doctor_name=doctor_name,
                phone=phone10,
                dob=customer_dob,
                name=effective_name,
                address=customer_address,
                utc_start=appointment_start,
                calendar_id=calendar_id
            )
            appointment_saved_internal = True
            debug_print("book_appt_confirm: ✅ internal appointment saved")
        except Exception as e:
            debug_print(f"book_appt_confirm: ❌ internal save failed → {e}")

        # ----------------------------------------------------------------------
        # Create Google Calendar event (simple: let Google assign the event ID)
        # ----------------------------------------------------------------------
        try:
            service = build("calendar", "v3", credentials=creds)

            event_body = {
                "summary": f"Appointment: {doctor_name}",
                "description": f"Clinic appointment for {effective_name or 'patient'}.",
                "start": {"dateTime": appointment_start, "timeZone": "UTC"},
                "end":   {"dateTime": appointment_end,   "timeZone": "UTC"},
                "transparency": "opaque",  # must block time
                "extendedProperties": {
                    "private": {
                        "patient_name": effective_name,
                        "phone10": phone10,
                        "dob": customer_dob,
                        "call_sid": call_sid,
                    }
                },
            }

            debug_print(f"book_appt_confirm: 📝 creating Google event cal={calendar_id} {appointment_start}→{appointment_end}")
            ev = service.events().insert(calendarId=calendar_id, body=event_body, sendUpdates="none").execute()
            google_event_id = ev.get("id")
            google_event_link = ev.get("htmlLink")
            debug_print(f"book_appt_confirm: ✅ Google event created id={google_event_id} link={google_event_link}")
        except Exception as e:
            debug_print(f"book_appt_confirm: ❌ Google Calendar insert failed → {e}")
            session_data[call_sid]["stage"] = "ask_time_date"
            gather = make_gather("Sorry, I couldn't confirm that slot. Please say a new date and time, for example, August 14th at 10 AM.")
            resp.append(gather)
            return str(resp)

        # ----------------------------------------------------------------------
        # Voice confirmation (success path)
        # ----------------------------------------------------------------------
        msg = f"Your appointment with {doctor_name} has been booked"
        if formatted_time:
            msg += f" on {formatted_time}"
        msg += ". We look forward to seeing you. Goodbye!"
        debug_print("book_appt_confirm: 🎉 success → speaking confirmation")
        resp.say(gpt_speak(msg), VOICE)

        # ----------------------------------------------------------------------
        # SMS confirmation (best-effort)
        # ----------------------------------------------------------------------
        try:
            e164 = f"+1{phone10}"
            sms_text = f"Hi {(effective_name or 'there')}, your appointment with {doctor_name} is confirmed"
            if formatted_time:
                sms_text += f" on {formatted_time}"
            sms_text += ". Thank you for choosing Epic Therapist Clinic."
            message = client.messages.create(body=sms_text, from_=TWILIO_PHONE_NUMBER, to=e164)
            debug_print(f"book_appt_confirm: 📩 SMS sent to {e164}, SID={message.sid}")
        except Exception as e:
            debug_print(f"book_appt_confirm: ⚠️ SMS failed → {e}")

        # ----------------------------------------------------------------------
        # Cleanup
        # ----------------------------------------------------------------------
        resp.hangup()
        session_data.pop(call_sid, None)
        debug_print("book_appt_confirm: ✅ session cleared and call ended")
        return str(resp)





    elif stage == "cancel_appointment":
        # ----------------------------------------------------------------------
        # 🔄 Stage: Cancel Appointment — after the caller says the doctor’s name
        #  1) Try direct partial match against known doctors.
        #  2) If no match, try GPT-based extraction.
        #  3) If still no match, re-prompt (with retry cap).
        #  4) On match, move to phone collection.
        # ----------------------------------------------------------------------
        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("cancel", {})

        # Safe punctuation list even if 'string' isn't imported globally
        try:
            import string as _string
            _PUNCT = _string.punctuation
        except Exception:
            _PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

        def _clean(s: str) -> str:
            """lowercase + strip punctuation + trim spaces"""
            return (s or "").lower().translate(str.maketrans("", "", _PUNCT)).strip()

        selected_text = (speech_result or "").strip()

        # Nothing heard → re-prompt (no next_stage arg)
        if not selected_text:
            debug_print("cancel_appointment: ⚠️ No speech detected — re-prompting for doctor name.")
            doctor_list = ", ".join(googleid_dr_name_map.values())
            retry_prompt = (
                f"I can't hear you. Available doctors are: {doctor_list}. "
                "Please say the name of the doctor whose appointment you want to cancel."
            )
            session_data[call_sid]["stage"] = "cancel_appointment"
            resp.append(make_gather(retry_prompt))
            return str(resp)

        selected_clean = _clean(selected_text)
        debug_print(f"cancel_appointment: 🗣️ Received doctor name → '{selected_clean}'")

        matched_id = None
        matched_name = None

        # 1) Partial substring match
        partial_matches = []
        for doc_id, friendly_name in googleid_dr_name_map.items():
            friendly_clean = _clean(friendly_name)
            if selected_clean in friendly_clean or friendly_clean in selected_clean:
                partial_matches.append((doc_id, friendly_name))

        if len(partial_matches) == 1:
            matched_id, matched_name = partial_matches[0]
            debug_print(f"cancel_appointment: ✅ Partial match → {matched_name} ({matched_id})")

        # 2) GPT fallback
        if not matched_id:
            try:
                extracted_name = extract_doctor_name(selected_text)
                debug_print(f"cancel_appointment: 🤖 GPT extracted name → '{extracted_name}'")
                if extracted_name:
                    extracted_clean = _clean(extracted_name)
                    for doc_id, friendly_name in googleid_dr_name_map.items():
                        friendly_clean = _clean(friendly_name)
                        if extracted_clean in friendly_clean or friendly_clean in extracted_clean:
                            matched_id, matched_name = doc_id, friendly_name
                            debug_print(f"cancel_appointment: ✅ GPT matched → {matched_name} ({matched_id})")
                            break
            except Exception as e:
                debug_print(f"cancel_appointment: ⚠️ GPT fallback error → {e}")

        # 3) Still no match → retry with cap
        if not matched_id:
            retries = session_data[call_sid].get("retry_booking", 0)
            session_data[call_sid]["retry_booking"] = retries + 1
            max_retries = globals().get("MAX_NUMBER_DR_RETRY", 3)

            if retries >= max_retries:
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
            session_data[call_sid]["stage"] = "cancel_appointment"
            resp.append(make_gather(retry_prompt))
            return str(resp)

        # 4) Proceed with matched doctor → next stage: phone number
        session_data[call_sid]["doctor_id"] = matched_id
        session_data[call_sid]["cancel"]["doctor"] = matched_name or googleid_dr_name_map.get(matched_id, "the doctor")
        session_data[call_sid]["stage"] = "cancel_appt_by_phone_number"

        resp.append(make_gather(
            "Thanks. What phone number did you use when booking the appointment?"
        ))
        return str(resp)


    elif stage == "cancel_appt_by_phone_number":
        # ----------------------------------------------------------------------
        # 📌 Collect the phone number used when booking, then move to date+time.
        # ----------------------------------------------------------------------
        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("cancel", {})

        phone_raw = (speech_result or "").strip()
        phone = extract_phone_number(phone_raw)
        debug_print(f"cancel_appt_by_phone_number: 📱 Extracted phone → '{phone}' (raw='{phone_raw}')")

        def _normalize_10(d: str) -> str:
            d = "".join(ch for ch in (d or "") if ch.isdigit())
            return d[1:] if len(d) == 11 and d.startswith("1") else d

        phone10 = _normalize_10(phone)

        if not phone10 or len(phone10) < 7:
            debug_print("cancel_appt_by_phone_number: ❌ invalid/missing phone → reprompt")
            session_data[call_sid]["stage"] = "cancel_appt_by_phone_number"
            resp.append(make_gather("I didn’t catch your phone number. Please say it again clearly, including the area code."))
            return str(resp)

        session_data[call_sid]["cancel"]["phone"] = phone10
        session_data[call_sid]["stage"] = "cancel_appt_by_date_time"

        resp.append(make_gather(
            "Thanks. Now, please tell me the date and time of the appointment you want to cancel. "
            "For example, say July 3rd at 9 AM."
        ))
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
        #   - Sets stage -> "cancel_appt_get_date_time" (we’ll ask for the appt’s date+time next).
        # Resilience:
        #   - 3 retries on parse or validation errors with polite reprompts.
        #   - Consistent "can't hear you" behavior via make_gather_dob() when available.
        # ----------------------------------------------------------------------
        debug_print("cancel_appt_get_dob: 📍 Stage entered")

        # Ensure buckets exist
        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("customer", {})

        # Local import guard for date
        try:
            from datetime import date
        except Exception:
            # Fallback: simple shim if import ever fails (highly unlikely)
            debug_print("cancel_appt_get_dob: ⚠️ could not import 'date' from datetime")
            class _FakeDate:  # minimal stub to avoid NameError
                @staticmethod
                def today(): return None
            date = _FakeDate  # type: ignore

        # --- Guard: require 10-digit phone first ---------------------------------
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

        # --- Pull inputs (DTMF preferred if provided by Twilio) -------------------
        try:
            dtmf_digits = (request.values.get("Digits") or "").strip()
        except Exception:
            dtmf_digits = ""
        speech_text = (speech_result or "").strip()
        debug_print(f"cancel_appt_get_dob: 🎙️ speech_text='{speech_text}', 🔢 dtmf_digits='{dtmf_digits}'")

        # --- Attempt to parse DOB (speech or keypad) ------------------------------
        # parse_dob_input should return a datetime (or None on failure)
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
                "Please say birth date, for example July third nineteen fifty six, "
                "or type MMDDYYYY, then press pound."
            )
            try:
                gather = make_gather_dob(prompt_text)  # if you have a specialized DOB gather
            except Exception:
                gather = make_gather(prompt_text, hints="zero one two three four five six seven eight nine")
            resp.append(gather)
            return str(resp)

        # --- Validate reasonable DOB range ----------------------------------------
        try:
            today = date.today()
            min_date = date(1900, 1, 1)
            dob_date = dt.date()
            if not (min_date <= dob_date <= today):
                session_data[call_sid]["retry_cancel_dob"] = session_data[call_sid].get("retry_cancel_dob", 0) + 1
                r = session_data[call_sid]["retry_cancel_dob"]
                debug_print(f"cancel_appt_get_dob: ⚠️ DOB out of range → {dob_date.isoformat()} Retry={r}")

                if r >= 3:
                    resp.say(gpt_speak("Sorry, that birth date still doesn’t look valid. Please call again later."), VOICE)
                    resp.hangup()
                    session_data.pop(call_sid, None)
                    return str(resp)

                prompt_text = (
                    "That doesn't sound like a valid birth date. Please say it again, "
                    "or type two digits for month, two for day, and four for year, then press pound. "
                    "For example, 07031956#."
                )
                try:
                    gather = make_gather_dob(prompt_text)
                except Exception:
                    gather = make_gather(prompt_text, hints="zero one two three four five six seven eight nine")
                resp.append(gather)
                return str(resp)
        except Exception as e:
            debug_print(f"cancel_appt_get_dob: ⚠️ Validation error → {e}")

        # --- Store ISO DOB and advance ----------------------------





    elif stage == "cancel_appt_by_time_date":
        # ----------------------------------------------------------------------
        # ❌ Stage: cancel_appt_by_time_date
        #
        # Goal:
        #   - Ask for (or receive) a spoken date+time and check the selected doctor's
        #     calendar using is_time_slot_available:
        #       • If slot is FREE  → there is no appointment at that time → go to iterate.
        #       • If slot is BUSY → there is an event → fetch it and go to confirm.
        #
        # Notes:
        #   - Uses smart_parse_time + build_timeslot_range to build UTC start/end.
        #   - Requires a chosen doctor (calendar_id) and a 10-digit phone (to prefer
        #     matching the correct patient when multiple overlaps exist).
        #   - We FIRST call is_time_slot_available (vendor-agnostic check).
        #     If it returns False (busy), we then list overlapping events to get event ID.
        #   - No use of next_stage; we set session_data[call_sid]["stage"] explicitly.
        # ----------------------------------------------------------------------
        debug_print("cancel_appt_by_time_date: 📍 Stage entered")

        # Ensure buckets
        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("cancel", {})
        cancel_ctx = session_data[call_sid]["cancel"]

        # --- Require selected doctor (calendar_id) --------------------------------
        calendar_id = cancel_ctx.get("calendar_id") or session_data[call_sid].get("doctor_id")
        if not calendar_id:
            debug_print("cancel_appt_by_time_date: ❌ no calendar_id — returning to cancel_appointment (choose doctor)")
            session_data[call_sid]["stage"] = "cancel_appointment"
            resp.append(make_gather("Which doctor's appointment would you like to cancel?"))
            return str(resp)

        # --- Require phone (10-digit) — helps disambiguate overlapping events -----
        def _normalize_10(d: str) -> str:
            d = "".join(ch for ch in (d or "") if ch.isdigit())
            return d[1:] if len(d) == 11 and d.startswith("1") else d

        phone_norm = _normalize_10(cancel_ctx.get("phone") or session_data[call_sid].get("customer", {}).get("phone"))
        if len(phone_norm) != 10:
            debug_print("cancel_appt_by_time_date: ❌ phone missing/invalid → collect_phone first")
            session_data[call_sid]["return_stage"] = "cancel_appt_by_time_date"
            session_data[call_sid]["stage"] = "collect_phone"
            resp.append(make_gather(
                "To locate your appointment, please provide your ten digit phone number including area code. "
                "You can say it, or type the digits and press pound.",
                hints="zero one two three four five six seven eight nine"
            ))
            return str(resp)

        cancel_ctx["phone"] = phone_norm

        # --- Get utterance and parse date+time ------------------------------------
        utter = (speech_result or "").strip()
        debug_print(f"cancel_appt_by_time_date: 🗣️ Raw speech → '{utter}'")

        if not utter:
            debug_print("cancel_appt_by_time_date: 🚫 No date/time spoken → iterate flow")
            cancel_ctx["iter_index"] = 0
            session_data[call_sid]["stage"] = "cancel_appt_iterate"
            resp.append(make_gather("Okay, I’ll list your upcoming appointments."))
            return str(resp)

        time_info = smart_parse_time(utter)
        debug_print(f"cancel_appt_by_time_date: 🧠 smart_parse_time → {time_info}")

        if not time_info or not isinstance(time_info, tuple) or len(time_info) != 2:
            debug_print("cancel_appt_by_time_date: ❌ Could not parse date/time → iterate flow")
            cancel_ctx["iter_index"] = 0
            session_data[call_sid]["stage"] = "cancel_appt_iterate"
            resp.append(make_gather("I couldn’t understand the date and time. I’ll list your upcoming appointments."))
            return str(resp)

        spoken_day, spoken_time = time_info
        debug_print(f"cancel_appt_by_time_date: 📆 Parsed → Day='{spoken_day}', Time='{spoken_time}'")

        # --- Make UTC window from spoken day/time ---------------------------------
        try:
            appointment_start, appointment_end = build_timeslot_range(spoken_day, spoken_time)
            cancel_ctx["utc_start"] = appointment_start
            cancel_ctx["utc_end"]   = appointment_end
            debug_print(f"cancel_appt_by_time_date: ⏰ UTC window → {appointment_start} → {appointment_end}")
        except Exception as e:
            debug_print(f"cancel_appt_by_time_date: ❌ build_timeslot_range failed → {e} → iterate flow")
            cancel_ctx["iter_index"] = 0
            session_data[call_sid]["stage"] = "cancel_appt_iterate"
            resp.append(make_gather("That didn’t look like a valid date and time. I’ll list your upcoming appointments."))
            return str(resp)

        # --- Use your strict availability checker (vendor-agnostic) ---------------
        # IMPORTANT: For cancel we invert the meaning:
        #   • is_time_slot_available == True  → slot is FREE  → nothing to cancel.
        #   • is_time_slot_available == False → slot is BUSY → there is an event to cancel.
        try:
            slot_free = is_time_slot_available(calendar_id, appointment_start, appointment_end, creds)
            debug_print(f"cancel_appt_by_time_date: 🔎 is_time_slot_available → {slot_free}")
        except Exception as e:
            debug_print(f"cancel_appt_by_time_date: ⚠️ availability check error → {e}")
            slot_free = True  # fail-open to avoid accidental delete; treat as "not found"

        if slot_free:
            # No event at that time → nothing to cancel; offer iterate path
            debug_print("cancel_appt_by_time_date: 🚫 Slot FREE → no appointment at that time → iterate")
            cancel_ctx["iter_index"] = 0
            session_data[call_sid]["stage"] = "cancel_appt_iterate"
            resp.append(make_gather("I didn’t find an appointment at that time. I’ll list your upcoming appointments."))
            return str(resp)

        # --- Slot is BUSY → fetch overlapping event(s) to get event ID ------------
        try:
            from googleapiclient.discovery import build
            from dateutil.parser import isoparse
            from datetime import timedelta

            service = build("calendar", "v3", credentials=creds)

            # Small padding to catch edge-inclusive overlaps (same as availability helper)
            tmin = (isoparse(appointment_start) - timedelta(seconds=60)).isoformat()
            tmax = (isoparse(appointment_end)   + timedelta(seconds=60)).isoformat()

            items = service.events().list(
                calendarId=calendar_id,
                timeMin=tmin,
                timeMax=tmax,
                singleEvents=True,
                showDeleted=False,
                orderBy="startTime",
                maxResults=250,
            ).execute().get("items", [])

            debug_print(f"cancel_appt_by_time_date: 📄 events().list returned {len(items)} item(s) in padded window")

            # Select the first overlapping, opaque event; prefer one that matches caller phone
            def _overlaps(ev, s, e):
                try:
                    es = isoparse(ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date"))
                    ee = isoparse(ev.get("end",   {}).get("dateTime") or ev.get("end",   {}).get("date"))
                    return s < ee and e > es
                except Exception:
                    return False

            sdt = isoparse(appointment_start)
            edt = isoparse(appointment_end)

            candidates = []
            for ev in items:
                if ev.get("status") == "cancelled": 
                    continue
                if ev.get("transparency") == "transparent":
                    continue
                if not _overlaps(ev, sdt, edt):
                    continue
                candidates.append(ev)

            debug_print(f"cancel_appt_by_time_date: 🔎 overlapping, opaque events → {len(candidates)}")

            # Prefer the event whose extendedProperties.private.phone10 matches the caller
            chosen = None
            for ev in candidates:
                try:
                    priv = (ev.get("extendedProperties", {}) or {}).get("private", {}) or {}
                    if (priv.get("phone10") or "").strip() == phone_norm:
                        chosen = ev
                        break
                except Exception:
                    pass
            if not chosen and candidates:
                chosen = candidates[0]

            if not chosen:
                debug_print("cancel_appt_by_time_date: ⚠️ busy per FreeBusy but no overlapping event found → iterate")
                cancel_ctx["iter_index"] = 0
                session_data[call_sid]["stage"] = "cancel_appt_iterate"
                resp.append(make_gather("I couldn’t find the event details. I’ll list your upcoming appointments instead."))
                return str(resp)

            # Persist chosen event for the confirm stage
            cancel_ctx["calendar_id"]    = calendar_id
            cancel_ctx["matching_event"] = {
                "id": chosen.get("id"),
                "summary": chosen.get("summary"),
                "start": chosen.get("start"),
                "end": chosen.get("end"),
                "htmlLink": chosen.get("htmlLink"),
            }
            debug_print(f"cancel_appt_by_time_date: ✅ matched event id={chosen.get('id')} summary='{chosen.get('summary','')}'")

            # Go to confirmation
            session_data[call_sid]["stage"] = "cancel_appt_confirm"
            # Friendly echo
            friendly = cancel_ctx.get("day") and cancel_ctx.get("time")
            prompt = "I found that appointment. Would you like me to cancel it now?"
            if friendly:
                prompt = f"I found your appointment on {cancel_ctx['day']} at {cancel_ctx['time']}. Shall I cancel it now?"
            resp.append(make_gather(prompt))
            return str(resp)

        except Exception as e:
            debug_print(f"cancel_appt_by_time_date: ❌ error retrieving overlapping event → {e}")
            cancel_ctx["iter_index"] = 0
            session_data[call_sid]["stage"] = "cancel_appt_iterate"
            resp.append(make_gather("I couldn’t look up the event details. I’ll list your upcoming appointments instead."))
            return str(resp)






    elif stage == "cancel_appt_iterate":
        # ----------------------------------------------------------------------
        # 🗂️ Stage: cancel_appt_iterate
        # Purpose:
        #   - Build/refresh a local (file-based) candidate list of this caller's
        #     appointments using the doctor JSON files (NOT Google Calendar).
        #   - Present one candidate at a time and ask: "Cancel this one?"
        #   - "Yes" (or '1') → store as matching_event and jump to cancel_appt_confirm
        #   - "No"  (or '2') → advance to the next candidate; if none left → apologize
        #
        # Inputs expected in session_data[call_sid]["cancel"]:
        #   {
        #     "phone": "4694633276",             # REQUIRED (normalized or not; we normalize here)
        #     "dob": "YYYY-MM-DD" or "",         # OPTIONAL (if provided we filter by it)
        #     "doctor": "Alfred hitchcock",      # OPTIONAL; if missing we search ALL doctors
        #     "candidates": [ ... ],             # OPTIONAL; built on first entry
        #     "iter_index": 0                    # OPTIONAL; current position in candidates
        #   }
        #
        # Notes:
        #   - Uses local helpers: get_doctor_appts_for / build_doctor_appt_index
        #   - Produces normalized "candidate" dicts compatible with cancel_appt_confirm:
        #       {
        #         "doctor_name": str,
        #         "start_utc":  str,     # ISO-8601 UTC (as stored in file; assumed UTC)
        #         "end_utc":    str,     # optional/blank if not present
        #         "friendly":   str,     # e.g., "Tuesday, August 12 at 9:00 AM"
        #         "phone":      str,     # 10-digit normalized
        #         "dob":        str      # ISO DOB if present
        #       }
        # ----------------------------------------------------------------------

        # ---------- tiny helpers ----------
        def _norm_phone_10(s: str) -> str:
            d = "".join(ch for ch in (s or "") if ch.isdigit())
            return d[1:] if len(d) == 11 and d.startswith("1") else d

        def _friendly_from_iso(utc_iso: str, tz_name: str = "America/Chicago") -> str:
            """Render a UTC ISO string into a caller-friendly local phrase."""
            try:
                dt_utc = dtparser.isoparse(utc_iso)
                local = dt_utc.astimezone(pytz.timezone(tz_name))
                try:
                    return local.strftime("%A, %B %-d at %-I:%M %p")
                except Exception:
                    return local.strftime("%A, %B %d at %I:%M %p").replace(" 0", " ")
            except Exception:
                return utc_iso or "the specified time"

        def _appt_to_candidate(appt: dict, doctor_name: str, phone10: str, dob_iso: str) -> dict:
            """
            Map a raw appointment record from the doctor's JSON file into a
            normalized candidate dict the iterate/confirm stages expect.
            Tries 'start' then 'time' for the UTC start; 'end' optional.
            """
            start_utc = (appt.get("start") or appt.get("time") or "").strip()
            end_utc   = (appt.get("end") or "").strip()
            friendly  = _friendly_from_iso(start_utc)
            return {
                "doctor_name": doctor_name,
                "start_utc": start_utc,
                "end_utc": end_utc,
                "friendly": friendly,
                "phone": phone10,
                "dob": (dob_iso or "").strip(),
                # keep a breadcrumb if you like:
                # "raw": appt
            }

        # ---------- ensure cancel context ----------
        cancel_ctx = session_data[call_sid].setdefault("cancel", {})
        phone_in = cancel_ctx.get("phone", "")
        dob_in   = cancel_ctx.get("dob", "") or ""     # DOB already verified in cancel_appt_get_dob
        doctor   = (cancel_ctx.get("doctor") or "").strip()

        phone10 = _norm_phone_10(phone_in)
        if len(phone10) != 10:
            debug_print("cancel_appt_iterate: ❌ missing/invalid phone → route back to cancel_appt_get_dob")
            session_data[call_sid]["stage"] = "cancel_appt_get_dob"
            gather = make_gather(
                "Before we cancel, please confirm your date of birth. "
                "You can say it, for example July third nineteen ninety, or enter month, day, and year, then press pound."
            )
            resp.append(gather)
            return str(resp)

        # ---------- Build candidates on first entry ----------
        if not cancel_ctx.get("candidates"):
            debug_print("cancel_appt_iterate: 🔎 building candidates from local doctor JSON")

            candidates = []

            # If doctor specified → only search that one
            if doctor:
                appts = get_doctor_appts_for(doctor, phone10, dob_in or None)
                debug_print(f"cancel_appt_iterate: doctor='{doctor}' → {len(appts)} appt(s) for caller")
                for ap in appts:
                    candidates.append(_appt_to_candidate(ap, doctor, phone10, dob_in))
            else:
                # No specific doctor → search ALL known doctors (from your global map)
                try:
                    doctor_map = googleid_dr_name_map  # {calendar_id: "Doctor Friendly Name"}
                except NameError:
                    doctor_map = {}
                # If you maintain a separate list of doctor display names, use it here.
                # We'll derive a set of unique friendly names from the map's values.
                doctor_names = sorted(set(doctor_map.values())) if doctor_map else []

                # Fallback: if you keep doctor files elsewhere, you could also list the folder.
                # For now rely on map values.
                for dr_name in doctor_names:
                    appts = get_doctor_appts_for(dr_name, phone10, dob_in or None)
                    if appts:
                        debug_print(f"cancel_appt_iterate: doctor='{dr_name}' → {len(appts)} appt(s)")
                    for ap in appts:
                        candidates.append(_appt_to_candidate(ap, dr_name, phone10, dob_in))

            # Sort candidates chronologically by start_utc (if present)
            try:
                candidates.sort(
                    key=lambda c: (c["start_utc"] == "", dtparser.isoparse(c["start_utc"]) if c["start_utc"] else None)
                )
            except Exception:
                pass

            cancel_ctx["candidates"] = candidates
            cancel_ctx["iter_index"] = 0

            debug_print(f"cancel_appt_iterate: ✅ built {len(candidates)} candidate(s)")

            # If none found → apologize and optionally offer reschedule or end.
            if not candidates:
                debug_print("cancel_appt_iterate: 🚫 no appointments found for caller")
                resp.say(gpt_speak("I couldn't find any upcoming appointments under that phone number."), VOICE)
                # If you want to offer rescheduling here, flip a flag and route to booking:
                if cancel_ctx.get("reschedule_after_cancel"):
                    session_data[call_sid]["stage"] = "booking"
                    doctor_list_str = ", ".join(googleid_dr_name_map.values())
                    gather = make_gather("Would you like to book a new appointment? Please say the doctor's name.", hints=doctor_list_str)
                    resp.append(gather)
                    return str(resp)
                # Otherwise end the call politely.
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

        # ---------- We have candidates; interpret the user's answer ----------
        # Pull raw inputs
        try:
            dtmf = (request.values.get("Digits") or "").strip()
        except Exception:
            dtmf = ""
        utter = (speech_result or "").strip().lower()

        debug_print(f"cancel_appt_iterate: 🎚️ user input → dtmf='{dtmf}' speech='{utter}'")

        # Simple intent: yes/no/next/back using speech OR DTMF
        YES = {"yes", "yeah", "yep", "correct", "confirm", "one", "1"}
        NO  = {"no", "nope", "next", "two", "2"}
        REP = {"repeat", "say again", "again"}
        BAK = {"back", "previous", "go back", "3"}

        def _is(intent_set):
            return (utter in intent_set) or (dtmf in {s for s in intent_set if s.isdigit()})

        # Current index & candidate
        idx = cancel_ctx.get("iter_index", 0)
        cands = cancel_ctx.get("candidates", [])
        total = len(cands)

        # Guard out-of-range
        if idx >= total:
            debug_print("cancel_appt_iterate: ⚠️ iterator exhausted unexpectedly; resetting to 0")
            idx = 0
            cancel_ctx["iter_index"] = 0

        cand = cands[idx] if cands else None

        # If we received a clear "yes"
        if _is(YES) and cand:
            debug_print(f"cancel_appt_iterate: ✅ user confirmed candidate #{idx+1}/{total}")
            # Stash as the event to cancel and jump to confirm stage
            cancel_ctx["matching_event"] = cand
            session_data[call_sid]["stage"] = "cancel_appt_confirm"
            # Go run the confirmation stage immediately
            return voice()

        # If "no/next"
        if _is(NO):
            debug_print(f"cancel_appt_iterate: ↪️ user skipped candidate #{idx+1}/{total}")
            idx += 1
            if idx >= total:
                # No more candidates → apologize (or route to booking)
                debug_print("cancel_appt_iterate: 🚫 no more candidates")
                resp.say(gpt_speak("That's all I found under your details. I couldn't find a matching appointment to cancel."), VOICE)
                if cancel_ctx.get("reschedule_after_cancel"):
                    session_data[call_sid]["stage"] = "booking"
                    doctor_list_str = ", ".join(googleid_dr_name_map.values())
                    gather = make_gather("Would you like to book a new appointment? Please say the doctor's name.", hints=doctor_list_str)
                    resp.append(gather)
                    return str(resp)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)
            # Advance to next candidate
            cancel_ctx["iter_index"] = idx
            cand = cands[idx]

        # If "back"
        if _is(BAK) and total > 0:
            idx = max(0, idx - 1)
            cancel_ctx["iter_index"] = idx
            cand = cands[idx]
            debug_print(f"cancel_appt_iterate: ⬅️ moved back to candidate #{idx+1}/{total}")

        # If "repeat" or anything else or just first display → (re)prompt this candidate
        if not cand:
            # Defensive: should not happen because we handle empty list above
            debug_print("cancel_appt_iterate: ❓ no candidate to present; ending.")
            resp.say(gpt_speak("I'm sorry, I couldn't find any appointment to review."), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)

        # Speak out the current candidate nicely and ask to confirm
        say_line = f"I found an appointment with {cand['doctor_name']} on {cand['friendly']}. " \
                f"Do you want to cancel this one? Say yes or no. You can also press 1 for yes, or 2 for no."
        debug_print(f"cancel_appt_iterate: 🗣️ prompting candidate #{idx+1}/{total} → {say_line}")

        # Use your standard gather (speech + DTMF; finish on '#')
        gather = make_gather(say_line, hints="yes no one two back repeat previous")
        resp.append(gather)
        return str(resp)



# ----------------------------------------------------------------------
        # 📌 Stage: cancel_appt_confirm
        #
        # What this does now (updated):
        #   • Always attempts LOCAL cancellation first via:
        #       cancel_appointment_by_name(doctor_name, phone, dob, utc_start)
        #     using doctor name + phone + DOB + exact UTC start time.
        #   • If a Google Calendar ID is available, it ALSO tries to delete the
        #     corresponding GCal event (best-effort; not required).
        #   • Speaks a friendly, local-time confirmation when successfully cancelled.
        #
        # Inputs expected in session_data[call_sid]["cancel"]:
        #   {
        #     "phone":        str,   # REQUIRED earlier in the flow
        #     "doctor":       str,   # friendly doctor name (used to locate local file)
        #     "dob":          str,   # ISO 'YYYY-MM-DD' (already verified upstream)
        #     "utc_start":    str,   # ISO UTC start of the appt (preferred)
        #     "utc_end":      str,   # optional
        #     "calendar_id":  str,   # optional; if given we attempt GCal delete too
        #     "matching_event": {    # optional; set by cancel_appt_iterate
        #         "doctor_name": str,
        #         "start_utc":   str,
        #         "end_utc":     str,
        #         "friendly":    str,
        #         "phone":       str,
        #         "dob":         str
        #     }
        #   }
        #
        # Output:
        #   • Speaks success or failure; optionally transitions to booking if
        #     reschedule_after_cancel is set.
        #
        # Notes:
        #   • This stage does NOT validate DOB/phone; that is done earlier.
        #   • Calendar deletion is best-effort; local JSON removal is primary.
        # ----------------------------------------------------------------------

   
    # ===== cancel_appt_confirm (stage) =====
    elif stage == "cancel_appt_confirm":
        debug_print("📍 Stage: cancel_appt_confirm")

        cancel_ctx  = session_data[call_sid].setdefault("cancel", {})
        phone       = (cancel_ctx.get("phone") or "").strip()
        doctor      = (cancel_ctx.get("doctor") or "").strip()
        spoken_day  = (cancel_ctx.get("day") or "").strip()
        spoken_time = (cancel_ctx.get("time") or "").strip()
        utc_start   = (cancel_ctx.get("utc_start") or "").strip()
        utc_end     = (cancel_ctx.get("utc_end") or "").strip()
        calendar_id = (cancel_ctx.get("calendar_id") or "").strip()
        dob         = (cancel_ctx.get("dob") or session_data[call_sid].get("customer", {}).get("dob") or "").strip()

        cand = cancel_ctx.get("matching_event") or {}
        if cand:
            doctor    = cand.get("doctor_name", doctor) or doctor
            utc_start = cand.get("start_utc",   utc_start) or utc_start
            utc_end   = cand.get("end_utc",     utc_end) or utc_end
            phone     = cand.get("phone") or phone
            dob       = cand.get("dob")   or dob

        def _friendly_from_iso(utc_iso: str, tz_name: str = "America/Chicago") -> str:
            try:
                dt_utc = dtparser.isoparse(utc_iso)
                local = dt_utc.astimezone(pytz.timezone(tz_name))
                try:
                    return local.strftime("%A, %B %-d at %-I:%M %p")
                except Exception:
                    return local.strftime("%A, %B %d at %I:%M %p").replace(" 0", " ")
            except Exception:
                return utc_iso or (spoken_day and f"{spoken_day} at {spoken_time}") or "the scheduled time"

        phone10 = _normalize_phone10(phone)

        # Primary: local JSON cancel (doctor + phone10 + dob + utc_start)
        local_ok = False
        if doctor and phone10 and dob and utc_start:
            try:
                local_ok = cancel_appointment_by_name(doctor_name=doctor, phone=phone10, dob=dob, utc_start=utc_start)
            except Exception as e:
                debug_print(f"cancel_appt_confirm: local cancel failed → {e}")
        else:
            debug_print("cancel_appt_confirm: insufficient info for local cancel (need doctor, phone, dob, utc_start)")

        # Secondary: best-effort Google Calendar delete
        gcal_ok = False
        if calendar_id and utc_start:
            try:
                

                start_dt = dtparser.isoparse(utc_start)
                win_start = (start_dt - timedelta(minutes=30)).astimezone(timezone.utc).isoformat()
                win_end   = (start_dt + timedelta(minutes=30)).astimezone(timezone.utc).isoformat()

                matched = get_upcoming_events(calendar_id, phone10, win_start, win_end, creds, debug=True)
                if isinstance(matched, list) and matched:
                    ev = matched[0]
                elif isinstance(matched, dict):
                    ev = matched
                else:
                    ev = None

                if ev and ev.get("id"):
                    service = build("calendar", "v3", credentials=creds)
                    service.events().delete(calendarId=calendar_id, eventId=ev["id"]).execute()
                    gcal_ok = True
                    debug_print(f"cancel_appt_confirm: 🗑️ GCal event deleted id={ev['id']}")
                else:
                    debug_print("cancel_appt_confirm: no GCal event found in ±30m window")
            except Exception as e:
                debug_print(f"cancel_appt_confirm: GCal delete failed → {e}")
        else:
            debug_print("cancel_appt_confirm: skipping GCal delete (no calendar_id or utc_start)")

        # Speak outcome
        if local_ok or gcal_ok:
            friendly = _friendly_from_iso(utc_start)
            resp.say(gpt_speak(f"Your appointment with {doctor} on {friendly} has been cancelled. Thank you!"), VOICE)
        else:
            resp.say(gpt_speak("I'm sorry, I couldn't find an appointment under that phone number and time to cancel."), VOICE)

        # Optional reschedule
        if session_data[call_sid].get("reschedule_after_cancel"):
            debug_print("cancel_appt_confirm: reschedule requested → booking")
            session_data[call_sid]["stage"] = "booking"
            session_data[call_sid].pop("cancel", None)
            doctor_list_str = ", ".join(googleid_dr_name_map.values())
            gather = make_gather("Now, please tell me which doctor you'd like to reschedule with.", hints=doctor_list_str)
            resp.append(gather)
            return str(resp)

        # End call
        session_data.pop(call_sid, None)
        resp.hangup()
        debug_print("cancel_appt_confirm: session cleared")
        return str(resp)




   



if __name__ == "__main__":
    
    # ------------------------
    # 🔁 Call this once at startup
    # ------------------------

    load_doctor_appointments()


    app.run(host="0.0.0.0", port=5000)
