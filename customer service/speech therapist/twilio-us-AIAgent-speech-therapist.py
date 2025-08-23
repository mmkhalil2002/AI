# update  08/22/25 2:45 pm
# =========================
# Standard library imports
# =========================
import os
import json
import string          # for string.punctuation
import calendar
import re as _re       # use _re everywhere to avoid UnboundLocalError
from uuid import uuid4
import pickle
import openai
import calendar as _calendar
import dateparser as _dp


from datetime import datetime as _dt
from datetime import time as dtime
from typing import Any, Optional, List, Dict, Tuple, Iterator, Iterable, Union
from datetime import datetime, date, time, timedelta, timezone
from datetime import datetime as _dt  # if code references _dt
from datetime import datetime as _dt_local, date as _date_local


# =========================
# Third-party libraries
# =========================
import dateparser
import pytz as _pytz
from dotenv import load_dotenv

from dateutil import parser as dtparser
from dateutil.parser import isoparse
from dateutil.tz import gettz

from googleapiclient.discovery import build
from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow  # keep only if you actually use OAuth user flow
from google.auth.transport.requests import Request

from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient
from twilio.rest import Client

from openai import OpenAI, APIConnectionError, AuthenticationError, RateLimitError, OpenAIError

from flask import Flask, request, url_for

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
        #from flask import request, url_for
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
    day = _re.sub(r'(\d+)(st|nd|rd|th)', r'\1', spoken_day.strip(), flags=_re.IGNORECASE)

    # Handle formats like "29 of July"
    match = _re.match(r"(\d+)\s+of\s+([A-Za-z]+)", day, flags=_re.IGNORECASE)
    if match:
        day = f"{match.group(2)} {match.group(1)}"

    # Remove "of", commas, etc.
    day = day.replace(",", "").replace("of", "").strip()

    # Combine with time
    return f"{day} {spoken_time}".strip()




def build_timeslot_range(spoken_day: Union[str, date],
                         spoken_time: Union[str, dtime],
                         tolerance_minutes: int = 30) -> Tuple[str, str]:
    """
    Convert a spoken date/time (or date/time objects) to a UTC ISO 8601 window.
    Returns (utc_start_iso, utc_end_iso). Always uses America/Chicago → UTC.

    Behavior / notes:
      - Accepts either strings ("Saturday, August 16", "5:00 AM") OR date/dtime objects.
      - Normalizes things like ordinals (“16th”), filler "at", and AM/PM variants (“a.m.”).
      - Tries a few explicit strptime formats first; if they fail, falls back to dateutil.
      - If the caller didn’t say a year, we infer the current year; if that local time
        is already in the past, we roll to next year (common speech UX).
      - Duration = tolerance_minutes (default 30), used as the end time offset.
    """
    debug_print(f"📥 build_timeslot_range: Input → Day: '{spoken_day}', Time: '{spoken_time}'")

    # -----------------------------------------
    # Timezone objects (clinic tz + UTC)
    # -----------------------------------------
    local_tz = _pytz.timezone("America/Chicago")
    utc_tz   = _pytz.UTC

    # -----------------------------------------
    # Fast path: both are concrete objects
    # -----------------------------------------
    if isinstance(spoken_day, date) and isinstance(spoken_time, dtime):
        combined = datetime.combine(spoken_day, spoken_time)
        try:
            localized = local_tz.localize(combined)
        except Exception:
            # Fallback if localization complains (rare DST edge)
            localized = combined.replace(tzinfo=local_tz)
        utc_start = localized.astimezone(utc_tz)
        utc_end   = utc_start + timedelta(minutes=tolerance_minutes)
        debug_print(f"📅 Local slot: {localized} → {localized + timedelta(minutes=tolerance_minutes)}")
        debug_print(f"🌍 UTC slot: {utc_start.isoformat()} → {utc_end.isoformat()}")
        return utc_start.isoformat(), utc_end.isoformat()

    # -----------------------------------------
    # String path: clean + parse
    # -----------------------------------------
    # Day cleanup: remove ordinals, commas, and " of " (e.g., "August 16th" → "August 16")
    day_str = str(spoken_day or "").strip()
    day_str = _re.sub(r'(\d+)(st|nd|rd|th)\b', r'\1', day_str, flags=_re.IGNORECASE)
    day_str = day_str.replace(",", " ").replace(" of ", " ")
    day_str = _re.sub(r"\s+", " ", day_str).strip()

    # Time cleanup: normalize a.m./p.m., trim trailing punctuation, collapse spaces
    time_str = str(spoken_time or "").strip()
    time_str = (time_str.lower()
                        .replace("a.m.", "am").replace("p.m.", "pm")
                        .replace("a. m.", "am").replace("p. m.", "pm")
                        .replace("a. m", "am").replace("p. m", "pm")
                        .replace("a m", "am").replace("p m", "pm"))
    time_str = _re.sub(r"[.!?]+$", "", time_str)       # trim sentence-ending punctuation
    time_str = _re.sub(r"\s+", " ", time_str).strip()

    # Common filler: remove a single " at " so "August 16 at 5:30 AM" → "August 16 5:30 AM"
    # (We remove only once to avoid nuking "Saturday" → "Saturd".)
    combo = f"{day_str} {time_str}".replace(" at ", " ", 1).strip()
    debug_print(f"🧽 Cleaned combined input: '{combo}'")

    # -----------------------------------------
    # Try explicit format templates first
    # -----------------------------------------
    fmt_candidates = [
        # With weekday + 12h
        "%A %B %d %I:%M %p",
        "%A %b %d %I:%M %p",
        "%A %B %d %I %p",
        "%A %b %d %I %p",
        # Without weekday + 12h
        "%B %d %I:%M %p",
        "%b %d %I:%M %p",
        "%B %d %I %p",
        "%b %d %I %p",
        # Numeric month/day + 12h
        "%m/%d %I:%M %p",
        "%m/%d %I %p",
        # 24h variants
        "%A %B %d %H:%M",
        "%A %b %d %H:%M",
        "%B %d %H:%M",
        "%b %d %H:%M",
        "%m/%d %H:%M",
    ]

    parsed = None
    for fmt in fmt_candidates:
        try:
            parsed = datetime.strptime(combo, fmt)
            debug_print(f"✅ Parsed datetime: {parsed} using format {fmt}")
            break
        except ValueError:
            continue

    # -----------------------------------------
    # Fallback: dateutil (more tolerant)
    # -----------------------------------------
    if not parsed:
        try:
            # Use current year as the default base so month/day fill correctly
            default_dt = datetime(datetime.now().year, 1, 1, 0, 0, 0)
            parsed = dtparser.parse(combo, default=default_dt)
            debug_print(f"✅ Parsed via dateutil: {parsed.isoformat()}")
        except Exception as e:
            raise ValueError(f"🛑 Could not parse datetime from '{combo}' → {e}")

    # -----------------------------------------
    # Year inference + roll forward if past
    # -----------------------------------------
    # Did the user explicitly say a year anywhere?
    explicit_year = bool(_re.search(r"\b(19|20)\d{2}\b", combo))

    # If no explicit year was spoken, force the current year, then roll forward if needed.
    if not explicit_year:
        parsed = parsed.replace(year=datetime.now().year)

    # Localize (clinic tz), then if that local time has already passed, roll to next year.
    try:
        localized = local_tz.localize(parsed)
    except Exception:
        localized = parsed.replace(tzinfo=local_tz)

    now_local = datetime.now(local_tz)
    if not explicit_year and localized < now_local:
        try:
            bumped = parsed.replace(year=parsed.year + 1)
            try:
                localized = local_tz.localize(bumped)
            except Exception:
                localized = bumped.replace(tzinfo=local_tz)
            debug_print(f"📅 Inferred year → {localized.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        except Exception as e:
            debug_print(f"⚠️ Year roll-forward failed → {e}")

    # -----------------------------------------
    # Convert to UTC + add duration
    # -----------------------------------------
    utc_start = localized.astimezone(utc_tz)
    utc_end   = utc_start + timedelta(minutes=tolerance_minutes)

    # Logs in your existing style
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
      - "August 16th. 3000 a.m."    → ("Saturday, August 16", "3:00 AM") # STT '3000' fix

    Returns:
      (spoken_day, spoken_time) where:
        - spoken_day  = "Friday, August 15"
        - spoken_time = "h:mm AM/PM"
      or None if too ambiguous.

    Notes:
      - Requires at least a recognizable month and day (named or numeric).
      - If AM/PM missing, uses `default_meridiem`.
      - Converts 24-hour inputs (e.g., 17:30, 1730) to 12-hour + PM automatically.
    """

    def _dbg(msg: str):
        try:
            debug_print(msg)
        except Exception:
            pass

    def _infer_meridiem(hh: int, mer: str) -> str:
        """Honor spoken meridiem; otherwise use default."""
        if mer:
            return mer.upper()
        _dbg(f"parse_time_fallback_noisy: ℹ️ inferring meridiem='{default_meridiem}' for hour={hh}")
        return (default_meridiem or "AM").upper()

    if not raw:
        return None

    s = str(raw).lower()

    # -------------------------------------------------------------------------
    # 1) Normalize punctuation/spacing and AM/PM spellings (keep colons)
    # -------------------------------------------------------------------------
    # "a. m." / "a.m." / "a m" → "am" ; same for pm
    s = (_re.sub(r"\b(a\s*\.?\s*m\.?)\b", "am", s, flags=_re.IGNORECASE)
           .replace("a. m", "am").replace("a m", "am"))
    s = (_re.sub(r"\b(p\s*\.?\s*m\.?)\b", "pm", s, flags=_re.IGNORECASE)
           .replace("p. m", "pm").replace("p m", "pm"))

    # Replace dots/commas/dashes with a single space; collapse multiple spaces.
    s = _re.sub(r"[,\.\-]+", " ", s)
    s = _re.sub(r"\s+", " ", s).strip()

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
    if month is None:
        mnum = _re.search(r"\b(\d{1,2})[\/\-](\d{1,2})\b", s)
        if mnum:
            mval, dval = int(mnum.group(1)), int(mnum.group(2))
            if 1 <= mval <= 12 and 1 <= dval <= 31:
                month = mval
                day = dval
                mi = 0
                day_index = 1

    if month is None:
        _dbg("parse_time_fallback_noisy: ❌ no recognizable month (named or numeric)")
        return None

    # 2C) If named month found but day still unknown, pick the first reasonable integer after it.
    if day is None:
        for j in range(mi + 1, min(mi + 4, len(tokens))):
            tj = _re.sub(r"\D", "", tokens[j])
            if tj.isdigit():
                val = int(tj)
                if 1 <= val <= 31:
                    day = val
                    day_index = j
                    _dbg(f"parse_time_fallback_noisy: 📅 day={day} from token '{tokens[j]}'")
                    break
        # If not found, try joining split digits like "1 5" → 15
        if day is None and mi + 2 < len(tokens):
            a = _re.sub(r"\D", "", tokens[mi + 1])
            b = _re.sub(r"\D", "", tokens[mi + 2])
            if len(a) == 1 and len(b) == 1 and a.isdigit() and b.isdigit():
                val = int(a + b)
                if 1 <= val <= 31:
                    day = val
                    day_index = mi + 2
                    _dbg(f"parse_time_fallback_noisy: 📅 day={day} by joining '{a}'+'{b}'")

    if day is None:
        _dbg("parse_time_fallback_noisy: ❌ could not find day")
        return None

    # -------------------------------------------------------------------------
    # 3) Extract a time AFTER the day token (or after month if numeric date used)
    #    Accepts: "h:mm", "h mm", "hmm"/"hhmm", or just "h"; am/pm optional.
    # -------------------------------------------------------------------------
    start_idx = (day_index + 1) if (day_index is not None) else (mi + 1)
    rest = " ".join(tokens[start_idx:]) if start_idx < len(tokens) else ""

    spoken_time = None

    # (a) h:mm (with optional am/pm)
    m = _re.search(r"\b(\d{1,2})\s*:\s*(\d{1,2})(?:\s*(am|pm))?\b", rest)
    if m:
        hh, mm, mer = int(m.group(1)), int(m.group(2)), (m.group(3) or "").upper()
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            _dbg("parse_time_fallback_noisy: ❌ invalid h:mm bounds")
            return None
        if hh == 0:
            hh, mer = 12, "AM"
        elif 1 <= hh <= 12:
            mer = _infer_meridiem(hh, mer)
        elif 13 <= hh <= 23:
            mer = "PM"; hh -= 12
        spoken_time = f"{hh}:{mm:02d} {mer}"

    # (b) h mm (with optional am/pm) → "5 30 am"
    if spoken_time is None:
        m2 = _re.search(r"\b(\d{1,2})\s+(\d{2})(?:\s*(am|pm))?\b", rest)
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

    # (c) hmm/hhmm (optional am/pm) → "530", "1730", **and STT '3000' fix**
    if spoken_time is None:
        m3 = _re.search(r"\b(\d{3,4})(?:\s*(am|pm))?\b", rest)
        if m3:
            digits, mer = m3.group(1), (m3.group(2) or "").upper()

            # --- STT '3000' style normalization (e.g., "3000 am" → "3:00 am") ---
            # If 4 digits, ends with '00', and the naïve hour (>23) would fail,
            # treat the *first* digit as the hour and the rest as "00".
            #   3000 → 3:00 ; 4000 → 4:00 ; etc.
            coerced = False
            if len(digits) == 4 and digits.endswith("00"):
                naive_hh = int(digits[:2])
                if naive_hh > 23 and digits[1] == "0":
                    hh = int(digits[0])
                    mm = 0
                    coerced = True
                    _dbg(f"parse_time_fallback_noisy: 🔧 coerced '{digits}' → {hh:01d}:00")

            if not coerced:
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

    # (d) bare hour (optional am/pm) → "5", "5 am"
    if spoken_time is None:
        m4 = _re.search(r"\b(\d{1,2})(?:\s*(am|pm))?\b", rest)
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
    # 4) Build "Weekday, Month Day" using the current year (for friendliness)
    # -------------------------------------------------------------------------
    try:
        #from datetime import datetime as _dt_local, date as _date_local
        #import calendar as _calendar
        year = _dt_local.now().year
        dt = _date_local(year, month, day)
        weekday = dt.strftime("%A")  # e.g., "Friday"
        month_name = _calendar.month_name[month]
        spoken_day = f"{weekday}, {month_name} {day}"
    except Exception:
        # If anything fails, fall back to "Month Day"
        import calendar as _calendar
        month_name = _calendar.month_name[month]
        spoken_day = f"{month_name} {day}"

    _dbg(f"parse_time_fallback_noisy: ✅ parsed → day='{spoken_day}' time='{spoken_time}'")
    return (spoken_day, spoken_time)







# Preserve any existing legacy parser so we can try it first.
try:
    _smart_parse_time_prev = smart_parse_time  # type: ignore[name-defined]
except Exception:
    _smart_parse_time_prev = None


def smart_parse_time(raw: str, *, tz_name: str = "America/Chicago"):
    """
    Unified, tolerant date+time parser for noisy ASR text.

    Strategy
    --------
    1) Pre-clean the raw text (normalize AM/PM, strip common filler, trim punctuation).
    2) If a *legacy* smart_parse_time exists, try it FIRST (backward compatibility).
    3) If legacy is unusable, try parse_time_fallback_noisy(...).
    4) Return a tuple (spoken_day, spoken_time) like:
         ("Saturday, August 16", "5:00 AM")
       or None if we can’t parse confidently.

    Notes
    -----
    • Uses `_re` (i.e., `import re as _re`) to avoid UnboundLocalError.
    • `tz_name` is passed to the fallback in case it needs locale context.
    • Calls to `debug_print(...)` are wrapped so absence won’t crash.
    """

    # ---- tiny local logger (safe if debug_print is absent) ----
    def _dbg(msg: str):
        try:
            debug_print(msg)
        except Exception:
            pass

    if not raw:
        return None

    # ---- 1) Pre-clean ASR text --------------------------------
    s = str(raw).strip()

    # Normalize “a. m.” / “a.m.” / “a m” → “am” (and pm)
    try:
        s = _re.sub(r"\b(a\s*\.?\s*m\.?)\b", "am", s, flags=_re.IGNORECASE)
        s = _re.sub(r"\b(p\s*\.?\s*m\.?)\b", "pm", s, flags=_re.IGNORECASE)
    except Exception:
        pass

    # Strip common filler ASR sometimes prepends:
    #   “I couldn’t hear August 16th at 5 am” → “August 16th at 5 am”
    s = _re.sub(
        r"^\s*(i\s+couldn['’]t\s+hear|you\s+said|caller\s+said|they\s+said|it(?:'s|\s+is))\b[,:;\-\s]*",
        "",
        s,
        flags=_re.IGNORECASE,
    )

    # Remove trailing terminal punctuation (keeps time colons intact)
    s = _re.sub(r"[.!?]\s*$", "", s)

    # ---- 2) Try legacy parser first (if present) ---------------
    if _smart_parse_time_prev:
        try:
            v = _smart_parse_time_prev(s)
            if isinstance(v, tuple) and len(v) == 2 and all(v):
                _dbg("smart_parse_time: ✅ legacy parser")
                return v
            else:
                _dbg("smart_parse_time: ℹ️ legacy unusable → trying fallback")
        except Exception as e:
            _dbg(f"smart_parse_time: ℹ️ legacy error → {e} ; trying fallback")

    # ---- 3) Fallback: tolerant parser for noisy inputs ---------
    try:
        v = parse_time_fallback_noisy(s, tz_name=tz_name, default_meridiem="AM")
        if isinstance(v, tuple) and len(v) == 2 and all(v):
            _dbg(f"smart_parse_time: ✅ fallback parsed → day='{v[0]}' time='{v[1]}'")
            return v
    except Exception as e:
        _dbg(f"smart_parse_time: ⚠️ fallback error → {e}")

    # ---- 4) Nothing worked ------------------------------------
    _dbg("smart_parse_time: ❌ both parsers failed")
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
    cleaned = _re.sub(r"[^\d\s\-]", "", speech_text)
    print(f"🔍 Cleaned speech (kept digits/spaces/dashes): '{cleaned}'")

    # 🔢 Step 2: Extract digits only
    digits = _re.sub(r"[^\d]", "", cleaned)
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

    clean_phone = _re.sub(r"[^\d]", "", phone)
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

        summary_digits = _re.sub(r"[^\d]", "", summary)
        description_digits = _re.sub(r"[^\d]", "", description)

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
    digits_only_phone = _re.sub(r"\D", "", phone or "")
    if not digits_only_phone:
        raise ValueError("Phone is required and must contain digits.")

    # -----------------------------------------
    # Normalize DOB into ISO YYYY-MM-DD (if any)
    # -----------------------------------------
    dob_iso = (dob or "").strip()
    if dob_iso:
        # Already ISO?
        if _re.fullmatch(r"\d{4}-\d{2}-\d{2}", dob_iso) is None:
            # Try MM/DD/YYYY or MM-DD-YYYY
            m = _re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", dob_iso)
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
            if _re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", s):
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
        p = _re.sub(r"\D", "", appt.get("phone", ""))
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
            appt_norm["phone"] = _re.sub(r"\D", "", appt_norm.get("phone", ""))
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
# --- Robust DOB parser (speech + keypad) --------------------------------------
def parse_dob_input(speech_text: str, dtmf_digits: str):
    """
    Parse DOB from either:
      - DTMF: MMDDYYYY (strictly 8 digits), OR
      - Speech: e.g., "February 3rd, 1956", "Feb 3 1956", "2/3/1956".

    Returns:
      - datetime(year, month, day) on success
      - None on failure

    Behavior:
      - Speech requires explicit 4-digit year (to avoid guessing).
      - Strips ordinals ("1st","2nd","3rd","4th") and punctuation.
      - Uses US ordering (MDY) and English language.

    Logging:
      - Uses debug_print if available, else print.
    """
    # ---- Local safe logger ---------------------------------------------------
    def _dbg(msg: str):
        try:
            debug_print(msg)  # type: ignore[name-defined]
        except Exception:
            try:
                print(msg)
            except Exception:
                pass

    # Import inside to avoid alias-name issues elsewhere in the file
    

    # 1) Prefer DTMF if provided: MMDDYYYY
    digits = "".join(ch for ch in (dtmf_digits or "") if ch.isdigit())
    if digits:
        _dbg(f"parse_dob_input: 🔢 DTMF received → '{digits}'")
        if len(digits) == 8:
            try:
                mm = int(digits[0:2])
                dd = int(digits[2:4])
                yyyy = int(digits[4:8])
                # Basic sanity
                if 1 <= mm <= 12 and 1 <= dd <= 31 and 1900 <= yyyy <= _dt.now().year:
                    dob = _dt(yyyy, mm, dd)
                    _dbg(f"parse_dob_input: ✅ DTMF parsed → {dob.strftime('%Y-%m-%d')}")
                    return dob
            except Exception as e:
                _dbg(f"parse_dob_input: ❌ DTMF parse error → {e}")
        else:
            _dbg(f"parse_dob_input: ❌ DTMF length != 8 (got {len(digits)})")

    # 2) Speech fallback
    raw = (speech_text or "").strip()
    if not raw:
        _dbg("parse_dob_input: ❌ no speech provided")
        return None

    s = raw.lower()
    _dbg(f"parse_dob_input: 🗣️ speech raw → '{s}'")

    # Normalize common forms:
    # - Ordinals: "3rd" → "3", "21st" → "21"
    s = _re.sub(r"\b(\d+)\s*(st|nd|rd|th)\b", r"\1", s, flags=_re.IGNORECASE)

    # Handle mis-hearings like "3d 1956" → "3 1956" (D from "3rd")
    s = _re.sub(r"\b(\d+)d\b", r"\1", s, flags=_re.IGNORECASE)

    # Remove commas/periods; collapse whitespace
    s = _re.sub(r"[,\.]+", " ", s)
    s = _re.sub(r"\s+", " ", s).strip()

    # Require a 4-digit year in the utterance (avoid guessing current year)
    has_year4 = bool(_re.search(r"\b(19|20)\d{2}\b", s))
    if not has_year4:
        _dbg("parse_dob_input: ❌ no 4-digit year present in speech → cannot parse DOB safely")
        return None

    # Try parsing with explicit US ordering (MDY), English language
    settings = {
        "RETURN_AS_TIMEZONE_AWARE": False,
        "PREFER_DAY_OF_MONTH": "first",
        "DATE_ORDER": "MDY",
        "STRICT_PARSING": True,
    }

    try:
        parsed = _dp.parse(s, languages=["en"], settings=settings)
    except Exception as e:
        _dbg(f"parse_dob_input: ❌ dateparser error → {e}")
        parsed = None

    if not parsed:
        _dbg("parse_dob_input: ❌ dateparser failed to parse speech")
        return None

    # Sanity: ensure parsed date contains a year within sensible DOB range
    yyyy = parsed.year
    if not (1900 <= yyyy <= _dt.now().year):
        _dbg(f"parse_dob_input: ❌ parsed year out of range → {yyyy}")
        return None

    _dbg(f"parse_dob_input: ✅ speech parsed → {parsed.strftime('%Y-%m-%d')}")
    return parsed




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
    return _re.sub(r"\s+", " ", (s or "").strip())



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
        if _re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            return s
        m = _re.match(r"^\s*(\d{1,2})[\/-](\d{1,2})[\/-](\d{4})\s*$", s)
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
    return _re.sub(r"[^a-zA-Z\s]", "", text).lower().strip()
@app.route("/voice", methods=["POST"])
@app.route("/voice/", methods=["POST"])  # Accepts trailing slash
def voice():
    # Create a new TwiML VoiceResponse object to build the voice reply to the caller
    resp = VoiceResponse()

    # Extract the unique call ID (SID) from the request parameters to track the session
    call_sid = request.values.get("CallSid", "")

    # Retrieve the customer's speech input (transcribed by Twilio's Speech-to-Text)
    speech_result = (request.values.get("SpeechResult") or "").strip()
    # Also grab any keypad input (DTMF) Twilio might have sent with the same webhook
    try:
        dtmf_digits = (request.values.get("Digits") or "").strip()
    except Exception:
        dtmf_digits = ""

    print(f"📢 voice :speech_result: {speech_result}")

    # Determine the current interaction stage (default to "intro" if not previously set)
    stage = session_data.get(call_sid, {}).get("stage", "intro")

    # ----------------------------------------------------------------------
    # 🔇 CENTRAL SILENCE GUARD
    # If we didn't hear *anything* (no speech, no DTMF), re-prompt with
    # stage-appropriate text. We skip stages that already have their own
    # robust silence handling (e.g., collect_cc).
    # ----------------------------------------------------------------------
    def _silence_prompt_for_stage(st: str) -> tuple[str, str]:
        """Return (prompt, hints) best suited for the current stage."""
        # Default: generic prompt, no hints
        hints = ""
        if st in ("intro", "intent"):
            return (
                "I didn’t hear anything. Would you like to book an appointment, cancel one, reschedule, or leave a message?",
                hints
            )
        if st == "booking":
            doctor_list = ", ".join(googleid_dr_name_map.values())
            hints = doctor_list
            return ("Please say the name of the doctor you'd like to book with.", hints)
        if st == "collect_phone":
            hints = "zero one two three four five six seven eight nine double triple"
            return ("Please say or enter your ten digit phone number including area code.", hints)
        if st == "collect_dob":
            return ("Please say your birth date, for example 'July third 1990'. Or type MMDDYYYY then press pound.", hints)
        if st == "ask_time_date":
            return ("Please say the appointment time, for example, 'August 15th at 5 AM'.", hints)
        if st == "collect_first_name":
            return ("Please say your first name.", hints)
        if st == "collect_last_name":
            return ("Please say your last name.", hints)
        if st == "collect_address":
            return ("Please say your street address, city, and ZIP. For example, '118 Briar Oak, Murphy, Texas 75094'.", hints)
        if st == "cancel_appointment":
            doctor_list = ", ".join(googleid_dr_name_map.values())
            hints = doctor_list
            return ("Please say the name of the doctor whose appointment you want to cancel.", hints)
        if st in ("cancel_appt_by_phone_number",):
            hints = "zero one two three four five six seven eight nine double triple"
            return ("Please say the phone number used when booking, including area code.", hints)
        if st in ("cancel_appt_by_time_date", "cancel_appt_by_date_time"):
            return ("Please say the date and time of the appointment you want to cancel, for example, 'July third at nine AM'.", hints)
        if st == "cancel_appt_get_dob":
            return ("Please say your birth date, for example 'July third nineteen fifty six'. Or type MMDDYYYY then press pound.", hints)
        if st == "voicemail":
            return ("Please leave your name, phone number, and message after the beep.", hints)

        # Fallback generic
        return ("Sorry, I didn’t hear anything. Please say that again.", hints)

    # Only run the guard outside of the very first greeting (intro),
    # and skip stages that handle silence internally (collect_cc).
    if stage not in ("intro", "collect_cc"):
        if not speech_result and not dtmf_digits:
            session_data.setdefault(call_sid, {})
            key = f"silence_{stage}"
            session_data[call_sid][key] = session_data[call_sid].get(key, 0) + 1
            tries = session_data[call_sid][key]
            debug_print(f"voice(): 🔇 silence detected at stage='{stage}' (tries={tries})")

            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            prompt, hints = _silence_prompt_for_stage(stage)
            try:
                gather = make_gather(prompt, hints=hints) if hints else make_gather(prompt)
            except Exception:
                # Very defensive fallback
                gather = make_gather("Sorry, I didn’t hear anything. Please try again.")
            resp.append(gather)
            # Redirect so Twilio posts again after Gather
            try:
                resp.redirect(url_for("voice"))
            except Exception:
                resp.redirect("/voice")
            return str(resp)

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
            debug_print("❌ Intent to cancel appointment detected → entering cancellation flow")
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
            debug_print(f"📅 Intent to book recognized → advancing to 'booking' stage")

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
            debug_print("📩 Intent to leave a message detected → recording voicemail")
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
            debug_print(f"❓ Unclear intent: '{lower}' → re-prompting for intent choice")

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

        # ✅ Safe punctuation constant (avoid using `string` directly)
        try:
            from string import punctuation as _PUNCT
        except Exception:
            _PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

        # 📻 Clean and normalize speech input
        spoken_text = (speech_result or "").lower().strip()
        # ⛑️ Use _PUNCT so we never reference `string` locally
        spoken_clean = spoken_text.translate(str.maketrans('', '', _PUNCT)).strip()
        print(f"📻 booking :speech_result: {spoken_clean}")

        # 🚫 Block common junk phrases often returned by Twilio hallucination
        junk_inputs = {
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening", "yo", "test",
            "1", "yes", "no", "i know", "huh", "what", "okay", "ok", "bye", "goodbye", ""
        }

        if not spoken_clean or spoken_clean in junk_inputs or len(spoken_clean) < 3:
            print(f"⏩ Skipping junk doctor input: '{spoken_clean}' — re-prompting without retry")
            doctor_list_str = ", ".join(googleid_dr_name_map.values())
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
            friendly_clean = friendly.lower().translate(str.maketrans('', '', _PUNCT)).strip()
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
            matched_id = partial_matches[0][0]  # or ask user to clarify

        # ------------------------------------------------------------------
        # 🤖 2. GPT fallback (only if 2+ words)
        # ------------------------------------------------------------------
        if matched_id is None and len(spoken_clean.split()) >= 2:
            try:
                extracted = extract_doctor_name(spoken_text)
                if extracted:
                    extracted_clean = extracted.lower().translate(str.maketrans('', '', _PUNCT)).strip()
                    for doc_id, friendly in googleid_dr_name_map.items():
                        friendly_clean = friendly.lower().translate(str.maketrans('', '', _PUNCT)).strip()
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
        session_data[call_sid]["stage"] = "collect_phone"

        friendly_name = googleid_dr_name_map[matched_id]
        phone_prompt = (
            f"Great, we'll book with {friendly_name}. "
            "Please say or enter your phone number including area code."
        )

        gather = make_gather(phone_prompt)
        resp.append(gather)
        return str(resp)



    elif stage == "collect_phone":
        # ----------------------------------------------------------------------
        # 📞 Stage: collect_phone  MOH
        #
        # Goal:
        #   - Capture the caller's phone number via DTMF or speech.
        #   - Normalize to 10 digits (strip non-digits; if 11 and starts with '1', drop leading 1).
        #   - Store at session_data[call_sid]["customer"]["phone"].
        #   - If we were sent here from another stage, return to that stage via
        #     session_data[call_sid]["return_stage"].
        #
        # 🆕 Silent mode handling:
        #   - If no SpeechResult and no Digits → re-prompt up to 3 times.
        # ----------------------------------------------------------------------
        debug_print("collect_phone: 📍 Stage entered")

        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("customer", {})

        # Pull inputs
        try:
            dtmf_digits = (request.values.get("Digits") or "").strip()
        except Exception:
            dtmf_digits = ""
        speech_text = (speech_result or "").strip()
        debug_print(f"collect_phone: speech='{speech_text}' DTMF='{dtmf_digits}'")

        # 🆕 Silent mode: nothing heard → re-prompt with cap 3
        if not (speech_text or dtmf_digits):
            tries = session_data[call_sid].get("silence_collect_phone", 0) + 1
            session_data[call_sid]["silence_collect_phone"] = tries
            debug_print(f"collect_phone: 🤐 no input heard (tries={tries})")

            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            prompt = (
                "Please say or type your ten digit phone number including area code. "
                "You can also type the digits, then press pound."
            )
            gather = make_gather(prompt, hints="zero one two three four five six seven eight nine double triple")
            resp.append(gather)
            return str(resp)

        # We heard something → clear stage silence counter
        session_data[call_sid].pop("silence_collect_phone", None)

        # --- helpers --------------------------------------------------------------
        def _spoken_to_digits(raw: str) -> str:
            """
            Convert spoken words to digits.
            Supports 'double'/'triple' and common homophones (oh/o for 0, to/too for 2, ate for 8).
            Also extracts any digits already present in the string.
            """
            if not raw:
                return ""
            words = (
                raw.lower()
                .replace("-", " ")
                .replace(",", " ")
                .replace(".", " ")
                .replace("(", " ")
                .replace(")", " ")
                .split()
            )
            m = {
                "zero": "0", "oh": "0", "o": "0",
                "one": "1", "two": "2", "to": "2", "too": "2",
                "three": "3", "four": "4", "for": "4",
                "five": "5", "six": "6", "seven": "7",
                "eight": "8", "ate": "8", "nine": "9",
            }
            out = []
            i = 0
            while i < len(words):
                w = words[i].strip()
                # handle "double X" / "triple X"
                if w in ("double", "triple") and i + 1 < len(words):
                    nxt = words[i + 1].strip()
                    if nxt in m:
                        out.extend([m[nxt]] * (2 if w == "double" else 3))
                        i += 2
                        continue
                # map word to digit
                if w in m:
                    out.append(m[w])
                else:
                    # copy any digits present in the token
                    out.extend([c for c in w if c.isdigit()])
                i += 1
            return "".join(out)

        def _normalize_10(s: str) -> str:
            """Keep only digits; if 11 starting with '1', strip to 10."""
            d = "".join(ch for ch in (s or "") if ch.isdigit())
            return d[1:] if len(d) == 11 and d.startswith("1") else d

        # Prefer DTMF; if none, use speech→digits
        if dtmf_digits:
            raw_digits = _re.sub(r"\D", "", dtmf_digits)
        else:
            raw_digits = _re.sub(r"\D", "", _spoken_to_digits(speech_text))

        debug_print(f"collect_phone: raw_digits='{raw_digits}'")

        # Normalize to 10 (truncate if user repeated extra digits)
        phone10 = _normalize_10(raw_digits)
        if len(phone10) > 10:
            phone10 = phone10[:10]

        # Validate
        if len(phone10) != 10:
            session_data[call_sid]["retry_phone"] = session_data[call_sid].get("retry_phone", 0) + 1
            r = session_data[call_sid]["retry_phone"]
            debug_print(f"collect_phone: ❌ invalid phone '{raw_digits}' (→ '{phone10}') retry={r}")

            if r >= 3:
                resp.say(gpt_speak("Sorry, I couldn't capture your phone number. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            prompt = (
                "Please say or type your ten digit phone number including area code. "
                "For example, four six nine four six three three two seven six. "
                "You can also type the digits, then press pound."
            )
            gather = make_gather(prompt, hints="zero one two three four five six seven eight nine double triple")
            resp.append(gather)
            return str(resp)

        # Save and reset retry
        session_data[call_sid]["customer"]["phone"] = phone10
        session_data[call_sid]["retry_phone"] = 0
        debug_print(f"collect_phone: ✅ saved phone10={phone10}")

        # If we were sent here by another stage, jump back there now
        return_stage = session_data[call_sid].pop("return_stage", None)
        if return_stage:
            session_data[call_sid]["stage"] = return_stage
            debug_print(f"collect_phone: ➡️ returning to {return_stage}")
            resp.redirect("/voice")
            return str(resp)

        # Decide next step by flow context
        if "cancel" in session_data[call_sid]:
            session_data[call_sid]["stage"] = "cancel_appt_get_date_time"
            gather = make_gather(
                "Thanks. Now tell me the date and time of the appointment you want to cancel. "
                "For example, August 15th at 5 AM."
            )
            resp.append(gather)
            return str(resp)

        # Default booking path: ask for DOB next
        session_data[call_sid]["stage"] = "collect_dob"
        gather = make_gather(
            "Thanks. Please provide your date of birth. You can say it, or enter 2 digits 4 Month and 2 4 day and 4 4 year, then press pound."
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
    # 🆕 Silent mode:
    #   - If neither speech nor digits were received, re-prompt up to 3 times using a
    #     separate counter (silence_dob), then hang up politely.
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

        # ⚠️ Avoid name shadowing: use `_date` for validation, not `date`
        from datetime import date as _date

        # Ensure session buckets exist
        session_data.setdefault(call_sid, {})

        # 1) Pull DTMF if present (Twilio sends digits on the same webhook), otherwise use speech.
        try:
            dtmf_digits = (request.values.get("Digits") or "").strip()
        except Exception:
            dtmf_digits = ""

        speech_text = (speech_result or "").strip()
        debug_print(f"collect_dob: 🎙️ speech_text='{speech_text}', 🔢 dtmf_digits='{dtmf_digits}'")

        # 1a) 🔁 SILENT MODE: nothing heard → re-ask up to 3 times (separate counter).
        if not dtmf_digits and not speech_text:
            tries = session_data[call_sid].get("silence_dob", 0) + 1
            session_data[call_sid]["silence_dob"] = tries
            debug_print(f"collect_dob: 🤐 no input heard; silence retries={tries}")

            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Re-prompt using short, consistent copy
            try:
                gather = make_gather_dob(DOB_PROMPT_SHORT)
            except Exception:
                gather = make_gather(DOB_PROMPT_SHORT)
            resp.append(gather)
            # Always redirect back so Twilio posts again after gather
            try:
                from flask import url_for
                resp.redirect(url_for("voice"))
            except Exception:
                resp.redirect("/voice")
            return str(resp)

        # We heard *something* → clear the silence counter for this stage
        session_data[call_sid].pop("silence_dob", None)

        # 2) Parse DOB input (helper handles speech and/or MMDDYYYY).
        #    parse_dob_input should return a datetime on success, or None if missing month/day/year.
        dt = parse_dob_input(speech_text, dtmf_digits)
        if not dt:
            # Retry counter (so we don’t loop forever on parsing errors)
            session_data[call_sid]["retry_dob"] = session_data[call_sid].get("retry_dob", 0) + 1
            r = session_data[call_sid]["retry_dob"]
            debug_print(f"collect_dob: ❌ Parse failed. retry_dob={r}")

            if r >= 3:
                # Fail out cleanly if user can’t provide a DOB we can parse
                resp.say(gpt_speak("Sorry, I couldn’t understand your date of birth. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Re-prompt using short, consistent copy
            try:
                gather = make_gather_dob(DOB_PROMPT_SHORT)
            except Exception:
                gather = make_gather(DOB_PROMPT_SHORT)
            resp.append(gather)
            try:
                from flask import url_for
                resp.redirect(url_for("voice"))
            except Exception:
                resp.redirect("/voice")
            return str(resp)

        # 3) Validate DOB sanity window (e.g., 1900..today)
        try:
            today = _date.today()
            min_date = _date(1900, 1, 1)
            dob_date = dt.date()
            if not (min_date <= dob_date <= today):
                debug_print(f"collect_dob: ⚠️ DOB out of range → {dob_date.isoformat()}")

                session_data[call_sid]["retry_dob"] = session_data[call_sid].get("retry_dob", 0) + 1
                try:
                    gather = make_gather_dob(DOB_PROMPT_SHORT)
                except Exception:
                    gather = make_gather(DOB_PROMPT_SHORT)
                resp.append(gather)
                try:
                    from flask import url_for
                    resp.redirect(url_for("voice"))
                except Exception:
                    resp.redirect("/voice")
                return str(resp)
        except Exception as e:
            # Do not fail the call; just log and re-prompt safely
            debug_print(f"collect_dob: ⚠️ Validation error → {e}")
            session_data[call_sid]["retry_dob"] = session_data[call_sid].get("retry_dob", 0) + 1
            try:
                gather = make_gather_dob(DOB_PROMPT_SHORT)
            except Exception:
                gather = make_gather(DOB_PROMPT_SHORT)
            resp.append(gather)
            try:
                from flask import url_for
                resp.redirect(url_for("voice"))
            except Exception:
                resp.redirect("/voice")
            return str(resp)

        # 4) Store ISO DOB in session
        iso_dob = dt.strftime("%Y-%m-%d")
        session_data[call_sid].setdefault("customer", {})
        session_data[call_sid]["customer"]["dob"] = iso_dob
        debug_print(f"collect_dob: ✅ Stored DOB → {iso_dob}")

        # Reset the parse retry counter on success so it doesn't affect later stages
        session_data[call_sid].pop("retry_dob", None)

        # 5) Always move to ask_time_date next (your booking flow expects this)
        session_data[call_sid]["stage"] = "ask_time_date"
        debug_print("collect_dob: ➡️ Next stage → ask_time_date")

        # 6) Prompt for appointment time/date using the short prompt
        try:
            gather = make_gather("Thanks. " + TIME_PROMPT_SHORT)
        except Exception:
            # Very defensive fallback (in case make_gather signature differs)
            gather = make_gather("Thanks. Please say the date and time, for example 'August 12 at 5 PM'.")
        resp.append(gather)
        try:
            from flask import url_for
            resp.redirect(url_for("voice"))
        except Exception:
            resp.redirect("/voice")
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
    # 🆕 Silent mode:
    #   - If we hear nothing, re-ask up to 3 times via a separate counter (silence_time).
    # ----------------------------------------------------------------------
    elif stage == "ask_time_date":
        debug_print(f"ask_time_date: 🗣️ Received speech: {speech_result}")

        # ----------------------------------------------------------------------
        # Prompt constants (short = consistent muscle memory for callers)
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

        # Ensure session bucket
        session_data.setdefault(call_sid, {})

        # --- tiny helpers for readability ---
        def _is_blank(x) -> bool:
            return (x is None) or (str(x).strip() == "")

        def _has_time_token(raw: str) -> bool:
            """
            Heuristic: if parse fully failed, check if caller likely said a time.
            Accepts 'am/pm', '5:30', or compact '0530'/'1730' tokens.
            """
            s = (raw or "").lower()
            return (
                ("am" in s) or ("pm" in s) or (":" in s)
                or ("o'clock" in s) or ("oclock" in s)
                or (_re.search(r"\b\d{3,4}\b", s) is not None)
            )

        def _has_date_token(raw: str) -> bool:
            """Heuristic: check for month/weekday/date words/tokens."""
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

        # 0) Guard: doctor must be chosen (per-doctor calendar)
        doctor_id = session_data.get(call_sid, {}).get("doctor_id")
        if not doctor_id:
            debug_print("ask_time_date: ❌ no doctor selected → sending user to pick a doctor")
            session_data[call_sid]["stage"] = "choose_doctor"
            doctor_list = ", ".join(googleid_dr_name_map.values())
            gather = make_gather("Which doctor would you like to see?", hints=doctor_list)
            resp.append(gather)
            return str(resp)

        calendar_id = doctor_id

        # --- Minimal pre-clean for AM/PM variants & trailing punctuation ---
        # Use `_re` (imported at top of file) to avoid UnboundLocalError.
        try:
            _raw = (speech_result or "").strip()
            _raw = _re.sub(r"\b(a\s*\.?\s*m\.?)\b", "am", _raw, flags=_re.IGNORECASE)
            _raw = _re.sub(r"\b(p\s*\.?\s*m\.?)\b", "pm", _raw, flags=_re.IGNORECASE)
            _raw = _re.sub(r"[.!?]\s*$", "", _raw)
        except Exception:
            _raw = (speech_result or "")

        # 🔈 Silent-mode handling (nothing heard)
        if not _raw:
            tries = session_data[call_sid].get("silence_time", 0) + 1
            session_data[call_sid]["silence_time"] = tries
            debug_print(f"ask_time_date: 🤐 no input; silence retries={tries}")

            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            gather = make_gather("Please say the date and time, for example, 'August 15th at 5 AM'.")
            resp.append(gather)
            # Redirect so Twilio re-posts after gather
            try:
                from flask import url_for
                resp.redirect(url_for("voice"))
            except Exception:
                resp.redirect("/voice")
            return str(resp)

        # We heard something → clear silence counter
        session_data[call_sid].pop("silence_time", None)

        # 1) Parse (day, time) from the caller’s phrase
        #    Expect a tuple like: ("Friday, August 15", "5:00 AM") or similar.
        time_info = smart_parse_time(_raw)

        # --- Branch A: parser returned nothing useful (None / wrong type / wrong length) ---
        if not time_info or not isinstance(time_info, tuple) or len(time_info) != 2:
            # Heuristics to give a specific prompt about what's missing
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
            # If we have both phone & DOB and the customer exists → go straight to booking confirmation
            if customer_phone and customer_dob and customer_search(customer_phone, customer_dob):
                debug_print("ask_time_date: 📋 Customer on file — skip name collection")
                # 🚩 IMPORTANT: actually transition to book_appt_confirm so it can reserve Google Calendar.
                session_data[call_sid]["stage"] = "book_appt_confirm"
                session_data[call_sid]["auto_confirm"] = True  # optional flag for that stage
                debug_print("ask_time_date: ➡️ Redirecting to book_appt_confirm (auto_confirm=True)")
                # Immediately re-enter handler; book_appt_confirm will execute now
                try:
                    from flask import url_for
                    resp.redirect(url_for("voice"))
                except Exception:
                    resp.redirect("/voice")
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
        # ----------------------------------------------------------------------
        # 🎯 Goal:
        #   - Capture FIRST name via speech.
        #   - Handle silence separately (up to 3 silent retries).
        #   - Clean & lightly validate (letters/'/- only).
        #   - Store into session_data[call_sid]["customer"]["first_name"].
        #   - Advance → collect_last_name.
        # ----------------------------------------------------------------------
        session_data.setdefault(call_sid, {}).setdefault("customer", {})
        raw = (speech_result or "").strip()
        debug_print(f"collect_first_name: raw='{raw}'")

        # 🔇 Silent mode: nothing heard → re-ask (no hangup until 3 tries)
        if not raw:
            tries = session_data[call_sid].get("silence_first_name", 0) + 1
            session_data[call_sid]["silence_first_name"] = tries
            debug_print(f"collect_first_name: 🤐 silence; tries={tries}")
            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)
            gather = make_gather("I didn’t hear your first name. Please say just your first name.")
            resp.append(gather)
            try:
                from flask import url_for
                resp.redirect(url_for("voice"))
            except Exception:
                resp.redirect("/voice")
            return str(resp)

        # We heard something → clear silence counter
        session_data[call_sid].pop("silence_first_name", None)

        # 🧽 Clean & normalize (remove punctuation, compress spaces; ignore fillers)
        import string  # ensure available; you already import at top, but safe to re-import
        cleaned = raw.translate(str.maketrans('', '', string.punctuation)).strip()
        cleaned = _re.sub(r"\s+", " ", cleaned)

        # Heuristics: drop leading fillers like "my name is", "this is", "i am", "i'm", "it's"
        # Keep it simple: if a filler exists, take tokens AFTER it.
        lower = cleaned.lower()
        filler_pat = _re.compile(r"\b(?:my name is|this is|i am|i'm|it is|it's)\b\s*", _re.IGNORECASE)
        cleaned = filler_pat.sub("", cleaned).strip()

        # Take the first token as FIRST name (avoid multiple words here)
        tokens = cleaned.split()
        first_name = tokens[0] if tokens else ""

        # ✅ Validate: letters, apostrophes or hyphens, 1–40 chars, no digits
        if not first_name or not _re.fullmatch(r"[A-Za-z][A-Za-z'\-]{0,39}", first_name):
            # Soft reprompt (up to 3 attempts)
            r = session_data[call_sid].get("retry_first_name", 0) + 1
            session_data[call_sid]["retry_first_name"] = r
            debug_print(f"collect_first_name: ❌ invalid first name '{first_name}' retry={r}")
            if r >= 3:
                resp.say(gpt_speak("Sorry, I couldn’t capture your first name. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)
            gather = make_gather("I didn't catch that clearly. Please say just your first name.")
            resp.append(gather)
            try:
                from flask import url_for
                resp.redirect(url_for("voice"))
            except Exception:
                resp.redirect("/voice")
            return str(resp)

        # Store & advance
        session_data[call_sid]["customer"]["first_name"] = first_name
        session_data[call_sid]["stage"] = "collect_last_name"
        # Reset name retry counter on success
        session_data[call_sid].pop("retry_first_name", None)
        debug_print(f"collect_first_name: ✅ saved first_name='{first_name}' → next=collect_last_name")

        gather = make_gather("Thank you. Now, what is your last name?")
        resp.append(gather)
        try:
            from flask import url_for
            resp.redirect(url_for("voice"))
        except Exception:
            resp.redirect("/voice")
        return str(resp)


    # ===== collect_last_name (stage) =====
    elif stage == "collect_last_name":
        # ----------------------------------------------------------------------
        # 🎯 Goal:
        #   - Capture LAST name via speech.
        #   - Handle silence separately (up to 3 silent retries).
        #   - Clean & lightly validate (letters/spaces/'/-; allow multi-token like "van dyke").
        #   - Store into session_data[call_sid]["customer"]["last_name"].
        #   - Advance → collect_address.
        # ----------------------------------------------------------------------
        session_data.setdefault(call_sid, {}).setdefault("customer", {})
        raw = (speech_result or "").strip()
        debug_print(f"collect_last_name: raw='{raw}'")

        # 🔇 Silent mode
        if not raw:
            tries = session_data[call_sid].get("silence_last_name", 0) + 1
            session_data[call_sid]["silence_last_name"] = tries
            debug_print(f"collect_last_name: 🤐 silence; tries={tries}")
            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)
            gather = make_gather("I didn’t hear your last name. Please say your last name now.")
            resp.append(gather)
            try:
                from flask import url_for
                resp.redirect(url_for("voice"))
            except Exception:
                resp.redirect("/voice")
            return str(resp)

        # We heard something → clear silence counter
        session_data[call_sid].pop("silence_last_name", None)

        # 🧽 Clean & normalize (keep inner spaces; strip punctuation except apostrophe/hyphen)
        import string
        # Remove punctuation except apostrophe and hyphen: build whitelist
        punct_keep = "'-"
        trans_table = str.maketrans('', '', "".join(ch for ch in string.punctuation if ch not in punct_keep))
        cleaned = raw.translate(trans_table).strip()
        cleaned = _re.sub(r"\s+", " ", cleaned)

        # Minimal validation: at least one letter, only letters/spaces/'/-
        if (not cleaned) or (not _re.search(r"[A-Za-z]", cleaned)) or (not _re.fullmatch(r"[A-Za-z'\- ]{1,60}", cleaned)):
            r = session_data[call_sid].get("retry_last_name", 0) + 1
            session_data[call_sid]["retry_last_name"] = r
            debug_print(f"collect_last_name: ❌ invalid last name '{cleaned}' retry={r}")
            if r >= 3:
                resp.say(gpt_speak("Sorry, I couldn’t capture your last name. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)
            gather = make_gather("Sorry, I didn't catch your last name. Please repeat it clearly.")
            resp.append(gather)
            try:
                from flask import url_for
                resp.redirect(url_for("voice"))
            except Exception:
                resp.redirect("/voice")
            return str(resp)

        # Store & advance
        session_data[call_sid]["customer"]["last_name"] = cleaned
        session_data[call_sid]["stage"] = "collect_address"
        # Reset retry counter on success
        session_data[call_sid].pop("retry_last_name", None)
        debug_print(f"collect_last_name: ✅ saved last_name='{cleaned}' → next=collect_address")

        gather = make_gather("Got it. What is your full address, please?")
        resp.append(gather)
        try:
            from flask import url_for
            resp.redirect(url_for("voice"))
        except Exception:
            resp.redirect("/voice")
        return str(resp)




   


    # ===== collect_address (stage) =====
    elif stage == "collect_address":
        # ---------------------------------------------------------------
        # 🏠 Stage: collect_address
        # Goal:
        #   - Capture the caller's address from speech.
        #   - Normalize whitespace & punctuation (common STT artifacts).
        #   - Handle silence with separate retry counter (no premature hangups).
        #   - Store under session_data[call_sid]["customer"]["address"].
        #   - Advance to collect_cc with a clear prompt.
        # Notes:
        #   - Uses `_re` (import re as _re) to avoid UnboundLocalError.
        #   - Keeps flow consistent by redirecting back to /voice after gather.
        # ---------------------------------------------------------------

        session_data.setdefault(call_sid, {}).setdefault("customer", {})

        # Safely pull raw text
        try:
            raw = (speech_result or request.values.get("SpeechResult") or "").strip()
        except Exception:
            raw = (speech_result or "").strip()

        debug_print(f"collect_address: 📬 Collected address (raw): {raw}")

        # 🔇 Silent-mode handling: nothing heard → ask again (up to 3 times)
        if not raw:
            tries = session_data[call_sid].get("silence_address", 0) + 1
            session_data[call_sid]["silence_address"] = tries
            debug_print(f"collect_address: 🤐 silence; tries={tries}")
            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            prompt = (
                "I didn't catch the address. Please say your street address, city, and ZIP. "
                "For example, '118 Briar Oak, Murphy, Texas 75094'."
            )
            gather = make_gather(prompt)
            resp.append(gather)
            # redirect so Twilio posts back after the gather
            try:
                from flask import url_for
                resp.redirect(url_for("voice"))
            except Exception:
                resp.redirect("/voice")
            return str(resp)

        # We heard something → clear silence counter
        session_data[call_sid].pop("silence_address", None)

        # ---------- Normalize ----------
        addr = raw

        # 1) Collapse multiple spaces
        addr = _re.sub(r"\s+", " ", addr)

        # 2) Normalize spacing around commas, hashes, and periods
        #    "Murphy , Texas . 75094" → "Murphy, Texas. 75094"
        addr = _re.sub(r"\s*([,#\.])\s*", r"\1 ", addr)

        # 3) Remove repeated punctuation like ".." or ",," → single instance
        addr = _re.sub(r"\.{2,}", ".", addr)
        addr = _re.sub(r",\s*,+", ", ", addr)

        # 4) Trim stray punctuation/spaces at the edges
        addr = addr.strip(" .,")
        # 5) Collapse any reintroduced doubles
        addr = _re.sub(r"\s+", " ", addr).strip()

        debug_print(f"collect_address: 🧽 Normalized → '{addr}'")

        # ---------- Light validation ----------
        # Must contain at least one letter; and be reasonably long
        # (We do NOT require digits because some addresses are like "PO Box Two")
        if (not addr) or (_re.search(r"[A-Za-z]", addr) is None) or (len(addr) < 6):
            r = session_data[call_sid].get("retry_address", 0) + 1
            session_data[call_sid]["retry_address"] = r
            debug_print(f"collect_address: ❌ looks invalid/too short → retry={r}")

            if r >= 3:
                resp.say(gpt_speak("Sorry, I couldn’t capture your address. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            prompt = (
                "Please repeat your full mailing address — street, city, state, and ZIP. "
                "For example, '118 Briar Oak, Murphy, Texas 75094'."
            )
            gather = make_gather(prompt)
            resp.append(gather)
            try:
                from flask import url_for
                resp.redirect(url_for("voice"))
            except Exception:
                resp.redirect("/voice")
            return str(resp)

        # ---------- Persist ----------
        session_data[call_sid]["customer"]["address"] = addr
        # Reset retry counter on success
        session_data[call_sid].pop("retry_address", None)
        debug_print("collect_address: ✅ Saved address to session")

        # ---------- Advance to next stage ----------
        session_data[call_sid]["stage"] = "collect_cc"
        gather = make_gather("Thank you. Now, please enter your card number, then press pound.")
        resp.append(gather)
        try:
            from flask import url_for
            resp.redirect(url_for("voice"))
        except Exception:
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
        #   - Silent-mode handling + DTMF-enforcement after repeated speech failures.
        # ----------------------------------------------------------------------

        # --- helpers --------------------------------------------------------------
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

        def normalize_spoken_digits(raw: str) -> str:
            """Map spoken words to digits; supports 'double'/'triple' and homophones."""
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

        def _normalize_10(d):
            d = "".join(ch for ch in (d or "") if ch.isdigit())
            return d[1:] if len(d) == 11 and d.startswith("1") else d

        def _gather_dtmf_only(prompt_text: str, num_digits=None):
            """Prefer a DTMF-only Gather; fallback to make_gather on error."""
            try:
                g = Gather(input="dtmf", finishOnKey="#", numDigits=(num_digits or None))
                g.say(prompt_text, voice=VOICE)
                resp.append(g)
            except Exception:
                # Fall back to your wrapper if needed
                resp.append(make_gather(prompt_text))

        def _reprompt(prompt: str, hints: str = ""):
            """Speech/DTMF reprompt with retry cap (separate from silence)."""
            session_data[call_sid]["retry_cc"] += 1
            if session_data[call_sid]["retry_cc"] >= 5:
                debug_print("collect_cc: ⛔ max CC retries. Ending.")
                resp.say(gpt_speak("Sorry, we’re having trouble collecting your details. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return True
            # Respect DTMF-only enforcement
            if session_data[call_sid].get("enforce_dtmf_cc"):
                _gather_dtmf_only(prompt)
            else:
                resp.append(make_gather(prompt, hints=hints))
            resp.say(gpt_speak("I didn't get that."), VOICE)
            resp.redirect("/voice")
            return True

        # Ensure session buckets exist
        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("customer", {})
        customer = session_data[call_sid]["customer"]

        # 🔒 Require phone + DOB before CC
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

        # Retries, speech tries, partial buffer for PAN, last-digit mode, and DTMF enforcement
        session_data[call_sid]["retry_cc"] = session_data[call_sid].get("retry_cc", 0)
        session_data[call_sid]["cc_speech_tries"] = session_data[call_sid].get("cc_speech_tries", 0)
        cc_partial = session_data[call_sid].get("cc_partial", "")
        cc_expect_last_digit = session_data[call_sid].get("cc_expect_last_digit", False)
        enforce_dtmf = session_data[call_sid].get("enforce_dtmf_cc", False)

        # Pull inputs
        try:
            dtmf_digits = (request.values.get("Digits") or "").strip()
        except Exception:
            dtmf_digits = ""
        speech_text = (speech_result or "").strip()
        debug_print(f"collect_cc: 📍 step={cc_step}, DTMF='{dtmf_digits}', speech='{speech_text}'")

        # 🔇 Silent-mode: if both DTMF and speech are empty → reprompt w/out penalizing Luhn tries
        if not dtmf_digits and not speech_text:
            tries = session_data[call_sid].get("silence_cc", 0) + 1
            session_data[call_sid]["silence_cc"] = tries
            debug_print(f"collect_cc: 🤐 silence/no input; tries={tries}")

            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            if cc_step == 1:
                prompt = "Please enter your card number now, then press pound."
                hints  = "zero one two three four five six seven eight nine double triple"
            elif cc_step == 2:
                prompt = "Please enter the expiration as four digits MMYY, then press pound."
                hints  = "zero one two three four five six seven eight nine"
            else:
                prompt = "Please enter the three or four digit security code, then press pound."
                hints  = "zero one two three four five six seven eight nine"

            if enforce_dtmf:
                _gather_dtmf_only(prompt)
            else:
                resp.append(make_gather(prompt, hints=hints))
            resp.redirect("/voice")
            return str(resp)

        # Clear silence counter once we hear something
        session_data[call_sid].pop("silence_cc", None)

        # Prefer DTMF; otherwise convert spoken words to digits
        def get_digits() -> str:
            if enforce_dtmf:
                # If we're enforcing DTMF and none provided, pretend we heard nothing
                if not dtmf_digits:
                    return ""
                return re_mod.sub(r"\D", "", dtmf_digits)
            # Not enforcing: take DTMF if present, else speech→digits
            if dtmf_digits:
                return re_mod.sub(r"\D", "", dtmf_digits)
            return re_mod.sub(r"\D", "", normalize_spoken_digits(speech_text))

        # -------------------------------
        # Step 1: Card Number (13–19)
        # -------------------------------
        if cc_step == 1:
            new_digits = get_digits()

            # Handle the special "expect last digit" path cleanly
            if cc_expect_last_digit:
                if len(new_digits) == 1:
                    digits = (cc_partial or "") + new_digits
                    debug_print(f"collect_cc: 🔚 appended last digit → '{digits}'")
                else:
                    # Treat as a fresh entry attempt
                    debug_print("collect_cc: ℹ️ expected 1 digit, got fresh entry → clearing partial")
                    session_data[call_sid]["cc_partial"] = ""
                    session_data[call_sid]["cc_expect_last_digit"] = False
                    digits = new_digits
            else:
                # Normal path
                digits = new_digits

            # If nothing meaningful yet, reprompt
            if not digits:
                debug_print("collect_cc: ℹ️ no digits heard → reprompt")
                if _reprompt(
                    "Please enter your card number now, then press pound.",
                    hints="zero one two three four five six seven eight nine double triple"
                ): return str(resp)

            # Keep max sane
            if len(digits) > 19:
                digits = digits[:19]

            # If we heard 15 digits via speech (not DTMF), ask for the last single digit
            if not enforce_dtmf and not dtmf_digits and len(digits) == 15:
                session_data[call_sid]["cc_partial"] = digits
                session_data[call_sid]["cc_expect_last_digit"] = True
                debug_print(f"collect_cc: 🧩 Heard 15 digits '{digits}'; asking for the last single digit")
                if _reprompt(
                    "I heard fifteen digits. Please say or type the last single digit now, then press pound.",
                    hints="zero one two three four five six seven eight nine"
                ): return str(resp)

            # Full validation
            if not (13 <= len(digits) <= 19) or not luhn_check(digits):
                # Count speech failures to decide DTMF enforcement
                if not dtmf_digits:  # speech path
                    session_data[call_sid]["cc_speech_tries"] += 1
                escalate = (session_data[call_sid]["cc_speech_tries"] >= 2 and not dtmf_digits)

                debug_print(f"collect_cc: ❌ Invalid card number: '{digits}' (len={len(digits)}), escalate={escalate}")

                # Always clear partial/expect flags on invalid so we don't glue next attempt
                session_data[call_sid]["cc_partial"] = ""
                session_data[call_sid]["cc_expect_last_digit"] = False

                if escalate:
                    # Force DTMF-only from now on for PAN entry
                    session_data[call_sid]["enforce_dtmf_cc"] = True
                    _gather_dtmf_only("That number didn’t sound clear. Please TYPE the full card number now, then press pound.")
                    resp.say(gpt_speak("Please use your keypad."), VOICE)
                    resp.redirect("/voice")
                    return str(resp)
                else:
                    if _reprompt(
                        "That card number doesn't look right. Please re-enter the full card number, then press pound.",
                        hints="zero one two three four five six seven eight nine double triple"
                    ): return str(resp)

            # Save and advance
            customer["cc_number"] = digits
            session_data[call_sid]["cc_step"] = 2
            session_data[call_sid]["cc_partial"] = ""
            session_data[call_sid]["cc_expect_last_digit"] = False
            session_data[call_sid]["cc_speech_tries"] = 0
            debug_print(f"collect_cc: ✅ Saved card number '{digits}'")

            prompt = (
                "Thank you. Now enter the expiration as two digits for month and two digits for year. "
                "For example, 0527. Then press pound."
            )
            if session_data[call_sid].get("enforce_dtmf_cc"):
                _gather_dtmf_only(prompt)
            else:
                resp.append(make_gather(prompt, hints="zero one two three four five six seven eight nine"))
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

            prompt = "Great. Finally, enter the three or four digit security code, then press pound."
            if session_data[call_sid].get("enforce_dtmf_cc"):
                _gather_dtmf_only(prompt)
            else:
                resp.append(make_gather(prompt, hints="zero one two three four five six seven eight nine"))
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
                resp.redirect(url_for("voice"))
            except Exception:
                resp.redirect(request.path)
            return str(resp)







    elif stage == "cancel_appointment":
        # ----------------------------------------------------------------------
        # 🔄 Stage: Cancel Appointment — after the caller says the doctor’s name
        #  1) Try direct partial match against known doctors.
        #  2) If no match, try GPT-based extraction.
        #  3) If still no match, re-prompt (with retry cap).
        #  4) On match, move to phone collection.
        #
        # Notes:
        #  - Includes "silent mode" handling (no speech heard) with its own counter.
        #  - Uses only built-ins already in scope; no local imports.
        #  - Uses make_gather(prompt, hints=...) only (no next_stage arg).
        # ----------------------------------------------------------------------
        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("cancel", {})

        # Safe punctuation set even if 'string' isn't globally imported elsewhere
        try:
            _PUNCT = string.punctuation  # string should be imported at top of file
        except Exception:
            _PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

        def _clean(s: str) -> str:
            """lowercase + strip punctuation + squeeze spaces"""
            s = (s or "").lower().translate(str.maketrans("", "", _PUNCT)).strip()
            # squeeze internal whitespace (defensive against odd STT spacing)
            return " ".join(s.split())

        # Pull speech
        selected_text = (speech_result or "").strip()

        # ------------------------------
        # 🔇 Silent-mode handling first
        # ------------------------------
        if not selected_text:
            tries = session_data[call_sid].get("silence_cancel_doc", 0) + 1
            session_data[call_sid]["silence_cancel_doc"] = tries
            debug_print(f"cancel_appointment: 🤐 No speech detected (silence count={tries})")

            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            doctor_list = ", ".join(googleid_dr_name_map.values())
            retry_prompt = (
                f"I didn't hear the doctor's name. Available doctors are: {doctor_list}. "
                "Please say the name of the doctor whose appointment you want to cancel."
            )
            # keep user in same stage
            session_data[call_sid]["stage"] = "cancel_appointment"
            resp.append(make_gather(retry_prompt, hints=doctor_list))
            return str(resp)

        # If we heard *something*, clear the silence counter
        session_data[call_sid].pop("silence_cancel_doc", None)

        # Normalize and block common junk inputs that aren’t names
        selected_clean = _clean(selected_text)
        debug_print(f"cancel_appointment: 🗣️ Received doctor name → '{selected_clean}'")

        junk_inputs = {
            "", "yes", "no", "yeah", "nope", "ok", "okay", "hello", "hi", "hey",
            "good morning", "good afternoon", "good evening", "test", "i know", "what"
        }
        if (not selected_clean) or (selected_clean in junk_inputs) or (len(selected_clean) < 2):
            doctor_list = ", ".join(googleid_dr_name_map.values())
            retry_prompt = (
                f"I didn't recognize that as a doctor's name. Available doctors are: {doctor_list}. "
                "Please say the name again."
            )
            session_data[call_sid]["stage"] = "cancel_appointment"
            resp.append(make_gather(retry_prompt, hints=doctor_list))
            return str(resp)

        # ------------------------------
        # 1) Partial substring / token match
        # ------------------------------
        matched_id = None
        matched_name = None
        partial_matches = []
        spoken_tokens = set(selected_clean.split())

        for doc_id, friendly_name in googleid_dr_name_map.items():
            friendly_clean = _clean(friendly_name)
            friendly_tokens = set(friendly_clean.split())
            if (
                selected_clean in friendly_clean
                or friendly_clean in selected_clean
                or (spoken_tokens & friendly_tokens)  # token overlap
            ):
                partial_matches.append((doc_id, friendly_name))

        if len(partial_matches) == 1:
            matched_id, matched_name = partial_matches[0]
            debug_print(f"cancel_appointment: ✅ Partial match → {matched_name} ({matched_id})")
        elif len(partial_matches) > 1:
            # If multiple candidates, pick the one with max token overlap
            best = None
            best_overlap = -1
            for doc_id, friendly_name in partial_matches:
                overlap = len(spoken_tokens & set(_clean(friendly_name).split()))
                if overlap > best_overlap:
                    best = (doc_id, friendly_name)
                    best_overlap = overlap
            if best:
                matched_id, matched_name = best
                debug_print(f"cancel_appointment: ✅ Multiple matches; chose best token overlap → {matched_name} ({matched_id})")

        # ------------------------------
        # 2) GPT fallback (only if not matched yet)
        # ------------------------------
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

        # ------------------------------
        # 3) Still no match → retry with cap
        # ------------------------------
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
            resp.append(make_gather(retry_prompt, hints=doctor_list))
            return str(resp)

        # ------------------------------
        # 4) Proceed with matched doctor → next stage: phone number
        # ------------------------------
        session_data[call_sid]["doctor_id"] = matched_id
        session_data[call_sid]["cancel"]["doctor"] = matched_name or googleid_dr_name_map.get(matched_id, "the doctor")
        session_data[call_sid]["stage"] = "cancel_appt_by_phone_number"

        resp.append(make_gather(
            "Thanks. What phone number did you use when booking the appointment?"
        ))
        return str(resp)







    elif stage == "cancel_appt_by_phone_number":
        # ----------------------------------------------------------------------
        # 📞 Collect the phone number used when booking, then move to date+time.
        #  - Silent-mode aware (re-prompts up to 3x if nothing is heard)
        #  - Normalizes to 10 digits (strip non-digits; drop leading 1 if 11)
        #  - Stores under session_data[call_sid]["cancel"]["phone"]
        #  - Next stage: cancel_appt_get_date_time
        # ----------------------------------------------------------------------
        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("cancel", {})

        phone_raw = (speech_result or "").strip()
        debug_print(f"cancel_appt_by_phone_number: 🗣️ raw='{phone_raw}'")

        # 🔇 Silent mode: nothing heard at all
        if not phone_raw:
            tries = session_data[call_sid].get("silence_cancel_phone", 0) + 1
            session_data[call_sid]["silence_cancel_phone"] = tries
            debug_print(f"cancel_appt_by_phone_number: 🤐 silence count={tries}")

            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            prompt = (
                "I didn’t hear your phone number. Please say or type your ten digit phone number including area code, "
                "then press pound."
            )
            session_data[call_sid]["stage"] = "cancel_appt_by_phone_number"
            resp.append(make_gather(prompt, hints="zero one two three four five six seven eight nine double triple"))
            resp.redirect("/voice")
            return str(resp)

        # If we DID hear something, clear the silence counter
        session_data[call_sid].pop("silence_cancel_phone", None)

        # Extract + normalize
        phone = extract_phone_number(phone_raw)  # your helper
        debug_print(f"cancel_appt_by_phone_number: 📱 extracted='{phone}'")

        def _normalize_10(d: str) -> str:
            d = "".join(ch for ch in (d or "") if ch.isdigit())
            return d[1:] if len(d) == 11 and d.startswith("1") else d

        phone10 = _normalize_10(phone)

        # Truncate to 10 if user repeated digits (defensive)
        if len(phone10) > 10:
            phone10 = phone10[:10]

        if len(phone10) != 10:
            debug_print(f"cancel_appt_by_phone_number: ❌ invalid → '{phone10}'")
            session_data[call_sid]["stage"] = "cancel_appt_by_phone_number"
            prompt = (
                "I didn’t catch a valid phone number. Please say or type your ten digit phone number including area code, "
                "then press pound."
            )
            resp.append(make_gather(prompt, hints="zero one two three four five six seven eight nine double triple"))
            resp.redirect("/voice")
            return str(resp)

        # ✅ Store and proceed
        session_data[call_sid]["cancel"]["phone"] = phone10
        session_data[call_sid]["stage"] = "cancel_appt_get_date_time"

        resp.append(make_gather(
            "Thanks. Now, please tell me the date and time of the appointment you want to cancel. "
            "For example, say July 3rd at 9 AM."
        ))
        resp.redirect("/voice")
        return str(resp)



    elif stage == "cancel_appt_get_dob":
        # ----------------------------------------------------------------------
        # 🎂 Collect caller DOB for cancellation lookup/verification.
        #  - Accepts speech (e.g., “July third nineteen fifty six”) or DTMF MMDDYYYY#
        #  - Silent-mode aware (re-prompts up to 3x if nothing is heard)
        #  - Stores ISO under session_data[call_sid]["customer"]["dob"]
        #  - Requires a 10-digit phone on file first
        #  - Next stage: cancel_appt_get_date_time
        # ----------------------------------------------------------------------
        debug_print("cancel_appt_get_dob: 📍 Stage entered")

        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("customer", {})

        # Guard: require phone first
        def _normalize_10(d: str) -> str:
            d = "".join(ch for ch in (d or "") if ch.isdigit())
            return d[1:] if len(d) == 11 and d.startswith("1") else d

        phone_norm = _normalize_10(session_data[call_sid]["customer"].get("phone"))
        if len(phone_norm) != 10:
            debug_print("cancel_appt_get_dob: ❌ phone missing/invalid → collect_phone")
            session_data[call_sid]["return_stage"] = "cancel_appt_get_dob"
            session_data[call_sid]["stage"] = "collect_phone"
            resp.append(make_gather(
                "To cancel your appointment, please provide your ten digit phone number including area code. "
                "You can say it, or type the digits and press pound.",
                hints="zero one two three four five six seven eight nine"
            ))
            resp.redirect("/voice")
            return str(resp)

        # Pull inputs
        try:
            dtmf_digits = (request.values.get("Digits") or "").strip()
        except Exception:
            dtmf_digits = ""
        speech_text = (speech_result or "").strip()
        debug_print(f"cancel_appt_get_dob: 🎙️ speech_text='{speech_text}', 🔢 dtmf_digits='{dtmf_digits}'")

        # 🔇 Silent-mode: nothing heard at all
        if not dtmf_digits and not speech_text:
            tries = session_data[call_sid].get("silence_cancel_dob", 0) + 1
            session_data[call_sid]["silence_cancel_dob"] = tries
            debug_print(f"cancel_appt_get_dob: 🤐 silence count={tries}")

            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            prompt_text = (
                "Please say your birth date, for example July third nineteen fifty six, "
                "or type MMDDYYYY, then press pound."
            )
            session_data[call_sid]["stage"] = "cancel_appt_get_dob"
            try:
                gather = make_gather_dob(prompt_text)
            except Exception:
                gather = make_gather(prompt_text, hints="zero one two three four five six seven eight nine")
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # If we DID hear something, clear the silence counter
        session_data[call_sid].pop("silence_cancel_dob", None)

        # Parse DOB
        dt = parse_dob_input(speech_text, dtmf_digits)  # should return datetime or None
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
                gather = make_gather_dob(prompt_text)
            except Exception:
                gather = make_gather(prompt_text, hints="zero one two three four five six seven eight nine")
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # Validate DOB in a sane range (1900..today)
        try:
            _Date = globals().get("_date", date)  # use top-level alias if present; else built-in 'date'
            today = _Date.today()
            min_date = _Date(1900, 1, 1)
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
                resp.redirect("/voice")
                return str(resp)
        except Exception as e:
            debug_print(f"cancel_appt_get_dob: ⚠️ Validation error → {e}")
            # soft re-prompt (don’t burn a retry here; we already parsed a dt)
            try:
                gather = make_gather_dob(
                    "Please repeat your birth date, for example July third nineteen fifty six, "
                    "or type MMDDYYYY, then press pound."
                )
            except Exception:
                gather = make_gather(
                    "Please repeat your birth date, for example July third nineteen fifty six, "
                    "or type MMDDYYYY, then press pound.",
                    hints="zero one two three four five six seven eight nine"
                )
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # ✅ Store and move on
        iso_dob = dt.strftime("%Y-%m-%d")
        session_data[call_sid]["customer"]["dob"] = iso_dob
        session_data[call_sid].pop("retry_cancel_dob", None)
        debug_print(f"cancel_appt_get_dob: ✅ Stored DOB → {iso_dob}")

        session_data[call_sid]["stage"] = "cancel_appt_get_date_time"
        resp.append(make_gather(
            "Thanks. Now, please tell me the date and time of the appointment you want to cancel. "
            "For example, say July 3rd at 9 AM."
        ))
        resp.redirect("/voice")
        return str(resp)






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
        #   - Silent-mode: re-prompt up to 3x if nothing is heard.
        #   - No inline imports; relies on top-level build/isoparse/timedelta.
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

        # --- Require phone (10-digit) ---------------------------------------------
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

        # --- Get utterance (with silent-mode handling) ----------------------------
        utter = (speech_result or "").strip()
        debug_print(f"cancel_appt_by_time_date: 🗣️ Raw speech → '{utter}'")

        if not utter:
            # 🔇 Silent-mode: re-prompt up to 3x before falling back to iterator
            tries = session_data[call_sid].get("silence_cancel_dt", 0) + 1
            session_data[call_sid]["silence_cancel_dt"] = tries
            debug_print(f"cancel_appt_by_time_date: 🤐 silence/no input; tries={tries}")

            if tries >= 3:
                debug_print("cancel_appt_by_time_date: ⬇️ falling back to iterator after repeated silence")
                cancel_ctx["iter_index"] = 0
                session_data[call_sid]["stage"] = "cancel_appt_iterate"
                resp.append(make_gather("Okay, I’ll list your upcoming appointments."))
                return str(resp)

            # Re-prompt for a date+time
            session_data[call_sid]["stage"] = "cancel_appt_by_time_date"
            prompt = "Please say the date and time of the appointment you want to cancel, for example 'August 15th at 5 AM'."
            resp.append(make_gather(prompt))
            return str(resp)

        # Heard something → clear the silence counter
        session_data[call_sid].pop("silence_cancel_dt", None)

        # --- Parse date+time ------------------------------------------------------
        time_info = smart_parse_time(utter)
        debug_print(f"cancel_appt_by_time_date: 🧠 smart_parse_time → {time_info}")

        if not time_info or not isinstance(time_info, tuple) or len(time_info) != 2:
            debug_print("cancel_appt_by_time_date: ❌ Could not parse date/time → iterate flow")
            cancel_ctx["iter_index"] = 0
            session_data[call_sid]["stage"] = "cancel_appt_iterate"
            resp.append(make_gather("I couldn’t understand the date and time. I’ll list your upcoming appointments."))
            return str(resp)

        spoken_day, spoken_time = time_info
        cancel_ctx["day"] = spoken_day
        cancel_ctx["time"] = spoken_time
        debug_print(f"cancel_appt_by_time_date: 📆 Parsed → Day='{spoken_day}', Time='{spoken_time}'")

        # --- Build UTC window -----------------------------------------------------
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

        # --- Availability (invert logic for cancel) --------------------------------
        try:
            slot_free = is_time_slot_available(calendar_id, appointment_start, appointment_end, creds)
            debug_print(f"cancel_appt_by_time_date: 🔎 is_time_slot_available → {slot_free}")
        except Exception as e:
            debug_print(f"cancel_appt_by_time_date: ⚠️ availability check error → {e}")
            slot_free = True  # fail-open: treat as no event at that time

        if slot_free:
            # No event at that time → nothing to cancel; offer iterate path
            debug_print("cancel_appt_by_time_date: 🚫 Slot FREE → no appointment at that time → iterate")
            cancel_ctx["iter_index"] = 0
            session_data[call_sid]["stage"] = "cancel_appt_iterate"
            resp.append(make_gather("I didn’t find an appointment at that time. I’ll list your upcoming appointments."))
            return str(resp)

        # --- Slot is BUSY → fetch overlapping event(s) to identify event -----------
        try:
            service = build("calendar", "v3", credentials=creds)

            # Pad the search window to catch edge-inclusive overlaps
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

            # Prefer the event whose private.phone10 matches the caller
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

            # Persist chosen event for confirm stage
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
            friendly = (cancel_ctx.get("day") and cancel_ctx.get("time"))
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
