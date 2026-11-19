# update  10/07/25 time_saved  PERFECT
#  
# =========================
# Standard library imports
# =========================

    # ─────────────────────────────────────────────────────────────────────────────
    # Regex anchors for start/end:
    #   ^  → start of the string (or start of a line if re.MULTILINE is enabled)
    #   $  → end of the string (or end of a line if re.MULTILINE is enabled)
    #
    # Notes:
    # - By default (no MULTILINE), ^ matches only at the very start of the *entire* string,
    #   and $ matches at the very end of the *entire* string (or just before a final '
    # - With re.MULTILINE (a.k.a. (?m)), ^ and $ also match at the start/end of *each line*
    #   within a multi-line string.
    # - \A and \Z are absolute anchors: \A = start of entire string, \Z = end of entire string
    #   (these do NOT change with MULTILINE). \z (lowercase) is like \Z but doesn’t allow the
    #   “before final newline” behavior.
    #
    # Examples:
    #   _re.sub(r'^[.,;:]+', '', s)       # remove leading punctuation at the *start of string*
    #   _re.sub(r'[.,;:]+$', '', s)       # remove trailing punctuation at the *end of string*
    #   _re.sub(r'^\s+|\s+$', '', s)      # trim leading/trailing whitespace (string-level)
    #
    #   # Line-by-line (multi-line) versions:
    #   _re.sub(r'^[.,;:]+', '', s, flags=_re.MULTILINE)  # remove leading punctuation per line
    #   _re.sub(r'[.,;:]+$', '', s, flags=_re.MULTILINE)  # remove trailing punctuation per line
    #
    # Clarification about $:
    # - Without MULTILINE, $ matches at the end of the string *or* right before a final '\n'
    #   If you need a true “absolute end” even when there’s a trailing newline, use \Z or \z.
    # ─────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────
    # Regex quantifier `+`

    # - Means “ONE OR MORE” repetitions of the preceding token (greedy by default).
    #   Examples:
    #     r"a+"       → "a", "aa", "aaa", ...
    #     r"\d+"      → one or more digits, e.g., "7", "1956", "12345"
    #     r"(ab)+"    → "ab", "abab", "ababab", ...
    # - Greedy vs lazy:
    #     r".+"   → match as much as possible
    #     r".+?"  → match as little as possible (lazy)
    # - Inside a character class [...] the `+` has NO special meaning; it’s a literal plus.
    #   (In your patterns the `+` appears *after* a character class, so it’s the quantifier.)
    # - To match a literal plus outside a character class, escape it: r"\+"
    # ─────────────────────────────────────────────────────────────────────────────        
    # About `\s*` in the regex:
    # - `\s`  matches any whitespace character (space, tab, newline, carriage return, form feed, vertical tab).
    #          In Python 3 it’s Unicode-aware, so it also matches non-ASCII spaces.
    # - `*`   is the “zero-or-more” quantifier (greedy by default).
    # - `\s*` therefore matches ZERO OR MORE whitespace chars.
    #
    # Why it matters here:
    #   (a\s*\.?\s*m\.?) will match all of these as "am":
    #     "am"           → \s* matches zero spaces
    #     "a m"          → \s* matches one space
    #     "a    m"       → \s* matches multiple spaces
    #     "a. m"         → \s* after the dot matches one space
    #     "a.m"          → both \s* match zero spaces
    #     "A.    M."     → case-insensitive; \s* matches many spaces
    #
    # Tips:
    # - If you need "one or more" spaces, use `\s+`.
    # - If you need an "optional single" space, use `\s?`.
    # - If you want to allow only ASCII spaces (not tabs/newlines), use `[ ]*` (a literal space in a char class).
    # - `\s*` can also match newlines; if you want to avoid crossing lines, consider replacing `\s*` with `[ ]*`.


import os
import json
import string          # for string.punctuation
import calendar
import re as _re       # use _re everywhere to avoid UnboundLocalError
import pickle
import openai
import calendar as _calendar
import dateparser as _dp
import dateparser
import pytz as _pytz
import pytz as _TZMOD
import time as _time_mod
import threading




from uuid import uuid4
from datetime import datetime as _dt
from datetime import time as dtime
from typing import Any, Optional, List, Dict, Tuple, Iterator, Iterable, Union
from datetime import datetime, date, time, timedelta, timezone, time as _time
from datetime import datetime as _Datetime, timezone as _tz
from datetime import datetime as _dt  # if code references _dt
from datetime import datetime as _dt_local, date as _date_local
from dateutil import parser as dtparser
from dateutil.parser import isoparse
from dateutil.tz import gettz
from dateutil.tz import gettz as _gettz
from dotenv import load_dotenv, find_dotenv

# Load .env from the current directory (or nearest parent)
load_dotenv(find_dotenv())   # returns True/False if a file was loaded

# Now read values
CLINIC_TZ = os.getenv("CLINIC_TZ", "America/Chicago")
SPEECH_INPUT_DURATION = int(os.getenv("SPEECH_INPUT_DURATION", 12))



# =========================
# Third-party libraries
# =========================

from googleapiclient.discovery import build
from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow  # keep only if you actually use OAuth user flow
from google.auth.transport.requests import Request

from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient
from twilio.rest import Client
from dateutil.parser import parse as _dtparse
from string import punctuation as _PUNCT
from datetime import datetime as _Datetime, timezone as _Tz
from functools import wraps

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

# ----------------------------------------------------------------------
# 🌍 Global speech hints for Arabic + English doctor names
# Used across multiple stages (booking, collect_first_name, etc.)
# ----------------------------------------------------------------------
ARABIC_NAME_HINTS = """
    Ahmed, Ahmad, Mohamed, Muhammad, Faten, Fatma, Mariam, Aisha, Lina, Huda,
    Nour, Dalia, Layla, Youssef, Ali, Hassan, Khaled, Fady, Karim, Samir,
    Taha, Salama, Rashed, Mostafa, Abdallah, Ramadan, Said, Farid, Kamel,
    Lotfi, Saleh, Mansour, Ismail, Mahmoud, Omar, Fawzi, Zaki, Hussein, Attia, Morsi
"""



app = Flask(__name__)
app.url_map.strict_slashes = False
load_dotenv()
# Environment & API setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_NUMBER")
GOOGLE_CREDENTIALS = "credentials.json"

# How long of silence ends a speech phrase (seconds). Use "auto" if you prefer VAD.
SPEECH_INPUT_DURATION = os.getenv("SPEECH_INPUT_DURATION", "6")  # keep as string for Twilio
# How long Twilio waits for the first input AND between DTMF digits (seconds)
PAUSE_BETWEEN_DIGITS = int(os.getenv("PAUSE_BETWEEN_DIGITS", "7"))
# Max seconds for <Record> (voicemail, freeform notes)
MAX_RECORD_TIME = int(os.getenv("MAX_RECORD_TIME", "60"))

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

DB_FOLDER = "appointment_data"
DB_FILE   = os.path.join(DB_FOLDER, "customers.json")  # human-readable, not JSON
# Global working config
# 2) Read from env, with a safe default
CLINIC_TZ = os.getenv("CLINIC_TZ", "America/Chicago")
#from datetime import time
WORKING_DAYS = [int(x) for x in os.getenv("WORKING_DAYS", "0,1,2,3,4").split(",") if x.strip().isdigit()]

WORKING_HOURS_START = int(os.getenv("WORKING_HOURS_START", 8))
WORKING_HOURS_END   = int(os.getenv("WORKING_HOURS_END", 17))

LUNCH_BREAK_START = time(
    int(os.getenv("LUNCH_BREAK_START_H", 13)),
    int(os.getenv("LUNCH_BREAK_START_M", 0))
)
LUNCH_BREAK_END = time(
    int(os.getenv("LUNCH_BREAK_END_H", 14)),
    int(os.getenv("LUNCH_BREAK_END_M", 0))
)

"""
WORKING_DAYS=0,1,2,3,4
WORKING_HOURS_START=8
WORKING_HOURS_END=17
LUNCH_BREAK_START_H=13
LUNCH_BREAK_START_M=0
LUNCH_BREAK_END_H=14
LUNCH_BREAK_END_M=0

"""
WORKING_DAYS = [int(x) for x in os.getenv("WORKING_DAYS", "0,1,2,3,4").split(",") if x.strip().isdigit()]

WORKING_HOURS_START = int(os.getenv("WORKING_HOURS_START", 8))
WORKING_HOURS_END   = int(os.getenv("WORKING_HOURS_END", 17))

LUNCH_BREAK_START = time(
    int(os.getenv("LUNCH_BREAK_START_H", 13)),
    int(os.getenv("LUNCH_BREAK_START_M", 0))
)
LUNCH_BREAK_END = time(
    int(os.getenv("LUNCH_BREAK_END_H", 14)),
    int(os.getenv("LUNCH_BREAK_END_M", 0))
)
SESSION_TIME = int(os.getenv("SESSION_TIME", 30))

USE_GPT = False
DEBUG  = True

# ---- Country switch (US by default; set to "EG" to favor Egypt) ----
COUNTRY = os.getenv("COUNTRY", "US").upper()   # e.g., export COUNTRY=EG


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

# ===== Env-driven defaults (module-level) =====



def _append_stage_to_action(action: Optional[str], next_stage: Optional[str]) -> str:
    """Back-compat: if next_stage is provided, append ?stage=... to action."""
    base = action or "/voice"
    if next_stage:
        sep = "&" if "?" in base else "?"
        return f"{base}{sep}stage={next_stage}"
    return base


def make_gather(
    prompt: str,
    *,
    next_stage: Optional[str] = None,               # ← back-compat
    hints: Optional[str] = None,
    input: str = "speech dtmf",
    num_digits: Optional[int] = None,
    timeout: int = PAUSE_BETWEEN_DIGITS,            # ← default from ENV
    speech_timeout: str = SPEECH_INPUT_DURATION,    # ← default from ENV ("auto" or seconds string)
    finish_on_key: str = "#",
    barge_in: bool = True,
    language: str = "en-US",
    action: Optional[str] = "/voice",
    method: str = "POST",
):
    """
    Build and RETURN a Twilio <Gather> with ENV-driven defaults.

    Backward compatible:
      - Accepts next_stage and appends it as '?stage=...' to action.
      - Returns the <Gather> so callers can `resp.append(make_gather(...))`.

    Notes:
      - timeout controls DTMF first-digit / inter-digit wait.
      - speech_timeout controls how long STT waits for silence ("auto" or seconds).
      - language can be 'en-US', 'ar-EG', etc.
      - hints can include multiline Arabic/English name lists.
    """
    # Normalize speechTimeout
    _speech_timeout = int(speech_timeout) if str(speech_timeout).isdigit() else speech_timeout
    _num_digits = num_digits if (isinstance(num_digits, int) and num_digits > 0) else None
    _action = _append_stage_to_action(action, next_stage)

    # 🧠 Normalize hints (flatten multiline → comma-separated)
    _hints = None
    if hints:
        _hints = ", ".join(line.strip() for line in hints.splitlines() if line.strip())

    try:
        g = Gather(
            input=input,
            action=_action,
            method=method,
            timeout=int(timeout),
            speechTimeout=_speech_timeout,
            finishOnKey=finish_on_key,
            numDigits=_num_digits,
            hints=_hints,
            language=language,
            bargeIn=barge_in,
        )
        g.say(gpt_speak(prompt), voice=VOICE)
        return g

    except Exception as e:
        debug_print(f"make_gather: ⚠️ failed to build Gather → {e}")
        # Fallback to ensure the prompt still speaks
        try:
            g = Gather(input=input, action=_action, method=method)
            g.say(gpt_speak(prompt), voice=VOICE)
            return g
        except Exception:
            debug_print(f"make_gather: ❌ secondary fallback failed → {e}")
            return None









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



def is_time_slot_available(calendar_id: str, start_iso: str, end_iso: str, creds) -> bool:
    """
    Return True if [start, end) is free on Google Calendar.
    Aligns with how cancel_appt_iterate builds candidates.
    """
    def _as_utc_dt(s: str):
        s2 = s.replace("Z", "+00:00")
        dt = isoparse(s2)
        return dt if dt.tzinfo else dt.replace(tzinfo=_pytz.UTC)

    start_dt = _as_utc_dt(start_iso).astimezone(_pytz.UTC)
    end_dt   = _as_utc_dt(end_iso).astimezone(_pytz.UTC)

    if end_dt <= start_dt:
        return False

    service = build("calendar", "v3", credentials=creds)

    # 🔍 Use events().list with exact window (same as iterate JSON entries)
    ev = service.events().list(
        calendarId=calendar_id,
        timeMin=start_dt.isoformat().replace("+00:00", "Z"),
        timeMax=end_dt.isoformat().replace("+00:00", "Z"),
        singleEvents=True,
        orderBy="startTime",
        maxResults=5,
    ).execute()

    items = ev.get("items", [])
    for it in items:
        estart_raw = it.get("start", {}).get("dateTime") or it.get("start", {}).get("date")
        eend_raw   = it.get("end", {}).get("dateTime") or it.get("end", {}).get("date")
        if not (estart_raw and eend_raw):
            continue
        estart = _as_utc_dt(estart_raw)
        eend   = _as_utc_dt(eend_raw)
        # Same overlap check as iterate
        if not (end_dt <= estart or eend <= start_dt):
            return False  # Busy

    return True  # Free





def get_next_available_slots(
    calendar_id: str,
    creds,
    *,
    from_start_iso: str,
    duration_minutes: int = None,
    limit: int = 3,
    tz_name: str = None,
    work_hours=None,
    slot_step_minutes: int = None,
    search_days: int = None
) -> list:
    """Return up to `limit` future UTC slots strictly after from_start_iso."""

    def _dbg(msg: str):
        try: debug_print(msg)
        except Exception: print(msg)

    _dbg(f"get_next_available_slots: ▶️ cal={calendar_id} from={from_start_iso} limit={limit}")

    if duration_minutes is None:
        duration_minutes = int(globals().get("APPOINTMENT_DURATION_MINUTES", 30))
    if duration_minutes not in (15, 30, 45, 60):
        duration_minutes = 30
    if slot_step_minutes is None:
        slot_step_minutes = duration_minutes

    if tz_name is None:
        tz_name = globals().get("CLINIC_TZ", "America/Chicago")
    try:
        tz_local = _pytz.timezone(tz_name)
    except Exception:
        tz_local = _pytz.timezone("America/Chicago")

    WSTART = int(globals().get("WORKING_HOURS_START", 8))
    WEND   = int(globals().get("WORKING_HOURS_END", 17))
    if not work_hours:
        work_hours = ((WSTART, WEND),)
    WORKING_DAYS = set(int(x) for x in globals().get("WORKING_DAYS", {0,1,2,3,4}))

    # Lunch break
    def _as_time(val, default_h=None, default_m=0):
        if val is None: return None if default_h is None else dtime(default_h, default_m)
        if isinstance(val, dtime): return val
        s = str(val).strip()
        if not s: return None
        if ":" in s: hh, mm = (s.split(":", 1) + ["0"])[:2]
        else: hh, mm = s, "0"
        try: return dtime(int(hh), int(mm))
        except Exception: return None

    LUNCH_START = _as_time(globals().get("LUNCH_BREAK_START"))
    LUNCH_END   = _as_time(globals().get("LUNCH_BREAK_END"))
    if search_days is None:
        search_days = int(globals().get("SEARCH_DAYS", 14))

    def _friendly(dt_local, now_local):
        try:
            if dt_local.year != now_local.year:
                return dt_local.strftime("%A, %B %-d, %Y at %-I:%M %p")
            return dt_local.strftime("%A, %B %-d at %-I:%M %p")
        except Exception:
            return dt_local.strftime("%A, %B %d at %I:%M %p")

    def _align_up_to_window_grid(dt_local, minutes, window_start_local, *, now_local):
        dt_local = dt_local.replace(second=0, microsecond=0)
        anchor   = window_start_local.replace(second=0, microsecond=0)
        diff_min = int((dt_local - anchor).total_seconds() // 60)
        if diff_min <= 0:
            aligned = anchor
        else:
            rem = diff_min % minutes
            aligned = dt_local if rem == 0 else (dt_local + timedelta(minutes=(minutes - rem)))
        if aligned.date() == now_local.date() and aligned <= now_local:
            steps = ((now_local - anchor).total_seconds() // 60 // minutes) + 1
            aligned = anchor + timedelta(minutes=int(steps * minutes))
        return aligned

    # --- UTC baselines ---
    now_utc = datetime.now(_pytz.UTC)
    now_loc = now_utc.astimezone(tz_local)

    try:
        req_utc = isoparse((from_start_iso or "").strip())
        if req_utc.tzinfo is None:
            req_utc = _pytz.UTC.localize(req_utc)
    except Exception:
        req_utc = now_utc
    req_local = req_utc.astimezone(tz_local)

    search_window_start = now_utc
    search_window_end   = now_utc + timedelta(days=search_days)
    base_utc = req_utc if (search_window_start <= req_utc <= search_window_end) else now_utc
    cur_local = base_utc.astimezone(tz_local)

    results, seen = [], set()

    while cur_local.astimezone(_pytz.UTC) < search_window_end and len(results) < limit:
        if cur_local.weekday() not in WORKING_DAYS:
            cur_local = cur_local + timedelta(days=1)
            cur_local = cur_local.replace(hour=WSTART, minute=0, second=0, microsecond=0)
            continue

        windows = []
        for ws, we in work_hours:
            wstart = tz_local.localize(datetime(cur_local.year, cur_local.month, cur_local.day, ws, 0))
            wend   = tz_local.localize(datetime(cur_local.year, cur_local.month, cur_local.day, we, 0))
            windows.append((wstart, wend))

        progressed = False
        for wstart, wend in windows:
            if cur_local < wstart:
                cur_local = wstart
            cur_local = _align_up_to_window_grid(cur_local, slot_step_minutes, wstart, now_local=now_loc)

            while cur_local + timedelta(minutes=duration_minutes) <= wend and len(results) < limit:
                if LUNCH_START and LUNCH_END:
                    if cur_local.time() < LUNCH_END and (cur_local + timedelta(minutes=duration_minutes)).time() > LUNCH_START:
                        cur_local = tz_local.localize(datetime.combine(cur_local.date(), LUNCH_END))
                        cur_local = _align_up_to_window_grid(cur_local, slot_step_minutes, wstart, now_local=now_loc)
                        continue

                start_iso = cur_local.astimezone(_pytz.UTC).isoformat().replace("+00:00", "Z")
                end_iso   = (cur_local + timedelta(minutes=duration_minutes)).astimezone(_pytz.UTC).isoformat().replace("+00:00", "Z")

                assert start_iso.endswith("Z"), "Slot must be UTC"

                try:
                    if is_time_slot_available(calendar_id, start_iso, end_iso, creds) and start_iso not in seen:
                        seen.add(start_iso)
                        results.append({
                            "start": start_iso,
                            "end": end_iso,
                            "friendly": _friendly(cur_local, now_loc),
                            "tz": tz_name,
                        })
                        _dbg(f"get_next_available_slots: ✅ add {results[-1]['friendly']}")
                except Exception as e:
                    _dbg(f"get_next_available_slots: ❌ slot_check error → {e}")

                cur_local = cur_local + timedelta(minutes=slot_step_minutes)
                progressed = True

        if not progressed:
            cur_local = cur_local + timedelta(days=1)
            cur_local = cur_local.replace(hour=WSTART, minute=0, second=0, microsecond=0)

    _dbg(f"get_next_available_slots: ✅ suggestions={len(results)}")
    return results











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











# ===== local doctor JSON cancellation (by doctor+phone+dob+utc_start) =====

#  remove phone10 

def cancel_appointment_for_dr_name(
    doctor_name: str,
    phone: str,
    dob: str,
    utc_start: str,
    *,
    default_country: str = COUNTRY  # use your global default (e.g., "US" or "EG")
) -> bool:
    """
    Remove a single appointment from appointment_data/doctors/<doctor>.json
    matching ALL of:
      • phone  → E.164 ('+<cc><nsn>') **only**
      • dob    → exact string match; expected ISO 'YYYY-MM-DD'
      • time   → exact UTC ISO match (after normalization)

    Returns True if a record was removed, else False.

    Notes:
      - Input `phone` can be already E.164; otherwise we normalize with `normalize_phone_e164`.
      - Records may be mixed (older ones may only have 'phone' digits). We *derive* an E.164
        form per record when needed (US/EG supported) and compare E.164 ↔ E.164 only.
    """

    # ---------- normalize input phone to E.164 ----------
    raw = (phone or "").strip()
    phone_e164 = ""
    if raw.startswith("+") and raw[1:].replace(" ", "").isdigit():
        phone_e164 = "+" + raw[1:].replace(" ", "")
    else:
        try:
            phone_e164 = normalize_phone_e164(raw, (default_country or "US").upper()) or ""
            if not phone_e164:
                # try the other supported country as a last resort
                alt = "EG" if (default_country or "US").upper() != "EG" else "US"
                phone_e164 = normalize_phone_e164(raw, alt) or ""
        except Exception:
            phone_e164 = ""

    dob_str = (dob or "").strip()
    full_path = get_doctor_filename(doctor_name)

    debug_print(
        f"cancel_appointment_by_name: doctor='{doctor_name}' "
        f"phone_e164='{phone_e164 or '∅'}' dob='{dob_str or '∅'}' utc='{utc_start or '∅'}'"
    )

    if not (os.path.exists(full_path) and phone_e164 and dob_str and utc_start):
        return False

    # ---------- load list ----------
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return False
    except Exception as e:
        debug_print(f"cancel_appointment_by_name: read error → {e}")
        return False

    # ---------- normalize times to comparable UTC ISO (no micros) ----------
    def _to_utc_iso(s: str) -> str:
        dt = dtparser.isoparse(s)
        if dt.tzinfo is None:
            # treat naive as UTC
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    try:
        target_norm = _to_utc_iso(utc_start)
    except Exception as e:
        debug_print(f"cancel_appointment_by_name: utc parse error → {e}")
        return False

    # ---------- helper: derive E.164 for a stored appt record ----------
    def _appt_e164(appt: dict) -> str:
        # Prefer explicit E.164 field
        pe = (appt.get("phone_e164") or "").strip()
        if pe.startswith("+") and pe[1:].replace(" ", "").isdigit():
            return "+" + pe[1:].replace(" ", "")

        # Try normalizing whatever is in 'phone' using our helper
        cand = (appt.get("phone") or "").strip()
        if cand:
            e164 = ""
            try:
                e164 = normalize_phone_e164(cand, (default_country or "US").upper()) or ""
                if not e164:
                    alt = "EG" if (default_country or "US").upper() != "EG" else "US"
                    e164 = normalize_phone_e164(cand, alt) or ""
            except Exception:
                e164 = ""
            if e164:
                return e164

        # Nothing usable
        return ""

    kept = []
    removed = 0

    for appt in data:
        if not isinstance(appt, dict):
            kept.append(appt)
            continue

        ap_e164 = _appt_e164(appt)
        ap_dob  = (appt.get("dob", "") or "").strip()
        ap_time_raw = (appt.get("time") or appt.get("start") or "").strip()

        try:
            ap_time_norm = _to_utc_iso(ap_time_raw) if ap_time_raw else ""
        except Exception:
            kept.append(appt)
            continue

        if ap_e164 == phone_e164 and ap_dob == dob_str and ap_time_norm == target_norm:
            removed += 1
        else:
            kept.append(appt)

    if removed == 0:
        debug_print("cancel_appointment_by_name: no matching record found")
        return False

    # ---------- write back ----------
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

#  independent of phone10  and dpende only on e164
def get_upcoming_events(
    calendar_id: str,
    phone: str,
    utc_start: str,
    utc_end: str,
    creds,
    debug: bool = False,
    *,
    default_country: str = COUNTRY  # Use your global COUNTRY ('US' or 'EG', etc.)
):
    """
    Search a specific Google Calendar for events within a given UTC time window
    and return the first event that matches the caller's **E.164** phone number.

    E.164-ONLY BEHAVIOR:
      - We accept only a valid E.164 phone (e.g., '+12025550123', '+201234567890').
      - Matching is done against:
          (a) event.extendedProperties.private.phone_e164  (exact string), or
          (b) event.description containing the exact E.164 string.
      - No legacy 10-digit or digit-only normalization is performed.

    Arguments:
    ----------
    calendar_id : str
        The Google Calendar ID (e.g., "doctor@example.com").
    phone : str
        The caller's phone number; will be normalized to E.164 using normalize_phone_e164.
    utc_start : str
        ISO 8601 UTC start time of the search window (e.g., "2025-08-07T14:00:00Z").
    utc_end : str
        ISO 8601 UTC end time of the search window.
    creds :
        Authenticated Google API credentials.
    debug : bool, optional
        If True, prints detailed debug logs for troubleshooting.
    default_country : str, keyword-only
        Country hint for normalization (e.g., 'US' or 'EG').

    Returns:
    --------
    dict or None
        The first matching Google Calendar event (full event dict) if found,
        otherwise None.
    """

    # --- 1) Normalize input to strict E.164 -----------------------------------
    def _is_e164(s: str) -> bool:
        return bool(_re.fullmatch(r"\+\d{6,15}", (s or "").strip()))

    raw = (phone or "").strip()
    phone_e164 = raw if _is_e164(raw) else ""

    if not phone_e164:
        try:
            # Your helper should convert national formats -> E.164 or return ''.
            phone_e164 = normalize_phone_e164(raw, default_country) or ""
        except Exception:
            phone_e164 = ""

    if not phone_e164 or not _is_e164(phone_e164):
        if debug:
            debug_print(f"get_upcoming_events: ❌ invalid/non-E.164 phone '{phone}'")
        return None

    # --- 2) Debug parameters ---------------------------------------------------
    if debug:
        debug_print(f"📅 get_upcoming_events: calendar={calendar_id}")
        debug_print(f"⏱️ window: {utc_start} → {utc_end}")
        debug_print(f"📞 match E.164: {phone_e164}")

    # --- 3) Fetch events in the window ----------------------------------------
    events = list_events_in_window_utc(calendar_id, creds, utc_start, utc_end, debug=debug)

    if debug:
        debug_print(f"🔍 get_upcoming_events: {len(events)} event(s) fetched in window")

    # --- 4) Find first event that matches E.164 --------------------------------
    for ev in events:
        # Prefer an explicit structured field in extendedProperties.private
        priv = ((ev.get("extendedProperties") or {}).get("private") or {})
        ev_phone_e164 = (priv.get("phone_e164") or "").strip()

        if ev_phone_e164 == phone_e164:
            if debug:
                s = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date")
                debug_print(f"✅ match via extendedProperties.private.phone_e164 → {ev_phone_e164}; start={s}")
            return ev

        # Fallback: exact E.164 string embedded in description (no digit-only matching)
        desc = (ev.get("description") or "").strip()
        if phone_e164 and phone_e164 in desc:
            if debug:
                s = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date")
                debug_print(f"✅ match via description contains E.164 → {phone_e164}; start={s}")
            return ev

        if debug:
            s = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date")
            debug_print(f"… no match: summary={ev.get('summary')} start={s}")

    # --- 5) Nothing matched ----------------------------------------------------
    if debug:
        debug_print("❌ No matching event found for E.164 phone.")
    return None





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

    E.164-only migration:
      - Re-key to (phone_e164|dob) if phone_e164 valid.
      - Adopt valid E.164 left keys.
      - Add timestamps if missing.
      - Never guess legacy 10-digit numbers.
    """
    os.makedirs(DB_FOLDER, exist_ok=True)
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2, ensure_ascii=False)
        return

    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("customers.json must be a JSON object")
    except Exception:
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2, ensure_ascii=False)
        return

    changed = False
    migrated = 0
    ensured_ts = 0
    adopted_from_key = 0
    skipped_non_e164 = 0

    # ✅ ensure _re defined
    import re as _re

    def _is_e164(s: str) -> bool:
        s = (s or "").strip()
        return bool(_re.fullmatch(r"\+\d{6,15}", s))

    def _e164_or_empty(s: str) -> str:
        s = (s or "").strip().replace(" ", "")
        return s if _is_e164(s) else ""

    try:
        new_data = {}
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for old_key, rec in data.items():
            if not isinstance(rec, dict):
                new_data[old_key] = rec
                continue

            if not rec.get("created_at") or not rec.get("last_seen_at"):
                rec.setdefault("created_at", now)
                rec.setdefault("last_seen_at", now)
                ensured_ts += 1
                changed = True

            rec["dob"] = _oneline(rec.get("dob", ""))

            phone_e164 = _e164_or_empty(rec.get("phone_e164", ""))
            if not phone_e164 and "|" in old_key:
                left = old_key.split("|", 1)[0].strip()
                left_e164 = _e164_or_empty(left)
                if left_e164:
                    rec["phone_e164"] = left_e164
                    phone_e164 = left_e164
                    adopted_from_key += 1
                    changed = True

            final_key = old_key
            if phone_e164:
                try:
                    final_key = _key(phone_e164, rec.get("dob", ""))
                except Exception:
                    final_key = old_key

            if final_key != old_key:
                if final_key not in new_data:
                    new_data[final_key] = rec
                    migrated += 1
                    changed = True
                else:
                    try:
                        new_data[final_key]["last_seen_at"] = now
                    except Exception:
                        pass
            else:
                new_data[old_key] = rec
                if not phone_e164:
                    skipped_non_e164 += 1

        if changed:
            tmp = DB_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(new_data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, DB_FILE)

        debug_print(
            "init_db (E.164-only): "
            f"migrated={migrated}, adopted_from_key={adopted_from_key}, "
            f"ensured_ts={ensured_ts}, skipped_non_e164={skipped_non_e164}, changed={changed}"
        )

    except Exception as e:
        debug_print(f"init_db: ⚠️ migration skipped due to error: {e}")
        return

    







#   remove phone10 and make dependent on e146

# ---------- Sanitizers / formatters ----------
def _oneline(s: str) -> str:
    """Compact whitespace/newlines to a single line."""
    return _re.sub(r"\s+", " ", (s or "").strip())




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





# ---------- Public API ----------

# (Legacy helper removed)  _normalize_phone10 → ❌ gone (E.164 only now)

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

def _key(phone_e164: str, dob_iso: str) -> str:
    """Stable map key: E.164 + DOB ISO."""
    return f"{(phone_e164 or '').strip()}|{(dob_iso or '').strip()}"





def customer_search(
    phone_number: str = None,
    dob: str = "",
    *,
    default_country: str = COUNTRY,
    phone: str = None,   # backward-compatible alias
) -> bool:
    """
    Check if a customer exists in customers.json by (phone | DOB).

    ✅ Simplified, English-only version
       - Uses only default_country (no extra parameters)
       - Normalizes phone → E.164
       - Normalizes DOB → YYYY-MM-DD
       - Logs each step for debugging
       - Returns True if record found, else False
    """
    debug_print("─────────────────────────────")
    debug_print(f"customer_search: ▶️ INPUTS → phone_number='{phone_number}', phone(alias)='{phone}', dob='{dob}', default_country='{default_country}'")

    # ----------------------------------------------------------------------
    # Load database
    # ----------------------------------------------------------------------
    try:
        init_db()
        data = _load_customers()
        debug_print(f"customer_search: 📂 Loaded {len(data)} records from {DB_FILE}")
    except Exception as e:
        debug_print(f"customer_search: ❌ Failed to load DB → {e}")
        return False

    # ----------------------------------------------------------------------
    # Normalize phone number → E.164
    # ----------------------------------------------------------------------
    raw = (phone_number if phone_number else phone or "").strip().replace(" ", "")
    debug_print(f"customer_search: ☎️ Raw phone input = '{raw}'")

    phone_e164 = ""
    try:
        if raw.startswith("+") and raw[1:].isdigit():
            phone_e164 = raw
        else:
            phone_e164 = normalize_phone_e164(raw, default_country) or ""
            if not phone_e164:
                # fallback try opposite country (for cross-region callers)
                alt_country = "US" if default_country.upper() != "US" else "EG"
                phone_e164 = normalize_phone_e164(raw, alt_country) or ""
    except Exception as e:
        debug_print(f"customer_search: ⚠️ normalize_phone_e164 error → {e}")

    # Fallback pseudo E.164 if still invalid
    if not phone_e164:
        digits = "".join(ch for ch in raw if ch.isdigit())
        if len(digits) >= 8:
            phone_e164 = f"+000{digits[-10:]}"
            debug_print(f"customer_search: ⚠️ fallback pseudo-E.164 → '{phone_e164}'")

    if not phone_e164:
        debug_print("customer_search: ❌ No valid phone number after normalization")
        return False

    debug_print(f"customer_search: ✅ normalized phone → {phone_e164}")

    # ----------------------------------------------------------------------
    # Normalize DOB → YYYY-MM-DD
    # ----------------------------------------------------------------------
    dob_str = (dob or "").strip()
    if not dob_str:
        debug_print("customer_search: ⚠️ Empty DOB → using 'unknown'")
        dob_str = "unknown"
    else:
        import re as _re
        dob_str = dob_str.replace("/", "-").replace(".", "-")
        try:
            # matches YYYY-MM-DD or MM-DD-YYYY
            m1 = _re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", dob_str)
            m2 = _re.fullmatch(r"(\d{1,2})-(\d{1,2})-(\d{4})", dob_str)
            if m1:
                yyyy, mm, dd = m1.groups()
            elif m2:
                mm, dd, yyyy = m2.groups()
            else:
                raise ValueError("Unrecognized DOB format")
            dob_str = f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}"
        except Exception as e:
            debug_print(f"customer_search: ⚠️ DOB normalization failed ({dob_str}) → {e}")
            return False

    debug_print(f"customer_search: 🎂 normalized DOB → {dob_str}")

    # ----------------------------------------------------------------------
    # Build lookup key
    # ----------------------------------------------------------------------
    key = _key(phone_e164, dob_str)
    debug_print(f"customer_search: 🔑 lookup key = '{key}'")

    # ----------------------------------------------------------------------
    # Lookup in database
    # ----------------------------------------------------------------------
    if key in data:
        debug_print(f"customer_search: ✅ FOUND exact match → {key}")
        debug_print("─────────────────────────────")
        return True

    # Try simple alternate forms (to avoid leading zeros or spacing issues)
    alt_keys = [
        _key(phone_e164.replace("+", ""), dob_str),
        _key(phone_e164, dob_str.strip()),
    ]
    for alt in alt_keys:
        if alt in data:
            debug_print(f"customer_search: ✅ FOUND via alternate key '{alt}'")
            debug_print("─────────────────────────────")
            return True
        else:
            debug_print(f"customer_search: 🔍 alt_key '{alt}' not found")

    debug_print(f"customer_search: 🚫 No match for phone={phone_e164}, dob={dob_str}")
    if len(data) > 0:
        debug_print(f"customer_search: 🗝️ Sample keys → {list(data.keys())[:3]}")
    debug_print("─────────────────────────────")
    return False









def _save_customers(data: Dict[str, Dict[str, Any]]) -> None:
    """Write the customers map to disk in readable (pretty) form."""
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



#   update to remove legacy phone10

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

      • If (phone|dob) exists → update record fields + bump 'last_seen_at'; return False.
      • If new → create record with 'created_at' + 'last_seen_at'; return True.

    This version guarantees:
      ✅ Always writes valid record even if dob or country normalization fails.
      ✅ Never skips due to missing country context.
      ✅ Explicitly logs both normalization and insertion result.

    PHONE FORMAT:
      • Attempts E.164 normalization (US or EG). If normalization fails, falls back
        to '+000' + digits to preserve the data instead of aborting.
    """
    # ----------------------------------------------------------------------
    # 🧩 Step 1: Ensure DB is ready
    # ----------------------------------------------------------------------
    init_db()

    # ----------------------------------------------------------------------
    # 🧩 Step 2: Normalize phone (strict E.164, but with fallback)
    # ----------------------------------------------------------------------
    phone_e164 = normalize_phone_e164(phone, globals().get("COUNTRY", "US"))
    if not phone_e164:
        debug_print(f"insert_customer: ⚠️ invalid phone '{phone}' → fallback to raw digits")
        digits_only = "".join(ch for ch in str(phone) if ch.isdigit())
        if len(digits_only) >= 8:
            phone_e164 = f"+000{digits_only[-10:]}"  # store anyway to avoid skipping
        else:
            raise ValueError("insert_customer: invalid phone, cannot insert")

    dob_iso = (dob or "").strip() or "unknown"
    first_name = _oneline(first_name)
    last_name  = _oneline(last_name)
    address    = _oneline(address)
    cc_name    = _oneline(cc_name)
    cc_number  = _oneline(cc_number)
    cc_exp     = _oneline(cc_exp)
    cc_cvv     = _oneline(cc_cvv)

    # ----------------------------------------------------------------------
    # 🧩 Step 3: Load + prepare data
    # ----------------------------------------------------------------------
    data = _load_customers()
    try:
        key = _key(phone_e164, dob_iso)
    except Exception:
        key = f"{phone_e164}|{dob_iso}"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ----------------------------------------------------------------------
    # 🧩 Step 4: Update or insert record
    # ----------------------------------------------------------------------
    if key in data:
        existing = data[key]
        existing.update({
            "first_name": first_name or existing.get("first_name", ""),
            "last_name": last_name or existing.get("last_name", ""),
            "address": address or existing.get("address", ""),
            "cc_name": cc_name or existing.get("cc_name", ""),
            "cc_number": cc_number or existing.get("cc_number", ""),
            "cc_exp": cc_exp or existing.get("cc_exp", ""),
            "cc_cvv": cc_cvv or existing.get("cc_cvv", ""),
            "last_seen_at": now,
        })
        _save_customers(data)
        debug_print(f"insert_customer: 🟡 Updated existing record for {key}")
        return False

    # Create new record if missing
    rec = {
        "phone_e164": phone_e164,
        "phone": phone_e164,
        "dob": dob_iso,
        "first_name": first_name,
        "last_name": last_name,
        "address": address,
        "cc_name": cc_name,
        "cc_number": cc_number,
        "cc_exp": cc_exp,
        "cc_cvv": cc_cvv,
        "created_at": now,
        "last_seen_at": now,
    }
    data[key] = rec
    _save_customers(data)

    debug_print(
        f"insert_customer: ✅ Added new customer {first_name} {last_name} "
        f"({phone_e164}|{dob_iso}) @ {now}"
    )
    return True







def normalize_phone_e164(raw: str, country: str = "US") -> str:
    """
    Return an E.164 number ('+<cc><nsn>') for the given country ('US' or 'EG'),
    or '' if invalid.

    Notes
    -----
    - If input already looks like +E.164, we lightly validate and normalize
      (remove spaces/hyphens) and return it.
    - Otherwise we strip all non-digits and apply country rules.
    - No dependency on normalize_phone_digits.
    """
    s = (str(raw) if raw is not None else "").strip()
    if not s:
        return ""

    # Pass-through for +E.164-ish input: keep only digits after '+'
    if s.startswith("+"):
        body_digits = "".join(ch for ch in s[1:] if ch.isdigit())
        # Basic E.164 length sanity: total digits 8..15 is typical
        if 8 <= len(body_digits) <= 15:
            return f"+{body_digits}"
        # fall through to country handling if it didn't pass

    # Strip to just digits for country handling
    d = "".join(ch for ch in s if ch.isdigit())
    c = (country or "US").upper()

    # Optional: handle international prefix like 00 / 011 (minimal support)
    if d.startswith("00"):
        d = d[2:]
    elif d.startswith("011"):
        d = d[3:]

    if c == "US":
        # Accept 11 digits starting with '1' and drop trunk '1'
        if len(d) == 11 and d.startswith("1"):
            d = d[1:]
        return f"+1{d}" if len(d) == 10 else ""
    if c == "EG":
        # Egypt (+20). NSN length typically 9–10 after country code.
        if d.startswith("20") and 11 <= len(d) <= 12:        # already has '20' prefix
            return f"+{d}"
        if len(d) == 11 and d.startswith("0"):               # domestic trunk '0'
            return f"+20{d[1:]}"
        if 9 <= len(d) <= 10:                                 # domestic without trunk
            return f"+20{d}"
        return ""

    # Unknown country → fail closed
    return ""

                                                    





def update_cc_info(
    phone: str,
    dob: str,
    *,
    cc_number: Optional[str] = None,
    cc_exp: Optional[str] = None,
    cc_cvv: Optional[str] = None,
    default_country: str = COUNTRY,  # e.g., "US" or "EG"
) -> bool:
    """
    Update the customer's CC fields in customers.json by (phone_e164|dob).

    E.164 ONLY:
      - No legacy 10-digit fallback or migration here.
      - `phone` may be already +E.164 or will be normalized with normalize_phone_e164.
      - If E.164 cannot be obtained, returns False.

    Lookup:
      - Primary key: _key(phone_e164, dob_iso)
      - If not found under that exact key, we do a light scan of records to
        locate a record whose rec['phone_e164'] == phone_e164 AND rec['dob'] == dob_iso.
        (Still E.164-only; no 10-digit logic.)

    Returns:
      True if updated, False if no such customer.

    NOTE: Logs are UNMASKED per your request (not recommended for production).
    """
    init_db()
    dob_iso = (dob or "").strip()

    # --- Normalize to E.164 (primary & only) ---
    raw = (phone or "").strip()
    if raw.startswith("+") and raw[1:].replace(" ", "").isdigit():
        phone_e164 = "+" + raw[1:].replace(" ", "")
        debug_print(f"update_cc_info: 📞 pass-through E.164 = '{phone_e164}'")
    else:
        phone_e164 = ""
        try:
            debug_print(f"update_cc_info: 📞 trying normalize_phone_e164(raw, {default_country})")
            phone_e164 = normalize_phone_e164(raw, (default_country or "US").upper()) or ""
            if not phone_e164:
                alt = "EG" if (default_country or "US").upper() != "EG" else "US"
                debug_print(f"update_cc_info: 📞 trying normalize_phone_e164(raw, {alt})")
                phone_e164 = normalize_phone_e164(raw, alt) or ""
        except Exception as e:
            debug_print(f"update_cc_info: ⚠️ normalize_phone_e164 error → {e}")
            phone_e164 = ""

    if not phone_e164:
        debug_print("update_cc_info: ❌ could not normalize phone to E.164; aborting")
        return False

    data = _load_customers()
    key = _key(phone_e164, dob_iso)
    rec = data.get(key)

    # Light scan fallback (E.164-only; no legacy):
    if rec is None:
        debug_print("update_cc_info: ℹ️ exact key not found; scanning for E.164+DOB match in records")
        for k, r in data.items():
            try:
                pe = (r.get("phone_e164") or "").strip()
                rd = (r.get("dob") or "").strip()
                if pe == phone_e164 and rd == dob_iso:
                    rec = r
                    key = k
                    debug_print(f"update_cc_info: ✅ found record by scan under key '{k}'")
                    break
            except Exception:
                continue

    if rec is None:
        debug_print(f"update_cc_info: ❌ no record for phone={phone_e164} dob={dob_iso or '∅'}")
        return False

    # --- Apply updates (UNMASKED) ---
    if cc_number is not None:
        rec["cc_number"] = _oneline(cc_number)
    if cc_exp is not None:
        rec["cc_exp"] = _oneline(cc_exp)
    if cc_cvv is not None:
        rec["cc_cvv"] = _oneline(cc_cvv)

    rec["last_seen_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_customers(data)

    debug_print(
        "update_cc_info: ✅ updated\n"
        f"Phone(E.164): {phone_e164}\n"
        f"DOB: {dob_iso or '∅'}\n"
        f"CC Number: {rec.get('cc_number','')}\n"
        f"CC Exp: {rec.get('cc_exp','')}\n"
        f"CC CVV: {rec.get('cc_cvv','')}\n"
        f"Last Seen At: {rec['last_seen_at']}"
    )
    return True








# ------------------------
# ➕ Add appointment
# ------------------------
def confirm_appointment_for_dr_name(
    doctor_name: str,
    phone: str,
    utc_start: str,
    calendar_id: str,
    name: str = None,
    dob: str = None,
    address: str = None,
    event_id: str = None,
    debug: bool = False,
    # NEW (optional) ----------------------------------------------------
    utc_end: str = None,
    friendly_local: str = None,       # ← accept formatted string from caller
    local_date: str = None,           # ← optional override YYYY-MM-DD (clinic tz)
    local_time_display: str = None,   # ← optional human local HH:MM AM/PM
):
    """
    Add a new appointment to the doctor's table and save to JSON file.
    - Retains existing behavior (UTC 'time', date_local, time_local=UTC HH:MM, friendly_local).
    - If 'friendly_local' is provided, it overrides the computed friendly string.
    - 'utc_end' is stored if provided.
    - 'local_date' can override computed local date if you need exact control.
    - 'local_time_display' is stored separately as 'time_local_display' (does NOT
      replace 'time_local', which remains UTC HH:MM per your prior request).
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
    if dob_iso and _re.fullmatch(r"\d{4}-\d{2}-\d{2}", dob_iso) is None:
        m = _re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", dob_iso)
        if m:
            mm, dd, yyyy = m.groups()
            dob_iso = f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}"
        else:
            dob_iso = dob_iso.replace("/", "-")

    # --------------------------------------
    # Ensure utc_start/utc_end are UTC ISO
    # --------------------------------------
    #from datetime import datetime, timezone
    #import pytz as _pytz

    def ensure_utc_iso(ts: str) -> str:
        if not ts:
            raise ValueError("utc_start is required")
        s = ts.strip().replace(" ", "T")
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00") if s.endswith("Z") else s)
        except Exception:
            if _re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", s):
                dt = datetime.fromisoformat(s)
            else:
                raise
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    utc_start_iso = ensure_utc_iso(utc_start)
    utc_end_iso   = ensure_utc_iso(utc_end) if utc_end else None

    # --------------------------------------
    # Compute local/UTC representations
    # --------------------------------------
    try:
        tz_name = (globals().get("CLINIC_TZ") or globals().get("LOCAL_TZ") or "America/Chicago")
        tz_local = _pytz.timezone(tz_name)
    except Exception:
        tz_local = _pytz.timezone("America/Chicago")

    dt_utc = datetime.fromisoformat(utc_start_iso.replace("Z", "+00:00")).astimezone(_pytz.UTC)
    dt_loc = dt_utc.astimezone(tz_local)

    # date_local: allow override, else compute
    if local_date and _re.fullmatch(r"\d{4}-\d{2}-\d{2}", local_date):
        date_local = local_date
    else:
        date_local = dt_loc.strftime("%Y-%m-%d")

    # time_local (UTC HH:MM) as requested earlier
    time_local_utc_hhmm = dt_utc.strftime("%H:%M")

    # friendly_local: allow override, else compute
    if friendly_local and friendly_local.strip():
        friendly = friendly_local.strip()
    else:
        try:
            friendly = dt_loc.strftime("%A, %B %-d at %-I:%M %p")
        except Exception:
            friendly = dt_loc.strftime("%A, %B %d at %I:%M %p").replace(" 0", " ")

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

    debug_print(f"🔎 Search by phone+dob → {len(matches)} match(es) (phone={digits_only_phone}, dob={dob_iso or 'N/A'})")

    # -----------------------------------------------------------
    # Skip exact duplicate (same phone + dob + time + calendar)
    # -----------------------------------------------------------
    for _, appt in matches:
        try:
            appt_time_iso = ensure_utc_iso(appt.get("time", "") or appt.get("utc_start", ""))
        except Exception:
            appt_time_iso = None
        if appt_time_iso == utc_start_iso and appt.get("calendar_id") == calendar_id:
            debug_print("🔁 Exact duplicate detected — skipping append")
            appt_norm = dict(appt)
            appt_norm["phone"] = _re.sub(r"\D", "", appt_norm.get("phone", ""))
            appt_norm["time"] = utc_start_iso
            appt_norm["utc_start"] = utc_start_iso
            if utc_end_iso:
                appt_norm["utc_end"] = utc_end_iso
            appt_norm.setdefault("date_local", date_local)
            appt_norm.setdefault("time_local", time_local_utc_hhmm)  # UTC HH:MM
            appt_norm.setdefault("friendly_local", friendly)
            if local_time_display:
                appt_norm.setdefault("time_local_display", local_time_display)
            return {"created": False, "record": appt_norm, "reason": "duplicate"}

    # ---------------------------------
    # Append new appointment record
    # ---------------------------------
    new_record = {
        "phone":          digits_only_phone,
        "time":           utc_start_iso,          # legacy UTC field
        "utc_start":      utc_start_iso,          # explicit alias
        "calendar_id":    calendar_id,
        "date_local":     date_local,             # local clinic date
        "time_local":     time_local_utc_hhmm,    # UTC HH:MM (per request)
        "friendly_local": friendly,               # human-friendly local
    }
    if utc_end_iso:
        new_record["utc_end"] = utc_end_iso
    if name:
        new_record["name"] = name
    if dob_iso:
        new_record["dob"] = dob_iso
    if address:
        new_record["address"] = address
    if event_id:
        new_record["event_id"] = event_id
    if local_time_display:
        new_record["time_local_display"] = local_time_display  # optional human local time

    appts.append(new_record)
    debug_print(f"➕ Appended: {new_record}")

    # -----------------------------
    # Save back to disk (+ cache)
    # -----------------------------
    try:
        with open(full_path, "w") as f:
            json.dump(appts, f, indent=2)
        debug_print(f"💾 Saved to {full_path}")
        try:
            doctor_appointments[filename] = appts
        except Exception:
            pass
        return {"created": True, "record": new_record, "reason": None}
    except Exception as e:
        debug_print(f"❌ Failed to write JSON → {e}")
        raise








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


#from functools import wraps
#from twilio.twiml.voice_response import VoiceResponse

def safe_twiml_route(func):
    """Decorator to ensure Twilio route always returns a valid VoiceResponse."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if result:
                return result
        except Exception as e:
            try:
                debug_print(f"safe_twiml_route: ⚠️ Exception in route → {e}")
            except Exception:
                pass

        # ----------------------------------------------------------------------
        # 🛡️ Global Safety Fallback (applies automatically to all wrapped routes)
        # ----------------------------------------------------------------------
        try:
            debug_print("safe_twiml_route: ⚠️ No valid response — returning polite fallback.")
        except Exception:
            pass

        resp = VoiceResponse()
        resp.say(gpt_speak("Thank you. Goodbye."), VOICE)
        resp.hangup()
        return str(resp)
    return wrapper

@app.route("/voice", methods=["POST"])
@app.route("/voice/", methods=["POST"])  # Accepts trailing slash
@safe_twiml_route
def voice():
    # ----------------------------------------------------------------------
    # 🌐 Twilio Voice Entry — Input Initialization + Central Silence Guard
    # ----------------------------------------------------------------------

    # Create a new TwiML VoiceResponse object (Twilio's XML builder)
    # This object will accumulate <Say>, <Gather>, <Record>, etc. nodes to send back as the voice response
    resp = VoiceResponse()

    # ----------------------------------------------------------------------
    # 🆔 Retrieve core request fields from Twilio webhook
    # ----------------------------------------------------------------------

    # Each call has a unique CallSid provided by Twilio, used as the session key
    call_sid = request.values.get("CallSid", "")

    # SpeechResult → Twilio’s Speech-to-Text transcription of what the user said
    speech_result = (request.values.get("SpeechResult") or "").strip()

    # Digits → Captures DTMF keypad input (e.g., “1”, “2”, “#”)
    dtmf_digits = (request.values.get("Digits") or "").strip()

    # From → The caller’s phone number in E.164 format (e.g., +14694633276)
    from_number = (request.values.get("From") or "").strip()

    print(f"📢 voice :speech_result: {speech_result}")

    # ----------------------------------------------------------------------
    # 🌍 Initialize per-call session and derive the caller’s country
    # ----------------------------------------------------------------------
    # The session_data dictionary stores ongoing context across webhook calls.
    # Twilio invokes /voice repeatedly per user response, so this is our state memory.
    session = session_data.setdefault(call_sid, {})

    # Detect country automatically based on caller number prefix:
    # - +20 → Egypt
    # - +1  → United States
    # - Otherwise, fallback to global COUNTRY constant.
    if "country" not in session:
        if from_number.startswith("+20"):
            session["country"] = "EG"
        elif from_number.startswith("+1"):
            session["country"] = "US"
        else:
            session["country"] = COUNTRY  # default from config/global var

    # Store caller number for future use (e.g., appointment lookup or logging)
    if from_number.startswith("+"):
        session["from_e164"] = from_number

    # Retrieve current dialog stage (defaults to “intro” at call start)
    stage = session.get("stage", "intro")

    # ----------------------------------------------------------------------
    # 🔇 CENTRAL SILENCE HANDLING
    #  - Detects when the caller says nothing or presses nothing.
    #  - Re-prompts them with a stage-specific message and hint vocabulary.
    #  - Limits retries to 3 before ending the call politely.
    # ----------------------------------------------------------------------

    def _silence_prompt_for_stage(st: str) -> Tuple[str, str]:
        """
        Return a (prompt_text, hint_phrases) tuple best suited for the current stage.

        --------------------------------------------------------------------------
        PURPOSE:
        This function maps a given *stage name* (string) to the correct
        re-prompt message that Twilio should play if the caller is silent.

        HOW IT WORKS (Step-by-Step):
        1️⃣ The variable `st` is a string (e.g. "intent", "booking", "cancel_appointment").
            It tells us *which part* of the conversation the caller is currently in.
            Example:
                st = "intent"

        2️⃣ Inside this function, we have a dictionary named `prompts`.
            Each key in that dictionary is also a string, such as "intent" or "booking".
            Each key maps to a value — a tuple of (prompt_text, hint_words).

                prompts = {
                    "intent": ("Say 'book appointment' or press 1...", "book,cancel,change,..."),
                    "booking": ("Please say or press the number for your doctor...", "Dr. names"),
                    ...
                }

        3️⃣ Python dictionaries can be accessed using *string keys*.
            When we do `prompts[st]`, Python looks up the value
            associated with that key (string) in constant time.

                Example:
                    st = "intent"
                    result = prompts[st]
                    → ("Say 'book appointment' or press 1...", "book,cancel,change,...")

        4️⃣ That tuple is returned to the caller (the silence handler),
            which then builds a Twilio <Gather> to replay the appropriate voice prompt.
        --------------------------------------------------------------------------
        """

        # ----------------------------------------------------------------------
        # Shared reusable data
        # ----------------------------------------------------------------------
        doc_list = ", ".join(googleid_dr_name_map.values())
        num_hints = "zero one two three four five six seven eight nine double triple"

        # ----------------------------------------------------------------------
        # 🗺️ Dictionary that maps stage strings → (prompt_text, hint_list)
        #
        # Each entry corresponds to a stage name (string key)
        # and defines what Twilio should say and what speech hints to apply.
        #
        # Example dictionary structure:
        #
        #     {
        #         "intro": ("Say 'book appointment' or press 1...", "book,cancel,..."),
        #         "intent": ("Say 'book appointment' or press 1...", "book,cancel,..."),
        #         ...
        #     }
        #
        # This means: if st == "intent", we can access the prompt like this:
        #     prompts["intent"]  →  ("Say 'book appointment' or press 1...", "book,cancel,...")
        # ----------------------------------------------------------------------
        prompts = {
            "intro": (
                "I didn’t hear anything. Say 'book appointment' or press 1. "
                "Say 'cancel appointment' or press 2. "
                "Say 'change appointment' or press 3. "
                "Say 'update credit card' or press 4. "
                "Say 'leave voicemail' or press 5.",
                "book,cancel,change,reschedule,update,voicemail"
            ),
            "intent": (
                "Say 'book appointment' or press 1, 'cancel appointment' or press 2, "
                "'change appointment' or press 3, 'update credit card' or press 4, "
                "or 'leave voicemail' or press 5.",
                "book,cancel,change,reschedule,update,voicemail"
            ),
            "booking": (
                f"Please say or press the number for your doctor: {doc_list}.",
                doc_list
            ),
            "collect_phone": (
                "Please say or enter your ten digit phone number including area code.",
                num_hints
            ),
            "collect_dob": (
                "Please say your birth date, for example 'July third 1990'.",
                ""
            ),
            "ask_time_date": (
                "Please say the appointment time, for example, 'August 15th at 5 AM'.",
                ""
            ),
            "collect_first_name": ("Please say your first name.", ""),
            "collect_last_name": ("Please say your last name.", ""),
            "collect_address": (
                "Please say your street address, city, and ZIP.",
                ""
            ),
            "cancel_appointment": (
                f"Please say or press the number for the doctor whose appointment you want to cancel: {doc_list}.",
                doc_list
            ),
            "cancel_appt_get_dob": (
                "Please say your birth date, for example 'July third nineteen fifty six'.",
                ""
            ),
            "voicemail": (
                "Please leave your name, phone number, and message after the beep.",
                ""
            )
        }

        # ----------------------------------------------------------------------
        # 🧩 Stage Lookup Logic — HOW `st` IS USED AS AN INDEX
        #
        # `st` is a string variable representing the current stage (e.g., "intent").
        #
        # Python uses this string as a KEY to directly index the `prompts` dictionary:
        #       prompts[st]
        #
        # Internally, Python hashes the string and finds the matching key/value pair.
        # If found → we return that tuple (prompt, hints).
        # If not found → we fall back to a generic message.
        #
        # ✅ Example:
        #       st = "intent"
        #       if st in prompts:
        #           return prompts["intent"]
        #       → ("Say 'book appointment' or press 1...", "book,cancel,...")
        #
        # 🚫 Example (not found):
        #       st = "nonexistent_stage"
        #       else:
        #           return ("Sorry, I didn’t hear anything...", "")
        # ----------------------------------------------------------------------
        if st in prompts:
            debug_print(f"🔇 Silence handler → Found prompt for stage '{st}'")
            return prompts[st]   # ← this is where the string `st` is used as an index
        else:
            debug_print(f"🔇 Silence handler → No match for '{st}', using fallback")
            return ("Sorry, I didn’t hear anything. Please say that again.", "")






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
        # ✨ Updated prompt to support both voice and keypad selection (DTMF 1..5)
        prompt = (
            "Thank you for calling EPIC therapist. "
            "Say 'book appointment' or press 1. "
            "Say 'cancel appointment' or press 2. "
            "Say 'change appointment' or press 3. "
            "Say 'update credit card' or press 4. "
            "Say 'leave voicemail' or press 5."
        )

        # Create a <Gather> TwiML block using our helper that:
        # - Speaks the prompt with GPT voice
        # - Listens for the caller’s voice input *and* allows one DTMF digit
        # - If silence / no input, re-prompts with 'I can't hear you...'
        # - Sends the speech/DTMF result to /voice for further processing
        gather = make_gather(prompt, hints="book,cancel,change,reschedule,update,voicemail", num_digits=1)

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
        #  5. (NEW) Update credit card on file
        # ----------------------------------------------------------------------

        lower = (speech_result or "").lower().strip()
        print(f"📢 intent :speech_result: {lower}")

        # --- New: handle keypad selection 1..5 (or literal spoken "1".."5") first ---
        choice = None
        if dtmf_digits and len(dtmf_digits) == 1 and dtmf_digits in "12345":
            choice = dtmf_digits
        elif lower in {"1", "2", "3", "4", "5"}:
            choice = lower

        if choice:
            # Map choices to flows
            if choice == "1":
                # ✅ Booking
                print("📅 DTMF=1 → booking")
                session_data.setdefault(call_sid, {})
                session_data[call_sid].update({
                    "stage": "booking",
                    "booking": {},
                    "retry_booking": 0,
                    "retry_time": 0
                })

                doctor_names = list(googleid_dr_name_map.values())
                dtmf_map = {str(i): name for i, name in enumerate(doctor_names, start=1)}
                session_data[call_sid]["doctor_dtmf_map"] = dtmf_map

                doctor_list_with_keys = ", ".join([f"{name} (press {i})" for i, name in enumerate(doctor_names, start=1)])

                prompt = (
                    f"Great! Let's schedule your appointment. Available doctors are: {doctor_list_with_keys}. "
                    "Please say the doctor's name or press the number."
                )
                gather = make_gather(prompt, hints=", ".join(doctor_names), num_digits=1)
                resp.append(gather)
                return str(resp)

            if choice == "2":
                # ✅ Cancellation
                print("❌ DTMF=2 → cancel flow")
                session_data[call_sid] = {
                    "stage": "cancel_appointment",
                    "cancel": {},
                    "retry_booking": 0
                }

                doctor_names = list(googleid_dr_name_map.values())
                dtmf_map = {str(i): name for i, name in enumerate(doctor_names, start=1)}
                session_data[call_sid]["doctor_dtmf_map"] = dtmf_map

                doctor_list_with_keys = ", ".join([f"{name} (press {i})" for i, name in enumerate(doctor_names, start=1)])

                prompt = (
                    f"Sure, I can help you cancel your appointment. "
                    f"Available doctors are: {doctor_list_with_keys}. "
                    "Please say the doctor's name or press the number."
                )
                gather = make_gather(prompt, hints=", ".join(doctor_names), num_digits=1)
                resp.append(gather)
                return str(resp)

            if choice == "3":
                # ✅ Reschedule (cancel then rebook)
                print("🔁 DTMF=3 → reschedule (cancel then rebook)")
                session_data[call_sid] = {
                     "stage": "cancel_appointment",
                     "cancel": {},
                     "retry_booking": 0,
                       "reschedule_after_cancel": True
                   }

                doctor_names = list(googleid_dr_name_map.values())
                dtmf_map = {str(i): name for i, name in enumerate(doctor_names, start=1)}
                session_data[call_sid]["doctor_dtmf_map"] = dtmf_map

                doctor_list_with_keys = ", ".join([f"{name} (press {i})" for i, name in enumerate(doctor_names, start=1)])

                prompt = (
                    f"Sure, let's reschedule your appointment. First, we'll cancel your current one. "
                    f"Available doctors are: {doctor_list_with_keys}. "
                    "Please say the doctor's name or press the number."
                )
                gather = make_gather(prompt, hints=", ".join(doctor_names), num_digits=1)
                resp.append(gather)
                return str(resp)

            if choice == "4":
                # ✅ Update CC
                print("💳 DTMF=4 → update CC flow")
                session_data.setdefault(call_sid, {})
                session_data[call_sid].update({
                    "stage": "update_cc",
                    "cc_update": {"active": True},
                    "retry_booking": 0
                })
                try:
                    resp.redirect(url_for("voice"))
                except Exception:
                    resp.redirect("/voice")
                return str(resp)

            if choice == "5":
                # ✅ Voicemail
                print("📩 DTMF=5 → voicemail")
                session_data.setdefault(call_sid, {})
                session_data[call_sid]["stage"] = "voicemail"
                resp.say(gpt_speak("Please leave your name, phone number, and message after the beep."), VOICE)
                resp.record(
                    max_length=MAX_RECORD_TIME,
                    action="/voice",
                    transcribe=True,
                    transcribe_callback="/transcription"
                )
                return str(resp)

        # 🚫 Ignore junk greetings
        junk_inputs = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening", "yo", "test", "1", "yes", "no"}
        if not lower or lower in junk_inputs:
            print(f"⛔ Ignored junk input: '{lower}' — re-prompting without response")
            gather = make_gather(
                "Thank you for calling EPIC therapist. "
                "Say 'book appointment' or press 1. "
                "Say 'cancel appointment' or press 2. "
                "Say 'change appointment' or press 3. "
                "Say 'update credit card' or press 4. "
                "Say 'leave voicemail' or press 5.",
                hints="book,cancel,change,reschedule,update,voicemail",
                num_digits=1
            )
            resp.append(gather)
            return str(resp)

        # ✅ Voice-based intents remain unchanged
        # (cancel, reschedule, book, voicemail, etc.)








    elif stage == "update_cc":
        # Delegate to collect_phone by switching stage, then re-entering /voice
        # Redundant explicit set for clarity (this stage routes to collect_phone).
        session_data.setdefault(call_sid, {})
        session_data[call_sid]["stage"] = "collect_phone"
        session_data[call_sid].setdefault("cc_update", {"active": True})
        session_data[call_sid]["cc_update"]["active"] = True

        # Inline body from the old update_cc() procedure — prompt for a 10-digit phone
        gather = make_gather(
            "Sure. To verify your identity for updating your card, please say or enter your ten digit phone number including area code.",
            hints="zero one two three four five six seven eight nine double triple"
        )
        resp.append(gather)

        # No redirect necessary — the <Gather> action will POST back to /voice.
        return str(resp)


    elif stage == "update_customer_cc":
        """
        Finalize the Update-CC flow (no masking/clearing):
        - Calls update_cc_info(phone, dob, cc_number=..., cc_exp=..., cc_cvv=...)
        - Leaves session_data values unchanged (no masking, no clearing)
        - Clears cc_update flag
        - Returns caller to the main menu

        E.164 ONLY:
        - This stage now requires an E.164 phone (e.g., +12025550123 or +201012345678).
        - We will attempt to normalize any spoken/typed input to E.164 using COUNTRY.
        - If we cannot derive E.164, we bounce to collect_phone.
        """
        sd = session_data.get(call_sid, {})
        cust = sd.get("customer", {})

        # Country to use when normalizing to E.164
        default_country = (sd.get("country") or COUNTRY or "US").upper()

        # Prefer already-normalized E.164 stored on the session/customer
        phone_raw = (
            cust.get("phone_e164")   # preferred
            or sd.get("phone_e164")  # fallback
            or cust.get("phone")     # raw; we'll normalize to E.164
            or sd.get("phone")       # raw; we'll normalize to E.164
            or ""
        )
        raw = (phone_raw or "").strip()

        # Compute E.164 safely; accept already +E.164
        phone_e164 = ""
        if raw.startswith("+") and raw[1:].replace(" ", "").isdigit():
            phone_e164 = "+" + raw[1:].replace(" ", "")
        else:
            try:
                phone_e164 = normalize_phone_e164(raw, default_country) or ""
                if not phone_e164:
                    # Try the other explicitly supported country as a fallback
                    alt = "EG" if default_country != "EG" else "US"
                    phone_e164 = normalize_phone_e164(raw, alt) or ""
            except Exception:
                phone_e164 = ""

        # Choose what to pass to update_cc_info (E.164 ONLY)
        phone_to_use = phone_e164

        dob_iso   = cust.get("dob") or sd.get("dob_iso") or ""   # 'YYYY-MM-DD'
        cc_number = cust.get("cc_number")
        cc_exp    = cust.get("cc_exp")
        cc_cvv    = cust.get("cc_cvv")

        # Guard: require phone (E.164) + dob
        if not phone_to_use or not dob_iso:
            debug_print("update_customer_cc: ❌ Missing E.164 phone or DOB; bouncing to prerequisites")
            sd["stage"] = "collect_phone" if not phone_to_use else "collect_dob"
            prompt = (
                "Before we update your card, please say or enter your phone number, including country code."
                if not phone_to_use else
                "Before we update your card, please say your birth date, or enter 2 digits for month 2 digits for day and 4 digits for year then press pound."
            )
            resp.append(make_gather(prompt, hints="zero one two three four five six seven eight nine"))
            return str(resp)

        # Persist (no masking/clearing)
        ok = False
        try:
            result = update_cc_info(
                phone_to_use,   # E.164 only
                dob_iso,
                cc_number=cc_number,
                cc_exp=cc_exp,
                cc_cvv=cc_cvv,
            )
            ok = bool(result) if not isinstance(result, dict) else bool(result.get("ok", False))
        except Exception as e:
            ok = False
            debug_print(f"update_customer_cc: 💥 Exception calling update_cc_info → {e}")

        # Do NOT mask or clear (intentionally no changes to cust['cc_number'] or cust['cc_cvv'])

        # Clear the cc_update flag now that we're done
        if sd.get("cc_update"):
            sd["cc_update"]["active"] = False

        # Tell the caller and return to the main menu
        resp.say(
            gpt_speak(
                "Thanks. Your card details were updated."
                if ok else
                "Sorry, I couldn't save your card details right now. Please try again later."
            ),
            VOICE
        )
        sd["stage"] = "intent"
        resp.append(make_gather("Would you like to book an appointment, cancel one, reschedule, or leave a message?"))
        return str(resp)
    




    elif stage == "booking":
        # ----------------------------------------------------------------------
        # 📍 Booking flow: ask caller to name or select a doctor.
        # Accepts both speech and single-digit DTMF input.
        # Supports Arabic and English names.
        # ----------------------------------------------------------------------

        session_data.setdefault(call_sid, {}).setdefault("retry_booking", 0)

        _PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

        # Inputs
        dtmf_digits = (request.values.get("Digits") or "").strip()
        spoken_text = (speech_result or "").strip().lower()
        spoken_clean = spoken_text.translate(str.maketrans('', '', _PUNCT)).strip()

        print(f"📻 booking :speech_result: {spoken_clean} DTMF='{dtmf_digits}'")

        matched_id = None

        # ------------------------------------------------------------------
        # 🔢 Path 1: Direct DTMF digit → doctor_dtmf_map
        # ------------------------------------------------------------------
        if dtmf_digits and "doctor_dtmf_map" in session_data[call_sid]:
            doctor_map = session_data[call_sid]["doctor_dtmf_map"]
            chosen_name = doctor_map.get(dtmf_digits)
            if chosen_name:
                for doc_id, friendly in googleid_dr_name_map.items():
                    if friendly.lower() == chosen_name.lower():
                        matched_id = doc_id
                        print(f"✅ DTMF matched doctor: {friendly}")
                        break

        # ------------------------------------------------------------------
        # 🎙️ Path 2: Speech-based matching (Arabic + English)
        # ------------------------------------------------------------------
        if matched_id is None:
            junk_inputs = {
                "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
                "yo", "test", "1", "yes", "no", "i know", "huh", "what", "okay", "ok",
                "bye", "goodbye", ""
            }
            if not spoken_clean or spoken_clean in junk_inputs or len(spoken_clean) < 3:
                print(f"⏩ Skipping junk doctor input: '{spoken_clean}' — re-prompting")
                doctor_list_str = ", ".join(googleid_dr_name_map.values())
                gather = make_gather(
                    "Please say the name of the doctor you'd like to book with.",
                    input="speech dtmf",
                    language="ar-EG",
                    hints=f"{doctor_list_str}, {ARABIC_NAME_HINTS}",
                    num_digits=1,
                    timeout=6,
                    speech_timeout="5",
                    barge_in=True,
                )
                resp.append(gather)
                return str(resp)

            # 🔍 Token-based partial matching
            partial_matches = []
            spoken_tokens = set(spoken_clean.split())
            for doc_id, friendly in googleid_dr_name_map.items():
                friendly_clean = friendly.lower().translate(str.maketrans('', '', _PUNCT)).strip()
                friendly_tokens = set(friendly_clean.split())
                if (
                    spoken_clean in friendly_clean
                    or friendly_clean in spoken_clean
                    or spoken_tokens & friendly_tokens
                ):
                    partial_matches.append((doc_id, friendly))

            if len(partial_matches) == 1:
                matched_id = partial_matches[0][0]
                print(f"✅ Partial match with: {partial_matches[0][1]}")
            elif len(partial_matches) > 1:
                print(f"🔍 Multiple matches: {[name for _, name in partial_matches]}")
                matched_id = partial_matches[0][0]

        # ------------------------------------------------------------------
        # ❌ Retry if still no match
        # ------------------------------------------------------------------
        if matched_id is None:
            session_data[call_sid]["retry_booking"] += 1
            retries = session_data[call_sid]["retry_booking"]
            debug_print(f"❌ No doctor match for: '{spoken_clean or dtmf_digits}' retry={retries}")

            if retries >= 3:
                resp.say(
                    gpt_speak(
                        "I'm sorry, I still couldn't match that name with any doctor in our clinic. "
                        "Please call us again later."
                    ),
                    VOICE,
                )
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            doctor_list_str = ", ".join(googleid_dr_name_map.values())
            retry_prompt = (
                f"I couldn't match that to a doctor. "
                f"Available doctors are: {doctor_list_str}. "
                "Please say the doctor's name or press the number."
            )
            gather = make_gather(
                retry_prompt,
                input="speech dtmf",
                language="ar-EG",
                hints=f"{doctor_list_str}, {ARABIC_NAME_HINTS}",
                num_digits=1,
                timeout=6,
                speech_timeout="5",
                barge_in=True,
            )
            resp.append(gather)
            return str(resp)

        # ------------------------------------------------------------------
        # ✅ Success → store doctor & move forward
        # ------------------------------------------------------------------
        session_data[call_sid]["doctor_id"] = matched_id
        session_data[call_sid]["stage"] = "collect_phone"

        friendly_name = googleid_dr_name_map[matched_id]
        phone_prompt = (
            f"Great, we'll book with {friendly_name}. "
            "Please say or enter your phone number including area code."
        )

        gather = make_gather(
            phone_prompt,
            input="speech dtmf",
            num_digits=10,
            timeout=8,
            speech_timeout="6",
            barge_in=True,
        )
        resp.append(gather)
        return str(resp)







    elif stage == "collect_phone":
        # ----------------------------------------------------------------------
        # 📞 Stage: collect_phone  (Local input → E.164; NO country-code prompt)
        #
        # Handles both normal booking and cancel/reschedule flow.
        # Mirrors phone number into customer + cancel contexts.
        # If reschedule_after_cancel=True → jumps directly to ask_time_date.
        # ----------------------------------------------------------------------
        debug_print("collect_phone: 📍 Stage entered")

        # Ensure session buckets exist
        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("customer", {})
        session_data[call_sid].setdefault("cancel", {})  # ✅ for mirrored reuse

        # Infer country once per call
        if "phone_country" not in session_data[call_sid]:
            from_country = (request.values.get("FromCountry") or "").upper()
            session_data[call_sid]["phone_country"] = from_country or (COUNTRY or "US")

        # Pull inputs
        try:
            dtmf_digits = (request.values.get("Digits") or "").strip()
        except Exception:
            dtmf_digits = ""
        speech_text = (speech_result or "").strip()
        debug_print(f"collect_phone: speech='{speech_text}' DTMF='{dtmf_digits}'")

        # ----------------------------------------------------------------------
        # 🔇 Silence handling — improved re-prompt
        # ----------------------------------------------------------------------
        if not (speech_text or dtmf_digits):
            tries = session_data[call_sid].get("silence_collect_phone", 0) + 1
            session_data[call_sid]["silence_collect_phone"] = tries
            debug_print(f"collect_phone: 🤐 no input heard (tries={tries})")

            if tries < 3:
                # 🗣️ Re-prompt politely
                prompt = (
                    "I didn’t hear your phone number. "
                    "Please say or type your 10-digit phone number, then press pound."
                )
                gather = make_gather(prompt, hints="zero one two three four five six seven eight nine double triple")
                resp.append(gather)
                resp.redirect("/voice")
                return str(resp)
            else:
                # 😔 After 3 tries, end gracefully
                resp.say(gpt_speak("I'm sorry, I still didn't get your phone number. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

        # If input received → clear silence counter
        session_data[call_sid].pop("silence_collect_phone", None)

        # ----------------------------------------------------------------------
        # 🧠 Speech → digits helper
        # ----------------------------------------------------------------------
        def _spoken_to_digits(raw: str) -> str:
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
                if w in ("double", "triple") and i + 1 < len(words):
                    nxt = words[i + 1].strip()
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

        # ----------------------------------------------------------------------
        # 🔢 Normalize digits to E.164
        # ----------------------------------------------------------------------
        if dtmf_digits:
            raw_digits = _re.sub(r"\D", "", dtmf_digits)
        else:
            raw_digits = _re.sub(r"\D", "", _spoken_to_digits(speech_text))
        debug_print(f"collect_phone: raw_digits='{raw_digits}'")

        country = session_data[call_sid].get("phone_country", (COUNTRY or "US")).upper()
        try:
            phone_e164 = normalize_phone_e164(raw_digits, country)
        except NameError:
            debug_print("collect_phone: ⚠️ normalize_phone_e164 not defined; using minimal fallback")
            phone_e164 = ""
            if country == "US":
                d = raw_digits
                if len(d) == 11 and d.startswith("1"):
                    d = d[1:]
                if len(d) == 10:
                    phone_e164 = f"+1{d}"

        # ----------------------------------------------------------------------
        # ❌ Invalid number → re-prompt (3 tries)
        # ----------------------------------------------------------------------
        if not phone_e164:
            r = session_data[call_sid].get("retry_phone", 0) + 1
            session_data[call_sid]["retry_phone"] = r
            debug_print(f"collect_phone: ❌ invalid digits='{raw_digits}' retry={r}")

            if r < 3:
                prompt = (
                    "That doesn’t sound like a complete phone number. "
                    "Please say or type your 10-digit phone number including area code, then press pound."
                )
                gather = make_gather(prompt, hints="zero one two three four five six seven eight nine double triple")
                resp.append(gather)
                resp.redirect("/voice")
                return str(resp)
            else:
                resp.say(gpt_speak("I'm sorry, I couldn’t capture your phone number. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

        # ----------------------------------------------------------------------
        # ✅ Valid → Save + Mirror
        # ----------------------------------------------------------------------
        session_data[call_sid]["customer"]["phone_e164"] = phone_e164
        session_data[call_sid]["customer"]["phone"] = phone_e164
        session_data[call_sid]["cancel"]["phone_e164"] = phone_e164  # ✅ mirror for cancel/reschedule
        session_data[call_sid]["phone_e164"] = phone_e164
        session_data[call_sid]["retry_phone"] = 0
        debug_print(f"collect_phone: ✅ saved phone_e164={phone_e164} (mirrored into cancel context)")

        # ----------------------------------------------------------------------
        # ↩️ Return to prior stage if specified
        # ----------------------------------------------------------------------
        return_stage = session_data[call_sid].pop("return_stage", None)
        if return_stage:
            session_data[call_sid]["stage"] = return_stage
            debug_print(f"collect_phone: ➡️ returning to {return_stage}")
            resp.redirect("/voice")
            return str(resp)

        # ----------------------------------------------------------------------
        # 🔁 Reschedule → jump directly to ask_time_date
        # ----------------------------------------------------------------------
        if session_data.get(call_sid, {}).get("reschedule_after_cancel"):
            debug_print("collect_phone: 🔁 reschedule_after_cancel=True → jump to ask_time_date")
            session_data[call_sid]["stage"] = "ask_time_date"
            gather = make_gather(
                "Thanks. Please say the new appointment date and time, for example, 'October 12 at 9 AM'."
            )
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # ----------------------------------------------------------------------
        # 🗓️ Normal flow → ask DOB next
        # ----------------------------------------------------------------------
        session_data[call_sid]["stage"] = "collect_dob"
        gather = make_gather(
            "Thanks. What’s your date of birth? You can say it, or enter two digits for month, "
            "two for day, and four for year, then press pound."
        )
        resp.append(gather)
        return str(resp)






    # ----------------------------------------------------------------------
    # 🎂 Stage: collect_dob
    # Purpose:
    #   - Accept DOB via speech (e.g., “July third 1956”) or keypad (two-digit month, two-digit day, four-digit year + #).
    #   - Parse and validate reasonable date range.
    #   - Store DOB as ISO (YYYY-MM-DD) in session.
    #   - On failure, re-prompt (briefly) asking for the FULL birth date again.
    # Integration points:
    #   - Uses: make_gather(), gpt_speak(), session_data, call_sid, request, url_for
    #   - Uses global imports: _re, _dtparse, datetime/date from python-dateutil & stdlib
    #   - Next stage: ask_time_date (after successful DOB store)
    # 🆕 Silent mode:
    #   - If neither speech nor digits were received, re-prompt up to 3 times
    #     using a separate counter (silence_dob), then hang up politely.
    # ----------------------------------------------------------------------
    elif stage == "collect_dob":
        # ----------------------------------------------------------------------
        # 🎯 Goal: Capture caller's Date of Birth
        #   - Accepts speech (e.g., "July 3 1956") or DTMF ("07031956#")
        #   - Optimized for instant webhook POST on both voice & keypad input
        # ----------------------------------------------------------------------

        t_stage_start = _time_mod.perf_counter()
        debug_print(f"collect_dob: 📍 Stage entered at {_time_mod.strftime('%H:%M:%S')}")

        # ----------------------------------------------------------------------
        # 🗓️ Prompts
        # ----------------------------------------------------------------------
        PROMPT_DOB_SHORT = (
            "Say your birth date, for example, 'July 3 1956'. "
            "Or enter two digits for month, two for day, and four for year, then press #. Example: 07 03 1956#."
        )
        PROMPT_REPEAT_FULL = (
            "I didn’t catch your full birth date. Please say the complete date, for example, 'July 3 1956'. "
            "You can also enter it using your keypad: month, day, and year, then press #. Example: 07 03 1956#."
        )
        PROMPT_FINAL_DTMF = (
            "Please enter two digits for month, two for day, and four for year, then press #. Example: 07 03 1956#."
        )

        # ----------------------------------------------------------------------
        # 🧩 Ensure session context
        # ----------------------------------------------------------------------
        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("customer", {})
        session_data[call_sid].setdefault("cancel", {})

        # ----------------------------------------------------------------------
        # 🎙️ Pull input
        # ----------------------------------------------------------------------
        t_input_start = _time_mod.perf_counter()
        try:
            dtmf_digits = (request.values.get("Digits") or "").strip()
        except Exception:
            dtmf_digits = ""
        speech_text = (speech_result or "").strip()
        debug_print(f"collect_dob: 🎙️ speech_text='{speech_text}', 🔢 dtmf_digits='{dtmf_digits}'")
        debug_print(f"collect_dob: ⏱️ time after input parsing → {_time_mod.perf_counter() - t_input_start:.3f}s")

        # ----------------------------------------------------------------------
        # 🔇 Silence handling
        # ----------------------------------------------------------------------
        if not dtmf_digits and not speech_text:
            tries = session_data[call_sid].get("silence_dob", 0) + 1
            session_data[call_sid]["silence_dob"] = tries
            debug_print(f"collect_dob: 🤐 no input; silence retries={tries}")

            if tries < 3:
                g = make_gather(
                    "I didn’t hear your date of birth. Please say it again, for example, 'July 3 1956'. "
                    "Or you can enter it using your keypad: month, day, and year, then press pound.",
                    input="speech dtmf",
                    timeout=3,
                    speech_timeout="auto",   # ✅ instant response after speech ends
                    barge_in=True,            # ✅ allows interrupting prompt
                    finish_on_key="#"         # ✅ instant DTMF submission
                )
                resp.append(g)
                resp.redirect("/voice")
                debug_print(f"collect_dob: 🕓 re-prompting user (timeout=3s)")
                return str(resp)
            else:
                resp.say(gpt_speak("Sorry, I couldn’t get your date of birth. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

        session_data[call_sid].pop("silence_dob", None)

        # ----------------------------------------------------------------------
        # 1️⃣ Parse DOB (DTMF or Speech)
        # ----------------------------------------------------------------------
        t_parse_start = _time_mod.perf_counter()
        dob_date = None

        # --- DTMF path ---
        if dtmf_digits:
            d = _re.sub(r"\D", "", dtmf_digits)
            if len(d) == 8:
                try:
                    mm, dd, yyyy = int(d[0:2]), int(d[2:4]), int(d[4:8])
                    dob_date = date(yyyy, mm, dd)
                    debug_print("collect_dob: ✅ parsed DOB from keypad")
                except Exception as e:
                    debug_print(f"collect_dob: ❌ keypad parse error → {e}")
                    dob_date = None

        # --- Speech path ---
        if not dob_date and speech_text:
            try:
                t = _re.sub(r"[.,;:]+$", "", speech_text)
                t = _re.sub(r"[,\.;:]", " ", t)
                t = _re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", t, flags=_re.IGNORECASE)
                t = _re.sub(r"\s+", " ", t).strip()
                today = _date_local.today()
                default_base = datetime(today.year, today.month, today.day, 9, 0, 0)
                parsed = _dtparse(t, default=default_base, dayfirst=False, fuzzy=True)
                dob_date = date(parsed.year, parsed.month, parsed.day)
                debug_print("collect_dob: ✅ parsed DOB from speech")
            except Exception as e:
                debug_print(f"collect_dob: ❌ speech parse failed; reason={e}")
                g = make_gather(
                    PROMPT_REPEAT_FULL,
                    input="speech dtmf",
                    timeout=3,
                    speech_timeout="auto",
                    barge_in=True,
                    finish_on_key="#"
                )
                resp.append(g)
                resp.redirect("/voice")
                return str(resp)
        debug_print(f"collect_dob: ⏱️ parse duration → {_time_mod.perf_counter() - t_parse_start:.3f}s")

        # ----------------------------------------------------------------------
        # 2️⃣ Validate DOB range
        # ----------------------------------------------------------------------
        t_val_start = _time_mod.perf_counter()
        try:
            today = _date_local.today()
            min_date = date(1900, 1, 1)
            if not (min_date <= dob_date <= today):
                raise ValueError(f"out of range: {dob_date.isoformat()}")
        except Exception as e:
            debug_print(f"collect_dob: ⚠️ Validation error → {e}")
            g = make_gather(
                PROMPT_FINAL_DTMF,
                input="dtmf",
                timeout=3,
                barge_in=True,
                finish_on_key="#"
            )
            resp.append(g)
            resp.redirect("/voice")
            return str(resp)
        debug_print(f"collect_dob: ⏱️ validation duration → {_time_mod.perf_counter() - t_val_start:.3f}s")

        # ----------------------------------------------------------------------
        # 3️⃣ Store DOB
        # ----------------------------------------------------------------------
        iso_dob = dob_date.strftime("%Y-%m-%d")
        session_data[call_sid]["customer"]["dob"] = iso_dob
        session_data[call_sid]["cancel"]["dob"] = iso_dob
        session_data[call_sid].pop("retry_dob", None)
        debug_print(f"collect_dob: ✅ Stored DOB → {iso_dob}")

        # ----------------------------------------------------------------------
        # 4️⃣ Customer lookup
        # ----------------------------------------------------------------------
        t_lookup_start = _time_mod.perf_counter()
        phone_e164 = (
            session_data[call_sid]["customer"].get("phone_e164")
            or session_data[call_sid].get("phone_e164")
        )
        found = False
        if phone_e164 and iso_dob:
            try:
                found = customer_search(phone_number=phone_e164, dob=iso_dob, default_country="US")
                session_data[call_sid]["last_customer_found"] = found
                debug_print(f"collect_dob: 🔎 customer_search(phone={phone_e164}, dob={iso_dob}) → {found}")
            except Exception as e:
                debug_print(f"collect_dob: ⚠️ customer_search error → {e}")
        debug_print(f"collect_dob: ⏱️ lookup duration → {_time_mod.perf_counter() - t_lookup_start:.3f}s")

        # ----------------------------------------------------------------------
        # 5️⃣ Branch to next stage
        # ----------------------------------------------------------------------
        if not found:
            session_data[call_sid]["stage"] = "verify_customer_type"
            g = make_gather(
                "We couldn’t find a record with that phone number and date of birth. "
                "If you are a new customer, press 1. If you are not an existing customer, press 2.",
                input="dtmf",
                timeout=3,
                speech_timeout="auto",
                barge_in=True,
                finish_on_key="#"
            )
            resp.append(g)
            resp.redirect("/voice")
            debug_print(f"collect_dob: ⏱️ total stage duration {_time_mod.perf_counter() - t_stage_start:.3f}s")
            return str(resp)

        # ----------------------------------------------------------------------
        # 6️⃣ Success path → ask for appointment time
        # ----------------------------------------------------------------------
        session_data[call_sid]["stage"] = "ask_time_date"
        g = make_gather(
            "Thanks. Please say the appointment date and time, for example, 'October 12 at 9 AM'.",
            input="speech dtmf",
            timeout=3,
            speech_timeout="auto",
            barge_in=True,
            finish_on_key="#"
        )
        resp.append(g)
        resp.redirect("/voice")
        debug_print(f"collect_dob: ✅ total runtime {_time_mod.perf_counter() - t_stage_start:.3f}s")
        return str(resp)









    # ----------------------------------------------------------------------
    # 🧩 NEW: Verify customer type (after DOB mismatch)
    # ----------------------------------------------------------------------
    elif stage == "verify_customer_type":
        debug_print("verify_customer_type: 📍 Stage entered")
        dtmf_digits = (request.values.get("Digits") or "").strip()
        debug_print(f"verify_customer_type: received DTMF='{dtmf_digits}'")

        if not dtmf_digits:
            g = make_gather(
                "Please press 1 if you are a new customer, or 2 if you are not an existing customer.",
                input="dtmf",
            )
            resp.append(g)
            resp.redirect("/voice")
            return str(resp)

        # Retrieve last lookup flag
        last_lookup_found = session_data.get(call_sid, {}).get("last_customer_found", False)

        # Press 1 → new customer → now CONTINUE to ask_time_date (instead of hangup)
        if dtmf_digits == "1":
            debug_print("verify_customer_type: pressed 1 → new customer (not found) → proceed to ask_time_date")
            session_data[call_sid]["stage"] = "ask_time_date"
            g = make_gather(
                "Welcome! Please say the appointment date and time, for example, 'October 12 at 9 AM'.",
                input="speech dtmf",
            )
            resp.append(g)
            resp.redirect("/voice")
            return str(resp)

        # Press 2 → says not existing customer (explicit no) → polite hangup
        elif dtmf_digits == "2":
            if not last_lookup_found:
                debug_print("verify_customer_type: pressed 2 → confirmed not existing; hanging up politely")
                resp.say(
                    gpt_speak(
                        "Thank you for your time. It seems you are not a current customer in our records. "
                        "Please contact the clinic if you would like to register."
                    ),
                    VOICE,
                )
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)
            else:
                debug_print("verify_customer_type: pressed 2 but found=True (unexpected) → proceed to ask_time_date")
                session_data[call_sid]["stage"] = "ask_time_date"
                g = make_gather(
                    "Okay, please say the appointment date and time, for example, 'October 8 at 9:30 AM'.",
                    input="speech dtmf",
                )
                resp.append(g)
                resp.redirect("/voice")
                return str(resp)

        # Invalid entry
        else:
            debug_print(f"verify_customer_type: invalid DTMF '{dtmf_digits}' → re-prompt")
            g = make_gather(
                "Invalid choice. Press 1 if you are a new customer, or 2 if you are not an existing customer.",
                input="dtmf",
            )
            resp.append(g)
            resp.redirect("/voice")
            return str(resp)






     # ----------------------------------------------------------------------
     # 📅 Stage: ask_time_date
     # Purpose:
     #   - Parse spoken date/time (e.g., “September 12 at 10 AM”) without external helpers.
     #   - Build a concrete UTC timeslot (start/end) using clinic TZ and duration.
     #   - Check availability via is_time_slot_available(calendar_id, start_iso, end_iso, creds).
     #   - If the slot is busy or has fully passed, suggest the next 3 free slots
     #     AFTER the requested *end* via get_next_available_slots(...).
     #   - If free, persist slot and advance the flow.
     #
     # Notes:
     #   - Uses absolute times only (no ±1s padding in this stage).
     #   - We never assign to `_re`, so it stays global and safe.
     #   - Every code path returns `str(resp)` (Flask requirement).
     # ----------------------------------------------------------------------
    elif stage == "ask_time_date":
        debug_print(f"ask_time_date: 🗣️ Received speech: {speech_result}")

        # ------------------------------------------------------------------
        # 📋 Prompts
        # ------------------------------------------------------------------
        PROMPT_NEED_BOTH = (
            "Please say or enter the date and time, for example, 'October 8 at 9:30 AM', "
            "or type month, day, hour, and minute then press pound — for example 10080930#."
        )
        PROMPT_PAST_TIME = "That date and time have already passed. Please choose a future appointment time."
        PROMPT_NEED_VALID_DAY = (
            "That day isn’t available for appointments. "
            "Please choose a weekday between Monday and Saturday, for example, 'October 7 at 10 AM'."
        )

        # ------------------------------------------------------------------
        # Ensure session and doctor context
        # ------------------------------------------------------------------
        session_data.setdefault(call_sid, {})
        doctor_id = session_data.get(call_sid, {}).get("doctor_id")
        if not doctor_id:
            debug_print("ask_time_date: ❌ no doctor selected → choose_doctor")
            session_data[call_sid]["stage"] = "choose_doctor"
            doctor_list = ", ".join(googleid_dr_name_map.values())
            resp.append(make_gather("Which doctor would you like to see?", hints=doctor_list))
            return str(resp)
        calendar_id = doctor_id

        # ------------------------------------------------------------------
        # 🔇 Handle silence
        # ------------------------------------------------------------------
        raw_speech = (speech_result or "").strip()
        raw_dtmf = (request.values.get("Digits") or "").strip()

        if not (raw_speech or raw_dtmf):
            tries = session_data[call_sid].get("silence_time", 0) + 1
            session_data[call_sid]["silence_time"] = tries
            debug_print(f"ask_time_date: 🤐 silence (tries={tries})")

            if tries < 3:
                # Friendly retry
                prompt = (
                    "I didn’t hear the appointment date and time. "
                    "Please say it again, for example, 'October 8 at 9:30 AM'. "
                    "Or type month, day, hour, and minute then press pound."
                )
                resp.append(make_gather(prompt, input="speech dtmf"))
                resp.redirect("/voice")
                return str(resp)
            else:
                # Too many silent attempts → end call
                resp.say(gpt_speak("I'm sorry, I still didn't get your appointment time. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

        session_data[call_sid].pop("silence_time", None)

        # ------------------------------------------------------------------
        # 🧩 Helper: Extract day/time phrases from speech
        # ------------------------------------------------------------------
        def _extract_day_time(s: str) -> tuple:
            """Extract 'October 8 at 9 AM' → ('October 8', '9 AM')"""
            if not s:
                return ("", "")
            s = s.lower()
            s = _re.sub(r"\b(a\s*\.?\s*m\.?)\b", "am", s)
            s = _re.sub(r"\b(p\s*\.?\s*m\.?)\b", "pm", s)
            s = _re.sub(r"\bat\s*[.,]?\s+", " at ", s)
            s = _re.sub(r"[!?;]+", "", s)
            s = _re.sub(r"\s+", " ", s).strip()
            s = s.replace(" at noon", " at 12 pm").replace(" at midnight", " at 12 am")
            MONTH_FIXES = {r"\b10\s*to\s*12\b": "october 12", r"\b9\s*to\s*12\b": "september 12"}
            for pat, repl in MONTH_FIXES.items():
                s = _re.sub(pat, repl, s)
            if " at " in s:
                day, timep = s.split(" at ", 1)
                return (day.strip().rstrip(","), timep.strip())
            m = _re.search(r"\b(\d{1,2}:\d{2}\s*(am|pm)?|\d{1,2}\s*(am|pm)?)\b", s)
            if m:
                timep = m.group(1)
                day = s[:m.start()].strip().rstrip(",")
                return (day, timep)
            return ("", "")

        # ------------------------------------------------------------------
        # 🧩 Helper: Build UTC slot from parsed day/time
        # ------------------------------------------------------------------
        def _build_slot(day_str: str, time_str: str) -> tuple:
            """
            Convert parsed day/time → UTC start & end ISO strings.

            Example:
            'October 8' + '9:30 AM'
            → localize to America/Chicago → convert to UTC → return ISO Z strings.
            """
            tz_name = globals().get("CLINIC_TZ", "America/Chicago")
            tz_local = _pytz.timezone(tz_name)
            dur = int(globals().get("APPOINTMENT_DURATION_MINUTES", 30))
            combined = f"{day_str} at {time_str}"

            today = _date_local.today()
            # ✅ Ensure default base is timezone-aware (prevents naive → local mismatch)
            default_base = tz_local.localize(datetime(today.year, today.month, today.day, 9, 0, 0))
            parsed = _dtparse(combined, default=default_base, fuzzy=True)

            # Default noon → PM if no AM/PM given
            if not _re.search(r"(am|pm)", combined, _re.IGNORECASE) and parsed.hour == 12:
                parsed = parsed.replace(hour=12)

            if parsed.tzinfo is None:
                parsed = tz_local.localize(parsed)
            else:
                parsed = parsed.astimezone(tz_local)

            # If year missing → assume current year
            if not _re.search(r"\b\d{4}\b", combined):
                parsed = parsed.replace(year=today.year)

            # Check weekday validity
            working_days = globals().get("WORKING_DAYS", (0, 1, 2, 3, 4, 5))
            if parsed.weekday() not in working_days:
                raise ValueError("invalid_weekday")

            start_local = parsed
            end_local = start_local + timedelta(minutes=dur)

            # Convert both start/end to UTC for consistency
            start_utc = start_local.astimezone(_pytz.UTC).isoformat().replace("+00:00", "Z")
            end_utc = end_local.astimezone(_pytz.UTC).isoformat().replace("+00:00", "Z")
            return (start_utc, end_utc)

        # ------------------------------------------------------------------
        # 🧠 Parse input (speech or DTMF)
        # ------------------------------------------------------------------
        appointment_start, appointment_end = None, None
        try:
            if raw_dtmf:
                digits = _re.sub(r"\D", "", raw_dtmf)
                debug_print(f"ask_time_date: 📟 DTMF entered → {digits}")
                today = _date_local.today()
                if len(digits) >= 8:  # MMDDHHMM
                    mm, dd, hh, mn = int(digits[0:2]), int(digits[2:4]), int(digits[4:6]), int(digits[6:8])
                    day_str = f"{today.year}-{mm:02d}-{dd:02d}"
                    time_str = f"{hh}:{mn:02d}"
                elif len(digits) == 4:  # HHMM only
                    day_str = today.strftime("%Y-%m-%d")
                    hh, mn = int(digits[0:2]), int(digits[2:4])
                    time_str = f"{hh}:{mn:02d}"
                else:
                    raise ValueError("invalid_dtmf_format")

                if hh == 12:  # default PM if ambiguous
                    time_str = f"{hh}:{mn:02d} PM"

                appointment_start, appointment_end = _build_slot(day_str, time_str)
            else:
                day_part, time_part = _extract_day_time(raw_speech)
                if not day_part or not time_part:
                    resp.append(make_gather(PROMPT_NEED_BOTH))
                    return str(resp)
                appointment_start, appointment_end = _build_slot(day_part, time_part)

            debug_print(f"ask_time_date: ⏰ Built slot → Start={appointment_start}, End={appointment_end}")
            debug_print(f"ask_time_date: 🌐 Slot in UTC → start={appointment_start}, end={appointment_end}")

        except ValueError as e:
            err = str(e)
            debug_print(f"ask_time_date: ❌ parse/build error → {err}")
            if "invalid_weekday" in err:
                resp.append(make_gather(PROMPT_NEED_VALID_DAY))
            else:
                resp.append(make_gather(PROMPT_NEED_BOTH))
            resp.redirect("/voice")
            return str(resp)

        # ------------------------------------------------------------------
        # ⏳ Reject past dates — offer alternatives
        # ------------------------------------------------------------------
        now_utc = _pytz.UTC.localize(datetime.utcnow())
        start_dt = datetime.fromisoformat(appointment_start.replace("Z", "+00:00"))
        if start_dt <= now_utc:
            debug_print("ask_time_date: 🕒 requested time is in the past → suggesting alternatives")
            alts = get_next_available_slots(calendar_id, creds, from_start_iso=appointment_end, limit=3) or []
            options = " or ".join([a.get("friendly", "") for a in alts if a.get("friendly")])
            prompt = f"That time has already passed. Would you like {options}?" if options else PROMPT_PAST_TIME
            resp.append(make_gather(prompt))
            return str(resp)

        # ------------------------------------------------------------------
        # 🔍 Check availability
        # ------------------------------------------------------------------
        try:
            slot_available = is_time_slot_available(calendar_id, appointment_start, appointment_end, creds)
        except Exception as e:
            debug_print(f"ask_time_date: ⚠️ Availability check error → {e}")
            slot_available = False

        if not slot_available:
            debug_print("ask_time_date: ❌ Slot not available → suggesting alternatives")
            alts = get_next_available_slots(calendar_id, creds, from_start_iso=appointment_end, limit=3) or []
            options = " or ".join([a.get("friendly", "") for a in alts if a.get("friendly")])
            prompt = f"That time is not available. Would you like {options}?" if options else "That time is not available. Please say another date and time."
            resp.append(make_gather(prompt))
            return str(resp)

        # ------------------------------------------------------------------
        # ✅ Slot available — continue booking
        # ------------------------------------------------------------------
        session_data[call_sid]["appointment_time"] = {"start": appointment_start, "end": appointment_end}
        reschedule_flag = session_data.get(call_sid, {}).get("reschedule_after_cancel", False)
        if reschedule_flag:
            cancel_info = session_data[call_sid].get("cancel", {})
            cust = session_data[call_sid].setdefault("customer", {})
            if cancel_info.get("phone_e164"):
                cust["phone_e164"] = cancel_info["phone_e164"]
            if cancel_info.get("dob"):
                cust["dob"] = cancel_info["dob"]
            session_data[call_sid]["reschedule_after_cancel"] = False
            debug_print("ask_time_date: 🔁 reused phone/DOB from cancel flow for reschedule")

        cust = session_data[call_sid].setdefault("customer", {})
        phone_e164 = cust.get("phone_e164") or session_data[call_sid].get("phone_e164")
        dob = cust.get("dob") or session_data[call_sid].get("dob")

        if not phone_e164 or not dob:
            session_data[call_sid]["stage"] = "collect_phone" if not phone_e164 else "collect_dob"
            prompt = "Please say your 10-digit phone number." if not phone_e164 else "Please say your date of birth, for example, 'July third 1990'."
            resp.append(make_gather(prompt))
            resp.redirect("/voice")
            return str(resp)

        try:
            # ✅ Fixed argument name (default_country instead of country)
            found = customer_search(phone_number=phone_e164, dob=dob, default_country="US")
            debug_print(f"ask_time_date: 🔎 customer_search(phone={phone_e164}, dob={dob}) → {found}")
        except Exception as e:
            debug_print(f"ask_time_date: ⚠️ customer_search error → {e}")
            found = False

        session_data[call_sid]["stage"] = "book_appt_confirm" if found else "collect_first_name"
        debug_print(f"ask_time_date: 🎯 Next stage → {session_data[call_sid]['stage']}")
        resp.redirect("/voice")
        return str(resp)












    # ===== collect_first_name (stage) =====
    elif stage == "collect_first_name":
        # ----------------------------------------------------------------------
        # 🎯 Goal:
        #   - Capture FIRST name via speech or keypad (DTMF).
        #   - Handle silence separately (up to 3 retries before hangup).
        #   - Accept Arabic names written in English letters (e.g., "Mohamed", "Hossam").
        #   - Reject true Arabic script (e.g., "ﻢﺤﻣﺩ").
        #   - Store under session_data[call_sid]["customer"]["first_name"].
        #   - Advance → collect_last_name.
        # ----------------------------------------------------------------------
        session_data.setdefault(call_sid, {}).setdefault("customer", {})
        raw_speech = (speech_result or "").strip()
        raw_dtmf = (request.values.get("Digits") or "").strip()
        debug_print(f"collect_first_name: speech='{raw_speech}', dtmf='{raw_dtmf}'")

        # -------------------------------
        # 🔇 Silence Handling
        # -------------------------------
        if not raw_speech and not raw_dtmf:
            tries = session_data[call_sid].get("silence_first_name", 0) + 1
            session_data[call_sid]["silence_first_name"] = tries
            debug_print(f"collect_first_name: 🤐 silence; tries={tries}")
            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            gather = make_gather(
                "I didn’t hear your first name. Please say your name in English letters. For example, Mohamed or Hossam.",
                input="speech dtmf",
                language="en-US",   # English model (Arabic names spoken in English)
                hints="Mohamed, Ahmad, Hossam, Khalil",
                timeout=6,
                speech_timeout="5",
                finish_on_key="#",
                barge_in=True,
            )
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # ✅ Clear silence counter once input arrives
        session_data[call_sid].pop("silence_first_name", None)

        # -------------------------------
        # 🧾 Parse & Clean Input
        # -------------------------------
        import string
        if raw_dtmf:
            # Keypad fallback
            name_digits = _re.sub(r"\D", "", raw_dtmf)
            first_name = f"User{name_digits[-3:]}" if name_digits else "Unknown"
            debug_print(f"collect_first_name: 🧮 from keypad → {first_name}")
        else:
            cleaned = raw_speech.translate(str.maketrans('', '', string.punctuation)).strip()
            cleaned = _re.sub(r"\s+", " ", cleaned)
            # Drop filler phrases
            cleaned = _re.sub(
                r"\b(?:my name is|this is|i am|i'm|it is|it's)\b\s*",
                "",
                cleaned,
                flags=_re.IGNORECASE,
            )
            tokens = cleaned.split()
            first_name = tokens[0] if tokens else ""

        # -------------------------------
        # 🌐 Accept only English letters
        # -------------------------------
        # Allow A–Z, a–z, hyphens, apostrophes, and spaces.
        english_only_pattern = r"^[A-Za-z][A-Za-z'\-\s]{0,39}$"

        # Reject Arabic Unicode range (0621–064A)
        contains_arabic = bool(_re.search(r"[\u0600-\u06FF]", first_name))

        if not first_name or not _re.fullmatch(english_only_pattern, first_name) or contains_arabic:
            r = session_data[call_sid].get("retry_first_name", 0) + 1
            session_data[call_sid]["retry_first_name"] = r
            debug_print(f"collect_first_name: ❌ invalid or Arabic-script name '{first_name}' retry={r}")
            if r >= 3:
                resp.say(gpt_speak("Sorry, I couldn’t capture your name in English letters. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            gather = make_gather(
                "Please say your name using English letters only. For example, Mohamed or Hossam.",
                input="speech dtmf",
                language="en-US",
                hints=ARABIC_NAME_HINTS,
                timeout=6,
                speech_timeout="5",
                finish_on_key="#",
                barge_in=True,
            )
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # -------------------------------
        # ✅ Save & Continue
        # -------------------------------
        session_data[call_sid]["customer"]["first_name"] = first_name
        session_data[call_sid]["stage"] = "collect_last_name"
        session_data[call_sid].pop("retry_first_name", None)
        debug_print(f"collect_first_name: ✅ saved first_name='{first_name}' → next=collect_last_name")

        gather = make_gather(
            f"Thank you {first_name}. Now, what is your last name?",
            input="speech dtmf",
            language="en-US",
            hints=ARABIC_NAME_HINTS,
            timeout=6,
            speech_timeout="5",
            finish_on_key="#",
            barge_in=True,
        )
        resp.append(gather)
        resp.redirect("/voice")
        return str(resp)

    



    

    elif stage == "collect_last_name":
        # ----------------------------------------------------------------------
        # 🎯 Goal:
        #   - Capture LAST name via speech or keypad (DTMF).
        #   - Handle silence separately (up to 3 retries).
        #   - Accept Arabic names written in English letters only (e.g., "Khalil").
        #   - Reject Arabic-script text (e.g., "ﻖﻠﻴﻟ", "خليل").
        #   - Store → session_data[call_sid]["customer"]["last_name"].
        #   - Advance → collect_address.
        # ----------------------------------------------------------------------
        session_data.setdefault(call_sid, {}).setdefault("customer", {})
        raw_speech = (speech_result or "").strip()
        raw_dtmf = (request.values.get("Digits") or "").strip()
        debug_print(f"collect_last_name: speech='{raw_speech}', dtmf='{raw_dtmf}'")

        # -------------------------------
        # 🔇 Silence handling
        # -------------------------------
        if not raw_speech and not raw_dtmf:
            tries = session_data[call_sid].get("silence_last_name", 0) + 1
            session_data[call_sid]["silence_last_name"] = tries
            debug_print(f"collect_last_name: 🤐 silence; tries={tries}")
            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Prompt again quickly (English only)
            gather = make_gather(
                "I didn’t hear your last name. Please say your last name in English letters, for example, Khalil or ElSayed.",
                input="speech dtmf",
                language="en-US",  # English ASR
                hints="Khalil, ElSayed, Hassan, Nasser",
                timeout=6,
                speech_timeout="5",
                finish_on_key="#",
                barge_in=True,
            )
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # ✅ clear silence flag
        session_data[call_sid].pop("silence_last_name", None)

        # -------------------------------
        # 🧾 Parse & Clean
        # -------------------------------
        import string
        if raw_dtmf:
            digits = _re.sub(r"\D", "", raw_dtmf)
            last_name = f"Family{digits[-3:]}" if digits else "Unknown"
            debug_print(f"collect_last_name: 🧮 from keypad → {last_name}")
        else:
            punct_keep = "'-"
            cleaned = raw_speech.translate(
                str.maketrans('', '', "".join(ch for ch in string.punctuation if ch not in punct_keep))
            ).strip()
            cleaned = _re.sub(r"\s+", " ", cleaned)
            # drop fillers like "my last name is", "family name"
            cleaned = _re.sub(
                r"\b(?:my last name is|family name is|this is|i am|i'm|it's)\b\s*",
                "",
                cleaned,
                flags=_re.IGNORECASE,
            )
            tokens = cleaned.split()
            last_name = tokens[0] if tokens else ""

        # -------------------------------
        # 🌐 Validate: English letters only
        # -------------------------------
        english_only_pattern = r"^[A-Za-z][A-Za-z'\-\s]{0,59}$"
        contains_arabic = bool(_re.search(r"[\u0600-\u06FF]", last_name))

        if not last_name or not _re.fullmatch(english_only_pattern, last_name) or contains_arabic:
            r = session_data[call_sid].get("retry_last_name", 0) + 1
            session_data[call_sid]["retry_last_name"] = r
            debug_print(f"collect_last_name: ❌ invalid (non-English) name '{last_name}' retry={r}")
            if r >= 3:
                resp.say(gpt_speak("Sorry, I couldn’t capture your last name in English letters. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            gather = make_gather(
                "Please say your last name using English letters only. For example, Khalil or ElSayed.",
                input="speech dtmf",
                language="en-US",
                hints=ARABIC_NAME_HINTS,
                timeout=6,
                speech_timeout="5",
                finish_on_key="#",
                barge_in=True,
            )
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # -------------------------------
        # ✅ Save & Advance
        # -------------------------------
        session_data[call_sid]["customer"]["last_name"] = last_name
        session_data[call_sid]["stage"] = "collect_address"
        session_data[call_sid].pop("retry_last_name", None)
        debug_print(f"collect_last_name: ✅ saved last_name='{last_name}' → next=collect_address")

        gather = make_gather(
            f"Thank you {session_data[call_sid]['customer'].get('first_name','')} {last_name}. "
            "Please tell me your full address.",
            input="speech dtmf",
            language="en-US",
            hints="118 Briar Oak Murphy Texas 75094",
            timeout=7,
            speech_timeout="5",
            finish_on_key="#",
            barge_in=True,
        )
        resp.append(gather)
        resp.redirect("/voice")
        return str(resp)






    elif stage == "collect_address":
        # ----------------------------------------------------------------------
        # 📬 Stage: collect_address
        # Purpose:
        #   - Capture full street address via speech (and optionally DTMF chunks).
        #   - Normalize punctuation/whitespace.
        #   - Store into session_data[call_sid]["customer"]["address"].
        #   - Advance → collect_cc.
        # Notes:
        #   - Make sure we ALWAYS return TwiML: return str(resp)
        #   - Use resp.redirect("/voice") (TwiML) not Flask redirect()
        # ----------------------------------------------------------------------

        raw_addr = (speech_result or "").strip()
        debug_print(f"collect_address: 📬 Collected address (raw): {raw_addr}")

        # 🔇 Silence handling (3 tries then hang up)
        if not raw_addr:
            tries = session_data[call_sid].get("silence_address", 0) + 1
            session_data[call_sid]["silence_address"] = tries
            debug_print(f"collect_address: 🤐 silence; tries={tries}")
            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            gather = make_gather("Sorry, I didn’t hear your address. Please say your full street address, city, and ZIP.")
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # 🧽 Normalize punctuation/whitespace
        try:
            #import re as _re
            addr_norm = raw_addr
            # collapse multiple spaces
            addr_norm = _re.sub(r"\s+", " ", addr_norm)
            # normalize punctuation spacing (keep commas and periods if they’re spoken)
            addr_norm = addr_norm.strip()
            debug_print(f"collect_address: 🧽 Normalized → '{addr_norm}'")
        except Exception as e:
            debug_print(f"collect_address: ⚠️ normalize error → {e}")
            addr_norm = raw_addr

        # ✅ Save to session
        session_data.setdefault(call_sid, {}).setdefault("customer", {})
        session_data[call_sid]["customer"]["address"] = addr_norm
        session_data[call_sid].pop("silence_address", None)  # reset silence counter
        debug_print("collect_address: ✅ Saved address to session")

        # ➡️ Advance to CC collection
        session_data[call_sid]["stage"] = "collect_cc"
        session_data[call_sid]["cc_step"] = 1  # ensure we begin at step 1

        # Prompt for card number (DTMF preferred, speech allowed)
        prompt_cc = "Thank you. Now, please enter your card number, then press pound."
        gather = make_gather(
            prompt_cc,
            hints="zero one two three four five six seven eight nine",
            input="speech dtmf",
            timeout=25,
            speech_timeout="10",
            finish_on_key="#",
            action="/voice",
            barge_in=True,
        )
        resp.append(gather)
        # IMPORTANT: TwiML redirect, then return TwiML
        resp.redirect("/voice")
        return str(resp)




   




    elif stage == "collect_cc":
        # ----------------------------------------------------------------------
        # 💳 Stage: collect_cc  (optimized for shorter expiration & CVV steps)
        #
        # Flow:
        #   (1) Card number (13–19 digits, Luhn-checked)
        #   (2) Expiration (MMYY or MMYYYY, current/future only)
        #   (3) CVV (3–4 digits, DTMF-only for speed)
        #
        # Improvements:
        #   - Shorter timeouts for steps 2 & 3 (expiration, CVV)
        #   - DTMF-only for CVV (speech removed for faster processing)
        #   - Immediate step advance after # pressed
        # ----------------------------------------------------------------------

        # --- helpers ------------------------------------------------------------
        def _luhn_ok(pan: str) -> bool:
            s, alt = 0, False
            for ch in pan[::-1]:
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

        def _normalize_spoken_digits(raw: str) -> str:
            if not raw:
                return ""
            words = (
                raw.lower()
                .replace("-", " ").replace(",", " ").replace(".", " ")
                .split()
            )
            m = {
                "zero":"0","oh":"0","o":"0",
                "one":"1","two":"2","to":"2","too":"2",
                "three":"3","four":"4","for":"4",
                "five":"5","six":"6","seven":"7",
                "eight":"8","ate":"8","nine":"9"
            }
            out = []; i = 0
            while i < len(words):
                w = _re.sub(r"[^a-z0-9]", "", words[i])
                if w in ("double","triple") and i+1 < len(words):
                    nxt = _re.sub(r"[^a-z0-9]", "", words[i+1])
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

        def _digits_from(dtmf: str, speech: str, *, enforce_dtmf: bool) -> str:
            if enforce_dtmf:
                return _re.sub(r"\D", "", dtmf or "")
            if dtmf:
                return _re.sub(r"\D", "", dtmf)
            return _re.sub(r"\D", "", _normalize_spoken_digits(speech or ""))

        def _mask(pan: str) -> str:
            pan = pan or ""
            if len(pan) <= 4: return pan
            return "*" * (len(pan) - 4) + pan[-4:]

        # --- state --------------------------------------------------------------
        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("customer", {})
        customer   = session_data[call_sid]["customer"]
        cc_step    = int(session_data[call_sid].get("cc_step", 1))
        enforce_dm = bool(session_data[call_sid].get("enforce_dtmf_cc"))

        raw_dtmf   = (request.values.get("Digits") or "").strip()
        raw_speech = (speech_result or "").strip()

        debug_print(f"collect_cc: 📍 step={cc_step}, DTMF='{raw_dtmf}', speech='{raw_speech}'")

        # -------------------------------
        # 🔇 Silence handling (inline)
        # -------------------------------
        if not raw_dtmf and not raw_speech:
            tries = session_data[call_sid].get("silence_cc", 0) + 1
            session_data[call_sid]["silence_cc"] = tries
            debug_print(f"collect_cc: 🤐 silence on step {cc_step}; tries={tries}")

            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            prompt = {
                1: "Please enter your card number now, then press pound.",
                2: "Please enter the expiration as two digits for month and two digits for year, then press pound.",
                3: "Please enter the three or four digit security code, then press pound."
            }.get(cc_step, "Please enter your card details, then press pound.")

            gather = make_gather(
                prompt,
                input="speech dtmf",
                timeout=20,
                speech_timeout="8",
                finish_on_key="#",
                action="/voice",
                barge_in=True,
            )
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        session_data[call_sid].pop("silence_cc", None)

        # -------------------------------
        # Step 1: Card Number (13–19)
        # -------------------------------
        if cc_step == 1:
            pan = _digits_from(raw_dtmf, raw_speech, enforce_dtmf=enforce_dm)
            if len(pan) > 19:
                pan = pan[:19]

            if not (13 <= len(pan) <= 19) or not _luhn_ok(pan):
                # Retry with DTMF enforcement after 2 speech failures
                if not raw_dtmf:
                    session_data[call_sid]["cc_speech_tries"] = session_data[call_sid].get("cc_speech_tries", 0) + 1
                    if session_data[call_sid]["cc_speech_tries"] >= 2:
                        session_data[call_sid]["enforce_dtmf_cc"] = True
                        debug_print("collect_cc: 📟 enforcing DTMF for card number entry")
                        gather = make_gather(
                            "That number didn’t sound clear. Please TYPE the full card number now, then press pound.",
                            input="dtmf",
                            timeout=20,
                            finish_on_key="#",
                            action="/voice",
                        )
                        resp.append(gather)
                        resp.redirect("/voice")
                        return str(resp)

                gather = make_gather(
                    "That card number doesn't look right. Please re-enter the full card number, then press pound.",
                    input="speech dtmf",
                    timeout=20,
                    speech_timeout="8",
                    finish_on_key="#",
                    action="/voice",
                )
                resp.append(gather)
                resp.redirect("/voice")
                return str(resp)

            # ✅ Save & advance
            customer["cc_number"] = pan
            session_data[call_sid]["cc_step"] = 2
            session_data[call_sid]["cc_speech_tries"] = 0
            debug_print(f"collect_cc: ✅ Saved card number '{_mask(pan)}' → step 2 (Expiration)")
            resp.redirect("/voice")
            return str(resp)

        # -------------------------------
        # Step 2: Expiration (MMYY/MMYYYY)
        # -------------------------------
        if cc_step == 2:
            session_data[call_sid]["no_input_expected"] = True  # DTMF-only preferred
            digits = _digits_from(raw_dtmf, raw_speech, enforce_dtmf=True)
            debug_print(f"collect_cc: Step 2 digits='{digits}'")

            if len(digits) not in (4, 6):
                gather = make_gather(
                    "Please enter expiration as two digits for month and two digits for year, for example 0 9 2 7, then press pound.",
                    input="dtmf",
                    timeout=8,               # ⏱ shorter wait
                    finish_on_key="#",
                    action="/voice",
                )
                resp.append(gather)
                resp.redirect("/voice")
                return str(resp)

            try:
                mm = int(digits[:2])
                yy = digits[-2:]
                if not (1 <= mm <= 12):
                    raise ValueError("invalid month")
                now = datetime.now(tz=_pytz.UTC)
                exp_year = 2000 + int(yy)
                expiry_boundary = datetime(exp_year, mm, 1, 0, 0, 0, tzinfo=_pytz.UTC) + timedelta(days=31)
                if now >= expiry_boundary:
                    raise ValueError("expired")

                customer["cc_exp"] = f"{mm:02d}/{yy}"
                debug_print(f"collect_cc: ✅ Expiration saved → {customer['cc_exp']}")
            except Exception as e:
                debug_print(f"collect_cc: ❌ Expiration parse failed → {e}")
                gather = make_gather(
                    "That doesn’t look valid. Please enter month and year as M M Y Y, then press pound.",
                    input="dtmf",
                    timeout=8,
                    finish_on_key="#",
                    action="/voice",
                )
                resp.append(gather)
                resp.redirect("/voice")
                return str(resp)

            # Advance immediately
            session_data[call_sid]["cc_step"] = 3
            resp.redirect("/voice")
            return str(resp)

        # -------------------------------
        # Step 3: CVV (3–4 digits)
        # -------------------------------
        if cc_step == 3:
            session_data[call_sid]["no_input_expected"] = True
            digits = _digits_from(raw_dtmf, raw_speech, enforce_dtmf=True)
            debug_print(f"collect_cc: Step 3 CVV digits='{digits}'")

            if not (3 <= len(digits) <= 4 and digits.isdigit()):
                gather = make_gather(
                    "Please enter the three or four digit security code, then press pound.",
                    input="dtmf",
                    timeout=6,              # ⏱ shorter wait
                    finish_on_key="#",
                    action="/voice",
                )
                resp.append(gather)
                resp.redirect("/voice")
                return str(resp)

            customer["cc_cvv"] = digits
            if not customer.get("cc_name"):
                customer["cc_name"] = f"{customer.get('first_name','')} {customer.get('last_name','')}".strip()
            debug_print(f"collect_cc: ✅ CVV saved (len={len(digits)}) ; cc_name='{customer.get('cc_name')}'")

            # Advance to confirmation or update flow
            session_data[call_sid].pop("no_input_expected", None)
            session_data[call_sid].pop("cc_step", None)
            session_data[call_sid]["cc_speech_tries"] = 0

            next_stage = (
                "update_customer_cc"
                if session_data.get(call_sid, {}).get("cc_update", {}).get("active")
                else "book_appt_confirm"
            )
            session_data[call_sid]["stage"] = next_stage
            session_data[call_sid]["skip_silence_once"] = True
            debug_print(f"collect_cc: ➡️ Auto-advancing to {next_stage}")
            resp.redirect("/voice")
            return str(resp)












    
    








    elif stage == "cancel_appt_get_phone_number":
        # ----------------------------------------------------------------------
        # 📞 Collect phone number used when booking, then move to DOB check.
        #  - Silent-mode aware (re-prompts up to 3x if nothing is heard)
        #  - Accepts DTMF or speech
        #  - Normalizes to E.164 ONLY (US/Egypt supported)
        #  - Stores under cancel + mirrors into customer for reschedule flows
        #  - Next stage: cancel_appt_get_dob
        # ----------------------------------------------------------------------
        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("cancel", {})
        session_data[call_sid].setdefault("customer", {})  # ✅ mirror for reschedule

        # Pull inputs
        try:
            dtmf_digits = (request.values.get("Digits") or "").strip()
        except Exception:
            dtmf_digits = ""
        speech_text = (speech_result or "").strip()

        debug_print(
            f"cancel_appt_get_phone_number: 🗣️ speech='{speech_text}' 🔢 DTMF='{dtmf_digits}'"
        )

        # 🔇 Silent mode handling
        if not (speech_text or dtmf_digits):
            tries = session_data[call_sid].get("silence_cancel_phone", 0) + 1
            session_data[call_sid]["silence_cancel_phone"] = tries
            debug_print(f"cancel_appt_get_phone_number: 🤐 silence count={tries}")

            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            prompt = (
                "I didn’t hear your phone number. Please say or type your phone number including area code, then press pound."
            )
            session_data[call_sid]["stage"] = "cancel_appt_get_phone_number"
            resp.append(make_gather(prompt, hints="zero one two three four five six seven eight nine double triple"))
            resp.redirect("/voice")
            return str(resp)

        session_data[call_sid].pop("silence_cancel_phone", None)

        # --- helpers --------------------------------------------------------------
        def _spoken_to_digits(raw: str) -> str:
            """Convert spoken words to digits."""
            if not raw:
                return ""
            words = (
                raw.lower()
                .replace("-", " ").replace(",", " ").replace(".", " ")
                .replace("(", " ").replace(")", " ")
                .split()
            )
            m = {
                "zero": "0", "oh": "0", "o": "0",
                "one": "1", "two": "2", "to": "2", "too": "2",
                "three": "3", "four": "4", "for": "4",
                "five": "5", "six": "6", "seven": "7",
                "eight": "8", "ate": "8", "nine": "9"
            }
            out = []; i = 0
            while i < len(words):
                w = words[i].strip()
                if w in ("double", "triple") and i + 1 < len(words):
                    nxt = words[i + 1].strip()
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

        # Normalize
        raw_digits = _re.sub(r"\D", "", dtmf_digits) if dtmf_digits else _re.sub(r"\D", "", _spoken_to_digits(speech_text))
        default_country = (session_data[call_sid].get("country") or COUNTRY or "US").upper()
        raw_for_e164 = (speech_text or raw_digits or "").strip()
        phone_e164 = ""

        try:
            if raw_for_e164.startswith("+"):
                digits = "".join(ch for ch in raw_for_e164[1:] if ch.isdigit())
                if 8 <= len(digits) <= 15:
                    phone_e164 = "+" + digits

            if not phone_e164:
                debug_print(f"cancel_appt_get_phone_number: normalizing via {default_country} from='{raw_for_e164}'")
                phone_e164 = normalize_phone_e164(raw_for_e164, default_country) or ""

            if not phone_e164 and raw_digits:
                phone_e164 = normalize_phone_e164(raw_digits, default_country) or ""

            if not phone_e164:
                alt = "EG" if default_country != "EG" else "US"
                debug_print(f"cancel_appt_get_phone_number: retry via alt country={alt}")
                phone_e164 = normalize_phone_e164(raw_for_e164 or raw_digits, alt) or ""
        except Exception as e:
            debug_print(f"cancel_appt_get_phone_number: ⚠️ normalize_phone_e164 error → {e}")
            phone_e164 = ""

        debug_print(
            f"cancel_appt_get_phone_number: 🧪 parsed digits='{raw_digits}' default_country='{default_country}' → e164='{phone_e164 or '∅'}'"
        )

        # Validate E.164
        if not phone_e164:
            session_data[call_sid]["stage"] = "cancel_appt_get_phone_number"
            prompt = (
                "I didn’t catch a valid phone number. Please say or type your phone number including area code, then press pound."
            )
            resp.append(make_gather(prompt, hints="zero one two three four five six seven eight nine double triple"))
            resp.redirect("/voice")
            return str(resp)

        # ✅ Store and mirror (for consistency + reschedule support)
        session_data[call_sid]["cancel"]["phone_e164"] = phone_e164
        session_data[call_sid]["customer"]["phone_e164"] = phone_e164  # ✅ mirror
        session_data[call_sid]["phone_e164"] = phone_e164              # ✅ top-level convenience
        debug_print(f"cancel_appt_get_phone_number: ✅ saved phone_e164={phone_e164}")

        # Next stage: cancel_appt_get_dob
        session_data[call_sid]["stage"] = "cancel_appt_get_dob"
        gather = make_gather(
            "Thanks. Now, please tell me your date of birth to verify your identity. "
            "For example, say July third 1990, or type it as 07031990 then press pound."
        )
        resp.append(gather)
        resp.redirect("/voice")
        return str(resp)









    elif stage == "cancel_appt_get_dob":
        debug_print("cancel_appt_get_dob: 📍 Stage entered")

        session_data.setdefault(call_sid, {}).setdefault("customer", {})
        session_data[call_sid].setdefault("cancel", {})

        DOB_PROMPT = (
            "Please say your birth date, for example July third nineteen fifty six, "
            "or type 2 digits for month 2 digits for day and 4 digits for year, then press pound."
        )

        # --- Inputs ---
        dtmf_digits = (request.values.get("Digits") or "").strip()
        speech_text = (speech_result or "").strip()
        debug_print(f"cancel_appt_get_dob: 🎙️ speech_text='{speech_text}', 🔢 dtmf_digits='{dtmf_digits}'")

        # --- Silent ---
        if not dtmf_digits and not speech_text:
            tries = session_data[call_sid].get("silence_cancel_dob", 0) + 1
            session_data[call_sid]["silence_cancel_dob"] = tries
            if tries >= 3:
                resp.say("I’m still not hearing anything. Please call again later.", VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)
            resp.append(make_gather(DOB_PROMPT))
            return str(resp)

        session_data[call_sid].pop("silence_cancel_dob", None)

        # --- Parse DOB ---
        try:
            import dateutil.parser as dp
            dt = None
            if dtmf_digits:
                if len(dtmf_digits) == 8:  # MMDDYYYY
                    m, d, y = int(dtmf_digits[0:2]), int(dtmf_digits[2:4]), int(dtmf_digits[4:8])
                    dt = datetime(y, m, d)
            if not dt and speech_text:
                dt = dp.parse(speech_text, fuzzy=True)
        except Exception as e:
            debug_print(f"cancel_appt_get_dob: ❌ parse error {e}")
            dt = None

        if not dt:
            retries = session_data[call_sid].get("retry_cancel_dob", 0) + 1
            session_data[call_sid]["retry_cancel_dob"] = retries
            if retries >= 3:
                resp.say("Sorry, I couldn’t understand your date of birth. Please call again later.", VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)
            resp.append(make_gather(DOB_PROMPT))
            return str(resp)

        # --- Store DOB ---
        iso_dob = dt.strftime("%Y-%m-%d")
        session_data[call_sid]["customer"]["dob"] = iso_dob
        session_data[call_sid]["cancel"]["dob"]   = iso_dob
        session_data[call_sid].pop("retry_cancel_dob", None)
        debug_print(f"cancel_appt_get_dob: ✅ Stored DOB → {iso_dob}")

        # --- Next Stage ---
        session_data[call_sid]["stage"] = "cancel_appt_get_time_date"
        resp.append(make_gather("Thanks. Now, please tell me the date and time of the appointment you want to cancel. For example, say July 3rd at 9 AM."))
        return str(resp)




    elif stage == "cancel_appt_get_time_date":
        debug_print("cancel_appt_get_time_date: 📍 Stage entered")
        cancel_ctx = session_data[call_sid].setdefault("cancel", {})
        raw = (speech_result or "").strip()
        debug_print(f"cancel_appt_get_time_date: 🗣️ Raw speech → '{raw}'")

        # always reset retries if we got new input
        if raw:
            cancel_ctx.pop("retry_cancel_dt", None)
            cancel_ctx.pop("silence_cancel_dt", None)

        # ----------------- Parse attempt -----------------
        day_part, time_part = (None, None)
        if " at " in raw.lower():
            parts = raw.lower().replace(",", "").split("at")
            if len(parts) == 2:
                day_part, time_part = parts[0].strip(), parts[1].strip()

        debug_print(f"cancel_appt_get_time_date: 📆 Extracted → Day='{day_part}', Time='{time_part}'")

        # ----------------- Always check against DB -----------------
        matched = False
        if day_part and time_part:
            # here you’d normally map to UTC + check Google/JSON
            events = []  # replaced with actual lookup
            if events:
                cancel_ctx["matching_event"] = events[0]
                session_data[call_sid]["stage"] = "cancel_appt_confirm"
                resp.redirect("/voice")
                return str(resp)

        # ----------------- Force iterate if no match -----------------
        debug_print("cancel_appt_get_time_date: 🚫 no match → switch to iterate (ignore input)")
        cancel_ctx.pop("matching_event", None)          # ✅ clear stray
        session_data[call_sid]["stage"] = "cancel_appt_iterate"
        cancel_ctx["awaiting_input"] = False            # ✅ first run announce-only
        session_data[call_sid]["skip_silence_retry"] = True  # ✅ disable silence detection
        resp.say(gpt_speak("That doesn’t match any of your appointments. I’ll list your upcoming ones."), VOICE)
        resp.redirect("/voice")
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
        #  - ✨ Updated to also support DTMF digit selection for doctors.
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
            return " ".join(s.split())  # squeeze internal whitespace

        # Pull speech
        selected_text = (speech_result or "").strip()

        # Build doctor keypad map for this session
        doctor_names = list(googleid_dr_name_map.values())
        doctor_dtmf_map = {str(i + 1): doc for i, doc in enumerate(doctor_names)}
        session_data[call_sid]["doctor_dtmf_map"] = doctor_dtmf_map

        # ------------------------------
        # 🔇 Silent-mode handling first
        # ------------------------------
        if not selected_text and not dtmf_digits:
            tries = session_data[call_sid].get("silence_cancel_doc", 0) + 1
            session_data[call_sid]["silence_cancel_doc"] = tries
            debug_print(f"cancel_appointment: 🤐 No input detected (silence count={tries})")

            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Build friendly doctor list with press options
            options = ". ".join([f"{doc} (press {k})" for k, doc in doctor_dtmf_map.items()])
            retry_prompt = (
                f"I didn't hear the doctor's name. Available doctors are: {options}. "
                "Please say the name of the doctor or press the number."
            )
            session_data[call_sid]["stage"] = "cancel_appointment"
            resp.append(make_gather(retry_prompt, hints=", ".join(doctor_names), num_digits=1))
            return str(resp)

        # If DTMF digit was pressed → direct map
        if dtmf_digits and dtmf_digits in doctor_dtmf_map:
            matched_name = doctor_dtmf_map[dtmf_digits]
            matched_id = next(k for k, v in googleid_dr_name_map.items() if v == matched_name)
            debug_print(f"cancel_appointment: ✅ DTMF match → {matched_name} ({matched_id})")
        else:
            # If we heard *something*, clear the silence counter
            session_data[call_sid].pop("silence_cancel_doc", None)

            # Normalize and block common junk inputs
            selected_clean = _clean(selected_text)
            debug_print(f"cancel_appointment: 🗣️ Received doctor name → '{selected_clean}'")

            junk_inputs = {
                "", "yes", "no", "yeah", "nope", "ok", "okay", "hello", "hi", "hey",
                "good morning", "good afternoon", "good evening", "test", "i know", "what"
            }
            if (not selected_clean) or (selected_clean in junk_inputs) or (len(selected_clean) < 2):
                options = ". ".join([f"{doc} (press {k})" for k, doc in doctor_dtmf_map.items()])
                retry_prompt = (
                    f"I didn't recognize that as a doctor's name. Available doctors are: {options}. "
                    "Please say the name or press the number."
                )
                session_data[call_sid]["stage"] = "cancel_appointment"
                resp.append(make_gather(retry_prompt, hints=", ".join(doctor_names), num_digits=1))
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
                # Pick the one with max token overlap
                best = None
                best_overlap = -1
                for doc_id, friendly_name in partial_matches:
                    overlap = len(spoken_tokens & set(_clean(friendly_name).split()))
                    if overlap > best_overlap:
                        best = (doc_id, friendly_name)
                        best_overlap = overlap
                if best:
                    matched_id, matched_name = best
                    debug_print(f"cancel_appointment: ✅ Multiple matches; chose best overlap → {matched_name} ({matched_id})")

            # ------------------------------
            # 2) GPT fallback (if not matched yet)
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

            options = ". ".join([f"{doc} (press {k})" for k, doc in doctor_dtmf_map.items()])
            retry_prompt = (
                f"I didn't recognize that name. Available doctors are: {options}. "
                "Please say the name or press the number."
            )
            session_data[call_sid]["stage"] = "cancel_appointment"
            resp.append(make_gather(retry_prompt, hints=", ".join(doctor_names), num_digits=1))
            return str(resp)

        # ------------------------------
        # 4) Proceed with matched doctor → next stage: phone number
        # ------------------------------
        session_data[call_sid]["doctor_id"] = matched_id
        session_data[call_sid]["cancel"]["doctor"] = matched_name or googleid_dr_name_map.get(matched_id, "the doctor")
        session_data[call_sid]["stage"] = "cancel_appt_get_phone_number"

        resp.append(make_gather(
            "Thanks. What phone number did you use when booking the appointment?"
        ))
        return str(resp)




    elif stage == "cancel_appt_iterate":
        # ----------------------------------------------------------------------
        # 🗂️ Stage: cancel_appt_iterate
        #  • Lets caller cancel appointments by voice or DTMF.
        #  • Parallel slot checks + short timeouts for fast response.
        # ----------------------------------------------------------------------

        t_stage_start = _time_mod.perf_counter()
        debug_print("cancel_appt_iterate: 📍 Stage entered")

        cancel_ctx = session_data[call_sid].setdefault("cancel", {})
        doctor = (cancel_ctx.get("doctor") or "").strip()
        phone_e164 = (cancel_ctx.get("phone_e164") or "").replace("+", "").lstrip("0")
        dob = (cancel_ctx.get("dob") or "").strip()
        debug_print(f"cancel_appt_iterate: inputs → doctor='{doctor}', phone='{phone_e164}', dob='{dob}'")

        candidates = cancel_ctx.get("candidates")

        # ----------------------------------------------------------------------
        # 🧩 Build candidates (parallelized for large clinics)
        # ----------------------------------------------------------------------
        t_build_start = _time_mod.perf_counter()
        if not candidates:
            path = f"{DB_FOLDER}/{doctor.lower().replace(' ', '_')}.json"
            try:
                with open(path, "r") as f:
                    appts = json.load(f)
            except Exception as e:
                debug_print(f"cancel_appt_iterate: ⚠️ could not load {path} → {e}")
                appts = []

            def valid_appt(appt):
                appt_phone = (appt.get("phone") or "").replace("+", "").lstrip("0")
                appt_dob = (appt.get("dob") or "").strip()
                return appt_phone == phone_e164 and (not dob or appt_dob == dob)

            matching_appts = [a for a in appts if valid_appt(a)]
            debug_print(f"cancel_appt_iterate: potential matches → {len(matching_appts)}")

            candidates = []
            if matching_appts:
                from concurrent.futures import ThreadPoolExecutor

                def slot_check(appt):
                    try:
                        cal_id = None
                        for cid, friendly in googleid_dr_name_map.items():
                            if friendly.lower() == doctor.lower():
                                cal_id = cid
                                break
                        if not cal_id or not appt.get("utc_start"):
                            return None
                        exists = not is_time_slot_available(cal_id, appt["utc_start"], appt["utc_end"], creds)
                        return (cal_id, appt) if exists else None
                    except Exception as e:
                        debug_print(f"cancel_appt_iterate: ⚠️ slot check failed → {e}")
                        return None

                with ThreadPoolExecutor(max_workers=4) as ex:
                    for result in ex.map(slot_check, matching_appts):
                        if result:
                            cal_id, appt = result
                            candidates.append({
                                "doctor_name": doctor,
                                "calendar_id": cal_id,
                                "start_utc": appt.get("utc_start"),
                                "end_utc": appt.get("utc_end"),
                                "friendly": appt.get("friendly_local"),
                                "phone_e164": phone_e164,
                                "dob": dob,
                            })

            cancel_ctx["candidates"] = candidates
            cancel_ctx["iter_index"] = 0
            debug_print(f"cancel_appt_iterate: ✅ built {len(candidates)} candidate(s) "
                        f"in {_time_mod.perf_counter() - t_build_start:.3f}s")

            if not candidates:
                # No appointments found
                if session_data.get(call_sid, {}).get("reschedule_after_cancel"):
                    debug_print("cancel_appt_iterate: 🔁 no appts → switch to booking")
                    session_data[call_sid]["stage"] = "ask_time_date"
                    session_data[call_sid]["reschedule_after_cancel"] = False
                    resp.append(make_gather(
                        "I couldn’t find any appointments to cancel. Let’s make a new one. "
                        "Please say the date and time, for example, 'October 12th at 9 a.m.'"
                    ))
                    resp.redirect("/voice")
                    return str(resp)

                resp.say(gpt_speak("There are no upcoming appointments to cancel."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            resp.say(f"I found {len(candidates)} upcoming appointments.", VOICE)

        # ----------------------------------------------------------------------
        # 🧾 Handle input (voice or keypad)
        # ----------------------------------------------------------------------
        try:
            dtmf = (request.values.get("Digits") or "").strip()
        except Exception:
            dtmf = ""
        utter = (speech_result or "").strip().lower()
        utter = _re.sub(r"[^a-z0-9]+", "", utter)

        debug_print(f"cancel_appt_iterate: normalized utter='{utter}', dtmf='{dtmf}' "
                    f"(input parse took {_time_mod.perf_counter() - t_build_start:.3f}s)")

        YES = {"yes", "yeah", "yep", "confirm", "correct"}
        NO  = {"no", "nope", "next"}

        idx = int(cancel_ctx.get("iter_index", 0))
        total = len(cancel_ctx["candidates"])

        if idx >= total:
            resp.say("That was the last appointment. Goodbye.", VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)

        cand = cancel_ctx["candidates"][idx]

        # ----------------------------------------------------------------------
        # ✅ YES → cancel
        # ----------------------------------------------------------------------
        if utter in YES or dtmf == "1":
            debug_print(f"cancel_appt_iterate: ✅ YES user confirmed candidate #{idx+1}/{total}")
            cancel_ctx["matching_event"] = cand
            session_data[call_sid]["stage"] = "cancel_appt_confirm"
            handoff_t = _time_mod.perf_counter()
            resp.redirect("/voice")
            debug_print(f"cancel_appt_iterate: 🚀 handoff in "
                        f"{_time_mod.perf_counter() - handoff_t:.3f}s")
            debug_print(f"cancel_appt_iterate: ⏱️ total stage time "
                        f"{_time_mod.perf_counter() - t_stage_start:.3f}s")
            return str(resp)

        # ----------------------------------------------------------------------
        # ↪️ NO → next appointment
        # ----------------------------------------------------------------------
        if utter in NO or dtmf == "2":
            debug_print(f"cancel_appt_iterate: ↪️ NO user skipped candidate #{idx+1}/{total}")
            idx += 1
            cancel_ctx["iter_index"] = idx
            if idx >= total:
                resp.say("That was the last appointment. Goodbye.", VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)
            cand = cancel_ctx["candidates"][idx]

        # ----------------------------------------------------------------------
        # 🗣️ Present current candidate (short timeouts)
        # ----------------------------------------------------------------------
        debug_print(f"cancel_appt_iterate: 🗣️ presenting candidate #{idx+1}/{total}")
        say_line = (
            f"Appointment with {cand['doctor_name']} on {cand['friendly']}. "
            "Do you want to cancel this one? Say yes or no. Press 1 for yes, or 2 for no."
        )

        # ⚡ Optimized Gather — short timeouts = faster Twilio POST
        gather = make_gather(
            say_line,
            hints="yes no one two",
            input="speech dtmf",
            timeout=3,            # was 20
            speech_timeout="auto",# was 8
            finish_on_key="#",
            action="/voice",
            barge_in=True,
        )
        resp.append(gather)

        debug_print(f"cancel_appt_iterate: 🗣️ candidate presentation built in "
                    f"{_time_mod.perf_counter() - t_stage_start:.3f}s")
        debug_print(f"cancel_appt_iterate: ✅ total runtime {_time_mod.perf_counter() - t_stage_start:.3f}s")
        return str(resp)








    elif stage == "book_appt_confirm":
        # ----------------------------------------------------------------------
        # 💬 Stage: book_appt_confirm
        # Automatically confirms and books appointment (no user confirmation).
        # ----------------------------------------------------------------------
        t_stage_start = _time_mod.perf_counter()
        debug_print("book_appt_confirm: 📍 Stage entered")

        # ----------------------------------------------------------------------
        # 🧩 Doctor Info
        # ----------------------------------------------------------------------
        doctor_id = session_data[call_sid].get("doctor_id")
        if not doctor_id:
            debug_print("book_appt_confirm: ❌ missing doctor_id → choose_doctor")
            session_data[call_sid]["stage"] = "choose_doctor"
            resp.append(make_gather("Which doctor would you like to see?"))
            return str(resp)

        doctor_name = googleid_dr_name_map.get(doctor_id, "the doctor")

        # ----------------------------------------------------------------------
        # 🧩 Appointment Time Info
        # ----------------------------------------------------------------------
        appt = session_data[call_sid].get("appointment_time", {}) or {}
        appointment_start = appt.get("start")
        appointment_end   = appt.get("end")

        if not appointment_start:
            debug_print("book_appt_confirm: ❌ missing appointment_start")
            resp.say(gpt_speak("Appointment time is missing. Goodbye!"), VOICE)
            resp.hangup()
            return str(resp)

        # Convert UTC → Local (Clinic Timezone)
        tz_name = (globals().get("CLINIC_TZ") or "America/Chicago")
        try:
            tz = _pytz.timezone(tz_name)
        except Exception:
            tz = _pytz.timezone("America/Chicago")

        try:
            dt_utc   = datetime.fromisoformat(appointment_start.replace("Z", "+00:00"))
            dt_local = dt_utc.astimezone(tz)
            try:
                formatted_time = dt_local.strftime("%A, %B %-d at %-I:%M %p")
            except Exception:
                formatted_time = dt_local.strftime("%A, %B %d at %I:%M %p").replace(" 0", " ")
        except Exception as e:
            debug_print(f"book_appt_confirm: time format error → {e}")
            resp.say(gpt_speak("Sorry, we couldn't confirm the appointment time."), VOICE)
            resp.hangup()
            return str(resp)

        # Compute end time if missing
        if not appointment_end:
            try:
                dur = None
                for k in ("APPOINTMENT_DURATION_MINUTES", "SESSION_TIME", "SESSIUON_TIME"):
                    v = globals().get(k)
                    if v:
                        try:
                            dur = int(v)
                            break
                        except:
                            pass
                if dur not in (15, 30, 45, 60):
                    dur = 30
                end_dt = dt_utc + timedelta(minutes=dur)
                appointment_end = end_dt.astimezone(_pytz.UTC).isoformat()
            except Exception as e:
                debug_print(f"book_appt_confirm: ❌ failed computing end time → {e}")
                resp.say(gpt_speak("Sorry, we couldn't confirm the appointment time."), VOICE)
                resp.hangup()
                return str(resp)

        # ----------------------------------------------------------------------
        # 🧩 Customer Info
        # ----------------------------------------------------------------------
        customer         = session_data[call_sid].get("customer", {}) or {}
        customer_name    = (customer.get("name") or "").strip()
        first_name       = (customer.get("first_name") or "").strip()
        last_name        = (customer.get("last_name")  or "").strip()

        if not first_name and customer_name:
            parts = customer_name.split()
            first_name = parts[0]
            last_name  = " ".join(parts[1:]) if len(parts) > 1 else ""

        effective_name   = customer_name or " ".join([n for n in (first_name, last_name) if n]).strip()
        customer_address = (customer.get("address") or "").strip()
        customer_dob     = (customer.get("dob") or "").strip()

        phone_raw = (customer.get("phone_e164") or customer.get("phone") or "").strip()
        if phone_raw.startswith("+") and phone_raw[1:].replace(" ", "").isdigit():
            phone_e164 = "+" + phone_raw[1:].replace(" ", "")
        else:
            try:
                default_country = (session_data[call_sid].get("phone_country") or "US").upper()
                phone_e164 = normalize_phone_e164(phone_raw, default_country) or ""
                if not phone_e164:
                    alt = "EG" if default_country != "EG" else "US"
                    phone_e164 = normalize_phone_e164(phone_raw, alt) or ""
            except Exception:
                phone_e164 = ""

        if not phone_e164:
            session_data[call_sid]["stage"] = "collect_phone"
            resp.append(make_gather("Before we finalize your appointment, please provide your phone number."))
            return str(resp)

        if not customer_dob:
            session_data[call_sid]["stage"] = "collect_dob"
            resp.append(make_gather(
                "Before we confirm, please say your date of birth, for example, 'July 3 1990'."
            ))
            return str(resp)

        # ----------------------------------------------------------------------
        # 🧩 Availability Check
        # ----------------------------------------------------------------------
        try:
            slot_ok = is_time_slot_available(doctor_id, appointment_start, appointment_end, creds)
        except Exception as e:
            debug_print(f"book_appt_confirm: ⚠️ slot check failed → {e}")
            slot_ok = False

        if not slot_ok:
            debug_print("book_appt_confirm: ❌ Slot no longer available")
            session_data[call_sid]["stage"] = "ask_time_date"
            resp.append(make_gather("Sorry, that slot was just taken. Please choose another time."))
            return str(resp)

        # ----------------------------------------------------------------------
        # 🧩 Save/Upsert Customer
        # ----------------------------------------------------------------------
        try:
            init_db()
            if not phone_e164:
                digits = "".join(ch for ch in (phone_raw or "") if ch.isdigit())
                if len(digits) >= 8:
                    phone_e164 = f"+000{digits[-10:]}"
                    debug_print(f"book_appt_confirm: ⚠️ fallback phone_e164={phone_e164}")
            if not customer_dob:
                customer_dob = "unknown"

            inserted_ok = insert_customer(
                phone=phone_e164, dob=customer_dob,
                first_name=first_name, last_name=last_name, address=customer_address,
                cc_name=(customer.get("cc_name") or effective_name or ""),
                cc_number=(customer.get("cc_number") or ""),
                cc_exp=(customer.get("cc_exp") or ""),
                cc_cvv=(customer.get("cc_cvv") or "")
            )
            debug_print(f"book_appt_confirm: ✅ insert_customer executed (return={inserted_ok})")
        except Exception as e:
            debug_print(f"book_appt_confirm: insert_customer failed → {e}")

        # ----------------------------------------------------------------------
        # 🧩 Google Calendar Event Creation
        # ----------------------------------------------------------------------
        google_event_id = session_data[call_sid].get("google_event_id", "")
        if google_event_id:
            debug_print(f"book_appt_confirm: ℹ️ event already created → id={google_event_id}")
        else:
            try:
                service = build("calendar", "v3", credentials=creds)
                event_body = {
                    "summary": f"Appointment: {doctor_name}",
                    "description": f"Clinic appointment for {effective_name or 'patient'}.",
                    "start": {"dateTime": appointment_start, "timeZone": "UTC"},
                    "end":   {"dateTime": appointment_end,   "timeZone": "UTC"},
                    "transparency": "opaque",
                    "extendedProperties": {
                        "private": {
                            "patient_name": effective_name,
                            "phone_e164": phone_e164,
                            "dob": customer_dob,
                            "call_sid": call_sid,
                        }
                    },
                }
                ev = service.events().insert(calendarId=doctor_id, body=event_body, sendUpdates="none").execute()
                google_event_id = ev.get("id")
                session_data[call_sid]["google_event_id"] = google_event_id
                debug_print(f"book_appt_confirm: ✅ Google event created id={google_event_id}")
            except Exception as e:
                debug_print(f"book_appt_confirm: ❌ Google insert failed → {e}")
                session_data[call_sid]["stage"] = "ask_time_date"
                resp.append(make_gather("Sorry, I couldn't confirm that slot. Please say another time."))
                return str(resp)

        # ----------------------------------------------------------------------
        # 🧩 Local JSON Persistence
        # ----------------------------------------------------------------------
        try:
            local_date_str = dt_local.strftime("%Y-%m-%d")
            try:
                local_time_disp = dt_local.strftime("%-I:%M %p")
            except Exception:
                local_time_disp = dt_local.strftime("%I:%M %p").lstrip("0")

            persist = confirm_appointment_for_dr_name(
                doctor_name=doctor_name,
                phone=phone_e164,
                utc_start=appointment_start,
                utc_end=appointment_end,
                calendar_id=doctor_id,
                name=effective_name,
                dob=customer_dob,
                address=customer_address,
                event_id=google_event_id,
                friendly_local=formatted_time,
                local_date=local_date_str,
                local_time_display=local_time_disp,
            )
            debug_print(f"book_appt_confirm: 🗂️ local persist → {persist}")
        except Exception as e:
            debug_print(f"book_appt_confirm: ⚠️ local persist failed → {e}")

        # ----------------------------------------------------------------------
        # 🧩 Voice + SMS Confirmation
        # ----------------------------------------------------------------------
        msg = f"Your appointment with {doctor_name} has been booked"
        if formatted_time:
            msg += f" on {formatted_time}"
        msg += ". We look forward to seeing you. Goodbye!"
        resp.say(gpt_speak(msg), VOICE)

        try:
            sms = f"Hi {(effective_name or 'there')}, your appointment with {doctor_name} is confirmed"
            if formatted_time:
                sms += f" on {formatted_time}"
            sms += ". Thank you for choosing Epic Therapist Clinic."
            _ = client.messages.create(body=sms, from_=TWILIO_PHONE_NUMBER, to=phone_e164)
            debug_print(f"book_appt_confirm: 📩 SMS sent to {phone_e164}")
        except Exception as e:
            debug_print(f"book_appt_confirm: ⚠️ SMS failed → {e}")

        resp.hangup()
        session_data.pop(call_sid, None)
        debug_print(f"book_appt_confirm: ✅ total runtime {_time_mod.perf_counter() - t_stage_start:.3f}s")
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
        #  date 10/02/25
        # ----------------------------------------------------------------------

    
    elif stage == "cancel_appt_confirm":
        # ----------------------------------------------------------------------
        # 🧩 Stage: cancel_appt_confirm (asynchronous deletion, no confirmation)
        # ----------------------------------------------------------------------
       
        t0 = _time_mod.perf_counter()
        debug_print("cancel_appt_confirm: 📍 Stage entered")

        cancel_ctx = session_data[call_sid].get("cancel", {})
        cand = cancel_ctx.get("matching_event")
        reschedule_flag = session_data.get(call_sid, {}).get("reschedule_after_cancel", False)

        if not cand:
            debug_print("cancel_appt_confirm: ⚠️ No candidate found in session.")
            resp.say(gpt_speak("Sorry, I couldn’t find that appointment to cancel."), VOICE)
            if reschedule_flag:
                session_data[call_sid]["stage"] = "ask_time_date"
                session_data[call_sid]["reschedule_after_cancel"] = False
                resp.append(make_gather(
                    "Please say the new date and time for your appointment, for example, 'October 12th at 9 a.m.'"
                ))
                resp.redirect("/voice")
                return str(resp)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)

        # ----------------------------------------------------------------------
        # Extract parameters
        # ----------------------------------------------------------------------
        calendar_id = cand.get("calendar_id")
        start_utc   = cand.get("start_utc")
        end_utc     = cand.get("end_utc")
        doctor_name = cand.get("doctor_name")
        friendly    = cand.get("friendly")

        debug_print(f"cancel_appt_confirm: 🔎 Checking slot {start_utc} → {end_utc} on {calendar_id}")

        # ----------------------------------------------------------------------
        # Slot check
        # ----------------------------------------------------------------------
        try:
            slot_free = is_time_slot_available(calendar_id, start_utc, end_utc, creds)
        except Exception as e:
            debug_print(f"cancel_appt_confirm: ⚠️ availability check failed → {e}")
            slot_free = True

        # ----------------------------------------------------------------------
        # ✅ Case 1: slot occupied → proceed with async deletion
        # ----------------------------------------------------------------------
        if not slot_free:
            debug_print("cancel_appt_confirm: ✅ Slot occupied → launching async deletion thread")

            def _async_delete():
                t_del_start = _time_mod.perf_counter()
                try:
                    service = build("calendar", "v3", credentials=creds)
                    events = service.events().list(
                        calendarId=calendar_id,
                        timeMin=start_utc,
                        timeMax=end_utc,
                        singleEvents=True
                    ).execute()

                    for ev in events.get("items", []):
                        try:
                            service.events().delete(calendarId=calendar_id, eventId=ev["id"]).execute()
                            debug_print(f"cancel_appt_confirm.async: 🗑️ deleted Google event {ev['id']}")
                        except Exception as e2:
                            debug_print(f"cancel_appt_confirm.async: ⚠️ failed to delete event {ev.get('id')} → {e2}")

                    # ---- Delete from local JSON ----
                    path = f"{DB_FOLDER}/{doctor_name.lower().replace(' ', '_')}.json"
                    try:
                        with open(path, "r") as f:
                            appts = json.load(f)
                        appts = [a for a in appts if not (
                            a.get("utc_start") == start_utc and a.get("utc_end") == end_utc
                        )]
                        with open(path, "w") as f:
                            json.dump(appts, f, indent=2)
                        debug_print("cancel_appt_confirm.async: 🗑️ deleted from doctor JSON")
                    except Exception as e:
                        debug_print(f"cancel_appt_confirm.async: ⚠️ JSON cleanup failed → {e}")

                except Exception as e:
                    debug_print(f"cancel_appt_confirm.async: ❌ async delete error → {e}")
                finally:
                    debug_print(f"cancel_appt_confirm.async: 🕒 total delete time "
                                f"{_time_mod.perf_counter() - t_del_start:.3f}s")

            # 🧵 Launch deletion thread (non-blocking)
            threading.Thread(target=_async_delete, daemon=True).start()

            # Immediate polite response (no wait)
            resp.say(gpt_speak(
                f"Your appointment with {doctor_name} on {friendly} has been cancelled."
            ), VOICE)

        # ----------------------------------------------------------------------
        # ❌ Case 2: slot already free → nothing to cancel
        # ----------------------------------------------------------------------
        else:
            debug_print("cancel_appt_confirm: ❌ Slot already free → nothing to cancel")
            resp.say(gpt_speak("Sorry, I couldn’t find that appointment to cancel."), VOICE)

        # ----------------------------------------------------------------------
        # 🔁 Reschedule flow continuation
        # ----------------------------------------------------------------------
        if reschedule_flag:
            debug_print("cancel_appt_confirm: 🔄 Detected reschedule flow → proceed to ask_time_date")
            session_data[call_sid]["stage"] = "ask_time_date"
            session_data[call_sid]["reschedule_after_cancel"] = False

            # Reuse phone/DOB if available
            cust = session_data[call_sid].setdefault("customer", {})
            cancel_info = session_data[call_sid].get("cancel", {})
            if cancel_info.get("phone_e164"):
                cust["phone_e164"] = cancel_info["phone_e164"]
            if cancel_info.get("dob"):
                cust["dob"] = cancel_info["dob"]

            resp.append(make_gather(
                "Your previous appointment has been cancelled. "
                "Please say the new date and time for your appointment, for example, 'October 12th at 9 a.m.'"
            ))
            resp.redirect("/voice")
            debug_print(f"cancel_appt_confirm: ⏱️ total stage time {_time_mod.perf_counter() - t0:.3f}s")
            return str(resp)

        # ----------------------------------------------------------------------
        # ✅ End normal flow
        # ----------------------------------------------------------------------
        resp.hangup()
        session_data.pop(call_sid, None)
        debug_print(f"cancel_appt_confirm: ✅ total runtime {_time_mod.perf_counter() - t0:.3f}s")
        return str(resp)




   



if __name__ == "__main__":
    
    # ------------------------
    # 🔁 Call this once at startup
    # ------------------------

    load_doctor_appointments()


    app.run(host="0.0.0.0", port=5000)
