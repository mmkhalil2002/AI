# update  10/10/25 time_saved  3:13 pm cancel is tested
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

    Notes on timing:
      - timeout controls DTMF first-digit and between-digit wait.
      - speech_timeout controls how long STT waits for silence; can be "auto" or seconds.
    """
    # Normalize speechTimeout for Twilio (int or "auto")
    _speech_timeout = speech_timeout
    if isinstance(_speech_timeout, str) and _speech_timeout.isdigit():
        _speech_timeout = int(_speech_timeout)

    _num_digits = num_digits if (isinstance(num_digits, int) and num_digits > 0) else None
    _action = _append_stage_to_action(action, next_stage)

    try:
        g = Gather(
            input=input,
            action=_action,
            method=method,
            timeout=int(timeout),
            speechTimeout=_speech_timeout,
            finishOnKey=finish_on_key,
            numDigits=_num_digits,
            hints=hints,
            language=language,
            bargeIn=barge_in,
        )
        g.say(gpt_speak(prompt), voice=VOICE)
        return g
    except Exception as e:
        # Soft fallback: at least speak the prompt so the flow doesn't crash
        try:
            g = Gather(input=input, action=_action, method=method)
            g.say(gpt_speak(prompt), voice=VOICE)
            return g
        except Exception:
            debug_print(f"make_gather: failed to build Gather → {e}")
            return None


def make_gather_dtmf(
    prompt: str,
    *,
    num_digits: Optional[int] = None,
    next_stage: Optional[str] = None,               # keep symmetry with make_gather
    finish_on_key: str = "#",
    action: Optional[str] = "/voice",
    method: str = "POST",
):
    """
    DTMF-only helper using the same ENV-driven defaults.
    """
    return make_gather(
        prompt,
        next_stage=next_stage,
        input="dtmf",
        num_digits=num_digits,
        timeout=PAUSE_BETWEEN_DIGITS,
        speech_timeout="auto",  # irrelevant for pure DTMF
        finish_on_key=finish_on_key,
        action=action,
        method=method,
    )


# Prefer speech+DTMF first; escalate to DTMF-only when caller struggles.
def prompt_for_value(prompt_text: str, *, dtmf_only: bool = False, max_digits: int = None):
    # DTMF-only path (e.g., after an invalid attempt or for CVV/ZIP)
    if dtmf_only and "make_gather_dtmf" in globals() and callable(globals()["make_gather_dtmf"]):
        # Keep '#" as the terminator so callers can press pound when done.
        return make_gather_dtmf(
            prompt_text,
            max_digits=max_digits,     # leave as None to require '#'
            finish_on_key="#",
        )

    # Speech + DTMF (default) — do NOT pass input="speech dtmf" unless your helper supports it
    if "make_gather" in globals() and callable(globals()["make_gather"]):
        return make_gather(
            prompt_text,
            hints="zero one two three four five six seven eight nine double triple",
        )

    # Ultra-compact fallback (prevents crashes if helpers aren’t available)
    try:
        #from twilio.twiml.voice_response import Gather
        g = Gather(input=("dtmf" if dtmf_only else "speech dtmf"))
        g.say(prompt_text)
        return g
    except Exception:
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
    Boundary rule: touching at start/end is OK (no +/- 1s padding).
    """
    #from googleapiclient.discovery import build
    #from dateutil.parser import isoparse
    #import pytz as _pytz

    # Parse to aware datetimes (UTC)
    def _as_utc_dt(s: str):
        s2 = s.replace("Z", "+00:00")
        dt = isoparse(s2)
        return dt if dt.tzinfo else dt.replace(tzinfo=_pytz.UTC)

    start_dt = _as_utc_dt(start_iso).astimezone(_pytz.UTC)
    end_dt   = _as_utc_dt(end_iso).astimezone(_pytz.UTC)

    if end_dt <= start_dt:
        return False  # invalid window

    service = build("calendar", "v3", credentials=creds)

    # ---- FreeBusy (no padding)
    fb = service.freebusy().query(body={
        "timeMin": start_dt.isoformat().replace("+00:00", "Z"),
        "timeMax": end_dt.isoformat().replace("+00:00", "Z"),
        "items": [{"id": calendar_id}],
    }).execute()

    busy_blocks = fb.get("calendars", {}).get(calendar_id, {}).get("busy", [])

    # Overlap check for half-open intervals
    def _overlaps(a_start, a_end, b_start, b_end):
        # True only if there is real overlap inside (touching edges is free)
        return (a_start < b_end) and (b_start < a_end)

    for b in busy_blocks:
        bs = _as_utc_dt(b["start"])
        be = _as_utc_dt(b["end"])
        if _overlaps(start_dt, end_dt, bs, be):
            return False

    # ---- Events list (no padding)
    ev = service.events().list(
        calendarId=calendar_id,
        timeMin=start_dt.isoformat().replace("+00:00", "Z"),
        timeMax=end_dt.isoformat().replace("+00:00", "Z"),
        singleEvents=True,
        maxResults=1,
        orderBy="startTime",
    ).execute()

    items = ev.get("items", [])
    if items:
        # Double-check overlap (some recurring instances can edge-touch)
        for it in items:
            # Get event start/end in UTC
            def _evt_to_dt(evt_key):
                val = it.get("start", {}).get(evt_key) or it.get("end", {}).get(evt_key)
                return _as_utc_dt(val) if val else None

            estart_raw = it.get("start", {}).get("dateTime") or it.get("start", {}).get("date")
            eend_raw   = it.get("end",   {}).get("dateTime") or it.get("end",   {}).get("date")
            if not (estart_raw and eend_raw):
                # All-day or malformed — conservatively block if there’s any real overlap
                return False
            estart = _as_utc_dt(estart_raw)
            eend   = _as_utc_dt(eend_raw)
            if _overlaps(start_dt, end_dt, estart, eend):
                return False

    return True






def get_next_available_slots(
    calendar_id: str,
    creds,
    *,
    from_start_iso: str,
    duration_minutes: int = None,   # falls back to APPOINTMENT_DURATION_MINUTES/SESSION_TIME
    limit: int = 3,
    tz_name: str = None,            # falls back to CLINIC_TZ or America/Chicago
    work_hours=None,                # e.g., ((8,12),(13,17)); falls back to WORKING_HOURS_START/END
    slot_step_minutes: int = None,  # default = duration
    search_days: int = None         # default = SEARCH_DAYS or 14
) -> list:
    """
    Return up to `limit` free slots strictly in the future, aligned to clinic policy.

    STRONG RULES (clinic schedule wins):
      - Only suggest slots on WORKING_DAYS.
      - Slot must fit FULLY inside provided work_hours windows (and W_START/END).
      - Exclude LUNCH window entirely (no overlap).
      - Google availability is necessary but NOT sufficient.

    If from_start_iso is past, start from NOW (rounded to grid, STRICTLY AFTER now).
    If from_start_iso is far in the future (beyond the search horizon), clamp to NOW.
    """
    # ---- local imports (3.8-safe) ----
    # (Assumes _pytz, isoparse, datetime/timedelta/dtime are imported at module scope)

    def _dbg(msg: str) -> None:
        try:
            debug_print(msg)
        except Exception:
            pass

    _dbg(f"get_next_available_slots: ▶️ cal={calendar_id} from={from_start_iso} limit={limit}")

    # ---- slot checker ----
    slot_check = globals().get("is_time_slot_available")
    if not callable(slot_check):
        _dbg("get_next_available_slots: ❌ no slot checker callable found")
        return []

    # Small, configurable tolerance to avoid 1-second boundary “phantom busy” at :00 / :30.
    # Example: FreeBusy returns a 1s busy exactly at 17:00:00Z; some implementations pad
    # by ±1s and mark 16:30–17:00 as "busy". By shrinking the check window *inside* the
    # requested slot by a couple seconds, we avoid false negatives while still staying
    # entirely within the slot.
    EDGE_LENIENCY_SECONDS = int(globals().get("EDGE_LENIENCY_SECONDS", 2))

    # ---- defaults from globals ----
    if duration_minutes is None:
        duration_minutes = int(globals().get(
            "APPOINTMENT_DURATION_MINUTES",
            globals().get("SESSION_TIME", globals().get("SESSIUON_TIME", 30))
        ))
    if duration_minutes not in (15, 30, 45, 60):
        duration_minutes = 30

    if slot_step_minutes is None:
        slot_step_minutes = duration_minutes

    if tz_name is None:
        tz_name = (globals().get("CLINIC_TZ") or globals().get("LOCAL_TZ") or "America/Chicago")
    try:
        tz_local = _pytz.timezone(tz_name)
    except Exception:
        tz_local = _pytz.timezone("America/Chicago")

    WSTART = int(globals().get("WORKING_HOURS_START", globals().get("WORKIN_HOURS_START", 8)))
    WEND   = int(globals().get("WORKING_HOURS_END", 17))
    if not work_hours:
        work_hours = ((WSTART, WEND),)

    try:
        wd_src = globals().get("WORKING_DAYS", {0,1,2,3,4})
        WORKING_DAYS = set(int(x) for x in (wd_src if isinstance(wd_src, (list,set,tuple)) else [0,1,2,3,4]))
    except Exception:
        WORKING_DAYS = {0,1,2,3,4}

    # ---- lunch window (optional) --------------------------------------------
    def _as_time(val, default_h=None, default_m=0):
        if val is None:
            return None if default_h is None else dtime(default_h, default_m)
        if isinstance(val, dtime):
            return val
        s = str(val).strip()
        if not s:
            return None if default_h is None else dtime(default_h, default_m)
        if ":" in s:
            hh, mm = (s.split(":", 1) + ["0"])[:2]
        else:
            hh, mm = s, "0"
        try:
            return dtime(max(0, min(23, int(hh))), max(0, min(59, int(mm))))
        except Exception:
            return None if default_h is None else dtime(default_h, default_m)

    LUNCH_START = _as_time(globals().get("LUNCH_BREAK_START"))
    LUNCH_END   = _as_time(globals().get("LUNCH_BREAK_END"))

    if search_days is None:
        search_days = int(globals().get("SEARCH_DAYS", 14))

    # ---- utilities -----------------------------------------------------------
    def _align_up_to_window_grid(dt_local, minutes, window_start_local, *, now_local):
        """
        Align 'dt_local' to the window's grid anchored at 'window_start_local'.
        - Always anchor to the window's own start (so 8:00/8:30/etc. is respected).
        - If TODAY and aligned <= now, push to the smallest tick strictly after 'now'.
        ⭐ Grace at opening: allow a tiny delay at the very first tick so we don't miss 8:00
          just because we're a few seconds late.
        """
        GRACE_SECONDS = int(globals().get("WINDOW_START_GRACE_SECONDS", 180))  # 3 minutes

        dt_local = dt_local.replace(second=0, microsecond=0)
        anchor   = window_start_local.replace(second=0, microsecond=0)

        diff_min = int((dt_local - anchor).total_seconds() // 60)
        if diff_min <= 0:
            aligned = anchor
        else:
            rem = diff_min % minutes
            aligned = dt_local if rem == 0 else (dt_local + timedelta(minutes=(minutes - rem)))

        if aligned.date() == now_local.date() and aligned <= now_local:
            if aligned == anchor:
                late_seconds = (now_local - aligned).total_seconds()
                if 0 < late_seconds <= GRACE_SECONDS:
                    return aligned  # keep the opening tick
            diff_now = int((now_local - anchor).total_seconds() // 60)
            steps = (diff_now // minutes) + 1
            aligned = anchor + timedelta(minutes=steps * minutes)

        return aligned

    def _friendly(dt_local, now_local):
        # Include year if different from current year to avoid "August confusion"
        if dt_local.year != now_local.year:
            try:
                return dt_local.strftime("%A, %B %-d, %Y at %-I:%M %p")
            except Exception:
                return dt_local.strftime("%A, %B %d, %Y at %I:%M %p").replace(" 0", " ")
        else:
            try:
                return dt_local.strftime("%A, %B %-d at %-I:%M %p")
            except Exception:
                return dt_local.strftime("%A, %B %d at %I:%M %p").replace(" 0", " ")

    def _inside_hours(start_loc):
        end_loc = start_loc + timedelta(minutes=duration_minutes)
        return (dtime(WSTART, 0) <= start_loc.time() and end_loc.time() <= dtime(WEND, 0))

    def _in_lunch(start_loc):
        if not (LUNCH_START and LUNCH_END):
            return False
        end_loc = start_loc + timedelta(minutes=duration_minutes)
        return (start_loc.time() < LUNCH_END and end_loc.time() > LUNCH_START)

    # Safe call that supports either slot_check signature:
    #   (calendar_id, start_iso, end_iso, creds)  OR  (calendar_id, creds, start_iso, end_iso)
    def _check_range(start_iso: str, end_iso: str) -> bool:
        try:
            return bool(slot_check(calendar_id, start_iso, end_iso, creds))
        except TypeError:
            return bool(slot_check(calendar_id, creds, start_iso, end_iso))

    # ---- seed start & clamp to NOW-window -----------------------------------
    now_loc = datetime.now(tz_local)

    # Parse requested; if naive, treat as LOCAL
    req_local = None
    try:
        parsed = isoparse((from_start_iso or "").strip())
        req_local = (tz_local.localize(parsed) if parsed.tzinfo is None else parsed.astimezone(tz_local))
    except Exception:
        req_local = None

    # Define the allowed search window: NOW → NOW + search_days
    search_window_start = now_loc
    search_window_end   = now_loc + timedelta(days=search_days)

    # Choose base start:
    #  - if req_local is within [now, now+search_days], use req_local
    #  - else, clamp to NOW
    if req_local and (search_window_start <= req_local <= search_window_end):
        base_local = req_local
    else:
        base_local = search_window_start

    # Do NOT pre-round here; we align per-window so we never skip the opening tick.
    cur_local = base_local

    _dbg(f"get_next_available_slots: ⏱️ now_local={now_loc.isoformat()} start_cursor={cur_local.isoformat()} (window_end={search_window_end.isoformat()})")

    # ---- main scan (clinic policy first, then Google) -----------------------
    results, seen = [], set()

    while cur_local < search_window_end and len(results) < limit:
        # enforce WORKING_DAYS
        if cur_local.weekday() not in WORKING_DAYS:
            _dbg(f"get_next_available_slots: 📅 non-working day {cur_local.weekday()} → next working day")
            d = (cur_local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            while d.weekday() not in WORKING_DAYS:
                d = d + timedelta(days=1)
            cur_local = d.replace(hour=int(work_hours[0][0]), minute=0, second=0, microsecond=0)
            continue

        # Build windows (tz-aware) for that day
        day = cur_local.date()
        windows = []
        for ws, we in work_hours:
            ws, we = int(ws), int(we)
            if ws >= we:
                continue
            wstart = tz_local.localize(datetime(day.year, day.month, day.day, ws, 0, 0))
            wend   = tz_local.localize(datetime(day.year, day.month, day.day, we, 0, 0))
            windows.append((wstart, wend))

        progressed = False
        for wstart, wend in windows:
            if cur_local >= wend:
                continue
            if cur_local < wstart:
                cur_local = wstart

            # 🔧 KEY: align to the window grid anchored at wstart so we probe 8:00, 8:30, 9:00, ...
            cur_local = _align_up_to_window_grid(cur_local, slot_step_minutes, wstart, now_local=now_loc)

            while cur_local + timedelta(minutes=duration_minutes) <= wend and len(results) < limit:
                # strictly inside working hours
                if not _inside_hours(cur_local):
                    break

                # exclude lunch overlap (no partial overlap)
                if _in_lunch(cur_local):
                    if LUNCH_END:
                        cur_local = tz_local.localize(datetime.combine(cur_local.date(), LUNCH_END))
                        # realign after lunch to the same window grid
                        cur_local = _align_up_to_window_grid(cur_local, slot_step_minutes, wstart, now_local=now_loc)
                        continue

                # after-now enforcement for TODAY
                if cur_local.date() == now_loc.date() and cur_local <= now_loc:
                    cur_local = _align_up_to_window_grid(now_loc, slot_step_minutes, wstart, now_local=now_loc)
                    if cur_local >= wend:
                        break

                # Google checks (UTC Z)
                start_iso = cur_local.astimezone(_pytz.UTC).isoformat().replace("+00:00", "Z")
                end_iso   = (cur_local + timedelta(minutes=duration_minutes)).astimezone(_pytz.UTC).isoformat().replace("+00:00", "Z")

                # Primary check
                ok = False
                try:
                    ok = _check_range(start_iso, end_iso)
                except Exception as e:
                    _dbg(f"get_next_available_slots: slot_check error → {e}")
                    ok = False

                # ⭐ Edge leniency: if the primary window looks busy, retry a slightly
                #    *shrunken* window inside the slot to ignore 1s boundary artifacts.
                if (not ok) and EDGE_LENIENCY_SECONDS > 0:
                    try:
                        inset_start = (cur_local + timedelta(seconds=EDGE_LENIENCY_SECONDS)).astimezone(_pytz.UTC).isoformat().replace("+00:00", "Z")
                        inset_end   = (cur_local + timedelta(minutes=duration_minutes, seconds=-EDGE_LENIENCY_SECONDS)).astimezone(_pytz.UTC).isoformat().replace("+00:00", "Z")
                        # Only retry if we still have a valid positive window
                        if inset_start < inset_end:
                            ok = _check_range(inset_start, inset_end)
                            if ok:
                                _dbg("get_next_available_slots: ✅ edge-leniency unblocked boundary artifact")
                    except Exception as e:
                        _dbg(f"get_next_available_slots: edge-leniency error → {e}")

                if ok and start_iso not in seen:
                    seen.add(start_iso)
                    results.append({
                        "start": start_iso,
                        "end": end_iso,
                        "friendly": _friendly(cur_local, now_loc),
                        "tz": tz_name,
                    })
                    _dbg(f"get_next_available_slots: ✅ add {results[-1]['friendly']}")
                    if len(results) >= limit:
                        break

                # step to next tick on this window's grid (ensures we test 11:30, 12:00, 12:30, 1:00, 1:30, …)
                cur_local = cur_local + timedelta(minutes=slot_step_minutes)

            progressed = True
            if len(results) >= limit:
                break

        if len(results) >= limit:
            break

        # advance to next working day’s first window
        d = (tz_local.localize(datetime(cur_local.year, cur_local.month, cur_local.day)) + timedelta(days=1)
             if progressed else (cur_local + timedelta(days=1)))
        while d.weekday() not in WORKING_DAYS:
            d = d + timedelta(days=1)
        cur_local = d.replace(hour=int(work_hours[0][0]), minute=0, second=0, microsecond=0)

    _dbg(f"get_next_available_slots: ✅ suggestions={len(results)}")
    return results





















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

    🆕 E.164-only migration (no legacy formatting):
      - If a record already has a valid E.164 'phone_e164', re-key to (phone_e164|dob).
      - If the existing key's left side is valid E.164, adopt it into 'phone_e164'.
      - Ensure 'created_at' and 'last_seen_at' exist.
      - Never derive from legacy 10-digit or trunked numbers; no digit munging.

    This function deliberately avoids any non-E.164 normalization. Records without
    a valid E.164 will be left as-is (preserved under their original keys).
    """
    os.makedirs(DB_FOLDER, exist_ok=True)
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2, ensure_ascii=False)
        return

    # Validate existing file is a JSON object; if not, reset to {}
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("customers.json must be a JSON object")
    except Exception:
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2, ensure_ascii=False)
        return

    # ---------- 🧰 migration (safe / idempotent; E.164-only) ----------
    changed = False
    migrated = 0
    ensured_ts = 0
    adopted_from_key = 0
    skipped_non_e164 = 0

    def _is_e164(s: str) -> bool:
        """Strict E.164 check: '+' followed by 6..15 digits (no spaces)."""
        s = (s or "").strip()
        return bool(_re.fullmatch(r"\+\d{6,15}", s))

    def _e164_or_empty(s: str) -> str:
        s = (s or "").strip().replace(" ", "")
        return s if _is_e164(s) else ""

    try:
        new_data: dict = {}
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for old_key, rec in data.items():
            if not isinstance(rec, dict):
                # Skip non-dict entries (preserve as-is)
                new_data[old_key] = rec
                continue

            # Ensure timestamps
            if not rec.get("created_at") or not rec.get("last_seen_at"):
                rec.setdefault("created_at", now)
                rec.setdefault("last_seen_at", now)
                ensured_ts += 1
                changed = True

            # Normalize DOB field to a single line (do NOT parse)
            rec["dob"] = _oneline(rec.get("dob", ""))

            # Ensure phone_e164 ONLY if it is already E.164 or can be read as E.164 from the key
            phone_e164 = _e164_or_empty(rec.get("phone_e164", ""))
            if not phone_e164 and "|" in old_key:
                # If the legacy key *already* uses E.164 on the left, adopt it
                left = old_key.split("|", 1)[0].strip()
                left_e164 = _e164_or_empty(left)
                if left_e164:
                    rec["phone_e164"] = left_e164
                    phone_e164 = left_e164
                    adopted_from_key += 1
                    changed = True

            # Decide final key:
            #   - If we have valid E.164, re-key to (phone_e164|dob)
            #   - Otherwise, keep the old key (no legacy conversion attempted)
            final_key = old_key
            if phone_e164:
                try:
                    final_key = _key(phone_e164, rec.get("dob", ""))
                except Exception:
                    final_key = old_key  # defensive

            if final_key != old_key:
                # Migrate if target key not occupied
                if final_key not in new_data:
                    new_data[final_key] = rec
                    migrated += 1
                    changed = True
                else:
                    # Collision: prefer existing; update its last_seen_at
                    try:
                        new_data[final_key]["last_seen_at"] = now
                    except Exception:
                        pass
            else:
                # Keep record under its original key
                new_data[old_key] = rec
                if not phone_e164:
                    skipped_non_e164 += 1

        if changed:
            # Atomic write to avoid corruption
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
        # Migration errors should never take down the app; keep existing data
        debug_print(f"init_db: ⚠️ migration skipped due to error: {e}")
        # Do not rewrite file in this case; leave it as we loaded it.
        return


#   remove phone10 and make dependent on e146

# ---------- Sanitizers / formatters ----------
def _oneline(s: str) -> str:
    """Compact whitespace/newlines to a single line."""
    return _re.sub(r"\s+", " ", (s or "").strip())


def _normalize_phone(s: str) -> str:
    """
    E.164-only sanitizer.

    Returns the input as E.164 (e.g., '+12025550123', '+201234567890') **only** if it
    already matches strict E.164 ('+' followed by 6–15 digits). Otherwise returns ''.

    NOTE:
      - No legacy normalization (no 10-digit US, no trunked EG, no digit stripping).
      - If you need to transform national numbers into E.164, call your dedicated
        normalize_phone_e164(raw, country) helper elsewhere; this function intentionally
        does not attempt any conversion.
    """
    s = (s or "").strip().replace(" ", "")
    return s if _re.fullmatch(r"\+\d{6,15}", s) else ""


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
          1: Phone (E.164)
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
            'phone_e164' (required for E.164-only display),
            'dob', 'first_name', 'last_name', 'address',
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
    phone        = rec.get("phone_e164") or "—"     # E.164 ONLY
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
    lines: List[str] = [
        _block_title(new),             # 0
        f"Phone: {phone}",             # 1  (E.164 only)
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

    # assert len(lines) == 12, "Rendered block must contain exactly 12 lines"
    return lines

# end finishing remove phone10 and make it strict on e164


# ---------- File parsing helpers ----------

# =============================================================================
# Block parsers / accessors (3.8-safe typing)
# =============================================================================
####   rewmove dependency on phone10
# BEFORE:
# def _iter_blocks(lines: list[str]):
# AFTER (3.8-safe):
##  reove dependency on phone 10 until save_customer

def _iter_blocks(lines: List[str]) -> Iterator[Tuple[int, int, List[str]]]:
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
    country: str = COUNTRY,
    phone: str = None,     # ← backward-compatible alias (if some callers still pass phone=)
) -> bool:
    """
    Return True if a customer (phone|dob) exists in customers.json, else False.

    Signature updated to match program-wide usage:
      - Primary param: phone_number
      - Back-compat alias: phone
      - Other behavior unchanged.

    Lookup (E.164 only):
      1) Normalize input as E.164 using default_country (falls back US↔EG).
      2) Normalize DOB to ISO (YYYY-MM-DD) when possible.
      3) Check key = _key(phone_e164, dob_iso).
    """
    # --- log inputs -----------------------------------------------------------
    try:
        debug_print(
            f"customer_search: ▶️ inputs phone_number='{phone_number}' "
            f"phone(alias)='{phone}' dob='{dob}' default_country='{default_country}'"
        )
    except Exception:
        pass

    init_db()

    # Prefer phone_number; fall back to phone alias
    raw = (phone_number if phone_number is not None else phone) or ""
    raw = raw.strip()
    dob_iso = (dob or "").strip()

    # ---- normalize phone to E.164 -------------------------------------------
    phone_e164 = ""
    if raw.startswith("+") and raw[1:].replace(" ", "").isdigit():
        phone_e164 = "+" + raw[1:].replace(" ", "")
    else:
        try:
            phone_e164 = normalize_phone_e164(raw, default_country) or ""
        except Exception:
            phone_e164 = ""
        if not phone_e164:
            try:
                alt = "EG" if str(default_country).upper() != "EG" else "US"
                phone_e164 = normalize_phone_e164(raw, alt) or ""
            except Exception:
                phone_e164 = ""

    if not phone_e164:
        debug_print(f"customer_search: ❌ invalid phone '{raw}' (no E.164)")
        return False

    # ---- normalize DOB to ISO if provided -----------------------------------
    try:
        if dob_iso:
            # If global _re is present and dob isn't already YYYY-MM-DD, try MM/DD/YYYY or MM-DD-YYYY
            if ('_re' in globals()) and (globals()['_re'].fullmatch(r"\d{4}-\d{2}-\d{2}", dob_iso) is None):
                m = globals()['_re'].match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", dob_iso)
                if m:
                    mm, dd, yyyy = m.groups()
                    dob_iso = f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}"
                else:
                    # Light normalization like 2025/08/07 → 2025-08-07
                    dob_iso = dob_iso.replace("/", "-")
    except Exception:
        # If regex not available or parsing fails, keep original dob_iso
        pass

    # ---- lookup --------------------------------------------------------------
    data = _load_customers()
    key_e164 = _key(phone_e164, dob_iso)
    exists = key_e164 in data

    debug_print(f"customer_search: phone_e164={phone_e164} dob_iso={dob_iso or '∅'} key={key_e164} → exists={exists}")
    return exists

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
      • If (phone|dob) exists: update 'last_seen_at' only; return False.
      • If new: create record with 'created_at' and 'last_seen_at'; return True.
    Never duplicates because the map key is unique.

    PHONE FORMAT:
      • Stores and keys by E.164 (e.g., +12025550123, +2011xxxxxxxx)
      • Also writes 'phone' (display) = E.164 for compatibility with renderers

    SECURITY (logging only):
      • Per your request, this version logs FULL values (no masking) for cc_number and cc_cvv.
        This is NOT recommended for production systems subject to PCI-DSS.

    DEPENDENCIES:
      • Requires: init_db(), _load_customers(), _save_customers(), _key(),
                  _oneline(), normalize_phone_e164(), debug_print,
                  and global COUNTRY
    """
    # Ensure DB exists and is a JSON object
    init_db()

    # --- normalize inputs ----------------------------------------------------
    # E.164 canonical phone (strict: must normalize)
    phone_e164 = normalize_phone_e164(phone, COUNTRY)
    if not phone_e164:
        raise ValueError("insert_customer: invalid phone (must normalize to E.164)")

    dob_iso = (dob or "").strip()  # upstream stages should already normalize to YYYY-MM-DD

    # Compact one-line fields (no newlines/tabs)
    first_name  = _oneline(first_name)
    last_name   = _oneline(last_name)
    address     = _oneline(address)
    cc_name     = _oneline(cc_name)
    cc_number   = _oneline(cc_number)
    cc_exp      = _oneline(cc_exp)   # MM/YY expected by your collector
    cc_cvv      = _oneline(cc_cvv)

    # --- load current map ----------------------------------------------------
    data = _load_customers()
    key = _key(phone_e164, dob_iso)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- existing customer: bump last_seen_at --------------------------------
    if key in data:
        data[key]["last_seen_at"] = now
        _save_customers(data)
        debug_print(f"insert_customer: ℹ️ exists; updated last_seen_at for {key}")
        return False

    # --- new record ----------------------------------------------------------
    rec: Dict[str, Any] = {
        # Canonical phone for storage + compatibility display field
        "phone_e164": phone_e164,   # canonical
        "phone":      phone_e164,   # for renderers that print "Phone: ..."

        "dob": dob_iso,

        "first_name": first_name,
        "last_name":  last_name,
        "address":    address,

        # Store CC fields (unmasked)
        "cc_name":   cc_name,
        "cc_number": cc_number,   # WARNING: stored unmasked per your request
        "cc_exp":    cc_exp,      # MM/YY
        "cc_cvv":    cc_cvv,      # WARNING: stored unmasked per your request

        "created_at":  now,
        "last_seen_at": now,
    }

    data[key] = rec
    _save_customers(data)

    # --- logging (UNMASKED per request) --------------------------------------
    debug_print(
        "insert_customer: ✅ Added new customer\n"
        f"Phone: {rec['phone_e164']}\n"
        f"DOB: {rec['dob'] or '∅'}\n"
        f"First Name: {rec['first_name']}\n"
        f"Last Name: {rec['last_name']}\n"
        f"Address: {rec['address']}\n"
        f"CC Name: {rec.get('cc_name','')}\n"
        f"CC Number: {rec.get('cc_number','')}\n"
        f"CC Exp: {rec.get('cc_exp','')}\n"
        f"CC CVV: {rec.get('cc_cvv','')}\n"
        f"Created At: {rec['created_at']}\n"
        f"Last Seen At: {rec['last_seen_at']}"
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

    # NEW: Seed per-call country once, using caller number if present; fallback to global COUNTRY
    session_data.setdefault(call_sid, {})
    if "country" not in session_data[call_sid]:
        from_number = (request.values.get("From") or "").strip()
        derived = COUNTRY
        if from_number.startswith("+20"):
            derived = "EG"
        elif from_number.startswith("+1"):
            derived = "US"
        session_data[call_sid]["country"] = derived
    # (optional) keep the raw caller E.164 for later use
    from_number = (request.values.get("From") or "").strip()
    if from_number.startswith("+"):
        session_data[call_sid]["from_e164"] = from_number

    print(f"📢 voice :speech_result: {speech_result}")

    # Determine the current interaction stage (default to "intro" if not previously set)
    stage = session_data.get(call_sid, {}).get("stage", "intro")




    # ----------------------------------------------------------------------
    # 🔇 CENTRAL SILENCE GUARD
    # If we didn't hear *anything* (no speech, no DTMF), re-prompt with
    # stage-appropriate text. We skip stages that already have their own
    # robust silence handling (e.g., collect_cc).
    # ----------------------------------------------------------------------
    def _silence_prompt_for_stage(st: str) -> Tuple[str, str]:
        """Return (prompt, hints) best suited for the current stage."""
        # Default: generic prompt, no hints
        hints = ""
        if st in ("intro", "intent"):
            # ✨ Updated to advertise both voice and keypad (DTMF 1..5)
            hints = "book,cancel,change,reschedule,update,update card,voicemail,leave message"
            return (
                "I didn’t hear anything. Say 'book appointment' or press 1. "
                "Say 'cancel appointment' or press 2. "
                "Say 'change appointment' or press 3. "
                "Say 'update credit card' or press 4. "
                "Say 'leave voicemail' or press 5.",
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
            return ("Please say your birth date, for example 'July third 1990'. Or type 2 digits for Month 2 digits for Day 4 digits for year then press pound.", hints)
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
            return ("Please say your birth date, for example 'July third nineteen fifty six'. Or type 2 digits for month 2 digits for day and 4 digis for year then press pound.", hints)
        if st == "voicemail":
            return ("Please leave your name, phone number, and message after the beep.", hints)

        # Fallback generic
        return ("Sorry, I didn’t hear anything. Please say that again.", hints)


    # ----------------------------------------------------------------------
    # Silence handling guard
    # ----------------------------------------------------------------------
    # Only run the guard outside of the very first greeting (intro),
    # and skip stages that handle silence internally.
    skip_silence = (
        "intro",
        "collect_cc",
        "book_appt_confirm",
        # 🚫 NEW: skip cancel flow stages too
        "cancel_appt_iterate",
        "cancel_appt_get_time_date",
        "cancel_appt_confirm",
    )

    if stage not in skip_silence:
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
                gather = make_gather(prompt, hints=hints, num_digits=1) if hints else make_gather(prompt, num_digits=1)
            except Exception:
                gather = make_gather("Sorry, I didn’t hear anything. Please try again.", num_digits=1)
            resp.append(gather)
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
        # 📍 Booking flow: the caller has just been asked to name a doctor.
        # Accepts either speech or single-digit DTMF input (from doctor_dtmf_map).
        # ----------------------------------------------------------------------

        if "retry_booking" not in session_data[call_sid]:
            session_data[call_sid]["retry_booking"] = 0

        # Safe punctuation constant (no string import)
        _PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

        # Pull DTMF and speech
        try:
            dtmf_digits = (request.values.get("Digits") or "").strip()
        except Exception:
            dtmf_digits = ""
        spoken_text = (speech_result or "").lower().strip()
        spoken_clean = spoken_text.translate(str.maketrans('', '', _PUNCT)).strip()

        print(f"📻 booking :speech_result: {spoken_clean} DTMF='{dtmf_digits}'")

        matched_id = None

        # ------------------------------------------------------------------
        # 🔢 Path 1: Direct keypad digit lookup
        # ------------------------------------------------------------------
        if dtmf_digits and "doctor_dtmf_map" in session_data[call_sid]:
            doctor_map = session_data[call_sid]["doctor_dtmf_map"]
            chosen_name = doctor_map.get(dtmf_digits)
            if chosen_name:
                # Find doctor_id by name match
                for doc_id, friendly in googleid_dr_name_map.items():
                    if friendly.lower() == chosen_name.lower():
                        matched_id = doc_id
                        print(f"✅ DTMF matched doctor: {friendly}")
                        break

        # ------------------------------------------------------------------
        # 🎙️ Path 2: Speech-based name lookup
        # ------------------------------------------------------------------
        if matched_id is None:
            # Skip obvious junk inputs
            junk_inputs = {
                "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
                "yo", "test", "1", "yes", "no", "i know", "huh", "what", "okay", "ok",
                "bye", "goodbye", ""
            }
            if not spoken_clean or spoken_clean in junk_inputs or len(spoken_clean) < 3:
                print(f"⏩ Skipping junk doctor input: '{spoken_clean}' — re-prompting without retry")
                doctor_list_str = ", ".join(googleid_dr_name_map.values())
                gather = make_gather("Please say the name of the doctor you'd like to book with.", 
                                    hints=doctor_list_str, num_digits=1)
                resp.append(gather)
                return str(resp)

            # 🔍 Partial token-based match
            partial_matches = []
            spoken_tokens = set(spoken_clean.split())
            for doc_id, friendly in googleid_dr_name_map.items():
                friendly_clean = friendly.lower().translate(str.maketrans('', '', _PUNCT)).strip()
                friendly_tokens = set(friendly_clean.split())
                if (spoken_clean in friendly_clean or
                    friendly_clean in spoken_clean or
                    spoken_tokens & friendly_tokens):
                    partial_matches.append((doc_id, friendly))

            if len(partial_matches) == 1:
                matched_id = partial_matches[0][0]
                print(f"✅ Partial match with: {partial_matches[0][1]}")
            elif len(partial_matches) > 1:
                print(f"🔍 Multiple matches: {[name for _, name in partial_matches]}")
                matched_id = partial_matches[0][0]

        # ------------------------------------------------------------------
        # ❌ Retry if no match
        # ------------------------------------------------------------------
        if matched_id is None:
            debug_print(f"❌ No doctor match for: '{spoken_clean or dtmf_digits}'")
            session_data[call_sid]["retry_booking"] += 1
            retries = session_data[call_sid]["retry_booking"]

            if retries >= 3:
                resp.say(gpt_speak(
                    "I'm sorry, I still couldn't match that name with any doctor in our clinic. "
                    "Please call us again later."
                ), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            doctor_list_str = ", ".join(googleid_dr_name_map.values())
            retry_prompt = (
                f"I couldn't match that to a doctor. Available doctors are: {doctor_list_str}. "
                "Please say the doctor name or press the number."
            )
            gather = make_gather(retry_prompt, hints=doctor_list_str, num_digits=1)
            resp.append(gather)
            return str(resp)

        # ------------------------------------------------------------------
        # ✅ Success → Store doctor, move to phone collection
        # ------------------------------------------------------------------
        session_data[call_sid]["doctor_id"] = matched_id
        session_data[call_sid]["stage"] = "collect_phone"

        friendly_name = googleid_dr_name_map[matched_id]
        phone_prompt = (
            f"Great, we'll book with {friendly_name}. "
            "Please say or enter your phone number including area code."
        )

        gather = make_gather(phone_prompt, num_digits=10)
        resp.append(gather)
        return str(resp)




    elif stage == "collect_phone":
        # ----------------------------------------------------------------------
        # 📞 Stage: collect_phone  (Local input → E.164; NO country-code prompt)
        #
        # Goal:
        #   - Caller provides *local/national* number only (e.g., 4694633276).
        #   - We normalize to **E.164** using normalize_phone_e164(raw, country),
        #     where `country` is inferred once per call:
        #         request.values["FromCountry"] or global COUNTRY (default "US").
        #   - Store at:
        #       session_data[call_sid]["customer"]["phone_e164"]  (primary)
        #       session_data[call_sid]["customer"]["phone"]       (mirror E.164)
        #       session_data[call_sid]["phone_e164"]              (top-level convenience)
        #   - If we were sent here from another stage, return via "return_stage".
        #
        # Silent-mode handling:
        #   - If no SpeechResult and no Digits → re-prompt up to 3 times.
        #
        # 🔒 Note:
        #   - We do NOT ask the caller to include a country code. We derive it
        #     from the known/default `country` server-side.
        # ----------------------------------------------------------------------
        debug_print("collect_phone: 📍 Stage entered")

       
        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("customer", {})

        # Infer country once per call (prefer Twilio signal if present); no user prompt about it.
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

        # 🔇 Silent mode: nothing heard → re-prompt with cap 3
        if not (speech_text or dtmf_digits):
            tries = session_data[call_sid].get("silence_collect_phone", 0) + 1
            session_data[call_sid]["silence_collect_phone"] = tries
            debug_print(f"collect_phone: 🤐 no input heard (tries={tries})")

            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Short, clear prompt; avoid over-explaining
            prompt = "Say or type your 10-digit phone number, then press #."
            gather = make_gather(prompt, hints="zero one two three four five six seven eight nine double triple")
            resp.append(gather)
            return str(resp)

        # We heard something → clear stage silence counter
        session_data[call_sid].pop("silence_collect_phone", None)

        # --- helper: speech→digits (for logging only; E.164 normalization uses digits we heard) ---
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

        # Prefer DTMF for the actual value we normalize; speech is kept for logging visibility
        if dtmf_digits:
            # Remove anything that isn't 0-9
            raw_digits = _re.sub(r"\D", "", dtmf_digits)
        else:
            raw_digits = _re.sub(r"\D", "", _spoken_to_digits(speech_text))

        debug_print(f"collect_phone: raw_digits='{raw_digits}'")

        # Build E.164 **without** asking caller for country code: we use server-side country.
        country = session_data[call_sid].get("phone_country", (COUNTRY or "US")).upper()
        try:
            phone_e164 = normalize_phone_e164(raw_digits, country)  # expects '+<cc><nsn>' or ''
        except NameError:
            # If helper is missing, do a minimal US-only fallback from local digits
            debug_print("collect_phone: ⚠️ normalize_phone_e164 not defined; using minimal US fallback")
            phone_e164 = ""
            if country == "US":
                d = raw_digits
                # Accept 11-digit NANP starting with '1'
                if len(d) == 11 and d.startswith("1"):
                    d = d[1:]
                if len(d) == 10:
                    phone_e164 = f"+1{d}"

        # Validate E.164
        if not phone_e164:
            session_data[call_sid]["retry_phone"] = session_data[call_sid].get("retry_phone", 0) + 1
            r = session_data[call_sid]["retry_phone"]
            debug_print(f"collect_phone: ❌ invalid local phone for country={country} (digits='{raw_digits}') retry={r}")

            if r >= 3:
                resp.say(gpt_speak("Sorry, I couldn't capture your phone number. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Short, consistent re-prompt
            prompt = "Say or type your 10-digit phone number, then press #."
            gather = make_gather(prompt, hints="zero one two three four five six seven eight nine double triple")
            resp.append(gather)
            return str(resp)

        # ✅ Save E.164 (primary) and mirror to 'phone' for compatibility
        session_data[call_sid]["customer"]["phone_e164"] = phone_e164
        session_data[call_sid]["customer"]["phone"] = phone_e164
        session_data[call_sid]["phone_e164"] = phone_e164
        session_data[call_sid]["retry_phone"] = 0
        debug_print(f"collect_phone: ✅ saved phone_e164={phone_e164}")

        # If we were sent here by another stage, jump back there now
        return_stage = session_data[call_sid].pop("return_stage", None)
        if return_stage:
            session_data[call_sid]["stage"] = return_stage
            debug_print(f"collect_phone: ➡️ returning to {return_stage}")
            resp.redirect("/voice")
            return str(resp)

        # Flow-based next step
        if "cancel" in session_data[call_sid]:
            session_data[call_sid]["stage"] = "cancel_appt_get_date_time"
            gather = make_gather(
                "Thanks. Now tell me the date and time of the appointment you want to cancel. "
                "For example, August 15th at 5 AM."
            )
            resp.append(gather)
            return str(resp)

        # Default: ask DOB (short, clear)
        session_data[call_sid]["stage"] = "collect_dob"
        gather = make_gather(
            "Thanks. What’s your date of birth? You can say it, or enter 2 digits for mounth 2 digits for day and 4 digits for year then press #."
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
    # ----------------------------------------------------------------------

    elif stage == "collect_dob":
        debug_print("collect_dob: 📍 Stage entered")

        # Short, clear prompts (no “MMDDYYYY” wording anywhere)
        PROMPT_DOB_SHORT = (
            "Say your birth date, for example, 'July 3 1956'. "
            "Or enter two digits for month, two for day, and four for year, then press #. Example: 07 03 1956#."
        )
        PROMPT_REPEAT_FULL = (
            "I didn’t catch your full birth date. Please say the complete date, for example, 'July 3 1956'. "
            "You can also enter two digits for month, two for day, and four for year, then press #. Example: 07 03 1956#."
        )
        PROMPT_FINAL_DTMF = (
            "Please enter two digits for month, two for day, and four for year, then press #. Example: 07 03 1956#."
        )

        # Ensure session buckets exist
        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("customer", {})

        # Pull inputs
        try:
            dtmf_digits = (request.values.get("Digits") or "").strip()
        except Exception:
            dtmf_digits = ""
        speech_text = (speech_result or "").strip()
        debug_print(f"collect_dob: 🎙️ speech_text='{speech_text}', 🔢 dtmf_digits='{dtmf_digits}'")

        # -----------------------------
        # 🔇 Silence handling (cap=3)
        # -----------------------------
        if not dtmf_digits and not speech_text:
            tries = session_data[call_sid].get("silence_dob", 0) + 1
            session_data[call_sid]["silence_dob"] = tries
            debug_print(f"collect_dob: 🤐 no input; silence retries={tries}")

            if tries >= 3:
                resp.say(gpt_speak("Sorry, I couldn’t get your birth date. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Re-prompt (speech+DTMF)
            g = make_gather(PROMPT_DOB_SHORT, input="speech dtmf")
            resp.append(g)
            try:
                resp.redirect(url_for("voice"))
            except Exception:
                resp.redirect("/voice")
            return str(resp)

        # We heard something → clear silence counter
        session_data[call_sid].pop("silence_dob", None)

        # -----------------------------------------------------
        # 1) KEYPAD path (preferred if provided)
        #    Accepts "07 03 1956#", "07031956#", "07-03-1956#", etc.
        # -----------------------------------------------------
        dob_date = None
        if dtmf_digits:
            d = _re.sub(r"\D", "", dtmf_digits)  # keep only digits
            if len(d) == 8:
                try:
                    mm = int(d[0:2]); dd = int(d[2:4]); yyyy = int(d[4:8])
                    dob_date = date(yyyy, mm, dd)
                except Exception:
                    dob_date = None
            else:
                dob_date = None

            if dob_date is None:
                # Invalid keypad DOB → ask for the FULL birth date again.
                r = session_data[call_sid].get("retry_dob", 0) + 1
                session_data[call_sid]["retry_dob"] = r
                debug_print(f"collect_dob: ❌ invalid keypad DOB '{dtmf_digits}' retry={r}")

                g = make_gather(PROMPT_FINAL_DTMF if r >= 3 else PROMPT_REPEAT_FULL,
                                input=("dtmf" if r >= 3 else "speech dtmf"))
                resp.append(g)
                try:
                    resp.redirect(url_for("voice"))
                except Exception:
                    resp.redirect("/voice")
                return str(resp)

        # -----------------------------------------------------
        # 2) SPEECH path (when no valid keypad DOB)
        #    - Handle slow speech with ordinals and punctuation
        #    - Require that a 4-digit year is present
        #    - If ANY part is unclear → ask for FULL date again
        # -----------------------------------------------------
        if dob_date is None:
            t = speech_text

            # Make STT punctuation harmless; collapse spaces
            t = _re.sub(r"[.,;:]+$", "", t)         # trim trailing punctuation
            t = _re.sub(r"[,\.;:]", " ", t)         # inner punctuation → space
            t = _re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", t, flags=_re.IGNORECASE)  # 3rd→3, 21st→21
            t = _re.sub(r"\s+", " ", t).strip()

            # If only a 4-digit year was heard → missing parts → ask for FULL date again
            only_digits = _re.sub(r"\D", "", t) or ""
            if _re.fullmatch(r"\d{4}", only_digits):
                r = session_data[call_sid].get("retry_dob", 0) + 1
                session_data[call_sid]["retry_dob"] = r
                debug_print(f"collect_dob: ❌ only year heard; retry_dob={r}")

                g = make_gather(PROMPT_FINAL_DTMF if r >= 3 else PROMPT_REPEAT_FULL,
                                input=("dtmf" if r >= 3 else "speech dtmf"))
                resp.append(g)
                try:
                    resp.redirect(url_for("voice"))
                except Exception:
                    resp.redirect("/voice")
                return str(resp)

            # Require an explicit 4-digit year in the spoken text (avoid parser’s default year)
            said_year = bool(_re.search(r"\b\d{4}\b", t))
            if not said_year:
                r = session_data[call_sid].get("retry_dob", 0) + 1
                session_data[call_sid]["retry_dob"] = r
                debug_print(f"collect_dob: ❌ year missing in speech; retry_dob={r}")

                g = make_gather(PROMPT_FINAL_DTMF if r >= 3 else PROMPT_REPEAT_FULL,
                                input=("dtmf" if r >= 3 else "speech dtmf"))
                resp.append(g)
                try:
                    resp.redirect(url_for("voice"))
                except Exception:
                    resp.redirect("/voice")
                return str(resp)

            # Try to parse month/day/year from speech
            try:
                today = _date_local.today()
                default_base = datetime(today.year, today.month, today.day, 9, 0, 0)
                parsed = _dtparse(t, default=default_base, dayfirst=False, fuzzy=True)

                # Build pure date (ignore time if any)
                dob_date = date(parsed.year, parsed.month, parsed.day)
            except Exception as e:
                r = session_data[call_sid].get("retry_dob", 0) + 1
                session_data[call_sid]["retry_dob"] = r
                debug_print(f"collect_dob: ❌ speech parse failed; retry_dob={r} reason={e}")

                g = make_gather(PROMPT_FINAL_DTMF if r >= 3 else PROMPT_REPEAT_FULL,
                                input=("dtmf" if r >= 3 else "speech dtmf"))
                resp.append(g)
                try:
                    resp.redirect(url_for("voice"))
                except Exception:
                    resp.redirect("/voice")
                return str(resp)

        # -----------------------------------------------------
        # 3) Validate DOB is reasonable (1900..today)
        # -----------------------------------------------------
        try:
            today = _date_local.today()
            min_date = date(1900, 1, 1)
            if not (min_date <= dob_date <= today):
                raise ValueError(f"out of range: {dob_date.isoformat()}")
        except Exception as e:
            r = session_data[call_sid].get("retry_dob", 0) + 1
            session_data[call_sid]["retry_dob"] = r
            debug_print(f"collect_dob: ⚠️ Validation error → {e} (retry={r})")

            g = make_gather(PROMPT_FINAL_DTMF if r >= 3 else PROMPT_REPEAT_FULL,
                            input=("dtmf" if r >= 3 else "speech dtmf"))
            resp.append(g)
            try:
                resp.redirect(url_for("voice"))
            except Exception:
                resp.redirect("/voice")
            return str(resp)

        # -----------------------------------------------------
        # 4) Success → store YYYY-MM-DD and advance
        # -----------------------------------------------------
        iso_dob = dob_date.strftime("%Y-%m-%d")
        session_data[call_sid]["customer"]["dob"] = iso_dob
        session_data[call_sid].pop("retry_dob", None)
        debug_print(f"collect_dob: ✅ Stored DOB → {iso_dob}")

        # Next stage: ask for appointment date/time
        session_data[call_sid]["stage"] = "ask_time_date"
        g = make_gather("Thanks. Please say the appointment date and time, for example, 'September 12 at 10 AM'.")
        resp.append(g)
        try:
            resp.redirect(url_for("voice"))
        except Exception:
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
        # Prompts
        # ------------------------------------------------------------------
        TIME_PROMPT_SHORT = "That doesn't sound like a valid date or time. Please say it again, for example, 'September 12 at 10 AM'."
        PROMPT_NEED_BOTH  = "Please say the date and the time, for example, 'September 12 at 10 AM'."
        PROMPT_NEED_DATE  = "I didn't hear the date. Please include it, for example, 'September 12 at 10 AM'."
        PROMPT_NEED_TIME  = "I didn't hear the time. Please include it, for example, 'September 12 at 10 AM'."

        # ------------------------------------------------------------------
        # Ensure session and doctor (per-doctor calendar)
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
        # Handle silence
        # ------------------------------------------------------------------
        raw = (speech_result or "").strip()
        if not raw:
            tries = session_data[call_sid].get("silence_time", 0) + 1
            session_data[call_sid]["silence_time"] = tries
            debug_print(f"ask_time_date: 🤐 silence (tries={tries})")
            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)
            resp.append(make_gather("Please say the date and time, for example, 'September 12 at 10 AM'."))
            resp.redirect("/voice")
            return str(resp)
        session_data[call_sid].pop("silence_time", None)

        # ------------------------------------------------------------------
        # Inline helpers (no imports)
        # ------------------------------------------------------------------
        def _has_time_token(s: str) -> bool:
            s = (s or "").lower()
            return (
                ("am" in s) or ("pm" in s) or (":" in s)
                or ("o'clock" in s) or ("oclock" in s)
                or (_re.search(r"\b\d{1,2}\s*(am|pm)\b", s) is not None)
                or (_re.search(r"\b\d{3,4}\b", s) is not None)   # 930, 1030
                or ("noon" in s) or ("midnight" in s)
            )

        def _has_date_token(s: str) -> bool:
            s = (s or "").lower()
            months = ("january","february","march","april","may","june","july",
                    "august","september","october","november","december",
                    "jan","feb","mar","apr","may","jun","jul","aug","sep","sept","oct","nov","dec")
            if any(m in s for m in months): return True
            if "/" in s or "-" in s: return True
            weekdays = ("monday","tuesday","wednesday","thursday","friday","saturday","sunday",
                        "mon","tue","tues","wed","thu","thur","thurs","fri","sat","sun")
            if any(w in s for w in weekdays): return True
            if _re.search(r"\b\d{1,2}\b", s): return True
            return False

        def _extract_day_time(s: str) -> tuple:
            if not s: return ("", "")
            s = _re.sub(r"\b(a\s*\.?\s*m\.?)\b", "am", s, flags=_re.IGNORECASE)
            s = _re.sub(r"\b(p\s*\.?\s*m\.?)\b", "pm", s, flags=_re.IGNORECASE)
            s = _re.sub(r"\bat\s*[.,]?\s+", " at ", s, flags=_re.IGNORECASE)
            s = _re.sub(r"[!?]+\s*$", "", s)
            s = _re.sub(r"[;,]+", " ", s)
            s = _re.sub(r"\.\s+(?=\d)", " ", s)
            s = _re.sub(r"\s+", " ", s).strip()
            s = _re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", s, flags=_re.IGNORECASE)

            s_low = s.lower()
            s_low = s_low.replace(" at noon", " at 12 pm").replace(" noon", " 12 pm")
            s_low = s_low.replace(" at midnight", " at 12 am").replace(" midnight", " 12 am")

            if " at " in s_low:
                day, timep = s_low.split(" at ", 1)
                return (day.strip().rstrip(","), timep.strip())

            m = _re.search(r"\b(\d{1,2}:\d{2}\s*(am|pm)?|\d{1,2}\s*(am|pm))\b", s_low)
            if m:
                timep = m.group(1)
                day = s_low[:m.start()].strip().rstrip(",")
                return (day, timep)

            m2 = _re.search(r"\b(\d{3,4})\b", s_low)
            if m2:
                t = m2.group(1)
                timep = (f"{int(t[0]):d}:{t[1:]}" if len(t) == 3 else f"{int(t[:-2]):d}:{t[-2:]}")
                day = s_low[:m2.start()].strip().rstrip(",")
                return (day, timep)

            return ("", "")

        def _build_slot(day_str: str, time_str: str) -> tuple:
            tz_name = (globals().get("CLINIC_TZ") or "America/Chicago")
            try:
                tz_local = _pytz.timezone(tz_name)
            except Exception:
                tz_local = _pytz.timezone("America/Chicago")

            dur = globals().get("APPOINTMENT_DURATION_MINUTES") or 30
            try: dur = int(dur)
            except Exception: dur = 30
            if dur not in (15,30,45,60): dur = 30

            d = (day_str or "").strip()
            t = (time_str or "").strip()
            if not d or not t:
                raise ValueError("missing date or time")

            t = _re.sub(r"\s*(am|pm)\b", r" \1", t)
            t = t.replace(" o'clock", "")
            combined = f"{d} at {t}"

            today = _date_local.today()
            default_base = datetime(today.year, today.month, today.day, 9, 0, 0)
            parsed = _dtparse(combined, default=default_base, dayfirst=False, fuzzy=True)

            if parsed.tzinfo is None:
                parsed = tz_local.localize(parsed)
            else:
                parsed = parsed.astimezone(tz_local)

            if not _re.search(r"\b\d{4}\b", combined):
                parsed = parsed.replace(year=today.year)

            start_local = parsed
            end_local   = start_local + timedelta(minutes=dur)
            start_utc = start_local.astimezone(_pytz.UTC).isoformat().replace("+00:00", "Z")
            end_utc   = end_local.astimezone(_pytz.UTC).isoformat().replace("+00:00", "Z")
            return (start_utc, end_utc)

        # ------------------------------------------------------------------
        # Extract (day, time)
        # ------------------------------------------------------------------
        day_part, time_part = _extract_day_time(raw)
        debug_print(f"ask_time_date: 📆 Extracted → Day: {day_part or '(none)'}, Time: {time_part or '(none)'}")

        need_date = not _has_date_token(day_part)
        need_time = not _has_time_token(time_part)
        if need_date or need_time:
            if need_date and need_time: prompt = PROMPT_NEED_BOTH
            elif need_date:             prompt = PROMPT_NEED_DATE
            else:                       prompt = PROMPT_NEED_TIME
            session_data[call_sid]["retry_time"] = session_data[call_sid].get("retry_time", 0) + 1
            if session_data[call_sid]["retry_time"] >= 3:
                resp.say(gpt_speak("Sorry, I still couldn't understand the date and time. Please try again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)
            resp.append(make_gather(prompt))
            return str(resp)

        # ------------------------------------------------------------------
        # Build UTC slot
        # ------------------------------------------------------------------
        try:
            appointment_start, appointment_end = _build_slot(day_part, time_part)
            session_data[call_sid]["retry_time"] = 0
            debug_print(f"ask_time_date: ⏰ Built slot → Start: {appointment_start}, End: {appointment_end}")
        except Exception as e:
            debug_print(f"ask_time_date: ❌ build slot failed → {e}")
            session_data[call_sid]["retry_time"] = session_data[call_sid].get("retry_time", 0) + 1
            if session_data[call_sid]["retry_time"] >= 3:
                resp.say(gpt_speak("Sorry, I couldn’t understand the time you mentioned. Please try again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)
            resp.append(make_gather(TIME_PROMPT_SHORT))
            return str(resp)

        # ------------------------------------------------------------------
        # Past-time guard
        # ------------------------------------------------------------------
        try:
            now_utc = datetime.utcnow().replace(tzinfo=_pytz.UTC)
            end_dt  = datetime.fromisoformat(appointment_end.replace("Z", "+00:00")).astimezone(_pytz.UTC)
            if end_dt <= now_utc:
                debug_print("ask_time_date: 🕒 requested time is in the past → suggest alternatives")
                try:
                    alts = get_next_available_slots(calendar_id, creds, from_start_iso=appointment_end, limit=3) or []
                except Exception as e:
                    debug_print(f"ask_time_date: ⚠️ get_next_available_slots error → {e}")
                    alts = []
                if alts:
                    options = " or ".join([a.get("friendly","") for a in alts if a.get("friendly")])
                    prompt = f"That time has already passed. Would you like {options}?" if options else "That time has already passed. Please say another date and time."
                else:
                    prompt = "That time has already passed. Please say another date and time."
                resp.append(make_gather(prompt))
                return str(resp)
        except Exception as e:
            debug_print(f"ask_time_date: ⚠️ past-time guard error → {e}")

        # ------------------------------------------------------------------
        # Availability check
        # ------------------------------------------------------------------
        debug_print(f"ask_time_date: 👨‍⚕️ Checking calendar → {calendar_id}")
        try:
            slot_available = is_time_slot_available(calendar_id, appointment_start, appointment_end, creds)
        except Exception as e:
            debug_print(f"ask_time_date: ⚠️ Availability check error → {e}")
            slot_available = False

        if not slot_available:
            debug_print("ask_time_date: ❌ Slot not available → suggesting alternatives")
            try:
                alts = get_next_available_slots(calendar_id, creds, from_start_iso=appointment_end, limit=3) or []
            except Exception as e:
                debug_print(f"ask_time_date: ⚠️ get_next_available_slots error → {e}")
                alts = []
            if alts:
                options = " or ".join([a.get("friendly","") for a in alts if a.get("friendly")])
                prompt = f"That time is not available. Would you like {options}?" if options else "That time is not available. Please say another date and time."
            else:
                prompt = "That time is not available. Please say another date and time."
            resp.append(make_gather(prompt))
            return str(resp)

        # ------------------------------------------------------------------
        # ✅ Slot free → save; then customer lookup
        # ------------------------------------------------------------------
        session_data[call_sid]["appointment_time"] = {"start": appointment_start, "end": appointment_end}

        cust = session_data[call_sid].setdefault("customer", {})
        phone_e164 = cust.get("phone_e164") or session_data[call_sid].get("phone_e164")
        dob        = cust.get("dob")        or session_data[call_sid].get("dob")

        if not phone_e164 or not dob:
            missing = "phone" if not phone_e164 else "dob"
            debug_print(f"ask_time_date: 🧩 missing {missing} → collect it before customer_search")
            if not phone_e164:
                session_data[call_sid]["stage"] = "collect_phone"
                prompt = "Please say your 10-digit phone number."
            else:
                session_data[call_sid]["stage"] = "collect_dob"
                prompt = "Please say your date of birth, for example, 'July third 1990'."
            resp.append(make_gather(prompt))
            resp.redirect("/voice")
            return str(resp)

        try:
            found = customer_search(phone_number=phone_e164, dob=dob, country="US")
            debug_print(f"ask_time_date: 🔎 customer_search(phone={phone_e164}, dob={dob}, country=US) → {found}")
        except Exception as e:
            debug_print(f"ask_time_date: ⚠️ customer_search error → {e}")
            found = False

        if found:
            debug_print("ask_time_date: 📋 Customer on file — skip name collection")
            session_data[call_sid]["stage"] = "book_appt_confirm"
        else:
            debug_print("ask_time_date: 🆕 New customer — go to collect_first_name")
            session_data[call_sid]["stage"] = "collect_first_name"

        resp.redirect("/voice")
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
                #from flask import url_for
                resp.redirect(url_for("voice"))
            except Exception:
                resp.redirect("/voice")
            return str(resp)

        # We heard something → clear silence counter
        session_data[call_sid].pop("silence_first_name", None)

        # 🧽 Clean & normalize (remove punctuation, compress spaces; ignore fillers)
        #import string  # ensure available; you already import at top, but safe to re-import
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
                #from flask import url_for
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
            #from flask import url_for
            resp.redirect(url_for("voice"))
        except Exception:
            resp.redirect("/voice")
        return str(resp)
    



    

    elif stage == "collect_last_name":
        # ----------------------------------------------------------------------
        # 🎯 Goal:
        #   - Capture LAST name via speech.
        #   - Handle silence separately (up to 3 silent retries).
        #   - Clean & lightly validate (letters/spaces/'/-; allow multi-token like "van dyke").
        #   - Require at least 2 letters to avoid short names like "K."
        #   - Store into session_data[call_sid]["customer"]["last_name"].
        #   - Advance → collect_address.
        # ----------------------------------------------------------------------

        # ---------- debug hook (prefer using_debug_print if present) ----------
        _udp = globals().get("using_debug_print")
        def _dbg(event: str, **kw):
            try:
                if callable(_udp):
                    _udp(event, **kw)
                else:
                    # fallback to regular debug_print
                    msg = ", ".join(f"{k}={kw[k]!r}" for k in kw)
                    debug_print(f"{event}: {msg}")
            except Exception as e:
                try: debug_print(f"collect_last_name: ⚠️ debug hook error → {e!r}")
                except: pass

        session_data.setdefault(call_sid, {}).setdefault("customer", {})
        raw = (speech_result or "").strip()
        debug_print(f"collect_last_name: raw='{raw}'")
        _dbg("collect_last_name.entry",
            call_sid=call_sid,
            raw=raw,
            stage=session_data[call_sid].get("stage"),
            prior_last_name=session_data[call_sid]["customer"].get("last_name"))

        # 🔇 Silent mode
        if not raw:
            tries = session_data[call_sid].get("silence_last_name", 0) + 1
            session_data[call_sid]["silence_last_name"] = tries
            debug_print(f"collect_last_name: 🤐 silence; tries={tries}")
            _dbg("collect_last_name.silence", tries=tries)

            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                _dbg("collect_last_name.hangup_silence_limit")
                return str(resp)

            gather = make_gather("I didn’t hear your last name. Please say your last name now.")
            resp.append(gather)
            try:
                resp.redirect(url_for("voice"))
                _dbg("collect_last_name.redirect", target="url_for('voice')")
            except Exception:
                resp.redirect("/voice")
                _dbg("collect_last_name.redirect", target="/voice", fallback=True)
            return str(resp)

        # ✅ We heard something → clear silence counter
        session_data[call_sid].pop("silence_last_name", None)

        # ✨ Clean: keep apostrophes and hyphens
        #import string
        punct_keep = "'-"
        trans_table = str.maketrans('', '', "".join(ch for ch in string.punctuation if ch not in punct_keep))
        cleaned = raw.translate(trans_table).strip()
        cleaned = _re.sub(r"\s+", " ", cleaned)
        _dbg("collect_last_name.cleaned", cleaned=cleaned)

        # ✅ Enhanced validation
        has_letter = bool(_re.search(r"[A-Za-z]", cleaned))
        allowed_chars = bool(_re.fullmatch(r"[A-Za-z'\- ]{1,60}", cleaned)) if cleaned else False
        min_len_ok = len(cleaned) >= 2
        valid = bool(cleaned and has_letter and allowed_chars and min_len_ok)
        _dbg("collect_last_name.validation",
            has_letter=has_letter,
            allowed_chars=allowed_chars,
            cleaned_len=len(cleaned),
            min_len_ok=min_len_ok,
            valid=valid)

        if not valid:
            r = session_data[call_sid].get("retry_last_name", 0) + 1
            session_data[call_sid]["retry_last_name"] = r
            debug_print(f"collect_last_name: ❌ invalid last name '{cleaned}' retry={r}")
            _dbg("collect_last_name.invalid", cleaned=cleaned, retry=r)

            if r >= 3:
                resp.say(gpt_speak("Sorry, I couldn’t capture your last name. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                _dbg("collect_last_name.hangup_invalid_limit")
                return str(resp)

            gather = make_gather("Sorry, I didn't catch your last name. Please repeat it clearly.")
            resp.append(gather)
            try:
                resp.redirect(url_for("voice"))
                _dbg("collect_last_name.redirect", target="url_for('voice')", reason="retry_invalid")
            except Exception:
                resp.redirect("/voice")
                _dbg("collect_last_name.redirect", target="/voice", fallback=True, reason="retry_invalid")
            return str(resp)

        # ✅ Success → store last name & advance
        session_data[call_sid]["customer"]["last_name"] = cleaned
        session_data[call_sid]["stage"] = "collect_address"
        session_data[call_sid].pop("retry_last_name", None)
        debug_print(f"collect_last_name: ✅ saved last_name='{cleaned}' → next=collect_address")
        _dbg("collect_last_name.store",
            last_name=cleaned,
            next_stage="collect_address")

        gather = make_gather("Got it. What is your full address, please?")
        resp.append(gather)
        try:
            resp.redirect(url_for("voice"))
            _dbg("collect_last_name.redirect", target="url_for('voice')", reason="advance_to_collect_address")
        except Exception:
            resp.redirect("/voice")
            _dbg("collect_last_name.redirect", target="/voice", fallback=True, reason="advance_to_collect_address")
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
        # 💳 Stage: collect_cc
        # Purpose:
        #   - Collect credit card info in three mini-steps:
        #       (1) Card number (13–19 digits, Luhn-checked)
        #       (2) Expiration (MMYY or MMYYYY) → saved 'MM/YY' (must be current/future)
        #       (3) CVV (3–4 digits)
        #   - Stores under session_data[call_sid]["customer"] as:
        #       cc_number, cc_exp, cc_cvv, cc_name
        #   - On success:
        #       - if cc_update.active → stage=update_customer_cc
        #       - else → stage=book_appt_confirm
        # Silence handling:
        #   - Inline reprompts using make_gather + resp.redirect("/voice") (no _reprompt)
        #   - Steps 2 & 3 mark session["no_input_expected"] = True (DTMF-only)
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
                .replace("-", " ")
                .replace(",", " ")
                .replace(".", " ")
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
                hints="zero one two three four five six seven eight nine",
                input="speech dtmf",
                timeout=25,
                speech_timeout="10",
                finish_on_key="#",
                action="/voice",
                barge_in=True,
            )
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # ✅ Clear silence count when input arrives
        session_data[call_sid].pop("silence_cc", None)

        # -------------------------------
        # Step 1: Card Number (13–19)
        # -------------------------------
        if cc_step == 1:
            pan = _digits_from(raw_dtmf, raw_speech, enforce_dtmf=enforce_dm)
            if len(pan) > 19:
                pan = pan[:19]

            # ❌ Treat exactly 15 digits as invalid; ask to re-enter FULL number
            if not enforce_dm and not raw_dtmf and len(pan) == 15:
                debug_print("collect_cc: ❌ heard 15 digits; invalid length → ask to re-enter full card number")
                gather = make_gather(
                    "That sounded like fifteen digits, which is not valid. "
                    "Please re-enter the full card number now, then press pound.",
                    hints="zero one two three four five six seven eight nine",
                    input="speech dtmf",
                    timeout=25,
                    speech_timeout="10",
                    finish_on_key="#",
                    action="/voice",
                    barge_in=True,
                )
                resp.append(gather)
                resp.redirect("/voice")
                return str(resp)

            if not (13 <= len(pan) <= 19) or not _luhn_ok(pan):
                # After a couple of misses via speech, enforce DTMF typing
                if not raw_dtmf:
                    session_data[call_sid]["cc_speech_tries"] = session_data[call_sid].get("cc_speech_tries", 0) + 1
                    if session_data[call_sid]["cc_speech_tries"] >= 2:
                        session_data[call_sid]["enforce_dtmf_cc"] = True
                        debug_print("collect_cc: 📟 enforcing DTMF for card number entry")
                        gather = make_gather(
                            "That number didn’t sound clear. Please TYPE the full card number now, then press pound.",
                            input="dtmf",
                            timeout=25,
                            finish_on_key="#",
                            action="/voice",
                        )
                        resp.append(gather)
                        resp.redirect("/voice")
                        return str(resp)

                gather = make_gather(
                    "That card number doesn't look right. Please re-enter the full card number, then press pound.",
                    hints="zero one two three four five six seven eight nine",
                    input="speech dtmf",
                    timeout=25,
                    speech_timeout="10",
                    finish_on_key="#",
                    action="/voice",
                    barge_in=True,
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
        # Step 2: Expiration (MMYY/MMYYYY, must be current/future)
        # -------------------------------
        if cc_step == 2:
            session_data[call_sid]["no_input_expected"] = True  # DTMF preferred here

            digits = _digits_from(raw_dtmf, raw_speech, enforce_dtmf=enforce_dm)
            if len(digits) not in (4, 6):
                gather = make_gather(
                    "Please enter the expiration as two digits for month and two digits for year, for example 0 9 2 7, then press pound.",
                    input="dtmf",
                    timeout=20,
                    finish_on_key="#",
                    action="/voice",
                )
                resp.append(gather)
                resp.redirect("/voice")
                return str(resp)

            mm = int(digits[:2]) if digits[:2].isdigit() else 0
            yy = digits[2:] if digits[2:].isdigit() else ""
            if len(yy) == 4:  # MMYYYY → use last two
                yy = yy[-2:]

            if not (1 <= mm <= 12) or not yy.isdigit():
                gather = make_gather(
                    "The month must be between 01 and 12. Please re-enter expiration as M M Y Y, then press pound.",
                    input="dtmf",
                    timeout=20,
                    finish_on_key="#",
                    action="/voice",
                )
                resp.append(gather)
                resp.redirect("/voice")
                return str(resp)

            # Validate not expired (valid through end of month)
            now = datetime.now(tz=_pytz.UTC)  # use existing _pytz (no new imports)
            exp_year = 2000 + int(yy)
            next_month = mm + 1 if mm < 12 else 1
            next_year  = exp_year + 1 if mm == 12 else exp_year
            expiry_boundary = datetime(next_year, next_month, 1, 0, 0, 0, tzinfo=_pytz.UTC)
            if now >= expiry_boundary:
                gather = make_gather(
                    "That card appears expired. Please enter a valid expiration date as M M Y Y, then press pound.",
                    input="dtmf",
                    timeout=20,
                    finish_on_key="#",
                    action="/voice",
                )
                resp.append(gather)
                resp.redirect("/voice")
                return str(resp)

            customer["cc_exp"] = f"{mm:02d}/{yy}"
            debug_print(f"collect_cc: ✅ Expiration saved → {customer['cc_exp']}")
            session_data[call_sid].pop("no_input_expected", None)

            session_data[call_sid]["cc_step"] = 3
            resp.redirect("/voice")
            return str(resp)

        # -------------------------------
        # Step 3: CVV (3–4 digits)
        # -------------------------------
        if cc_step == 3:
            session_data[call_sid]["no_input_expected"] = True  # DTMF preferred here

            digits = _digits_from(raw_dtmf, raw_speech, enforce_dtmf=enforce_dm)

            # Try a forgiving parse again if STT gave words like "two eight eight"
            if (not digits) and (not raw_dtmf) and raw_speech:
                digits = _re.sub(r"\D", "", _normalize_spoken_digits(raw_speech))

            if not (3 <= len(digits) <= 4 and digits.isdigit()):
                # Escalate to DTMF-only after first speech miss
                if not raw_dtmf:
                    session_data[call_sid]["cc_speech_tries"] = session_data[call_sid].get("cc_speech_tries", 0) + 1
                    if session_data[call_sid]["cc_speech_tries"] >= 1:
                        session_data[call_sid]["enforce_dtmf_cc"] = True
                        enforce_dm = True
                        debug_print("collect_cc: 📟 enforcing DTMF for CVV")

                gather = make_gather(
                    "Please enter the three or four digit security code from your card, then press pound.",
                    input="dtmf" if enforce_dm else "speech dtmf",
                    hints=None if enforce_dm else "zero one two three four five six seven eight nine",
                    timeout=15,
                    speech_timeout="8" if not enforce_dm else None,
                    finish_on_key="#",
                    action="/voice",
                    barge_in=True if not enforce_dm else None,
                )
                resp.append(gather)
                resp.redirect("/voice")
                return str(resp)

            customer["cc_cvv"] = digits
            if not customer.get("cc_name"):
                customer["cc_name"] = f"{customer.get('first_name','') } {customer.get('last_name','')}".strip()
            debug_print(f"collect_cc: ✅ CVV saved (len={len(digits)}) ; cc_name='{customer.get('cc_name')}'")

            # Clear flags and advance
            session_data[call_sid].pop("no_input_expected", None)
            session_data[call_sid].pop("cc_step", None)
            session_data[call_sid]["cc_speech_tries"] = 0

            next_stage = "update_customer_cc" if session_data.get(call_sid, {}).get("cc_update", {}).get("active") else "book_appt_confirm"
            session_data[call_sid]["stage"] = next_stage

            # 🔑 One-shot bypass so the next empty POST doesn't trigger the central silence guard
            session_data[call_sid]["skip_silence_once"] = True

            debug_print(f"collect_cc: ➡️ Auto-advancing to {next_stage}")

            resp.redirect("/voice")
            return str(resp)












    elif stage == "book_appt_confirm":
        debug_print("book_appt_confirm: 📍 Stage entered")

        # ----------------------------------------------------------------------
        # 🔇 LOCALIZED SILENCE HANDLING FOR CONFIRM STAGE
        # If there's no speech or DTMF input, retry up to 3 times before hanging up.
        # This is localized and does not touch global silence logic.
        # ----------------------------------------------------------------------
        raw_speech = (speech_result or "").strip()
        raw_dtmf   = (request.values.get("Digits") or "").strip()
        if not raw_speech and not raw_dtmf:
            silent_tries = session_data[call_sid].get("silence_book_appt_confirm", 0) + 1
            session_data[call_sid]["silence_book_appt_confirm"] = silent_tries
            debug_print(f"book_appt_confirm: 🤐 silence detected (tries={silent_tries})")

            if silent_tries >= 3:
                resp.say(gpt_speak("I'm still not hearing anything. Let's try again later."), VOICE)
                resp.hangup()
                return str(resp)

            # Re-prompt with a friendly fallback message
            prompt = "Would you like to confirm your appointment now? You can say yes to continue."
            resp.append(make_gather(prompt))
            return str(resp)

        # ----------------------------------------------------------------------
        # Ignore any incidental speech/DTMF here — this stage auto-books.
        try:
            if raw_speech or raw_dtmf:
                debug_print("book_appt_confirm: 🛡️ ignoring incidental input at confirm stage")
        except Exception:
            pass

        # ---- Doctor info ----
        doctor_id = session_data[call_sid].get("doctor_id")
        if not doctor_id:
            debug_print("book_appt_confirm: ❌ missing doctor_id → choose_doctor")
            session_data[call_sid]["stage"] = "choose_doctor"
            resp.append(make_gather("Which doctor would you like to see?"))
            return str(resp)

        doctor_name = googleid_dr_name_map.get(doctor_id, "the doctor")

        # ---- Appointment time ----
        appt = session_data[call_sid].get("appointment_time", {}) or {}
        appointment_start = appt.get("start")
        appointment_end   = appt.get("end")
        if not appointment_start:
            debug_print("book_appt_confirm: ❌ missing appointment_start")
            resp.say(gpt_speak("Appointment time is missing. Goodbye!"), VOICE)
            resp.hangup()
            return str(resp)

        # Local-friendly time
        tz_name = (globals().get("CLINIC_TZ") or globals().get("LOCAL_TZ") or "America/Chicago")
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

        # Compute end if missing (default 30m)
        if not appointment_end:
            try:
                dur = None
                for k in ("APPOINTMENT_DURATION_MINUTES", "SESSION_TIME", "SESSIUON_TIME"):
                    v = globals().get(k)
                    if v:
                        try:
                            dur = int(v); break
                        except Exception:
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

        # ---- Customer info ----
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
                default_country = (session_data[call_sid].get("phone_country") or globals().get("COUNTRY") or "US").upper()
                phone_e164 = normalize_phone_e164(phone_raw, default_country) or ""
                if not phone_e164:
                    alt = "EG" if default_country != "EG" else "US"
                    phone_e164 = normalize_phone_e164(phone_raw, alt) or ""
            except Exception:
                phone_e164 = ""

        if not phone_e164:
            session_data[call_sid]["stage"] = "collect_phone"
            resp.append(make_gather("Before we confirm your appointment, please provide your phone number."))
            return str(resp)

        if not customer_dob:
            session_data[call_sid]["stage"] = "collect_dob"
            resp.append(make_gather(
                "Before we confirm, please say your date of birth, for example, 'July 3 1990'. "
                "You can also enter month, day, and year, then press pound."
            ))
            return str(resp)

        # ---- Upsert customer (best-effort) ----
        try:
            init_db()
            insert_customer(
                phone=phone_e164, dob=customer_dob,
                first_name=first_name, last_name=last_name, address=customer_address,
                cc_name=(customer.get("cc_name") or effective_name or ""),
                cc_number=(customer.get("cc_number") or ""),
                cc_exp=(customer.get("cc_exp") or ""),
                cc_cvv=(customer.get("cc_cvv") or "")
            )
        except Exception as e:
            debug_print(f"book_appt_confirm: insert_customer failed → {e}")

        # ---- Create Google Calendar event (no re-check) ----
        google_event_id = session_data[call_sid].get("google_event_id", "")
        if google_event_id:
            debug_print(f"book_appt_confirm: ℹ️ event already created earlier → id={google_event_id}")
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
                resp.append(make_gather("Sorry, I couldn't confirm that slot. Please say a new date and time, for example, September 14th at 10 AM."))
                return str(resp)

        # ---- Persist locally (JSON) via confirm_appointment_by_name ----
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
            debug_print(
                "book_appt_confirm: 🗂️ local persist → "
                f"created={persist.get('created')} reason={persist.get('reason')}"
            )
        except Exception as e:
            debug_print(f"book_appt_confirm: ⚠️ local persist failed → {e}")

        # ---- Voice confirmation + SMS, then hang up ----
        msg = f"Your appointment with {doctor_name} has been booked"
        if formatted_time: msg += f" on {formatted_time}"
        msg += ". We look forward to seeing you. Goodbye!"
        resp.say(gpt_speak(msg), VOICE)

        try:
            sms = f"Hi {(effective_name or 'there')}, your appointment with {doctor_name} is confirmed"
            if formatted_time: sms += f" on {formatted_time}"
            sms += ". Thank you for choosing Epic Therapist Clinic."
            _ = client.messages.create(body=sms, from_=TWILIO_PHONE_NUMBER, to=phone_e164)
        except Exception as e:
            debug_print(f"book_appt_confirm: ⚠️ SMS failed → {e}")

        resp.hangup()
        session_data.pop(call_sid, None)
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






    elif stage == "cancel_appt_get_phone_number":
        # ----------------------------------------------------------------------
        # 📞 Collect the phone number used when booking, then move to DOB check.
        #  - Silent-mode aware (re-prompts up to 3x if nothing is heard)
        #  - Accepts DTMF or speech
        #  - Normalizes to E.164 ONLY (US/Egypt supported via normalize_phone_e164)
        #  - Stores under session_data[call_sid]["cancel"]["phone_e164"]
        #  - Next stage: cancel_appt_get_dob
        # ----------------------------------------------------------------------
        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("cancel", {})

        # Pull inputs (DTMF + speech)
        try:
            dtmf_digits = (request.values.get("Digits") or "").strip()
        except Exception:
            dtmf_digits = ""
        speech_text = (speech_result or "").strip()

        debug_print(
            f"cancel_appt_get_phone_number: 🗣️ speech='{speech_text}' "
            f"🔢 DTMF='{dtmf_digits}'"
        )

        # 🔇 Silent mode: nothing heard at all
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
                "I didn’t hear your phone number. Please say or type your phone number including area code, "
                "then press pound."
            )
            session_data[call_sid]["stage"] = "cancel_appt_get_phone_number"
            resp.append(make_gather(prompt, hints="zero one two three four five six seven eight nine double triple"))
            resp.redirect("/voice")
            return str(resp)

        # If we DID hear something, clear the silence counter
        session_data[call_sid].pop("silence_cancel_phone", None)

        # --- helpers --------------------------------------------------------------
        def _spoken_to_digits(raw: str) -> str:
            """
            Convert spoken words to digits.
            Supports 'double'/'triple' and common homophones (oh/o=0, to/too=2, ate=8).
            Also extracts any digits already present in the string.
            """
            if not raw:
                return ""
            words = (
                raw.lower()
                .replace("-", " ").replace(",", " ").replace(".", " ")
                .replace("(", " ").replace(")", " ")
                .split()
            )
            m = {"zero":"0","oh":"0","o":"0","one":"1","two":"2","to":"2","too":"2",
                "three":"3","four":"4","for":"4","five":"5","six":"6","seven":"7",
                "eight":"8","ate":"8","nine":"9"}
            out = []; i = 0
            while i < len(words):
                w = words[i].strip()
                if w in ("double","triple") and i+1 < len(words):
                    nxt = words[i+1].strip()
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

        # Prefer DTMF; else speech→digits (for fallback/validation messages)
        raw_digits = _re.sub(r"\D", "", dtmf_digits) if dtmf_digits else _re.sub(r"\D", "", _spoken_to_digits(speech_text))
        default_country = (session_data[call_sid].get("country") or COUNTRY).upper()
        raw_for_e164 = (speech_text or raw_digits or "").strip()

        # Try to build E.164 (US/Egypt); accept already +E.164 with light validation
        phone_e164 = ""
        try:
            if raw_for_e164.startswith("+"):
                body_digits = "".join(ch for ch in raw_for_e164[1:] if ch.isdigit())
                if 8 <= len(body_digits) <= 15:
                    phone_e164 = "+" + body_digits

            if not phone_e164:
                debug_print(f"cancel_appt_get_phone_number: normalizing via {default_country} from='{raw_for_e164}'")
                phone_e164 = normalize_phone_e164(raw_for_e164, default_country) or ""

            if not phone_e164 and raw_digits:
                # secondary attempt with bare digits
                phone_e164 = normalize_phone_e164(raw_digits, default_country) or ""

            if not phone_e164:
                # try the other supported country as a last resort (still E.164 only)
                alt = "EG" if default_country != "EG" else "US"
                debug_print(f"cancel_appt_get_phone_number: retry via alt country={alt}")
                phone_e164 = normalize_phone_e164(raw_for_e164 or raw_digits, alt) or ""
        except Exception as e:
            debug_print(f"cancel_appt_by_phone_number: ⚠️ normalize_phone_e164 error → {e}")
            phone_e164 = ""

        debug_print(
            f"cancel_appt_by_phone_number: 🧪 parsed digits='{raw_digits}' "
            f"default_country='{default_country}' → e164='{phone_e164 or '∅'}'"
        )

        # Validate (E.164 required)
        if not phone_e164:
            session_data[call_sid]["stage"] = "cancel_appt_get_phone_number"
            prompt = (
                "I didn’t catch a valid phone number. Please say or type your phone number including area code, "
                "then press pound."
            )
            resp.append(make_gather(prompt, hints="zero one two three four five six seven eight nine double triple"))
            resp.redirect("/voice")
            return str(resp)

        # ✅ Store E.164 only and proceed
        session_data[call_sid]["cancel"]["phone_e164"] = phone_e164
        session_data[call_sid]["stage"] = "cancel_appt_get_dob"

        resp.append(make_gather(
            "Thanks. Now, please tell me your date of birth to verify your identity. "
            "For example, say July third 1990, or type it as 07031990 then press pound."
        ))
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







    elif stage == "cancel_appt_iterate":
        # ----------------------------------------------------------------------
        # 🗂️ Stage: cancel_appt_iterate
        #
        # Purpose:
        #   - Load caller’s appointments for the specific doctor (JSON file).
        #   - Filter by phone + DOB to build candidate list.
        #   - Present candidates one by one, ask if cancel.
        #
        # Behavior:
        #   - "yes"/"1" → store candidate and jump to cancel_appt_confirm.
        #   - "no"/"2" → move to next candidate, or end if none left.
        #   - Silence / unclear → just re-present candidate (no nag message).
        # ----------------------------------------------------------------------

        debug_print("cancel_appt_iterate: 📍 Stage entered")

        cancel_ctx = session_data[call_sid].setdefault("cancel", {})
        doctor = (cancel_ctx.get("doctor") or "").strip()
        phone_e164 = (cancel_ctx.get("phone_e164") or "").replace("+", "").lstrip("0")
        dob = (cancel_ctx.get("dob") or "").strip()

        debug_print(f"cancel_appt_iterate: inputs → doctor='{doctor}', phone='{phone_e164}', dob='{dob}'")

        candidates = cancel_ctx.get("candidates")

        # ---------- Build candidates on first entry ----------
        if not candidates:
            path = f"{DB_FOLDER}/{doctor.lower().replace(' ', '_')}.json"
            try:
                with open(path, "r") as f:
                    appts = json.load(f)
            except Exception as e:
                debug_print(f"cancel_appt_iterate: ⚠️ could not load {path} → {e}")
                appts = []

            candidates = []
            for appt in appts:
                appt_phone = (appt.get("phone") or "").replace("+", "").lstrip("0")
                appt_dob = (appt.get("dob") or "").strip()

                debug_print(
                    f"cancel_appt_iterate: 🔍 compare → input phone={phone_e164}, "
                    f"appt phone={appt_phone}; input dob={dob}, appt dob={appt_dob}"
                )

                if appt_phone == phone_e164 and (not dob or appt_dob == dob):
                    candidates.append({
                        "doctor_name": doctor,
                        "start_utc": appt.get("utc_start"),
                        "end_utc": appt.get("utc_end"),
                        "friendly": appt.get("friendly_local"),
                        "phone_e164": appt_phone,
                        "dob": appt_dob,
                    })

            cancel_ctx["candidates"] = candidates
            cancel_ctx["iter_index"] = 0
            debug_print(f"cancel_appt_iterate: ✅ built {len(candidates)} candidate(s)")

            if not candidates:
                resp.say("There are no upcoming events to cancel. Goodbye.", VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Announce total once at the beginning
            resp.say(f"I found {len(candidates)} upcoming appointments.", VOICE)

        # ---------- Handle user input ----------
        try:
            dtmf = (request.values.get("Digits") or "").strip()
        except Exception:
            dtmf = ""
        utter = (speech_result or "").strip().lower()

        # Normalize utter → remove punctuation, spaces
        utter = _re.sub(r"[^a-z0-9]+", "", utter)

        debug_print(f"cancel_appt_iterate: normalized utter='{utter}', dtmf='{dtmf}'")

        YES = {"yes", "yeah", "yep", "correct", "confirm"}
        NO  = {"no", "nope", "next"}

        idx = int(cancel_ctx.get("iter_index", 0))
        total = len(cancel_ctx["candidates"])

        # Guard out of range
        if idx >= total:
            resp.say("That was the last appointment. Goodbye.", VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)

        cand = cancel_ctx["candidates"][idx]

        # --- YES (confirm cancel) ---
        if utter in YES or dtmf == "1":
            debug_print(f"cancel_appt_iterate: ✅ YES user confirmed candidate #{idx+1}/{total}")
            cancel_ctx["matching_event"] = cand
            session_data[call_sid]["stage"] = "cancel_appt_confirm"
            resp.redirect("/voice")
            return str(resp)

        # --- NO (move to next) ---
        if utter in NO or dtmf == "2":
            debug_print(f"cancel_appt_iterate: ↪️ NO user skipped candidate #{idx+1}/{total}")
            idx += 1
            if idx >= total:
                resp.say("That was the last appointment. Goodbye.", VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)
            cancel_ctx["iter_index"] = idx
            cand = cancel_ctx["candidates"][idx]

        # --- SILENCE or unclear input → just present current candidate again ---
        debug_print(f"cancel_appt_iterate: 🗣️ presenting candidate #{idx+1}/{total}")
        say_line = (
            f"Appointment with {cand['doctor_name']} on {cand['friendly']}. "
            "Do you want to cancel this one? Say yes or no. Press 1 for yes, or 2 for no."
        )
        gather = make_gather(
            say_line,
            hints="yes no one two",
            input="speech dtmf",
            timeout=20,
            speech_timeout="8",
            finish_on_key="#",
            action="/voice",
            barge_in=True,
        )
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

    elif stage == "cancel_appt_confirm":
            debug_print("📍 Stage: cancel_appt_confirm (auto-execute, no confirmation prompt)")

            cancel_ctx  = session_data[call_sid].setdefault("cancel", {})
            cand        = cancel_ctx.get("matching_event") or {}

            doctor      = cand.get("doctor_name") or cancel_ctx.get("doctor") or ""
            utc_start   = cand.get("start_utc")   or cancel_ctx.get("utc_start") or ""
            utc_end     = cand.get("end_utc")     or cancel_ctx.get("utc_end")   or ""
            phone_raw   = (cand.get("phone_e164") or cancel_ctx.get("phone_e164") or "").strip()
            dob         = cand.get("dob") or cancel_ctx.get("dob") or session_data[call_sid].get("customer", {}).get("dob") or ""

            # ------------------------------------------------------------------
            # Helper: normalize phone to E.164
            # ------------------------------------------------------------------
            def _normalize_phone(phone_str: str, country: str = "US") -> str:
                import re as _re
                phone_digits = _re.sub(r"\D", "", phone_str)
                if not phone_digits:
                    return ""
                if phone_digits.startswith("1") and len(phone_digits) == 11:
                    return "+" + phone_digits
                if len(phone_digits) == 10 and country.upper() == "US":
                    return "+1" + phone_digits
                return "+" + phone_digits

            default_country = (session_data[call_sid].get("phone_country") or COUNTRY or "US").upper()
            phone_e164 = _normalize_phone(phone_raw, default_country)

            # ------------------------------------------------------------------
            # Helper: friendly date string
            # ------------------------------------------------------------------
            def _friendly_from_iso(utc_iso: str, tz_name: str = "America/Chicago") -> str:
                try:
                    import dateutil.parser as dtparser
                    import pytz
                    dt_utc = dtparser.isoparse(utc_iso)
                    local = dt_utc.astimezone(pytz.timezone(tz_name))
                    return local.strftime("%A, %B %-d at %-I:%M %p")
                except Exception:
                    return utc_iso or "the scheduled time"

            friendly = _friendly_from_iso(utc_start)

            # ------------------------------------------------------------------
            # ✅ Check slot existence before attempting cancel
            # ------------------------------------------------------------------
            calendar_id = cancel_ctx.get("calendar_id")
            slot_exists_before = False
            if calendar_id and utc_start and utc_end:
                try:
                    slot_exists_before = not is_time_slot_available(calendar_id, utc_start, utc_end, creds)
                    if slot_exists_before:
                        debug_print(f"cancel_appt_confirm: ✅ Slot exists BEFORE deletion ({utc_start} → {utc_end})")
                    else:
                        debug_print(f"cancel_appt_confirm: ❌ Slot does NOT exist BEFORE deletion ({utc_start} → {utc_end})")
                except Exception as e:
                    debug_print(f"cancel_appt_confirm: ⚠️ slot pre-check failed → {e}")

            # ------------------------------------------------------------------
            # Cancel appointment in local JSON
            # ------------------------------------------------------------------
            local_ok = False
            if doctor and phone_e164 and dob and utc_start and slot_exists_before:
                try:
                    local_ok = cancel_appointment_for_dr_name(
                        doctor_name=doctor,
                        phone=phone_e164,
                        dob=dob,
                        utc_start=utc_start
                    )
                    if local_ok:
                        debug_print(f"cancel_appt_confirm: 🗑️ Local file cancel succeeded for {doctor}")
                except Exception as e:
                    debug_print(f"cancel_appt_confirm: local cancel failed → {e}")

            # ------------------------------------------------------------------
            # Cancel appointment in Google Calendar
            # ------------------------------------------------------------------
            gcal_ok = False
            if calendar_id and utc_start and phone_e164 and slot_exists_before:
                try:
                    start_dt  = dtparser.isoparse(utc_start)
                    win_start = (start_dt - timedelta(minutes=30)).astimezone(timezone.utc).isoformat()
                    win_end   = (start_dt + timedelta(minutes=30)).astimezone(timezone.utc).isoformat()

                    matched = get_upcoming_events(calendar_id, phone_e164, win_start, win_end, creds, debug=True)
                    ev = matched[0] if isinstance(matched, list) and matched else (matched if isinstance(matched, dict) else None)
                    if ev and ev.get("id"):
                        service = build("calendar", "v3", credentials=creds)
                        service.events().delete(calendarId=calendar_id, eventId=ev["id"]).execute()
                        gcal_ok = True
                        debug_print(f"cancel_appt_confirm: 🗑️ GCal event deleted id={ev['id']}")
                except Exception as e:
                    debug_print(f"cancel_appt_confirm: GCal delete failed → {e}")

            # ------------------------------------------------------------------
            # 🔍 Check slot availability AFTER deletion
            # ------------------------------------------------------------------
            if calendar_id and utc_start and utc_end:
                try:
                    available_after = is_time_slot_available(calendar_id, utc_start, utc_end, creds)
                    if available_after:
                        debug_print(f"cancel_appt_confirm: ✅ Slot is FREE after deletion ({utc_start} → {utc_end})")
                    else:
                        debug_print(f"cancel_appt_confirm: ❌ Slot is STILL blocked after deletion ({utc_start} → {utc_end})")
                except Exception as e:
                    debug_print(f"cancel_appt_confirm: ⚠️ slot post-check failed → {e}")

            # ------------------------------------------------------------------
            # Respond to caller
            # ------------------------------------------------------------------
            if (local_ok or gcal_ok) and slot_exists_before:
                resp.say(
                    gpt_speak(f"Your appointment with {doctor} on {friendly} has been cancelled. Thank you!"),
                    VOICE
                )
            else:
                resp.say(
                    gpt_speak("I'm sorry, I couldn't find an appointment under that phone number and time to cancel."),
                    VOICE
                )

            # Cleanup
            session_data.pop(call_sid, None)
            resp.hangup()
            return str(resp)



   



if __name__ == "__main__":
    
    # ------------------------
    # 🔁 Call this once at startup
    # ------------------------

    load_doctor_appointments()


    app.run(host="0.0.0.0", port=5000)
