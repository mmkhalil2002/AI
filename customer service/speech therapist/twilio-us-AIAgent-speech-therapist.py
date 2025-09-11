# update  09/11/25 time_saved 02:3f pm
#  am
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
    #   and $ matches at the very end of the *entire* string (or just before a final '\n').
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
    # - Without MULTILINE, $ matches at the end of the string *or* right before a final '\n'.
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



from uuid import uuid4
from datetime import datetime as _dt
from datetime import time as dtime
from typing import Any, Optional, List, Dict, Tuple, Iterator, Iterable, Union
from datetime import datetime, date, time, timedelta, timezone, time as _time
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
from datetime import time
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
    from googleapiclient.discovery import build
    from dateutil.parser import isoparse
    import pytz as _pytz

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










def normalize_phone_digits(phone: str) -> str:
    """Digits-only normalization for matching (calendar description & JSON)."""
    return ''.join(ch for ch in (phone or "") if ch.isdigit())


# ===== local doctor JSON cancellation (by doctor+phone+dob+utc_start) =====

#  remove phone10 

def cancel_appointment_by_name(
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

def customer_search(phone: str, dob: str, *, default_country: str = COUNTRY) -> bool:
    """
    Return True if a customer (phone|dob) exists in customers.json, else False.

    Lookup (E.164 only):
      1) Normalize input as E.164 using default_country (falls back US↔EG).
      2) Check key = _key(phone_e164, dob_iso).
    """
    init_db()
    dob_iso = (dob or "").strip()
    raw = (phone or "").strip()

    # Try E.164 first (accept +E.164 as-is; otherwise normalize)
    phone_e164 = ""
    if raw.startswith("+") and raw[1:].replace(" ", "").isdigit():
        phone_e164 = "+" + raw[1:].replace(" ", "")
    else:
        try:
            phone_e164 = normalize_phone_e164(raw, default_country) or ""
        except Exception:
            phone_e164 = ""
        # Secondary guess (useful when caller is EG and default is US, or vice versa)
        if not phone_e164:
            try:
                alt_country = "EG" if default_country.upper() != "EG" else "US"
                phone_e164 = normalize_phone_e164(raw, alt_country) or ""
            except Exception:
                phone_e164 = ""

    if not phone_e164:
        debug_print(f"customer_search: ❌ invalid phone '{raw}' (no E.164)")
        return False

    data = _load_customers()
    key_e164 = _key(phone_e164, dob_iso)
    exists = key_e164 in data
    debug_print(f"customer_search: phone_e164={phone_e164} dob={dob_iso or '∅'} → {exists}")
    return exists

def _save_customers(data: Dict[str, Dict[str, Any]]) -> None:
    """Write the customers map to disk in readable (pretty) form."""
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

#  update to remove legact phone 10

def _update_existing_block_in_place(
    phone_norm: str,
    dob_clean: str,
    updates: dict,
    *,
    default_country: str = COUNTRY  # uses global COUNTRY by default
) -> bool:
    """
    Edit the matching block in place:
      - Always bump 'Last Seen At' to now
      - If updates contain non-empty values, refresh:
          First Name, Last Name, Address, CC Name, CC Number, CC Exp, CC CVV
      - Preserve original 'Created At' and title line
    Returns True if a block was updated.

    E.164 ONLY:
      - phone_norm is normalized to E.164.
      - We normalize the block's "Phone:" line to E.164 and match by strict equality.
      - No 10-digit (phone10) paths remain. Legacy blocks should still normalize.
    """
    if not os.path.exists(DB_FILE):
        return False

    with open(DB_FILE, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    changed = False
    out: list[str] = []
    i = 0

    # ---- build target E.164 --------------------------------------------------
    raw_in = (phone_norm or "").strip()

    def _to_e164(raw: str, pref_country: str) -> str:
        raw = (raw or "").strip()
        if raw.startswith("+") and raw[1:].replace(" ", "").isdigit():
            return "+" + raw[1:].replace(" ", "")
        try:
            e = normalize_phone_e164(raw, pref_country) or ""
            if not e:
                alt = "EG" if pref_country.upper() != "EG" else "US"
                e = normalize_phone_e164(raw, alt) or ""
            return e
        except Exception:
            return ""

    target_e164 = _to_e164(raw_in, default_country)
    if not target_e164:
        # If we cannot normalize the input, we cannot match anything safely.
        return False

    # ---- DOB → ISO -----------------------------------------------------------
    def _dob_iso(s: str) -> str:
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
                return ""
        return ""
    dob_target_iso = _dob_iso(dob_clean)

    # ---- phone matcher (E.164 only) -----------------------------------------
    def _phones_match(block_phone: str) -> bool:
        b_e164 = _to_e164(block_phone, default_country)
        return bool(b_e164) and (b_e164 == target_e164)

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("insert_customer:"):
            start = i
            i += 1
            while i < len(lines) and not lines[i].startswith("insert_customer:"):
                i += 1
            block = lines[start:i]

            b_phone, b_dob = _extract_phone_dob(block)
            b_dob_iso = _dob_iso(b_dob)

            # Strict: E.164 phone equality AND ISO DOB equality
            if _phones_match(b_phone) and (b_dob_iso == dob_target_iso):
                # Pull existing values
                cur = {
                    "title":        block[0],
                    "phone":        _get_value(block, "Phone") or "",
                    "dob":          _get_value(block, "DOB") or "",
                    "first_name":   _get_value(block, "First Name") or "",
                    "last_name":    _get_value(block, "Last Name") or "",
                    "address":      _get_value(block, "Address") or "",
                    "cc_name":      _get_value(block, "CC Name") or "",
                    "cc_number":    _get_value(block, "CC Number") or "",  # renderer may mask
                    "cc_exp":       _get_value(block, "CC Exp") or "",
                    "cc_cvv":       _get_value(block, "CC CVV") or "",     # renderer may mask
                    "created_at":   _get_value(block, "Created At") or "—",
                    "last_seen_at": now,
                }

                # Apply non-empty updates (one-line sanitize)
                def pick(new_val, old_val):
                    new_val = _oneline(new_val)
                    return new_val if new_val else old_val

                cur["first_name"] = pick(updates.get("first_name"), cur["first_name"])
                cur["last_name"]  = pick(updates.get("last_name"),  cur["last_name"])
                cur["address"]    = pick(updates.get("address"),    cur["address"])
                cur["cc_name"]    = pick(updates.get("cc_name"),    cur["cc_name"])

                nv = _oneline(updates.get("cc_number"))
                if nv:
                    cur["cc_number"] = nv
                nv = _oneline(updates.get("cc_exp"))
                if nv:
                    cur["cc_exp"] = nv
                nv = _oneline(updates.get("cc_cvv"))
                if nv:
                    cur["cc_cvv"] = nv

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









# ===== local doctor JSON cancellation (by doctor+phone+dob+utc_start) =====



def cancel_appointment_by_name(
    doctor_name: str,
    phone: str,
    dob: str,
    utc_start: str,
    *,
    default_country: str = COUNTRY  # e.g., "US" or "EG"
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
        per record (US/EG supported) and compare E.164 ↔ E.164 only.
    """

    # ---------- inputs (raw) ----------
    debug_print(
        "cancel_appointment_by_name: 🟡 inputs"
        f"\n  doctor_name = '{doctor_name}'"
        f"\n  phone(raw)  = '{(phone or '').strip()}'"
        f"\n  dob         = '{(dob or '').strip()}'"
        f"\n  utc_start   = '{(utc_start or '').strip()}'"
        f"\n  country     = '{(default_country or 'US')}'"
    )

    # ---------- normalize input phone to E.164 ----------
    raw = (phone or "").strip()
    phone_e164 = ""
    if raw.startswith("+") and raw[1:].replace(" ", "").isdigit():
        phone_e164 = "+" + raw[1:].replace(" ", "")
        debug_print(f"cancel_appointment_by_name: 📞 pass-through E.164 = '{phone_e164}'")
    else:
        try:
            debug_print(f"cancel_appointment_by_name: 📞 trying normalize_phone_e164(raw, {default_country})")
            phone_e164 = normalize_phone_e164(raw, (default_country or "US").upper()) or ""
            debug_print(f"cancel_appointment_by_name: 📞 result (default) = '{phone_e164 or '∅'}'")
            if not phone_e164:
                alt = "EG" if (default_country or "US").upper() != "EG" else "US"
                debug_print(f"cancel_appointment_by_name: 📞 trying normalize_phone_e164(raw, {alt})")
                phone_e164 = normalize_phone_e164(raw, alt) or ""
                debug_print(f"cancel_appointment_by_name: 📞 result (alt) = '{phone_e164 or '∅'}'")
        except Exception as e:
            debug_print(f"cancel_appointment_by_name: ⚠️ normalize_phone_e164 error → {e}")
            phone_e164 = ""

    dob_str = (dob or "").strip()
    full_path = get_doctor_filename(doctor_name)
    debug_print(
        "cancel_appointment_by_name: 🟡 normalized inputs"
        f"\n  phone_e164  = '{phone_e164 or '∅'}'"
        f"\n  dob_iso     = '{dob_str or '∅'}'"
        f"\n  file_path   = '{full_path}'"
    )

    if not os.path.exists(full_path):
        debug_print("cancel_appointment_by_name: ❌ file not found")
        return False
    if not phone_e164 or not dob_str or not utc_start:
        debug_print("cancel_appointment_by_name: ❌ missing one of (phone_e164, dob, utc_start)")
        return False

    # ---------- load list ----------
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            debug_print("cancel_appointment_by_name: ❌ JSON root is not a list")
            return False
        debug_print(f"cancel_appointment_by_name: 📄 loaded {len(data)} record(s)")
    except Exception as e:
        debug_print(f"cancel_appointment_by_name: ❌ read/parse error → {e}")
        return False

    # ---------- normalize times to comparable UTC ISO (no micros) ----------
    def _to_utc_iso(s: str) -> str:
        dt = dtparser.isoparse(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        out = dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        debug_print(f"_to_utc_iso: '{s}' → '{out}'")
        return out

    try:
        target_norm = _to_utc_iso(utc_start)
        debug_print(f"cancel_appointment_by_name: 🎯 target UTC   = '{target_norm}'")
    except Exception as e:
        debug_print(f"cancel_appointment_by_name: ❌ utc parse error → {e}")
        return False

    # ---------- helper: derive E.164 for a stored appt record ----------
    def _appt_e164(appt: dict) -> str:
        pe = (appt.get("phone_e164") or "").strip()
        if pe.startswith("+") and pe[1:].replace(" ", "").isdigit():
            debug_print(f"_appt_e164: using record phone_e164 = '{pe}'")
            return "+" + pe[1:].replace(" ", "")
        cand = (appt.get("phone") or "").strip()
        if cand:
            try:
                e164 = normalize_phone_e164(cand, (default_country or "US").upper()) or ""
                if not e164:
                    alt = "EG" if (default_country or "US").upper() != "EG" else "US"
                    e164 = normalize_phone_e164(cand, alt) or ""
                debug_print(f"_appt_e164: derived from 'phone'='{cand}' → '{e164 or '∅'}'")
                if e164:
                    return e164
            except Exception as e:
                debug_print(f"_appt_e164: ⚠️ normalize error for '{cand}' → {e}")
        debug_print("_appt_e164: no usable phone on record")
        return ""

    kept = []
    removed = 0

    for idx, appt in enumerate(data):
        if not isinstance(appt, dict):
            debug_print(f"[{idx}] skip non-dict record")
            kept.append(appt)
            continue

        ap_e164 = _appt_e164(appt)
        ap_dob  = (appt.get("dob", "") or "").strip()
        ap_time_raw = (appt.get("time") or appt.get("start") or "").strip()

        try:
            ap_time_norm = _to_utc_iso(ap_time_raw) if ap_time_raw else ""
        except Exception as e:
            debug_print(f"[{idx}] time parse error for '{ap_time_raw}' → {e} (keeping record)")
            kept.append(appt)
            continue

        debug_print(
            f"[{idx}] record"
            f"\n     phone_e164(rec) = '{ap_e164 or '∅'}'"
            f"\n     dob(rec)        = '{ap_dob or '∅'}'"
            f"\n     time(raw)       = '{ap_time_raw or '∅'}'"
            f"\n     time(norm)      = '{ap_time_norm or '∅'}'"
        )

        matched = (ap_e164 == phone_e164) and (ap_dob == dob_str) and (ap_time_norm == target_norm)
        debug_print(f"[{idx}] match? {matched}")
        if matched:
            removed += 1
        else:
            kept.append(appt)

    if removed == 0:
        debug_print("cancel_appointment_by_name: ❌ no matching record found")
        return False

    # ---------- write back ----------
    try:
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(kept, f, indent=2, ensure_ascii=False)
        debug_print(f"cancel_appointment_by_name: ✅ deleted {removed} appt(s); kept {len(kept)}")
        return True
    except Exception as e:
        debug_print(f"cancel_appointment_by_name: ❌ write error → {e}")
        return False






# update remove phone 10
def get_doctor_appts_for(doctor_name: str, phone: str, dob: str = None) -> list:
    """
    Read appointment_data/doctors/<doctor_name>.json and return all
    appointment dicts that match the given caller:
      - phone is REQUIRED (E.164 string, e.g., +12025550123 or +2011xxxxxxx)
      - dob is OPTIONAL (normalized to YYYY-MM-DD if possible)

    Returned list is sorted chronologically by start time if present.
    Uses debug_print for logging (falls back to print if unavailable).

    E.164 behavior:
      - We normalize the input with normalize_phone_e164 using a default country
        (GLOBAL COUNTRY if defined, else 'US'), and falling back to the other
        supported country (US/EG) if needed.
      - Records are matched by 'phone_e164'. If a record lacks that field but
        has legacy 'phone' digits, we try to derive an E.164 using the input
        phone’s country as a hint (US: +1, EG: +20).
    """
    # ---------- local helpers (self-contained) ----------

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
                return ""
        return ""

    def _extract_start_iso(appt: dict) -> str:
        """Prefer 'start' then 'time' field; may be empty if not present."""
        return (appt.get("start") or appt.get("time") or "").strip()

    # ---------- normalize inputs ----------
    # Choose a default country from global COUNTRY if present; else US
    try:
        _default_country = COUNTRY
    except NameError:
        _default_country = "US"

    # Primary: E.164 using your helper (accepts '+E.164' as-is when valid)
    phone_e164 = ""
    try:
        phone_e164 = normalize_phone_e164(phone or "", _default_country) or ""
        if not phone_e164:
            alt = "EG" if _default_country.upper() != "EG" else "US"
            phone_e164 = normalize_phone_e164(phone or "", alt) or ""
    except Exception:
        phone_e164 = ""

    if not phone_e164:
        debug_print(f"get_doctor_appts_for: ❌ invalid phone '{phone}' (no E.164)")
        return []

    # Infer country hint from input E.164 (used only to up-convert legacy record phones)
    if phone_e164.startswith("+1"):
        inferred_country = "US"
    elif phone_e164.startswith("+20"):
        inferred_country = "EG"
    else:
        inferred_country = _default_country

    dob_iso = _normalize_dob_iso(dob) if dob else ""

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

        # Prefer E.164 stored on the record
        ap_e164 = (appt.get("phone_e164") or "").strip()

        # If record lacks E.164, derive it from the legacy 'phone' field using the hint country
        if not ap_e164:
            raw_legacy = (appt.get("phone") or "").strip()
            try:
                ap_e164 = normalize_phone_e164(raw_legacy, inferred_country) or ""
                if not ap_e164:
                    # one last pass with the opposite of the hint if we only support US/EG
                    alt = "EG" if inferred_country.upper() != "EG" else "US"
                    ap_e164 = normalize_phone_e164(raw_legacy, alt) or ""
            except Exception:
                ap_e164 = ""

        # Phone match (E.164 only)
        if not ap_e164 or ap_e164 != phone_e164:
            continue

        # Optional DOB exact match (normalized)
        if dob_iso:
            ap_dob = _normalize_dob_iso(appt.get("dob", "") or "")
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

    debug_print(
        f"get_doctor_appts_for: ✅ doctor='{doctor_name}' phone='{phone_e164}' "
        f"dob='{dob_iso or '∅'}' → {len(matches)} appt(s)"
    )
    return matches

#   update remove phone 10
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
# - Adds ±1s boundary "fuzz" to avoid edge inclusivity issues. (configurable)
# - Logs any blocking busy windows for debugging.
# - Robust tz handling + all-day events in fallback overlap test.
# - Enforces clinic policy (working days/hours/lunch/past) before Google.
# - UPDATED: signature order to (calendar_id, creds, start_iso, end_iso)
# - UPDATED: reject if start <= now (strictly future)
# - UPDATED: safe debug wrapper + single client build for FB and events()
# =============================================================================
def is_time_slot_available(calendar_id: str, start_iso: str, end_iso: str, creds) -> bool:
    """
    True iff the slot passes clinic policy AND no overlapping event exists.

    Clinic policy (local clinic TZ):
      • Not fully in the past  → end > now   (half-open window, so equality means past)
      • Working day            → weekday ∈ WORKING_DAYS (defaults Mon–Fri)
      • Working hours          → WSTART:00 ≤ start  AND  end ≤ WEND:00
      • Lunch                  → no real overlap with [LUNCH_START, LUNCH_END)
                                 (ending exactly at lunch start is allowed)

    Calendar checks (UTC):
      • Primary: FreeBusy      → reject if any block overlaps [start, end)
      • Fallback: events().list→ reject if any event overlaps [start, end)
    """
    # ---- imports / aliases (module-level imports already exist) -------------
    # relies on: isoparse, datetime, timedelta, date, time as _time, pytz as _TZMOD
    def _dbg(msg: str) -> None:
        try: debug_print(msg)
        except Exception: pass

    # ---- small helpers ------------------------------------------------------
    def _as_utc(dt):
        return dt if dt.tzinfo else _TZMOD.UTC.localize(dt)

    def _parse_iso_utc(s: str):
        s2 = s.replace("Z", "+00:00")
        return _as_utc(isoparse(s2)).astimezone(_TZMOD.UTC)

    def _overlap_half_open(a_start, a_end, b_start, b_end) -> bool:
        # Real overlap only; touching at the edge is OK.
        return (a_start < b_end) and (b_start < a_end)

    def _opt_time(val, default_h=None, default_m=0):
        """Coerce env/global value to datetime.time or None."""
        if val is None:
            return None if default_h is None else _time(default_h, default_m)
        if isinstance(val, _time):
            return val
        s = str(val).strip()
        if not s:
            return None if default_h is None else _time(default_h, default_m)
        if ":" in s:
            hh, mm = (s.split(":", 1) + ["0"])[:2]
        else:
            hh, mm = s, "0"
        try:
            return _time(max(0, min(23, int(hh))), max(0, min(59, int(mm))))
        except Exception:
            return None if default_h is None else _time(default_h, default_m)

    # ---- normalize requested window (UTC aware) -----------------------------
    try:
        start_dt = _parse_iso_utc(start_iso)
        end_dt   = _parse_iso_utc(end_iso)
    except Exception as e:
        _dbg(f"is_time_slot_available: ❌ invalid start/end iso → {e}")
        return False
    if end_dt <= start_dt:
        _dbg("is_time_slot_available: ❌ end <= start")
        return False

    # ---- clinic policy (local tz) -------------------------------------------
    tz_name = (globals().get("CLINIC_TZ") or globals().get("LOCAL_TZ") or "America/Chicago")
    try:
        tz_local = _TZMOD.timezone(tz_name)
    except Exception:
        tz_local = _TZMOD.timezone("America/Chicago")

    s_loc = start_dt.astimezone(tz_local)
    e_loc = end_dt.astimezone(tz_local)
    now_loc = datetime.now(tz_local)

    # Past only if slot fully ended
    if e_loc <= now_loc:
        _dbg("is_time_slot_available: ⛔ blocked by clinic policy → fully_past")
        return False

    # Working days
    try:
        wd_src = globals().get("WORKING_DAYS", {0,1,2,3,4})
        WORKING_DAYS = set(int(x) for x in (wd_src if isinstance(wd_src, (list,set,tuple,set)) else [0,1,2,3,4]))
    except Exception:
        WORKING_DAYS = {0,1,2,3,4}
    if s_loc.weekday() not in WORKING_DAYS:
        _dbg(f"is_time_slot_available: ⛔ non_working_day (weekday={s_loc.weekday()})")
        return False

    # Working hours (allow edge touch via half-open)
    WSTART = int(globals().get("WORKING_HOURS_START", globals().get("WORKIN_HOURS_START", 8)))
    WEND   = int(globals().get("WORKING_HOURS_END", 17))
    if not (_time(WSTART, 0) <= s_loc.time() and e_loc.time() <= _time(WEND, 0)):
        _dbg(f"is_time_slot_available: ⛔ outside_hours ({WSTART}:00–{WEND}:00)")
        return False

    # Lunch (no real overlap with [LUNCH_START, LUNCH_END))
    LUNCH_START = _opt_time(globals().get("LUNCH_BREAK_START"))
    LUNCH_END   = _opt_time(globals().get("LUNCH_BREAK_END"))
    if LUNCH_START and LUNCH_END:
        # Compare as times on the same local day; only real intersection blocks.
        if (s_loc.time() < LUNCH_END) and (e_loc.time() > LUNCH_START):
            _dbg("is_time_slot_available: ⛔ lunch_overlap")
            return False

    # ---- Google Calendar checks (no padding; exact [start, end) semantics) --
    try:
        service = build("calendar", "v3", credentials=creds)
    except Exception as e:
        _dbg(f"is_time_slot_available: ❌ google client build error → {e}")
        return False  # fail closed

    tmin = start_dt.isoformat().replace("+00:00", "Z")
    tmax = end_dt.isoformat().replace("+00:00", "Z")

    # 1) FreeBusy primary
    try:
        fb = service.freebusy().query(body={
            "timeMin": tmin,
            "timeMax": tmax,
            "items": [{"id": calendar_id}],
            "timeZone": "UTC",
        }).execute()
        blocks = (fb.get("calendars", {}).get(calendar_id, {}) or {}).get("busy", []) or []
        if blocks:
            for b in blocks:
                bs = _parse_iso_utc(b["start"])
                be = _parse_iso_utc(b["end"])
                if _overlap_half_open(start_dt, end_dt, bs, be):
                    _dbg(f"is_time_slot_available: 🚫 FreeBusy overlap {bs.isoformat()}→{be.isoformat()}")
                    return False
        _dbg("is_time_slot_available: ✅ FreeBusy clear")
    except Exception as e:
        _dbg(f"is_time_slot_available: ⚠️ FreeBusy error → {e}; will try events().list")

    # 2) events().list fallback
    try:
        evs = service.events().list(
            calendarId=calendar_id,
            timeMin=tmin,
            timeMax=tmax,
            singleEvents=True,
            showDeleted=False,
            orderBy="startTime",
            maxResults=250,
        ).execute().get("items", [])

        for ev in evs:
            if ev.get("status") == "cancelled":   # skip cancelled
                continue
            if ev.get("transparency") == "transparent":  # free-time holds
                continue
            # derive UTC bounds from dateTime/date
            s_raw = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date")
            e_raw = ev.get("end",   {}).get("dateTime") or ev.get("end",   {}).get("date")
            if not (s_raw and e_raw):
                # malformed/all-day → be conservative and block
                _dbg("is_time_slot_available: ⚠️ malformed/all-day event → blocking")
                return False
            es = _parse_iso_utc(s_raw if "T" in s_raw or "Z" in s_raw else s_raw + "T00:00:00Z")
            ee = _parse_iso_utc(e_raw if "T" in e_raw or "Z" in e_raw else e_raw + "T00:00:00Z")
            if _overlap_half_open(start_dt, end_dt, es, ee):
                _dbg(f"is_time_slot_available: 🚫 events overlap {es.isoformat()}→{ee.isoformat()} title='{ev.get('summary','')}'")
                return False

        _dbg("is_time_slot_available: ✅ No overlaps via events().list")
    except Exception as e:
        _dbg(f"is_time_slot_available: ❗ events().list error → {e}; fail-closed")
        return False

    _dbg("is_time_slot_available: ✅ Slot FREE (final)")
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
                # ✨ num_digits=1 so the menu can accept a single DTMF digit as well
                gather = make_gather(prompt, hints=hints, num_digits=1) if hints else make_gather(prompt, num_digits=1)
            except Exception:
                # Very defensive fallback
                gather = make_gather("Sorry, I didn’t hear anything. Please try again.", num_digits=1)
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

        lower = speech_result.lower()
        print(f"📢 intent :speech_result: {lower.strip()}")

        # --- New: handle keypad selection 1..5 (or literal spoken "1".."5") first ---
        choice = None
        if dtmf_digits and len(dtmf_digits) == 1 and dtmf_digits in "12345":
            choice = dtmf_digits
        elif lower.strip() in {"1", "2", "3", "4", "5"}:
            choice = lower.strip()

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
                doctor_list = ", ".join(googleid_dr_name_map.values())
                prompt = (
                    f"Great! Let's schedule your appointment. Here is the list of doctors: {doctor_list}. "
                    "Please say the name of the doctor you want to book with."
                )
                gather = make_gather(prompt, hints=", ".join(googleid_dr_name_map.values()))
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
                doctor_list = ", ".join(doctor_names[:-1]) + ", or " + doctor_names[-1] if len(doctor_names) > 1 else doctor_names[0]
                prompt = (
                    f"Sure, I can help you cancel your appointment. "
                    f"We currently have the following doctors: {doctor_list}. "
                    f"Please say the name of the doctor you had booked with."
                )
                gather = make_gather(prompt, hints=", ".join(doctor_names))
                resp.append(gather)
                return str(resp)

            if choice == "3":
                # ✅ Reschedule (change → cancel then rebook)
                print("🔁 DTMF=3 → reschedule (cancel then rebook)")
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
                gather = make_gather(prompt, hints=", ".join(doctor_names))
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
                # Let the update_cc stage/procedure run
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

        # 🚫 Ignore junk or greeting phrases commonly returned by Twilio
        junk_inputs = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening", "yo", "test", "1", "yes", "no"}
        if not lower.strip() or lower.strip() in junk_inputs:
            print(f"⛔ Ignored junk input: '{lower}' — re-prompting without response")

            # ⬇️ CHANGED: use make_gather (same behavior, cleaner + consistent)
            # ✨ Updated to show both voice and keypad options, allow 1 digit
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

        # ✅ Rescheduling intent
        elif any(word in lower for word in ["change", "move", "reschedule"]):
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

        # ✅ Update credit card (voice intent)
        elif any(kw in lower for kw in [
            "update card", "update credit card", "update my card", "update cc",
            "change card", "new card", "update payment", "update payment method",
            "update billing", "change billing", "update card number",
            "update visa", "update mastercard", "update american express", "update amex"
        ]):
            print("💳 Intent to update credit card detected → starting CC update flow")

            # Flag the CC update path and start identity verification at collect_phone
            session_data.setdefault(call_sid, {})
            session_data[call_sid].update({
                "stage": "update_cc",
                "cc_update": {"active": True},   # used later to route to collect_cc after DOB
                "retry_booking": 0
            })

            # For the update_cc stage, we redirect so its handler can do the phone gather
            try:
                resp.redirect(url_for("voice"))
            except Exception:
                resp.redirect("/voice")
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
            # ✨ Re-prompt includes both voice choices and digits, allows single DTMF
            gather = make_gather(
                "Sorry, I didn’t catch that. "
                "Say 'book appointment' or press 1, "
                "'cancel appointment' or press 2, "
                "'change appointment' or press 3, "
                "'update credit card' or press 4, "
                "or 'leave voicemail' or press 5.",
                hints="book,cancel,change,reschedule,update,voicemail",
                num_digits=1
            )
            resp.append(gather)
            return str(resp)


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
                "Before we update your card, please say your birth date, or enter MMDDYYYY then press pound."
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

        # ✅ Ensure the regex module alias exists at runtime (prevents UnboundLocalError)
        try:
            _re  # type: ignore[name-defined]
        except NameError:
            import re as _re  # single import; never reassign inside functions

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
        import re as _re
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
        import re as _re
        # ------------------------------------------------------------------
        # Prompts
        # ------------------------------------------------------------------
        TIME_PROMPT_SHORT = (
            "That doesn't sound like a valid date or time. "
            "Please say it again, for example, 'September 12 at 10 AM'."
        )
        PROMPT_NEED_BOTH = "Please say the date and the time, for example, 'September 12 at 10 AM'."
        PROMPT_NEED_DATE = "I didn't hear the date. Please include it, for example, 'September 12 at 10 AM'."
        PROMPT_NEED_TIME = "I didn't hear the time. Please include it, for example, 'September 12 at 10 AM'."

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
            try: resp.redirect(url_for("voice"))
            except Exception: resp.redirect("/voice")
            return str(resp)
        session_data[call_sid].pop("silence_time", None)

        # ------------------------------------------------------------------
        # Tiny helpers (local to this stage)
        # ------------------------------------------------------------------
        def _has_time_token(s: str) -> bool:
            """Heuristic to decide if a time was said."""
            s = (s or "").lower()
            return (
                ("am" in s) or ("pm" in s) or (":" in s)
                or ("o'clock" in s) or ("oclock" in s)
                or (_re.search(r"\b\d{1,2}\s*(am|pm)\b", s) is not None)
                or (_re.search(r"\b\d{3,4}\b", s) is not None)  # 930, 1030
                or ("noon" in s) or ("midnight" in s)
            )

        def _has_date_token(s: str) -> bool:
            """Heuristic to decide if a date was said."""
            s = (s or "").lower()
            months = ("january","february","march","april","may","june","july",
                      "august","september","october","november","december",
                      "jan","feb","mar","apr","may","jun","jul","aug","sep","sept","oct","nov","dec")
            if any(m in s for m in months): return True
            if "/" in s or "-" in s: return True  # 9/12, 09-12
            weekdays = ("monday","tuesday","wednesday","thursday","friday","saturday","sunday",
                        "mon","tue","tues","wed","thu","thur","thurs","fri","sat","sun")
            if any(w in s for w in weekdays): return True
            if _re.search(r"\b\d{1,2}\b", s): return True  # day of month spoken alone
            return False

        def _extract_day_time(s: str) -> tuple:
            """
            Normalize and split into (day_str, time_str).
            Accepts forms like:
              'September 11 at 10 am'
              'Thu, September 11, 10:30 am'
              '9/11 at 10'
            """
            if not s: return ("", "")

            # Normalize AM/PM variants & punctuation; keep ':' for times
            s = _re.sub(r"\b(a\s*\.?\s*m\.?)\b", "am", s, flags=_re.IGNORECASE)
            s = _re.sub(r"\b(p\s*\.?\s*m\.?)\b", "pm", s, flags=_re.IGNORECASE)
            s = _re.sub(r"[!?]+\s*$", "", s)
            # Remove extra commas/periods but keep colons
            s = _re.sub(r"[;,]+", " ", s)
            s = _re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", s, flags=_re.IGNORECASE)  # 11th→11
            s = _re.sub(r"\s+", " ", s).strip()

            # Map "noon"/"midnight"
            s_low = s.lower()
            s_low = s_low.replace(" at noon", " at 12 pm").replace(" noon", " 12 pm")
            s_low = s_low.replace(" at midnight", " at 12 am").replace(" midnight", " 12 am")

            # If there's an explicit " at " split, use it
            if " at " in s_low:
                day, timep = s_low.split(" at ", 1)
                return (day.strip().rstrip(","), timep.strip())

            # Otherwise, try to locate a time token and split around it
            m = _re.search(r"\b(\d{1,2}(:\d{2})?\s*(am|pm)?)\b", s_low)
            if m:
                timep = m.group(1)
                day = s_low[:m.start()].strip().rstrip(",")
                # If day ended up empty (e.g., "10 am"), leave it ""
                return (day, timep)

            # Last resort: maybe compact "930", "1000"
            m2 = _re.search(r"\b(\d{3,4})\b", s_low)
            if m2:
                t = m2.group(1)
                if len(t) == 3:  # "930" → "9:30"
                    timep = f"{int(t[0]):d}:{t[1:]}"
                else:            # "1030" → "10:30"
                    timep = f"{int(t[:-2]):d}:{t[-2:]}"
                day = s_low[:m2.start()].strip().rstrip(",")
                return (day, timep)

            return ("", "")

        def _build_slot(day_str: str, time_str: str) -> tuple:
            """
            Build (start_iso_utc, end_iso_utc) using clinic TZ and duration.
            If year not said, force current year (do NOT auto-roll to next year).
            """
            tz_name = (globals().get("CLINIC_TZ") or globals().get("LOCAL_TZ") or "America/Chicago")
            try:
                tz_local = _pytz.timezone(tz_name)
            except Exception:
                tz_local = _pytz.timezone("America/Chicago")

            # Duration (allowed set)
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

            # Clean pieces
            d = (day_str or "").strip()
            t = (time_str or "").strip()
            if not d or not t:
                raise ValueError("missing date or time")

            # Ensure common formats
            t = _re.sub(r"\s*(am|pm)\b", r" \1", t)  # "10am" → "10 am"
            t = t.replace(" o'clock", "")

            # Compose a single phrase: "<day> at <time>"
            combined = f"{d} at {t}"

            # Parse with a default baseline (today in local tz)
            today = _date_local.today()
            default_base = datetime(today.year, today.month, today.day, 9, 0, 0)

            parsed = _dtparse(combined, default=default_base, dayfirst=False, fuzzy=True)
            # Attach/normalize tz
            if parsed.tzinfo is None:
                parsed = tz_local.localize(parsed)
            else:
                parsed = parsed.astimezone(tz_local)

            # If the caller did NOT say a year, force current year
            said_year = bool(_re.search(r"\b\d{4}\b", combined))
            if not said_year:
                parsed = parsed.replace(year=today.year)

            start_local = parsed
            end_local   = start_local + timedelta(minutes=dur)

            start_utc = start_local.astimezone(_pytz.UTC).isoformat().replace("+00:00", "Z")
            end_utc   = end_local.astimezone(_pytz.UTC).isoformat().replace("+00:00", "Z")
            return (start_utc, end_utc)

        def _friendly(dt_utc_iso: str) -> str:
            """Make a friendly local label for prompts."""
            tz_name = (globals().get("CLINIC_TZ") or globals().get("LOCAL_TZ") or "America/Chicago")
            try:
                dt_utc = datetime.fromisoformat(dt_utc_iso.replace("Z", "+00:00"))
                dt_loc = dt_utc.astimezone(_pytz.timezone(tz_name))
                try:
                    return dt_loc.strftime("%A, %B %-d at %-I:%M %p")
                except Exception:
                    return dt_loc.strftime("%A, %B %d at %I:%M %p").replace(" 0", " ")
            except Exception:
                return ""

        # ------------------------------------------------------------------
        # Extract (day, time)
        # ------------------------------------------------------------------
        day_part, time_part = _extract_day_time(raw)
        debug_print(f"ask_time_date: 📆 Extracted → Day: {day_part or '(none)'}, Time: {time_part or '(none)'}")

        need_date = not _has_date_token(day_part)
        need_time = not _has_time_token(time_part)

        if need_date or need_time:
            if need_date and need_time: prompt = PROMPT_NEED_BOTH
            elif need_date:              prompt = PROMPT_NEED_DATE
            else:                        prompt = PROMPT_NEED_TIME
            session_data[call_sid]["retry_time"] = session_data[call_sid].get("retry_time", 0) + 1
            if session_data[call_sid]["retry_time"] >= 3:
                resp.say(gpt_speak("Sorry, I still couldn't understand the date and time. Please try again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)
            resp.append(make_gather(prompt))
            return str(resp)

        # ------------------------------------------------------------------
        # Build concrete UTC slot (start/end)
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
        # Past-time guard (absolute): only if the slot has fully ended
        # ------------------------------------------------------------------
        try:
            now_utc  = datetime.utcnow().replace(tzinfo=_pytz.UTC)
            end_dt   = datetime.fromisoformat(appointment_end.replace("Z", "+00:00")).astimezone(_pytz.UTC)
            if end_dt <= now_utc:
                debug_print("ask_time_date: 🕒 requested time is in the past → suggest next slots AFTER requested time")
                # Ask for the next 3 free slots AFTER the requested end
                try:
                    alts = get_next_available_slots(
                        calendar_id,
                        creds,
                        from_start_iso=appointment_end,
                        limit=3
                    ) or []
                except Exception as e:
                    debug_print(f"ask_time_date: ⚠️ get_next_available_slots error → {e}")
                    alts = []

                if alts:
                    options = " or ".join([a.get("friendly","") for a in alts if a.get("friendly")])
                    prompt = f"That time has already passed. Would you like {options}?" if options else \
                             "That time has already passed. Please say another date and time."
                else:
                    prompt = "That time has already passed. Please say another date and time."

                resp.append(make_gather(prompt))
                return str(resp)
        except Exception as e:
            debug_print(f"ask_time_date: ⚠️ past-time guard error → {e}")

        # ------------------------------------------------------------------
        # Availability check for the exact requested slot
        # ------------------------------------------------------------------
        debug_print(f"ask_time_date: 👨‍⚕️ Checking calendar → {calendar_id}")
        try:
            slot_available = is_time_slot_available(calendar_id, appointment_start, appointment_end, creds)
        except Exception as e:
            debug_print(f"ask_time_date: ⚠️ Availability check error → {e}")
            slot_available = False

        if not slot_available:
            debug_print("ask_time_date: ❌ Slot not available → suggesting alternatives")
            # Next 3 free AFTER the requested end (absolute)
            try:
                alts = get_next_available_slots(
                    calendar_id,
                    creds,
                    from_start_iso=appointment_end,
                    limit=3
                ) or []
            except Exception as e:
                debug_print(f"ask_time_date: ⚠️ get_next_available_slots error → {e}")
                alts = []

            if alts:
                options = " or ".join([a.get("friendly","") for a in alts if a.get("friendly")])
                prompt = f"That time is not available. Would you like {options}?" if options \
                         else "That time is not available. Please say another date and time."
            else:
                prompt = "That time is not available. Please say another date and time."

            resp.append(make_gather(prompt))
            return str(resp)

        # ------------------------------------------------------------------
        # Slot free → persist and advance
        # ------------------------------------------------------------------
        debug_print("ask_time_date: ✅ Slot free → proceed to confirmation flow")
        session_data[call_sid]["appointment_time"] = {
            "start": appointment_start,
            "end": appointment_end
        }

        # Keep the flow simple/independent: go to confirm stage.
        session_data[call_sid]["stage"] = "book_appt_confirm"
        try: resp.redirect(url_for("voice"))
        except Exception: resp.redirect("/voice")
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
                #from flask import url_for
                resp.redirect(url_for("voice"))
            except Exception:
                resp.redirect("/voice")
            return str(resp)

        # We heard something → clear silence counter
        session_data[call_sid].pop("silence_last_name", None)

        # 🧽 Clean & normalize (keep inner spaces; strip punctuation except apostrophe/hyphen)
        #import string
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
        #       (2) Expiration (MMYY or MMYYYY) → saved 'MM/YY' (must be current/future)
        #       (3) CVV (3–4 digits)
        #   - Stores under session_data[call_sid]["customer"]:
        #       cc_number, cc_exp, cc_cvv, cc_name
        #   - On success:
        #       - if cc_update.active → stage=update_customer_cc
        #       - else → stage=book_appt_confirm
        # Notes:
        #   - Uses make_gather() (speech + DTMF). DTMF preferred; speech digits supported.
        #   - Requires phone **E.164** and DOB before collecting CC.
        #   - Never store full PAN/CVV in production (tokenize with Twilio <Pay> instead).
        # ----------------------------------------------------------------------

        # --- helpers ----------------------------------------------------------
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
            """Map spoken words to digits; supports 'double'/'triple' and common homophones."""
            if not raw:
                return ""
            words = raw.lower().strip().replace("-", " ").replace(",", " ").replace(".", " ").split()
            m = {
                "zero":"0","oh":"0","o":"0",
                "one":"1","two":"2","to":"2","too":"2",
                "three":"3","four":"4","for":"4",
                "five":"5","six":"6","seven":"7",
                "eight":"8","ate":"8","nine":"9"
            }
            out = []; i = 0
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

        def _reprompt(prompt: str, hints: str = "") -> str:
            """
            Speech/DTMF reprompt with retry cap (separate from silence). Returns TwiML string.
            Always append a <Gather> and redirect, then return str(resp).
            """
            session_data[call_sid]["retry_cc"] = session_data[call_sid].get("retry_cc", 0) + 1
            if session_data[call_sid]["retry_cc"] >= 5:
                debug_print("collect_cc: ⛔ max CC retries. Ending.")
                resp.say(gpt_speak("Sorry, we’re having trouble collecting your card details. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # If we've escalated to DTMF-only, force keypad
            if session_data[call_sid].get("enforce_dtmf_cc"):
                resp.append(make_gather_dtmf(prompt_text=prompt, num_digits=None))
            else:
                # Longer timeouts so speech has time to finish the digits
                resp.append(make_gather(
                    prompt,
                    hints=hints,
                    input="speech dtmf",
                    timeout=25,          # overall wait before user starts
                    speech_timeout="10", # wait for up to 10s of speech chunk
                    finish_on_key="#",
                    barge_in=True,
                    action="/voice",
                ))
            resp.redirect("/voice")
            return str(resp)

        # --- session buckets --------------------------------------------------
        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("customer", {})
        customer = session_data[call_sid]["customer"]

        # Are we here via the update_cc path?
        is_cc_update = bool(session_data.get(call_sid, {}).get("cc_update", {}).get("active"))

        # 🔒 Require phone **E.164** + DOB before CC (no phone10 fallback)
        phone_e164 = (customer.get("phone_e164") or session_data[call_sid].get("phone_e164") or "").strip()
        if not phone_e164 or not customer.get("dob"):
            debug_print(f"collect_cc: ❌ Missing E.164 phone or DOB → redirecting (phone_e164='{phone_e164}', dob_present={bool(customer.get('dob'))})")
            session_data[call_sid]["stage"] = "collect_phone" if not phone_e164 else "collect_dob"
            prompt_txt = (
                "Before payment details, please provide your phone number including area code."
                if not phone_e164 else
                "Before payment details, please provide your date of birth. You can say it, or enter MMDDYYYY then press pound."
            )
            resp.append(make_gather(prompt_txt, hints="zero one two three four five six seven eight nine", action="/voice"))
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
                resp.append(make_gather_dtmf(prompt_text=prompt, num_digits=None))
            else:
                resp.append(make_gather(prompt, hints=hints, input="speech dtmf",
                                        timeout=25, speech_timeout="10", finish_on_key="#", action="/voice"))
            resp.redirect("/voice")
            return str(resp)

        # Clear silence counter once we hear something
        session_data[call_sid].pop("silence_cc", None)

        # Prefer DTMF; otherwise convert spoken words to digits
        def get_digits() -> str:
            if enforce_dtmf:
                if not dtmf_digits:
                    return ""
                return _re.sub(r"\D", "", dtmf_digits)
            if dtmf_digits:
                return _re.sub(r"\D", "", dtmf_digits)
            return _re.sub(r"\D", "", normalize_spoken_digits(speech_text))

        # -------------------------------
        # Step 1: Card Number (13–19)
        # -------------------------------
        if cc_step == 1:
            new_digits = get_digits()

            # Handle "expect last digit" path (after spoken 15 digits)
            if cc_expect_last_digit:
                if len(new_digits) == 1:
                    digits = (cc_partial or "") + new_digits
                    debug_print(f"collect_cc: 🔚 appended last digit → '{digits}'")
                else:
                    debug_print("collect_cc: ℹ️ expected 1 digit, got fresh entry → clearing partial")
                    session_data[call_sid]["cc_partial"] = ""
                    session_data[call_sid]["cc_expect_last_digit"] = False
                    digits = new_digits
            else:
                digits = new_digits

            if not digits:
                debug_print("collect_cc: ℹ️ no digits heard → reprompt")
                return _reprompt(
                    "Please enter your card number now, then press pound.",
                    hints="zero one two three four five six seven eight nine double triple"
                )

            if len(digits) > 19:
                digits = digits[:19]

            # If we heard exactly 15 digits via speech (not DTMF), ask for final digit
            if not enforce_dtmf and not dtmf_digits and len(digits) == 15:
                session_data[call_sid]["cc_partial"] = digits
                session_data[call_sid]["cc_expect_last_digit"] = True
                debug_print(f"collect_cc: 🧩 Heard 15 digits '{digits}'; asking for the last single digit")
                return _reprompt(
                    "I heard fifteen digits. Please say or type the last single digit now, then press pound.",
                    hints="zero one two three four five six seven eight nine"
                )

            # Full validation
            if not (13 <= len(digits) <= 19) or not luhn_check(digits):
                # Count speech failures to decide DTMF enforcement
                if not dtmf_digits:  # speech path
                    session_data[call_sid]["cc_speech_tries"] += 1
                escalate = (session_data[call_sid]["cc_speech_tries"] >= 2 and not dtmf_digits)

                debug_print(f"collect_cc: ❌ Invalid card number: '{digits}' (len={len(digits)}), escalate={escalate}")

                # Always clear partial/expect flags on invalid
                session_data[call_sid]["cc_partial"] = ""
                session_data[call_sid]["cc_expect_last_digit"] = False

                if escalate:
                    # Force DTMF-only from now on for PAN entry
                    session_data[call_sid]["enforce_dtmf_cc"] = True
                    resp.append(make_gather_dtmf("That number didn’t sound clear. Please TYPE the full card number now, then press pound.", num_digits=None))
                    resp.redirect("/voice")
                    return str(resp)
                else:
                    return _reprompt(
                        "That card number doesn't look right. Please re-enter the full card number, then press pound.",
                        hints="zero one two three four five six seven eight nine double triple"
                    )

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
                resp.append(make_gather_dtmf(prompt_text=prompt, num_digits=None))
            else:
                resp.append(make_gather(prompt, hints="zero one two three four five six seven eight nine",
                                        input="speech dtmf", timeout=25, speech_timeout="10",
                                        finish_on_key="#", action="/voice"))
            resp.redirect("/voice")
            return str(resp)

        # -------------------------------
        # Step 2: Expiration (MMYY/MMYYYY, must be current/future)
        # -------------------------------
        if cc_step == 2:
            digits = get_digits()
            if len(digits) not in (4, 6):
                debug_print(f"collect_cc: ❌ Exp bad length: '{digits}'")
                return _reprompt(
                    "Please enter the expiration as four digits MMYY, then press pound.",
                    hints="zero one two three four five six seven eight nine"
                )

            mm = int(digits[:2]) if digits[:2].isdigit() else 0
            yy = digits[2:]
            if not (1 <= mm <= 12):
                debug_print(f"collect_cc: ❌ Invalid month: '{digits}'")
                return _reprompt(
                    "The month must be 01 through 12. Please re-enter expiration MMYY, then press pound.",
                    hints="zero one two three four five six seven eight nine"
                )

            # Normalize year to 2-digit (handle MMYYYY too)
            if len(yy) == 4:
                yy = yy[-2:]

            # Reject past month
            from datetime import datetime as _Datetime
            now = _Datetime.now()
            exp_year = 2000 + int(yy)
            exp_cmp  = exp_year * 100 + mm
            now_cmp  = now.year * 100 + now.month
            if exp_cmp < now_cmp:
                debug_print(f"collect_cc: ❌ Expired card: {mm:02d}/{yy}")
                return _reprompt(
                    "That card appears expired. Please enter a valid expiration date as MMYY, then press pound.",
                    hints="zero one two three four five six seven eight nine"
                )

            customer["cc_exp"] = f"{mm:02d}/{yy}"
            session_data[call_sid]["cc_step"] = 3
            debug_print(f"collect_cc: ✅ Saved expiration {customer['cc_exp']}")

            prompt = "Great. Finally, enter the three or four digit security code, then press pound."
            if session_data[call_sid].get("enforce_dtmf_cc"):
                resp.append(make_gather_dtmf(prompt_text=prompt, num_digits=None))
            else:
                resp.append(make_gather(prompt, hints="zero one two three four five six seven eight nine",
                                        input="speech dtmf", timeout=25, speech_timeout="10",
                                        finish_on_key="#", action="/voice"))
            resp.redirect("/voice")
            return str(resp)

        # -------------------------------
        # Step 3: CVV (3–4 digits)
        # -------------------------------
        if cc_step == 3:
            digits = get_digits()

            # If speech path produced nothing meaningful, try parsing the raw speech again (more tolerant)
            if (not digits) and (not dtmf_digits) and speech_text:
                digits = _re.sub(r"\D", "", normalize_spoken_digits(speech_text))

            # Validate
            if not (3 <= len(digits) <= 4 and digits.isdigit()):
                # Count speech failures and escalate to DTMF-only after first miss
                if not dtmf_digits:
                    session_data[call_sid]["cc_speech_tries"] += 1
                enforce_now = session_data[call_sid]["cc_speech_tries"] >= 1 and not dtmf_digits

                debug_print(f"collect_cc: ❌ Invalid CVV '{digits or speech_text}' "
                            f"len={len(digits)} enforce_dtmf={enforce_now}")

                # Pin the flag so next prompt uses keypad-only
                if enforce_now:
                    session_data[call_sid]["enforce_dtmf_cc"] = True

                # Reprompt (DTMF if escalated)
                session_data[call_sid]["retry_cc"] += 1
                if session_data[call_sid]["retry_cc"] >= 5:
                    resp.say(gpt_speak("Sorry, we’re having trouble collecting your card details. Please call again later."), VOICE)
                    resp.hangup()
                    session_data.pop(call_sid, None)
                    return str(resp)

                prompt = "That security code doesn't sound right. Please enter the three or four digit code, then press pound."
                g = prompt_for_value(prompt_text=prompt, dtmf_only=session_data[call_sid].get("enforce_dtmf_cc", False))
                if g: resp.append(g)
                resp.redirect("/voice")
                return str(resp)

            # ✅ Good CVV
            customer["cc_cvv"] = digits
            if not customer.get("cc_name"):
                name = customer.get("name") or " ".join(p for p in [customer.get("first_name"), customer.get("last_name")] if p)
                customer["cc_name"] = (name or "").strip() or None

            debug_print(f"collect_cc: ✅ Saved CVV '{digits}'; cc_name='{customer.get('cc_name')}'")

            # Clear step tracker and branch based on origin
            session_data[call_sid].pop("cc_step", None)
            next_stage = "update_customer_cc" if is_cc_update else "book_appt_confirm"
            session_data[call_sid]["stage"] = next_stage
            debug_print(f"collect_cc: ➡️ Auto-advancing to {next_stage}")

            # Move to next stage now
            resp.redirect("/voice")
            return str(resp)



        elif stage == "book_appt_confirm":
            debug_print("book_appt_confirm: 📍 Stage entered")

            # ----------------------------------------------------------------------
            # 📌 We DO NOT ignore speech here anymore.
            #     - Accept 'confirm/yes/book' (or DTMF 1)
            #     - Accept 'change/no/different' (or DTMF 2)
            #     - Accept 'cancel' (or DTMF 3)
            #     - Small silence/bad-input retry budget
            # ----------------------------------------------------------------------

            # ---- Doctor info --------------------------------------------------------
            doctor_id = session_data[call_sid].get("doctor_id")
            if not doctor_id:
                debug_print("book_appt_confirm: ❌ missing doctor_id → choose_doctor")
                session_data[call_sid]["stage"] = "choose_doctor"
                resp.append(make_gather("Which doctor would you like to see?"))
                return str(resp)

            doctor_name = googleid_dr_name_map.get(doctor_id, "the doctor")
            debug_print(f"book_appt_confirm: doctor_id={doctor_id} name={doctor_name}")

            # ---- Appointment time (need start; compute end if missing) --------------
            appt_payload      = session_data[call_sid].get("appointment_time", {}) or {}
            appointment_start = appt_payload.get("start")
            appointment_end   = appt_payload.get("end")
            debug_print(f"book_appt_confirm: utc_start={appointment_start} utc_end={appointment_end}")

            if not appointment_start:
                debug_print("book_appt_confirm: ❌ missing appointment_start")
                resp.say(gpt_speak("Appointment time is missing. Goodbye!"), VOICE)
                resp.hangup()
                return str(resp)

            # Format local friendly time (uses CLINIC_TZ or America/Chicago)
            tz_name = (globals().get("CLINIC_TZ") or globals().get("LOCAL_TZ") or "America/Chicago")
            try:
                tz = _pytz.timezone(tz_name)
            except Exception:
                tz = _pytz.timezone("America/Chicago")

            try:
                dt_utc   = datetime.fromisoformat(appointment_start.replace("Z", "+00:00"))
                dt_local = dt_utc.astimezone(tz)
                try:
                    formatted_time = dt_local.strftime("%A, %B %-d at %-I:%M %p")  # Unix-like
                except Exception:
                    formatted_time = dt_local.strftime("%A, %B %d at %I:%M %p").replace(" 0", " ")  # Windows fallback
            except Exception as e:
                debug_print(f"book_appt_confirm: time format error → {e}")
                resp.say(gpt_speak("Sorry, we couldn't confirm the appointment time."), VOICE)
                resp.hangup()
                return str(resp)

            # Compute end if missing (duration from globals, default 30m)
            if not appointment_end:
                try:
                    dur_min = None
                    for k in ("APPOINTMENT_DURATION_MINUTES", "SESSION_TIME", "SESSIUON_TIME"):
                        v = globals().get(k)
                        if v:
                            try:
                                dur_min = int(v)
                                break
                            except Exception:
                                pass
                    if dur_min not in (15, 30, 45, 60):
                        dur_min = 30
                    end_dt = dt_utc + timedelta(minutes=dur_min)
                    appointment_end = end_dt.astimezone(_pytz.UTC).isoformat().replace("+00:00", "Z")
                    debug_print(f"book_appt_confirm: computed utc_end={appointment_end} (duration={dur_min}m)")
                except Exception as e:
                    debug_print(f"book_appt_confirm: ❌ failed computing end time → {e}")
                    resp.say(gpt_speak("Sorry, we couldn't confirm the appointment time."), VOICE)
                    resp.hangup()
                    return str(resp)

            # ---- Customer info (E.164 + DOB) ---------------------------------------
            customer           = session_data[call_sid].get("customer", {}) or {}
            customer_name      = (customer.get("name") or "").strip()
            first_name         = (customer.get("first_name") or "").strip()
            last_name          = (customer.get("last_name")  or "").strip()
            if not first_name and customer_name:
                parts = customer_name.split()
                first_name = parts[0]
                last_name  = " ".join(parts[1:]) if len(parts) > 1 else ""
            effective_name     = customer_name or " ".join([n for n in (first_name, last_name) if n]).strip()
            customer_address   = (customer.get("address") or "").strip()
            customer_dob       = (customer.get("dob") or "").strip()

            # E.164 normalization (prefer already-stored phone_e164)
            phone_raw           = (customer.get("phone_e164") or customer.get("phone") or "").strip()
            customer_phone_e164 = ""
            if phone_raw.startswith("+") and phone_raw[1:].replace(" ", "").isdigit():
                customer_phone_e164 = "+" + phone_raw[1:].replace(" ", "")
            else:
                try:
                    default_country = (session_data[call_sid].get("phone_country") or globals().get("COUNTRY") or "US").upper()
                    customer_phone_e164 = normalize_phone_e164(phone_raw, default_country) or ""
                    if not customer_phone_e164:
                        alt = "EG" if default_country != "EG" else "US"
                        customer_phone_e164 = normalize_phone_e164(phone_raw, alt) or ""
                except Exception:
                    customer_phone_e164 = ""

            if not customer_phone_e164:
                debug_print("book_appt_confirm: ❌ missing/invalid E.164 phone → collect_phone")
                session_data[call_sid]["stage"] = "collect_phone"
                resp.append(make_gather("Before we confirm your appointment, please provide your phone number."))
                return str(resp)

            if not customer_dob:
                debug_print("book_appt_confirm: ❌ missing DOB → collect_dob")
                session_data[call_sid]["stage"] = "collect_dob"
                resp.append(make_gather(
                    "Before we confirm, please say your date of birth, for example, 'July 3 1990'. "
                    "You can also enter two digits for month, two for day, and four for year, then press #."
                ))
                return str(resp)

            # ---- Upsert customer (best-effort) --------------------------------------
            try:
                init_db()
                insert_customer(
                    phone=customer_phone_e164,
                    dob=customer_dob,
                    first_name=first_name,
                    last_name=last_name,
                    address=customer_address,
                    cc_name=(customer.get("cc_name") or effective_name or ""),
                    cc_number=(customer.get("cc_number") or ""),
                    cc_exp=(customer.get("cc_exp") or ""),
                    cc_cvv=(customer.get("cc_cvv") or "")
                )
                debug_print("book_appt_confirm: customers DB → inserted/updated")
            except Exception as e:
                debug_print(f"book_appt_confirm: insert_customer failed → {e}")

            # ---- Availability check (enforces working hours/days/past-time) ---------
            calendar_id = doctor_id
            debug_print(f"book_appt_confirm: 🔎 availability cal={calendar_id} {appointment_start}→{appointment_end}")
            try:
                slot_free = is_time_slot_available(calendar_id, appointment_start, appointment_end, creds)
            except Exception as e:
                debug_print(f"book_appt_confirm: ⚠️ availability check error → {e}")
                slot_free = False

            # Fast path: auto-confirm callers (e.g., returning patients) when slot is free
            if session_data[call_sid].get("auto_confirm", False) and slot_free:
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
                                "phone_e164": customer_phone_e164,
                                "dob": customer_dob,
                                "call_sid": call_sid,
                            }
                        },
                    }
                    ev = service.events().insert(calendarId=calendar_id, body=event_body, sendUpdates="none").execute()
                    debug_print(f"book_appt_confirm: ✅ Google event created id={ev.get('id')}")
                    resp.say(gpt_speak(f"Your appointment with {doctor_name} is booked for {formatted_time}. See you then!"), VOICE)
                    resp.hangup()
                    session_data.pop(call_sid, None)
                    return str(resp)
                except Exception as e:
                    debug_print(f"book_appt_confirm: ❌ auto-confirm booking error → {e}")
                    # fall through to interactive confirm

            # If slot not free → suggest next options and bounce back to ask_time_date.
            if not slot_free:
                debug_print("book_appt_confirm: ❌ Slot not free → alternatives")
                try:
                    alts = get_next_available_slots(
                        calendar_id,
                        creds,
                        from_start_iso=appointment_end,  # absolute; no +/- 1s
                        limit=3
                    ) or []
                    if alts:
                        options = " or ".join([a.get("friendly","") for a in alts if a.get("friendly")])
                        session_data[call_sid]["stage"] = "ask_time_date"
                        resp.append(make_gather(f"That time is not available. Would you like {options}?"))
                        return str(resp)
                except Exception as e:
                    debug_print(f"book_appt_confirm: ⚠️ get_next_available_slots error → {e}")

                session_data[call_sid]["stage"] = "ask_time_date"
                resp.append(make_gather("That time is not available. Please say another date and time, for example, 'today at 3:30 PM'."))
                return str(resp)

            # ----------------------------------------------------------------------
            # Interactive confirm (speech or DTMF)
            # ----------------------------------------------------------------------
            dtmf   = (request.values.get("Digits") or "").strip()
            speech = (speech_result or "").strip().lower()

            # Silence handling (2 tries max)
            if not (dtmf or speech):
                tries = session_data[call_sid].get("silence_confirm", 0) + 1
                session_data[call_sid]["silence_confirm"] = tries
                if tries > 2:
                    resp.say(gpt_speak("I’m still not hearing anything. Let’s try again later."), VOICE)
                    resp.hangup()
                    session_data.pop(call_sid, None)
                    return str(resp)

                prompt = f"To book {formatted_time} with {doctor_name}, say 'confirm' or press 1. To change, say 'change' or press 2. To cancel, say 'cancel' or press 3."
                g = make_gather(prompt, input="speech dtmf", hints="yes confirm book no change different cancel")
                resp.append(g)
                return str(resp)

            # Map intents
            intent = ""
            if dtmf == "1":
                intent = "confirm"
            elif dtmf == "2":
                intent = "change"
            elif dtmf == "3":
                intent = "cancel"
            else:
                if any(k in speech for k in ("yes","confirm","book","okay","ok")):
                    intent = "confirm"
                elif any(k in speech for k in ("no","change","different","another")):
                    intent = "change"
                elif "cancel" in speech:
                    intent = "cancel"

            # Branch on intent
            if intent == "confirm":
                # Re-check slot just before booking, then create the event.
                try:
                    if not is_time_slot_available(calendar_id, appointment_start, appointment_end, creds):
                        debug_print("book_appt_confirm: ❌ slot became busy at confirm")
                        session_data[call_sid]["stage"] = "ask_time_date"
                        resp.append(make_gather(f"Sorry, {formatted_time} just became unavailable. Please say another time."))
                        return str(resp)

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
                                "phone_e164": customer_phone_e164,
                                "dob": customer_dob,
                                "call_sid": call_sid,
                            }
                        },
                    }
                    ev = service.events().insert(calendarId=calendar_id, body=event_body, sendUpdates="none").execute()
                    google_event_id   = ev.get("id")
                    google_event_link = ev.get("htmlLink")
                    debug_print(f"book_appt_confirm: ✅ Google event created id={google_event_id} link={google_event_link}")

                    # Voice + SMS confirmation
                    msg = f"Your appointment with {doctor_name} has been booked"
                    if formatted_time: msg += f" on {formatted_time}"
                    msg += ". We look forward to seeing you. Goodbye!"
                    debug_print("book_appt_confirm: 🎉 success → speaking confirmation")
                    resp.say(gpt_speak(msg), VOICE)

                    try:
                        sms_text = f"Hi {(effective_name or 'there')}, your appointment with {doctor_name} is confirmed"
                        if formatted_time: sms_text += f" on {formatted_time}"
                        sms_text += ". Thank you for choosing Epic Therapist Clinic."
                        message = client.messages.create(body=sms_text, from_=TWILIO_PHONE_NUMBER, to=customer_phone_e164)
                        debug_print(f"book_appt_confirm: 📩 SMS sent to {customer_phone_e164}, SID={message.sid}")
                    except Exception as e:
                        debug_print(f"book_appt_confirm: ⚠️ SMS failed → {e}")

                    # Cleanup
                    resp.hangup()
                    session_data.pop(call_sid, None)
                    debug_print("book_appt_confirm: ✅ session cleared and call ended")
                    return str(resp)
                except Exception as e:
                    debug_print(f"book_appt_confirm: ❌ booking error → {e}")
                    session_data[call_sid]["stage"] = "ask_time_date"
                    resp.append(make_gather("Sorry, I couldn’t finish the booking. Please say another date and time."))
                    return str(resp)

            if intent == "change":
                session_data[call_sid]["stage"] = "ask_time_date"
                resp.append(make_gather("Okay. Please say the new date and time you prefer."))
                return str(resp)

            if intent == "cancel":
                session_data[call_sid]["stage"] = "cancel_appt_get_date_time"
                resp.append(make_gather("Okay. Tell me the date and time of the appointment you want to cancel."))
                return str(resp)

            # Didn’t recognize answer → reprompt (short budget)
            tries = session_data[call_sid].get("bad_confirm", 0) + 1
            session_data[call_sid]["bad_confirm"] = tries
            if tries > 2:
                resp.say(gpt_speak("Sorry, I’m not getting that. Let’s try again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            resp.append(make_gather(
                f"I heard {speech_result or dtmf}. To confirm {formatted_time} with {doctor_name}, say 'confirm' or press 1. "
                "To change, say 'change' or press 2. To cancel, say 'cancel' or press 3."
            ))
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
        #  - Accepts DTMF or speech
        #  - Normalizes to E.164 ONLY (US/Egypt supported via normalize_phone_e164)
        #  - Stores under session_data[call_sid]["cancel"]["phone_e164"]
        #  - Next stage: cancel_appt_get_date_time
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
            f"cancel_appt_by_phone_number: 🗣️ speech='{speech_text}' "
            f"🔢 DTMF='{dtmf_digits}'"
        )

        # 🔇 Silent mode: nothing heard at all
        if not (speech_text or dtmf_digits):
            tries = session_data[call_sid].get("silence_cancel_phone", 0) + 1
            session_data[call_sid]["silence_cancel_phone"] = tries
            debug_print(f"cancel_appt_by_phone_number: 🤐 silence count={tries}")

            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            prompt = (
                "I didn’t hear your phone number. Please say or type your phone number including area code, "
                "then press pound."
            )
            session_data[call_sid]["stage"] = "cancel_appt_by_phone_number"
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
                debug_print(f"cancel_appt_by_phone_number: normalizing via {default_country} from='{raw_for_e164}'")
                phone_e164 = normalize_phone_e164(raw_for_e164, default_country) or ""

            if not phone_e164 and raw_digits:
                # secondary attempt with bare digits
                phone_e164 = normalize_phone_e164(raw_digits, default_country) or ""

            if not phone_e164:
                # try the other supported country as a last resort (still E.164 only)
                alt = "EG" if default_country != "EG" else "US"
                debug_print(f"cancel_appt_by_phone_number: retry via alt country={alt}")
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
            session_data[call_sid]["stage"] = "cancel_appt_by_phone_number"
            prompt = (
                "I didn’t catch a valid phone number. Please say or type your phone number including area code, "
                "then press pound."
            )
            resp.append(make_gather(prompt, hints="zero one two three four five six seven eight nine double triple"))
            resp.redirect("/voice")
            return str(resp)

        # ✅ Store E.164 only and proceed
        session_data[call_sid]["cancel"]["phone_e164"] = phone_e164
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
        #  - Requires a phone on file first (E.164 ONLY)
        #  - Next stage: cancel_appt_get_date_time
        # ----------------------------------------------------------------------
        debug_print("cancel_appt_get_dob: 📍 Stage entered")

        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("customer", {})

        # Guard: require E.164 phone first (set by cancel_appt_by_phone_number / collect_phone)
        cust_phone_e164 = (
            session_data[call_sid].get("cancel", {}).get("phone_e164")
            or session_data[call_sid]["customer"].get("phone_e164")
            or session_data[call_sid].get("phone_e164")
            or ""
        ).strip()

        if not cust_phone_e164:
            debug_print("cancel_appt_get_dob: ❌ E.164 phone missing → collect_phone")
            session_data[call_sid]["return_stage"] = "cancel_appt_get_dob"
            session_data[call_sid]["stage"] = "collect_phone"
            resp.append(make_gather(
                "To cancel your appointment, please provide your phone number including area code. "
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
                "or type 2 digits for month 2 digits for day and 4 digits for year, then press pound."
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
                "Please say your birth date, for example July third nineteen fifty six, "
                "or type 2 digits for month 2 digits for day and 4 digits for year, then press pound."
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
            _Date = globals().get("_date")
            if _Date is None:
                from datetime import date as _Date  # local fallback
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
        #   - Requires a chosen doctor (calendar_id) and a PHONE IN E.164 to prefer
        #     matching the correct patient when multiple overlaps exist.
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

        # --- Require phone (E.164 ONLY) ------------------------------------------
        phone_e164 = (
            cancel_ctx.get("phone_e164")
            or session_data[call_sid].get("phone_e164")
            or session_data[call_sid].get("customer", {}).get("phone_e164")
            or ""
        ).strip()

        if not phone_e164:
            debug_print("cancel_appt_by_time_date: ❌ E.164 phone missing → collect_phone first")
            session_data[call_sid]["return_stage"] = "cancel_appt_by_time_date"
            session_data[call_sid]["stage"] = "collect_phone"
            resp.append(make_gather(
                "To locate your appointment, please provide your phone number including area code. "
                "You can say it, or type the digits and press pound.",
                hints="zero one two three four five six seven eight nine"
            ))
            return str(resp)

        # Store for downstream stages
        cancel_ctx["phone_e164"] = phone_e164

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

        # --- Availability (invert logic for cancel) -------------------------------
        try:
            slot_free = slot_available = is_time_slot_available(calendar_id, appointment_start, appointment_end, creds)
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

        # --- Slot is BUSY → fetch overlapping event(s) to identify event ----------
        try:
            service = build("calendar", "v3", credentials=creds)

            # Pad the search window to catch edge-inclusive overlaps
            sdt = isoparse(appointment_start)
            edt = isoparse(appointment_end)
            tmin = (sdt - timedelta(seconds=60)).isoformat()
            tmax = (edt + timedelta(seconds=60)).isoformat()

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

            # Prefer the event whose private.patient_phone_e164 (or phone_e164) matches the caller.
            chosen = None
            for ev in candidates:
                try:
                    priv = (ev.get("extendedProperties", {}) or {}).get("private", {}) or {}
                    ev_e164 = (priv.get("patient_phone_e164") or
                            priv.get("phone_e164") or
                            priv.get("phone") or "").strip()
                    if ev_e164 == phone_e164:
                        chosen = ev
                        break
                except Exception:
                    pass

            # Fallback: compare digits of e164 against digits in description (best-effort)
            if not chosen and candidates:
                try:
                    e164_digits = "".join(ch for ch in phone_e164 if ch.isdigit())
                    for ev in candidates:
                        desc = (ev.get("description") or "") or ""
                        desc_digits = "".join(ch for ch in desc if ch.isdigit())
                        if e164_digits and e164_digits in desc_digits:
                            chosen = ev
                            break
                except Exception:
                    pass

            # If still not chosen, pick the first overlapping event
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
                # optional: store who we matched for traceability
                "matched_phone_e164": phone_e164,
            }
            debug_print(
                f"cancel_appt_by_time_date: ✅ matched event id={chosen.get('id')} "
                f"summary='{chosen.get('summary','')}' phone_e164='{phone_e164}'"
            )

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
        #     "phone_e164": "+14694633276",         # REQUIRED (E.164 only)
        #     "dob": "YYYY-MM-DD" or "",            # OPTIONAL (if provided we filter by it)
        #     "doctor": "Alfred Hitchcock",         # OPTIONAL; if missing we search ALL doctors
        #     "candidates": [ ... ],                # OPTIONAL; built on first entry
        #     "iter_index": 0                       # OPTIONAL; current position in candidates
        #   }
        #
        # Notes:
        #   - Uses local helpers: get_doctor_appts_for / build_doctor_appt_index
        #   - Produces normalized "candidate" dicts compatible with cancel_appt_confirm:
        #       {
        #         "doctor_name": str,
        #         "start_utc":  str,   # ISO-8601 UTC (as stored in file; assumed UTC)
        #         "end_utc":    str,   # optional/blank if not present
        #         "friendly":   str,   # e.g., "Tuesday, August 12 at 9:00 AM"
        #         "phone_e164": str,   # E.164 only
        #         "dob":        str    # ISO DOB if present
        #       }
        # ----------------------------------------------------------------------

        # ---------- tiny helpers ----------
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

        def _appt_to_candidate(appt: dict, doctor_name: str, phone_e164: str, dob_iso: str) -> dict:
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
                "phone_e164": (phone_e164 or "").strip(),
                "dob": (dob_iso or "").strip(),
                # "raw": appt  # (optional breadcrumb)
            }

        # ---------- ensure cancel context ----------
        cancel_ctx = session_data[call_sid].setdefault("cancel", {})
        phone_e164 = (cancel_ctx.get("phone_e164")
                    or session_data[call_sid].get("phone_e164")
                    or session_data[call_sid].get("customer", {}).get("phone_e164")
                    or "").strip()
        dob_in   = (cancel_ctx.get("dob") or "").strip()       # DOB already verified upstream
        doctor   = (cancel_ctx.get("doctor") or "").strip()

        if not phone_e164:
            debug_print("cancel_appt_iterate: ❌ missing E.164 phone → route back to collect_phone")
            session_data[call_sid]["stage"] = "collect_phone"
            gather = make_gather(
                "To locate your appointment, please say or type your phone number including area code, then press pound.",
                hints="zero one two three four five six seven eight nine"
            )
            resp.append(gather)
            return str(resp)

        # ---------- Build candidates on first entry ----------
        if not cancel_ctx.get("candidates"):
            debug_print(f"cancel_appt_iterate: 🔎 building candidates from local doctor JSON for phone_e164='{phone_e164}' dob='{dob_in or '∅'}'")

            candidates = []

            # If doctor specified → only search that one
            if doctor:
                appts = get_doctor_appts_for(doctor, phone_e164, dob_in or None)
                debug_print(f"cancel_appt_iterate: doctor='{doctor}' → {len(appts)} appt(s) for caller")
                for ap in appts:
                    candidates.append(_appt_to_candidate(ap, doctor, phone_e164, dob_in))
            else:
                # No specific doctor → search ALL known doctors (from your global map)
                try:
                    doctor_map = googleid_dr_name_map  # {calendar_id: "Doctor Friendly Name"}
                except NameError:
                    doctor_map = {}
                doctor_names = sorted(set(doctor_map.values())) if doctor_map else []

                for dr_name in doctor_names:
                    appts = get_doctor_appts_for(dr_name, phone_e164, dob_in or None)
                    if appts:
                        debug_print(f"cancel_appt_iterate: doctor='{dr_name}' → {len(appts)} appt(s)")
                    for ap in appts:
                        candidates.append(_appt_to_candidate(ap, dr_name, phone_e164, dob_in))

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
                debug_print("cancel_appt_iterate: 🚫 no appointments found for caller (E.164)")
                resp.say(gpt_speak("I couldn't find any upcoming appointments under that phone number."), VOICE)
                if cancel_ctx.get("reschedule_after_cancel"):
                    session_data[call_sid]["stage"] = "booking"
                    doctor_list_str = ", ".join(googleid_dr_name_map.values())
                    gather = make_gather(
                        "Would you like to book a new appointment? Please say the doctor's name.",
                        hints=doctor_list_str
                    )
                    resp.append(gather)
                    return str(resp)
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
                    gather = make_gather(
                        "Would you like to book a new appointment? Please say the doctor's name.",
                        hints=doctor_list_str
                    )
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
        say_line = (
            f"I found an appointment with {cand['doctor_name']} on {cand['friendly']}. "
            f"Do you want to cancel this one? Say yes or no. You can also press 1 for yes, or 2 for no."
        )
        debug_print(f"cancel_appt_iterate: 🗣️ prompting candidate #{idx+1}/{total} "
                    f"doctor='{cand['doctor_name']}' start='{cand['start_utc']}' "
                    f"phone_e164='{cand['phone_e164']}' dob='{cand['dob'] or '∅'}'")

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
   # ----------------------------------------------------------------------
# 📌 Stage: cancel_appt_confirm
#
# What this does now (updated):
#   • Always attempts LOCAL cancellation first via:
#       cancel_appointment_by_name(doctor_name, phone, dob, utc_start)
#     using doctor name + phone (E.164) + DOB + exact UTC start time.
#   • If a Google Calendar ID is available, it ALSO tries to delete the
#     corresponding GCal event (best-effort; not required).
#   • Speaks a friendly, local-time confirmation when successfully cancelled.
#
# Inputs expected in session_data[call_sid]["cancel"]:
#   {
#     "phone_e164":   str,   # REQUIRED earlier in the flow (E.164)
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
#         "phone_e164":  str,
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
        debug_print("📍 Stage: cancel_appt_confirm")

        cancel_ctx  = session_data[call_sid].setdefault("cancel", {})
        # Prefer already-stored E.164; fall back to normalizing any raw 'phone'
        phone_raw   = (cancel_ctx.get("phone_e164") or cancel_ctx.get("phone") or "").strip()
        doctor      = (cancel_ctx.get("doctor") or "").strip()
        spoken_day  = (cancel_ctx.get("day") or "").strip()
        spoken_time = (cancel_ctx.get("time") or "").strip()
        utc_start   = (cancel_ctx.get("utc_start") or "").strip()
        utc_end     = (cancel_ctx.get("utc_end") or "").strip()
        calendar_id = (cancel_ctx.get("calendar_id") or "").strip()
        dob         = (cancel_ctx.get("dob") or session_data[call_sid].get("customer", {}).get("dob") or "").strip()

        # Use call-detected country if available, else your global COUNTRY
        default_country = (session_data[call_sid].get("phone_country") or COUNTRY or "US").upper()

        # If a matching candidate was found earlier, prefer its values
        cand = cancel_ctx.get("matching_event") or {}
        if cand:
            doctor    = cand.get("doctor_name", doctor) or doctor
            utc_start = cand.get("start_utc",   utc_start) or utc_start
            utc_end   = cand.get("end_utc",     utc_end) or utc_end
            phone_raw = (cand.get("phone_e164") or cand.get("phone") or phone_raw).strip()
            dob       = cand.get("dob") or dob

        # -------- Phone: E.164 only (no phone10 fallback) --------
        phone_e164 = ""
        raw = (phone_raw or "").strip()
        if raw.startswith("+") and raw[1:].replace(" ", "").isdigit():
            phone_e164 = "+" + raw[1:].replace(" ", "")
        else:
            try:
                phone_e164 = normalize_phone_e164(raw, default_country) or ""
                if not phone_e164:
                    # try the other supported country as a last resort
                    alt = "EG" if default_country != "EG" else "US"
                    phone_e164 = normalize_phone_e164(raw, alt) or ""
            except Exception:
                phone_e164 = ""

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

        # Primary: local JSON cancel (doctor + phone_e164 + dob + utc_start)
        local_ok = False
        if doctor and phone_e164 and dob and utc_start:
            try:
                local_ok = cancel_appointment_by_name(
                    doctor_name=doctor,
                    phone=phone_e164,   # ← E.164 ONLY
                    dob=dob,
                    utc_start=utc_start
                )
            except Exception as e:
                debug_print(f"cancel_appt_confirm: local cancel failed → {e}")
        else:
            debug_print("cancel_appt_confirm: insufficient info for local cancel (need doctor, phone_e164, dob, utc_start)")

        # Secondary: best-effort Google Calendar delete (search by E.164)
        gcal_ok = False
        if calendar_id and utc_start and phone_e164:
            try:
                start_dt  = dtparser.isoparse(utc_start)
                win_start = (start_dt - timedelta(minutes=30)).astimezone(timezone.utc).isoformat()
                win_end   = (start_dt + timedelta(minutes=30)).astimezone(timezone.utc).isoformat()

                # get_upcoming_events is assumed to use E.164-only matching now
                matched = get_upcoming_events(calendar_id, phone_e164, win_start, win_end, creds, debug=True)
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
            debug_print("cancel_appt_confirm: skipping GCal delete (missing calendar_id, utc_start, or phone_e164)")

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
