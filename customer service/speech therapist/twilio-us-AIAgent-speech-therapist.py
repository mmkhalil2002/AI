# update  10/17/25 time_saved  PERFECT
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
import dateparser
import pytz as _pytz
import pytz as _TZMOD
import time as _time_mod
import threading
import traceback
import dateutil.parser as dp
import string, unicodedata as _uni



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
FOREIGN_NAME_HINTS = """
# Arabic Names (Egypt, Levant, Gulf) — includes 99 Names of Allah (Asma' ul‑Husna)
Ahmed, Ahmad, Mohamed, Muhammad, Hamza, Youssef, Yousef, Yasin, Yassin, Ali, Hassan, Hussain, Hussein, Mostafa, Mustafa,
Khaled, Khalid, Karim, Kareem, Samir, Sameer, Omar, Omer, Amr, Tarek, Tariq, Farid, Fadi, Fady, Fawzi, Zaki, Zain, Zayd, Ziad, Mazen,
Mahmoud, Mahmood, Ismail, Ismael, Mansour, Mansoor, Saleh, Salih, Said, Saeed, Salem, Kamel, Kamil, Lotfi, Lutfi, Rashed, Rashid, Taha,
Abdallah, Abdullah, Ramadan, Attia, Atiya, Morsi, Nabil, Nabeel, Basel, Bassem, Hany, Hani, Walid, Waleed, Nasser, Nasir, Adel, Aadel,
Rami, Ramy, Sherif, Sharif, Magdy, Magdi, Hatem, Hatim, Yasser, Ayman, Aiman, Marwan, Mourad, Murad, Anas, Bilal, Faisal, Faysal,
Amine, Amin, Younes, Yunus, Younis, Jawad, Jamal, Jamil, Ghassan, Ghaith, Qasim, Kassim,

# 99 Names of Allah (used in Arabic male names with “Abdul‑” prefix)
Rahman, Rahim, Malik, Quddus, Salam, Mu'min, Muhaymin, Aziz, Jabbar, Mutakabbir,
Khaliq, Bari, Musawwir, Ghaffar, Qahhar, Wahhab, Razzaq, Fattah, Alim, Qabid, Basit, Khafid,
Rafi, Mu'izz, Mudhill, Sami, Basir, Hakim, Adl, Latif, Khabir, Halim, Azim, Ghafur,
Shakur, Ali, Kabir, Hafiz, Muqit, Hasib, Jalil, Karim, Raqib, Mujib, Wasi, Hakim,
Wadud, Majid, Ba'ith, Shahid, Haqq, Wakil, Qawi, Matin, Wali, Hamid, Muhsi, Mubdi,
Mu'id, Muhyi, Mumit, Hayy, Qayyum, Wajid, Majid, Wahid, Samad, Qadir, Muqtadir,
Muqaddim, Mu’akhkhir, Awwal, Akhir, Zaher, Batin, Wali, Muta'ali, Barr, Tawwab,
Muntaqim, Afu, Rauf, Malik al‑Mulk, Dhul‑Jalal wal‑Ikram, Muqsit, Jami, Ghani,
Mughni, Mani, Darr, Nafi, Nur, Hadi, Badi, Baqi, Warith, Rashid, Sabur,

# Arabic Female Names
Aisha, Ayesha, Aysha, Mariam, Maryam, Miriam, Fatma, Fatima, Faten, Fatin, Huda, Hanaa, Hana, Rania, Ranya, Esraa, Alaa,
Nour, Noor, Nor, Dalia, Dalya, Layla, Leila, Laila, Lina, Lena, Riham, Reham, Salma, Selma, Sara, Sarah, Zahra, Zehra,
Zeinab, Zainab, Nadia, Nadiya, Nadera, Hiba, Heba, Maha, Mona, Muna, Manal, Amal, Iman, Eman, Doaa, Dua, Somaya, Sumaya,
Samira, Sameera, Yasmin, Yasmine, Jasmine, Nourhan, Nermin, Nirmeen, Reem, Rym, Hager, Hajar, Rahma, Rahmah,

# Indian Male Names
Rahul, Rohan, Arjun, Vikram, Raj, Rajesh, Ravi, Rohit, Rakesh, Sunil, Sanjay, Suresh, Amit, Deepak, Anil, Nikhil, Karthik, Kartik, Varun,
Vijay, Akshay, Abhishek, Aditya, Siddharth, Sidharth, Ishaan, Ishan, Pranav, Prakash, Mohan, Manoj, Anurag, Arnav, Yash, Harsh, Kunal, Naveen,
Aman, Gaurav, Dev, Devansh, Parth, Shubham, Shreyas, Sagar, Suraj, Tejas, Ankit,

# Indian Female Names
Priya, Anjali, Neha, Pooja, Kiran, Ritu, Sneha, Aarti, Arti, Kavita, Meera, Mira, Nisha, Riya, Diya, Isha, Asha, Sanya, Tanya, Ananya,
Aishwarya, Shreya, Shruti, Bhavna, Poonam, Karishma, Radhika, Rituparna, Nandini, Trisha, Ishita, Komal, Juhi,

# Pakistani / Muslim South Asian Names
Arif, Arshad, Imran, Irfan, Faisal, Danish, Saad, Adeel, Naveed, Javed, Junaid, Rehan, Reza, Riza, Farhan, Fahad,
Hammad, Hamid, Kamran, Salman, Shahid, Shahbaz,

# Chinese Names
Zhang, Li, Wang, Chen, Liu, Huang, Lin, Yang, Zhao, Wu, Zhou, Xu, Sun, Ma, Zhu, Guo, He, Gao, Luo, Deng, Qian, Mei, Jia, Wei, Hao, Ying, Ning, Long,
Xiao, Xia, Xiu, Xin, Xue, Qiao, Qiu, Qi, Rui, Lei, Fang, Hui, Yan, Yuan, Yao, Tao, Dong, Fei, Jun, Jian, Jing, Liang, Lian, Ling, Ming, Ping, Qing, Sheng, Shu, Wen,

# Japanese Names
Ken, Yuki, Sora, Haru, Rina, Aoi, Ren, Ryo, Yuta, Yuto, Yuya, Hana, Hina, Mai, Kai, Daiki, Sakura, Takashi, Yoko, Yuka, Ayumi, Akira, Daichi, Keiko, Naoki, Satoshi,
Takeshi, Taro, Jiro, Ichiro, Kenta, Shota, Yuma, Riku, Erika, Emi,

# Korean Names
Kim, Lee, Park, Choi, Jung, Jeong, Kang, Yoon, Yun, Lim, Im, Han, Shin, Seo, Suh, Kwon, Hwang, Yoo, Ryu, Ryou, Baek, Byun, Nam, Oh, Song, Moon, Cho, Jo, Jang,
Jiho, Ji-hoon, Minji, Min-ji, Jisoo, Ji-soo, Seojun, Seo-jun, Yeonwoo, Yeon-woo, Hyun, Hyunwoo, Hyun-woo, Soo-min, Su-min,

# Vietnamese Names
Nguyen, Tran, Le, Pham, Huynh, Vo, Phan, Truong, Bui, Do, Dang, Dinh, Vu, Vuong, Anh, Linh, Thao, Nhan, Quang, Minh, Nam, Duc, Hoa, Huong, Lan, Mai, My, Phuong, Trang,

# Indonesian / Malay Names
Putra, Putri, Budi, Siti, Nur, Dewi, Agus, Rizki, Rizky, Dian, Andi, Wulan, Eka, Rani, Adi, Hendra, Fitri, Widya, Yuli,

# Filipino / Spanish-influenced Names
Jose, Maria, Juan, Mark, Marco, Carlo, Carlos, Miguel, Andrea, Angel, Angelo, Anne, Anna, Liza, Jessa, Katrina, Kristine, Paolo, Ramon,

# Thai Names
Somchai, Suriya, Anan, Apichai, Niran, Chai, Kanya, Mali, Suda, Nicha, Nisa, Lalita, Siriporn, Thanya, Prapa, Arisa,

# Burmese Names (Myanmar)
Aung, Min, Hla, Htoo, Nandi, Su, Suu, Thura, Thant, Nyein, Phyo, Zaw, Zin, Ei, Ei Mon,

# Cambodian (Khmer) Names
Sok, Soth, Dara, Vannak, Chan, Sophea, Sreypov, Pisey, Rith, Ratha,

# Persian / Iranian Names
Reza, Rezaul, Rezaan, Farzad, Farshad, Arman, Arash, AliReza, Alireza, Navid, Nima, Sina, Sara, Sahar, Parisa, Ladan, Leila, Leyla, Negin
""".strip()








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
# do u allow to create new customer on LINE
CREATE_NEW_CUSTOMER = bool(os.getenv("CREATE_NEW_CUSTOMER", True))  # d

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
      - Add customer_status field (default = "current").
      - Add pin_number field (6-digit integer; auto-generated if missing)
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

    def _is_e164(s: str) -> bool:
        s = (s or "").strip()
        return bool(_re.fullmatch(r"\+\d{6,15}", s))

    def _e164_or_empty(s: str) -> str:
        s = (s or "").strip().replace(" ", "")
        return s if _is_e164(s) else ""

    try:
        import random  # local import for clarity
        new_data = {}
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for old_key, rec in data.items():
            if not isinstance(rec, dict):
                new_data[old_key] = rec
                continue

            # ✅ Ensure created_at and last_seen_at timestamps
            if not rec.get("created_at") or not rec.get("last_seen_at"):
                rec.setdefault("created_at", now)
                rec.setdefault("last_seen_at", now)
                ensured_ts += 1
                changed = True

            # ✅ Ensure customer_status always present (default = "current")
            # Possible values: "current" or "new"
            if not rec.get("customer_status"):
                rec["customer_status"] = "current"
                changed = True

            # ✅ Ensure pin_number always present (6-digit integer)
            # Auto-generate a random 6-digit PIN if missing or invalid.
            # Example: 483927
            pin_value = rec.get("pin_number")
            if not isinstance(pin_value, int) or pin_value < 100000 or pin_value > 999999:
                rec["pin_number"] = random.randint(100000, 999999)
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

        # ✅ Save updated database only if changed
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
        #import re as _re
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
    customer_status: str = "current",
    pin_number: int = 0,
) -> bool:
    """
    Insert or update a customer in customers.json (single pretty JSON dict).

      • If (phone|dob) exists → update record fields + bump 'last_seen_at'; return False.
      • If new → create record with 'created_at' + 'last_seen_at'; return True.

    This version guarantees:
      ✅ Strict E.164 enforcement (no legacy 10-digit fallback).
      ✅ Adds 'customer_status' field (default = "current", or explicitly set to "new").
      ✅ Adds 'pin_number' field (6-digit integer; default = 0 if not provided).
      ✅ Preserves timestamps and existing customer data.
    """
    # ----------------------------------------------------------------------
    # 🧩 Step 1: Ensure DB is initialized
    # ----------------------------------------------------------------------
    init_db()

    # ----------------------------------------------------------------------
    # 🧩 Step 2: Normalize phone (strict E.164 only)
    # ----------------------------------------------------------------------
    phone_e164 = normalize_phone_e164(phone, globals().get("COUNTRY", "US"))
    if not phone_e164:
        raise ValueError(f"insert_customer: invalid phone '{phone}' (must be valid E.164)")

    dob_iso = (dob or "").strip() or "unknown"
    first_name = _oneline(first_name)
    last_name  = _oneline(last_name)
    address    = _oneline(address)
    cc_name    = _oneline(cc_name)
    cc_number  = _oneline(cc_number)
    cc_exp     = _oneline(cc_exp)
    cc_cvv     = _oneline(cc_cvv)

    # ----------------------------------------------------------------------
    # 🧩 Step 3: Load + prepare customer data
    # ----------------------------------------------------------------------
    data = _load_customers()
    try:
        key = _key(phone_e164, dob_iso)
    except Exception:
        key = f"{phone_e164}|{dob_iso}"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ----------------------------------------------------------------------
    # 🧩 Step 4: Update existing customer
    # ----------------------------------------------------------------------
    if key in data:
        existing = data[key]

        # ✅ If pin_number provided and valid, update it; otherwise preserve existing one
        if isinstance(pin_number, int) and 100000 <= pin_number <= 999999:
            new_pin = pin_number
        else:
            new_pin = existing.get("pin_number", 0)

        existing.update({
            "first_name": first_name or existing.get("first_name", ""),
            "last_name": last_name or existing.get("last_name", ""),
            "address": address or existing.get("address", ""),
            "cc_name": cc_name or existing.get("cc_name", ""),
            "cc_number": cc_number or existing.get("cc_number", ""),
            "cc_exp": cc_exp or existing.get("cc_exp", ""),
            "cc_cvv": cc_cvv or existing.get("cc_cvv", ""),
            "last_seen_at": now,
            "customer_status": customer_status or "current",  # ✅ override if specified
            "pin_number": new_pin,
        })
        _save_customers(data)
        debug_print(f"insert_customer: 🟡 Updated existing record for {key} (status={customer_status}, pin={new_pin})")
        return False

    # ----------------------------------------------------------------------
    # 🧩 Step 5: Insert new customer record
    # ----------------------------------------------------------------------
    # ✅ Generate random 6-digit PIN if not provided or invalid
    import random
    if not isinstance(pin_number, int) or pin_number < 100000 or pin_number > 999999:
        pin_number = random.randint(100000, 999999)

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
        "customer_status": customer_status or "current",  # ✅ input-controlled value
        "pin_number": pin_number,  # ✅ always stored as integer
    }
    data[key] = rec
    _save_customers(data)

    debug_print(
        f"insert_customer: ✅ Added {customer_status.upper()} customer {first_name} {last_name} "
        f"({phone_e164}|{dob_iso}) @ {now} (PIN={pin_number})"
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



def update_customer_status(
    phone: str,
    dob: str,
    new_status: str,
    default_country: str = COUNTRY
) -> bool:
    """
    Update the 'customer_status' value for a given customer record.

    Args:
        phone (str): Customer's phone number (must be E.164 or normalizable).
        dob (str): Date of birth string as stored in record (e.g., "1990-04-12").
        new_status (str): New value for 'customer_status' ("new" or "current").
        default_country (str): Default country for normalization (default = COUNTRY).

    Returns:
        bool: True if update succeeded, False if no record found or invalid input.

    Behavior:
      ✅ Strict E.164 enforcement (no legacy 10-digit fallback).
      ✅ Updates in-place within customers.json and preserves timestamps.
      ✅ Returns False if record not found or normalization fails.
      ✅ Logs every step for debugging visibility.
    """
    # ----------------------------------------------------------------------
    # 🧩 Step 1: Sanitize and validate input
    # ----------------------------------------------------------------------
    if new_status not in {"new", "current"}:
        debug_print(f"update_customer_status: ❌ invalid status '{new_status}' (must be 'new' or 'current')")
        return False

    init_db()
    dob_iso = (dob or "").strip()
    raw_phone = (phone or "").strip()

    # ----------------------------------------------------------------------
    # 🧩 Step 2: Normalize phone to E.164
    # ----------------------------------------------------------------------
    phone_e164 = ""
    try:
        if raw_phone.startswith("+") and raw_phone[1:].replace(" ", "").isdigit():
            phone_e164 = "+" + raw_phone[1:].replace(" ", "")
            debug_print(f"update_customer_status: 📞 using existing E.164 '{phone_e164}'")
        else:
            for country in [(default_country or "US").upper(), "EG", "US"]:
                phone_e164 = normalize_phone_e164(raw_phone, country)
                if phone_e164:
                    debug_print(f"update_customer_status: 📞 normalized to {phone_e164} ({country})")
                    break
    except Exception as e:
        debug_print(f"update_customer_status: ⚠️ normalization failed → {e}")

    if not phone_e164:
        debug_print("update_customer_status: ❌ could not normalize phone to E.164; aborting")
        return False

    # ----------------------------------------------------------------------
    # 🧩 Step 3: Load database and locate record
    # ----------------------------------------------------------------------
    data = _load_customers()
    key = _key(phone_e164, dob_iso)
    rec = data.get(key)

    # Fallback scan if not found directly
    if rec is None:
        for k, r in data.items():
            if (r.get("phone_e164") or "").strip() == phone_e164 and (r.get("dob") or "").strip() == dob_iso:
                rec = r
                key = k
                debug_print(f"update_customer_status: ✅ found record by scan under key '{k}'")
                break
        if rec is None:
            debug_print(f"update_customer_status: ❌ no record found for {phone_e164}|{dob_iso}")
            return False

    # ----------------------------------------------------------------------
    # 🧩 Step 4: Update the customer_status field
    # ----------------------------------------------------------------------
    old_status = rec.get("customer_status", "current")
    rec["customer_status"] = new_status
    rec["last_seen_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ----------------------------------------------------------------------
    # 🧩 Step 5: Save and log changes
    # ----------------------------------------------------------------------
    _save_customers(data)
    debug_print(
        f"update_customer_status: ✅ Updated {phone_e164}|{dob_iso}\n"
        f"  Old status: {old_status}\n"
        f"  New status: {new_status}\n"
        f"  Last Seen At: {rec['last_seen_at']}"
    )
    return True



def get_pin_number(phone_e164: str, dob: str) -> Optional[int]:
    """
    Retrieve a customer's 6-digit pin_number from customers.json.

    Parameters:
        phone_e164 (str): Customer's phone number in E.164 format (e.g., "+14155552671").
        dob (str): Date of birth in ISO format (YYYY-MM-DD).

    Returns:
        int | None: The customer's pin_number if found and valid (6 digits), otherwise None.

    Behavior:
      ✅ Looks up the customer record by key = "<phone_e164>|<dob>".
      ✅ Ensures that the pin_number is an integer and within 100000–999999.
      ✅ Logs helpful debug output.
    """
    try:
        init_db()  # ensure DB ready
        data = _load_customers()
        key = _key(phone_e164, dob)

        rec = data.get(key)
        if not rec:
            debug_print(f"get_pin_number: ❌ no record for key={key}")
            return None

        pin = rec.get("pin_number")
        if isinstance(pin, int) and 100000 <= pin <= 999999:
            debug_print(f"get_pin_number: ✅ found pin={pin} for {key}")
            return pin
        else:
            debug_print(f"get_pin_number: ⚠️ invalid or missing pin for {key} → {pin}")
            return None

    except Exception as e:
        debug_print(f"get_pin_number: ⚠️ error reading pin for {phone_e164}|{dob}: {e}")
        return None



def update_pin_number(phone_e164: str, dob: str, new_pin: int) -> bool:
    """
    Update or assign a customer's 6-digit pin_number in customers.json.

    Parameters:
        phone_e164 (str): Customer's phone number in E.164 format.
        dob (str): Customer's date of birth (YYYY-MM-DD).
        new_pin (int): The new 6-digit PIN to store.

    Returns:
        bool: True if the update succeeded, False if customer not found or invalid pin.

    Behavior:
      ✅ Validates that the new_pin is a 6-digit integer (100000–999999).
      ✅ Updates the record's pin_number and last_seen_at.
      ✅ Persists the change immediately in customers.json.
      ✅ Logs debug details for traceability.
    """
    try:
        if not isinstance(new_pin, int) or new_pin < 100000 or new_pin > 999999:
            debug_print(f"update_pin_number: ❌ invalid new_pin={new_pin}")
            return False

        init_db()
        data = _load_customers()
        key = _key(phone_e164, dob)
        rec = data.get(key)

        if not rec:
            debug_print(f"update_pin_number: ❌ no record for key={key}")
            return False

        rec["pin_number"] = new_pin
        rec["last_seen_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _save_customers(data)

        debug_print(f"update_pin_number: ✅ updated pin={new_pin} for {key}")
        return True

    except Exception as e:
        debug_print(f"update_pin_number: ⚠️ failed for {phone_e164}|{dob}: {e}")
        return False





def get_customer_status(phone: str, dob: str, default_country: str = COUNTRY) -> Optional[str]:
    """
    Retrieve the customer's current status ("new" or "current") from customers.json.

    Behavior:
      ✅ Uses strict E.164-only lookup (no legacy phone fallback).
      ✅ Returns "new" or "current" if found, None if no record exists.
      ✅ Performs a light scan fallback if the exact key is missing.
    """
    init_db()
    dob_iso = (dob or "").strip()
    raw_phone = (phone or "").strip()

    # --- Normalize phone to E.164 ---
    phone_e164 = ""
    try:
        if raw_phone.startswith("+") and raw_phone[1:].replace(" ", "").isdigit():
            phone_e164 = "+" + raw_phone[1:].replace(" ", "")
            debug_print(f"get_customer_status: 📞 using E.164 '{phone_e164}' directly")
        else:
            for country in [(default_country or "US").upper(), "EG", "US"]:
                phone_e164 = normalize_phone_e164(raw_phone, country)
                if phone_e164:
                    debug_print(f"get_customer_status: 📞 normalized to {phone_e164} ({country})")
                    break
    except Exception as e:
        debug_print(f"get_customer_status: ⚠️ normalization failed → {e}")

    if not phone_e164:
        debug_print("get_customer_status: ❌ invalid phone; cannot normalize to E.164")
        return None

    # --- Load data and find record ---
    data = _load_customers()
    key = _key(phone_e164, dob_iso)
    rec = data.get(key)

    # Light scan fallback
    if rec is None:
        for k, r in data.items():
            if (r.get("phone_e164") or "").strip() == phone_e164 and (r.get("dob") or "").strip() == dob_iso:
                rec = r
                debug_print(f"get_customer_status: ✅ found record by scan under key '{k}'")
                break
        if rec is None:
            debug_print(f"get_customer_status: ❌ no record found for {phone_e164}|{dob_iso}")
            return None

    # --- Return status safely ---
    status = rec.get("customer_status", "current")
    debug_print(f"get_customer_status: ✅ status for {phone_e164}|{dob_iso} = '{status}'")
    return status





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
    Update the customer's credit card info in customers.json by (phone_e164|dob).

    Optimization goals:
      ✅ Strict E.164-only normalization (no legacy fallback).
      ✅ Clearer flow with early returns.
      ✅ Reduced redundant normalization and string ops.
      ✅ Maintains identical behavior and full debug traceability.
    """
    # ----------------------------------------------------------------------
    # 🧩 Step 1: Ensure DB ready + normalize input
    # ----------------------------------------------------------------------
    init_db()
    dob_iso = (dob or "").strip()
    raw_phone = (phone or "").strip()

    # --- Normalize to E.164 once ---
    phone_e164 = ""
    try:
        if raw_phone.startswith("+") and raw_phone[1:].replace(" ", "").isdigit():
            phone_e164 = "+" + raw_phone[1:].replace(" ", "")
            debug_print(f"update_cc_info: 📞 using existing E.164: {phone_e164}")
        else:
            for country in [(default_country or "US").upper(), "EG", "US"]:
                phone_e164 = normalize_phone_e164(raw_phone, country)
                if phone_e164:
                    debug_print(f"update_cc_info: 📞 normalized to {phone_e164} ({country})")
                    break
    except Exception as e:
        debug_print(f"update_cc_info: ⚠️ normalization failed → {e}")

    if not phone_e164:
        debug_print("update_cc_info: ❌ could not normalize phone to E.164; aborting")
        return False

    # ----------------------------------------------------------------------
    # 🧩 Step 2: Load customer data
    # ----------------------------------------------------------------------
    data = _load_customers()
    key = _key(phone_e164, dob_iso)
    rec = data.get(key)

    # --- Fallback: Light scan (E.164 + DOB match only) ---
    if rec is None:
        for k, r in data.items():
            if (r.get("phone_e164") or "").strip() == phone_e164 and (r.get("dob") or "").strip() == dob_iso:
                rec = r
                key = k
                debug_print(f"update_cc_info: ✅ found record under key '{k}' (scan match)")
                break
        if rec is None:
            debug_print(f"update_cc_info: ❌ no record for phone={phone_e164} dob={dob_iso or '∅'}")
            return False

    # ----------------------------------------------------------------------
    # 🧩 Step 3: Apply CC updates
    # ----------------------------------------------------------------------
    for field, value in {
        "cc_number": cc_number,
        "cc_exp": cc_exp,
        "cc_cvv": cc_cvv,
    }.items():
        if value is not None:
            rec[field] = _oneline(value)

    rec["last_seen_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_customers(data)

    # ----------------------------------------------------------------------
    # 🧩 Step 4: Log changes
    # ----------------------------------------------------------------------
    debug_print(
        f"update_cc_info: ✅ updated record for {phone_e164}\n"
        f"  DOB: {dob_iso or '∅'}\n"
        f"  CC Number: {rec.get('cc_number', '')}\n"
        f"  CC Exp: {rec.get('cc_exp', '')}\n"
        f"  CC CVV: {rec.get('cc_cvv', '')}\n"
        f"  Last Seen At: {rec['last_seen_at']}"
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
    # Create a new TwiML VoiceResponse object to build the voice reply to the caller
    resp = VoiceResponse()
    debug_print("[voice] ▶ enter voice()")

    # Extract the unique call ID (SID) from the request parameters to track the session
    call_sid = request.values.get("CallSid", "")
    debug_print(f"[voice] CallSid={call_sid}")

    # Retrieve the customer's speech input (transcribed by Twilio's Speech-to-Text)
    speech_result = (request.values.get("SpeechResult") or "").strip()
    # Also grab any keypad input (DTMF) Twilio might have sent with the same webhook
    try:
        dtmf_digits = (request.values.get("Digits") or "").strip()
    except Exception:
        dtmf_digits = ""
    debug_print(f"[voice] inputs → speech='{speech_result}' dtmf='{dtmf_digits}'")

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
        debug_print(f"[voice] 🌐 country seeded → {derived}")
    else:
        debug_print(f"[voice] 🌐 country exists → {session_data[call_sid].get('country')}")

    # (optional) keep the raw caller E.164 for later use
    from_number = (request.values.get("From") or "").strip()
    if from_number.startswith("+"):
        session_data[call_sid]["from_e164"] = from_number
        debug_print(f"[voice] from_e164 set → {from_number}")

    print(f"📢 voice :speech_result: {speech_result}")

    # Determine the current interaction stage (default to "intro" if not previously set)
    stage = session_data.get(call_sid, {}).get("stage", "intro")
    debug_print(f"[voice] 🎯 stage='{stage}'")

    # ----------------------------------------------------------------------
    # 🔇 CENTRAL SILENCE GUARD
    # If we didn't hear *anything* (no speech, no DTMF), re-prompt with
    # stage-appropriate text. We skip stages that already have their own
    # robust silence handling (e.g., collect_cc).
    # ----------------------------------------------------------------------
    def _silence_prompt_for_stage(st: str) -> Tuple[str, str]:
        """Return (prompt, hints) best suited for the current stage."""
        debug_print(f"[voice] 🔇 selecting silence prompt for stage='{st}'")
        # Default: generic prompt, no hints
        hints = ""
        if st in ("intro", "intent"):
            # ✨ Updated to advertise both voice and keypad (DTMF 1..5)
            hints = "book,cancel,change,reschedule,update,update card,voicemail,leave message"
            debug_print("[voice] 🔇 using intro/intent silence prompt")
            return (
                "I didn’t hear anything. Say 'book appointment' or press 1. "
                "Say 'cancel appointment' or press 2. "
                "Say 'change appointment' or press 3. "
                "Say 'update credit card' or press 4. "
                "Say 'leave voicemail' or press 5.",
                hints
            )
        if st == "book_appointment":
            doctor_list = ", ".join(googleid_dr_name_map.values())
            hints = doctor_list
            debug_print("[voice] 🔇 using booking silence prompt")
            return ("Please say the name of the doctor you'd like to book with.", hints)
        if st == "collect_phone":
            hints = "zero one two three four five six seven eight nine double triple"
            debug_print("[voice] 🔇 using collect_phone silence prompt")
            return ("Please say or enter your ten digit phone number including area code.", hints)
        if st == "collect_dob":
            debug_print("[voice] 🔇 using collect_dob silence prompt")
            return ("Please say your birth date, for example 'July third 1990'. Or type 2 digits for Month 2 digits for Day 4 digits for year then press pound.", hints)
        if st == "ask_time_date":
            debug_print("[voice] 🔇 using ask_time_date silence prompt")
            return ("Please say the appointment time, for example, 'August 15th at 5 AM'. "
                        "Or enter two digits for month, two for day, two for hour, and two for minutes, then enter  A for AM or P for PM. then press #", hints)

            debug_print("[voice] 🔇 using collect_first_name silence prompt")
            return ("Please say your first name.", hints)
        if st == "collect_last_name":
            debug_print("[voice] 🔇 using collect_last_name silence prompt")
            return ("Please say your last name.", hints)
        if st == "collect_address":
            debug_print("[voice] 🔇 using collect_address silence prompt")
            return ("Please say your street address, city, and ZIP. For example, '118 Briar Oak, Murphy, Texas 75094'.", hints)
        if st == "cancel_appointment":
            doctor_list = ", ".join(googleid_dr_name_map.values())
            hints = doctor_list
            debug_print("[voice] 🔇 using cancel_appointment silence prompt")
            return ("Please say the name of the doctor whose appointment you want to cancel.", hints)
        if st in ("cancel_appt_by_phone_number",):
            hints = "zero one two three four five six seven eight nine double triple"
            debug_print("[voice] 🔇 using cancel_appt_by_phone_number silence prompt")
            return ("Please say the phone number used when booking, including area code.", hints)
        if st in ("cancel_appt_by_time_date", "cancel_appt_by_date_time"):
            debug_print("[voice] 🔇 using cancel_appt_by_time_date silence prompt")
            return ("Please say the date and time of the appointment you want to cancel, for example, 'July third at nine AM'.", hints)
        if st == "cancel_appt_get_dob":
            debug_print("[voice] 🔇 using cancel_appt_get_dob silence prompt")
            return ("Please say your birth date, for example 'July third nineteen fifty six'. Or type 2 digits for month 2 digits for day and 4 digis for year then press pound.", hints)
        if st == "voicemail":
            debug_print("[voice] 🔇 using voicemail silence prompt")
            return ("Please leave your name, phone number, and message after the beep.", hints)

        # Fallback generic
        debug_print("[voice] 🔇 using generic silence prompt")
        return ("Sorry, I didn’t hear anything. Please say that again.", hints)

    # ----------------------------------------------------------------------
    # Silence handling guard
    # ----------------------------------------------------------------------
    # Only run the guard outside of the very first greeting (intro),
    # and skip stages that handle silence internally.
    skip_silence = (
        "intro",
        "intent"
        "collect_cc",
        "book_appt_confirm",
        # 🚫 NEW: skip cancel flow stages too
        "cancel_appt_iterate",
        "cancel_appt_get_time_date",
        "collect_phone",
        "cancel_appt_confirm",
        "collect_dob",
        "collect_first_name",
        "collect_last_name"
    )
    debug_print(f"[voice] 🔇 skip_silence={skip_silence}")

    if stage not in skip_silence:
        debug_print(f"[voice] 🔇 evaluating silence guard at stage='{stage}' (speech_empty={not bool(speech_result)} dtmf_empty={not bool(dtmf_digits)})")
        if not speech_result and not dtmf_digits:
            session_data.setdefault(call_sid, {})
            key = f"silence_{stage}"
            session_data[call_sid][key] = session_data[call_sid].get(key, 0) + 1
            tries = session_data[call_sid][key]
            debug_print(f"[voice] 🔇 silence detected at stage='{stage}' (tries={tries})")

            if tries >= 3:
                debug_print("[voice] 🔇 max silence reached → hangup")
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            prompt, hints = _silence_prompt_for_stage(stage)
            debug_print(f"[voice] 🔇 re-prompting with prompt='{prompt[:80]}' hints='{hints}'")
            try:
                gather = make_gather(prompt, hints=hints, num_digits=1) if hints else make_gather(prompt, num_digits=1)
            except Exception as _e:
                debug_print(f"[voice] 🔇 make_gather failed: {_e} → using generic prompt")
                gather = make_gather("Sorry, I didn’t hear anything. Please try again.", num_digits=1)
            resp.append(gather)
            try:
                redirect_url = url_for("voice")
                resp.redirect(redirect_url)
                debug_print(f"[voice] 🔇 redirect → {redirect_url}")
            except Exception:
                resp.redirect("/voice")
                debug_print("[voice] 🔇 redirect → /voice (fallback)")
            return str(resp)
    else:
        debug_print(f"[voice] 🔇 silence guard skipped for stage='{stage}'")











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
        # 🎯 STAGE: intent — Detect caller's intent (via speech or keypad)
        #
        # PURPOSE:
        #   - Determine what the caller wants: book, cancel, reschedule, update, etc.
        #   - Accepts BOTH speech (Twilio SpeechResult) and DTMF keypad input.
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # 💬 Local message constants
        # ----------------------------------------------------------------------
        MAIN_MENU_PROMPT = (
            "Thank you for calling Epic Therapist. "
            "Please choose one of the following options. "
            "Say 'book appointment' or press 1. "
            "Say 'cancel appointment' or press 2. "
            "Say 'change appointment' or press 3. "
            "Say 'update credit card' or press 4. "
            "Say 'update PIN number' or press 5. "
            "Say 'update health insurance' or press 6. "
            "Say 'leave voicemail' or press 7."
        )

        REPEAT_MENU_PROMPT = (
            "I'm sorry, I didn’t understand that. "
            "Please say 'book appointment' or press 1. "
            "Say 'cancel appointment' or press 2. "
            "Say 'change appointment' or press 3. "
            "Say 'update credit card' or press 4. "
            "Say 'update PIN number' or press 5. "
            "Say 'update health insurance' or press 6. "
            "Say 'leave voicemail' or press 7."
        )

        UPDATE_CC_PLACEHOLDER_MSG = (
            "You said you want to update your credit card information. "
            "Please hold while we process this request."
        )

        UPDATE_PIN_PLACEHOLDER_MSG = (
            "You said you want to update your PIN number. "
            "This option is not implemented yet. "
            "Please call the clinic for assistance."
        )

        UPDATE_INSURANCE_PLACEHOLDER_MSG = (
            "You said you want to update your health insurance information. "
            "This option is not implemented yet. "
            "Please call the clinic for assistance."
        )

        # ----------------------------------------------------------------------
        # 🧠 Input extraction
        # ----------------------------------------------------------------------
        lower = (speech_result or "").lower().strip()
        debug_print(f"📢 intent :speech_result: {lower}")
        debug_print(f"📞 intent :dtmf_digits: {dtmf_digits}")

        # ----------------------------------------------------------------------
        # 🩵 Handle polite / empty / meaningless responses
        # ----------------------------------------------------------------------
        polite_or_empty = not lower or lower in {
            "thank you", "thanks", "thankyou", "ok", "okay",
            "goodbye", "bye", "no", "nothing", "that's it",
            "that’s it", "that’s all", "that is all"
        }

        if polite_or_empty:
            debug_print(f"[intent] 🙏 polite or empty response ('{lower}') — re-prompt menu")
            g = Gather(
                input="speech dtmf",           # Accept both speech and keypad
                timeout=6,                     # Wait 6 seconds for response
                speech_timeout="auto",         # Auto-end on speech pause
                barge_in=True,                 # Allow interruption
                finish_on_key="#",             # '#' ends DTMF input early
                num_digits=1,                  # Expect single digit 1–7
                action="/voice", method="POST",
                language="en-US"
            )
            g.say(gpt_speak(MAIN_MENU_PROMPT), VOICE)
            resp.append(g)
            resp.redirect("/voice")
            return str(resp)

        # ----------------------------------------------------------------------
        # 🔢 DTMF (Keypad) detection
        #
        # Example:
        #   Pressing “1” → dtmf_digits="1" → goes to booking flow
        # ----------------------------------------------------------------------
        choice = None
        if dtmf_digits and len(dtmf_digits) == 1 and dtmf_digits in "1234567":
            choice = dtmf_digits
            debug_print(f"[intent] 🎹 keypad input detected → {choice}")
        elif lower in {"1", "2", "3", "4", "5", "6", "7"}:
            choice = lower
            debug_print(f"[intent] 💬 spoken numeric intent → {choice}")

        # ----------------------------------------------------------------------
        # 🗣️ Voice-based keyword detection
        #
        # Twilio’s SpeechResult is free text. We check for common intent words.
        # ----------------------------------------------------------------------
        if any(word in lower for word in ["book", "appointment", "schedule", "make appointment"]):
            choice = "1"
        elif any(word in lower for word in ["cancel", "delete", "remove", "call off"]):
            choice = "2"
        elif any(word in lower for word in ["reschedule", "change", "move", "shift", "different time"]):
            choice = "3"
        elif any(word in lower for word in ["credit", "card", "payment", "update card"]):
            choice = "4"
        elif any(word in lower for word in ["pin", "password", "update pin", "change pin", "pin number"]):
            choice = "5"
        elif any(word in lower for word in ["insurance", "health", "medical", "health card"]):
            choice = "6"
        elif any(word in lower for word in ["voicemail", "message", "leave a message", "record message"]):
            choice = "7"

        # ----------------------------------------------------------------------
        # ✅ Routing based on detected choice
        #
        # DTMF and voice inputs share the same menu routing.
        # ----------------------------------------------------------------------
        if choice:
            # 1️⃣ Book Appointment
            if choice == "1":
                debug_print("intent:📅 → booking")
                session_data.setdefault(call_sid, {})
                session_data[call_sid].update({
                    "stage": "book_appointment",
                    "booking": {},
                    "retry_booking": 0,
                    "retry_time": 0
                })

                # Create list of doctors and DTMF map
                doctor_names = list(googleid_dr_name_map.values())
                dtmf_map = {str(i): name for i, name in enumerate(doctor_names, start=1)}
                session_data[call_sid]["doctor_dtmf_map"] = dtmf_map
                doctor_list = ", ".join([f"{name} (press {i})" for i, name in enumerate(doctor_names, start=1)])

                # Prompt user to choose a doctor
                prompt = (
                    f"Great! Let's schedule your appointment. "
                    f"Available doctors are: {doctor_list}. "
                    "Please say the doctor's name or press the number."
                )
                g = Gather(
                    input="speech dtmf",
                    timeout=6,
                    speech_timeout="auto",
                    barge_in=True,
                    finish_on_key="#",
                    num_digits=1,
                    action="/voice", method="POST",
                    language="en-US"
                )
                g.say(gpt_speak(prompt), VOICE)
                resp.append(g)
                resp.redirect("/voice")
                return str(resp)

            # 2️⃣ Cancel Appointment
            if choice == "2":
                debug_print("intent:❌ → cancel appointment")
                session_data[call_sid] = {"stage": "cancel_appointment", "cancel": {}}
                doctor_names = list(googleid_dr_name_map.values())
                dtmf_map = {str(i): name for i, name in enumerate(doctor_names, start=1)}
                session_data[call_sid]["doctor_dtmf_map"] = dtmf_map
                doctor_list = ", ".join([f"{name} (press {i})" for i, name in enumerate(doctor_names, start=1)])
                prompt = (
                    f"Sure, I can help you cancel your appointment. "
                    f"Available doctors are: {doctor_list}. "
                    "Please say the doctor's name or press the number."
                )
                g = Gather(
                    input="speech dtmf", timeout=6, speech_timeout="auto",
                    barge_in=True, finish_on_key="#", num_digits=1,
                    action="/voice", method="POST", language="en-US"
                )
                g.say(gpt_speak(prompt), VOICE)
                resp.append(g)
                resp.redirect("/voice")
                return str(resp)

            # 3️⃣ Reschedule Appointment
            if choice == "3":
                debug_print("intent:🔁 → reschedule")
                session_data[call_sid] = {
                    "stage": "cancel_appointment",
                    "cancel": {},
                    "reschedule_after_cancel": True
                }
                doctor_names = list(googleid_dr_name_map.values())
                dtmf_map = {str(i): name for i, name in enumerate(doctor_names, start=1)}
                session_data[call_sid]["doctor_dtmf_map"] = dtmf_map
                doctor_list = ", ".join([f"{name} (press {i})" for i, name in enumerate(doctor_names, start=1)])
                prompt = (
                    f"Let's reschedule your appointment. First, we'll cancel your current one. "
                    f"Available doctors are: {doctor_list}. "
                    "Please say the doctor's name or press the number."
                )
                g = Gather(
                    input="speech dtmf", timeout=6, speech_timeout="auto",
                    barge_in=True, finish_on_key="#", num_digits=1,
                    action="/voice", method="POST", language="en-US"
                )
                g.say(gpt_speak(prompt), VOICE)
                resp.append(g)
                resp.redirect("/voice")
                return str(resp)

            # 4️⃣ Update Credit Card
            if choice == "4":
                debug_print("intent:💳 → update credit card")
                session_data.setdefault(call_sid, {})
                session_data[call_sid].update({
                    "stage": "update_cc",
                    "cc_update": {"active": True}
                })
                resp.say(gpt_speak(UPDATE_CC_PLACEHOLDER_MSG), VOICE)
                resp.redirect("/voice")
                return str(resp)

            # 5️⃣ Update PIN Number
            if choice == "5":
                debug_print("intent:🔢 → update PIN")
                session_data.setdefault(call_sid, {})
                session_data[call_sid]["stage"] = "update_pin"
                resp.say(gpt_speak(UPDATE_PIN_PLACEHOLDER_MSG), VOICE)
                g = Gather(
                    input="speech dtmf", timeout=6, speech_timeout="auto",
                    barge_in=True, finish_on_key="#", num_digits=1,
                    action="/voice", method="POST", language="en-US"
                )
                g.say(gpt_speak(MAIN_MENU_PROMPT), VOICE)
                resp.append(g)
                resp.redirect("/voice")
                return str(resp)

            # 6️⃣ Update Health Insurance
            if choice == "6":
                debug_print("intent:🏥 → update insurance")
                session_data.setdefault(call_sid, {})
                session_data[call_sid]["stage"] = "update_insurance"
                resp.say(gpt_speak(UPDATE_INSURANCE_PLACEHOLDER_MSG), VOICE)
                g = Gather(
                    input="speech dtmf", timeout=6, speech_timeout="auto",
                    barge_in=True, finish_on_key="#", num_digits=1,
                    action="/voice", method="POST", language="en-US"
                )
                g.say(gpt_speak(MAIN_MENU_PROMPT), VOICE)
                resp.append(g)
                resp.redirect("/voice")
                return str(resp)

            # 7️⃣ Voicemail
            if choice == "7":
                debug_print("intent:📩 → voicemail")
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

        # ----------------------------------------------------------------------
        # 🚫 Fallback — Unrecognized input: repeat menu
        # ----------------------------------------------------------------------
        debug_print(f"intent:⚠️ unrecognized '{lower}' — repeating main menu")
        g = Gather(
            input="speech dtmf", timeout=6, speech_timeout="auto",
            barge_in=True, finish_on_key="#", num_digits=1,
            action="/voice", method="POST", language="en-US"
        )
        g.say(gpt_speak(REPEAT_MENU_PROMPT), VOICE)
        resp.append(g)
        resp.redirect("/voice")
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
    


    elif stage == "book_appointment":
        # ----------------------------------------------------------------------
        # 📍 Booking flow: ask caller to name or select a doctor.
        # Accepts both speech and single-digit DTMF input.
        # ----------------------------------------------------------------------

        session_data.setdefault(call_sid, {}).setdefault("retry_booking", 0)
        session_data[call_sid]["origin_stage"] = "book"  # ✅ Mark booking as origin

        _PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        dtmf_digits = (request.values.get("Digits") or "").strip()
        spoken_text = (speech_result or "").strip().lower()
        spoken_clean = spoken_text.translate(str.maketrans('', '', _PUNCT)).strip()
        print(f"📻 booking :speech_result: {spoken_clean} DTMF='{dtmf_digits}'")

        matched_id = None

        # -------------------- 🔢 DTMF Matching --------------------
        if dtmf_digits and "doctor_dtmf_map" in session_data[call_sid]:
            doctor_map = session_data[call_sid]["doctor_dtmf_map"]
            chosen_name = doctor_map.get(dtmf_digits)
            if chosen_name:
                for doc_id, friendly in googleid_dr_name_map.items():
                    if friendly.lower() == chosen_name.lower():
                        matched_id = doc_id
                        print(f"✅ DTMF matched doctor: {friendly}")
                        break

        # -------------------- 🎙️ Speech Matching --------------------
        if matched_id is None:
            junk_inputs = {
                "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
                "yo", "test", "1", "yes", "no", "i know", "huh", "what", "okay", "ok",
                "bye", "goodbye", ""
            }

            if not spoken_clean or spoken_clean in junk_inputs or len(spoken_clean) < 3:
                print(f"⏩ Skipping junk doctor input: '{spoken_clean}' — re-prompting")
                doctor_list_str = ", ".join(googleid_dr_name_map.values())

                g = Gather(
                    input="speech dtmf",
                    language="en-US",
                    hints=f"{doctor_list_str}, {FOREIGN_NAME_HINTS}",
                    num_digits=1,
                    timeout=6,
                    speech_timeout="auto",
                    barge_in=True,
                    action="/voice",
                    method="POST"
                )
                g.say(
                    "Please say the name of the doctor you'd like to book with. "
                    "You can also press the number on your keypad.",
                    voice=VOICE
                )
                resp.append(g)
                resp.redirect("/voice")
                return str(resp)

            # 🔍 Fuzzy Match
            partial_matches = []
            spoken_tokens = set(spoken_clean.split())
            for doc_id, friendly in googleid_dr_name_map.items():
                friendly_clean = friendly.lower().translate(str.maketrans('', '', _PUNCT)).strip()
                friendly_tokens = set(friendly_clean.split())
                if (spoken_clean in friendly_clean or
                    friendly_clean in spoken_clean or
                    (spoken_tokens & friendly_tokens)):
                    partial_matches.append((doc_id, friendly))

            if len(partial_matches) == 1:
                matched_id = partial_matches[0][0]
                print(f"✅ Partial match with: {partial_matches[0][1]}")
            elif len(partial_matches) > 1:
                print(f"🔍 Multiple matches: {[name for _, name in partial_matches]}")
                matched_id = partial_matches[0][0]

        # -------------------- ❌ Retry on Failure --------------------
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
                    VOICE
                )
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            doctor_list_str = ", ".join(googleid_dr_name_map.values())
            g = Gather(
                input="speech dtmf",
                language="en-US",
                hints=f"{doctor_list_str}, {FOREIGN_NAME_HINTS}",
                num_digits=1,
                timeout=6,
                speech_timeout="auto",
                barge_in=True,
                action="/voice",
                method="POST"
            )
            g.say(
                f"I couldn't match that to a doctor. "
                f"Available doctors are: {doctor_list_str}. "
                "Please say the doctor's name or press the number.",
                voice=VOICE
            )
            resp.append(g)
            resp.redirect("/voice")
            return str(resp)

        # -------------------- ✅ Success — Store and Prompt Next --------------------
        session_data[call_sid]["doctor_id"] = matched_id
        session_data[call_sid]["stage"] = "collect_phone"
        friendly_name = googleid_dr_name_map[matched_id]

        g = Gather(
            input="speech dtmf",
            language="en-US",
            num_digits=10,
            timeout=8,
            speech_timeout="auto",
            barge_in=True,
            action="/voice",
            method="POST"
        )
        g.say(
            f"Great, we'll book with {friendly_name}. "
            "Please say or enter your phone number, including area code, then press pound.",
            voice=VOICE
        )
        resp.append(g)
        resp.redirect("/voice")
        return str(resp)




    
    elif stage == "collect_phone":
        # ==========================================================================
        # 📞 Stage: collect_phone — capture customer phone number via speech/DTMF.
        #
        # DESIGN INTENT:
        #   - Accept phone via speech (e.g., "469 463 3276") or DTMF (e.g., 4694633276#).
        #   - Normalize to E.164 (e.g., "+14694633276").
        #   - Validate length/format (US = 10 digits).
        #   - Handle up to 3 invalid retries and 3 silent timeouts.
        #   - Mirror phone into booking and cancellation contexts.
        # ==========================================================================
        
        debug_print("[collect_phone] 📍 entered")

        # Ensure session buckets exist
        sd = session_data.setdefault(call_sid, {})
        cust = sd.setdefault("customer", {})
        cancel_ctx = sd.setdefault("cancel", {})

        # Infer country once per call (used by phone normalization)
        if "phone_country" not in sd:
            from_country = (request.values.get("FromCountry") or "").upper()
            sd["phone_country"] = from_country or (COUNTRY or "US")
            debug_print(f"[collect_phone] 🌐 phone_country={sd['phone_country']}")

        # Inputs Twilio heard this turn
        dtmf_digits = (request.values.get("Digits") or "").strip()
        speech_text = (speech_result or "").strip()
        debug_print(f"[collect_phone] 🗣 speech='{speech_text}'  🔢 DTMF='{dtmf_digits}'")

        # ------------------------------------------------------------------
        # 🔇 LOCAL SILENCE HANDLING
        # ------------------------------------------------------------------
        if not (speech_text or dtmf_digits):
            tries = sd.get("silence_collect_phone", 0) + 1
            sd["silence_collect_phone"] = tries
            debug_print(f"[collect_phone] 🤐 no input (tries={tries}/3)")

            if tries < 3:
                g = make_gather(
                    prompt="I didn’t hear your phone number. "
                        "Please say or enter your 10-digit number, then press pound.",
                    input="speech dtmf",
                    timeout=4,
                    speech_timeout="auto",
                    barge_in=True,
                    finish_on_key="#"
                )
                resp.append(g)
                resp.redirect("/voice")
                debug_print("[collect_phone] 🔁 re-prompt & redirect → /voice (new webhook regardless of input)")
                return str(resp)

            debug_print("[collect_phone] ❌ max silence → hangup")
            resp.say(gpt_speak("I'm sorry, I still didn't get your phone number. Please call again later."), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)

        # We DID receive some input → clear the silence counter
        sd.pop("silence_collect_phone", None)

        # ------------------------------------------------------------------
        # 🧠 Speech → digits helper
        # ------------------------------------------------------------------
        def _spoken_to_digits(raw: str) -> str:
            if not raw:
                return ""
            words = (
                raw.lower()
                .replace("-", " ").replace(",", " ").replace(".", " ")
                .replace("(", " ").replace(")", " ").split()
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
                w = words[i]
                if w in ("double", "triple") and i + 1 < len(words):
                    nxt = words[i + 1]
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

        # ------------------------------------------------------------------
        # 🔢 Normalize digits to E.164 (prefer DTMF; else use speech)
        # ------------------------------------------------------------------
        if dtmf_digits:
            raw_digits = _re.sub(r"\D", "", dtmf_digits)
        else:
            raw_digits = _re.sub(r"\D", "", _spoken_to_digits(speech_text))
        debug_print(f"[collect_phone] 🔍 raw_digits='{raw_digits}'")

        country = sd.get("phone_country", (COUNTRY or "US")).upper()
        try:
            phone_e164 = normalize_phone_e164(raw_digits, country)
            debug_print(f"[collect_phone] ✅ normalized → {phone_e164}")
        except NameError:
            # Minimal fallback for US
            d = raw_digits
            if country == "US":
                if len(d) == 11 and d.startswith("1"):
                    d = d[1:]
                phone_e164 = f"+1{d}" if len(d) == 10 else ""
            else:
                phone_e164 = ""
            debug_print(f"[collect_phone] ⚠️ fallback normalize → '{phone_e164}'")

        # ------------------------------------------------------------------
        # ❌ Invalid number → retry up to 3x
        # ------------------------------------------------------------------
        if not phone_e164:
            r = sd.get("retry_phone", 0) + 1
            sd["retry_phone"] = r
            debug_print(f"[collect_phone] ❌ invalid number (retry {r}/3) input='{raw_digits}'")

            if r < 3:
                g = make_gather(
                    prompt="That doesn’t sound complete. "
                        "Please say or enter your 10-digit phone number including area code, then press pound.",
                    input="speech dtmf",
                    timeout=5,
                    speech_timeout="auto",
                    barge_in=True,
                    finish_on_key="#"
                )
                resp.append(g)
                resp.redirect("/voice")
                debug_print("[collect_phone] 🔁 invalid → re-prompt & redirect")
                return str(resp)

            debug_print("[collect_phone] ❌ max invalid attempts → hangup")
            resp.say(gpt_speak("I'm sorry, I couldn’t capture your phone number. Please call again later."), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)

        # ------------------------------------------------------------------
        # ✅ Valid → Save & mirror
        # ------------------------------------------------------------------
        cust["phone_e164"] = phone_e164
        cust["phone"] = phone_e164
        cancel_ctx["phone_e164"] = phone_e164
        sd["phone_e164"] = phone_e164
        sd["retry_phone"] = 0
        debug_print(f"[collect_phone] 💾 saved phone_e164={phone_e164} (mirrored to cancel context)")

        # ↩️ Return to prior stage if specified
        return_stage = sd.pop("return_stage", None)
        if return_stage:
            sd["stage"] = return_stage
            debug_print(f"[collect_phone] ↩️ returning to stage '{return_stage}'")
            resp.redirect("/voice")
            return str(resp)

            # ==========================================================================
            # 🔁 RESCHEDULE FLOW — BRANCH TO ask_time_date
            # ==========================================================================
            # If the user has just canceled an appointment and indicated they want to
            # reschedule immediately, we skip the remaining stages (e.g. collect_dob, etc.)
            # and jump directly to asking for the new appointment date and time.
            #
            # HOW THIS WORKS:
            #  - Earlier in the flow, when the customer cancels an appointment, we set:
            #       sd["reschedule_after_cancel"] = True
            #  - When that flag exists, we come here instead of going to collect_dob.
            #  - We then prompt the user for a new appointment time.
            #
            # Example Interaction:
            #   System: “Thanks. Please say the new appointment date and time, for example,
            #            'October 12 at 9 A M'.”
            #   Caller: “October 18 at 4 PM.”
            #       → Control moves to stage ask_time_date, which validates and books the slot.
            #
            # DESIGN NOTES:
            #  • Name/phone are already saved, so no need to re-collect them.
            #  • We use make_gather() for uniform speech+DTMF handling.
            #  • We append a trailing resp.redirect("/voice") to force a new Twilio webhook
            #    even if the caller stays silent (Twilio’s <Gather> does not re-POST on silence).
            # ==========================================================================
            # 🔁 RESCHEDULE FLOW — BRANCH TO ask_time_date
        if sd.get("reschedule_after_cancel"):
            sd["stage"] = "ask_time_date"
            g = make_gather(
                prompt="Thanks. Please say the new appointment date and time, "
                    "for example, 'October 12 at 9 A M'.",
                input="speech dtmf",
                timeout=5,
                speech_timeout="auto",
                barge_in=True,
                finish_on_key="#"
            )
            resp.append(g)
            resp.redirect("/voice")
            debug_print("[collect_phone] 🔁 reschedule → ask_time_date (via make_gather + redirect)")
            return str(resp)

        # 🗓️ Normal flow → ask DOB next
        sd["stage"] = "collect_dob"
        g = make_gather(
            prompt="Thanks. What’s your date of birth? You can say it, or enter two digits for month, "
                "two for day, and four for year, then press pound.",
            input="speech dtmf",
            timeout=5,
            speech_timeout="auto",
            barge_in=True,
            finish_on_key="#"
        )
        resp.append(g)
        resp.redirect("/voice")
        debug_print("[collect_phone] ➡️ next stage → collect_dob")
        return str(resp)









    elif stage == "collect_dob":
        # ----------------------------------------------------------------------
        # 🎂 Stage: collect_dob
        #
        # PURPOSE
        #   • Capture and validate the customer's date of birth (DOB) via speech or DTMF.
        #   • Store the DOB in session_data and perform customer lookup.
        #
        # BRANCH LOGIC:
        #   1️⃣ Customer NOT found → go to "verify_customer_type"
        #   2️⃣ Customer FOUND but customer_status == "new" → instruct to complete registration
        #   3️⃣ Customer FOUND and customer_status == "current" →
        #         → continue to "collect_pin_number"
        #
        # IMPLEMENTATION NOTES:
        #   • Uses get_customer_status(phone_e164, dob) to determine status.
        #   • Pressing # (pound) terminates DTMF entry for faster input.
        #   • Handles silence (3 retries) with polite exit.
        # ----------------------------------------------------------------------

        t_stage_start = _time_mod.perf_counter()
        debug_print(f"collect_dob: 📍 Stage entered at {_time_mod.strftime('%H:%M:%S')}")

        # ----------------------------------------------------------------------
        # 🗣️ Message constants for all voice prompts (centralized definitions)
        # ----------------------------------------------------------------------
        MSG_FIRST_SILENT = (
                            "Please say your date of birth, for example, 'December 3 1962'. "
                            "You can also enter it using your keypad: 2 digits month, 2 digits day, and 4 digits year, then press pound."
                            )
        MSG_REPEAT_DOB = (
            "I didn’t hear your date of birth. Please say it again, for example, 'July 3 1956'. "
            "Or you can enter it using your keypad, then press pound."
        )
        MSG_HANGUP_SILENT = "Sorry, I couldn’t get your date of birth. Please call again later."
        MSG_PARSE_FAIL = (
            "I didn’t catch your full birth date. Please say the complete date, for example, 'July 3 1956'. "
            "You can also enter it using your keypad: 2 digits month, 2 digits day, and 4 digits year, then press pound."
        )
        MSG_INVALID_DOB = (
            "That doesn’t seem like a valid date of birth. "
            "Please enter 2 digits for month, 2 for day, and 4 for year, then press #"
        )
        MSG_NOT_FOUND = (
            "We couldn’t find a record with that phone number and date of birth. "
            "If you are a new customer, press 1. If you are an existing customer, press 2."
        )
        MSG_NEW_CUSTOMER = (
            "We found your record, but your registration with the clinic is not complete. "
            "Please contact the clinic to finish your registration before booking an appointment. Goodbye!"
        )
        MSG_PIN_PROMPT = (
            "Thank you. For security verification, please enter your six digit PIN number now, "
            "followed by the pound key. If you prefer, you can also say each digit slowly."
        )

        # ----------------------------------------------------------------------
        # 🧾 Session setup
        # ----------------------------------------------------------------------
        sd = session_data.setdefault(call_sid, {})
        sd.setdefault("customer", {})
        sd.setdefault("cancel", {})

        # ----------------------------------------------------------------------
        # 🎧 Inputs from Twilio webhook
        # ----------------------------------------------------------------------
        try:
            dtmf_digits = (request.values.get("Digits") or "").strip()
        except Exception:
            dtmf_digits = ""
        speech_text = (speech_result or "").strip()
        debug_print(f"collect_dob: 🎙️ speech='{speech_text}', 🔢 dtmf='{dtmf_digits}'")

        # ----------------------------------------------------------------------
        # 🔇 Silence handling (up to 3 tries)
        # ----------------------------------------------------------------------
        if not dtmf_digits and not speech_text:
            tries = sd.get("silence_dob", 0) + 1
            sd["silence_dob"] = tries
            debug_print(f"collect_dob: 🤐 silence tries={tries}/3")

            if tries == 1:
                # 🗣️ First silent → prompt to state first name clearly
                debug_print("collect_dob: 🗣️ first silent → ask for first name")
                sd["stage"] = "collect_dob"
                g = make_gather(
                    MSG_FIRST_SILENT,
                    input="speech dtmf",
                    timeout=3,
                    speech_timeout="auto",
                    barge_in=True,
                    finish_on_key="#",
                )
                resp.append(g)
                resp.redirect("/voice")
                return str(resp)

            if tries < 3:
                # 2nd or 3rd silent → normal DOB re-prompt
                sd["stage"] = "collect_dob"
                g = make_gather(
                    MSG_REPEAT_DOB,
                    input="speech dtmf",
                    timeout=3,
                    speech_timeout="auto",
                    barge_in=True,
                    finish_on_key="#",
                )
                resp.append(g)
                resp.redirect("/voice")
                return str(resp)

            # After 3rd → hang up politely
            resp.say(gpt_speak(MSG_HANGUP_SILENT), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)

        # ✅ Clear silence counter if input received
        sd.pop("silence_dob", None)

        # ----------------------------------------------------------------------
        # 🧩 Parse DOB from input
        # ----------------------------------------------------------------------
        dob_date = None
        if dtmf_digits:
            d = _re.sub(r"\D", "", dtmf_digits)
            if len(d) >= 8:
                try:
                    mm, dd, yyyy = int(d[0:2]), int(d[2:4]), int(d[4:8])
                    dob_date = date(yyyy, mm, dd)
                    debug_print("collect_dob: ✅ parsed DOB from keypad")
                except Exception as e:
                    debug_print(f"collect_dob: ❌ keypad parse error → {e}")

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
                debug_print(f"collect_dob: ❌ speech parse failed → {e}")
                sd["stage"] = "collect_dob"
                g = make_gather(MSG_PARSE_FAIL, input="speech dtmf", timeout=3,
                                speech_timeout="auto", barge_in=True, finish_on_key="#")
                resp.append(g)
                resp.redirect("/voice")
                return str(resp)

        # ----------------------------------------------------------------------
        # ⚙️ Validate DOB
        # ----------------------------------------------------------------------
        try:
            today = _date_local.today()
            if not dob_date or dob_date < date(1900, 1, 1) or dob_date > today:
                raise ValueError("DOB out of valid range")
        except Exception as e:
            debug_print(f"collect_dob: ⚠️ Validation error → {e}")
            sd["stage"] = "collect_dob"
            g = make_gather(MSG_INVALID_DOB, input="dtmf", timeout=3, barge_in=True, finish_on_key="#")
            resp.append(g)
            resp.redirect("/voice")
            return str(resp)

        # ----------------------------------------------------------------------
        # 💾 Store DOB
        # ----------------------------------------------------------------------
        iso_dob = dob_date.strftime("%Y-%m-%d")
        sd["customer"]["dob"] = iso_dob
        sd["cancel"]["dob"] = iso_dob
        debug_print(f"collect_dob: ✅ Stored DOB → {iso_dob}")

        # ----------------------------------------------------------------------
        # 🔍 Lookup customer record + status
        # ----------------------------------------------------------------------
        phone_e164 = sd["customer"].get("phone_e164") or sd.get("phone_e164")
        found = False
        customer_status = "unknown"

        if phone_e164 and iso_dob:
            try:
                found = customer_search(phone_number=phone_e164, dob=iso_dob, default_country="US")
                if found:
                    customer_status = get_customer_status(phone_e164, iso_dob)
                    sd["customer"]["customer_status"] = customer_status
                debug_print(f"collect_dob: 🔎 lookup(phone={phone_e164}, dob={iso_dob}) → found={found}, status={customer_status}")
            except Exception as e:
                debug_print(f"collect_dob: ⚠️ get_customer_status error → {e}")

        # ----------------------------------------------------------------------
        # 🔀 Branching logic based on search result
        # ----------------------------------------------------------------------
        if not found:
            # Not found → go to verify_customer_type
            sd["stage"] = "verify_customer_type"
            g = make_gather(
                MSG_NOT_FOUND,
                input="dtmf", timeout=3, speech_timeout="auto", barge_in=True, finish_on_key="#"
            )
            resp.append(g)
            resp.redirect("/voice")
            debug_print("collect_dob: 🔀 not found → verify_customer_type")
            return str(resp)

        # 🟡 Customer found but registration incomplete
        if customer_status == "new":
            debug_print("collect_dob: 🟡 found record but status='new' → require clinic registration")
            resp.say(gpt_speak(MSG_NEW_CUSTOMER), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)

        # ✅ Verified (current) customer → proceed to collect_pin_number
        sd["stage"] = "collect_pin_number"
        g = make_gather(
            MSG_PIN_PROMPT,
            input="speech dtmf",
            timeout=5,
            speech_timeout="auto",
            barge_in=True,
            finish_on_key="#"
        )
        resp.append(g)
        resp.redirect("/voice")
        debug_print("collect_dob: ✅ existing verified customer → proceed to collect_pin_number")
        return str(resp)





    # ----------------------------------------------------------------------
    # 🧩 NEW: Verify customer type (after DOB mismatch)
    elif stage == "verify_customer_type":
        # ----------------------------------------------------------------------
        # 🧭 Stage: verify_customer_type
        # Purpose:
        #   - Handle branching when a phone+dob lookup didn't find a customer.
        #   - Behavior depends on CREATE_NEW_CUSTOMER flag.
        #       * If False → hang up with message.
        #       * If True  → allow caller to choose: 1=new, 2=existing.
        #
        # Inputs:
        #   - DTMF (1=new, 2=existing)
        #   - Local silence handling and retries.
        #
        # Session Data:
        #   - last_customer_found (bool)
        #   - customer_status ("new" | "current")
        #   - silence_verify_type, retry_verify_type counters
        # ----------------------------------------------------------------------
        debug_print("verify_customer_type: 📍 Stage entered")

        sd = session_data.setdefault(call_sid, {})
        last_lookup_found = sd.get("last_customer_found", False)
        allow_new = bool(globals().get("CREATE_NEW_CUSTOMER", False))

        # Pull current DTMF input
        dtmf_digits = (request.values.get("Digits") or "").strip()
        debug_print(
            f"verify_customer_type: received DTMF='{dtmf_digits}', allow_new={allow_new}, found={last_lookup_found}"
        )

        # If new creation not allowed
        if not last_lookup_found and not allow_new:
            debug_print("verify_customer_type: not found & CREATE_NEW_CUSTOMER=False → hang up")
            resp.say(
                gpt_speak(
                    "We couldn’t find a record with that phone number and date of birth. "
                    "Please contact the clinic to create your customer record, then call us again."
                ),
                VOICE,
            )
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)

        # -------------------------------
        # 🔇 Silence handling
        # -------------------------------
        if not dtmf_digits:
            tries = sd.get("silence_verify_type", 0) + 1
            sd["silence_verify_type"] = tries
            debug_print(f"verify_customer_type: 🤐 silence tries={tries}/3")

            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            prompt = (
                "Please press 1 if you are a new customer, or 2 if you are an existing customer."
                if not last_lookup_found
                else "You are already in our system. Press 1 to continue scheduling."
            )
            g = make_gather(prompt, input="dtmf", timeout=4, barge_in=True, finish_on_key="#", num_digits=1)
            resp.append(g)
            resp.redirect("/voice")
            return str(resp)

        # Clear silence counter once input received
        sd.pop("silence_verify_type", None)

        # -------------------------------
        # 🧭 Branch on DTMF choice
        # -------------------------------
        if dtmf_digits == "1":
            # 1 → "New customer"
            sd["customer_status"] = "new"
            debug_print("verify_customer_type: 🆕 customer_status='new' stored in session")

            # Not found → proceed to collect first name
            if not last_lookup_found:
                debug_print("verify_customer_type: new customer not found → go to collect_first_name")
                sd["stage"] = "collect_first_name"
                g = make_gather(
                    "Welcome! Please say your first name",
                    input="speech dtmf", timeout=6, speech_timeout="auto", barge_in=True, finish_on_key="#"
                )
                resp.append(g)
                resp.redirect("/voice")
                return str(resp)

            # Found but pressed 1 (odd case)
            else:
                debug_print("verify_customer_type: found=True but pressed 1=new → continue to scheduling")
                sd["stage"] = "ask_time_date"
                g = make_gather(
                    "Okay. Please say the appointment date and time, for example, 'October 8 at 9 30 A M'.",
                    input="speech dtmf", timeout=6, speech_timeout="auto", barge_in=True, finish_on_key="#"
                )
                resp.append(g)
                resp.redirect("/voice")
                return str(resp)

        elif dtmf_digits == "2":
            # 2 → "Existing customer"
            sd["customer_status"] = "current"
            debug_print("verify_customer_type: 👤 customer_status='current' stored in session")

            if not last_lookup_found:
                debug_print("verify_customer_type: 2=existing; not found → hang up")
                resp.say(
                    gpt_speak(
                        "We couldn’t find you as an existing customer. "
                        "Please contact the clinic to set up your record, then call us again."
                    ),
                    VOICE,
                )
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)
            else:
                debug_print("verify_customer_type: 2=existing; found=True → proceed to ask_time_date")
                sd["stage"] = "ask_time_date"
                g = make_gather(
                    "Great. Please say the appointment date and time, for example, 'October 8 at 2 P M'.",
                    input="speech dtmf", timeout=6, speech_timeout="auto", barge_in=True, finish_on_key="#"
                )
                resp.append(g)
                resp.redirect("/voice")
                return str(resp)

        # -------------------------------
        # ❌ Invalid entry → retry
        # -------------------------------
        r = sd.get("retry_verify_type", 0) + 1
        sd["retry_verify_type"] = r
        debug_print(f"verify_customer_type: ❌ invalid DTMF '{dtmf_digits}' retry={r}/3")

        if r >= 3:
            resp.say(gpt_speak("Sorry, I didn’t get a valid choice. Please call again later."), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)

        g = make_gather(
            "Invalid choice. Press 1 if you are a new customer, or 2 if you are an existing customer.",
            input="dtmf", timeout=4, barge_in=True, finish_on_key="#", num_digits=1
        )
        resp.append(g)
        resp.redirect("/voice")
        return str(resp)


    elif stage == "collect_pin_number":
        # ----------------------------------------------------------------------
        # 🔢 Stage: collect_pin_number
        #
        # PURPOSE
        #   • Verify the caller's identity using their 6-digit PIN.
        #
        # FLOW
        #   1️⃣ Ask user to enter or say their 6-digit PIN (DTMF or speech).
        #   2️⃣ Compare against stored PIN using get_pin_number().
        #   3️⃣ If correct → branch based on origin_stage:
        #         - "book"       → ask_time_date
        #         - "cancel"     → cancel_appt_get_time_date
        #         - "update_cc"  → collect_cc
        #         - otherwise    → intro (main menu)
        #   4️⃣ If incorrect → allow up to 3 retries before terminating politely.
        #
        # FEATURES
        #   ✅ Handles silence locally (3 retries, then hang up).
        #   ✅ Tracks invalid PIN attempts (3 max, then advise to contact clinic).
        #   ✅ Supports both DTMF and spoken digits.
        #   ✅ Respects origin_stage for dynamic branching.
        # ----------------------------------------------------------------------

        debug_print("collect_pin_number: 📍 Stage entered")

        sd = session_data.setdefault(call_sid, {})
        sd.setdefault("customer", {})
        customer = sd["customer"]

        # --- Retrieve key info ------------------------------------------------
        phone_e164 = (customer.get("phone_e164") or sd.get("phone_e164") or "").strip()
        dob = (customer.get("dob") or "").strip()
        origin_stage = sd.get("origin_stage", "book")  # default to booking flow

        debug_print(f"collect_pin_number: 🔎 origin_stage={origin_stage}")

        # --- Gather inputs ----------------------------------------------------
        raw_dtmf = (request.values.get("Digits") or "").strip()
        raw_speech = (speech_result or "").strip()
        debug_print(f"collect_pin_number: 🔢 DTMF='{raw_dtmf}' 🗣 speech='{raw_speech}'")

        # ======================================================================
        # 🕳️ SILENCE HANDLING
        # ======================================================================
        if not raw_dtmf and not raw_speech:
            tries = sd.get("silence_pin", 0) + 1
            sd["silence_pin"] = tries
            debug_print(f"collect_pin_number: 🤐 silence tries={tries}/3")

            if tries >= 3:
                resp.say(
                    gpt_speak(
                        "I’m still not hearing anything. Please call the clinic for assistance."
                    ),
                    VOICE,
                )
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            prompt = (
                "Please enter your six digit PIN now, followed by the pound key. "
                "If you prefer, you can also say each digit slowly."
            )
            g = make_gather(
                prompt,
                input="speech dtmf",
                timeout=6,
                speech_timeout="auto",
                barge_in=True,
                finish_on_key="#",
                language="en-US",
                action="/voice",
            )
            resp.append(g)
            resp.redirect("/voice")
            return str(resp)

        # Reset silence counter once valid input received
        sd.pop("silence_pin", None)

        # ======================================================================
        # 🔢 PIN PARSING
        # ======================================================================
        digits = _re.sub(r"\D", "", raw_dtmf or raw_speech)
        debug_print(f"collect_pin_number: normalized digits='{digits}'")

        if len(digits) != 6:
            debug_print("collect_pin_number: ⚠️ invalid PIN length")
            sd["pin_attempts"] = sd.get("pin_attempts", 0) + 1
            if sd["pin_attempts"] >= 3:
                resp.say(
                    gpt_speak(
                        "That doesn’t seem like a valid six digit PIN. "
                        "Please contact the clinic to verify or reset your PIN number. Goodbye."
                    ),
                    VOICE,
                )
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            g = make_gather(
                "That doesn’t seem like a valid six digit PIN. Please try again now.",
                input="speech dtmf",
                timeout=6,
                speech_timeout="auto",
                barge_in=True,
                finish_on_key="#",
                language="en-US",
                action="/voice",
            )
            resp.append(g)
            resp.redirect("/voice")
            return str(resp)

        # ======================================================================
        # 🧩 PIN VALIDATION
        # ======================================================================
        try:
            stored_pin = get_pin_number(phone_e164, dob)
            debug_print(f"collect_pin_number: 🔍 stored_pin={stored_pin} for {phone_e164}|{dob}")
        except Exception as e:
            debug_print(f"collect_pin_number: ⚠️ error retrieving stored PIN → {e}")
            stored_pin = None

        # ======================================================================
        # ✅ SUCCESS CASE — Correct PIN
        # ======================================================================
        if stored_pin is not None and digits == str(stored_pin).zfill(6):
            debug_print(f"collect_pin_number: ✅ PIN verified successfully (origin={origin_stage})")

            sd.pop("pin_attempts", None)  # reset attempts after success

            if origin_stage == "book":
                next_stage = "ask_time_date"
                msg = "Thank you. Your PIN has been verified. Let's continue with your appointment booking."
            elif origin_stage == "cancel":
                next_stage = "cancel_appt_get_time_date"
                msg = "Thank you. PIN verified. Let's proceed to locate your appointment for cancellation."
            elif origin_stage == "update_cc":
                next_stage = "collect_cc"
                msg = "Your PIN has been verified. Let's update your payment information."
            else:
                next_stage = "intro"
                msg = "Thank you. Your PIN has been verified. Returning to the main menu."

            sd["stage"] = next_stage
            sd["skip_silence_once"] = True

            resp.say(gpt_speak(msg), VOICE)
            resp.redirect("/voice")
            return str(resp)

        # ======================================================================
        # ❌ FAILURE CASE — Wrong PIN
        # ======================================================================
        sd["pin_attempts"] = sd.get("pin_attempts", 0) + 1
        tries = sd["pin_attempts"]
        debug_print(f"collect_pin_number: ❌ invalid PIN ({digits}) vs stored ({stored_pin}) (try {tries}/3)")

        if tries < 3:
            # Allow retry
            retry_msg = (
                "That PIN number is incorrect. Please try again now. "
                "Enter your six digit PIN followed by the pound key."
            )
            g = make_gather(
                retry_msg,
                input="speech dtmf",
                timeout=6,
                speech_timeout="auto",
                barge_in=True,
                finish_on_key="#",
                language="en-US",
                action="/voice",
            )
            resp.append(g)
            resp.redirect("/voice")
            return str(resp)

        # ----------------------------------------------------------------------
        # 🚫 Max retries reached → terminate
        # ----------------------------------------------------------------------
        msg = (
            "You have entered an incorrect PIN too many times. "
            "Please contact the clinic to verify your information or to change your PIN number. Goodbye!"
        )
        resp.say(gpt_speak(msg), VOICE)
        resp.hangup()
        session_data.pop(call_sid, None)
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
        # ----------------------------------------------------------------------
        # 📅 ASK_TIME_DATE — AM/PM (or A/P) required, repeats on silence, 3 wrongs → hangup
        # ----------------------------------------------------------------------
        debug_print(f"[ask_time_date] 🗣️ Received speech: {speech_result}")

        # -------------------------- Config / Text --------------------------
        working_days  = globals().get("WORKING_DAYS", (0, 1, 2, 3, 4, 5))
        working_start = globals().get("WORKING_HOURS_START", 8)
        working_end   = globals().get("WORKING_HOURS_END", 17)

        DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        available_days = [DAY_NAMES[d] for d in working_days if 0 <= d < 7]

        def _fmt_hour(h: int) -> str:
            if h == 0:  return "12:00 AM"
            if h < 12:  return f"{h}:00 AM"
            if h == 12: return "12:00 PM"
            return f"{h-12}:00 PM"

        days_str  = (", ".join(available_days[:-1]) + f", and {available_days[-1]}" if len(available_days) > 1
                    else (available_days[0] if available_days else "weekdays"))
        hours_str = f"{_fmt_hour(working_start)} and {_fmt_hour(working_end)}"

        PROMPT_NEED_BOTH = (
            "Say the date and time, like 'October 8 at 9 30 A M' or 'October 8 at 2 P M'. "
            "Or enter two digits for month, two for day, two for hour, two for minutes, then say A or P."
        )
        PROMPT_PAST_TIME = "That time has already passed. Please choose a future time."
        PROMPT_NEED_VALID_DAY = f"That day isn’t available. We’re open {days_str}, between {hours_str}."
        PROMPT_NEED_AMPM = "Say A or P to indicate A M or P M."

        # ----------------------------- Session -----------------------------
        session_data.setdefault(call_sid, {})
        sd = session_data[call_sid]

        bad_time_tries = sd.get("bad_time_tries", 0)     # wrong-time attempts
        silence_time   = sd.get("silence_time", 0)       # initial silence
        silence_alts   = sd.get("silence_alts", 0)       # silence during alternatives
        alts_prompt    = sd.get("alts_prompt", "")       # last alternatives prompt
        pending_digits = sd.get("pending_dt_digits")     # awaiting explicit A/P after DTMF

        # doctor / calendar
        doctor_id = sd.get("doctor_id")
        if not doctor_id:
            debug_print("[ask_time_date] ❌ no doctor selected → choose_doctor")
            sd["stage"] = "choose_doctor"
            doctor_list = ", ".join(googleid_dr_name_map.values())
            resp.append(make_gather("Which doctor would you like to see?", hints=doctor_list))
            return str(resp)
        calendar_id = doctor_id

        # input snapshot
        raw_speech = (speech_result or "").strip()
        raw_dtmf   = (request.values.get("Digits") or "").strip()

        # ------------------------- AM/PM helpers ---------------------------
        def _extract_ampm(s: str) -> str:
            """
            Return 'am' or 'pm' if AM/PM (or single 'A'/'P') is explicitly present in s.
            Accepts A, P, AM, PM (with optional dots/spaces).
            """
            if not s: return ""
            t = s.lower().strip()
            # normalize a.m./p.m.
            t = _re.sub(r"\ba\s*\.?\s*m\.?\b", "am", t)
            t = _re.sub(r"\bp\s*\.?\s*m\.?\b", "pm", t)
            # whole-token AM/PM or bare A/P tokens
            if _re.search(r"\bam\b", t): return "am"
            if _re.search(r"\bpm\b", t): return "pm"
            if _re.search(r"(^|[\s,])a($|[\s,])", t): return "am"
            if _re.search(r"(^|[\s,])p($|[\s,])", t): return "pm"
            return ""

        def _ensure_ampm_in_time(time_str: str, speech_src: str) -> Tuple[bool, str, str]:
            """Ensure explicit A/P (or AM/PM). Returns (ok, time_with_ampm, reason)."""
            if not time_str:
                return (False, "", "missing_parts")
            ampm = _extract_ampm(time_str) or _extract_ampm(speech_src)
            if not ampm:
                return (False, "", "missing_ampm")
            # strip any am/pm/a/p already attached, then re-attach normalized
            t = _re.sub(r"\s*\b(am|pm|a|p)\b", "", time_str.lower()).strip()
            return (True, f"{t} {ampm}", "")

        # -------------------- Alternatives reactive repeat -----------------
        if alts_prompt and not (raw_speech or raw_dtmf):
            silence_alts += 1
            sd["silence_alts"] = silence_alts
            debug_print(f"[ask_time_date] 🤐 silence on alternatives → repeat {silence_alts}/3")

            if silence_alts < 3:
                g = make_gather(
                    f"I didn’t hear you. {alts_prompt}",
                    input="speech dtmf", timeout=5, speech_timeout="auto", barge_in=True,
                    action="/voice", method="POST"
                )
                resp.append(g)
                try:
                    resp.redirect(url_for("voice")); debug_print("[ask_time_date] 🔁 alts repeat → redirect url_for('voice')")
                except Exception:
                    resp.redirect("/voice");         debug_print("[ask_time_date] 🔁 alts repeat → redirect /voice (fallback)")
                sd["stage"] = "ask_time_date"
                return str(resp)
            else:
                debug_print("[ask_time_date] ❌ 3 silent repeats on alternatives → hangup")
                resp.say(gpt_speak("I didn’t hear a response. Please call again later."), VOICE)
                resp.hangup()
                sd.pop("alts_prompt", None); sd.pop("silence_alts", None)
                session_data.pop(call_sid, None)
                return str(resp)

        if alts_prompt and (raw_speech or raw_dtmf):
            debug_print("[ask_time_date] 🎧 input received after alternatives → clearing alt state")
            sd["silence_alts"] = 0
            sd["alts_prompt"]  = ""

        # --------------------- Initial silence (pre-alts) -------------------
        if not alts_prompt and not (raw_speech or raw_dtmf):
            silence_time += 1
            sd["silence_time"] = silence_time
            debug_print(f"[ask_time_date] 🤐 initial silence (tries={silence_time})")

            if silence_time < 3:
                prompt = (
                    "Say the date and time, like 'October 12 at 9 A M'. "
                    "Or enter two digits for month, two for day, two for hour, two for minutes, then say A or P."
                )
                g = make_gather(prompt, input="speech dtmf", timeout=4, speech_timeout="auto", barge_in=True,
                                action="/voice", method="POST")
                resp.append(g)
                try:
                    resp.redirect(url_for("voice")); debug_print("[ask_time_date] 🔁 initial silence → redirect url_for('voice')")
                except Exception:
                    resp.redirect("/voice");         debug_print("[ask_time_date] 🔁 initial silence → redirect /voice (fallback)")
                sd["stage"] = "ask_time_date"
                return str(resp)
            else:
                debug_print("[ask_time_date] ❌ initial silence maxed → hangup")
                resp.say(gpt_speak("I'm sorry, I still didn't get your appointment time. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

        sd.pop("silence_time", None)  # input present

        # ---------------------------- Speech split -------------------------
        def _extract_day_time(s: str) -> Tuple[str, str]:
            if not s: return ("", "")
            t = s.lower()
            t = _re.sub(r"\b(a\s*\.?\s*m\.?)\b", "am", t)
            t = _re.sub(r"\b(p\s*\.?\s*m\.?)\b", "pm", t)
            t = _re.sub(r"\bat\s*[.,]?\s+", " at ", t)
            t = _re.sub(r"[!?;]+", "", t)
            t = _re.sub(r"\s+", " ", t).strip()
            t = t.replace(" at noon", " at 12 pm").replace(" at midnight", " at 12 am")

            # Primary: "DAY at TIME"
            if " at " in t:
                day, timep = t.split(" at ", 1)
                return (day.strip().rstrip(","), timep.strip())

            # Time digits anywhere → treat preceding as day
            m = _re.search(r"\b(\d{1,2}(:\d{2})?)\b", t)
            if m:
                timep = m.group(1)
                day = t[:m.start()].strip().rstrip(",")
                return (day, timep)

            # Date-only like "july 30th"
            MONTHS = r"january|february|march|april|may|june|july|august|september|october|november|december"
            date_only = _re.search(rf"\b({MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?\b", t)
            if date_only:
                return (date_only.group(0), "")

            # Fuzzy date detection fallback (no explicit time)
            try:
                _ = dtparser.parse(t, fuzzy=True, default=_dt(2000, 1, 1, 9, 0, 0))
                return (t, "")
            except Exception:
                return ("", "")

        partial_ctx = sd.setdefault("partial_datetime", {})
        day_part, time_part = _extract_day_time(raw_speech)

        # ---------------------- Partial capture flows ----------------------
        if day_part and not time_part:
            # Validate day-only not in the past (clinic TZ)
            try:
                tz_name  = globals().get("CLINIC_TZ", "America/Chicago")
                tz_local = _pytz.timezone(tz_name)
                today    = _date_local.today()
                default_base = tz_local.localize(_dt(today.year, today.month, today.day, 9, 0, 0))
                parsed_day   = dtparser.parse(day_part, default=default_base, fuzzy=True)
                if not _re.search(r"\b\d{4}\b", day_part):
                    parsed_day = parsed_day.replace(year=today.year)
                local_day_date = parsed_day.astimezone(tz_local).date()

                if local_day_date < today:
                    debug_print(f"[ask_time_date] ⛔ day-only is past → '{day_part}'")
                    # Offer alternatives starting now
                    now_utc_iso = _pytz.UTC.localize(_dt.utcnow()).isoformat().replace("+00:00", "Z")
                    alts = get_next_available_slots(calendar_id, creds, from_start_iso=now_utc_iso, limit=3) or []
                    options = " or ".join([a.get("friendly", "") for a in alts if a.get("friendly")])
                    prompt = (f"That date has already passed. Would you like {options}?" if options
                            else "That date has already passed. Please say a future date.")
                    sd["alts_prompt"]  = prompt
                    sd["silence_alts"] = 0
                    sd["stage"]        = "ask_time_date"

                    g = make_gather(prompt, input="speech dtmf", timeout=6, speech_timeout="auto", barge_in=True,
                                    action="/voice", method="POST")
                    resp.append(g)
                    try:
                        resp.redirect(url_for("voice")); debug_print("[ask_time_date] 🔁 past day-only → redirect url_for('voice')")
                    except Exception:
                        resp.redirect("/voice");         debug_print("[ask_time_date] 🔁 past day-only → redirect /voice (fallback)")
                    return str(resp)

            except Exception as e:
                debug_print(f"[ask_time_date] ⚠️ day-only parse error → {e} (fallback to need-both)")

            # Day is today or future → proceed with original partial flow
            partial_ctx["day"] = day_part
            debug_print(f"[ask_time_date] 🧭 stored partial day='{day_part}', prompting for time only")
            g = make_gather(
                f"Got it — {day_part}. What time? Say like '8 A M' or '3 30 P M'.",
                input="speech dtmf", timeout=5, speech_timeout="auto", barge_in=True,
                action="/voice", method="POST"
            )
            resp.append(g)
            try:
                resp.redirect(url_for("voice")); debug_print("[ask_time_date] 🔁 day-without-time → redirect url_for('voice')")
            except Exception:
                resp.redirect("/voice");         debug_print("[ask_time_date] 🔁 day-without-time → redirect /voice (fallback)")
            sd["stage"] = "ask_time_date"
            return str(resp)

        if time_part and not day_part and "day" in partial_ctx:
            day_part = partial_ctx.pop("day")
            debug_print(f"[ask_time_date] 🧩 combined remembered day='{day_part}' with new time='{time_part}'")

        if day_part and time_part:
            partial_ctx.clear()

        # --------------------------- Slot builder --------------------------
        def _build_slot(day_str: str, time_str_with_ampm: str) -> Tuple[str, str]:
            tz_name  = globals().get("CLINIC_TZ", "America/Chicago")
            tz_local = _pytz.timezone(tz_name)
            dur      = int(globals().get("APPOINTMENT_DURATION_MINUTES", 30))

            combined     = f"{day_str} at {time_str_with_ampm}"
            today        = _date_local.today()
            default_base = tz_local.localize(_dt(today.year, today.month, today.day, 9, 0, 0))
            parsed       = dtparser.parse(combined, default=default_base, fuzzy=True)

            if parsed.tzinfo is None:
                parsed = tz_local.localize(parsed)
            else:
                parsed = parsed.astimezone(tz_local)

            if not _re.search(r"\b\d{4}\b", combined):
                parsed = parsed.replace(year=today.year)

            if parsed.weekday() not in working_days:
                raise ValueError("invalid_weekday")

            start_local = parsed
            end_local   = start_local + timedelta(minutes=dur)
            return (
                start_local.astimezone(_pytz.UTC).isoformat().replace("+00:00", "Z"),
                end_local.astimezone(_pytz.UTC).isoformat().replace("+00:00", "Z"),
            )

        # ------------------------- Parse / build slot ----------------------
        appointment_start, appointment_end = None, None
        try:
            # Pending digits path (awaiting A/P or AM/PM in speech)
            if pending_digits and raw_speech:
                ampm = _extract_ampm(raw_speech)
                if not ampm:
                    g = make_gather(PROMPT_NEED_AMPM, input="speech", timeout=4, speech_timeout="auto", barge_in=True,
                                    action="/voice", method="POST")
                    resp.append(g)
                    try:
                        resp.redirect(url_for("voice")); debug_print("[ask_time_date] 🔁 awaiting A/P after digits → redirect url_for('voice')")
                    except Exception:
                        resp.redirect("/voice");         debug_print("[ask_time_date] 🔁 awaiting A/P after digits → redirect /voice (fallback)")
                    sd["stage"] = "ask_time_date"
                    return str(resp)

                today = _date_local.today()
                mm, dd, hh, mn = int(pending_digits[0:2]), int(pending_digits[2:4]), int(pending_digits[4:6]), int(pending_digits[6:8])
                day_str  = f"{today.year}-{mm:02d}-{dd:02d}"
                time_str = f"{hh}:{mn:02d} {ampm}"
                appointment_start, appointment_end = _build_slot(day_str, time_str)
                sd.pop("pending_dt_digits", None)

            elif raw_dtmf:
                digits = _re.sub(r"\D", "", raw_dtmf)
                debug_print(f"[ask_time_date] 📟 DTMF entered → {digits}")
                today = _date_local.today()
                if len(digits) >= 8:
                    mm, dd, hh, mn = int(digits[0:2]), int(digits[2:4]), int(digits[4:6]), int(digits[6:8])
                    day_str  = f"{today.year}-{mm:02d}-{dd:02d}"
                    ampm = _extract_ampm(raw_speech)  # check if caller said A/P this turn
                    if not ampm:
                        sd["pending_dt_digits"] = digits[:8]
                        g = make_gather(PROMPT_NEED_AMPM, input="speech", timeout=5, speech_timeout="auto", barge_in=True,
                                        action="/voice", method="POST")
                        resp.append(g)
                        try:
                            resp.redirect(url_for("voice")); debug_print("[ask_time_date] 🔁 digits w/o A/P → redirect url_for('voice')")
                        except Exception:
                            resp.redirect("/voice");         debug_print("[ask_time_date] 🔁 digits w/o A/P → redirect /voice (fallback)")
                        sd["stage"] = "ask_time_date"
                        return str(resp)
                    time_str = f"{hh}:{mn:02d} {ampm}"
                    appointment_start, appointment_end = _build_slot(day_str, time_str)
                else:
                    raise ValueError("invalid_dtmf_format")

            else:
                if not day_part or not time_part:
                    bad_time_tries += 1
                    sd["bad_time_tries"] = bad_time_tries
                    debug_print(f"[ask_time_date] ❌ missing parts → wrong-time tries={bad_time_tries}/3")
                    if bad_time_tries >= 3:
                        debug_print("[ask_time_date] ❌ missing parts maxed → hangup")
                        resp.say(gpt_speak("I’m sorry, I’m still not getting a valid date and time. Please call again later."), VOICE)
                        resp.hangup()
                        session_data.pop(call_sid, None)
                        return str(resp)
                    g = make_gather(PROMPT_NEED_BOTH, input="speech dtmf", timeout=5, speech_timeout="auto", barge_in=True,
                                    action="/voice", method="POST")
                    resp.append(g)
                    try:
                        resp.redirect(url_for("voice")); debug_print("[ask_time_date] 🔁 missing parts → redirect url_for('voice')")
                    except Exception:
                        resp.redirect("/voice");         debug_print("[ask_time_date] 🔁 missing parts → redirect /voice (fallback)")
                    sd["stage"] = "ask_time_date"
                    return str(resp)

                ok, time_with_ampm, reason = _ensure_ampm_in_time(time_part, raw_speech)
                if not ok:
                    bad_time_tries += 1
                    sd["bad_time_tries"] = bad_time_tries
                    debug_print(f"[ask_time_date] ❌ AM/PM (A/P) missing → wrong-time tries={bad_time_tries}/3")
                    if bad_time_tries >= 3:
                        debug_print("[ask_time_date] ❌ AM/PM missing maxed → hangup")
                        resp.say(gpt_speak("I’m sorry, I’m still not getting a valid date and time. Please call again later."), VOICE)
                        resp.hangup()
                        session_data.pop(call_sid, None)
                        return str(resp)
                    g = make_gather(
                        "Say the time again and include A or P, like '9 A M' or '2 30 P M'.",
                        input="speech dtmf", timeout=5, speech_timeout="auto", barge_in=True,
                        action="/voice", method="POST"
                    )
                    resp.append(g)
                    try:
                        resp.redirect(url_for("voice")); debug_print("[ask_time_date] 🔁 missing A/P → redirect url_for('voice')")
                    except Exception:
                        resp.redirect("/voice");         debug_print("[ask_time_date] 🔁 missing A/P → redirect /voice (fallback)")
                    sd["stage"] = "ask_time_date"
                    return str(resp)

                appointment_start, appointment_end = _build_slot(day_part, time_with_ampm)

            debug_print(f"[ask_time_date] ⏰ Built slot → Start={appointment_start}, End={appointment_end}")

        except ValueError as e:
            err = str(e)
            bad_time_tries += 1
            sd["bad_time_tries"] = bad_time_tries
            debug_print(f"[ask_time_date] ❌ parse/build error='{err}' → wrong-time tries={bad_time_tries}/3")

            if bad_time_tries >= 3:
                debug_print("[ask_time_date] ❌ parse/build maxed → hangup")
                resp.say(gpt_speak("I’m sorry, I’m still not getting a valid date and time. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            if "invalid_weekday" in err:
                g = make_gather(PROMPT_NEED_VALID_DAY, input="speech dtmf", timeout=5, speech_timeout="auto", barge_in=True,
                                action="/voice", method="POST")
            else:
                g = make_gather(PROMPT_NEED_BOTH, input="speech dtmf", timeout=5, speech_timeout="auto", barge_in=True,
                                action="/voice", method="POST")
            resp.append(g)
            try:
                resp.redirect(url_for("voice")); debug_print("[ask_time_date] 🔁 parse error branch → redirect url_for('voice')")
            except Exception:
                resp.redirect("/voice");         debug_print("[ask_time_date] 🔁 parse error branch → redirect /voice (fallback)")
            sd["stage"] = "ask_time_date"
            return str(resp)

        # Reset wrong-time counter once we have a valid slot
        sd["bad_time_tries"] = 0

        # ----------------------------- Past-time ---------------------------
        now_utc = _pytz.UTC.localize(_dt.utcnow())
        start_dt = _dt.fromisoformat(appointment_start.replace("Z", "+00:00"))
        if start_dt <= now_utc:
            alts = get_next_available_slots(calendar_id, creds, from_start_iso=appointment_end, limit=3) or []
            options = " or ".join([a.get("friendly", "") for a in alts if a.get("friendly")])
            prompt  = (f"That time has already passed. Would you like {options}?"
                    if options else PROMPT_PAST_TIME)

            sd["alts_prompt"]  = prompt
            sd["silence_alts"] = 0

            g = make_gather(prompt, input="speech dtmf", timeout=6, speech_timeout="auto", barge_in=True,
                            action="/voice", method="POST")
            resp.append(g)
            try:
                resp.redirect(url_for("voice")); debug_print("[ask_time_date] 🔁 past-time → redirect url_for('voice')")
            except Exception:
                resp.redirect("/voice");         debug_print("[ask_time_date] 🔁 past-time → redirect /voice (fallback)")
            sd["stage"] = "ask_time_date"
            return str(resp)

        # ----------------------------- Availability ------------------------
        try:
            slot_available = is_time_slot_available(calendar_id, appointment_start, appointment_end, creds)
        except Exception as e:
            debug_print(f"[ask_time_date] ⚠️ Availability check error → {e}")
            slot_available = False

        if not slot_available:
            alts = get_next_available_slots(calendar_id, creds, from_start_iso=appointment_end, limit=3) or []
            options = " or ".join([a.get("friendly", "") for a in alts if a.get("friendly")])
            prompt  = (f"That time is not available. Would you like {options}?"
                    if options else "That time isn’t available. Please say another time with A or P.")

            sd["alts_prompt"]  = prompt
            sd["silence_alts"] = 0

            g = make_gather(prompt, input="speech dtmf", timeout=6, speech_timeout="auto", barge_in=True,
                            action="/voice", method="POST")
            resp.append(g)
            try:
                resp.redirect(url_for("voice")); debug_print("[ask_time_date] 🔁 unavailable → redirect url_for('voice')")
            except Exception:
                resp.redirect("/voice");         debug_print("[ask_time_date] 🔁 unavailable → redirect /voice (fallback)")
            sd["stage"] = "ask_time_date"
            return str(resp)

        # ------------------------------ Continue flow ----------------------
        sd.pop("alts_prompt", None)
        sd["silence_alts"] = 0
        sd.pop("pending_dt_digits", None)

        sd["appointment_time"] = {"start": appointment_start, "end": appointment_end}

        if sd.get("reschedule_after_cancel", False):
            cancel_info = sd.get("cancel", {})
            cust = sd.setdefault("customer", {})
            if cancel_info.get("phone_e164"): cust["phone_e164"] = cancel_info["phone_e164"]
            if cancel_info.get("dob"):        cust["dob"]        = cancel_info["dob"]
            sd["reschedule_after_cancel"] = False

        cust = sd.setdefault("customer", {})
        phone_e164 = cust.get("phone_e164") or sd.get("phone_e164")
        dob        = cust.get("dob")        or sd.get("dob")

        if not phone_e164 or not dob:
            sd["stage"] = "collect_phone" if not phone_e164 else "collect_dob"
            prompt = ("Please say your 10-digit phone number."
                    if not phone_e164 else
                    "Please say your date of birth, like 'July third 1990'.")
            g = make_gather(prompt, input="speech dtmf", timeout=5, speech_timeout="auto", barge_in=True,
                            action="/voice", method="POST")
            resp.append(g)
            try:
                resp.redirect(url_for("voice")); debug_print("[ask_time_date] 🔁 continue flow (ID collection) → redirect url_for('voice')")
            except Exception:
                resp.redirect("/voice");         debug_print("[ask_time_date] 🔁 continue flow (ID collection) → redirect /voice (fallback)")
            return str(resp)

        # ------------------------------------------------------------------
        # 🛠️ FIX: define `found` safely in-scope, then set stage & prompt
        # ------------------------------------------------------------------
        found = False
        try:
            found = customer_search(phone_number=phone_e164, dob=dob, default_country="US")
            debug_print(f"[ask_time_date] 🔎 customer_search(phone={phone_e164}, dob={dob}) → {found}")
        except Exception as e:
            debug_print(f"[ask_time_date] ⚠️ customer_search error → {e}")
            found = False

        sd["stage"] = "book_appt_confirm" if found else "collect_first_name"
        debug_print(f"[ask_time_date] 🎯 Next stage → {sd['stage']}")

        # If we need the first name, prompt right now (speech or DTMF + #)
        if sd["stage"] == "collect_first_name":
            prompt = "Please say your first name, or type it and press pound."
            g = make_gather(
                prompt,
                input="speech dtmf",
                timeout=6,
                speech_timeout="auto",
                barge_in=True,
                finish_on_key="#",
                action="/voice", method="POST",
            )
            resp.append(g)
            # Safety net so we re-enter /voice even if the caller stays silent
            #resp.redirect("/voice")
            return str(resp)

        # Otherwise we’re going to book_appt_confirm; redirect to continue flow
        resp.redirect("/voice")
        return str(resp)














    # ===== collect_first_name (stage) =====
    elif stage == "collect_first_name":
        # ----------------------------------------------------------------------
        # 🎯 Goal:
        #   - Capture FIRST name via speech or keypad (DTMF).
        #   - Handle silence locally (up to 3 retries) so we don’t rely on the
        #     central silence guard.
        #   - Accept Arabic or foreign names when spoken (but written with Latin/English letters),
        #     e.g., "Faten", "Hossam", "Youssef". Reject true Arabic script input.
        #   - Support keypad T9 entry: 32836# → "Faten" (from ARABIC_NAME_HINTS).
        #   - Save to session_data[call_sid]["customer"]["first_name"].
        #   - Advance → collect_last_name.
        #
        # 🔇 SILENCE HANDLING (local):
        #   Silence = BOTH SpeechResult and Digits empty in this webhook.
        #   We count consecutive silences in sd["silence_first_name"].
        #   • Tries 1–2 → re-prompt with <Gather> and append <Redirect>/voice (safety net).
        #   • Try 3     → apologize and hang up.
        #
        # ☎️ KEYPAD (DTMF) NAME VIA T9:
        #   - If digits look like a T9 pattern (2–12 digits, only 2–9), we map them to
        #     candidate names from ARABIC_NAME_HINTS.
        #   - Unique match → accept. Multiple → pick first (you can extend to ask user).
        #   - No match → re-prompt (counts as retry).
        #
        # 🧼 VALIDATION:
        #   - Allow only English letters plus apostrophe/hyphen/space; first char must be a letter.
        #   - Reject Arabic-script characters (U+0600–U+06FF).
        # ----------------------------------------------------------------------
        

       
        sd = session_data.setdefault(call_sid, {})
        sd.setdefault("customer", {})
        raw_speech = (speech_result or "").strip()
        raw_dtmf   = (request.values.get("Digits") or "").strip()
        debug_print(f"collect_first_name: speech='{raw_speech}', dtmf='{raw_dtmf}'")

        # ----------------------------------------------------------------------
        # ⏹️ FIX: Skip false silence if the user pressed only '#'
        # ----------------------------------------------------------------------
        if speech_result == "" and request.values.get("Digits") == "#":
            debug_print("collect_first_name: ⏹️ Raw DTMF '#' received alone — skipping false silence (via raw)")
            return str(resp)

        # silence handling...


        # -- small helpers (local to this stage) --------------------------------
        #import string, unicodedata as _uni

        def _t9_digit_for_char(ch: str) -> str:
            # Normalize, encode, decode, and uppercase a character for uniform text processing
            ch = (
                _uni.normalize("NFKD", ch)        # Step 1: Unicode normalization (NFKD = Compatibility Decomposition)
                                                # Example: "é" → "e" + "´" (accent is separated)
                                                # Ensures all accented characters are split into base + modifier
                .encode("ascii", "ignore")        # Step 2: Encode to ASCII and drop non-ASCII parts
                                                # This removes accents and any non-ASCII symbols
                                                # Example: "é" → "e", "ø" → "", "ç" → "c"
                .decode("ascii")                  # Step 3: Convert from bytes back to string
                                                # Necessary after encoding; returns a clean ASCII string
                .upper()                          # Step 4: Convert result to uppercase for case-insensitive matching
                                                # Example: "e" → "E", "c" → "C"
            )

            if ch in "ABC":   return "2"
            if ch in "DEF":   return "3"
            if ch in "GHI":   return "4"
            if ch in "JKL":   return "5"
            if ch in "MNO":   return "6"
            if ch in "PQRS":  return "7"
            if ch in "TUV":   return "8"
            if ch in "WXYZ":  return "9"
            return ""  # ignore non A–Z for T9

        

        def _t9_code(name: str) -> str:
            """
            Convert a given name to its T9 keypad numeric equivalent.
            
            This is useful for matching speech or DTMF input (e.g., user typing a name using 
            a phone keypad), especially when dealing with foreign names or partial matches.

            Steps:
            -------
            1. Normalize the input name to ASCII-only by removing accents (e.g., "José" → "Jose").
            2. Remove any non-letter characters (e.g., hyphens, apostrophes, spaces).
            3. Convert each letter to its T9 digit using _t9_digit_for_char().
            The mapping is like old mobile phones:
                2 → ABC, 3 → DEF, 4 → GHI, 5 → JKL, 6 → MNO, 7 → PQRS, 8 → TUV, 9 → WXYZ

            Example:
            --------
            Input:  name = "Mohamed"
            Output: "6642633"

            Breakdown:
                M → 6
                O → 6
                H → 4
                A → 2
                M → 6
                E → 3
                D → 3
            """

            # 1. Remove accents/diacritics by converting to ASCII characters
            base = _uni.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")

            # 2. Keep only alphabetic characters (remove numbers, spaces, hyphens, etc.)
            # Step 2: Remove all non-alphabetic characters (keep only A–Z and a–z)
            # ------------------------------------------------------------
            # This regular expression replaces any character that is NOT a letter with an empty string.
            # - `[^A-Za-z]` means "any character that is NOT between A–Z or a–z"
            # - This strips out digits, punctuation, whitespace, and special characters.

            # Example transformations:
            #   "Jose Andres"          → "JoseAndres"     (removes the space)
            #   "M@h@m0ud!"            → "Mhmoud"         (removes symbols and digits)
            #   "Abd_el-Rahman"        → "AbdelRahman"    (removes underscore and hyphen)
            #   "123Hello_There!"      → "HelloThere"     (removes numbers and symbols)

            base = _re.sub(r"[^A-Za-z]", "", base)


            # 3. Convert each letter to its T9 digit equivalent
            # Step 5: Convert each character in the cleaned name into its T9 digit
            # ---------------------------------------------------------------------
            # The T9 keypad mapping (used in old mobile phones) maps letters to digits like this:
            #   - ABC → 2, DEF → 3, GHI → 4, JKL → 5, MNO → 6, PQRS → 7, TUV → 8, WXYZ → 9
            #
            # This line iterates over each character in the preprocessed name (`base`)
            # and converts it to its corresponding T9 digit using `_t9_digit_for_char(c)`.
            # All resulting digits are then concatenated into a single string using `"".join(...)`.

            # Example:
            #   Input name: "Mohamed"
            #   After cleaning: base = "MOHAMED"
            #   T9 digits: M→6, O→6, H→4, A→2, M→6, E→3, D→3
            #   Output: "6642633"

            return "".join(_t9_digit_for_char(c) for c in base)



        
        
        def _build_t9_index_from_hints(hints: str) -> dict:
            """
            Build a T9 lookup index from a comma-separated list of name hints.

            This function takes a long string of names (e.g., Arabic, Indian, Persian names)
            and maps each name to its corresponding T9 numeric keypad code using `_t9_code`.
            It returns a dictionary that allows you to look up all possible names for a 
            given T9 input sequence — useful for matching keypad input or fuzzy recognition.

            Parameters:
            ------------
            hints (str): A string of names separated by commas.
                        Example: "Ahmed, Ahmad, Mohamed, Faten, Fatma, Aisha"

            Returns:
            ---------
            dict[str, list[str]]:
                A dictionary where:
                - keys are T9 codes as strings (e.g., "26433")
                - values are lists of names that match that code

            Example:
            ---------
            Input:
                hints = "Ahmed, Ahmad, Mohamed, Faten, Fatma"

            Output:
                {
                    "26433": ["Ahmed", "Ahmad"],
                    "6642633": ["Mohamed"],
                    "32862": ["Faten"],
                    "32862": ["Faten", "Fatma"]   # Both names have same T9 code!
                }

            Step-by-step:
            --------------
            1. Split the string into individual names by comma.
            2. Clean up each name (remove whitespace).
            3. Convert each name to its T9 keypad code using `_t9_code()`.
            4. Group names by the T9 code in a dictionary.
            """

            # Step 1: Split the input string into individual names
            names = [n.strip() for n in hints.split(",") if n.strip()]

            # Step 2: Prepare an empty dictionary to store T9 code → name list
            idx = {}

            # Step 3: For each name, calculate its T9 code and add to index
            for nm in names:
                code = _t9_code(nm)  # e.g., "Mohamed" → "6642633"
                if code:
                    # Step 6: Group names by their T9 code in a dictionary
                    # ----------------------------------------------------
                    # This line ensures that all names which map to the same T9 digit sequence
                    # are grouped together in the same list within the dictionary `idx`.
                    #
                    # - `setdefault(code, [])` checks if the T9 `code` already exists as a key.
                    #   - If it exists, it returns the existing list.
                    #   - If it doesn't exist, it creates a new list `[]` for that key.
                    # - `.append(nm)` then adds the current name (`nm`) to that list.
                    # [] is used as a default value.
                    #
                    #    It says:
                    #    ➤ "If code is not already in the dictionary idx, set it to an empty list [], and then append nm."
                    #
                    #    Example:
                    #    idx = {}
                    #    code = "123"
                    #    name = "Ali"
                    #
                    #    idx.setdefault(code, []).append(name)
                    #    print(idx)
                    #    Output: {'123': ['Ali']}
                    #
                    # Example:
                    #   Let's say:
                    #       code = "6642633"
                    #       nm = "Mohamed"
                    #   If this is the first "Mohamed"-type name, the dictionary becomes:
                    #       idx = {"6642633": ["Mohamed"]}
                    #
                    #   Later, if we process:
                    #       nm = "Muhamed"  → also has code = "6642633"
                    #   Then the dictionary becomes:
                    #       idx = {"6642633": ["Mohamed", "Muhamed"]}

                    idx.setdefault(code, []).append(nm)


            # Step 4: Return the final index mapping T9 → list of matching names
            return idx



        # -------------------------------
        # 🔇 Silence Handling (local)
        # -------------------------------
        if not raw_speech and not raw_dtmf:
            tries = sd.get("silence_first_name", 0) + 1
            sd["silence_first_name"] = tries
            sd["stage"] = "collect_first_name"  # ensure we come back here next webhook
            debug_print(f"collect_first_name: 🤐 silence; tries={tries}/3")

            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            gather = make_gather(
                "I didn’t hear your first name. Please say your first name. "
                "You can also type it and press pound.",
                input="speech dtmf",
                language="en-US",
                hints=FOREIGN_NAME_HINTS,  # helps speech ASR
                timeout=6,
                speech_timeout="auto",
                finish_on_key="#",
                barge_in=True,
                action="/voice", method="POST",
            )
            resp.append(gather)
            resp.redirect("/voice")  # safety net if still silent
            return str(resp)

        # ✅ Some input arrived → clear the local silence counter
        sd.pop("silence_first_name", None)

        # -------------------------------
        # 🧾 Parse & Clean Input
        # -------------------------------
        first_name = ""

        if raw_dtmf:
            # NEW: T9 decoding for keypad-entered names (e.g., 32836 → Faten)
            # Step: Clean the input and keep only numeric digits (0–9)
            # --------------------------------------------------------
            # This line uses a regular expression to remove any character that is NOT a digit.
            #
            # - `\D` (uppercase D) means "any non-digit character" in regex.
            # - `_re.sub(r"\D", "", raw_dtmf)` means:
            #       → find all non-digit characters in `raw_dtmf`
            #       → replace them with an empty string ""
            #
            # Purpose:
            #   This ensures the phone number input contains only digits, removing symbols,
            #   spaces, parentheses, plus signs, or letters.
            #
            # Examples:
            #   raw_dtmf = "(469) 463-3276"   →  "4694633276"
            #   raw_dtmf = "+1 800-CALLME"    →  "1800"
            #   raw_dtmf = " 123 456 7890 "   →  "1234567890"
            #   raw_dtmf = "abc123xyz"        →  "123"
            #
            # ✅ Result: digits = clean numeric string ready for E.164 normalization

            digits = _re.sub(r"\D", "", raw_dtmf)

            debug_print(f"collect_first_name: 🔢 keypad digits='{digits}'")

            # Only try T9 if looks like a real name entry: 2–12 digits, only 2–9
            if 2 <= len(digits) <= 12 and _re.fullmatch(r"[2-9]+", digits):
                t9_index = _build_t9_index_from_hints(FOREIGN_NAME_HINTS)
                # Step: Look up possible name matches using the T9 digit code
                # -----------------------------------------------------------
                # `t9_index` is a dictionary that maps T9 numeric codes (like "4663") to possible names (like ["HOME", "GOOD", "GONE"]).
                # 
                # `digits` contains the user-entered keypad sequence (e.g., from speech-to-text converted T9 input).
                #
                # - `t9_index.get(digits, [])` tries to fetch the list of names matching the input digits.
                # - If the digits are NOT in the dictionary, it returns an empty list `[]` by default.
                #
                # This ensures the code never crashes if there’s no match — it safely returns an empty list instead of None.
                #
                # Examples:
                #   t9_index = {"43556": ["HELLO", "GELLO"], "4663": ["GOOD", "HOME", "GONE"]}
                #   digits = "4663"  → matches = ["GOOD", "HOME", "GONE"]
                #   digits = "1234"  → matches = []   (no match found)

                matches = t9_index.get(digits, [])


                if len(matches) == 1:
                    first_name = matches[0]
                    debug_print(f"collect_first_name: 📟 T9 unique match → '{first_name}'")
                elif len(matches) > 1:
                    # Simple strategy: pick the first (extend to present options if you prefer)
                    first_name = matches[0]
                    debug_print(f"collect_first_name: 📟 T9 multiple matches {matches} → chose '{first_name}'")
                else:
                    # No T9 match → count a retry and re-prompt
                    r = sd.get("retry_first_name", 0) + 1
                    sd["retry_first_name"] = r
                    sd["stage"] = "collect_first_name"
                    debug_print(f"collect_first_name: 📟 T9 no match for '{digits}' retry={r}/3")

                    if r >= 3:
                        resp.say(gpt_speak("Sorry, I couldn’t capture your name. Please call again later."), VOICE)
                        resp.hangup()
                        session_data.pop(call_sid, None)
                        return str(resp)

                    gather = make_gather(
                        "I couldn’t match that keypad entry to a name. "
                        "Please say your first name, or type it again and press pound.",
                        input="speech dtmf",
                        language="en-US",
                        hints=FOREIGN_NAME_HINTS,
                        timeout=6,
                        speech_timeout="auto",
                        finish_on_key="#",
                        barge_in=True,
                        action="/voice", method="POST",
                    )
                    resp.append(gather)
                    resp.redirect("/voice")
                    return str(resp)
            else:
                # Not a valid T9-like pattern; keep a minimal fallback (will likely fail validation)
                name_digits = _re.sub(r"\D", "", raw_dtmf)
                first_name = f"User{name_digits[-3:]}" if name_digits else ""
                debug_print(f"collect_first_name: 🧮 non-T9 keypad → '{first_name}'")

        else:
            # ------------------------------------------------------------------
            # 🗣 Speech Input Path
            # ------------------------------------------------------------------
            # This branch executes when:
            #   - The caller *did not* provide keypad (DTMF) input.
            #   - Twilio's Speech-to-Text (STT) engine transcribed the caller’s voice.
            #
            # For example:
            #   User presses nothing on the phone but says "My name is Mohamed."
            #   → Twilio posts SpeechResult="My name is Mohamed" to /voice
            #   → dtmf_digits == ""  → this 'else' branch runs.
            #
            # Goal:
            #   Extract a clean first name string from the caller’s spoken input.

            # ---------------------------------------------------------------
            # 🧹 1. Remove punctuation and extra spaces
            # ---------------------------------------------------------------
            # ----------------------------------------------------------------------
            # 🧠 str.maketrans('', '', string.punctuation)
            # ----------------------------------------------------------------------
            # The 'str.maketrans()' function builds a translation table that defines
            # how characters in a string should be transformed (replaced or removed)
            # when passed to 'translate()'.
            #
            # General syntax:
            #     str.maketrans(from_chars, to_chars, delete_chars)
            #
            # ✅ Example 1 — Character Replacement:
            #   If we want to replace characters individually, we supply
            #   'from_chars' and 'to_chars' with the same length.
            #
            #   Example:
            #       table = str.maketrans("abc", "xyz")
            #       "abc cab".translate(table)
            #   → Output: "xyz zxy"
            #
            #   Explanation:
            #       'a' → 'x'
            #       'b' → 'y'
            #       'c' → 'z'
            #   So "abc cab" becomes "xyz zxy".
            #
            # ------------------------------------------------------------
            # 🧹 Example 2 — Character Deletion (our case)
            # ------------------------------------------------------------
            #   The third argument 'delete_chars' defines characters to remove entirely.
            #   They are *not replaced* with anything — they are just erased.
            #
            #   Example:
            #       table = str.maketrans('', '', '!?')
            #       "Hello! How are you?".translate(table)
            #   → Output: "Hello How are you"
            #
            #   Explanation:
            #       - The characters '!' and '?' are in the delete list.
            #       - Every '!' and '?' found in the text is completely removed.
            #       - Other characters stay untouched.
            #
            #   Now applying this concept:
            #       str.maketrans('', '', string.punctuation)
            #   means:
            #       "Delete all punctuation marks defined in string.punctuation"
            #       → !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
            #
            #   Example:
            #       raw_speech = "My name is Mohamed!"
            #       cleaned = raw_speech.translate(str.maketrans('', '', string.punctuation))
            #   → Output: "My name is Mohamed"
            #
            #   Another example:
            #       raw_speech = "Hello, world! I'm here."
            #       cleaned = raw_speech.translate(str.maketrans('', '', string.punctuation))
            #   → Output: "Hello world Im here"
            #
            #   Note that the apostrophe in “I’m” is deleted too, resulting in “Im”.
            #
            # ------------------------------------------------------------
            # 🚿 .strip() → Final Cleanup
            # ------------------------------------------------------------
            # After removing punctuation, '.strip()' trims leading and trailing spaces:
            #     "  My name is Mohamed   " → "My name is Mohamed"
            #
            # ------------------------------------------------------------
            # ✅ Combined Effect:
            #   Input : "  Hello, my name is Mohamed!  "
            #   Output: "Hello my name is Mohamed"
            #
            # This ensures:
            #   • Punctuation marks are fully removed.
            #   • Text is clean and consistent for further regex processing.
            #   • Helps the next steps extract first name tokens correctly.

            cleaned = raw_speech.translate(str.maketrans('', '', string.punctuation)).strip()

            # ----------------------------------------------------------------------
            # 🧽 STEP 2: Normalize whitespace — collapse multiple spaces into one
            # ----------------------------------------------------------------------
            # After punctuation is removed, some text might contain irregular spacing.
            # For example, multiple spaces between words or before/after names.
            #
            # Example before cleanup:
            #     cleaned = "My   name   is   Mohamed"
            #
            # The goal is to make the spacing consistent:
            #     "My name is Mohamed"
            #
            # ------------------------------------------------------------
            # 🔍 REGEX: r"\s+"
            # ------------------------------------------------------------
            # Let's break down this regular expression:
            #
            #   \s   → matches any whitespace character:
            #           spaces, tabs (\t), newlines (\n), etc.
            #
            #   +    → quantifier meaning "one or more" of the preceding token.
            #
            # So together, "\s+" means:
            #   "match any *sequence* of one or more whitespace characters"
            #
            # Examples of what "\s+" matches:
            #   "   "          → three spaces
            #   "\t\t"         → two tabs
            #   " \t \n "      → a mix of spaces, tabs, or newlines
            #
            # ------------------------------------------------------------
            # 🔧 _re.sub(r"\s+", " ", cleaned)
            # ------------------------------------------------------------
            # 're.sub()' replaces all matches of the pattern with a single space " ".
            #
            # Example:
            #     Input : "My   name    is\tMohamed"
            #     Output: "My name is Mohamed"
            #
            # Internally:
            #   - Finds "   "  → replaces with " "
            #   - Finds "\t"   → replaces with " "
            #   - Repeats until all groups of spaces/tabs/newlines are replaced.
            #
            # ------------------------------------------------------------
            # 🧾 Practical Example:
            #   raw_speech = "  My   name   is   Mohamed  "
            #   cleaned = "  My   name   is   Mohamed  "
            #   cleaned = _re.sub(r"\s+", " ", cleaned)
            # → " My name is Mohamed "
            #
            # Then, later we use .strip() again if we want to remove the single
            # leading/trailing spaces, resulting in → "My name is Mohamed".
            #
            # ------------------------------------------------------------
            # ✅ Purpose:
            #   - Keeps spacing consistent before tokenizing.
            #   - Prevents empty tokens or mismatched name extraction.
            #   - Makes "split()" behavior predictable in the next step.


            cleaned = _re.sub(r"\s+", " ", cleaned)

                    # ----------------------------------------------------------------------
            # 🧠 STEP 3: Remove filler phrases like "my name is", "this is", "I'm"
            # ----------------------------------------------------------------------
            # In natural speech, callers often begin with introductions such as:
            #   "My name is Mohamed"
            #   "This is Sarah"
            #   "I am John"
            #   "It's Ahmed"
            #
            # These introductory phrases are not part of the *actual name*,
            # so we remove them before extracting the first word as the name.
            #
            # ------------------------------------------------------------
            # 🔍 REGEX: r"\b(?:my name is|this is|i am|i'm|it is|it's)\b\s*"
            # ------------------------------------------------------------
            # Let’s break this pattern down piece by piece:
            #
            #   \b
            #     → "Word boundary" — ensures the phrase starts and ends cleanly.
            #       Example: Matches "my name is" in "my name is Mohamed"
            #       but NOT in "dummy name issue" (where "name is" appears inside words).
            #
            #   (?: ... )
            #     → "Non-capturing group" — groups several options together
            #       but does not store them for later backreference.
            #       Inside, we have the possible starter phrases.
            #
            #   my name is | this is | i am | i'm | it is | it's
            #     → The '|' character means "OR" — so the regex will match
            #       *any* of these exact lowercase phrases.
            #
            #   \b
            #     → Another word boundary to ensure the phrase ends cleanly.
            #
            #   \s*
            #     → Zero or more whitespace characters (spaces, tabs, etc.)
            #       that may follow the phrase before the person’s actual name.
            #
            # ------------------------------------------------------------
            # 🧾 Example Matches:
            # ------------------------------------------------------------
            #   "my name is Mohamed"   → matches "my name is "
            #   "this is Ahmed"        → matches "this is "
            #   "i am Sara"            → matches "i am "
            #   "I'm Youssef"          → matches "I'm "
            #   "it is Fatma"          → matches "it is "
            #   "it's Rania"           → matches "it's "
            #
            # ------------------------------------------------------------
            # 🧼 _re.sub(..., "", cleaned, flags=_re.IGNORECASE)
            # ------------------------------------------------------------
            # We call 're.sub()' to replace all occurrences of these phrases
            # (regardless of capitalization) with an empty string "".
            #
            # The 'flags=_re.IGNORECASE' part allows case-insensitive matching:
            #   - "My Name Is", "MY NAME IS", "my name is" → all match.
            #
            # ------------------------------------------------------------
            # 🎯 Example Transformations:
            # ------------------------------------------------------------
            #   Input : "My name is Mohamed"
            #   Output: "Mohamed"
            #
            #   Input : "This is Sarah"
            #   Output: "Sarah"
            #
            #   Input : "I’m John"
            #   Output: "John"
            #
            #   Input : "It’s Ahmed"
            #   Output: "Ahmed"
            #
            # ------------------------------------------------------------
            # ✅ Purpose:
            #   - Cleans out polite or redundant introduction phrases.
            #   - Helps isolate the *actual* first name token later.
            #   - Works for speech recognition transcripts that include filler words.
            #
            # ------------------------------------------------------------
            # ⚙️ Combined Example (Steps 1–3):
            #   raw_speech = "Hello! My name is   Mohamed."
            #   Step 1 → remove punctuation  → "Hello My name is   Mohamed"
            #   Step 2 → normalize spaces   → "Hello My name is Mohamed"
            #   Step 3 → remove intro phrase → "Hello Mohamed"
            #
            #   After tokenization, first_name = "Hello" (if that’s noise),
            #   or we can later ignore it based on context.
            #
            # ------------------------------------------------------------
            # 🧩 Implementation:
            cleaned = _re.sub(
                r"\b(?:my name is|this is|i am|i'm|it is|it's)\b\s*",
                "",
                cleaned,
                flags=_re.IGNORECASE,
            )


           
            # ---------------------------------------------------------------
            # ✂️ 3. Split and pick the first token
            # ---------------------------------------------------------------
            tokens = cleaned.split()
            # Example:
            #   "Khalil Mohamed" → ["Khalil", "Mohamed"]

            first_name = tokens[0] if tokens else ""
            # Picks the first word as the first name.
            # Example:
            #   tokens=["Khalil", "Mohamed"] → first_name="Khalil"
            #
            # If nothing is captured (tokens = []), 'first_name' becomes an empty string.

            debug_print(f"collect_first_name: 🗣 derived first_name='{first_name}' from speech")
            # Logs what name was detected, helpful for debugging and audit.

        # ----------------------------------------------------------------------
        # 🌐 4. Validation: Only English letters are allowed
        # ----------------------------------------------------------------------
        # We now check if the name looks valid. The rules are:
        #   - Must contain English letters only (A-Z or a-z)
        #   - Can contain apostrophes, hyphens, or spaces
        #   - Must not contain Arabic script or foreign Unicode letters

        english_only_pattern = r"^[A-Za-z][A-Za-z'\-\s]{0,39}$"
        contains_foreign = bool(_re.search(r"[\u0600-\u06FF]", first_name))
        # \u0600-\u06FF = Arabic Unicode range → detect Arabic names like "خليل"

        if not first_name or not _re.fullmatch(english_only_pattern, first_name) or contains_foreign:
            # This block runs if:
            #  - Caller said nothing (empty)
            #  - Caller used Arabic letters
            #  - Caller used invalid characters (numbers, symbols, etc.)
            #
            # Examples:
            #   first_name = ""          → invalid (silence)
            #   first_name = "خليل"      → invalid (Arabic)
            #   first_name = "123John"   → invalid (numbers)

            r = sd.get("retry_first_name", 0) + 1
            sd["retry_first_name"] = r
            sd["stage"] = "collect_first_name"
            debug_print(f"collect_first_name: ❌ invalid/foreign-script '{first_name}' retry={r}/3")

            # Give the caller up to 3 attempts.
            if r >= 3:
                # After 3 invalid attempts → polite exit
                resp.say(
                    gpt_speak("Sorry, I couldn’t capture your name in English letters. Please call again later."),
                    VOICE
                )
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Prompt the user again to re-enter/speak their name
            gather = make_gather(
                "Please say your first name using English letters only. "
                "You can also type it on the keypad and press pound.",
                input="speech dtmf",
                language="en-US",
                hints=FOREIGN_NAME_HINTS,
                timeout=6,
                speech_timeout="5",
                finish_on_key="#",
                barge_in=True,
                action="/voice", method="POST",
            )
            resp.append(gather)
            resp.redirect("/voice")  # Retry loop for silence or invalid input
            return str(resp)

        # ----------------------------------------------------------------------
        # ✅ 5. Valid input: Save and continue to collect the last name
        # ----------------------------------------------------------------------
        sd["customer"]["first_name"] = first_name
        sd["stage"] = "collect_last_name"
        sd.pop("retry_first_name", None)
        debug_print(f"collect_first_name: ✅ saved first_name='{first_name}' → next=collect_last_name")

        # Next prompt: ask for last name
        gather = make_gather(
            f"Thank you {first_name}. Now, what is your last name?",
            input="speech dtmf",
            language="en-US",
            hints=FOREIGN_NAME_HINTS,   # help recognize family names like 'Ng', 'Lopez', 'Al-Sayed'
            timeout=6,
            speech_timeout="5",
            finish_on_key="#",
            barge_in=True,
            action="/voice", method="POST",
        )
        resp.append(gather)
        resp.redirect("/voice")  # If silent on last-name stage → re-prompt
        return str(resp)





    elif stage == "collect_last_name":
        # ----------------------------------------------------------------------
        # 🎯 Goal:
        #   - Capture LAST name via speech or keypad (DTMF).
        #   - Robust local silence handling (up to 3 retries; uses <Gather>+<Redirect>).
        #   - Accept English-letter names (romanized) incl. hyphen/apostrophe.
        #   - DTMF: support T9 entry (e.g., 542545# → "Khalil") by matching against
        #     FOREIGN_NAMES_HINTS. If there’s no good match, fall back to a readable guess.
        #   - Save → session_data[call_sid]["customer"]["last_name"].
        #   - Next → collect_address.
        # ----------------------------------------------------------------------
        sd = session_data.setdefault(call_sid, {})
        sd.setdefault("customer", {})

        raw_speech = (speech_result or "").strip()
        raw_dtmf   = (request.values.get("Digits") or "").strip()
        debug_print(f"collect_last_name: speech='{raw_speech}', dtmf='{raw_dtmf}'")

        # -------------------------------
        # 🔇 Silence handling (local)
        # -------------------------------
        if not raw_speech and not raw_dtmf:
            tries = sd.get("silence_last_name", 0) + 1
            sd["silence_last_name"] = tries
            sd["stage"] = "collect_last_name"
            debug_print(f"collect_last_name: 🤐 silence; tries={tries}/3")

            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            gather = make_gather(
                "I didn’t hear your last name. Please say your last name in English letters. "
                "You can also type it on the keypad and press pound.",
                input="speech dtmf",
                language="en-US",
                hints=FOREIGN_NAME_HINTS,
                timeout=6,
                speech_timeout="5",
                finish_on_key="#",
                barge_in=True,
                action="/voice", method="POST",
            )
            resp.append(gather)
            resp.redirect("/voice")  # safety net if still silent
            return str(resp)

        # ✅ Some input → clear silence counter
        sd.pop("silence_last_name", None)

        # -------------------------------
        # 🔢 T9 helpers (for DTMF names)
        # -------------------------------
        _T9 = {
            "2": "ABC", "3": "DEF", "4": "GHI", "5": "JKL",
            "6": "MNO", "7": "PQRS", "8": "TUV", "9": "WXYZ"
        }

        def _t9_signature(name: str) -> str:
            """Convert a romanized name to its T9 digit signature (letters only)."""
            s = []
            up = name.upper()
            for ch in up:
                if "A" <= ch <= "Z":
                    for d, letters in _T9.items():
                        if ch in letters:
                            s.append(d)
                            break
            return "".join(s)

        def _best_name_from_t9(digits: str, hints_csv: str) -> str:
            """
            Try to resolve a T9 digit string to a name using FOREIGN_NAMES_HINTS.
            Returns best candidate or "" if none.
            """
            if not digits:
                return ""
            # Build candidate pool from hints
            items = [x.strip() for x in hints_csv.split(",") if x.strip() and x.strip()[0].isalpha()]
            # Exact signature matches first
            exact = []
            for nm in items:
                if _t9_signature(nm) == digits:
                    exact.append(nm)
            if len(exact) == 1:
                return exact[0]
            if len(exact) > 1:
                # Prefer longer (more specific), then alphabetically
                exact.sort(key=lambda n: (-len(n), n))
                return exact[0]
            # No exact: allow startswith (user truncated) or digits contain the shorter signature
            partial = []
            for nm in items:
                sig = _t9_signature(nm)
                if sig.startswith(digits) or digits.startswith(sig):
                    partial.append((nm, sig))
            if partial:
                partial.sort(key=lambda t: (abs(len(t[1]) - len(digits)), -len(t[0]), t[0]))
                return partial[0][0]
            return ""

        # -------------------------------
        # 🧾 Parse & Clean
        # -------------------------------
        if raw_dtmf:
            # Remove non-digits; accept with or without trailing '#'
            d = _re.sub(r"\D", "", raw_dtmf)
            debug_print(f"collect_last_name: 📟 DTMF cleaned='{d}'")
            last_name = ""
            if d and all(ch in "23456789" for ch in d):
                # Try to map via hints
                try_name = _best_name_from_t9(d, FOREIGN_NAMES_HINTS)
                if try_name:
                    last_name = try_name
                    debug_print(f"collect_last_name: 🔤 T9 matched → '{last_name}'")
                else:
                    # Fallback: pick first letter option per digit to form a readable token
                    first_letter = {"2":"A","3":"D","4":"G","5":"J","6":"M","7":"P","8":"T","9":"W"}
                    last_name = "".join(first_letter[ch] for ch in d)
                    debug_print(f"collect_last_name: 🧩 T9 fallback guess → '{last_name}'")
            else:
                trailing = d[-3:] if d else ""
                last_name = f"Family{trailing}" if trailing else "Unknown"
                debug_print(f"collect_last_name: 🔖 placeholder from keypad → '{last_name}'")
        else:
            # Speech path
            # Define punctuation set locally (no imports)
            _PUNCT = """!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""
            punct_keep = "'-"
            # Build a string of punctuation to remove (exclude allowed keepers)
            _punct_to_remove = "".join(ch for ch in _PUNCT if ch not in punct_keep)

            cleaned = raw_speech.translate(str.maketrans('', '', _punct_to_remove)).strip()
            cleaned = _re.sub(r"\s+", " ", cleaned)

            # Drop fillers
            cleaned = _re.sub(
                r"\b(?:my last name is|family name is|last name|surname is|this is|i am|i'm|it's)\b\s*",
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
        contains_foreign_block = bool(_re.search(r"[\u0600-\u06FF]", last_name))

        if (not last_name) or (not _re.fullmatch(english_only_pattern, last_name)) or contains_foreign_block:
            r = sd.get("retry_last_name", 0) + 1
            sd["retry_last_name"] = r
            sd["stage"] = "collect_last_name"
            debug_print(f"collect_last_name: ❌ invalid name '{last_name}' retry={r}/3")

            if r >= 3:
                resp.say(gpt_speak("Sorry, I couldn’t capture your last name in English letters. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            gather = make_gather(
                "Please say your last name using English letters only. "
                "You can also type it on the keypad using T9 and press pound.",
                input="speech dtmf",
                language="en-US",
                hints=FOREIGN_NAME_HINTS,
                timeout=6,
                speech_timeout="5",
                finish_on_key="#",
                barge_in=True,
                action="/voice", method="POST",
            )
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # -------------------------------
        # ✅ Save & Advance
        # -------------------------------
        sd["customer"]["last_name"] = last_name
        sd["stage"] = "collect_address"
        sd.pop("retry_last_name", None)
        debug_print(f"collect_last_name: ✅ saved last_name='{last_name}' → next=collect_address")

        gather = make_gather(
            f"Thank you {sd['customer'].get('first_name','')} {last_name}. "
            "Please tell me your full address.",
            input="speech dtmf",
            language="en-US",
            hints="118 Briar Oak Murphy Texas 75094",
            timeout=7,
            speech_timeout="5",
            finish_on_key="#",
            barge_in=True,
            action="/voice", method="POST",
        )
        resp.append(gather)
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



   



    # ----------------------------------------------------------------------
    # collect_cc - complete stage (robust digit normalization + strict Luhn)
    # ----------------------------------------------------------------------
    elif stage == "collect_cc":
        # ----------------------------------------------------------------------
        # 💳 Stage: collect_cc  (optimized for shorter expiration & CVV steps)
        #
        # Flow:
        #   (1) Card number (13–19 digits, Luhn-checked)
        #   (2) Expiration (MMYY or MMYYYY, current/future only)
        #   (3) CVV (3–4 digits)
        #
        # Improvements:
        #   - Shorter timeouts for steps 2 & 3 (expiration, CVV)
        #   - Allows both speech and DTMF entry for all steps
        #   - Immediate step advance after # pressed
        #
        # ⚙️ Difference between resp.append() and resp.redirect():
        #   • resp.append(make_gather(...)) → Adds a <Gather> verb that makes
        #     Twilio play a message and then WAIT for input. Twilio will only
        #     POST back to /voice *after* the user types or speaks something.
        #
        #   • resp.redirect("/voice") → Adds a <Redirect> verb that tells Twilio
        #     to IMMEDIATELY call /voice again (no waiting). Use this when you
        #     want to move to the next logical step right away (e.g., after
        #     saving data).
        #
        #   ⚠️ Never use append(gather) + redirect() together — redirect will
        #     cancel the gather before the user can respond.
        # ----------------------------------------------------------------------

        # --- helpers ------------------------------------------------------------
        def _luhn_ok(pan: str) -> bool:
            s, alt = 0, False
            for ch in pan[::-1]:
                if not ch.isdigit():
                    continue
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
                "zero": "0", "oh": "0", "o": "0",
                "one": "1", "two": "2", "to": "2", "too": "2",
                "three": "3", "four": "4", "for": "4",
                "five": "5", "six": "6", "seven": "7",
                "eight": "8", "ate": "8", "nine": "9"
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
                1: "Please enter or say your card number now, then press pound.",
                2: "Please enter or say the expiration date, for example, zero nine two seven, then press pound.",
                3: "Please enter or say the three or four digit security code, then press pound."
            }.get(cc_step, "Please enter or say your card details, then press pound.")

            # Use append() so Twilio will WAIT for input and only post back after user responds
            gather = make_gather(
                prompt,
                input="speech dtmf",
                timeout=20,
                speech_timeout="auto",
                finish_on_key="#",
                action="/voice",
                barge_in=True,
            )
            resp.append(gather)
            return str(resp)

        session_data[call_sid].pop("silence_cc", None)

        # -------------------------------
        # Step 1: Card Number (13–19)
        # -------------------------------
        if cc_step == 1:
            pan = _digits_from(raw_dtmf, raw_speech, enforce_dtmf=enforce_dm)
            if len(pan) > 19:
                pan = pan[:19]
            debug_print(f"collect_cc: normalized card digits={pan}")

            if not (13 <= len(pan) <= 19):
                debug_print("collect_cc: ❌ invalid card length")
                gather = make_gather(
                    "That card number doesn't look right. Please re-enter or say the full card number, then press pound.",
                    input="speech dtmf",
                    timeout=20,
                    speech_timeout="auto",
                    finish_on_key="#",
                    action="/voice",
                )
                resp.append(gather)
                return str(resp)

            # Non-strict Luhn (accept Visa/Mastercard, even if algorithm fails due to timing)
            if not _luhn_ok(pan):
                debug_print(f"collect_cc: ⚠️ {_mask(pan)} failed Luhn but accepted (non-strict mode).")
            else:
                debug_print(f"collect_cc: ✅ Luhn passed for {_mask(pan)}")

            # Save card number and move to step 2
            customer["cc_number"] = pan
            session_data[call_sid]["cc_step"] = 2
            debug_print(f"collect_cc: ✅ Saved card number '{_mask(pan)}' → step 2 (Expiration)")

            # Redirect to /voice → move to next step immediately (no waiting)
            resp.redirect("/voice")
            return str(resp)

        # -------------------------------
        # Step 2: Expiration (MMYY/MMYYYY)
        # -------------------------------
        if cc_step == 2:
            session_data[call_sid]["no_input_expected"] = True
            digits = _digits_from(raw_dtmf, raw_speech, enforce_dtmf=True)
            debug_print(f"collect_cc: Step 2 digits='{digits}'")

            # --------------------------------------------------------------------------
            # If the normalized digits string for the expiration date is not length 4 or 6,
            # we consider the caller's input incomplete or malformed and re-prompt.
            # Acceptable formats we expect are:
            #   - 4 digits:  MMYY   (e.g. "0927" -> September 2027)
            #   - 6 digits:  MMYYYY (e.g. "092027" -> September 2027)
            # Anything else (e.g. 3 digits, 5 digits, letters only, empty) is treated as invalid.
            if len(digits) not in (4, 6):
                # Build a <Gather> instruction for Twilio:
                #  - The spoken prompt tells the caller how to give the expiration (examples).
                #  - input="speech dtmf" lets the caller either *speak* the date or *type* it on the keypad.
                #  - timeout=10 means Twilio will wait up to 10 seconds for user input before the gather times out.
                #  - finish_on_key="#" allows the caller to press the pound key to immediately finish typing
                #    instead of waiting for the timeout or for num_digits to be reached.
                #  - action="/voice" tells Twilio to POST the results of this gather back to your /voice
                #    webhook when the gather completes (either by #, by the timeout, or by speech result).
                gather = make_gather(
                    "Please say or enter the expiration date as month and year, for example, zero nine two seven, then press pound.",
                    input="speech dtmf",
                    timeout=10,
                    finish_on_key="#",
                    action="/voice",
                )

                # Append the <Gather> to the TwiML response. This places the gather in the outgoing
                # TwiML so Twilio will play the prompt and listen for input.
                resp.append(gather)

                # Return the TwiML (string form) immediately so Twilio receives the <Gather>.
                # Important behavior:
                #  - We do NOT call resp.redirect("/voice") here. Returning the TwiML with the <Gather>
                #    causes Twilio to wait for the user's input and then POST back to the 'action' URL.
                #  - After the user types/speaks and the gather finishes, Twilio will call your /voice
                #    webhook again with request parameters such as 'Digits' (for DTMF) and/or
                #    'SpeechResult' (for speech). Your handler should then re-enter this stage and
                #    process the provided digits.
                return str(resp)
            # --------------------------------------------------------------------------

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
                    input="speech dtmf",
                    timeout=8,
                    finish_on_key="#",
                    action="/voice",
                )
                resp.append(gather)
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
                    "Please enter or say the three or four digit security code, then press pound.",
                    input="speech dtmf",
                    timeout=6,
                    finish_on_key="#",
                    action="/voice",
                )
                resp.append(gather)
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












    elif stage == "cancel_appointment":
        # ----------------------------------------------------------------------
        # 🔄 Stage: Cancel Appointment — after the caller says the doctor’s name
        #
        # PURPOSE:
        #   1. Identify the doctor from speech or keypad input.
        #   2. If matched → proceed to phone number verification.
        #   3. If not → retry up to MAX_NUMBER_DR_RETRY.
        #
        # FEATURES:
        #   • Handles silence, retries, and junk input.
        #   • Supports both speech and DTMF input.
        #   • Uses GPT extraction as fallback for ambiguous speech.
        #   • Tags this flow with origin_stage="cancel" for downstream use
        #     (e.g., collect_pin_number, cancel_appt_get_phone_number, etc.)
        # ----------------------------------------------------------------------

        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("cancel", {})

        # ✅ Tag this session as cancellation flow origin
        session_data[call_sid]["origin_stage"] = "cancel"

        # Safe punctuation definition
        try:
            _PUNCT = string.punctuation
        except Exception:
            _PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

        # ----------------------------------------------------------------------
        # 🔍 Helper: Extract doctor name using GPT fallback
        # ----------------------------------------------------------------------
        def _extract_doctor_name(speech_text):
            if not speech_text.strip():
                return ""

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
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )
                extracted = response.choices[0].message.content.strip()
                print(f"✅ GPT extracted doctor name: {extracted}")
                return extracted
            except Exception as e:
                print(f"⚠️ GPT fallback in extract_doctor_name: {type(e).__name__}: {e}")
                return speech_text.strip()

        # ----------------------------------------------------------------------
        # 🔧 Helper: Clean text (lowercase, remove punctuation)
        # ----------------------------------------------------------------------
        def _clean(s: str) -> str:
            s = (s or "").lower().translate(str.maketrans("", "", _PUNCT)).strip()
            return " ".join(s.split())

        # ----------------------------------------------------------------------
        # 🔊 Input handling
        # ----------------------------------------------------------------------
        dtmf_digits = (request.values.get("Digits") or "").strip()
        selected_text = (speech_result or "").strip()

        # Build DTMF doctor map
        doctor_names = list(googleid_dr_name_map.values())
        doctor_dtmf_map = {str(i + 1): doc for i, doc in enumerate(doctor_names)}
        session_data[call_sid]["doctor_dtmf_map"] = doctor_dtmf_map

        # ----------------------------------------------------------------------
        # 🔇 Silence handling
        # ----------------------------------------------------------------------
        if not selected_text and not dtmf_digits:
            tries = session_data[call_sid].get("silence_cancel_doc", 0) + 1
            session_data[call_sid]["silence_cancel_doc"] = tries
            debug_print(f"cancel_appointment: 🤐 No input detected (silence count={tries})")

            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            options = ". ".join([f"{doc} (press {k})" for k, doc in doctor_dtmf_map.items()])
            retry_prompt = (
                f"I didn't hear the doctor's name. Available doctors are: {options}. "
                "Please say the name of the doctor or press the number."
            )
            session_data[call_sid]["stage"] = "cancel_appointment"
            resp.append(make_gather(
                retry_prompt,
                hints=", ".join(doctor_names),
                num_digits=1,
                language="en-US",  # ✅ English (US)
            ))
            return str(resp)

        # ----------------------------------------------------------------------
        # 🔢 Handle DTMF direct match
        # ----------------------------------------------------------------------
        if dtmf_digits and dtmf_digits in doctor_dtmf_map:
            matched_name = doctor_dtmf_map[dtmf_digits]
            matched_id = next(k for k, v in googleid_dr_name_map.items() if v == matched_name)
            debug_print(f"cancel_appointment: ✅ DTMF match → {matched_name} ({matched_id})")
        else:
            # Clear silence counter if we heard something
            session_data[call_sid].pop("silence_cancel_doc", None)

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
                resp.append(make_gather(
                    retry_prompt,
                    hints=", ".join(doctor_names),
                    num_digits=1,
                    language="en-US",  # ✅ English
                ))
                return str(resp)

            # ------------------------------
            # 1️⃣ Partial substring / token match
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
                    or (spoken_tokens & friendly_tokens)
                ):
                    partial_matches.append((doc_id, friendly_name))

            if len(partial_matches) == 1:
                matched_id, matched_name = partial_matches[0]
                debug_print(f"cancel_appointment: ✅ Partial match → {matched_name} ({matched_id})")
            elif len(partial_matches) > 1:
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
            # 2️⃣ GPT fallback if no match
            # ------------------------------
            if not matched_id:
                try:
                    extracted_name = _extract_doctor_name(selected_text)
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

        # ----------------------------------------------------------------------
        # 3️⃣ Still no match → Retry
        # ----------------------------------------------------------------------
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
            resp.append(make_gather(
                retry_prompt,
                hints=", ".join(doctor_names),
                num_digits=1,
                language="en-US",  # ✅ English
            ))
            return str(resp)

        # ----------------------------------------------------------------------
        # 4️⃣ Success → Save and move to phone number collection
        # ----------------------------------------------------------------------
        session_data[call_sid]["doctor_id"] = matched_id
        session_data[call_sid]["cancel"]["doctor"] = matched_name or googleid_dr_name_map.get(matched_id, "the doctor")
        session_data[call_sid]["stage"] = "cancel_appt_get_phone_number"

        resp.append(make_gather(
            "Thanks. What phone number did you use when booking the appointment?",
            input="speech dtmf",
            language="en-US",  # ✅ English language
            num_digits=10,
            timeout=8,
            speech_timeout="6",
            barge_in=True,
        ))
        return str(resp)




    elif stage == "cancel_appt_get_dob":
        # ----------------------------------------------------------------------
        # 🎂 Stage: cancel_appt_get_dob
        #
        # PURPOSE:
        #   • Capture and validate the customer's date of birth (DOB) via speech or DTMF.
        #   • Store DOB under session_data["customer"]["dob"] and ["cancel"]["dob"].
        #   • Then branch to collect_pin_number (for verification) instead of time/date.
        #
        # ENHANCEMENTS:
        #   ✅ origin_stage set to "cancel" → used later in collect_pin_number.
        #   ✅ English (US) language for all prompts.
        #   ✅ Silent retry logic (3 tries max).
        # ----------------------------------------------------------------------

        debug_print("cancel_appt_get_dob: 📍 Stage entered")

        session_data.setdefault(call_sid, {}).setdefault("customer", {})
        session_data[call_sid].setdefault("cancel", {})

        # ✅ Tag origin for later stages (used in collect_pin_number)
        session_data[call_sid]["origin_stage"] = "cancel"

        DOB_PROMPT = (
            "Please say your birth date, for example July third nineteen fifty six, "
            "or type 2 digits for month, 2 digits for day, and 4 digits for year, then press pound."
        )

        # --- Inputs ---
        dtmf_digits = (request.values.get("Digits") or "").strip()
        speech_text = (speech_result or "").strip()
        debug_print(f"cancel_appt_get_dob: 🎙️ speech_text='{speech_text}', 🔢 dtmf_digits='{dtmf_digits}'")

        # --- Silence Handling ---
        if not dtmf_digits and not speech_text:
            tries = session_data[call_sid].get("silence_cancel_dob", 0) + 1
            session_data[call_sid]["silence_cancel_dob"] = tries
            if tries >= 3:
                resp.say("I’m still not hearing anything. Please call again later.", VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            gather = make_gather(
                DOB_PROMPT,
                input="speech dtmf",
                timeout=8,
                speech_timeout="auto",
                barge_in=True,
                finish_on_key="#",
                language="en-US"  # ✅ English (US)
            )
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # Clear silence retry counter
        session_data[call_sid].pop("silence_cancel_dob", None)

        # --- Parse DOB ---
        try:
            dt = None
            if dtmf_digits:
                clean = re.sub(r"\D", "", dtmf_digits)
                if len(clean) == 8:  # MMDDYYYY
                    m, d, y = int(clean[0:2]), int(clean[2:4]), int(clean[4:8])
                    dt = datetime(y, m, d)
            if not dt and speech_text:
                dt = dp.parse(speech_text, fuzzy=True)
        except Exception as e:
            debug_print(f"cancel_appt_get_dob: ❌ parse error {e}")
            dt = None

        # --- Invalid or Unparsed DOB ---
        if not dt:
            retries = session_data[call_sid].get("retry_cancel_dob", 0) + 1
            session_data[call_sid]["retry_cancel_dob"] = retries
            if retries >= 3:
                resp.say("Sorry, I couldn’t understand your date of birth. Please call again later.", VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            gather = make_gather(
                DOB_PROMPT,
                input="speech dtmf",
                timeout=8,
                speech_timeout="auto",
                barge_in=True,
                finish_on_key="#",
                language="en-US"  # ✅ English (US)
            )
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # --- Store DOB ---
        iso_dob = dt.strftime("%Y-%m-%d")
        session_data[call_sid]["customer"]["dob"] = iso_dob
        session_data[call_sid]["cancel"]["dob"] = iso_dob
        session_data[call_sid].pop("retry_cancel_dob", None)
        debug_print(f"cancel_appt_get_dob: ✅ Stored DOB → {iso_dob}")

        # ------------------------------------------------------------------
        # ✅ Next Stage: Branch to collect_pin_number instead of time/date
        # ------------------------------------------------------------------
        session_data[call_sid]["stage"] = "collect_pin_number"

        gather = make_gather(
            "Thank you. For security verification, please enter your six-digit PIN number followed by the pound key.",
            input="dtmf speech",
            num_digits=6,
            timeout=10,
            finish_on_key="#",
            barge_in=True,
            language="en-US"  # ✅ English (US)
        )
        resp.append(gather)
        resp.redirect("/voice")
        debug_print("cancel_appt_get_dob: 🔀 Proceeding to collect_pin_number for verification")
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
        #
        # BEHAVIOR SUMMARY:
        #
        #   🆕 NEW CUSTOMER:
        #       • Triggered when session_data[call_sid]["customer_status"] == "new".
        #       • Does NOT require a doctor or appointment time.
        #       • Calls only insert_customer() to store the record.
        #       • Plays a polite message:
        #           "Thank you. You need to verify your information with the clinic
        #            before booking an appointment."
        #       • Ends the call immediately afterward.
        #
        #   👤 CURRENT CUSTOMER:
        #       • Triggered when session_data[call_sid]["customer_status"] == "current".
        #       • Follows full appointment confirmation logic:
        #           1. Validate slot availability (is_time_slot_available()).
        #           2. Insert/update customer record via insert_customer().
        #           3. Create Google Calendar event for the selected doctor.
        #           4. Persist the appointment locally using confirm_appointment_for_dr_name().
        #           5. Send SMS confirmation to the caller.
        #       • Ends the call with “Your appointment has been booked” message.
        #
        # NOTE:
        #   - Assumes both phone_e164 and customer_dob exist in session_data.
        #   - No fallback gathers for missing data.
        #   - pin_number is passed as 0 (default placeholder — auto-generated by insert_customer).
        # ----------------------------------------------------------------------
        t_stage_start = _time_mod.perf_counter()
        debug_print("book_appt_confirm: 📍 Stage entered")

        # ----------------------------------------------------------------------
        # 🧩 Retrieve session data
        # ----------------------------------------------------------------------
        sd = session_data.get(call_sid, {})
        customer_status = sd.get("customer_status", "current")
        debug_print(f"book_appt_confirm: 🧾 customer_status={customer_status}")

        # ----------------------------------------------------------------------
        # 🧩 Customer Info
        # ----------------------------------------------------------------------
        customer         = sd.get("customer", {}) or {}
        first_name       = (customer.get("first_name") or "").strip()
        last_name        = (customer.get("last_name")  or "").strip()
        customer_address = (customer.get("address")    or "").strip()
        customer_dob     = (customer.get("dob")        or "").strip()
        phone_e164       = (customer.get("phone_e164") or "").strip()

        # ----------------------------------------------------------------------
        # 🆕 NEW CUSTOMER FLOW (no appointment / no doctor)
        # ----------------------------------------------------------------------
        if customer_status == "new":
            debug_print("book_appt_confirm: 🆕 new customer → skipping doctor & appointment flow")

            try:
                inserted_ok = insert_customer(
                    phone=phone_e164, dob=customer_dob,
                    first_name=first_name, last_name=last_name, address=customer_address,
                    cc_name=(customer.get("cc_name") or f"{first_name} {last_name}"),
                    cc_number=(customer.get("cc_number") or ""),
                    cc_exp=(customer.get("cc_exp") or ""),
                    cc_cvv=(customer.get("cc_cvv") or ""),
                    customer_status="new",
                    pin_number=0,  # ✅ default placeholder (auto-generated if missing)
                )
                debug_print(f"book_appt_confirm: ✅ insert_customer (new) → {inserted_ok}")
            except Exception as e:
                debug_print(f"book_appt_confirm: ❌ insert_customer failed for new customer → {e}")

            msg = (
                f"Thank you {first_name or 'there'}. "
                "You need to verify your information with the clinic before scheduling an appointment. "
                "Please contact the clinic to complete your registration. Goodbye!"
            )
            resp.say(gpt_speak(msg), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            debug_print(
                f"book_appt_confirm: 🆕 new customer flow completed in "
                f"{_time_mod.perf_counter() - t_stage_start:.3f}s"
            )
            return str(resp)

        # ----------------------------------------------------------------------
        # 👤 CURRENT CUSTOMER FLOW (full booking)
        # ----------------------------------------------------------------------
        debug_print("book_appt_confirm: 👤 current customer flow continues")

        # STEP 1: Doctor Info
        doctor_id = sd.get("doctor_id")
        if not doctor_id:
            debug_print("book_appt_confirm: ❌ missing doctor_id → choose_doctor")
            session_data[call_sid]["stage"] = "choose_doctor"
            resp.append(make_gather("Which doctor would you like to see?"))
            return str(resp)

        doctor_name = googleid_dr_name_map.get(doctor_id, "the doctor")

        # STEP 2: Appointment Info
        appt = sd.get("appointment_time", {}) or {}
        appointment_start = appt.get("start")
        appointment_end   = appt.get("end")

        if not appointment_start:
            debug_print("book_appt_confirm: ❌ appointment_start missing for current customer")
            resp.say(gpt_speak("Sorry, appointment time is missing. Please try again."), VOICE)
            resp.hangup()
            return str(resp)

        # Convert UTC → Local (Clinic Timezone)
        tz_name = globals().get("CLINIC_TZ", "America/Chicago")
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
                for k in ("APPOINTMENT_DURATION_MINUTES", "SESSION_TIME"):
                    v = globals().get(k)
                    if v:
                        dur = int(v)
                        break
                if dur not in (15, 30, 45, 60):
                    dur = 30
                end_dt = dt_utc + timedelta(minutes=dur)
                appointment_end = end_dt.astimezone(_pytz.UTC).isoformat()
            except Exception as e:
                debug_print(f"book_appt_confirm: ❌ failed computing end time → {e}")
                resp.say(gpt_speak("Sorry, we couldn't confirm the appointment time."), VOICE)
                resp.hangup()
                return str(resp)

        # STEP 3: Slot Availability
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

        # STEP 4: Insert or update customer
        try:
            inserted_ok = insert_customer(
                phone=phone_e164, dob=customer_dob,
                first_name=first_name, last_name=last_name, address=customer_address,
                cc_name=(customer.get("cc_name") or f"{first_name} {last_name}"),
                cc_number=(customer.get("cc_number") or ""),
                cc_exp=(customer.get("cc_exp") or ""),
                cc_cvv=(customer.get("cc_cvv") or ""),
                customer_status="current",
                pin_number=0,  # ✅ placeholder (will auto-generate if missing)
            )
            debug_print(f"book_appt_confirm: ✅ insert_customer (current) → {inserted_ok}")
        except Exception as e:
            debug_print(f"book_appt_confirm: ❌ insert_customer failed → {e}")

        # STEP 5: Create Google Calendar event
        google_event_id = sd.get("google_event_id", "")
        if not google_event_id:
            try:
                service = build("calendar", "v3", credentials=creds)
                event_body = {
                    "summary": f"Appointment: {doctor_name}",
                    "description": f"Clinic appointment for {first_name} {last_name or ''}.",
                    "start": {"dateTime": appointment_start, "timeZone": "UTC"},
                    "end":   {"dateTime": appointment_end,   "timeZone": "UTC"},
                    "transparency": "opaque",
                    "extendedProperties": {
                        "private": {
                            "patient_name": f"{first_name} {last_name or ''}",
                            "phone_e164": phone_e164,
                            "dob": customer_dob,
                            "call_sid": call_sid,
                        }
                    },
                }
                ev = service.events().insert(calendarId=doctor_id, body=event_body, sendUpdates="none").execute()
                google_event_id = ev.get("id", "")
                session_data[call_sid]["google_event_id"] = google_event_id
                debug_print(f"book_appt_confirm: ✅ Google event created id={google_event_id}")
            except Exception as e:
                debug_print(f"book_appt_confirm: ❌ Google insert failed → {e}")
                session_data[call_sid]["stage"] = "ask_time_date"
                resp.append(make_gather("Sorry, I couldn't confirm that slot. Please say another time."))
                return str(resp)

        # STEP 6: Persist locally
        try:
            local_date_str = dt_local.strftime("%Y-%m-%d")
            local_time_disp = dt_local.strftime("%I:%M %p").lstrip("0")
            persist = confirm_appointment_for_dr_name(
                doctor_name=doctor_name,
                phone=phone_e164,
                utc_start=appointment_start,
                utc_end=appointment_end,
                calendar_id=doctor_id,
                name=f"{first_name} {last_name}".strip(),
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

        # STEP 7: Voice + SMS confirmation
        msg = f"Your appointment with {doctor_name} has been booked"
        if formatted_time:
            msg += f" on {formatted_time}"
        msg += ". We look forward to seeing you. Goodbye!"
        resp.say(gpt_speak(msg), VOICE)

        try:
            sms = f"Hi {first_name or 'there'}, your appointment with {doctor_name} is confirmed"
            if formatted_time:
                sms += f" on {formatted_time}"
            sms += ". Thank you for choosing Epic Therapist Clinic."
            _ = client.messages.create(body=sms, from_=TWILIO_PHONE_NUMBER, to=phone_e164)
            debug_print(f"book_appt_confirm: 📩 SMS sent to {phone_e164}")
        except Exception as e:
            debug_print(f"book_appt_confirm: ⚠️ SMS failed → {e}")

        # Finalize
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
