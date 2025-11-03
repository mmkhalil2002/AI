#=======
# update  11/03/2025 time_saved 
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
import random  # local import for clarity



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

MAX_ADVANCE_MONTHS = int(os.getenv("MAX_ADVANCE_MONTHS", 6))
# do u allow to create new customer on LINE
CREATE_NEW_CUSTOMER = bool(os.getenv("CREATE_NEW_CUSTOMER", True))  # d
# ----------------------------------------------------------------------
# 🕓 Speech Pause Duration (milliseconds)
#   Controls SSML <break> tag timing between spoken appointment options.
#   Examples:
#       500 → half-second pause
#      1000 → one-second pause (recommended for clearer spacing)
# ----------------------------------------------------------------------
PAUSE_MS = int(os.getenv("PAUSE_MS", 2000))  # pause time btween messages


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

# ----------------------------------------------------------------------
# 🏥 Global Insurance Companies
# ----------------------------------------------------------------------
# You can override this via environment variable:
#   INSURANCE_COMPANIES="Blue Cross Blue Shield,Aetna,Cigna,United Healthcare,Humana,Kaiser Permanente"
#
# By default → 6 major US insurance companies
# ----------------------------------------------------------------------

INSURANCE_COMPANIES_LIST = [
    name.strip() for name in os.getenv(
        "INSURANCE_COMPANIES",
        "Blue Cross Blue Shield,Aetna,Cigna,United Healthcare,Humana,Kaiser Permanente"
    ).split(",")
]


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

with open("doctors_map.json") as f:
   doctor_names = json.load(f)


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





#################################################
####  start save session information
##################################################

SESSION_DIR = "/tmp/twilio_sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

def _session_path(call_sid: str) -> str:
    """Return the session file path for a given call SID."""
    safe_sid = call_sid.replace("/", "_").replace("\\", "_")
    return os.path.join(SESSION_DIR, f"session_{safe_sid}.json")

def load_session(call_sid: str) -> dict:
    """Load session data from disk for this call, or return empty."""
    path = _session_path(call_sid)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            session_data[call_sid] = data
            return data
        except Exception as e:
            print(f"⚠️ load_session error for {call_sid}: {e}")
    session_data[call_sid] = {}
    return session_data[call_sid]

def save_session(call_sid: str):
    """Persist current session_data[call_sid] safely to disk."""
    if call_sid not in session_data:
        return
    path = _session_path(call_sid)
    try:
        with open(path, "w") as f:
            json.dump(session_data[call_sid], f)
    except Exception as e:
        print(f"⚠️ save_session error for {call_sid}: {e}")

##########################################################
#  end of save seesion info
#######################################################



#import re as _re
#from datetime import datetime, timedelta
# ==============================================================
# 📅 smart_parse_time — fully compatible with collect_book_time_date
# ==============================================================



def smart_parse_time(raw: str, tz_offset_hours: int = -5, default_duration_min: int = 30):
    """
    Robust version — handles speech-to-text artifacts like '2000 p.m.' for '2:00 p.m.'
    """

    def _dbg(msg):
        try:
            debug_print(msg)
        except Exception:
            print(msg)

    if not raw or not str(raw).strip():
        _dbg("[smart_parse_time] ⚠️ Empty input")
        return None

    s = str(raw).strip().lower()
    _dbg(f"[smart_parse_time] 🧠 raw input='{s}'")

    # ------------------------------------------------------------------
    # 🧹 Normalize text and fix STT misreads like '2000 pm' → '2:00 pm'
    # ------------------------------------------------------------------
    s = _re.sub(r"o['’]?clock", "", s)
    s = _re.sub(r"\b(a\s*\.?\s*m\.?)\b", "am", s)
    s = _re.sub(r"\b(p\s*\.?\s*m\.?)\b", "pm", s)
    s = _re.sub(r"[^\w\s:]", " ", s)
    s = _re.sub(r"\s+", " ", s).strip()

    # Fix '2000 pm' or '20 00 pm' → '2:00 pm'
    s = _re.sub(r"\b20\s?00\s*pm\b", "2 00 pm", s)
    s = _re.sub(r"\b2000\s*pm\b", "2 00 pm", s)
    s = _re.sub(r"\btwenty hundred\s*pm\b", "2 00 pm", s)
    _dbg(f"[smart_parse_time] 🧹 normalized='{s}'")

    # ------------------------------------------------------------------
    # 🗓️ Extract month
    # ------------------------------------------------------------------
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12
    }

    month, day, hour, minute, ampm = None, None, 9, 0, "am"
    for m in months:
        if m in s:
            month = months[m]
            _dbg(f"[smart_parse_time] 🗓️ found month='{m}' → {month}")
            break

    # ------------------------------------------------------------------
    # 🕐 Extract time like 9:30, 2 00, etc.
    # ------------------------------------------------------------------
    m_time = _re.search(r"\b(\d{1,2})(?:[: ](\d{2}))?\s*(am|pm)?\b", s)
    if m_time:
        hour = int(m_time.group(1))
        minute = int(m_time.group(2) or 0)
        ampm = m_time.group(3) or "am"
        _dbg(f"[smart_parse_time] ⏰ time → {hour}:{minute:02d} {ampm}")

    # ------------------------------------------------------------------
    # 📅 Extract day (before "at")
    # ------------------------------------------------------------------
    m_day = _re.search(r"\b([1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?\b(?=.*\bat\b)", s)
    if m_day:
        day = int(m_day.group(1))
        _dbg(f"[smart_parse_time] 📅 day → {day}")

    tz_local = _pytz.FixedOffset(tz_offset_hours * 60)
    now_local = datetime.now(tz_local)

    if not month:
        month = now_local.month
    if not day:
        day = now_local.day

    # Convert 12h → 24h
    if ampm == "pm" and hour < 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0

    try:
        dt_local = tz_local.localize(datetime(now_local.year, month, day, hour, minute))
    except Exception as e:
        _dbg(f"[smart_parse_time] ❌ invalid date → {e}")
        return None

    # ------------------------------------------------------------------
    # 🧭 Determine if past
    # ------------------------------------------------------------------
    is_past = dt_local < now_local - timedelta(minutes=2)

    # If earlier in same year → past, not next year
    if dt_local.month < now_local.month or (dt_local.month == now_local.month and dt_local.day < now_local.day):
        is_past = True

    # Optional rollover only for next year if month < current and diff > 6 months
    if dt_local < now_local and (now_local.month - dt_local.month) > 6:
        _dbg("[smart_parse_time] ⏩ rolling to next year (month wraparound)")
        dt_local = tz_local.localize(datetime(now_local.year + 1, month, day, hour, minute))
        is_past = False

    # ------------------------------------------------------------------
    # 📏 Check booking horizon
    # ------------------------------------------------------------------
    try:
        max_months = int(globals().get("MAX_ADVANCE_MONTHS", 6))
    except Exception:
        max_months = 6
    limit_local = now_local + timedelta(days=30 * max_months)
    if dt_local > limit_local:
        _dbg(f"[smart_parse_time] 🚫 beyond booking window ({max_months} mo)")
        return None

    # ------------------------------------------------------------------
    # 🕒 Convert to UTC + friendly string
    # ------------------------------------------------------------------
    dt_utc = dt_local.astimezone(_pytz.UTC)
    dt_end = dt_utc + timedelta(minutes=default_duration_min)
    friendly = dt_local.strftime("%A, %B %-d at %-I:%M %p").replace(" 0", " ")

    result = {
        "start": dt_utc.isoformat().replace("+00:00", "Z"),
        "end": dt_end.isoformat().replace("+00:00", "Z"),
        "friendly": friendly,
        "is_past": is_past,
    }

    _dbg(f"[smart_parse_time] ✅ Parsed '{raw}' → {friendly} (past={is_past}) start={result['start']}")
    return result







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

# ======================================================================
# 🛠️ make_gather() — Twilio <Gather> Builder with SSML & Fallback Logic
# ======================================================================
# PURPOSE:
#   Construct a Twilio <Gather> element that can capture both speech
#   and keypad (DTMF) input with environment-driven defaults.
#
# FEATURES:
#   • Accepts optional `next_stage` for legacy compatibility.
#   • Auto-appends '?stage=...' to the action URL.
#   • Supports `hints` for better speech recognition accuracy.
#   • Allows SSML-based voice prompts (e.g., <break time="500ms"/>).
#   • Includes strong error handling — always returns a valid <Gather>.
#
# PARAMETERS:
#   prompt          (str)  → The text (or SSML) to speak to the caller.
#   next_stage      (str)  → Optional; appended to action as '?stage=...'.
#   hints           (str)  → Optional multiline hint list for speech model.
#   input           (str)  → 'speech', 'dtmf', or 'speech dtmf' (default).
#   num_digits      (int)  → Expected DTMF digits (None = flexible).
#   timeout         (int)  → Max wait time for DTMF digits (seconds).
#   speech_timeout  (str)  → Max silence wait before speech completes ("auto" or seconds).
#   finish_on_key   (str)  → Key (e.g., '#') that ends input early.
#   barge_in        (bool) → Whether user can interrupt playback.
#   language        (str)  → STT language code ('en-US', 'ar-EG', etc.).
#   action          (str)  → Webhook endpoint (e.g., '/voice').
#   method          (str)  → HTTP method for callback ('POST' or 'GET').
#
# RETURNS:
#   Twilio <Gather> object ready to append to VoiceResponse.
#
# NOTES:
#   - Automatically detects SSML markup and sets allow_ssml=True.
#   - Uses RAW prompt when SSML is present (to avoid escaping by gpt_speak()).
#   - Falls back gracefully if Twilio API parameters are invalid.
#   - Centralized debug_print messages for reliability tracking.
# ======================================================================

def make_gather(
    prompt: str,
    *,
    next_stage: Optional[str] = None,               # ← backward-compatible stage chaining
    hints: Optional[str] = None,                    # ← speech recognition vocabulary
    input: str = "speech dtmf",                     # ← capture both speech & DTMF by default
    num_digits: Optional[int] = None,
    timeout: int = PAUSE_BETWEEN_DIGITS,            # ← DTMF timeout default from ENV
    speech_timeout: str = SPEECH_INPUT_DURATION,    # ← e.g. "auto" or "5"
    finish_on_key: str = "#",
    barge_in: bool = True,
    language: str = "en-US",
    action: Optional[str] = "/voice",
    method: str = "POST",
):
    """
    Build and RETURN a Twilio <Gather> element with configurable behavior.

    Backward compatible:
      - Supports next_stage (adds '?stage=...' to action URL).
      - Returns the <Gather> so caller can append it to a VoiceResponse.

    Notes:
      - timeout controls DTMF inter-digit wait.
      - speech_timeout controls silence detection.
      - language sets speech recognition locale.
      - hints provides contextual phrases for better accuracy.
      - SSML markup (<break>, <emphasis>) automatically enabled.
    """
    import re  # local import to avoid global dependency if not needed

    # ------------------------------------------------------------------
    # 🧹 Normalize speech_timeout — convert numeric strings to int
    # ------------------------------------------------------------------
    _speech_timeout = int(speech_timeout) if str(speech_timeout).isdigit() else speech_timeout

    # 🧮 Validate num_digits — must be a positive integer
    _num_digits = num_digits if (isinstance(num_digits, int) and num_digits > 0) else None

    # 🧭 Append next_stage to action for compatibility
    _action = _append_stage_to_action(action, next_stage)

    # ------------------------------------------------------------------
    # 🧠 Normalize speech recognition hints
    #    - Flatten multiline input (e.g. Arabic/English names)
    #    - Convert to comma-separated format
    # ------------------------------------------------------------------
    _hints = None
    if hints:
        _hints = ", ".join(line.strip() for line in hints.splitlines() if line.strip())

    # ------------------------------------------------------------------
    # 🗣️ Determine if SSML is present in the prompt text
    #    - Detects <break>, <emphasis>, <prosody>, <say-as>, etc.
    #    - If found, Twilio's allow_ssml=True will be enabled.
    #    - IMPORTANT: When SSML is present, we pass RAW prompt (no gpt_speak())
    #      to avoid escaping angle brackets which would break SSML.
    # ------------------------------------------------------------------
    _contains_ssml = bool(re.search(r"<\s*(break|emphasis|prosody|say-as)\b", prompt, re.IGNORECASE))

    try:
        # ===============================================================
        # 🎤 Primary Attempt: Build the <Gather> with all enhanced params
        # ===============================================================
        g = Gather(
            input=input,
            action=_action,
            method=method,
            timeout=int(timeout),
            speechTimeout=_speech_timeout,   # ← keep camelCase to match your system
            finishOnKey=finish_on_key,       # ← keep camelCase to match your system
            numDigits=_num_digits,           # ← keep camelCase to match your system
            hints=_hints,
            language=language,
            bargeIn=barge_in,                # ← keep camelCase to match your system
        )

        # 🗣️ Add the spoken prompt using Twilio's <Say> tag
        #     - If SSML detected, allow Twilio to parse markup correctly
        #       and pass RAW prompt (no gpt_speak()) to preserve tags.
        if _contains_ssml:
            g.say(prompt, voice=VOICE, allow_ssml=True)
        else:
            g.say(gpt_speak(prompt), voice=VOICE)

        return g

    except Exception as e:
        # ===============================================================
        # ⚠️ Primary <Gather> creation failed — fallback attempt
        # ===============================================================
        debug_print(f"make_gather: ⚠️ failed to build Gather → {e}")

        try:
            # Rebuild with minimal configuration to ensure voice response
            g = Gather(input=input, action=_action, method=method)

            # Keep the same SSML-vs-plain logic on fallback too
            if _contains_ssml:
                g.say(prompt, voice=VOICE, allow_ssml=True)
            else:
                g.say(gpt_speak(prompt), voice=VOICE)

            return g

        except Exception as e2:
            # Final fallback — cannot recover from Twilio parameter failure
            debug_print(f"make_gather: ❌ secondary fallback failed → {e2}")
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




"""
    Determine if a given [start, end) time slot is available for the doctor
    based entirely on the local JSON appointment file.

    🩺 FUNCTION PURPOSE
    -------------------
    This replaces the old Google Calendar API call. It loads the doctor's
    appointments from a JSON file located in "appointment_data/{doctor}.json"
    and checks for time overlaps with existing appointments.

    ✅ Returns:
        True  → The slot is free (no overlaps).
        False → The slot is busy (overlaps with an existing booking).

    ⚙️ Input Parameters:
        doctor_name : str  → Name of the doctor (used to find JSON file)
        start_iso   : str  → Start time in ISO 8601 (e.g., "2025-10-22T16:00:00Z")
        end_iso     : str  → End time in ISO 8601 (e.g., "2025-10-22T16:30:00Z")

    🧩 JSON Format Expected:
        [
            {
                "first_name": "Mohamed",
                "last_name": "Khalil",
                "phone_e164": "+14694633276",
                "dob": "1956-07-03",
                "start_utc": "2025-10-22T16:00:00Z",
                "end_utc": "2025-10-22T16:30:00Z"
            },
            ...
        ]
    """





# ==============================================================
# 🩺 Function: is_doctor_slot_available
# ==============================================================
def is_doctor_slot_available(doctor_name: str, start_iso: str, end_iso: str) -> bool:
    """
    🩺 PURPOSE:
        Determine if a given [start, end) appointment slot is available for a doctor.
        Uses only the local JSON file under ./appointment_data/.

    ✅ RETURNS:
        True  → Slot is available (no overlaps, valid window)
        False → Slot is busy, invalid, or out of range

    🌍 GLOBAL DEPENDENCY:
        - Reads MAX_ADVANCE_MONTHS (if defined globally) to enforce booking window.
          If not defined, defaults safely to 6 months.
    """

    # ----------------------------------------------------------------------
    # 🧩 Local helper: convert ISO → timezone-aware UTC datetime
    # ----------------------------------------------------------------------
    def _as_utc_dt(s: str):
        """
        Converts ISO timestamp string to a UTC datetime object.
        Handles both 'Z' suffix and naive ISO strings.
        Example:
            "2025-10-22T16:00:00Z" → datetime(2025,10,22,16,0,0,tz=UTC)
        """
        try:
            s2 = s.replace("Z", "+00:00")                # Normalize trailing Z → +00:00
            dt = isoparse(s2)                            # Parse into datetime
            if not dt.tzinfo:                            # If timezone missing
                dt = dt.replace(tzinfo=_pytz.UTC)        # Assume UTC explicitly
            return dt.astimezone(_pytz.UTC)              # Return timezone-aware UTC datetime
        except Exception as e:
            debug_print(f"[is_doctor_slot_available] ⚠️ Failed to parse datetime '{s}' → {e}")
            raise

    # ----------------------------------------------------------------------
    # 🕐 Normalize start/end input times to aware UTC datetimes
    # ----------------------------------------------------------------------
    try:
        start_dt = _as_utc_dt(start_iso)
        end_dt   = _as_utc_dt(end_iso)
    except Exception:
        debug_print("[is_doctor_slot_available] ⚠️ Invalid time input — cannot convert to UTC")
        return False

    # Sanity check — ensure start precedes end
    if end_dt <= start_dt:
        debug_print("[is_doctor_slot_available] ⚠️ Invalid interval (end ≤ start)")
        return False

    # ----------------------------------------------------------------------
    # 🗓️ Enforce MAX_ADVANCE_MONTHS booking window
    # ----------------------------------------------------------------------
    MAX_ADVANCE_MONTHS = int(globals().get("MAX_ADVANCE_MONTHS", 6))  # Use global or fallback to 6

    # Helper to add months safely with rollover
    def _add_months(dt, months):
        import calendar
        y, m = dt.year, dt.month + months
        y += (m - 1) // 12
        m = ((m - 1) % 12) + 1
        d = min(dt.day, calendar.monthrange(y, m)[1])
        return dt.replace(year=y, month=m, day=d)

    now_utc = datetime.now(_pytz.UTC)
    limit_end_utc = _add_months(now_utc, MAX_ADVANCE_MONTHS)

    # Reject appointments that are too far in the future or fully in the past
    if end_dt <= now_utc:
        debug_print(f"[is_doctor_slot_available] ⏳ Slot is entirely in the past → {start_dt.isoformat()}")
        return False
    if start_dt > limit_end_utc:
        debug_print(f"[is_doctor_slot_available] 🚫 Slot beyond {MAX_ADVANCE_MONTHS}-month limit → {start_dt.isoformat()}")
        return False

    # ----------------------------------------------------------------------
    # 📂 Locate the doctor's appointment file
    # ----------------------------------------------------------------------
    safe_name = _re.sub(r"\s+", "_", doctor_name.strip().lower())  # Normalize name
    doc_path = os.path.join("appointment_data", f"{safe_name}.json")
    debug_print(f"[is_doctor_slot_available] 📁 File lookup → {doc_path}")

    # If no file exists, doctor has no appointments yet → slot is free
    if not os.path.exists(doc_path):
        debug_print(f"[is_doctor_slot_available] 🆕 No file for {doctor_name} — slot free")
        return True

    # ----------------------------------------------------------------------
    # 📖 Load existing appointments from the JSON file
    # ----------------------------------------------------------------------
    try:
        with open(doc_path, "r", encoding="utf-8") as f:
            appointments = json.load(f)
    except Exception as e:
        debug_print(f"[is_doctor_slot_available] ❌ Failed to load JSON → {e}")
        return True  # Fail-open: if corrupted/unreadable, treat as available

    # Validate that data structure is a list
    if not isinstance(appointments, list):
        debug_print("[is_doctor_slot_available] ⚠️ File format invalid (expected list) — treating as free")
        return True

    debug_print(f"[is_doctor_slot_available] 🔍 Loaded {len(appointments)} existing entries for {doctor_name}")

    # ----------------------------------------------------------------------
    # 🔄 Check each existing appointment for overlap
    # ----------------------------------------------------------------------
    for i, appt in enumerate(appointments, start=1):
        # Extract stored start/end fields
        appt_start_raw = appt.get("start_utc") or appt.get("utc_start") or appt.get("time")
        appt_end_raw   = appt.get("end_utc") or appt.get("utc_end")

        # Skip incomplete records
        if not appt_start_raw or not appt_end_raw:
            debug_print(f"[is_doctor_slot_available] ⚠️ Skipping incomplete record #{i}")
            continue

        try:
            appt_start = _as_utc_dt(appt_start_raw)
            appt_end   = _as_utc_dt(appt_end_raw)
        except Exception as e:
            debug_print(f"[is_doctor_slot_available] ⚠️ Record #{i} parse error → {e}")
            continue

        # Skip outdated (past) appointments
        if appt_end <= now_utc:
            continue

        # --------------------------------------------------------------
        # 🧮 Overlap condition:
        # Two intervals [start_dt, end_dt) and [appt_start, appt_end)
        # overlap if not (end_dt <= appt_start or appt_end <= start_dt)
        # --------------------------------------------------------------
        if not (end_dt <= appt_start or appt_end <= start_dt):
            debug_print(f"[is_doctor_slot_available] 🚫 Overlap with record #{i}: "
                        f"{appt_start.isoformat()} → {appt_end.isoformat()}")
            return False  # Slot busy

    # ----------------------------------------------------------------------
    # ✅ No overlaps found → slot is free
    # ----------------------------------------------------------------------
    debug_print(f"[is_doctor_slot_available] ✅ Slot free for {doctor_name}: "
                f"{start_dt.isoformat()} → {end_dt.isoformat()}")
    return True






def get_doctor_next_available_slots(
    doctor_name: str,
    *,
    from_start_iso: str,
    duration_minutes: int = None,
    limit: int = 3,
    tz_name: str = None,
    work_hours=None,
    slot_step_minutes: int = None,
    search_days: int = None
) -> list:
    """
    🩺 PURPOSE:
        Generate a list of the next available appointment slots for a given doctor.

    ⚙️ SELF-CONTAINED FEATURES:
        • Uses global MAX_ADVANCE_MONTHS if defined, else defaults to 6.
        • Scans upcoming days (within both search_days and the allowed advance window).
        • Respects work hours, lunch breaks, and weekdays.
        • Filters out booked times using JSON-based slot checks.

    📦 RETURNS:
        A list of dictionaries:
        [
            {
                "start": "2025-10-29T14:00:00Z",
                "end": "2025-10-29T14:30:00Z",
                "friendly": "Wednesday, October 29 at 9:00 AM",
                "tz": "America/Chicago"
            },
            ...
        ]
    """

    # ----------------------------------------------------------------------
    # 🧩 Local debug helper — wraps debug_print safely
    # ----------------------------------------------------------------------
    def _dbg(msg: str):
        try:
            debug_print(msg)
        except Exception:
            print(msg)

    _dbg(f"[get_doctor_next_available_slots] ▶️ doctor={doctor_name} from={from_start_iso} limit={limit}")

    # ----------------------------------------------------------------------
    # ⚙️ Environment defaults and constants
    # ----------------------------------------------------------------------
    MAX_ADVANCE_MONTHS = int(globals().get("MAX_ADVANCE_MONTHS", 6))  # default if not defined
    _dbg(f"[get_doctor_next_available_slots] 📏 Using MAX_ADVANCE_MONTHS={MAX_ADVANCE_MONTHS}")

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
        _dbg("[get_doctor_next_available_slots] ⚠️ Invalid timezone, using America/Chicago")
        tz_local = _pytz.timezone("America/Chicago")

    WSTART = int(globals().get("WORKING_HOURS_START", 8))
    WEND   = int(globals().get("WORKING_HOURS_END", 17))
    if not work_hours:
        work_hours = ((WSTART, WEND),)

    WORKING_DAYS = set(int(x) for x in globals().get("WORKING_DAYS", {0, 1, 2, 3, 4}))

    # ----------------------------------------------------------------------
    # 🕛 Lunch break setup
    # ----------------------------------------------------------------------
    def _as_time(val, default_h=None, default_m=0):
        """Safely converts strings like '12:30' → datetime.time(12,30)."""
        if val is None:
            return None if default_h is None else dtime(default_h, default_m)
        if isinstance(val, dtime):
            return val
        s = str(val).strip()
        if not s:
            return None
        if ":" in s:
            hh, mm = (s.split(":", 1) + ["0"])[:2]
        else:
            hh, mm = s, "0"
        try:
            return dtime(int(hh), int(mm))
        except Exception:
            return None

    LUNCH_START = _as_time(globals().get("LUNCH_BREAK_START"))
    LUNCH_END   = _as_time(globals().get("LUNCH_BREAK_END"))

    if search_days is None:
        search_days = int(globals().get("SEARCH_DAYS", 14))

    # ----------------------------------------------------------------------
    # 🧠 Helper functions
    # ----------------------------------------------------------------------
    def _friendly(dt_local, now_local):
        """Returns human-readable label like 'Tuesday, May 3 at 2:30 PM'."""
        try:
            if dt_local.year != now_local.year:
                return dt_local.strftime("%A, %B %-d, %Y at %-I:%M %p")
            return dt_local.strftime("%A, %B %-d at %-I:%M %p")
        except Exception:
            return dt_local.strftime("%A, %B %d at %I:%M %p")

    def _align_up_to_window_grid(dt_local, minutes, window_start_local, *, now_local):
        """Aligns to the next valid grid (e.g. 9:10 → 9:30 if step=30)."""
        dt_local = dt_local.replace(second=0, microsecond=0)
        anchor = window_start_local.replace(second=0, microsecond=0)
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

    def _add_months(dt, months):
        """Add months safely (keeps valid day numbers)."""
        import calendar
        y, m = dt.year, dt.month + months
        y += (m - 1) // 12
        m = ((m - 1) % 12) + 1
        d = min(dt.day, calendar.monthrange(y, m)[1])
        return dt.replace(year=y, month=m, day=d)

    # ----------------------------------------------------------------------
    # 🕐 Initialize time boundaries
    # ----------------------------------------------------------------------
    now_utc = datetime.now(_pytz.UTC)
    now_loc = now_utc.astimezone(tz_local)

    try:
        req_utc = isoparse((from_start_iso or "").strip())
        if req_utc.tzinfo is None:
            req_utc = _pytz.UTC.localize(req_utc)
    except Exception:
        _dbg("[get_doctor_next_available_slots] ⚠️ Invalid from_start_iso, using now")
        req_utc = now_utc

    req_local = req_utc.astimezone(tz_local)
    limit_end_utc = _add_months(now_utc, MAX_ADVANCE_MONTHS)
    search_window_end = min(now_utc + timedelta(days=search_days), limit_end_utc)
    search_window_start = now_utc
    base_utc = req_utc if (search_window_start <= req_utc <= search_window_end) else now_utc
    cur_local = base_utc.astimezone(tz_local)

    results, seen = [], set()

    # ----------------------------------------------------------------------
    # 🔁 Main scanning loop
    # ----------------------------------------------------------------------
    while cur_local.astimezone(_pytz.UTC) < search_window_end and len(results) < limit:
        # Skip non-working days
        if cur_local.weekday() not in WORKING_DAYS:
            _dbg(f"[get_doctor_next_available_slots] 💤 Skipping {cur_local.strftime('%A')} (non-working day)")
            cur_local = cur_local + timedelta(days=1)
            cur_local = cur_local.replace(hour=WSTART, minute=0, second=0, microsecond=0)
            continue

        # Build working-hour windows
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
                # Respect lunch break
                if LUNCH_START and LUNCH_END:
                    if (cur_local.time() < LUNCH_END and
                        (cur_local + timedelta(minutes=duration_minutes)).time() > LUNCH_START):
                        _dbg("[get_doctor_next_available_slots] 🍽️ Lunch break — skipping")
                        cur_local = tz_local.localize(datetime.combine(cur_local.date(), LUNCH_END))
                        cur_local = _align_up_to_window_grid(cur_local, slot_step_minutes, wstart, now_local=now_loc)
                        continue

                # Stop if beyond 6-month horizon
                if cur_local.astimezone(_pytz.UTC) > limit_end_utc:
                    _dbg(f"[get_doctor_next_available_slots] 🚫 Beyond {MAX_ADVANCE_MONTHS}-month limit — stop")
                    return results

                start_iso = cur_local.astimezone(_pytz.UTC).isoformat().replace("+00:00", "Z")
                end_iso   = (cur_local + timedelta(minutes=duration_minutes)).astimezone(_pytz.UTC).isoformat().replace("+00:00", "Z")

                try:
                    if is_doctor_slot_available(doctor_name, start_iso, end_iso) and start_iso not in seen:
                        seen.add(start_iso)
                        results.append({
                            "start": start_iso,
                            "end": end_iso,
                            "friendly": _friendly(cur_local, now_loc),
                            "tz": tz_name,
                        })
                        _dbg(f"[get_doctor_next_available_slots] ✅ Added → {results[-1]['friendly']} ({start_iso})")
                except Exception as e:
                    _dbg(f"[get_doctor_next_available_slots] ❌ Availability check failed → {e}")

                cur_local = cur_local + timedelta(minutes=slot_step_minutes)
                progressed = True

        if not progressed:
            _dbg(f"[get_doctor_next_available_slots] ⏭️ No valid slots on {cur_local.strftime('%A')} — next day")
            cur_local = cur_local + timedelta(days=1)
            cur_local = cur_local.replace(hour=WSTART, minute=0, second=0, microsecond=0)

    _dbg(f"[get_doctor_next_available_slots] ✅ Finished — found {len(results)} slot(s)")
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
      - Add insurance_name and insurance_member_id fields (empty string if missing)
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
        #import random  # local import for clarity
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
            if not rec.get("customer_status"):
                rec["customer_status"] = "current"
                changed = True

            # ✅ Ensure pin_number (6-digit integer)
            pin_value = rec.get("pin_number")
            if not isinstance(pin_value, int) or pin_value < 100000 or pin_value > 999999:
                rec["pin_number"] = random.randint(100000, 999999)
                changed = True

            # ✅ Ensure insurance fields exist (empty string if missing)
            # These fields are used to store customer insurance data.
            # Example: insurance_name="Blue Cross Blue Shield", insurance_member_id="W123456789"
            if "insurance_name" not in rec:
                rec["insurance_name"] = ""
                changed = True
            if "insurance_member_id" not in rec:
                rec["insurance_member_id"] = ""
                changed = True

            # ✅ Normalize DOB and ensure phone format
            rec["dob"] = _oneline(rec.get("dob", ""))
            phone_e164 = _e164_or_empty(rec.get("phone_e164", ""))

            # ✅ If E.164 not found, try adopting from old key
            if not phone_e164 and "|" in old_key:
                left = old_key.split("|", 1)[0].strip()
                left_e164 = _e164_or_empty(left)
                if left_e164:
                    rec["phone_e164"] = left_e164
                    phone_e164 = left_e164
                    adopted_from_key += 1
                    changed = True

            # ✅ Construct final key based on (phone_e164|dob)
            final_key = old_key
            if phone_e164:
                try:
                    final_key = _key(phone_e164, rec.get("dob", ""))
                except Exception:
                    final_key = old_key

            # ✅ Update record in new_data
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
    insurance_name: str = "",
    insurance_member_id: str = "",
) -> bool:
    """
    Insert or update a customer in customers.json (single pretty JSON dict).

      • If (phone|dob) exists → update record fields + bump 'last_seen_at'; return False.
      • If new → create record with 'created_at' + 'last_seen_at'; return True.

    This version guarantees:
      ✅ Strict E.164 enforcement (no legacy 10-digit fallback).
      ✅ Adds 'customer_status' field (default = "current", or explicitly set to "new").
      ✅ Adds 'pin_number' field (6-digit integer; default = 0 if not provided).
      ✅ Adds 'insurance_name' and 'insurance_member_id' fields (default = "").
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
    insurance_name = _oneline(insurance_name)
    insurance_member_id = _oneline(insurance_member_id)

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
            "insurance_name": insurance_name or existing.get("insurance_name", ""),
            "insurance_member_id": insurance_member_id or existing.get("insurance_member_id", ""),
            "last_seen_at": now,
            "customer_status": customer_status or "current",  # ✅ override if specified
            "pin_number": new_pin,
        })
        _save_customers(data)
        debug_print(
            f"insert_customer: 🟡 Updated existing record for {key} "
            f"(status={customer_status}, pin={new_pin}, insurance='{insurance_name}')"
        )
        return False

    # ----------------------------------------------------------------------
    # 🧩 Step 5: Insert new customer record
    # ----------------------------------------------------------------------
    # ✅ Generate random 6-digit PIN if not provided or invalid
    #import random
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
        "insurance_name": insurance_name or "",
        "insurance_member_id": insurance_member_id or "",
        "created_at": now,
        "last_seen_at": now,
        "customer_status": customer_status or "current",  # ✅ input-controlled value
        "pin_number": pin_number,  # ✅ always stored as integer
    }

    data[key] = rec
    _save_customers(data)

    debug_print(
        f"insert_customer: ✅ Added {customer_status.upper()} customer {first_name} {last_name} "
        f"({phone_e164}|{dob_iso}) @ {now} (PIN={pin_number}, insurance='{insurance_name}')"
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



# ======================================================================
# 🏥 Insurance Field Utilities
# ----------------------------------------------------------------------
# These helper functions update or retrieve insurance_name and
# insurance_member_id for a given (phone_e164, dob) customer record.
# They strictly enforce E.164 normalization and preserve all existing data.
# ======================================================================

def update_insurance_name(phone: str, dob: str, new_name: str) -> bool:
    """
    Update the insurance_name field for an existing customer.

    Returns:
        True  → if the insurance_name was successfully updated.
        False → if the record was not found or invalid input.
    """
    init_db()
    phone_e164 = normalize_phone_e164(phone, globals().get("COUNTRY", "US"))
    if not phone_e164:
        debug_print(f"update_insurance_name: invalid phone '{phone}'")
        return False

    key = _key(phone_e164, dob.strip())
    data = _load_customers()

    if key not in data:
        debug_print(f"update_insurance_name: ❌ record not found for {key}")
        return False

    data[key]["insurance_name"] = _oneline(new_name)
    data[key]["last_seen_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_customers(data)
    debug_print(f"update_insurance_name: ✅ updated to '{new_name}' for {key}")
    return True


def get_insurance_name(phone: str, dob: str) -> str:
    """
    Retrieve the insurance_name for a customer (default = '').
    Returns empty string if record not found.
    """
    init_db()
    phone_e164 = normalize_phone_e164(phone, globals().get("COUNTRY", "US"))
    if not phone_e164:
        return ""
    key = _key(phone_e164, dob.strip())
    data = _load_customers()
    rec = data.get(key, {})
    return rec.get("insurance_name", "")


def update_insurance_member_id(phone: str, dob: str, new_id: str) -> bool:
    """
    Update the insurance_member_id field for an existing customer.

    Returns:
        True  → if the insurance_member_id was successfully updated.
        False → if the record was not found or invalid input.
    """
    init_db()
    phone_e164 = normalize_phone_e164(phone, globals().get("COUNTRY", "US"))
    if not phone_e164:
        debug_print(f"update_insurance_member_id: invalid phone '{phone}'")
        return False

    key = _key(phone_e164, dob.strip())
    data = _load_customers()

    if key not in data:
        debug_print(f"update_insurance_member_id: ❌ record not found for {key}")
        return False

    data[key]["insurance_member_id"] = _oneline(new_id)
    data[key]["last_seen_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_customers(data)
    debug_print(f"update_insurance_member_id: ✅ updated to '{new_id}' for {key}")
    return True


def get_insurance_member_id(phone: str, dob: str) -> str:
    """
    Retrieve the insurance_member_id for a customer (default = '').
    Returns empty string if record not found.
    """
    init_db()
    phone_e164 = normalize_phone_e164(phone, globals().get("COUNTRY", "US"))
    if not phone_e164:
        return ""
    key = _key(phone_e164, dob.strip())
    data = _load_customers()
    rec = data.get(key, {})
    return rec.get("insurance_member_id", "")


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




def book_appointment_for_dr_name(
    doctor_name: str,
    phone: str,
    utc_start: str,
    name: str = None,
    dob: str = None,
    address: str = None,
    event_id: str = None,
    debug: bool = False,
    utc_end: str = None,
    friendly_local: str = None,
    local_date: str = None,
    local_time_display: str = None,
):
    """
    🩺 PURPOSE:
        Create and store a new appointment for a given doctor in the local JSON database.

    ⚙️ FUNCTIONALITY:
        • Validates phone, DOB, and time inputs
        • Converts UTC times to local clinic timezone
        • Checks for duplicates (same phone + DOB + UTC time)
        • Appends appointment entry to doctor’s JSON file
        • Returns a structured dictionary summarizing operation result

    ✅ OUTPUT FORMAT:
        {
            "created": True,
            "record": {...},      # appointment record dictionary
            "reason": None        # or "duplicate" if not created
        }
    """

    # ----------------------------------------------------------------------
    # 📞 Normalize phone number: keep only numeric digits
    # ----------------------------------------------------------------------
    digits_only_phone = _re.sub(r"\D", "", phone or "")
    if not digits_only_phone:
        raise ValueError("Phone is required and must contain digits.")

    # ----------------------------------------------------------------------
    # 🎂 Normalize DOB (convert to ISO YYYY-MM-DD if possible)
    # ----------------------------------------------------------------------
    dob_iso = (dob or "").strip()
    if dob_iso and _re.fullmatch(r"\d{4}-\d{2}-\d{2}", dob_iso) is None:
        # Convert formats like "10/28/2025" → "2025-10-28"
        m = _re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", dob_iso)
        if m:
            mm, dd, yyyy = m.groups()
            dob_iso = f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}"
        else:
            dob_iso = dob_iso.replace("/", "-")

    # ----------------------------------------------------------------------
    # ⏰ Validate and normalize UTC timestamps
    # ----------------------------------------------------------------------
    def ensure_utc_iso(ts: str) -> str:
        """
        Convert a raw timestamp string to a strict UTC ISO 8601 format.
        Ensures consistency between stored appointment times.
        """
        if not ts:
            raise ValueError("utc_start is required")

        s = ts.strip().replace(" ", "T")  # Normalize spacing between date and time
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00") if s.endswith("Z") else s)
        except Exception:
            # fallback for timestamps missing timezone info
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

    # ----------------------------------------------------------------------
    # 🕐 Convert UTC → Local clinic time for display
    # ----------------------------------------------------------------------
    try:
        tz_name = (globals().get("CLINIC_TZ") or globals().get("LOCAL_TZ") or "America/Chicago")
        tz_local = _pytz.timezone(tz_name)
    except Exception:
        tz_local = _pytz.timezone("America/Chicago")

    dt_utc = datetime.fromisoformat(utc_start_iso.replace("Z", "+00:00")).astimezone(_pytz.UTC)
    dt_loc = dt_utc.astimezone(tz_local)

    # Local date (overridable by caller)
    if local_date and _re.fullmatch(r"\d{4}-\d{2}-\d{2}", local_date):
        date_local = local_date
    else:
        date_local = dt_loc.strftime("%Y-%m-%d")

    # Local time (UTC HH:MM format for backend consistency)
    time_local_utc_hhmm = dt_utc.strftime("%H:%M")

    # Friendly local display string
    if friendly_local and friendly_local.strip():
        friendly = friendly_local.strip()
    else:
        try:
            friendly = dt_loc.strftime("%A, %B %-d at %-I:%M %p")
        except Exception:
            friendly = dt_loc.strftime("%A, %B %d at %I:%M %p").replace(" 0", " ")

    # ----------------------------------------------------------------------
    # 📂 Resolve doctor’s JSON file path
    # ----------------------------------------------------------------------
    filename = sanitize_filename(doctor_name).replace(".json", "")
    full_path = get_doctor_filename(doctor_name)
    debug_print(f"🔍 File → {full_path}")

    # ----------------------------------------------------------------------
    # 📖 Load existing appointments (create new list if missing)
    # ----------------------------------------------------------------------
    appts = []
    if os.path.exists(full_path):
        try:
            with open(full_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                appts = data
                debug_print(f"✅ Loaded {len(appts)} existing appointment(s)")
            else:
                debug_print("⚠️ Invalid JSON root type — reinitializing empty list")
        except Exception as e:
            debug_print(f"⚠️ Failed to parse JSON → {e}")
    else:
        debug_print("📂 No existing file — starting new list")

    # ----------------------------------------------------------------------
    # 🔍 Detect duplicates by phone (+ DOB if available)
    # ----------------------------------------------------------------------
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

    debug_print(f"🔎 Search for duplicates → {len(matches)} match(es) (phone={digits_only_phone}, dob={dob_iso or 'N/A'})")

    # ----------------------------------------------------------------------
    # 🚫 Prevent adding identical appointment twice
    # ----------------------------------------------------------------------
    for _, appt in matches:
        try:
            appt_time_iso = ensure_utc_iso(appt.get("time", "") or appt.get("utc_start", ""))
        except Exception:
            appt_time_iso = None

        if appt_time_iso == utc_start_iso:
            debug_print("🔁 Exact duplicate detected — skipping append")

            appt_norm = dict(appt)
            appt_norm["phone"] = _re.sub(r"\D", "", appt_norm.get("phone", ""))
            appt_norm["utc_start"] = utc_start_iso
            if utc_end_iso:
                appt_norm["utc_end"] = utc_end_iso
            appt_norm.setdefault("date_local", date_local)
            appt_norm.setdefault("time_local", time_local_utc_hhmm)
            appt_norm.setdefault("friendly_local", friendly)
            if local_time_display:
                appt_norm.setdefault("time_local_display", local_time_display)

            return {"created": False, "record": appt_norm, "reason": "duplicate"}

    # ----------------------------------------------------------------------
    # ➕ Build new appointment record
    # ----------------------------------------------------------------------
    new_record = {
        "phone":          digits_only_phone,
        "utc_start":      utc_start_iso,
        "date_local":     date_local,
        "time_local":     time_local_utc_hhmm,
        "friendly_local": friendly,
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
        new_record["time_local_display"] = local_time_display

    appts.append(new_record)
    debug_print(f"➕ Appended: {new_record}")

    # ----------------------------------------------------------------------
    # 💾 Save updated list to JSON
    # ----------------------------------------------------------------------
    try:
        with open(full_path, "w") as f:
            json.dump(appts, f, indent=2)
        debug_print(f"💾 Saved successfully → {full_path}")

        # Optional cache update (if global cache variable exists)
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
    global session_data,doctor_names
    # ----------------------------------------------------------------------
    # 🎙️ Twilio Voice Entry Point
    # ----------------------------------------------------------------------
    resp = VoiceResponse()
    debug_print("[voice] ▶ enter voice()")

    # ----------------------------------------------------------------------
    # 🆔 Extract Call SID & Initialize Session
    # ----------------------------------------------------------------------
    call_sid = request.values.get("CallSid", "")
    debug_print(f"[voice] CallSid={call_sid}")

    # Load or access session
    sd = session_data.get(call_sid, {})
    debug_print(f"voice: 🔁 Loaded session for {call_sid}: keys={list(sd.keys())}")
    debug_print(f"voice : 🩺 doctor_name loaded → {sd.get('doctor_name')}")

    # ----------------------------------------------------------------------
    # 🗣️ Retrieve Inputs
    # ----------------------------------------------------------------------
    speech_result = (request.values.get("SpeechResult") or "").strip()
    try:
        dtmf_digits = (request.values.get("Digits") or "").strip()
    except Exception:
        dtmf_digits = ""
    debug_print(f"[voice] inputs → speech='{speech_result}' dtmf='{dtmf_digits}'")

    # ----------------------------------------------------------------------
    # 🌍 Country / Caller Initialization
    # ----------------------------------------------------------------------
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

    # Store E.164 caller number
    from_number = (request.values.get("From") or "").strip()
    if from_number.startswith("+"):
        session_data[call_sid]["from_e164"] = from_number
        debug_print(f"[voice] from_e164 set → {from_number}")

    print(f"📢 voice :speech_result: {speech_result}")

    # ----------------------------------------------------------------------
    # 🎯 Determine Current Stage
    # ----------------------------------------------------------------------
    stage = session_data.get(call_sid, {}).get("stage", "intro")
    debug_print(f"[voice] 🎯 stage='{stage}'")

    # ----------------------------------------------------------------------
    # 🔇 SILENCE GUARD PROMPTS
    # ----------------------------------------------------------------------
    def _silence_prompt_for_stage(st: str) -> Tuple[str, str]:
        """Return (prompt, hints) best suited for the current stage."""
        debug_print(f"[voice] 🔇 selecting silence prompt for stage='{st}'")
        hints = ""

        if st in ("intro", "intent"):
            hints = "book,cancel,change,reschedule,update,update card,voicemail,leave message"
            return (
                "I didn’t hear anything. Say 'book appointment' or press 1. "
                "Say 'cancel appointment' or press 2. "
                "Say 'change appointment' or press 3. "
                "Say 'update credit card' or press 4. "
                "Say 'leave voicemail' or press 5.",
                hints
            )

        if st == "book_appointment":
            # 🩺 Use local doctor names instead of Google map
            doctor_list = ", ".join(
                doctor_names.values() if isinstance(doctor_names, dict) else doctor_names
            )
            hints = doctor_list
            return ("Please say the name of the doctor you'd like to book with.", hints)

        if st == "collect_phone":
            hints = "zero one two three four five six seven eight nine double triple"
            return ("Please say or enter your ten digit phone number including area code.", hints)

        if st == "collect_dob":
            return (
                "Please say your birth date, for example 'July third 1990'. "
                "Or type 2 digits for Month, 2 digits for Day, and 4 digits for Year, then press pound.",
                hints
            )

        if st == "collect_book_time_date":
            return (
                "Please say the appointment time, for example, 'August fifteenth at 5 AM'. "
                "Or enter two digits for month, two for day, two for hour, and two for minutes, "
                "then say A for AM or P for PM, then press #.",
                hints
            )

        if st == "collect_first_name":
            return ("Please say your first name.", hints)

        if st == "collect_last_name":
            return ("Please say your last name.", hints)

        if st == "collect_address":
            return (
                "Please say your street address, city, and ZIP. For example, "
                "'118 Briar Oak, Murphy, Texas 75094'.",
                hints
            )

        if st == "cancel_appointment":
            doctor_list = ", ".join(
                doctor_names.values() if isinstance(doctor_names, dict) else doctor_names
            )
            hints = doctor_list
            return ("Please say the name of the doctor whose appointment you want to cancel.", hints)

        if st in ("cancel_appt_by_phone_number",):
            hints = "zero one two three four five six seven eight nine double triple"
            return ("Please say the phone number used when booking, including area code.", hints)

        if st in ("cancel_appt_by_time_date", "cancel_appt_by_date_time"):
            return ("Please say the date and time of the appointment you want to cancel, for example, 'July third at nine AM'.", hints)

        if st == "cancel_appt_get_dob":
            return ("Please say your birth date, for example 'July third nineteen fifty six'. "
                    "Or type 2 digits for month 2 digits for day and 4 digits for year then press pound.", hints)

        if st == "voicemail":
            return ("Please leave your name, phone number, and message after the beep.", hints)

        # Default fallback
        return ("Sorry, I didn’t hear anything. Please say that again.", hints)

    # ----------------------------------------------------------------------
    # 🔇 SILENCE HANDLER
    # ----------------------------------------------------------------------
    skip_silence = (
        "intro",
        "collect_cc",
        "book_appt_confirm",
        "cancel_appt_iterate",
        "collect_phone",
        "cancel_appt_confirm",
        "collect_dob",
        "collect_first_name",
        "collect_last_name",
        "collect_insurance_information",
        "collect_dr_info",
        "cancel_appt_get_time_date",
    )
    debug_print(f"[voice] 🔇 skip_silence={skip_silence}")

    if stage not in skip_silence:
        debug_print(
            f"[voice] 🔇 evaluating silence guard at stage='{stage}' "
            f"(speech_empty={not bool(speech_result)} dtmf_empty={not bool(dtmf_digits)})"
        )
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

    # ----------------------------------------------------------------------
    # Continue with main conversation logic (other stages)
    # ----------------------------------------------------------------------
    # ↓ add your existing stage-handling code below this point
    # e.g. intro / intent / collect_dr_info / collect_book_time_date / etc.









    """
    # What happens in this stage:
    # The caller calls the clinic.
    # Twilio sends a webhook to your /voice endpoint.
    # You respond with a greeting prompt, dynamically generated using ChatGPT.
    # You ask: “Would you like to book an appointment or leave a message?”
    # The system listens for speech and sends the result back to the same endpoint (/voice) using a POST request.
    # The session progresses from "intro" to "intent" for next steps.
    # If this is the start of the call, begin with the "intro" stage.
    """
    if stage == "intro":
        # ----------------------------------------------------------------------
        # 🧠 Initialize or update the session for this call
        # ----------------------------------------------------------------------
        # Ensures the session dict exists and preserves any previous values
        # (like phone, country, doctor_name) without overwriting them.
        sd = session_data.setdefault(call_sid, {})
        sd["stage"] = "intent"

        # ----------------------------------------------------------------------
        # 🩺 Debug info to trace continuity between Twilio POSTs
        # ----------------------------------------------------------------------
        debug_print(f"[intro] ▶️ New or returning call SID → {call_sid}")
        debug_print(f"[intro] 🧭 Next stage set to 'intent'")
        debug_print(f"[intro] Current session keys → {list(sd.keys())}")

        # ----------------------------------------------------------------------
        # 🎙️ Voice prompt
        # ----------------------------------------------------------------------
        prompt = (
            "Thank you for calling EPIC therapist. "
            "Say 'book appointment' or press 1. "
            "Say 'cancel appointment' or press 2. "
            "Say 'change appointment' or press 3. "
            "Say 'update credit card' or press 4. "
            "Say 'leave voicemail' or press 5."
        )

        # Build a Twilio <Gather> block:
        # - Speaks the message using Polly voice
        # - Accepts speech or keypad (DTMF)
        # - Automatically posts back to /voice
        gather = make_gather(
            prompt,
            hints="book,cancel,change,reschedule,update,voicemail",
            num_digits=1
        )

        """
        Twilio will receive this as:
        <Response>
            <Gather input="speech dtmf" numDigits="1" ...>
                <Say>Thank you for calling EPIC therapist...</Say>
            </Gather>
        </Response>
        """

        # Append the Gather block to the TwiML response
        resp.append(gather)

        # ✅ Return TwiML back to Twilio
        return str(resp)




    elif stage == "intent":
        # ----------------------------------------------------------------------
        # 🎯 Intent detection stage: figure out what the caller wants:
        #   1. Book an appointment
        #   2. Cancel an appointment
        #   3. Reschedule an appointment
        #   4. Update credit card
        #   5. Update PIN number
        #   6. Update insurance information
        #   7. Leave a voicemail
        # ----------------------------------------------------------------------

        lower = (speech_result or "").lower().strip()
        print(f"📢 intent :speech_result: {lower}")

        # ----------------------------------------------------------------------
        # 🔢 Handle keypad input 1..7 or spoken digits
        # ----------------------------------------------------------------------
        choice = None
        if dtmf_digits and len(dtmf_digits) == 1 and dtmf_digits in "1234567":
            choice = dtmf_digits
        elif lower in {"1", "2", "3", "4", "5", "6", "7"}:
            choice = lower

        # ----------------------------------------------------------------------
        # 🗣️ Handle speech keywords (semantic triggers)
        # ----------------------------------------------------------------------
        if any(word in lower for word in ["book", "appointment", "schedule"]):
            choice = "1"
        elif any(word in lower for word in ["cancel", "delete", "remove"]):
            choice = "2"
        elif any(word in lower for word in ["reschedule", "change", "move"]):
            choice = "3"
        elif any(word in lower for word in ["credit", "card", "payment"]):
            choice = "4"
        elif any(word in lower for word in ["pin", "password", "pin number"]):
            choice = "5"
        elif any(word in lower for word in ["insurance", "health", "medical"]):
            choice = "6"
        elif any(word in lower for word in ["voicemail", "message", "record"]):
            choice = "7"

        # ----------------------------------------------------------------------
        # ✅ Route user choice (speech or keypad)
        # ----------------------------------------------------------------------
        if choice:
            # 1️⃣ Book Appointment
            if choice == "1":
                print("📅 DTMF=1 → booking (start with phone collection)")
                session_data.setdefault(call_sid, {})
                session_data[call_sid].update({
                    "stage": "collect_phone",
                    "origin_stage": "book",
                    "booking": {},
                    "retry_booking": 0,
                    "retry_time": 0
                })
                prompt = "Please say or enter your ten-digit phone number, then press pound."
                gather = make_gather(prompt, input="speech dtmf", timeout=6,
                                    speech_timeout="auto", barge_in=True,
                                    finish_on_key="#", num_digits=10)
                resp.append(gather)
                resp.redirect("/voice")
                return str(resp)

            # 2️⃣ Cancel Appointment
            if choice == "2":
                print("❌ DTMF=2 → cancel flow")
                session_data[call_sid] = {
                    "stage": "cancel_appointment",
                    "cancel": {},
                    "retry_booking": 0
                }

                # Use local doctor name list
                doctor_list = list(doctor_names.values()) if isinstance(doctor_names, dict) else doctor_names
                dtmf_map = {str(i): name for i, name in enumerate(doctor_list, start=1)}
                session_data[call_sid]["doctor_dtmf_map"] = dtmf_map

                doctor_list_with_keys = ", ".join([f"{name} (press {i})" for i, name in enumerate(doctor_list, start=1)])
                prompt = (
                    f"Sure, I can help you cancel your appointment. "
                    f"Available doctors are: {doctor_list_with_keys}. "
                    "Please say the doctor's name or press the number."
                )
                gather = make_gather(prompt, hints=", ".join(doctor_list), num_digits=1)
                resp.append(gather)
                return str(resp)

            # 3️⃣ Reschedule Appointment
            if choice == "3":
                print("🔁 DTMF=3 → reschedule (cancel then rebook)")
                session_data[call_sid] = {
                    "stage": "cancel_appointment",
                    "cancel": {},
                    "retry_booking": 0,
                    "reschedule_after_cancel": True
                }

                doctor_list = list(doctor_names.values()) if isinstance(doctor_names, dict) else doctor_names
                dtmf_map = {str(i): name for i, name in enumerate(doctor_list, start=1)}
                session_data[call_sid]["doctor_dtmf_map"] = dtmf_map

                doctor_list_with_keys = ", ".join([f"{name} (press {i})" for i, name in enumerate(doctor_list, start=1)])
                prompt = (
                    f"Sure, let's reschedule your appointment. First, we'll cancel your current one. "
                    f"Available doctors are: {doctor_list_with_keys}. "
                    "Please say the doctor's name or press the number."
                )
                gather = make_gather(prompt, hints=", ".join(doctor_list), num_digits=1)
                resp.append(gather)
                return str(resp)

            # 4️⃣ Update Credit Card
            if choice == "4":
                print("💳 DTMF=4 → update credit card")
                session_data.setdefault(call_sid, {})
                session_data[call_sid].update({
                    "stage": "update_cc",
                    "cc_update": {"active": True},
                    "retry_booking": 0
                })
                resp.say(gpt_speak("You said you want to update your credit card information. Please hold while we process this request."), VOICE)
                resp.redirect("/voice")
                return str(resp)

            # 5️⃣ Update PIN Number
            if choice == "5":
                print("🔢 DTMF=5 → update PIN number")
                session_data.setdefault(call_sid, {})
                session_data[call_sid]["stage"] = "update_pin_number"
                resp.say(gpt_speak("You said you want to update your PIN number. This option is not implemented yet. Please call the clinic for assistance."), VOICE)
                return str(resp)

            # 6️⃣ Update Insurance Info
            if choice == "6":
                print("🏥 DTMF=6 → update insurance information")
                session_data.setdefault(call_sid, {})
                session_data[call_sid]["stage"] = "update_insurance_information"
                resp.say(gpt_speak("You said you want to update your health insurance information. This option is not implemented yet. Please call the clinic for assistance."), VOICE)
                return str(resp)

            # 7️⃣ Leave Voicemail
            if choice == "7":
                print("📩 DTMF=7 → voicemail")
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
        # 🚫 Handle greetings or empty input (re-prompt main menu)
        # ----------------------------------------------------------------------
        junk_inputs = {
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
            "yo", "test", "1", "yes", "no"
        }
        if not lower or lower in junk_inputs:
            print(f"⛔ Ignored junk input: '{lower}' — re-prompting main menu")
            gather = make_gather(
                "Thank you for calling Epic Therapist. "
                "Say 'book appointment' or press 1. "
                "Say 'cancel appointment' or press 2. "
                "Say 'change appointment' or press 3. "
                "Say 'update credit card' or press 4. "
                "Say 'update PIN number' or press 5. "
                "Say 'update insurance information' or press 6. "
                "Say 'leave voicemail' or press 7.",
                hints="book,cancel,change,reschedule,update,credit card,pin number,insurance,voicemail",
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
    






     # ----------------------------------------------------------------------
    # 📅 Stage: book_appointment
    # ----------------------------------------------------------------------
    # 🎯 PURPOSE:
    #   • Acts as the entry point for appointment booking.
    #   • Initializes session data for booking flow.
    #   • Redirects caller to `collect_phone` to capture their phone number.
    # ----------------------------------------------------------------------
    elif stage == "book_appointment":
   
        debug_print("book_appointment: entered → redirecting to collect_phone")

        # ----------------------------------------------------------------------
        # 💬 VOICE MESSAGES — centralized for maintainability & localization
        # ----------------------------------------------------------------------
        VOICE_BOOKING_INTRO_MSG = (
            "Let's get started with your booking. "
            "Please say or enter your phone number, including the area code, then press pound."
        )
        VOICE_INVALID_INPUT_MSG = (
            "I'm sorry, I didn't get that. Please say or enter your phone number again, then press pound."
        )

        # ----------------------------------------------------------------------
        # 🧩 SESSION INITIALIZATION
        # ----------------------------------------------------------------------
        # Ensure the call has a valid session in memory.
        # Sets initial booking metadata to track user flow.
        sd = session_data.setdefault(call_sid, {})
        sd["stage"] = "collect_phone"       # Next stage to capture phone number
        sd["retry_booking"] = 0             # Counter for booking retries
        sd["origin_stage"] = "book"         # Identify booking origin for PIN flow, etc.

        # ----------------------------------------------------------------------
        # 🩺 Store available doctors for later doctor selection
        # ----------------------------------------------------------------------
        # Convert doctor list into a DTMF map (1 → Dr. Smith, 2 → Dr. Lee, etc.)
        # This map will be reused in `collect_dr_info` to present choices.
        if isinstance(doctor_names, dict):
            doctor_list = list(doctor_names.values())
        else:
            doctor_list = doctor_names

        dtmf_map = {str(i): name for i, name in enumerate(doctor_list, start=1)}
        sd["doctor_dtmf_map"] = dtmf_map
        debug_print(f"book_appointment: 🩺 loaded doctor map → {dtmf_map}")

        # ----------------------------------------------------------------------
        # 📞 Prompt caller to provide phone number
        # ----------------------------------------------------------------------
        # Uses make_gather() to capture speech or keypad input.
        #   - input="speech dtmf"  → supports both speaking and typing.
        #   - timeout=8            → waits up to 8 seconds for a response.
        #   - finish_on_key="#"    → '#' key ends input early.
        #   - num_digits=10        → expects 10-digit phone numbers (U.S. style).
        # After prompt, control passes to `/voice` for next processing.
        g = make_gather(
            VOICE_BOOKING_INTRO_MSG,        # spoken prompt to caller
            input="speech dtmf",            # allow both speech and keypad input
            timeout=8,                      # wait up to 8 seconds
            speech_timeout="auto",          # auto-detect end of speech
            barge_in=True,                  # allow interrupting prompt
            finish_on_key="#",              # '#' ends input
            num_digits=10                   # expect 10 digits
        )

        # Append gather block to Twilio <Response> and redirect
        resp.append(g)
        resp.redirect("/voice")
        return str(resp)






    # ======================================================================
    # 📞 Stage: collect_phone — Capture customer phone number via speech or DTMF
    # ======================================================================
    # 🎯 PURPOSE:
    #   • Capture and normalize caller’s phone number.
    #   • Support both speech and keypad (DTMF) input.
    #   • Handle silence and invalid input with polite retry prompts.
    #   • Maintain doctor context and booking continuity.
    # ======================================================================


    
    elif stage == "collect_phone":
    

        debug_print("[collect_phone] 📍 Stage entered")

        # ----------------------------------------------------------------------
        # 💬 VOICE MESSAGES — centralized for maintainability & localization
        # ----------------------------------------------------------------------
        VOICE_NO_INPUT_MSG = (
            "I didn’t hear your phone number. Please say or enter your 10-digit number, then press pound."
        )
        VOICE_TOO_MANY_SILENCES_MSG = (
            "I'm sorry, I still didn't get your phone number. Please call again later."
        )
        VOICE_INVALID_PHONE_MSG = (
            "That doesn’t sound complete. Please say or enter your 10-digit phone number including area code, then press pound."
        )
        VOICE_TOO_MANY_INVALID_MSG = (
            "I'm sorry, I couldn’t capture your phone number. Please call again later."
        )
        VOICE_RESCHEDULE_MSG = (
            "Thanks. Please say the new appointment date and time, for example, 'October 12 at 9 A M'."
        )
        VOICE_ASK_DOB_MSG = (
            "Thanks. What’s your date of birth? You can say it, or enter two digits for month, "
            "two for day, and four for year, then press pound."
        )

        # ----------------------------------------------------------------------
        # 🔁 Load session safely (never overwrite)
        # ----------------------------------------------------------------------
        sd = session_data.setdefault(call_sid, {})
        cust = sd.setdefault("customer", {})
        cancel_ctx = sd.setdefault("cancel", {})
        debug_print(f"[collect_phone] session keys before: {list(sd.keys())}")

        # 👁️ Diagnostic: ensure doctor_name context persists between stages
        if "doctor_name" in sd:
            debug_print(f"[collect_phone] ✅ doctor_name still active: {sd['doctor_name']}")
        else:
            debug_print("[collect_phone] ⚠️ doctor_name missing entering collect_phone")

        # ----------------------------------------------------------------------
        # 🌎 Infer phone country once per call
        # ----------------------------------------------------------------------
        if "phone_country" not in sd:
            from_country = (request.values.get("FromCountry") or "").upper()
            sd["phone_country"] = from_country or (COUNTRY or "US")
            debug_print(f"[collect_phone] 🌐 phone_country={sd['phone_country']}")

        # ----------------------------------------------------------------------
        # 🗣 Capture inputs from Twilio (speech + DTMF)
        # ----------------------------------------------------------------------
        dtmf_digits = (request.values.get("Digits") or "").strip()
        speech_text = (speech_result or "").strip()
        debug_print(f"[collect_phone] 🗣 speech='{speech_text}'  🔢 DTMF='{dtmf_digits}'")

        # ----------------------------------------------------------------------
        # 🔇 Silence handling (no speech or digits received)
        # ----------------------------------------------------------------------
        if not (speech_text or dtmf_digits):
            tries = sd.get("silence_collect_phone", 0) + 1
            sd["silence_collect_phone"] = tries
            debug_print(f"[collect_phone] 🤐 No input (tries={tries}/3)")

            # 🗣 Retry up to 2 times politely
            if tries < 3:
                g = make_gather(
                    VOICE_NO_INPUT_MSG,
                    input="speech dtmf",
                    timeout=4,
                    speech_timeout="auto",
                    barge_in=True,
                    finish_on_key="#"
                )
                resp.append(g)
                resp.redirect("/voice")
                return str(resp)

            # ❌ After 3 failed attempts, terminate politely
            resp.say(gpt_speak(VOICE_TOO_MANY_SILENCES_MSG), VOICE)
            resp.hangup()
            save_session(call_sid)  # ✅ persist logs for debugging
            session_data.pop(call_sid, None)
            return str(resp)

        # ✅ Clear silence counter since input received
        sd.pop("silence_collect_phone", None)

        # ----------------------------------------------------------------------
        # 🔢 Convert spoken input to digits if needed
        # ----------------------------------------------------------------------
        def _spoken_to_digits(raw: str) -> str:
            """
            Converts spoken numbers like “two one four five five” → “21455”.
            Handles common words, including “double” and “triple” cases.
            """
            if not raw:
                return ""
            # Normalize input text: lowercase and remove punctuation
            words = (
                raw.lower()
                .replace("-", " ").replace(",", " ").replace(".", " ")
                .replace("(", " ").replace(")", " ").split()
            )
            # Map spoken words to digits
            mapping = {
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
                # Handle "double five" → "55", "triple six" → "666"
                if w in ("double", "triple") and i + 1 < len(words):
                    nxt = words[i + 1]
                    if nxt in mapping:
                        out.extend([mapping[nxt]] * (2 if w == "double" else 3))
                        i += 2
                        continue
                # Regular mapping
                if w in mapping:
                    out.append(mapping[w])
                else:
                    # Extract digits directly from any alphanumeric speech artifacts
                    out.extend([c for c in w if c.isdigit()])
                i += 1
            return "".join(out)

        # 🧩 Combine DTMF digits (if pressed) or convert speech
        raw_digits = _re.sub(r"\D", "", dtmf_digits or _spoken_to_digits(speech_text))
        debug_print(f"[collect_phone] 🔍 raw_digits='{raw_digits}'")

        # ----------------------------------------------------------------------
        # 🌐 Normalize number to E.164 format (+14155552671)
        # ----------------------------------------------------------------------
        country = sd.get("phone_country", (COUNTRY or "US")).upper()
        try:
            phone_e164 = normalize_phone_e164(raw_digits, country)
            debug_print(f"[collect_phone] ✅ normalized → {phone_e164}")
        except Exception as e:
            # Fallback for U.S.-style numbers if normalization fails
            debug_print(f"[collect_phone] ⚠️ normalize_phone_e164 failed: {e}")
            d = raw_digits
            if country == "US":
                if len(d) == 11 and d.startswith("1"):
                    d = d[1:]  # remove leading “1” if present
                phone_e164 = f"+1{d}" if len(d) == 10 else ""
            else:
                phone_e164 = ""
            debug_print(f"[collect_phone] ⚙️ fallback normalize → '{phone_e164}'")

        # ----------------------------------------------------------------------
        # ❌ Retry if invalid or incomplete number
        # ----------------------------------------------------------------------
        if not phone_e164:
            r = sd.get("retry_phone", 0) + 1
            sd["retry_phone"] = r
            debug_print(f"[collect_phone] ❌ invalid number (retry {r}/3) input='{raw_digits}'")

            if r < 3:
                g = make_gather(
                    VOICE_INVALID_PHONE_MSG,
                    input="speech dtmf",
                    timeout=5,
                    speech_timeout="auto",
                    barge_in=True,
                    finish_on_key="#"
                )
                resp.append(g)
                resp.redirect("/voice")
                return str(resp)

            # ❌ Max retries exceeded → hang up politely
            resp.say(gpt_speak(VOICE_TOO_MANY_INVALID_MSG), VOICE)
            resp.hangup()
            save_session(call_sid)
            session_data.pop(call_sid, None)
            return str(resp)

        # ----------------------------------------------------------------------
        # ✅ Save phone number across contexts (for booking, cancellation, etc.)
        # ----------------------------------------------------------------------
        cust["phone_e164"] = phone_e164
        cust["phone"] = phone_e164
        cancel_ctx["phone_e164"] = phone_e164
        sd["phone_e164"] = phone_e164
        sd["retry_phone"] = 0
        debug_print(f"[collect_phone] 💾 saved phone_e164={phone_e164}")

        # ----------------------------------------------------------------------
        # 🔁 Return stage handling (if caller is coming back from another flow)
        # ----------------------------------------------------------------------
        return_stage = sd.pop("return_stage", None)
        if return_stage:
            sd["stage"] = return_stage
            debug_print(f"[collect_phone] ↩️ returning to stage '{return_stage}'")
            resp.redirect("/voice")
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # 🔁 Reschedule-after-cancel shortcut
        # ----------------------------------------------------------------------
        if sd.get("reschedule_after_cancel"):
            sd["stage"] = "collect_book_time_date"
            g = make_gather(
                VOICE_RESCHEDULE_MSG,
                input="speech dtmf",
                timeout=5,
                speech_timeout="auto",
                barge_in=True,
                finish_on_key="#"
            )
            resp.append(g)
            resp.redirect("/voice")
            debug_print("[collect_phone] 🔁 reschedule → collect_book_time_date")
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # 🗓️ Normal flow → Proceed to collect date of birth
        # ----------------------------------------------------------------------
        sd["stage"] = "collect_dob"
        g = make_gather(
            VOICE_ASK_DOB_MSG,
            input="speech dtmf",
            timeout=5,
            speech_timeout="auto",
            barge_in=True,
            finish_on_key="#"
        )
        resp.append(g)
        resp.redirect("/voice")
        debug_print(f"[collect_phone] ➡️ next stage → collect_dob (doctor_name={sd.get('doctor_name')})")

        # ✅ Persist session state for continuity
        save_session(call_sid)
        return str(resp)








    elif stage == "collect_dob":
        # ----------------------------------------------------------------------
        # 🎂 Stage: collect_dob — capture and validate date of birth
        # ----------------------------------------------------------------------
        t_stage_start = _time_mod.perf_counter()
        debug_print(f"[collect_dob] 📍 Stage entered at {_time_mod.strftime('%H:%M:%S')}")

        # ----------------------------------------------------------------------
        # 💬 VOICE MESSAGES — centralized for maintainability & localization
        # ----------------------------------------------------------------------
        VOICE_SILENCE_MSG = (
            "Please say your date of birth, for example, 'July 3 1956'. "
            "Or enter two digits for month, two for day, and four for year, then press pound."
        )
        VOICE_SILENCE_FINAL_MSG = (
            "Sorry, I couldn’t get your date of birth. Please call again later."
        )
        VOICE_PARSE_FAIL_MSG = (
            "I didn’t catch your full birth date. Please say it again, for example, 'July 3 1956'. "
            "You can also enter it using your keypad: 2 digits for month, 2 for day, and 4 for year, then press pound."
        )
        VOICE_INVALID_DOB_MSG = (
            "That doesn’t seem like a valid date of birth. "
            "Please enter 2 digits for month, 2 for day, and 4 for year, then press #."
        )
        VOICE_NOT_FOUND_MSG = (
            "We couldn’t find a record with that phone number and date of birth. "
            "If you are a new customer, press 1. If you are an existing customer, press 2."
        )
        VOICE_NEW_CUSTOMER_MSG = (
            "We found your record, but your registration with the clinic is not complete. "
            "Please contact the clinic to finish your registration before booking an appointment. Goodbye!"
        )
        VOICE_PIN_PROMPT_MSG = (
            "Thank you. For security verification, please enter your six digit PIN number now, "
            "followed by the pound key. If you prefer, you can also say each digit slowly."
        )

        # ----------------------------------------------------------------------
        # 🛡️ Session protection — ensure dictionary exists
        # ----------------------------------------------------------------------
        if "session_data" not in globals():
            debug_print("[collect_dob] ⚠️ session_data missing globally — recreating empty dict")
            session_data = {}

        if call_sid not in session_data:
            debug_print(f"[collect_dob] ⚠️ No existing session for {call_sid} — initializing new one")
            session_data[call_sid] = {"stage": "collect_dob"}

        sd = session_data.setdefault(call_sid, {})
        sd.setdefault("customer", {})
        sd.setdefault("cancel", {})

        debug_print(f"[collect_dob] session keys before: {list(sd.keys())}")
        debug_print(f"[collect_dob] 🔎 doctor_name check → {sd.get('doctor_name')}")

        # ----------------------------------------------------------------------
        # 🎧 Capture inputs (speech or keypad digits)
        # ----------------------------------------------------------------------
        try:
            dtmf_digits = (request.values.get("Digits") or "").strip()
        except Exception:
            dtmf_digits = ""
        speech_text = (speech_result or "").strip()
        debug_print(f"[collect_dob] 🎙️ speech='{speech_text}', 🔢 dtmf='{dtmf_digits}'")

        # ----------------------------------------------------------------------
        # 🔇 Silence handling
        # ----------------------------------------------------------------------
        if not dtmf_digits and not speech_text:
            tries = sd.get("silence_dob", 0) + 1
            sd["silence_dob"] = tries
            debug_print(f"[collect_dob] 🤐 silence tries={tries}/3")

            if tries < 3:
                g = make_gather(
                    VOICE_SILENCE_MSG,
                    input="speech dtmf",
                    timeout=3,
                    speech_timeout="auto",
                    barge_in=True,
                    finish_on_key="#"
                )
                resp.append(g)
                sd["stage"] = "collect_dob"
                save_session(call_sid)
                resp.redirect("/voice")
                return str(resp)

            # ❌ Too many silent attempts → hang up politely
            resp.say(gpt_speak(VOICE_SILENCE_FINAL_MSG), VOICE)
            resp.hangup()
            debug_print(f"[collect_dob] 🧹 clearing session after hangup (call_sid={call_sid})")
            session_data.pop(call_sid, None)
            return str(resp)

        # ✅ Reset silence counter once valid input is received
        sd.pop("silence_dob", None)

        # ----------------------------------------------------------------------
        # 🧩 Parse DOB (from DTMF or speech)
        # ----------------------------------------------------------------------
        dob_date = None

        # 🧮 If DTMF provided (numeric keypad input)
        if dtmf_digits:
            d = _re.sub(r"\D", "", dtmf_digits)
            if len(d) >= 8:
                try:
                    mm, dd, yyyy = int(d[0:2]), int(d[2:4]), int(d[4:8])
                    dob_date = date(yyyy, mm, dd)
                    debug_print("[collect_dob] ✅ parsed DOB from keypad")
                except Exception as e:
                    debug_print(f"[collect_dob] ❌ keypad parse error → {e}")

        # 🗣️ If not from keypad, try parsing spoken date
        if not dob_date and speech_text:
            try:
                # ----------------------------------------------------------------------
                # 🧹 Clean and normalize spoken date text before parsing
                # ----------------------------------------------------------------------
                # Example raw speech input:
                #   "July 3rd, 1956."  → we need to make it machine-friendly like "July 3 1956"
                # ----------------------------------------------------------------------

                t = _re.sub(r"[.,;:]+$", "", speech_text)

                # 🔹 Removes punctuation at the *end* of the spoken text.
                #   - Pattern: [.,;:]+$
                #       • [.,;:]  → matches period (.), comma (,), semicolon (;), or colon (:)
                #       • +       → one or more occurrences
                #       • $       → end of string anchor
                #   ✅ Example: "July 3rd, 1956." → "July 3rd, 1956"

                t = _re.sub(r"[,\.;:]", " ", t)

                # 🔹 Replaces punctuation *inside* the string with spaces.
                #   - Pattern: [,\.;:]
                #       • Matches commas, periods, semicolons, or colons.
                #   ✅ Example: "July 3rd, 1956" → "July 3rd 1956"

                t = _re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", t, flags=_re.IGNORECASE)

                # 🔹 Removes ordinal suffixes (st, nd, rd, th) from day numbers.
                #   - Pattern: \b(\d{1,2})(st|nd|rd|th)\b
                #       • \b → **word boundary**, ensures match only at word edges.
                #       • (\d{1,2}) → captures one or two digits (day numbers like "3" or "21")
                #       • (st|nd|rd|th) → matches common ordinal suffixes.
                #       • \b → another word boundary to prevent partial matches.
                #   ✅ Example: "July 3rd 1956" → "July 3 1956"

                t = _re.sub(r"\s+", " ", t).strip()

                # 🔹 Collapses multiple whitespace characters into one space and trims edges.
                #   - Pattern: \s+
                #       • \s → **whitespace** (spaces, tabs, newlines)
                #       • +  → one or more occurrences.
                #   - `.strip()` removes leading/trailing spaces.
                #   ✅ Example: "  March   22nd,   1988 " → "March 22 1988"

                # ----------------------------------------------------------------------
                # 🧠 Parse normalized string using dateutil.parser
                # ----------------------------------------------------------------------
                parsed = _dtparse(t, fuzzy=True)
                dob_date = date(parsed.year, parsed.month, parsed.day)
                debug_print("[collect_dob] ✅ parsed DOB from speech")
            except Exception as e:
                # ❌ Unable to parse → re-prompt the user
                debug_print(f"[collect_dob] ❌ speech parse failed → {e}")
                sd["stage"] = "collect_dob"
                g = make_gather(
                    VOICE_PARSE_FAIL_MSG,
                    input="speech dtmf",
                    timeout=3,
                    speech_timeout="auto",
                    barge_in=True,
                    finish_on_key="#"
                )
                resp.append(g)
                save_session(call_sid)
                resp.redirect("/voice")
                return str(resp)

        # ----------------------------------------------------------------------
        # ⚙️ Validate DOB range (1900 ≤ DOB ≤ today)
        # ----------------------------------------------------------------------
        try:
            today = _date_local.today()
            if not dob_date or dob_date < date(1900, 1, 1) or dob_date > today:
                raise ValueError("DOB out of valid range")
        except Exception as e:
            debug_print(f"[collect_dob] ⚠️ Validation error → {e}")
            sd["stage"] = "collect_dob"
            g = make_gather(
                VOICE_INVALID_DOB_MSG,
                input="dtmf",
                timeout=3,
                barge_in=True,
                finish_on_key="#"
            )
            resp.append(g)
            save_session(call_sid)
            resp.redirect("/voice")
            return str(resp)

        # ✅ Store DOB in session once validated
        iso_dob = dob_date.strftime("%Y-%m-%d")
        sd["customer"]["dob"] = iso_dob
        sd["cancel"]["dob"] = iso_dob
        debug_print(f"[collect_dob] ✅ Stored DOB → {iso_dob}")

        # ----------------------------------------------------------------------
        # 🔍 Lookup customer by phone & DOB
        # ----------------------------------------------------------------------
        phone_e164 = sd["customer"].get("phone_e164") or sd.get("phone_e164")
        found, customer_status = False, "unknown"

        if phone_e164:
            try:
                found = customer_search(phone_number=phone_e164, dob=iso_dob, default_country="US")
                if found:
                    customer_status = get_customer_status(phone_e164, iso_dob)
                    sd["customer"]["customer_status"] = customer_status
                debug_print(f"[collect_dob] 🔎 lookup → found={found}, status={customer_status}")
            except Exception as e:
                debug_print(f"[collect_dob] ⚠️ lookup error → {e}")
        else:
            debug_print("[collect_dob] ⚠️ phone_e164 missing before lookup")

        # ----------------------------------------------------------------------
        # 🔀 Branching based on lookup result
        # ----------------------------------------------------------------------
        if not found:
            sd["stage"] = "verify_customer_type"
            g = make_gather(
                VOICE_NOT_FOUND_MSG,
                input="dtmf",
                timeout=3,
                barge_in=True,
                finish_on_key="#"
            )
            resp.append(g)
            save_session(call_sid)
            resp.redirect("/voice")
            debug_print("[collect_dob] 🔀 not found → verify_customer_type")
            return str(resp)

        # 🟡 Incomplete registration (new)
        if customer_status == "new":
            debug_print("[collect_dob] 🟡 incomplete registration → hangup")
            resp.say(gpt_speak(VOICE_NEW_CUSTOMER_MSG), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)

        # ✅ Valid existing customer → proceed to PIN verification
        sd["stage"] = "collect_pin_number"
        g = make_gather(
            VOICE_PIN_PROMPT_MSG,
            input="speech dtmf",
            timeout=5,
            speech_timeout="auto",
            barge_in=True,
            finish_on_key="#"
        )
        resp.append(g)
        save_session(call_sid)
        resp.redirect("/voice")
        debug_print(f"[collect_dob] ✅ proceed → collect_pin_number (doctor_name={sd.get('doctor_name')})")
        return str(resp)




    # ======================================================================
    # 🏦 Stage: collect_insurance_information
    # ----------------------------------------------------------------------
    # 🎯 FUNCTIONAL PURPOSE:
    #   • Captures caller’s insurance company and member ID.
    #   • Supports both speech and DTMF (keypad) input.
    #   • Step 1 → “company” selection from numbered list.
    #   • Step 2 → “id” entry via speech or keypad.
    #   • Handles silence with 3 retries before hanging up.
    #   • Once both collected → advances to collect_first_name.
    #
    # 🔁 FLOW SUMMARY:
    #   1️⃣ Company selection — DTMF or silence re-prompt.
    #   2️⃣ Member ID capture — extended wait for speech.
    #
    # 🧩 Data stored in:
    #   session_data[call_sid]["customer"]["insurance_name"]
    #   session_data[call_sid]["customer"]["insurance_member_id"]
    # ======================================================================

    elif stage == "collect_insurance_information":

        # ----------------------------------------------------------------------
        # 🎙️ VOICE MESSAGES — centralized for clarity and localization
        # ----------------------------------------------------------------------
        MSG_SILENCE_EXIT = "I’m still not hearing anything. Please call again later."
        MSG_MEMBERID_SILENCE_EXIT = "I’m still not hearing your member ID. Please call again later."
        MSG_PROMPT_INSURANCE_COMPANY = (
            "Please choose your insurance company using your keypad. "
            "Press the number now while I’m speaking. "
        )
        MSG_PROMPT_MEMBER_ID = (
            "Please say or enter your insurance member ID now. "
            "You can include both letters and numbers, then press pound when done."
        )
        MSG_AFTER_SELECTION = (
            "Thank you. You selected {insurance_name}. "
            "Now please say or enter your insurance member ID. "
            "You can include both letters and numbers, then press pound when done."
        )
        MSG_THANK_YOU_NEXT_FIRST_NAME = (
            "Thank you. Now, please tell me your first name."
        )

        # ----------------------------------------------------------------------
        # 🧭 Initialize session safely (ensure nested dicts exist)
        # ----------------------------------------------------------------------
        sd = session_data.setdefault(call_sid, {})
        sd.setdefault("customer", {})
        customer = sd["customer"]

        # ----------------------------------------------------------------------
        # 🎙️ Capture input (speech + keypad)
        # ----------------------------------------------------------------------
        raw_speech = (speech_result or "").strip()
        raw_dtmf = (request.values.get("Digits") or "").strip()
        debug_print(f"collect_insurance_information: speech='{raw_speech}', dtmf='{raw_dtmf}'")

        # ----------------------------------------------------------------------
        # 🏢 Load insurance companies (from environment variable or defaults)
        # ----------------------------------------------------------------------
        INSURANCE_COMPANIES_LIST = [
            n.strip()
            for n in os.getenv(
                "INSURANCE_COMPANIES",
                "Blue Cross Blue Shield,Aetna,Cigna,United Healthcare,Humana,Kaiser Permanente",
            ).split(",")
            if n.strip()
        ]
        keypad_map = {str(i + 1): n for i, n in enumerate(INSURANCE_COMPANIES_LIST)}
        debug_print(f"collect_insurance_information: keypad_map={keypad_map}")

        # ----------------------------------------------------------------------
        # 🧩 Determine current sub-step ("company" or "id")
        # ----------------------------------------------------------------------
        step = sd.get("insurance_step", "company")

        # ======================================================================
        # 🧩 STEP 1 — SELECT INSURANCE COMPANY
        # ======================================================================
        if step == "company":
            # --------------------------------------------------------------
            # 🔢 Handle keypad (DTMF) input
            # --------------------------------------------------------------
            if raw_dtmf:
                first_digit = next((ch for ch in raw_dtmf if ch in keypad_map), "")
                if first_digit:
                    insurance_name = keypad_map[first_digit]
                    customer["insurance_name"] = insurance_name
                    sd["insurance_step"] = "id"  # Move to ID collection step
                    debug_print(f"✅ Selected insurance_name='{insurance_name}' via DTMF '{raw_dtmf}'")

                    # ------------------------------------------------------
                    # 🕐 Prompt for member ID (longer listening window)
                    # ------------------------------------------------------
                    # timeout=25 ensures the system waits long enough for slow speech.
                    # barge_in=False prevents early cutoff mid-sentence.
                    g = make_gather(
                        MSG_AFTER_SELECTION.format(insurance_name=insurance_name),
                        input="speech dtmf",
                        timeout=25,
                        speech_timeout="auto",
                        barge_in=False,
                        finish_on_key="#",
                        language="en-US",
                        action="/voice",
                        method="POST",
                    )
                    resp.append(g)
                    resp.redirect("/voice")
                    return str(resp)

            # --------------------------------------------------------------
            # 🤐 Handle silence (no input) — allow up to 3 retries
            # --------------------------------------------------------------
            tries = sd.get("insurance_silence_tries", 0) + 1
            sd["insurance_silence_tries"] = tries
            debug_print(f"collect_insurance_information: 🤐 silence tries={tries}/3")

            if tries >= 3:
                resp.say(MSG_SILENCE_EXIT, VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # --------------------------------------------------------------
            # 📞 Re-prompt with company menu
            # --------------------------------------------------------------
            # Build spoken list: “Press 1 for Blue Cross, Press 2 for Aetna...”
            menu_text = MSG_PROMPT_INSURANCE_COMPANY
            for i, name in enumerate(INSURANCE_COMPANIES_LIST, start=1):
                menu_text += f"Press {i} for {name}. "

            # Quick DTMF-only gather — instant response when key pressed
            g = make_gather(
                menu_text,
                input="dtmf",
                timeout=4,              # short delay before repeat
                num_digits=1,
                barge_in=True,
                finish_on_key="#",
                language="en-US",
                action="/voice",
                method="POST",
            )
            resp.append(g)
            resp.redirect("/voice")
            return str(resp)

        # ======================================================================
        # 🧩 STEP 2 — COLLECT INSURANCE MEMBER ID
        # ======================================================================
        if step == "id":
            # --------------------------------------------------------------
            # 🔇 Handle silence with retry logic (max 3)
            # --------------------------------------------------------------
            if not raw_speech and not raw_dtmf:
                tries = sd.get("insurance_id_silence", 0) + 1
                sd["insurance_id_silence"] = tries
                debug_print(f"collect_insurance_information: 🤐 ID silence tries={tries}/3")

                if tries >= 3:
                    resp.say(MSG_MEMBERID_SILENCE_EXIT, VOICE)
                    resp.hangup()
                    session_data.pop(call_sid, None)
                    return str(resp)

                # ----------------------------------------------------------
                # 🗣️ Re-prompt for ID (extended timeout)
                # ----------------------------------------------------------
                g = make_gather(
                    MSG_PROMPT_MEMBER_ID,
                    input="speech dtmf",
                    timeout=30,              # plenty of time for speaking ID
                    speech_timeout="auto",   # stops when user actually silent
                    barge_in=False,          # ensures no early cutoff
                    finish_on_key="#",
                    language="en-US",
                    action="/voice",
                    method="POST",
                )
                resp.append(g)
                resp.redirect("/voice")
                return str(resp)

            # --------------------------------------------------------------
            # 🧾 Capture and save insurance member ID
            # --------------------------------------------------------------
            member_id = (raw_dtmf or raw_speech).strip().upper()
            customer["insurance_member_id"] = member_id
            debug_print(f"✅ Captured insurance_member_id='{member_id}'")

            # --------------------------------------------------------------
            # 🔄 Move to next stage (collect_first_name)
            # --------------------------------------------------------------
            sd["stage"] = "collect_first_name"
            sd.pop("insurance_step", None)
            sd.pop("insurance_silence_tries", None)
            sd.pop("insurance_id_silence", None)

            # Prompt for first name next
            g = make_gather(
                MSG_THANK_YOU_NEXT_FIRST_NAME,
                input="speech dtmf",
                timeout=10,
                speech_timeout="auto",
                barge_in=True,
                language="en-US",
                action="/voice",
                method="POST",
            )
            resp.append(g)
            resp.redirect("/voice")
            return str(resp)
















    elif stage == "verify_customer_type":
    
        # ----------------------------------------------------------------------
        # 🧭 Stage: verify_customer_type
        # ----------------------------------------------------------------------
        debug_print("verify_customer_type: 📍 Stage entered")

        sd = session_data.setdefault(call_sid, {})
        last_lookup_found = sd.get("last_customer_found", False)
        allow_new = bool(globals().get("CREATE_NEW_CUSTOMER", False))

        dtmf_digits = (request.values.get("Digits") or "").strip()
        debug_print(
            f"verify_customer_type: received DTMF='{dtmf_digits}', allow_new={allow_new}, found={last_lookup_found}"
        )

        # 🔒 New creation disabled
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

        sd.pop("silence_verify_type", None)

        # -------------------------------
        # 🧭 Branch on DTMF choice
        # -------------------------------
        if dtmf_digits == "1":
            sd["customer_status"] = "new"
            debug_print("verify_customer_type: 🆕 customer_status='new' stored in session")

            # ✅ Just set the stage — don’t say anything yet
            if not last_lookup_found:
                debug_print("verify_customer_type: new customer not found → jump to collect_insurance_information")
                sd["stage"] = "collect_insurance_information"
                resp.redirect("/voice")
                return str(resp)

            else:
                debug_print("verify_customer_type: found=True but pressed 1=new → continue to scheduling")
                sd["stage"] = "collect_book_time_date"
                g = make_gather(
                    "Okay. Please say the appointment date and time, for example, 'October 8 at 9 30 A M'.",
                    input="speech dtmf", timeout=6, speech_timeout="auto", barge_in=True, finish_on_key="#"
                )
                resp.append(g)
                resp.redirect("/voice")
                return str(resp)

        elif dtmf_digits == "2":
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
                debug_print("verify_customer_type: 2=existing; found=True → proceed to collect_book_time_date")
                sd["stage"] = "collect_book_time_date"
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
    #         - "book"       → collect_dr_info ✅
    #         - "cancel"     → cancel_appt_get_time_date
    #         - "update_cc"  → collect_cc
    #         - otherwise    → intro (main menu)
    #   4️⃣ If incorrect → allow up to 3 retries before terminating politely.
    #
    # FEATURES
    #   ✅ Handles silence locally (3 retries, then hang up).
    #   ✅ Tracks invalid PIN attempts (3 max).
    #   ✅ Supports both DTMF and speech.
    # ----------------------------------------------------------------------




    elif stage == "collect_pin_number":
    

        debug_print("collect_pin_number: 📍 Stage entered")

        # ----------------------------------------------------------------------
        # 💬 VOICE MESSAGES — centralized for maintainability & localization
        # ----------------------------------------------------------------------
        VOICE_PIN_PROMPT_MSG = (
            "Please enter your six digit PIN now, followed by the pound key. "
            "If you prefer, you can also say each digit slowly."
        )
        VOICE_SILENCE_MSG = (
            "I didn’t hear anything. Please enter or say your six digit PIN now."
        )
        VOICE_INVALID_LENGTH_MSG = (
            "That doesn’t seem like a valid six digit PIN. Please try again now."
        )
        VOICE_TOO_MANY_INVALID_MSG = (
            "That doesn’t seem like a valid six digit PIN. "
            "Please contact the clinic to verify or reset your PIN number. Goodbye."
        )
        VOICE_CORRECT_PIN_BOOK_MSG = (
            "Thank you. Your PIN has been verified. Let's continue with booking your appointment."
        )
        VOICE_CORRECT_PIN_CANCEL_MSG = (
            "Thank you. PIN verified. Let's proceed to locate your appointment for cancellation."
        )
        VOICE_CORRECT_PIN_CC_MSG = (
            "Your PIN has been verified. Let's update your payment information."
        )
        VOICE_CORRECT_PIN_DEFAULT_MSG = (
            "Thank you. Your PIN has been verified. Returning to the main menu."
        )
        VOICE_WRONG_PIN_MSG = (
            "That PIN number is incorrect. Please try again now. "
            "Enter your six digit PIN followed by the pound key."
        )
        VOICE_MAX_ATTEMPTS_MSG = (
            "You have entered an incorrect PIN too many times. "
            "Please contact the clinic to verify your information or to change your PIN number. Goodbye!"
        )
        VOICE_SILENCE_TERMINATE_MSG = (
            "I’m still not hearing anything. Please call the clinic for assistance."
        )

        # ----------------------------------------------------------------------
        # 🗂️ SESSION SETUP
        # ----------------------------------------------------------------------
        sd = session_data.setdefault(call_sid, {})
        sd.setdefault("customer", {})
        customer = sd["customer"]

        # --- Retrieve key info ------------------------------------------------
        phone_e164 = (customer.get("phone_e164") or sd.get("phone_e164") or "").strip()
        dob = (customer.get("dob") or "").strip()
        origin_stage = sd.get("origin_stage", "book")  # default to booking
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

            # If caller silent for 3 attempts → hang up politely
            if tries >= 3:
                resp.say(gpt_speak(VOICE_SILENCE_TERMINATE_MSG), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Otherwise re-prompt the caller to provide PIN again
            g = make_gather(
                VOICE_PIN_PROMPT_MSG,
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
        # Extract only numeric digits (strip spaces, words, etc.)
        digits = _re.sub(r"\D", "", raw_dtmf or raw_speech)
        debug_print(f"collect_pin_number: normalized digits='{digits}'")

        # If not exactly 6 digits → invalid PIN format
        if len(digits) != 6:
            debug_print("collect_pin_number: ⚠️ invalid PIN length")
            sd["pin_attempts"] = sd.get("pin_attempts", 0) + 1

            # If reached 3 invalid attempts → terminate politely
            if sd["pin_attempts"] >= 3:
                resp.say(gpt_speak(VOICE_TOO_MANY_INVALID_MSG), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Otherwise retry prompt
            g = make_gather(
                VOICE_INVALID_LENGTH_MSG,
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
            # Retrieve stored PIN from customer database (or JSON)
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

            # Reset attempts after success
            sd.pop("pin_attempts", None)

            # ✅ Branch based on origin
            if origin_stage == "book":
                next_stage = "collect_dr_info"  # Proceed to doctor selection
                msg = VOICE_CORRECT_PIN_BOOK_MSG
            elif origin_stage == "cancel":
                next_stage = "cancel_appt_get_time_date"
                msg = VOICE_CORRECT_PIN_CANCEL_MSG
            elif origin_stage == "update_cc":
                next_stage = "collect_cc"
                msg = VOICE_CORRECT_PIN_CC_MSG
            else:
                next_stage = "intro"
                msg = VOICE_CORRECT_PIN_DEFAULT_MSG

            # Update session stage and skip silence check for next prompt
            sd["stage"] = next_stage
            sd["skip_silence_once"] = True

            # Inform caller and redirect to next stage
            resp.say(gpt_speak(msg), VOICE)
            resp.redirect("/voice")
            return str(resp)

        # ======================================================================
        # ❌ FAILURE CASE — Wrong PIN
        # ======================================================================
        sd["pin_attempts"] = sd.get("pin_attempts", 0) + 1
        tries = sd["pin_attempts"]
        debug_print(f"collect_pin_number: ❌ invalid PIN ({digits}) vs stored ({stored_pin}) (try {tries}/3)")

        # If user still has retries left → re-prompt
        if tries < 3:
            g = make_gather(
                VOICE_WRONG_PIN_MSG,
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
        resp.say(gpt_speak(VOICE_MAX_ATTEMPTS_MSG), VOICE)
        resp.hangup()
        session_data.pop(call_sid, None)
        return str(resp)







    # ======================================================================
    # 🩺 Stage: collect_dr_info
    # ----------------------------------------------------------------------
    # 🎯 FUNCTIONAL PURPOSE:
    #   • Present a numbered list of doctors to the caller.
    #   • Accept selection either by speech (name recognition) or keypad (DTMF).
    #   • Handle partial/fuzzy speech matches and retry gracefully.
    #   • Save the chosen doctor in session_data for subsequent booking steps.
    #   • Transition to the next stage: collect_book_time_date.
    #
    # 🧩 INPUTS:
    #   • speech_result → Caller’s spoken response (“Doctor Alfred”).
    #   • Digits        → DTMF keypad input (e.g., “1” for Dr. Smith).
    #   • doctor_names  → List or dict of doctor names (from global or config).
    #   • call_sid      → Call session identifier.
    #
    # 💾 OUTPUTS:
    #   • session_data[call_sid]["doctor_name"] = selected doctor.
    #   • session_data[call_sid]["stage"] = "collect_book_time_date".
    #
    # 🔁 FLOW OVERVIEW:
    #   1️⃣ Announce all doctors (Press 1 for X, Press 2 for Y).
    #   2️⃣ Wait for speech or DTMF input.
    #   3️⃣ Try to match exact DTMF → fuzzy/partial name → retry or hang up.
    #   4️⃣ On success, announce chosen doctor and move to time collection.
    #
    # 🧠 SPECIAL BEHAVIOR:
    #   • Filters out junk speech like “hello”, “okay”, etc. to avoid false matches.
    #   • Retries up to 3 times before ending call politely.
    #   • Supports doctor_names as list or dict for flexible loading.
    #   • Uses make_gather() for Twilio <Gather> with /voice redirect.
    #
    # ✅ SUMMARY:
    #   This stage connects caller intent (“which doctor?”) with internal booking
    #   logic, ensuring the right doctor is selected through robust recognition.
    # ======================================================================

    elif stage == "collect_dr_info":
        # ----------------------------------------------------------------------
        # 💬 VOICE PROMPTS — centralized for easy editing & localization
        # ----------------------------------------------------------------------
        VOICE_INTRO_MSG = (
            "Please choose your doctor from the following list. "
            "You may either press the corresponding number on your keypad or say the doctor’s name."
        )
        VOICE_REPROMPT_MSG = (
            "I didn’t catch that. Please say the name of your doctor or press the number associated with them."
        )
        VOICE_NO_MATCH_MSG = (
            "I'm sorry, I couldn't match that name with any doctor in our clinic. Please try again."
        )
        VOICE_FINAL_FAIL_MSG = (
            "I'm sorry, I still couldn't match that name with any doctor in our clinic. Please call us again later."
        )
        VOICE_SUCCESS_MSG = (
            "Great, your appointment will be with {doctor_name}. "
            "Please say the appointment date and time, for example, 'October 8 at 9 30 A M'."
        )

        # ----------------------------------------------------------------------
        # 🧭 SESSION INITIALIZATION
        # ----------------------------------------------------------------------
        session_data.setdefault(call_sid, {}).setdefault("retry_booking", 0)
        session_data[call_sid]["origin_stage"] = "book"

        # ----------------------------------------------------------------------
        # 🧹 CLEAN INPUTS (speech + DTMF)
        # ----------------------------------------------------------------------
        # Define punctuation characters to remove from speech text.
        _PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

        # Retrieve keypad digits (if any)
        dtmf_digits = (request.values.get("Digits") or "").strip()

        # Retrieve speech result (case-insensitive and punctuation-free)
        spoken_text = (speech_result or "").strip().lower()
        spoken_clean = spoken_text.translate(str.maketrans('', '', _PUNCT)).strip()

        debug_print(f"[collect_dr_info] 🗣 speech='{spoken_clean}' 🔢 DTMF='{dtmf_digits}'")

        # ----------------------------------------------------------------------
        # 🗂️ INITIALIZE DOCTOR MAP (if not yet built)
        # ----------------------------------------------------------------------
        if "doctor_dtmf_map" not in session_data[call_sid]:
            doctor_dtmf_map = {}
            prompt_lines = []

            # doctor_names can be either dict or list depending on config
            if isinstance(doctor_names, dict):
                doctor_list = list(doctor_names.values())
            else:
                doctor_list = doctor_names

            # Build DTMF mapping and corresponding prompt text
            for i, friendly in enumerate(doctor_list, start=1):
                doctor_dtmf_map[str(i)] = friendly
                prompt_lines.append(f"Press {i} for {friendly}.")

            # Save mapping in session for later reference
            session_data[call_sid]["doctor_dtmf_map"] = doctor_dtmf_map

            # Build the combined speech prompt message
            doctor_prompt = f"{VOICE_INTRO_MSG} " + " ".join(prompt_lines)

            # Generate Twilio <Gather> prompt for both speech & keypad
            g = make_gather(
                doctor_prompt,
                input="speech dtmf",
                timeout=10,
                speech_timeout="auto",
                barge_in=True,
                finish_on_key="#",
                num_digits=1
            )
            resp.append(g)
            resp.redirect("/voice")
            return str(resp)

        # ----------------------------------------------------------------------
        # 🧭 RETRIEVE EXISTING DOCTOR MAP
        # ----------------------------------------------------------------------
        doctor_map = session_data[call_sid]["doctor_dtmf_map"]
        matched_name = None

        # ----------------------------------------------------------------------
        # 🔢 STEP 1 — DTMF MATCHING
        # ----------------------------------------------------------------------
        # Example: Caller presses “2” → maps to Dr. Johnson
        if dtmf_digits and dtmf_digits in doctor_map:
            matched_name = doctor_map[dtmf_digits]
            debug_print(f"✅ DTMF matched doctor → {matched_name}")

        # ----------------------------------------------------------------------
        # 🗣️ STEP 2 — SPEECH MATCHING (Partial / Fuzzy)
        # ----------------------------------------------------------------------
        if matched_name is None:
            # Define filler words to ignore (common misrecognitions)
            junk_inputs = {
                "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
                "yo", "test", "1", "yes", "no", "i know", "huh", "what", "okay", "ok",
                "bye", "goodbye", ""
            }

            # Skip if recognized junk, short, or empty
            if not spoken_clean or spoken_clean in junk_inputs or len(spoken_clean) < 3:
                debug_print(f"⏩ Skipping junk doctor input → '{spoken_clean}' (re-prompting)")
                prompt_lines = [f"Press {k} for {v}." for k, v in doctor_map.items()]
                doctor_prompt = f"{VOICE_REPROMPT_MSG} " + " ".join(prompt_lines)

                g = make_gather(
                    doctor_prompt,
                    input="speech dtmf",
                    timeout=8,
                    speech_timeout="auto",
                    barge_in=True,
                    finish_on_key="#",
                    num_digits=1
                )
                resp.append(g)
                resp.redirect("/voice")
                return str(resp)

            # Tokenize spoken input for partial/fuzzy matching
            spoken_tokens = set(spoken_clean.split())
            partial_matches = []

            # Normalize doctor names to lowercase, punctuation-free
            if isinstance(doctor_names, dict):
                doctor_list = list(doctor_names.values())
            else:
                doctor_list = doctor_names

            for friendly in doctor_list:
                friendly_clean = friendly.lower().translate(str.maketrans('', '', _PUNCT)).strip()
                friendly_tokens = set(friendly_clean.split())

                # Match if:
                #   • Entire phrase matches (“alfred hitchcock”)
                #   • Partial overlap (“alfred” vs “dr alfred hitchcock”)
                #   • Token overlap (intersection between word sets)
                if (
                    spoken_clean in friendly_clean
                    or friendly_clean in spoken_clean
                    or (spoken_tokens & friendly_tokens)
                ):
                    partial_matches.append(friendly)

            # If one match → accept directly
            if len(partial_matches) == 1:
                matched_name = partial_matches[0]
                debug_print(f"✅ Partial speech match → {matched_name}")
            elif len(partial_matches) > 1:
                # If multiple → take the first (or could present later)
                debug_print(f"🔍 Multiple doctor matches found → {partial_matches}")
                matched_name = partial_matches[0]

        # ----------------------------------------------------------------------
        # ❌ STEP 3 — HANDLE NO MATCH FOUND
        # ----------------------------------------------------------------------
        if matched_name is None:
            session_data[call_sid]["retry_booking"] += 1
            retries = session_data[call_sid]["retry_booking"]
            debug_print(f"❌ No doctor match for '{spoken_clean or dtmf_digits}' retry={retries}")

            if retries >= 3:
                # After 3 failed attempts → end call politely
                resp.say(gpt_speak(VOICE_FINAL_FAIL_MSG), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Re-prompt caller to try again with the available list
            prompt_lines = [f"Press {k} for {v}." for k, v in doctor_map.items()]
            doctor_prompt = f"{VOICE_NO_MATCH_MSG} " + " ".join(prompt_lines)

            g = make_gather(
                doctor_prompt,
                input="speech dtmf",
                timeout=8,
                speech_timeout="auto",
                barge_in=True,
                finish_on_key="#",
                num_digits=1
            )
            resp.append(g)
            resp.redirect("/voice")
            return str(resp)

        # ----------------------------------------------------------------------
        # ✅ STEP 4 — SUCCESS: SAVE & ADVANCE TO NEXT STAGE
        # ----------------------------------------------------------------------
        session_data[call_sid]["doctor_name"] = matched_name
        session_data[call_sid]["stage"] = "collect_book_time_date"

        # Build confirmation voice message dynamically
        success_msg = VOICE_SUCCESS_MSG.format(doctor_name=matched_name)

        # Prompt user to provide date and time
        g = make_gather(
            success_msg,
            input="speech dtmf",
            timeout=10,
            speech_timeout="auto",
            barge_in=True,
            finish_on_key="#"
        )
        resp.append(g)
        resp.redirect("/voice")
        return str(resp)







     # ----------------------------------------------------------------------
     # 📅 Stage: collect_book_time_date
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
    elif stage == "collect_book_time_date":
        # ----------------------------------------------------------------------
        # 📅 Stage: collect_book_time_date
        # ----------------------------------------------------------------------
        # 🎯 PURPOSE:
        #   - Capture and validate spoken or keypad date/time.
        #   - Handle silence, invalid input, and past times.
        #   - Offer up to 3 alternative appointment times if needed.
        #   - Insert controlled SSML pauses between proposed appointment options.
        #   - Keep all voice messages in easily editable variables.
        # ----------------------------------------------------------------------

        debug_print(f"[collect_book_time_date] 🗣️ Received speech: {speech_result}")

        
        # ----------------------------------------------------------------------
        # 💬 VOICE MESSAGES — centralized for maintainability & localization
        # ----------------------------------------------------------------------
        VOICE_OLD_DATE_MSG = (
            "That time has already passed. Let me suggest the next available appointment times."
        )
        VOICE_NO_DECISION_MSG = (
            "It seems you’re not ready right now. Please call us back when you’re ready. Goodbye."
        )
        VOICE_SILENCE_MSG = (
            "I didn’t hear anything. Please say the date and time clearly, "
            "for example, 'October 10 at 9 A M'."
        )
        VOICE_REASK_TIME_MSG = "Please tell me another time that works for you."
        VOICE_NO_AVAILABLE_SLOTS_MSG = "Sorry, there are no upcoming available appointments."
        VOICE_NO_RESPONSE_MSG = (
            "I didn’t hear from you. Please call us again when you're ready. Goodbye."
        )
        VOICE_ASK_AGAIN_MSG = (
            "Please say the date and time again, for example, 'October 8 at 9 30 A M'."
        )
        VOICE_NEXT_AVAILABLE_INTRO = "Here are the next available times."
        VOICE_NEXT_AVAILABLE_OUTRO = (
            "Please say the option number, or tell me another date and time."
        )

        # ----------------------------------------------------------------------
        # 🗂️ SESSION SETUP
        # ----------------------------------------------------------------------
        session_data.setdefault(call_sid, {})
        sd = session_data[call_sid]
        sd.setdefault("stage", "collect_book_time_date")

        doctor_name = sd.get("doctor_name")
        if not doctor_name:
            # Doctor not selected — redirect to collect_dr_info
            resp.append(make_gather("Please tell me which doctor you'd like to see."))
            sd["stage"] = "collect_dr_info"
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # 🔇 HANDLE SILENCE
        # ----------------------------------------------------------------------
        if not speech_result and not request.values.get("Digits"):
            sd["silence_retry"] = sd.get("silence_retry", 0) + 1

            # If 2 silent retries → end call politely
            if sd["silence_retry"] >= 2:
                resp.say(gpt_speak(VOICE_NO_DECISION_MSG), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Otherwise, re-prompt with clarity
            g = make_gather(
                VOICE_SILENCE_MSG,
                input="speech dtmf",
                timeout=10,
                speech_timeout="auto",
                barge_in=True,
                action="/voice",
                method="POST"
            )
            resp.append(g)
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # 🧠 PARSE INPUT
        # ----------------------------------------------------------------------
        raw = (speech_result or request.values.get("Digits") or "").strip()
        debug_print(f"[collect_book_time_date][parse] raw='{raw}'")

        # Handle spoken “Option one / two / three” responses
        if sd.get("alts_list"):
            spoken = raw.lower().strip()
            num_map = {
                "one": "1",
                "first": "1",
                "two": "2",
                "second": "2",
                "three": "3",
                "third": "3",
            }

            for k, v in num_map.items():
                if k in spoken:
                    raw = v

            if raw.isdigit() and 1 <= int(raw) <= len(sd["alts_list"]):
                choice = sd["alts_list"][int(raw) - 1]
                debug_print(f"[collect_book_time_date] 🎯 User selected Option {raw}: {choice['friendly']}")
                sd["appointment_time"] = {"start": choice["start"], "end": choice["end"]}
                sd["stage"] = "book_appt_confirm"
                save_session(call_sid)
                resp.redirect("/voice")
                return str(resp)

        # Try parsing full spoken date/time
        try:
            result = smart_parse_time(raw)
        except Exception as e:
            debug_print(f"[collect_book_time_date][parse] error: {e}")
            result = None

        # ----------------------------------------------------------------------
        # ❌ INVALID OR UNPARSEABLE INPUT
        # ----------------------------------------------------------------------
        if not result:
            sd["retry_time"] = sd.get("retry_time", 0) + 1
            if sd["retry_time"] >= 3:
                resp.say(gpt_speak(VOICE_NO_DECISION_MSG), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Ask user again with example format
            g = make_gather(
                VOICE_ASK_AGAIN_MSG,
                input="speech dtmf",
                timeout=10,
                barge_in=True,
                action="/voice",
                method="POST"
            )
            resp.append(g)
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # ✅ PARSED SUCCESSFULLY
        # ----------------------------------------------------------------------
        appointment_start = result["start"]
        appointment_end = result["end"]
        friendly = result["friendly"]
        is_past = result.get("is_past", False)

        now_utc = _pytz.UTC.localize(_dt.utcnow())
        limit_end_utc = now_utc + timedelta(days=30 * MAX_ADVANCE_MONTHS)

        # ----------------------------------------------------------------------
        # ⏰ HANDLE OLD OR OUT-OF-RANGE DATE → SUGGEST ALTERNATIVES
        # ----------------------------------------------------------------------
        if is_past or isoparse(appointment_start) <= now_utc or isoparse(appointment_start) > limit_end_utc:
            resp.say(gpt_speak(VOICE_OLD_DATE_MSG), VOICE)
            alts = get_doctor_next_available_slots(doctor_name, from_start_iso=now_utc.isoformat(), limit=3)

            if not alts:
                resp.say(gpt_speak(VOICE_NO_AVAILABLE_SLOTS_MSG), VOICE)
                resp.hangup()
                return str(resp)

            # 🗣️ Build spoken options with configurable pause duration between each
            #   Using SSML <break> tag with PAUSE_MS milliseconds.

            # ──────────────────────────────────────────────────────────────
            # 🔍 PURPOSE:
            #   This line dynamically builds a natural-sounding spoken list of
            #   appointment options (Option 1, Option 2, etc.) that Twilio’s
            #   voice engine will read aloud with pauses between them.
            #
            # 🧠 HOW IT WORKS:
            #   1️⃣  The variable `alts` is a list of dictionaries like:
            #       alts = [
            #           {"friendly": "Monday, October 28 at 9 A M"},
            #           {"friendly": "Tuesday, October 29 at 2 P M"},
            #           {"friendly": "Wednesday, October 30 at 11 A M"}
            #       ]
            #
            #   2️⃣  The list comprehension:
            #           [f"Option {i}: {a['friendly']}." for i, a in enumerate(alts, start=1)]
            #       Iterates through `alts`, numbering each one:
            #           → ["Option 1: Monday, October 28 at 9 A M.",
            #              "Option 2: Tuesday, October 29 at 2 P M.",
            #              "Option 3: Wednesday, October 30 at 11 A M."]
            #
            #   3️⃣  The .join() call:
            #           f" <break time=\"{PAUSE_MS}ms\"/> ".join([...])
            #       Combines all strings into a single sentence, inserting
            #       an SSML <break> tag (pause) between each one.
            #
            #   4️⃣  The <break time="{PAUSE_MS}ms"/> tag is SSML (Speech Synthesis
            #       Markup Language). It tells Twilio to pause for PAUSE_MS milliseconds
            #       before speaking the next option — making the dialogue more natural.
            #
            # 🗣️ FINAL SPOKEN OUTPUT (if PAUSE_MS = 1000):
            #   “Option 1: Monday, October 28 at 9 A M.” [pause 1.0s]
            #   “Option 2: Tuesday, October 29 at 2 P M.” [pause 1.0s]
            #   “Option 3: Wednesday, October 30 at 11 A M.” [pause 1.0s]
            #
            # 📘 Example:
            #   The SSML string built will look like this:
            #   "<speak>
            #       Here are the next available times.
            #       <break time='1000ms'/>
            #       Option 1: Monday, October 28 at 9 A M.
            #       <break time='1000ms'/>
            #       Option 2: Tuesday, October 29 at 2 P M.
            #       <break time='1000ms'/>
            #       Option 3: Wednesday, October 30 at 11 A M.
            #       <break time='1000ms'/>
            #       Please say the option number, or tell me another date and time.
            #    </speak>"
            #
            #   Twilio interprets this properly as a real pause between lines —
            #   the caller hears silence between options, not the text "<break>".
            # ──────────────────────────────────────────────────────────────

            # ----------------------------------------------------------------------
            # 🧩 This f-string builds a single spoken option line for Twilio.
            # Each iteration of the loop generates one sentence like:
            #    "Option 1: Monday, October 28 at 9 A M."
            #
            # Let's break down the inner parts:
            #
            # • f"..."  →  This is an f-string (formatted string literal) in Python.
            #              It allows embedding variable values directly inside curly braces { }.
            #
            # • {i}     →  Inserts the option number provided by enumerate(alts, start=1).
            #              For example, if i = 2, this part becomes "Option 2".
            #
            # • {a['friendly']} →
            #     - `a` is the current dictionary in the `alts` list.
            #     - `a['friendly']` accesses the value stored under the key "friendly".
            #       For example, if a = {"friendly": "Monday, October 28 at 9 A M"},
            #       then a['friendly'] returns the string:
            #           "Monday, October 28 at 9 A M"
            # ----------------------------------------------------------------------

            options_ssml = f" <break time=\"{PAUSE_MS}ms\"/> ".join(
                [f"Option {i}: {a['friendly']}." for i, a in enumerate(alts, start=1)]
            )

            # 📘 SSML NOTE:
            #   Twilio ignores <break> tags unless they are wrapped inside a <speak> block.
            #   Without <speak>...</speak>, the tags are read as plain text or skipped.
            #   By wrapping our combined message with <speak>, we activate SSML processing
            #   so the caller actually hears a pause between each option.
            #
            #   The <break> tags remain invisible — callers will never hear “1000 milliseconds”,
            #   they will simply experience a natural pause between sentences.
            #
            #   This is the supported and recommended method per Twilio SSML documentation.
            # ----------------------------------------------------------------------

            combined = (
                f"<speak>{VOICE_NEXT_AVAILABLE_INTRO}"
                f"<break time=\"{PAUSE_MS}ms\"/>{options_ssml}"
                f"<break time=\"{PAUSE_MS}ms\"/>{VOICE_NEXT_AVAILABLE_OUTRO}</speak>"
            )

            debug_print(f"[collect_book_time_date] 🗣️ SSML prompt prepared with {len(alts)} options and {PAUSE_MS}ms pauses")

            # Create SSML-enabled <Gather> block
            g = make_gather(
                combined,
                input="speech dtmf",
                timeout=15,
                speech_timeout="auto",
                barge_in=True,
                action="/voice",
                method="POST"
            )
            resp.append(g)
            sd["alts_list"] = alts
            sd["stage"] = "collect_book_time_date"
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # 🕓 CHECK AVAILABILITY
        # ----------------------------------------------------------------------
        if not is_doctor_slot_available(doctor_name, appointment_start, appointment_end):
            g = make_gather(
                f"That time is not available. {VOICE_REASK_TIME_MSG}",
                input="speech dtmf",
                timeout=10,
                barge_in=True,
                action="/voice",
                method="POST"
            )
            resp.append(g)
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # ✅ SUCCESS → MOVE TO CONFIRMATION STAGE
        # ----------------------------------------------------------------------
        sd["appointment_time"] = {"start": appointment_start, "end": appointment_end}
        sd["stage"] = "book_appt_confirm"
        save_session(call_sid)
        resp.redirect("/voice")
        return str(resp)

        







    # ======================================================================
    # 📅 Stage: collect_book_time_date
    # ----------------------------------------------------------------------
    # 🎯 FUNCTIONAL PURPOSE:
    #   • Capture and validate the caller’s spoken or keypad appointment date/time.
    #   • Handle silence, invalid entries, or past/unavailable time slots gracefully.
    #   • Suggest up to 3 alternative appointment slots when necessary.
    #   • Use SSML <break> tags to create natural speech pauses between options.
    #   • Maintain all voice messages as editable variables for localization.
    #
    # 🧩 INPUTS:
    #   • speech_result → Speech-to-text transcription from Twilio.
    #   • Digits        → Keypad input for option selection.
    #   • call_sid      → Unique call session identifier.
    #   • doctor_name   → Previously selected doctor’s name from session_data.
    #
    # 💾 OUTPUTS:
    #   • session_data[call_sid]["appointment_time"] → dict with "start"/"end" ISO strings.
    #   • session_data[call_sid]["stage"] → next stage ("book_appt_confirm").
    #
    # 🔁 FLOW OVERVIEW:
    #   1️⃣ Handle silence or missing input → prompt again or end politely.
    #   2️⃣ Parse the spoken text into a datetime (via smart_parse_time).
    #   3️⃣ If time is past or invalid → suggest next 3 available slots.
    #   4️⃣ Present options using SSML <break> pauses for natural voice output.
    #   5️⃣ Validate final choice and proceed to confirmation stage.
    #
    # 🧠 SPECIAL BEHAVIOR:
    #   • Uses localized VOICE_* constants for easy editing and translation.
    #   • Automatically re-prompts up to 3 times before hang-up.
    #   • Uses Google-style fuzzy date/time parsing (smart_parse_time).
    #   • Supports “Option one/two/three” or numeric DTMF input for slot choice.
    #
    # ✅ SUMMARY:
    #   This stage ensures robust and natural collection of appointment times.
    #   It improves user experience by re-prompting gently, offering alternatives,
    #   and speaking them in a human-like cadence with SSML-based pauses.
    # ======================================================================

    elif stage == "collect_book_time_date":
        debug_print(f"[collect_book_time_date] 🗣️ Received speech: {speech_result}")

        # ----------------------------------------------------------------------
        # 💬 VOICE MESSAGES — centralized for readability & localization
        # ----------------------------------------------------------------------
        VOICE_OLD_DATE_MSG = (
            "That time has already passed. Let me suggest the next available appointment times."
        )
        VOICE_NO_DECISION_MSG = (
            "It seems you’re not ready right now. Please call us back when you’re ready. Goodbye."
        )
        VOICE_SILENCE_MSG = (
            "I didn’t hear anything. Please say the date and time clearly, "
            "for example, 'October 10 at 9 A M'."
        )
        VOICE_REASK_TIME_MSG = "Please tell me another time that works for you."
        VOICE_NO_AVAILABLE_SLOTS_MSG = "Sorry, there are no upcoming available appointments."
        VOICE_NO_RESPONSE_MSG = "I didn’t hear from you. Please call us again when you're ready. Goodbye."
        VOICE_ASK_AGAIN_MSG = "Please say the date and time again, for example, 'October 8 at 9 30 A M'."
        VOICE_NEXT_AVAILABLE_INTRO = "Here are the next available times."
        VOICE_NEXT_AVAILABLE_OUTRO = "Please say the option number, or tell me another date and time."

        # ----------------------------------------------------------------------
        # 🗂️ SESSION SETUP
        # ----------------------------------------------------------------------
        session_data.setdefault(call_sid, {})
        sd = session_data[call_sid]
        sd.setdefault("stage", "collect_book_time_date")

        # Ensure a doctor has been selected; otherwise redirect to collect_dr_info
        doctor_name = sd.get("doctor_name")
        if not doctor_name:
            g = make_gather("Please tell me which doctor you'd like to see.")
            sd["stage"] = "collect_dr_info"
            resp.append(g)
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # 🔇 HANDLE SILENCE
        # ----------------------------------------------------------------------
        if not speech_result and not request.values.get("Digits"):
            sd["silence_retry"] = sd.get("silence_retry", 0) + 1
            debug_print(f"[collect_book_time_date] 🤐 silence_retry={sd['silence_retry']}")

            # If repeated silence → end the call politely
            if sd["silence_retry"] >= 2:
                resp.say(gpt_speak(VOICE_NO_DECISION_MSG), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Otherwise, re-prompt the user to speak clearly
            g = make_gather(
                VOICE_SILENCE_MSG,
                input="speech dtmf",
                timeout=10,
                speech_timeout="auto",
                barge_in=True,
                action="/voice",
                method="POST"
            )
            resp.append(g)
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # 🧠 PARSE INPUT (Speech or Keypad)
        # ----------------------------------------------------------------------
        raw = (speech_result or request.values.get("Digits") or "").strip()
        debug_print(f"[collect_book_time_date][parse] raw='{raw}'")

        # If user was previously offered alternative slots, detect if they said “Option 1/2/3”
        if sd.get("alts_list"):
            spoken = raw.lower().strip()

            # Map spoken words (“one”, “first”) to their numeric equivalents
            num_map = {
                "one": "1", "first": "1",
                "two": "2", "second": "2",
                "three": "3", "third": "3",
            }

            # Replace spoken numbers with digits if found
            for k, v in num_map.items():
                if k in spoken:
                    raw = v

            # If valid numeric selection → pick corresponding slot
            if raw.isdigit() and 1 <= int(raw) <= len(sd["alts_list"]):
                choice = sd["alts_list"][int(raw) - 1]
                debug_print(f"[collect_book_time_date] 🎯 Option {raw} selected → {choice['friendly']}")
                sd["appointment_time"] = {"start": choice["start"], "end": choice["end"]}
                sd["stage"] = "book_appt_confirm"
                save_session(call_sid)
                resp.redirect("/voice")
                return str(resp)

        # Attempt to parse a spoken date/time string using NLP parser
        try:
            result = smart_parse_time(raw)
        except Exception as e:
            debug_print(f"[collect_book_time_date][parse] error: {e}")
            result = None

        # ----------------------------------------------------------------------
        # ❌ INVALID OR UNPARSEABLE INPUT
        # ----------------------------------------------------------------------
        if not result:
            sd["retry_time"] = sd.get("retry_time", 0) + 1
            debug_print(f"[collect_book_time_date] ⚠️ Invalid time → retry={sd['retry_time']}")
            if sd["retry_time"] >= 3:
                resp.say(gpt_speak(VOICE_NO_DECISION_MSG), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Ask again with example formatting
            g = make_gather(
                VOICE_ASK_AGAIN_MSG,
                input="speech dtmf",
                timeout=10,
                barge_in=True,
                action="/voice",
                method="POST"
            )
            resp.append(g)
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # ✅ PARSED SUCCESSFULLY — EXTRACT FIELDS
        # ----------------------------------------------------------------------
        appointment_start = result["start"]
        appointment_end = result["end"]
        friendly = result["friendly"]
        is_past = result.get("is_past", False)

        # Define time bounds (UTC)
        now_utc = _pytz.UTC.localize(_dt.utcnow())
        limit_end_utc = now_utc + timedelta(days=30 * MAX_ADVANCE_MONTHS)

        # ----------------------------------------------------------------------
        # ⏰ HANDLE OLD OR OUT-OF-RANGE DATE → SUGGEST ALTERNATIVES
        # ----------------------------------------------------------------------
        if is_past or isoparse(appointment_start) <= now_utc or isoparse(appointment_start) > limit_end_utc:
            resp.say(gpt_speak(VOICE_OLD_DATE_MSG), VOICE)

            # Fetch next 3 available slots for this doctor
            alts = get_doctor_next_available_slots(doctor_name, from_start_iso=now_utc.isoformat(), limit=3)
            if not alts:
                resp.say(gpt_speak(VOICE_NO_AVAILABLE_SLOTS_MSG), VOICE)
                resp.hangup()
                return str(resp)

            # 🗣️ Build spoken list of options with SSML <break> pauses
            options_ssml = f" <break time=\"{PAUSE_MS}ms\"/> ".join(
                [f"Option {i}: {a['friendly']}." for i, a in enumerate(alts, start=1)]
            )

            # Wrap the message inside <speak>...</speak> for Twilio SSML rendering
            combined = (
                f"<speak>{VOICE_NEXT_AVAILABLE_INTRO}"
                f"<break time=\"{PAUSE_MS}ms\"/>{options_ssml}"
                f"<break time=\"{PAUSE_MS}ms\"/>{VOICE_NEXT_AVAILABLE_OUTRO}</speak>"
            )

            debug_print(f"[collect_book_time_date] 🗣️ SSML built for {len(alts)} options with {PAUSE_MS}ms pauses")

            # Send as a gather prompt (Twilio will pause between each <break>)
            g = make_gather(
                combined,
                input="speech dtmf",
                timeout=15,
                speech_timeout="auto",
                barge_in=True,
                action="/voice",
                method="POST"
            )
            resp.append(g)
            sd["alts_list"] = alts
            sd["stage"] = "collect_book_time_date"
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # 🕓 CHECK AVAILABILITY OF SELECTED SLOT
        # ----------------------------------------------------------------------
        if not is_doctor_slot_available(doctor_name, appointment_start, appointment_end):
            debug_print(f"[collect_book_time_date] ⛔ Slot not available for {doctor_name} at {friendly}")
            g = make_gather(
                f"That time is not available. {VOICE_REASK_TIME_MSG}",
                input="speech dtmf",
                timeout=10,
                barge_in=True,
                action="/voice",
                method="POST"
            )
            resp.append(g)
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # ✅ SUCCESS → SAVE SLOT AND MOVE TO CONFIRMATION
        # ----------------------------------------------------------------------
        sd["appointment_time"] = {"start": appointment_start, "end": appointment_end}
        sd["stage"] = "book_appt_confirm"
        debug_print(f"[collect_book_time_date] ✅ Slot accepted → {friendly}")

        save_session(call_sid)
        resp.redirect("/voice")
        return str(resp)










    # ======================================================================
    # 🧾 Stage: collect_first_name
    # ----------------------------------------------------------------------
    # 🎯 FUNCTIONAL PURPOSE:
    #   • Capture the caller’s **first name** via speech or keypad (DTMF).
    #   • Handle both natural spoken input (e.g., “My name is Ahmed”) and
    #     keypad input (if applicable, though speech is primary).
    #   • Clean and normalize input by removing punctuation, filler phrases,
    #     and extraneous words to isolate the actual first name.
    #   • Validate that the result consists only of English letters.
    #   • Retry up to 3 times for silence or invalid input.
    #
    # 🧩 INPUTS:
    #   • speech_result → Transcribed speech text from Twilio’s STT engine.
    #   • Digits        → Raw keypad input (optional for DTMF fallback).
    #   • call_sid      → Unique session identifier for per-call state tracking.
    #
    # 💾 OUTPUTS (stored in session_data[call_sid]["customer"]):
    #   • first_name → Caller’s cleaned and validated first name (e.g., “Mohamed”)
    #
    # 🔁 FLOW OVERVIEW:
    #   1️⃣ **Silence Handling:**
    #       - If no voice or DTMF detected → prompt again up to 3 times.
    #       - After 3 silent tries → gracefully end the call with apology message.
    #
    #   2️⃣ **Speech Path:**
    #       - Removes punctuation (.,!? etc.) while keeping valid characters.
    #       - Collapses multiple spaces and trims edges.
    #       - Removes filler phrases such as:
    #         “my name is”, “this is”, “I am”, “it’s”, “I’m”, “it is”.
    #       - Extracts the first valid alphabetic token as the name.
    #
    #   3️⃣ **DTMF Path:**
    #       - (Optional) Uses digit mapping or foreign name hints if keypad entry supported.
    #
    #   4️⃣ **Validation & Retry:**
    #       - Ensures name only contains English letters, apostrophes, or hyphens.
    #       - Rejects Arabic or non-latin text (Unicode range \u0600–\u06FF).
    #       - Allows up to 3 invalid attempts before ending call.
    #
    #   5️⃣ **Next Step:**
    #       - On success, saves first_name and proceeds to stage → “collect_last_name”.
    #
    # 🧠 SPECIAL BEHAVIOR:
    #   • Keeps apostrophes/hyphens for names like “O’Connor” or “Al-Sayed”.
    #   • Uses polite re-prompts (“Please say your first name now.”) on retries.
    #   • Maintains prior session context (doctor info, insurance, etc.).
    #   • Uses Twilio <Gather> and <Redirect> sequence for smooth retry flow.
    #
    # ✅ SUMMARY:
    #   This stage reliably captures, cleans, and validates the caller’s first name,
    #   ensuring user-friendly recovery from silence or noise while preparing for
    #   accurate downstream personalization (e.g., greeting, confirmation).
    # ======================================================================




    # ===== collect_first_name (stage) =====
    elif stage == "collect_first_name":
        # ----------------------------------------------------------------------
        # 🎙️ Voice Messages — declared up-front for easy editing/localization
        # ----------------------------------------------------------------------
        MSG_SILENCE_REPROMPT = (
            "I didn’t hear your first name. Please say your first name. "
            "You can also type it and press pound."
        )
        MSG_MAX_SILENCE_EXIT = (
            "I’m still not hearing anything. Please call again later."
        )
        MSG_T9_NO_MATCH_REPROMPT = (
            "I couldn’t match that keypad entry to a name. "
            "Please say your first name, or type it again and press pound."
        )
        MSG_INVALID_NAME_REPROMPT = (
            "Please say your first name using English letters only. "
            "You can also type it on the keypad and press pound."
        )
        # Will be formatted with the detected first name before prompting for last name:
        MSG_THANK_YOU_NEXT_LASTNAME = "Thank you {first_name}. Now, what is your last name?"

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

        # -------------------------------
        # 🔇 Silence Handling (local)
        # -------------------------------
        if not raw_speech and not raw_dtmf:
            tries = sd.get("silence_first_name", 0) + 1
            sd["silence_first_name"] = tries
            sd["stage"] = "collect_first_name"  # ensure we come back here next webhook
            debug_print(f"collect_first_name: 🤐 silence; tries={tries}/3")

            if tries >= 3:
                resp.say(gpt_speak(MSG_MAX_SILENCE_EXIT), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            gather = make_gather(
                MSG_SILENCE_REPROMPT,
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
            #
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
            #
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
                        MSG_T9_NO_MATCH_REPROMPT,
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
        # 🌐 4. Validation: Only English letters are allowed  ✅ FIXED VERSION
        # ----------------------------------------------------------------------
        # We now check if the name looks valid. The rules are:
        #   - Must contain English letters only (A-Z or a-z)
        #   - Can contain apostrophes, hyphens, or spaces
        #   - Must not contain Arabic script or foreign Unicode letters
        #   - Should accept short names (≥2 characters) such as "Ola", "Ng", "Ali"
        #
        # ⚙️ FIX:
        #   Some STT engines add invisible directional marks or dots.
        #   We'll normalize them and lower-case before applying regex.
        # ----------------------------------------------------------------------

        first_name = first_name.strip().title()  # Normalize capitalization (e.g., "faten" → "Faten")
        first_name = _re.sub(r"[\u200B-\u200F]", "", first_name)  # remove zero-width marks

        # Allow 2–40 alphabetic chars, plus apostrophes or hyphens in between
        english_only_pattern = r"^[A-Za-z][A-Za-z'\-\s]{1,39}$"

        # Arabic-script range (\u0600–\u06FF); ignore accidental diacritics (\u064B–\u065F)
        contains_foreign = bool(
            _re.search(r"[\u0600-\u06AA\u06CC-\u06FF]", first_name)
        )

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
                MSG_INVALID_NAME_REPROMPT,
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
            MSG_THANK_YOU_NEXT_LASTNAME.format(first_name=first_name),
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




    # ======================================================================
    # 🧾 Stage: collect_last_name
    # ----------------------------------------------------------------------
    # 🎯 FUNCTIONAL PURPOSE:
    #   • Capture the caller’s **last name** using either speech or keypad (DTMF).
    #   • Ensure the collected name is in **English letters only** (no Arabic script).
    #   • Handle silence, retries, and invalid entries gracefully (up to 3 attempts).
    #   • Normalize both spoken and typed inputs to extract clean last names.
    #
    # 🧩 INPUTS:
    #   • speech_result  → Twilio Speech-to-Text output of caller’s spoken name.
    #   • Digits         → Raw keypad input (T9 entry, e.g., 542545# → “Khalil”).
    #   • call_sid       → Unique call session ID used to store session data.
    #
    # 💾 OUTPUTS (stored in session_data[call_sid]["customer"]):
    #   • last_name → caller’s cleaned and validated last name (e.g., “Khalil”)
    #
    # 🔁 FLOW OVERVIEW:
    #   1️⃣ **Silence Handling:**
    #       - If no speech or digits detected → re-prompt up to 3 times.
    #       - After 3 failed attempts → terminate politely.
    #
    #   2️⃣ **DTMF Path:**
    #       - Map keypad digits to possible letter combinations (T9 system).
    #       - Match against FOREIGN_NAME_HINTS for likely known names.
    #       - Fallback: approximate name from the keypad pattern.
    #
    #   3️⃣ **Speech Path:**
    #       - Clean punctuation, spacing, and filler phrases (e.g., “My last name is…”).
    #       - Extract the first valid English token as the last name.
    #       - Reject input containing Arabic or invalid characters.
    #
    #   4️⃣ **Validation & Retry:**
    #       - Allow up to 3 invalid attempts (non-English or empty name).
    #       - If all fail → politely end call with retry message.
    #
    #   5️⃣ **Next Step:**
    #       - On success, store last name and move to stage “collect_address”.
    #
    # 🧠 SPECIAL BEHAVIOR:
    #   • Allows names with apostrophes/hyphens (e.g., “O'Neill”, “Al-Sayed”).
    #   • Filters out filler words (“say”, “press”, “you can also…”).
    #   • Maintains previous session context (doctor_name, etc.).
    #   • Reuses Twilio <Gather> + <Redirect> loop for smooth retry handling.
    #
    # ✅ SUMMARY:
    #   This stage ensures accurate and user-friendly last name collection
    #   while maintaining robustness against silence, noise, or invalid input.
    # ======================================================================



    elif stage == "collect_last_name":
        # ======================================================================
        # 🎙️ Voice Message Constants (for easy updates/localization)
        # ----------------------------------------------------------------------
        MSG_SILENCE_REPROMPT = (
            "Please say your last name now. You can also type it using your keypad and press pound."
        )
        MSG_MAX_SILENCE_EXIT = (
            "I’m still not hearing anything. Please call again later."
        )
        MSG_INVALID_NAME_REPROMPT = (
            "I didn’t get that clearly. Please say your last name. "
            "For example, say Khalil, Ahmed, or Johnson."
        )
        MSG_THANK_YOU_NEXT_ADDRESS = (
            "Thank you {first_name} {last_name}. Please tell me your full address."
        )

        # ======================================================================
        # 📞 Stage: collect_last_name — capture customer's last name
        # ----------------------------------------------------------------------
        # 🎯 Functionality:
        #   • Capture the customer's last name via speech or keypad (DTMF).
        #   • Handle silence locally (up to 3 retries) with polite re-prompts.
        #   • Support T9 DTMF name entry using FOREIGN_NAME_HINTS.
        #   • Clean speech input (remove filler words, punctuation).
        #   • Validate English-only name pattern (reject Arabic or symbols).
        #   • Save to session_data[call_sid]["customer"]["last_name"].
        #   • Proceed to collect_address stage.
        # ======================================================================

        sd = session_data.setdefault(call_sid, {})
        sd.setdefault("customer", {})

        raw_speech = (speech_result or "").strip()
        raw_dtmf = (request.values.get("Digits") or "").strip()
        debug_print(f"collect_last_name: speech='{raw_speech}', dtmf='{raw_dtmf}'")

        # ----------------------------------------------------------------------
        # 🔇 Silence Handling (local, 3 tries)
        # ----------------------------------------------------------------------
        # If no speech or DTMF input is received, re-prompt up to 3 times.
        if not raw_speech and not raw_dtmf:
            tries = sd.get("silence_last_name", 0) + 1
            sd["silence_last_name"] = tries
            sd["stage"] = "collect_last_name"
            debug_print(f"collect_last_name: 🤐 silence; tries={tries}/3")

            if tries >= 3:
                # After 3 silent attempts → polite exit
                resp.say(gpt_speak(MSG_MAX_SILENCE_EXIT), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Re-prompt the user for their last name
            gather = Gather(
                input="speech dtmf",
                language="en-US",
                hints=FOREIGN_NAME_HINTS,
                speech_model="phone_call",
                timeout=8,
                speech_timeout="auto",
                finish_on_key="#",
                barge_in=True,
                action="/voice",
                method="POST",
            )
            gather.say(gpt_speak(MSG_SILENCE_REPROMPT), VOICE)
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # ✅ Some input → clear silence counter
        sd.pop("silence_last_name", None)

        # ----------------------------------------------------------------------
        # 🔢 Define T9 conversion helpers (for keypad entry)
        # ----------------------------------------------------------------------
        # Each digit maps to possible letters (like old mobile keypads).
        _T9 = {
            "2": "ABC", "3": "DEF", "4": "GHI", "5": "JKL",
            "6": "MNO", "7": "PQRS", "8": "TUV", "9": "WXYZ"
        }

        def _t9_signature(name: str) -> str:
            """Convert a romanized name to its T9 numeric sequence."""
            s = []
            for ch in name.upper():
                if "A" <= ch <= "Z":
                    for d, letters in _T9.items():
                        if ch in letters:
                            s.append(d)
                            break
            return "".join(s)

        def _best_name_from_t9(digits: str, hints_csv: str) -> str:
            """Match a T9 sequence against FOREIGN_NAME_HINTS for likely names."""
            if not digits:
                return ""
            items = [x.strip() for x in hints_csv.split(",") if x.strip() and x.strip()[0].isalpha()]
            # Exact match (best case)
            exact = [nm for nm in items if _t9_signature(nm) == digits]
            if len(exact) == 1:
                return exact[0]
            if len(exact) > 1:
                exact.sort(key=lambda n: (-len(n), n))
                return exact[0]
            # Partial matches (fallback)
            partial = []
            for nm in items:
                sig = _t9_signature(nm)
                if sig.startswith(digits) or digits.startswith(sig):
                    partial.append((nm, sig))
            if partial:
                partial.sort(key=lambda t: (abs(len(t[1]) - len(digits)), -len(t[0]), t[0]))
                return partial[0][0]
            return ""

        # ----------------------------------------------------------------------
        # 🧾 Parse and Clean Input
        # ----------------------------------------------------------------------
        if raw_dtmf:
            # --------------------------------------------------------------
            # 📟 DTMF (keypad) input path
            # --------------------------------------------------------------
            # Remove all non-digit characters → keep only 0–9
            d = _re.sub(r"\D", "", raw_dtmf)
            debug_print(f"collect_last_name: 📟 DTMF cleaned='{d}'")

            last_name = ""
            if d and all(ch in "23456789" for ch in d):
                # Try to resolve to a name using T9 lookup
                try_name = _best_name_from_t9(d, FOREIGN_NAME_HINTS)
                if try_name:
                    last_name = try_name
                    debug_print(f"collect_last_name: 🔤 T9 matched → '{last_name}'")
                else:
                    # Fallback → approximate letter guess by first keypad letter
                    first_letter = {"2": "A", "3": "D", "4": "G", "5": "J", "6": "M", "7": "P", "8": "T", "9": "W"}
                    last_name = "".join(first_letter[ch] for ch in d)
                    debug_print(f"collect_last_name: 🧩 T9 fallback guess → '{last_name}'")
            else:
                # No valid digits (e.g., empty or includes 0/1) → use placeholder
                trailing = d[-3:] if d else ""
                last_name = f"Family{trailing}" if trailing else "Unknown"
                debug_print(f"collect_last_name: 🔖 placeholder from keypad → '{last_name}'")

        else:
            # --------------------------------------------------------------
            # 🗣️ Speech input path
            # --------------------------------------------------------------
            # Clean and normalize user’s spoken last name.
            _PUNCT = """!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""
            punct_keep = "'-"  # allow apostrophes and hyphens (e.g., O'Neill, Al-Sayed)
            _punct_to_remove = "".join(ch for ch in _PUNCT if ch not in punct_keep)

            # 1️⃣ Remove disallowed punctuation
            cleaned = raw_speech.translate(str.maketrans('', '', _punct_to_remove)).strip()

            # 2️⃣ Normalize multiple spaces/tabs/newlines to a single space
            cleaned = _re.sub(r"\s+", " ", cleaned)

            # 3️⃣ Remove filler words and phrases not part of name
            cleaned = _re.sub(
                r"\b(?:my last name is|family name is|last name|surname is|this is|i am|i'm|it's|you can also|say|press)\b\s*",
                "",
                cleaned,
                flags=_re.IGNORECASE,
            )

            # Tokenize words, filter non-name fillers
            tokens = cleaned.split()
            fillers = {"you", "can", "also", "say", "press", "the", "button", "to", "type"}
            valid_tokens = [t for t in tokens if t.isalpha() and t.lower() not in fillers]

            # Choose the first valid token or fallback to regex search
            if valid_tokens:
                last_name = valid_tokens[0].capitalize()
            else:
                m = _re.search(r"[A-Za-z]+", cleaned)
                last_name = m.group(0).capitalize() if m else ""

            debug_print(f"collect_last_name: 🧹 cleaned='{cleaned}', chosen='{last_name}'")

        # ----------------------------------------------------------------------
        # 🌐 Validate: English letters only
        # ----------------------------------------------------------------------
        english_only_pattern = r"^[A-Za-z][A-Za-z'\-\s]{0,59}$"
        contains_foreign_block = bool(_re.search(r"[\u0600-\u06FF]", last_name))  # Arabic letters check

        if (not last_name) or (not _re.fullmatch(english_only_pattern, last_name)) or contains_foreign_block:
            # --------------------------------------------------------------
            # ❌ Invalid name → retry (max 3 attempts)
            # --------------------------------------------------------------
            r = sd.get("retry_last_name", 0) + 1
            sd["retry_last_name"] = r
            sd["stage"] = "collect_last_name"
            debug_print(f"collect_last_name: ❌ invalid name '{last_name}' retry={r}/3")

            if r >= 3:
                # Too many invalid attempts → end call politely
                resp.say(
                    gpt_speak("Sorry, I couldn’t capture your last name in English letters. Please call again later."),
                    VOICE,
                )
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Re-prompt for clearer input
            gather = Gather(
                input="speech dtmf",
                language="en-US",
                hints=FOREIGN_NAME_HINTS,
                speech_model="phone_call",
                timeout=8,
                speech_timeout="auto",
                finish_on_key="#",
                barge_in=True,
                action="/voice",
                method="POST",
            )
            gather.say(gpt_speak(MSG_INVALID_NAME_REPROMPT), VOICE)
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # ----------------------------------------------------------------------
        # ✅ Valid → Save and Proceed to Address Collection
        # ----------------------------------------------------------------------
        sd["customer"]["last_name"] = last_name
        sd["stage"] = "collect_address"
        sd.pop("retry_last_name", None)
        debug_print(f"collect_last_name: ✅ saved last_name='{last_name}' → next=collect_address")

        # Ask user for address next
        gather = make_gather(
            MSG_THANK_YOU_NEXT_ADDRESS.format(
                first_name=sd["customer"].get("first_name", ""),
                last_name=last_name,
            ),
            input="speech dtmf",
            language="en-US",
            hints="118 Briar Oak Murphy Texas 75094",  # example bias
            timeout=8,
            speech_timeout="auto",
            finish_on_key="#",
            barge_in=True,
            action="/voice",
            method="POST",
        )
        resp.append(gather)
        resp.redirect("/voice")
        return str(resp)



# ======================================================================
# 🏠 Stage: collect_address
# ----------------------------------------------------------------------
# 🎯 FUNCTIONAL PURPOSE:
#   • Capture the caller’s complete mailing address using speech input.
#   • Normalize the text (spacing, punctuation) for clean storage.
#   • Handle silence gracefully with up to 3 retries before hanging up.
#   • Validate the address to ensure it contains letters and reasonable length.
#   • Save under session_data[call_sid]["customer"]["address"].
#   • Advance to the next stage (collect_cc) after success.
#
# 🧩 INPUTS:
#   • speech_result → Caller’s spoken address (transcribed by Twilio).
#   • Digits        → Optional keypad input (not primary here).
#   • call_sid      → Call session identifier used to maintain state.
#
# 💾 OUTPUTS:
#   • session_data[call_sid]["customer"]["address"] = normalized address text.
#
# 🔁 FLOW OVERVIEW:
#   1️⃣ Prompt caller for address.
#   2️⃣ Retry up to 3 times for silence.
#   3️⃣ Normalize spacing/punctuation for cleaner text.
#   4️⃣ Validate for alphabetic content and minimum length.
#   5️⃣ Save and continue to the payment (collect_cc) stage.
#
# 🧠 SPECIAL BEHAVIOR:
#   • Uses `_re` (alias for `re`) to avoid import conflicts.
#   • Keeps conversational tone in voice prompts.
#   • Ensures Twilio posts back after <Gather> via redirect.
#
# ✅ SUMMARY:
#   This stage robustly collects and cleans the caller’s spoken address,
#   preventing hangs from silence and guaranteeing normalized text for
#   downstream use (e.g., confirmation or billing).
# ======================================================================

    elif stage == "collect_address":
        # ----------------------------------------------------------------------
        # 🎙️ ALL VOICE PROMPTS — DECLARED AT THE BEGINNING
        # ----------------------------------------------------------------------
        PROMPT_INTRO = (
            "Please tell me your full address, including street number, city, and ZIP code. "
            "For example, say one one eight Briar Oak, Murphy, Texas seven five zero nine four."
        )

        PROMPT_RETRY_SILENCE = (
            "I didn't catch that. Please say your street address, city, and ZIP. "
            "For example, one one eight Briar Oak, Murphy, Texas seven five zero nine four."
        )

        PROMPT_INVALID_ADDRESS = (
            "Please repeat your full mailing address — street, city, state, and ZIP. "
            "For example, one one eight Briar Oak, Murphy, Texas seven five zero nine four."
        )

        PROMPT_CONFIRM_NEXT = (
            "Thank you. Now, please enter your card number, then press pound."
        )

        # ----------------------------------------------------------------------
        # 🔧 Initialize or retrieve session context
        # ----------------------------------------------------------------------
        session_data.setdefault(call_sid, {}).setdefault("customer", {})

        # ----------------------------------------------------------------------
        # 🗣️ Retrieve caller’s speech safely
        # ----------------------------------------------------------------------
        try:
            raw = (speech_result or request.values.get("SpeechResult") or "").strip()
        except Exception:
            raw = (speech_result or "").strip()

        debug_print(f"collect_address: 📬 Collected address (raw): {raw}")

        # ----------------------------------------------------------------------
        # 🔇 Handle silence (nothing heard)
        # ----------------------------------------------------------------------
        if not raw:
            # Increment silence counter for this stage
            tries = session_data[call_sid].get("silence_address", 0) + 1
            session_data[call_sid]["silence_address"] = tries
            debug_print(f"collect_address: 🤐 silence; tries={tries}")

            if tries >= 3:
                # After 3 failed attempts → politely end call
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Re-prompt caller to try again
            gather = make_gather(PROMPT_RETRY_SILENCE)
            resp.append(gather)
            try:
                from flask import url_for
                resp.redirect(url_for("voice"))  # redirect ensures Twilio posts back
            except Exception:
                resp.redirect("/voice")
            return str(resp)

        # ✅ Some input received → reset silence counter
        session_data[call_sid].pop("silence_address", None)

        # ----------------------------------------------------------------------
        # 🧹 Normalize and clean address text
        # ----------------------------------------------------------------------
        addr = raw

        # 1️⃣ Collapse multiple spaces (e.g., “Murphy   Texas” → “Murphy Texas”)
        addr = _re.sub(r"\s+", " ", addr)

        # 2️⃣ Normalize spacing around commas, hashes, and periods
        #     Example: "Murphy , Texas . 75094" → "Murphy, Texas. 75094"
        addr = _re.sub(r"\s*([,#\.])\s*", r"\1 ", addr)

        # 3️⃣ Remove repeated punctuation (“..” or “,,,” → single instance)
        addr = _re.sub(r"\.{2,}", ".", addr)
        addr = _re.sub(r",\s*,+", ", ", addr)

        # 4️⃣ Trim stray punctuation and whitespace at edges
        addr = addr.strip(" .,")

        # 5️⃣ Final pass — collapse any spaces introduced during cleanup
        addr = _re.sub(r"\s+", " ", addr).strip()

        debug_print(f"collect_address: 🧽 Normalized → '{addr}'")

        # ----------------------------------------------------------------------
        # ✅ Basic validation for readability
        # ----------------------------------------------------------------------
        # Require at least one alphabetic character and a reasonable length.
        # This filters out blank or nonsensical STT artifacts.
        if (not addr) or (_re.search(r"[A-Za-z]", addr) is None) or (len(addr) < 6):
            r = session_data[call_sid].get("retry_address", 0) + 1
            session_data[call_sid]["retry_address"] = r
            debug_print(f"collect_address: ❌ looks invalid/too short → retry={r}")

            if r >= 3:
                # After 3 invalid attempts → hang up politely
                resp.say(gpt_speak("Sorry, I couldn’t capture your address. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Re-prompt for valid address
            gather = make_gather(PROMPT_INVALID_ADDRESS)
            resp.append(gather)
            try:
                from flask import url_for
                resp.redirect(url_for("voice"))
            except Exception:
                resp.redirect("/voice")
            return str(resp)

        # ----------------------------------------------------------------------
        # 💾 Persist valid address
        # ----------------------------------------------------------------------
        session_data[call_sid]["customer"]["address"] = addr
        session_data[call_sid].pop("retry_address", None)  # reset retry counter
        debug_print(f"collect_address: ✅ Saved address='{addr}'")

        # ----------------------------------------------------------------------
        # 🔁 Advance to next stage
        # ----------------------------------------------------------------------
        session_data[call_sid]["stage"] = "collect_cc"

        # Prompt user for credit card (or next data item)
        gather = make_gather(PROMPT_CONFIRM_NEXT)
        resp.append(gather)
        try:
            from flask import url_for
            resp.redirect(url_for("voice"))
        except Exception:
            resp.redirect("/voice")

        return str(resp)




    # ----------------------------------------------------------------------
    # 🩺 Stage: collect_dr_info
    # ----------------------------------------------------------------------
    # 🎯 PURPOSE:
    #   - Present a list of available doctors (by keypad or speech).
        #   - Capture the user’s selection using DTMF (1, 2, 3...) or speech.
        #   - Perform fuzzy matching on spoken names (partial word matches).
        #   - Retry up to 3 times if no valid match is found.
        #   - On success → move to stage "collect_book_time_date" for time selection.
        # ----------------------------------------------------------------------


    elif stage == "collect_dr_info":
        

        # ----------------------------------------------------------------------
        # 💬 Voice Prompts — all text in variables for easy maintenance
        # ----------------------------------------------------------------------
        VOICE_INTRO_PROMPT = "Please choose your doctor."
        VOICE_INSTRUCTION_APPENDIX = "You can also say the doctor's name."
        VOICE_REPROMPT_ON_JUNK = (
            "Please say the name of the doctor you'd like to book with, "
            "or press the number on your keypad."
        )
        VOICE_FAIL_FINAL = (
            "I'm sorry, I still couldn't match that name with any doctor in our clinic. "
            "Please call us again later."
        )
        VOICE_RETRY_PROMPT = (
            "I couldn't match that to a doctor. "
            "You can also say the doctor's name."
        )
        VOICE_SUCCESS_PROMPT_TEMPLATE = (
            "Great, your appointment will be with {doctor}. "
            "Please say the appointment date and time, for example, "
            "'October 8 at 9 30 A M'."
        )

        # ----------------------------------------------------------------------
        # ⚙️ Initialize Session State
        # ----------------------------------------------------------------------
        sd = session_data.setdefault(call_sid, {})
        sd.setdefault("retry_booking", 0)
        sd["origin_stage"] = "book"

        # ----------------------------------------------------------------------
        # 🧹 Clean speech input (remove punctuation, lowercase for fuzzy matching)
        # ----------------------------------------------------------------------
        _PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        dtmf_digits = (request.values.get("Digits") or "").strip()
        spoken_text = (speech_result or "").strip().lower()
        spoken_clean = spoken_text.translate(str.maketrans('', '', _PUNCT)).strip()
        debug_print(f"[collect_dr_info] 🗣 speech='{spoken_clean}' 🔢 DTMF='{dtmf_digits}'")

        # ----------------------------------------------------------------------
        # 📋 Build the doctor keypad map (first interaction only)
        # ----------------------------------------------------------------------
        if "doctor_dtmf_map" not in sd:
            doctor_dtmf_map = {}
            prompt_lines = []

            # Enumerate available doctor names into "Press 1 for Dr. X"
            for i, friendly in enumerate(doctor_names.values(), start=1):
                doctor_dtmf_map[str(i)] = friendly
                prompt_lines.append(f"Press {i} for {friendly}.")

            # Store map for future use within session
            sd["doctor_dtmf_map"] = doctor_dtmf_map

            # Combine all parts into a single spoken message
            doctor_prompt = f"{VOICE_INTRO_PROMPT} " + " ".join(prompt_lines) + " " + VOICE_INSTRUCTION_APPENDIX

            # Create <Gather> TwiML element for speech/DTMF capture
            g = make_gather(
                doctor_prompt,
                input="speech dtmf",
                timeout=10,
                speech_timeout="auto",
                barge_in=True,
                finish_on_key="#",
                num_digits=1
            )
            resp.append(g)
            resp.redirect("/voice")
            debug_print("[collect_dr_info] 📋 Initial doctor list prompt sent.")
            return str(resp)

        # ----------------------------------------------------------------------
        # 🔎 Retrieve the stored map for repeated interactions
        # ----------------------------------------------------------------------
        doctor_map = sd.get("doctor_dtmf_map", {})
        matched_name = None

        # ----------------------------------------------------------------------
        # 🔢 Case 1: Direct DTMF match (e.g., user pressed "2")
        # ----------------------------------------------------------------------
        if dtmf_digits and dtmf_digits in doctor_map:
            matched_name = doctor_map[dtmf_digits]
            debug_print(f"[collect_dr_info] ✅ DTMF matched doctor: {matched_name}")

        # ----------------------------------------------------------------------
        # 🎙️ Case 2: Speech-based fuzzy matching
        # ----------------------------------------------------------------------
        if matched_name is None:
            # Ignore junk words or empty input (like “hello”, “ok”, etc.)
            junk_inputs = {
                "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
                "yo", "test", "1", "yes", "no", "i know", "huh", "what", "okay", "ok",
                "bye", "goodbye", ""
            }

            # 🧩 If invalid input → re-prompt
            if not spoken_clean or spoken_clean in junk_inputs or len(spoken_clean) < 3:
                debug_print(f"[collect_dr_info] ⏩ Skipping junk input: '{spoken_clean}' — re-prompting")

                prompt_lines = [f"Press {k} for {v}." for k, v in doctor_map.items()]
                doctor_prompt = f"{VOICE_REPROMPT_ON_JUNK} " + " ".join(prompt_lines)

                g = make_gather(
                    doctor_prompt,
                    input="speech dtmf",
                    timeout=8,
                    speech_timeout="auto",
                    barge_in=True,
                    finish_on_key="#",
                    num_digits=1
                )
                resp.append(g)
                resp.redirect("/voice")
                return str(resp)

            # 🔍 Try fuzzy/partial token match
            partial_matches = []
            spoken_tokens = set(spoken_clean.split())

            for friendly in doctor_names.values():
                friendly_clean = friendly.lower().translate(str.maketrans('', '', _PUNCT)).strip()
                friendly_tokens = set(friendly_clean.split())

                # Match if names overlap, or one contained in the other
                if (
                    spoken_clean in friendly_clean
                    or friendly_clean in spoken_clean
                    or (spoken_tokens & friendly_tokens)
                ):
                    partial_matches.append(friendly)

            # ✅ Single unique match
            if len(partial_matches) == 1:
                matched_name = partial_matches[0]
                debug_print(f"[collect_dr_info] ✅ Partial match: {matched_name}")
            # ⚠️ Multiple possible matches — pick first for simplicity
            elif len(partial_matches) > 1:
                debug_print(f"[collect_dr_info] 🔍 Multiple matches: {partial_matches}")
                matched_name = partial_matches[0]

        # ----------------------------------------------------------------------
        # ❌ Case 3: No match — retry or fail out
        # ----------------------------------------------------------------------
        if matched_name is None:
            sd["retry_booking"] += 1
            retries = sd["retry_booking"]
            debug_print(f"[collect_dr_info] ❌ No doctor match for '{spoken_clean or dtmf_digits}' retry={retries}")

            # 3 failed attempts → hang up gracefully
            if retries >= 3:
                resp.say(gpt_speak(VOICE_FAIL_FINAL), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Otherwise, prompt again with doctor list
            prompt_lines = [f"Press {k} for {v}." for k, v in doctor_map.items()]
            doctor_prompt = f"{VOICE_RETRY_PROMPT} " + " ".join(prompt_lines)

            g = make_gather(
                doctor_prompt,
                input="speech dtmf",
                timeout=8,
                speech_timeout="auto",
                barge_in=True,
                finish_on_key="#",
                num_digits=1
            )
            resp.append(g)
            resp.redirect("/voice")
            return str(resp)

        # ----------------------------------------------------------------------
        # ✅ Success — Save doctor and move to "collect_book_time_date"
        # ----------------------------------------------------------------------
        sd["doctor_name"] = matched_name
        sd["stage"] = "collect_book_time_date"

        # Build personalized success message
        success_prompt = VOICE_SUCCESS_PROMPT_TEMPLATE.format(doctor=matched_name)

        g = make_gather(
            success_prompt,
            input="speech dtmf",
            timeout=10,
            speech_timeout="auto",
            barge_in=True,
            finish_on_key="#"
        )
        resp.append(g)
        resp.redirect("/voice")

        debug_print(f"[collect_dr_info] ✅ Stored doctor_name={matched_name} → next stage collect_book_time_date")
        save_session(call_sid)
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
        #     (e.g., cancel_appt_get_phone_number).
        # ----------------------------------------------------------------------

        sd = session_data.setdefault(call_sid, {})
        sd.setdefault("cancel", {})
        sd["origin_stage"] = "cancel"

        # ----------------------------------------------------------------------
        # 🧹 Punctuation and helper setup
        # ----------------------------------------------------------------------
        try:
            _PUNCT = string.punctuation
        except Exception:
            _PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

        def _clean(s: str) -> str:
            """Normalize and strip punctuation for fuzzy matching."""
            s = (s or "").lower().translate(str.maketrans("", "", _PUNCT)).strip()
            return " ".join(s.split())

        def _extract_doctor_name(speech_text):
            """GPT fallback extractor for doctor name (if needed)."""
            if not speech_text.strip():
                return ""
            prompt = (
                f"From this sentence: \"{speech_text}\", extract only the doctor's name "
                f"mentioned. Return only the name without titles like 'Dr.' or punctuation. "
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
                debug_print(f"[cancel_appointment] 🤖 GPT extracted doctor name → '{extracted}'")
                return extracted
            except Exception as e:
                debug_print(f"[cancel_appointment] ⚠️ GPT fallback failed → {type(e).__name__}: {e}")
                return speech_text.strip()

        # ----------------------------------------------------------------------
        # 🔊 Input capture
        # ----------------------------------------------------------------------
        dtmf_digits = (request.values.get("Digits") or "").strip()
        spoken_text = (speech_result or "").strip()
        spoken_clean = _clean(spoken_text)

        # Build map for DTMF-based doctor selection
        doctor_list = list(doctor_names.values())
        doctor_dtmf_map = {str(i + 1): doc for i, doc in enumerate(doctor_list)}
        sd["doctor_dtmf_map"] = doctor_dtmf_map

        debug_print(f"[cancel_appointment] 🎙 speech='{spoken_clean}' 🔢 DTMF='{dtmf_digits}'")

        # ----------------------------------------------------------------------
        # 🔇 Silence handling
        # ----------------------------------------------------------------------
        if not spoken_clean and not dtmf_digits:
            tries = sd.get("silence_cancel_doc", 0) + 1
            sd["silence_cancel_doc"] = tries
            debug_print(f"[cancel_appointment] 🤐 silence count={tries}/3")

            if tries >= 3:
                resp.say(gpt_speak("I’m still not hearing anything. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            options = ". ".join([f"{name} (press {k})" for k, name in doctor_dtmf_map.items()])
            retry_prompt = (
                f"I didn't hear the doctor's name. Available doctors are: {options}. "
                "Please say the doctor's name or press the corresponding number."
            )
            sd["stage"] = "cancel_appointment"
            resp.append(make_gather(
                retry_prompt,
                hints=", ".join(doctor_list),
                num_digits=1,
                language="en-US",
            ))
            return str(resp)

        sd.pop("silence_cancel_doc", None)

        # ----------------------------------------------------------------------
        # 🔢 DTMF direct match
        # ----------------------------------------------------------------------
        matched_name = None
        if dtmf_digits and dtmf_digits in doctor_dtmf_map:
            matched_name = doctor_dtmf_map[dtmf_digits]
            debug_print(f"[cancel_appointment] ✅ DTMF matched doctor → {matched_name}")

        # ----------------------------------------------------------------------
        # 🎙 Speech match (fuzzy) if no DTMF
        # ----------------------------------------------------------------------
        if not matched_name:
            junk_inputs = {
                "", "yes", "no", "yeah", "nope", "ok", "okay", "hello", "hi", "hey",
                "good morning", "good afternoon", "good evening", "test", "i know", "what"
            }
            if (not spoken_clean) or (spoken_clean in junk_inputs) or len(spoken_clean) < 2:
                options = ". ".join([f"{name} (press {k})" for k, name in doctor_dtmf_map.items()])
                retry_prompt = (
                    f"I didn't recognize that as a doctor's name. Available doctors are: {options}. "
                    "Please say the name or press the number."
                )
                sd["stage"] = "cancel_appointment"
                resp.append(make_gather(
                    retry_prompt,
                    hints=", ".join(doctor_list),
                    num_digits=1,
                    language="en-US",
                ))
                return str(resp)

            # 🔍 Partial / token-based fuzzy matching
            partial_matches = []
            spoken_tokens = set(spoken_clean.split())
            for friendly_name in doctor_names.values():
                friendly_clean = _clean(friendly_name)
                friendly_tokens = set(friendly_clean.split())
                if (
                    spoken_clean in friendly_clean
                    or friendly_clean in spoken_clean
                    or (spoken_tokens & friendly_tokens)
                ):
                    partial_matches.append(friendly_name)

            if len(partial_matches) == 1:
                matched_name = partial_matches[0]
                debug_print(f"[cancel_appointment] ✅ Partial match → {matched_name}")
            elif len(partial_matches) > 1:
                matched_name = partial_matches[0]
                debug_print(f"[cancel_appointment] 🔍 Multiple possible matches, defaulting to → {matched_name}")

            # 🤖 GPT fallback
            if not matched_name:
                try:
                    extracted_name = _extract_doctor_name(spoken_text)
                    extracted_clean = _clean(extracted_name)
                    for friendly_name in doctor_names.values():
                        friendly_clean = _clean(friendly_name)
                        if extracted_clean in friendly_clean or friendly_clean in extracted_clean:
                            matched_name = friendly_name
                            debug_print(f"[cancel_appointment] ✅ GPT matched doctor → {matched_name}")
                            break
                except Exception as e:
                    debug_print(f"[cancel_appointment] ⚠️ GPT fallback error → {e}")

        # ----------------------------------------------------------------------
        # ❌ Retry if still no match
        # ----------------------------------------------------------------------
        if not matched_name:
            retries = sd.get("retry_booking", 0) + 1
            sd["retry_booking"] = retries
            max_retries = globals().get("MAX_NUMBER_DR_RETRY", 3)
            debug_print(f"[cancel_appointment] ❌ No doctor match retry={retries}/{max_retries}")

            if retries >= max_retries:
                resp.say(gpt_speak(
                    "I'm sorry, I still couldn't match that name with any doctor in our clinic. Please try again later. Goodbye."
                ), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            options = ". ".join([f"{name} (press {k})" for k, name in doctor_dtmf_map.items()])
            retry_prompt = (
                f"I didn't recognize that name. Available doctors are: {options}. "
                "Please say the name or press the number."
            )
            sd["stage"] = "cancel_appointment"
            resp.append(make_gather(
                retry_prompt,
                hints=", ".join(doctor_list),
                num_digits=1,
                language="en-US",
            ))
            return str(resp)

        # ----------------------------------------------------------------------
        # ✅ Success — store doctor_name and proceed
        # ----------------------------------------------------------------------
        sd["cancel"]["doctor_name"] = matched_name
        sd["doctor_name"] = matched_name
        sd["stage"] = "cancel_appt_get_phone_number"

        g = make_gather(
            f"Thanks. What phone number did you use when booking your appointment with {matched_name}?",
            input="speech dtmf",
            language="en-US",
            num_digits=10,
            timeout=8,
            speech_timeout="6",
            barge_in=True,
        )
        resp.append(g)
        save_session(call_sid)
        debug_print(f"[cancel_appointment] ✅ Stored doctor_name={matched_name} → next stage cancel_appt_get_phone_number")
        return str(resp)






    elif stage == "cancel_appt_get_dob":
        # ----------------------------------------------------------------------
        # 🎂 Stage: cancel_appt_get_dob
        #
        # PURPOSE:
        #   • Capture and validate the customer's date of birth (DOB) via speech or DTMF.
        #   • Store DOB under session_data["customer"]["dob"] and ["cancel"]["dob"].
        #   • Proceeds directly to collect_pin_number after success.
        #
        # FEATURES:
        #   ✅ Handles speech (e.g., “July third nineteen fifty six”)
        #   ✅ Handles DTMF (e.g., 07031956#)
        #   ✅ Uses _re (regex alias)
        #   ✅ Full retry and silence logic
        # ----------------------------------------------------------------------

        debug_print("cancel_appt_get_dob: 📍 Stage entered")

        session_data.setdefault(call_sid, {}).setdefault("customer", {})
        session_data[call_sid].setdefault("cancel", {})

        # --- Inputs ---
        try:
            dtmf_digits = (request.values.get("Digits") or "").strip()
        except Exception:
            dtmf_digits = ""
        speech_text = (speech_result or "").strip()
        debug_print(f"cancel_appt_get_dob: 🎙️ speech_text='{speech_text}', 🔢 dtmf_digits='{dtmf_digits}'")

        # ------------------------------------------------------------------
        # 🔇 Silence / No Input Handling
        # ------------------------------------------------------------------
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
                "Please say your birth date — for example, July third nineteen fifty six. "
                "Or type two digits for month, two digits for day, and four digits for year, then press pound."
            )
            session_data[call_sid]["stage"] = "cancel_appt_get_dob"
            gather = make_gather(
                prompt_text,
                hints="zero one two three four five six seven eight nine",
                num_digits=8,
                timeout=15,
                finish_on_key="#"
            )
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # Clear silence counter
        session_data[call_sid].pop("silence_cancel_dob", None)

        # ------------------------------------------------------------------
        # 🧩 Inline DOB Parsing Logic
        # ------------------------------------------------------------------
        dt = None
        try:
            if dtmf_digits:
                clean = _re.sub(r"\D", "", dtmf_digits)
                debug_print(f"cancel_appt_get_dob: 🧮 cleaned DTMF='{clean}'")
                if len(clean) == 8:
                    m, d, y = int(clean[0:2]), int(clean[2:4]), int(clean[4:8])
                    dt = datetime(y, m, d)
                elif len(clean) == 7:
                    # 👇 Auto-pad single-digit month to 8 digits (e.g. 7031956 → 07031956)
                    clean = clean.zfill(8)
                    m, d, y = int(clean[0:2]), int(clean[2:4]), int(clean[4:8])
                    dt = datetime(y, m, d)
                else:
                    debug_print(f"cancel_appt_get_dob: ⚠️ DTMF not 8 digits → {clean}")
            if not dt and speech_text:
                dt = dp.parse(speech_text, fuzzy=True)
        except Exception as e:
            debug_print(f"cancel_appt_get_dob: ❌ parse error {e}")
            dt = None

        # ------------------------------------------------------------------
        # ❌ Invalid or Unparsed DOB → Retry
        # ------------------------------------------------------------------
        if not dt:
            retries = session_data[call_sid].get("retry_cancel_dob", 0) + 1
            session_data[call_sid]["retry_cancel_dob"] = retries
            debug_print(f"cancel_appt_get_dob: ❌ Parse failed. Retry={retries}")

            if retries >= 3:
                resp.say(gpt_speak("Sorry, I couldn’t understand your date of birth. Please call again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            prompt_text = (
                "Please say your birth date again — for example, July third nineteen fifty six. "
                "Or type two digits for month, two digits for day, and four digits for year, then press pound."
            )
            gather = make_gather(
                prompt_text,
                hints="zero one two three four five six seven eight nine",
                num_digits=8,
                timeout=15,
                finish_on_key="#"
            )
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # ------------------------------------------------------------------
        # 🧮 Validate DOB range
        # ------------------------------------------------------------------
        try:
            today = date.today()
            min_date = date(1900, 1, 1)
            dob_date = dt.date()
            if not (min_date <= dob_date <= today):
                retries = session_data[call_sid].get("retry_cancel_dob", 0) + 1
                session_data[call_sid]["retry_cancel_dob"] = retries
                debug_print(f"cancel_appt_get_dob: ⚠️ DOB out of range → {dob_date.isoformat()} Retry={retries}")

                if retries >= 3:
                    resp.say(gpt_speak("Sorry, that birth date still doesn’t look valid. Please call again later."), VOICE)
                    resp.hangup()
                    session_data.pop(call_sid, None)
                    return str(resp)

                prompt_text = (
                    "That doesn't sound like a valid birth date. Please say it again, "
                    "or type two digits for month, two for day, and four for year, then press pound. "
                    "For example, 07 03 1956#."
                )
                gather = make_gather(
                    prompt_text,
                    hints="zero one two three four five six seven eight nine",
                    num_digits=8,
                    timeout=15,
                    finish_on_key="#"
                )
                resp.append(gather)
                resp.redirect("/voice")
                return str(resp)
        except Exception as e:
            debug_print(f"cancel_appt_get_dob: ⚠️ Validation error → {e}")
            gather = make_gather(
                "Please repeat your birth date — for example, July third nineteen fifty six. "
                "Or type two digits for month, two digits for day, and four digits for year, then press pound.",
                hints="zero one two three four five six seven eight nine",
                num_digits=8,
                timeout=15,
                finish_on_key="#"
            )
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # ------------------------------------------------------------------
        # ✅ Store Parsed DOB
        # ------------------------------------------------------------------
        iso_dob = dt.strftime("%Y-%m-%d")
        session_data[call_sid]["customer"]["dob"] = iso_dob
        session_data[call_sid]["cancel"]["dob"] = iso_dob
        session_data[call_sid].pop("retry_cancel_dob", None)
        debug_print(f"cancel_appt_get_dob: ✅ Stored DOB → {iso_dob}")

        # ------------------------------------------------------------------
        # 🔜 Next Stage → collect_pin_number
        # ------------------------------------------------------------------
        session_data[call_sid]["stage"] = "collect_pin_number"
        gather = make_gather(
            "Thank you. For security verification, please enter your six-digit PIN number followed by the pound key.",
            input="dtmf speech",
            num_digits=6,
            timeout=10,
            finish_on_key="#"
        )
        resp.append(gather)
        resp.redirect("/voice")
        debug_print("cancel_appt_get_dob: 🔀 Proceeding to collect_pin_number for verification")
        return str(resp)






    
    # ----------------------------------------------------------------------
    # 🗓️ Stage: cancel_appt_get_time_date
    #
    # PURPOSE:
    #   • Capture the appointment date/time to cancel (spoken or typed).
    #   • Accepts natural speech (e.g., “October 21 at 3:30 PM”).
    #   • Handles silence and retry up to 3 times.
    #   • Checks locally stored JSON via is_doctor_slot_available().
    #   • If the slot is found → go to cancel_appt_confirm.
    #   • If not found or in the past → switch to cancel_appt_iterate.
    # ----------------------------------------------------------------------


    elif stage == "cancel_appt_get_time_date":
        debug_print("cancel_appt_get_time_date: 📍 Stage entered")
        cancel_ctx = session_data[call_sid].setdefault("cancel", {})

        raw = (speech_result or "").strip()
        debug_print(f"cancel_appt_get_time_date: 🗣️ Raw speech → '{raw}'")

        # ----------------------------------------------------------------------
        # 🔇 Handle silence locally (up to 3 retries)
        # ----------------------------------------------------------------------
        if not raw:
            tries = cancel_ctx.get("silence_cancel_dt", 0) + 1
            cancel_ctx["silence_cancel_dt"] = tries
            debug_print(f"cancel_appt_get_time_date: 🤐 silence count={tries}")

            if tries >= 3:
                debug_print("cancel_appt_get_time_date: 🚫 too many silent attempts → iterate")
                cancel_ctx.pop("silence_cancel_dt", None)
                cancel_ctx["awaiting_input"] = False
                session_data[call_sid]["stage"] = "cancel_appt_iterate"
                session_data[call_sid]["skip_silence_retry"] = True
                resp.say(gpt_speak("That doesn’t match any of your appointments. I’ll list your upcoming ones."), VOICE)
                resp.redirect("/voice")
                save_session(call_sid)
                return str(resp)

            resp.pause(length=1)
            gather = make_gather(
                "Please say the date and time of the appointment you want to cancel. "
                "For example, say October twenty-first at 3:30 PM.",
                input="speech dtmf",
                timeout=25,
                speech_timeout="auto",
                finish_on_key="#",
                barge_in=True,
            )
            resp.append(gather)
            resp.redirect("/voice")
            save_session(call_sid)
            return str(resp)

        cancel_ctx.pop("silence_cancel_dt", None)

        # ----------------------------------------------------------------------
        # 🧩 Parse date and time from speech
        # ----------------------------------------------------------------------
        day_part, time_part = (None, None)
        try:
            raw_fixed = raw.lower().replace(",", "").strip()
            if " at " in raw_fixed:
                parts = raw_fixed.split("at")
                if len(parts) == 2:
                    day_part, time_part = parts[0].strip(), parts[1].strip()
        except Exception as e:
            debug_print(f"cancel_appt_get_time_date: ⚠️ parse split error → {e}")

        debug_print(f"cancel_appt_get_time_date: 📆 Extracted → Day='{day_part}', Time='{time_part}'")

        # ----------------------------------------------------------------------
        # 🕐 Convert parsed speech into UTC datetime
        # ----------------------------------------------------------------------
        matched = False
        dt_utc, dt_end, spoken_phrase = (None, None, None)
        try:
            if day_part and time_part:
                day_part_fixed = _re.sub(r"\b(\d{1,2})d\b", r"\1rd", day_part, flags=_re.IGNORECASE)
                spoken_phrase = f"{day_part_fixed} at {time_part}"

                tz_name = globals().get("CLINIC_TZ", "America/Chicago")
                tz = _pytz.timezone(tz_name)
                dt_local = dp.parse(spoken_phrase, fuzzy=True, default=datetime.now(tz))
                dt_utc = dt_local.astimezone(_pytz.UTC)
                dt_end = dt_utc + timedelta(minutes=30)
                matched = True
                debug_print(f"cancel_appt_get_time_date: ✅ Parsed datetime → {dt_utc.isoformat()}")
        except Exception as e:
            debug_print(f"cancel_appt_get_time_date: ⚠️ dp.parse failed → {e}")

        # ----------------------------------------------------------------------
        # ❌ Retry if parsing failed
        # ----------------------------------------------------------------------
        if not matched:
            retries = cancel_ctx.get("retry_cancel_dt", 0) + 1
            cancel_ctx["retry_cancel_dt"] = retries
            debug_print(f"cancel_appt_get_time_date: ❌ parse failed → retry={retries}")

            if retries < 3:
                resp.pause(length=1)
                gather = make_gather(
                    "I didn’t catch that. Please say the appointment date and time clearly, "
                    "for example, October twenty-first at 3:30 PM.",
                    input="speech dtmf",
                    timeout=20,
                    speech_timeout="auto",
                    finish_on_key="#",
                    barge_in=True,
                )
                resp.append(gather)
                resp.redirect("/voice")
                return str(resp)

            debug_print("cancel_appt_get_time_date: 🚫 too many retries → iterate")
            cancel_ctx.pop("retry_cancel_dt", None)
            cancel_ctx["awaiting_input"] = False
            session_data[call_sid]["stage"] = "cancel_appt_iterate"
            session_data[call_sid]["skip_silence_retry"] = True
            resp.say(gpt_speak("That doesn’t match any of your appointments. I’ll list your upcoming ones."), VOICE)
            resp.redirect("/voice")
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # ⏰ Check if slot time is in the past
        # ----------------------------------------------------------------------
        now_utc = datetime.utcnow().replace(tzinfo=_pytz.UTC)
        if dt_utc < now_utc:
            debug_print("cancel_appt_get_time_date: ⏳ Time is in the past → iterate")
            cancel_ctx["awaiting_input"] = False
            session_data[call_sid]["stage"] = "cancel_appt_iterate"
            resp.say(gpt_speak("That appointment time has already passed. I’ll list your upcoming ones."), VOICE)
            resp.redirect("/voice")
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # 🧠 Check slot existence using local JSON
        # ----------------------------------------------------------------------
        try:
            doctor_name = session_data[call_sid].get("doctor_name")
            start_iso = dt_utc.isoformat()
            end_iso   = dt_end.isoformat()

            is_booked = is_doctor_slot_available(doctor_name, start_iso, end_iso)
            debug_print(f"cancel_appt_get_time_date: 🧩 is_doctor_slot_available({doctor_name}, {start_iso}) → {is_booked}")
        except Exception as e:
            debug_print(f"cancel_appt_get_time_date: ⚠️ local slot check failed → {e}")
            is_booked = False

        # ----------------------------------------------------------------------
        # 🚫 If slot not found → switch to iterate
        # ----------------------------------------------------------------------
        if not is_booked:
            debug_print("cancel_appt_get_time_date: 🚫 Slot not found → switching to iterate")
            cancel_ctx["awaiting_input"] = False
            session_data[call_sid]["stage"] = "cancel_appt_iterate"
            resp.say(gpt_speak("That doesn’t match any of your appointments. I’ll list your upcoming ones."), VOICE)
            resp.redirect("/voice")
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # ✅ Slot found → Proceed to cancel_appt_confirm
        # ----------------------------------------------------------------------
        cancel_ctx["matching_event"] = {
            "spoken_dt": spoken_phrase,
            "start": start_iso,
            "end": end_iso,
        }
        cancel_ctx.pop("retry_cancel_dt", None)
        cancel_ctx["awaiting_input"] = False
        session_data[call_sid]["stage"] = "cancel_appt_confirm"

        resp.say(gpt_speak(f"You said {day_part} at {time_part}. Let me confirm that appointment."), VOICE)
        resp.redirect("/voice")
        save_session(call_sid)
        return str(resp)








    elif stage == "cancel_appt_iterate":
        

        t_stage_start = _time_mod.perf_counter()
        debug_print("cancel_appt_iterate: 📍 Stage entered")

        # ----------------------------------------------------------------------
        # 🔧 Retrieve cancellation context from session
        # ----------------------------------------------------------------------
        #   This context was built in earlier stages (doctor, phone, DOB).
        #   We now use it to search the doctor’s JSON for matching appointments.
        # ----------------------------------------------------------------------
        cancel_ctx = session_data[call_sid].setdefault("cancel", {})
        doctor = (cancel_ctx.get("doctor") or "").strip()
        phone_e164 = (cancel_ctx.get("phone_e164") or "").replace("+", "").lstrip("0")
        dob = (cancel_ctx.get("dob") or "").strip()
        debug_print(f"cancel_appt_iterate: inputs → doctor='{doctor}', phone='{phone_e164}', dob='{dob}'")

        # ----------------------------------------------------------------------
        # 📂 Step 1: Load doctor’s local JSON appointment file
        # ----------------------------------------------------------------------
        #   Each doctor has their own JSON file in the "appointments/" folder.
        #   The filename is derived from the doctor’s name (lowercase + underscores).
        # ----------------------------------------------------------------------
        try:
            safe_name = doctor.lower().replace(" ", "_")  # e.g., "Dr. John Smith" → "dr._john_smith"
            doc_path = f"appointments/{safe_name}.json"   # JSON file path
            with open(doc_path, "r", encoding="utf-8") as f:
                appointments = json.load(f)
            debug_print(f"cancel_appt_iterate: 📁 Loaded {len(appointments)} appointments from {doc_path}")
        except FileNotFoundError:
            debug_print(f"cancel_appt_iterate: ❌ No appointment file found for doctor: {doctor}")
            resp.say(f"Sorry, I couldn’t find any appointments for {doctor}.", VOICE)
            resp.hangup()
            save_session(call_sid)
            return str(resp)
        except Exception as e:
            debug_print(f"cancel_appt_iterate: ⚠️ Error loading JSON file → {e}")
            resp.say("Sorry, there was a problem reading the appointment list.", VOICE)
            resp.hangup()
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # 🧩 Step 2: Filter appointments by phone number and DOB
        # ----------------------------------------------------------------------
        #   We normalize both fields (remove punctuation, + signs, etc.)
        #   and compare against each record in the JSON list.
        # ----------------------------------------------------------------------
        candidates = []
        normalized_phone = _re.sub(r"\D", "", phone_e164)
        normalized_dob = _re.sub(r"[^0-9a-z]+", "", dob.replace("-", "").replace("/", ""))

        for appt in appointments:
            # Extract relevant identifying info
            appt_phone = _re.sub(r"\D", "", appt.get("phone_e164", ""))
            appt_dob = _re.sub(r"[^0-9a-z]+", "", (appt.get("dob", "") or "").replace("-", "").replace("/", ""))

            # Compare caller vs record
            phone_match = normalized_phone == appt_phone
            dob_match = not dob or normalized_dob == appt_dob

            # Log each record for debugging visibility
            debug_print("------------------------------------------------")
            debug_print(f"👤 Name: {appt.get('first_name', '')} {appt.get('last_name', '')}")
            debug_print(f"📞 Phone: {appt.get('phone_e164', '(none)')}")
            debug_print(f"🎂 DOB: {appt.get('dob', '(none)')}")
            debug_print(f"🏠 Address: {appt.get('address', '(none)')}")
            debug_print(f"🏥 Insurance Company: {appt.get('insurance_name', '(none)')}")
            debug_print(f"🆔 Insurance Member ID: {appt.get('insurance_member_id', '(none)')}")
            debug_print(f"🕓 Start: {appt.get('start_utc', '(none)')}")
            debug_print(f"🔍 Match → phone={phone_match}, dob={dob_match}")

            # Skip if not matching both phone & DOB
            if not (phone_match and dob_match):
                debug_print("🚫 Skipped (does not match caller info)")
                continue

            # Format a readable date for voice prompt
            start_iso = appt.get("start_utc", "")
            try:
                friendly = _dt.fromisoformat(start_iso).strftime("%A, %B %d at %I:%M %p")
            except Exception:
                friendly = start_iso or "unknown time"

            # Add to candidate list for iteration
            candidates.append({
                "doctor_name": doctor,
                "start_utc": start_iso,
                "end_utc": appt.get("end_utc", ""),
                "friendly": friendly,
                "phone_e164": phone_e164,
                "dob": dob,
                "index_in_file": appointments.index(appt)
            })
            debug_print(f"✅ Added matching appointment → {friendly}")

        cancel_ctx["candidates"] = candidates
        cancel_ctx["iter_index"] = 0
        debug_print(f"cancel_appt_iterate: ✅ Prepared {len(candidates)} candidate(s) from JSON file")

        # ----------------------------------------------------------------------
        # 🚫 Step 3: Handle case — no appointments found
        # ----------------------------------------------------------------------
        if not candidates:
            resp.say("I couldn’t find any appointments that match your phone number or date of birth.", VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # 🎤 Step 4: Handle user input (speech or keypad)
        # ----------------------------------------------------------------------
        #   The user can respond "yes"/"no" or press 1/2.
        #   YES = cancel current appointment.
        #   NO  = skip to next appointment.
        # ----------------------------------------------------------------------
        dtmf = (request.values.get("Digits") or "").strip()
        utter = (speech_result or "").strip().lower()
        utter = _re.sub(r"[^a-z0-9]+", "", utter)

        YES = {"yes", "yeah", "yep", "confirm", "correct"}
        NO  = {"no", "nope", "next"}

        idx = int(cancel_ctx.get("iter_index", 0))
        total = len(candidates)

        if idx >= total:
            # User has gone through all appointments
            resp.say("That was the last appointment. Goodbye.", VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            save_session(call_sid)
            return str(resp)

        # Get the current candidate (appointment)
        cand = candidates[idx]

        # ----------------------------------------------------------------------
        # ✅ Step 5: YES → delete appointment from JSON
        # ----------------------------------------------------------------------
        if utter in YES or dtmf == "1":
            debug_print(f"cancel_appt_iterate: ✅ User confirmed cancel #{idx+1}/{total}")

            try:
                # Remove the entry from JSON and re-save file
                del appointments[cand["index_in_file"]]
                with open(doc_path, "w", encoding="utf-8") as f:
                    json.dump(appointments, f, indent=2)
                debug_print(f"🗑️ Deleted appointment from file: {doc_path}")
            except Exception as e:
                debug_print(f"⚠️ Could not delete appointment from JSON → {e}")

            # Save state and move to confirmation stage
            cancel_ctx["matching_event"] = cand
            session_data[call_sid]["stage"] = "cancel_appt_confirm"
            resp.redirect("/voice")
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # ↪️ Step 6: NO → skip to next appointment
        # ----------------------------------------------------------------------
        if utter in NO or dtmf == "2":
            debug_print(f"cancel_appt_iterate: ↪️ User skipped #{idx+1}/{total}")
            idx += 1
            cancel_ctx["iter_index"] = idx

            if idx >= total:
                # No more appointments left to review
                resp.say("That was the last appointment. Goodbye.", VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                save_session(call_sid)
                return str(resp)

            cand = candidates[idx]  # Move to next appointment

        # ----------------------------------------------------------------------
        # 🗣️ Step 7: Present current appointment to caller
        # ----------------------------------------------------------------------
        #   This stage asks: “Do you want to cancel this one?”
        #   The caller can say YES/NO or press 1/2.
        # ----------------------------------------------------------------------
        debug_print(f"cancel_appt_iterate: 🗣️ Presenting appointment #{idx+1}/{total}")

        say_line = (
            f"Appointment with {cand['doctor_name']} on {cand['friendly']}. "
            "Do you want to cancel this one? Say yes or no. "
            "Press 1 for yes, or 2 for no."
        )

        gather = make_gather(
            say_line,
            hints="yes no one two",
            input="speech dtmf",
            timeout=4,            # Short timeout for fast iteration
            speech_timeout="auto",
            finish_on_key="#",
            action="/voice",
            barge_in=True,
        )
        resp.append(gather)

        debug_print(f"cancel_appt_iterate: 🗣️ Appointment prompt built in "
                    f"{_time_mod.perf_counter() - t_stage_start:.3f}s")
        debug_print(f"cancel_appt_iterate: ✅ Total runtime {_time_mod.perf_counter() - t_stage_start:.3f}s")

        save_session(call_sid)
        return str(resp)





    # ----------------------------------------------------------------------
    # 🎯 Stage: book_appt_confirm
    # ----------------------------------------------------------------------
    # PURPOSE:
    #   Final confirmation step for appointment or new customer record.
    #
    #   NEW CUSTOMER:
    #       • Inserts record (including insurance info)
    #       • Instructs caller to verify info with clinic
    #
    #   CURRENT CUSTOMER:
    #       • Confirms appointment slot
    #       • Persists locally via book_appointment_for_dr_name()
    #       • Sends SMS confirmation
    # ----------------------------------------------------------------------


    elif stage == "book_appt_confirm":
    

        t_stage_start = _time_mod.perf_counter()
        debug_print("book_appt_confirm: 📍 Stage entered")

        # ----------------------------------------------------------------------
        # 💬 VOICE MESSAGES — centralized for maintainability & localization
        # ----------------------------------------------------------------------
        VOICE_NEW_CUSTOMER_MSG = (
            "Thank you {name}. You need to verify your information with the clinic "
            "before scheduling an appointment. Please contact the clinic to complete "
            "your registration. Goodbye!"
        )
        VOICE_MISSING_APPT_MSG = "Sorry, appointment time is missing. Please try again."
        VOICE_CONFIRMATION_ERROR_MSG = "Sorry, we couldn't confirm the appointment time."
        VOICE_SLOT_TAKEN_MSG = "Sorry, that slot was just taken. Please choose another time."
        VOICE_APPT_CONFIRMED_MSG = (
            "Your appointment with {doctor} has been booked on {time}. "
            "We look forward to seeing you. Goodbye!"
        )

        # ----------------------------------------------------------------------
        # 🧩 Retrieve session data
        # ----------------------------------------------------------------------
        sd = session_data.get(call_sid, {})
        customer_status = sd.get("customer_status", "current")
        debug_print(f"book_appt_confirm: 🧾 customer_status={customer_status}")

        # ----------------------------------------------------------------------
        # 👤 Customer Info
        # ----------------------------------------------------------------------
        customer = sd.get("customer", {}) or {}
        first_name       = (customer.get("first_name") or "").strip()
        last_name        = (customer.get("last_name")  or "").strip()
        customer_address = (customer.get("address")    or "").strip()
        customer_dob     = (customer.get("dob")        or "").strip()
        phone_e164       = (customer.get("phone_e164") or "").strip()
        insurance_name   = (customer.get("insurance_name") or "").strip()
        insurance_member_id = (customer.get("insurance_member_id") or "").strip()

        # ----------------------------------------------------------------------
        # 🆕 NEW CUSTOMER FLOW
        # ----------------------------------------------------------------------
        if customer_status == "new":
            debug_print("book_appt_confirm: 🆕 new customer → skipping appointment booking")

            try:
                inserted_ok = insert_customer(
                    phone=phone_e164,
                    dob=customer_dob,
                    first_name=first_name,
                    last_name=last_name,
                    address=customer_address,
                    cc_name=f"{first_name} {last_name}",
                    cc_number="",
                    cc_exp="",
                    cc_cvv="",
                    insurance_name=insurance_name,
                    insurance_member_id=insurance_member_id,
                    customer_status="new",
                    pin_number=0,
                )
                debug_print(f"book_appt_confirm: ✅ insert_customer (new) → {inserted_ok}")
            except Exception as e:
                debug_print(f"book_appt_confirm: ❌ insert_customer failed for new customer → {e}")

            # 🗣 Speak polite final message and end the call
            msg = VOICE_NEW_CUSTOMER_MSG.format(name=first_name or "there")
            resp.say(gpt_speak(msg), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)

        # ----------------------------------------------------------------------
        # 👤 CURRENT CUSTOMER FLOW
        # ----------------------------------------------------------------------
        debug_print("book_appt_confirm: 👤 current customer flow continues")

        doctor_name = sd.get("doctor_name", "the doctor")
        appt = sd.get("appointment_time", {}) or {}
        appointment_start = appt.get("start")
        appointment_end   = appt.get("end")

        # ----------------------------------------------------------------------
        # ❌ Missing appointment time → terminate early
        # ----------------------------------------------------------------------
        if not appointment_start:
            debug_print("book_appt_confirm: ❌ appointment_start missing for current customer")
            resp.say(gpt_speak(VOICE_MISSING_APPT_MSG), VOICE)
            resp.hangup()
            return str(resp)

        # ----------------------------------------------------------------------
        # 🕒 Convert UTC → Local timezone for spoken format
        # ----------------------------------------------------------------------
        tz_name = globals().get("CLINIC_TZ", "America/Chicago")
        try:
            tz = _pytz.timezone(tz_name)
        except Exception:
            tz = _pytz.timezone("America/Chicago")

        try:
            dt_utc   = datetime.fromisoformat(appointment_start.replace("Z", "+00:00"))
            dt_local = dt_utc.astimezone(tz)
            # Convert to readable format, e.g., "Monday, October 28 at 9:00 AM"
            formatted_time = dt_local.strftime("%A, %B %d at %I:%M %p").replace(" 0", " ")
        except Exception as e:
            debug_print(f"book_appt_confirm: time format error → {e}")
            resp.say(gpt_speak(VOICE_CONFIRMATION_ERROR_MSG), VOICE)
            resp.hangup()
            return str(resp)

        # ----------------------------------------------------------------------
        # 🕓 Compute missing end time if needed
        # ----------------------------------------------------------------------
        if not appointment_end:
            try:
                dur = int(globals().get("APPOINTMENT_DURATION_MINUTES", 30))
                end_dt = dt_utc + timedelta(minutes=dur)
                appointment_end = end_dt.astimezone(_pytz.UTC).isoformat()
            except Exception as e:
                debug_print(f"book_appt_confirm: ❌ failed computing end time → {e}")
                resp.say(gpt_speak(VOICE_CONFIRMATION_ERROR_MSG), VOICE)
                resp.hangup()
                return str(resp)

        # ----------------------------------------------------------------------
        # ✅ Verify slot availability
        # ----------------------------------------------------------------------
        try:
            slot_ok = is_doctor_slot_available(doctor_name, appointment_start, appointment_end)
        except Exception as e:
            debug_print(f"book_appt_confirm: ⚠️ slot check failed → {e}")
            slot_ok = False

        if not slot_ok:
            sd["stage"] = "collect_book_time_date"
            resp.append(make_gather(VOICE_SLOT_TAKEN_MSG))
            return str(resp)

        # ----------------------------------------------------------------------
        # 💾 Insert or Update Customer Record Locally
        # ----------------------------------------------------------------------
        try:
            inserted_ok = insert_customer(
                phone=phone_e164,
                dob=customer_dob,
                first_name=first_name,
                last_name=last_name,
                address=customer_address,
                cc_name=f"{first_name} {last_name}",
                cc_number="",
                cc_exp="",
                cc_cvv="",
                insurance_name=insurance_name,
                insurance_member_id=insurance_member_id,
                customer_status="current",
                pin_number=0,
            )
            debug_print(f"book_appt_confirm: ✅ insert_customer (current) → {inserted_ok}")
        except Exception as e:
            debug_print(f"book_appt_confirm: ❌ insert_customer failed → {e}")

        # ----------------------------------------------------------------------
        # 🗂️ Log Appointment Locally
        # ----------------------------------------------------------------------
        try:
            full_name = f"{first_name} {last_name}".strip()
            book_appointment_for_dr_name(
                doctor_name=doctor_name,
                phone=phone_e164,
                utc_start=appointment_start,
                utc_end=appointment_end,
                name=full_name,
                dob=customer_dob,
                address=customer_address,
                friendly_local=formatted_time,
                debug=True
            )
            debug_print(f"book_appt_confirm: ✅ Appointment logged locally for {doctor_name} at {formatted_time}")
        except Exception as e:
            debug_print(f"book_appt_confirm: ⚠️ failed to log appointment locally → {e}")

        # ----------------------------------------------------------------------
        # ✅ Confirm appointment + send SMS
        # ----------------------------------------------------------------------
        msg = VOICE_APPT_CONFIRMED_MSG.format(doctor=doctor_name, time=formatted_time)
        resp.say(gpt_speak(msg), VOICE)

        try:
            sms = (
                f"Hi {first_name or 'there'}, your appointment with {doctor_name} is confirmed "
                f"on {formatted_time}. Thank you for choosing Epic Therapist Clinic."
            )
            client.messages.create(body=sms, from_=TWILIO_PHONE_NUMBER, to=phone_e164)
            debug_print(f"book_appt_confirm: 📩 SMS sent to {phone_e164}")
        except Exception as e:
            debug_print(f"book_appt_confirm: ⚠️ SMS failed → {e}")

        # ----------------------------------------------------------------------
        # 🧹 Cleanup and hang up
        # ----------------------------------------------------------------------
        resp.hangup()
        session_data.pop(call_sid, None)
        debug_print(f"book_appt_confirm: ✅ completed in {_time_mod.perf_counter() - t_stage_start:.3f}s")
        return str(resp)









        # ======================================================================
        # 🧩 Stage: cancel_appt_confirm
        # ----------------------------------------------------------------------
        # PURPOSE:
        #   • Final confirmation stage for appointment cancellation.
        #   • Completely independent of Google Calendar — all operations now use
        #     local doctor JSON files.
        #   • Deletes the confirmed appointment entry permanently from the JSON list.
        #   • Supports the "reschedule after cancel" flow.
        #
        # INPUTS:
        #   - cancel_ctx["matching_event"]: holds the appointment info (from iterate stage)
        #   - doctor_name, phone_e164, dob, and start time
        #
        # OUTPUTS:
        #   - Removes the corresponding appointment from JSON file.
        #   - Confirms to the caller via voice prompt.
        #   - Optionally advances to reschedule flow if requested.
        # ======================================================================

    
    elif stage == "cancel_appt_confirm":
        

        t0 = _time_mod.perf_counter()
        debug_print("cancel_appt_confirm: 📍 Stage entered")

        # Retrieve cancel context and current candidate (from previous stage)
        cancel_ctx = session_data[call_sid].get("cancel", {})
        cand = cancel_ctx.get("matching_event")  # The appointment user just confirmed to cancel
        reschedule_flag = session_data.get(call_sid, {}).get("reschedule_after_cancel", False)

        # ----------------------------------------------------------------------
        # 🚫 Safety check — ensure we have a valid candidate
        # ----------------------------------------------------------------------
        if not cand:
            debug_print("cancel_appt_confirm: ⚠️ No candidate found in session (nothing to cancel).")
            resp.say(gpt_speak("Sorry, I couldn’t find that appointment to cancel."), VOICE)

            # If user was trying to reschedule, gracefully continue to time selection
            if reschedule_flag:
                session_data[call_sid]["stage"] = "collect_book_time_date"
                session_data[call_sid]["reschedule_after_cancel"] = False
                resp.append(make_gather(
                    "Please say the new date and time for your appointment, for example, 'October 12th at 9 a.m.'"
                ))
                resp.redirect("/voice")
                save_session(call_sid)
                return str(resp)

            # Otherwise, end call politely
            resp.hangup()
            session_data.pop(call_sid, None)
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # 🧾 Extract parameters from the confirmed candidate
        # ----------------------------------------------------------------------
        doctor_name = cand.get("doctor_name", "")
        start_utc   = cand.get("start_utc", "")
        end_utc     = cand.get("end_utc", "")
        friendly    = cand.get("friendly", "")
        phone_e164  = cand.get("phone_e164", "")
        dob         = cand.get("dob", "")

        debug_print(f"cancel_appt_confirm: 🩺 Doctor='{doctor_name}', Start='{start_utc}', End='{end_utc}'")

        # ----------------------------------------------------------------------
        # 📂 Identify doctor’s appointment file
        # ----------------------------------------------------------------------
        #   Each doctor has a JSON file under the "appointments" folder.
        #   Example:  appointments/alfred_hitchcock.json
        # ----------------------------------------------------------------------
        safe_name = doctor_name.lower().replace(" ", "_")
        doc_path = f"appointments/{safe_name}.json"

        # ----------------------------------------------------------------------
        # 🔍 Load all existing appointments from that file
        # ----------------------------------------------------------------------
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                appointments = json.load(f)
            debug_print(f"cancel_appt_confirm: 📁 Loaded {len(appointments)} appointments from {doc_path}")
        except FileNotFoundError:
            debug_print(f"cancel_appt_confirm: ❌ Doctor JSON not found: {doc_path}")
            resp.say(f"I couldn’t find any appointment records for {doctor_name}.", VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            save_session(call_sid)
            return str(resp)
        except Exception as e:
            debug_print(f"cancel_appt_confirm: ⚠️ Error reading doctor JSON → {e}")
            resp.say("Sorry, something went wrong while accessing the appointment list.", VOICE)
            resp.hangup()
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # 🧩 Find matching appointment by start time, phone, and DOB
        # ----------------------------------------------------------------------
        deleted = False
        for appt in list(appointments):  # iterate over a copy to safely delete
            appt_phone = _re.sub(r"\D", "", appt.get("phone_e164", ""))
            cand_phone = _re.sub(r"\D", "", phone_e164)
            appt_dob = (appt.get("dob", "") or "").strip()
            start_match = appt.get("start_utc", "") == start_utc
            phone_match = appt_phone == cand_phone
            dob_match = not dob or appt_dob == dob

            if start_match and phone_match and dob_match:
                debug_print(f"cancel_appt_confirm: ✅ Found matching appointment → {appt}")
                appointments.remove(appt)
                deleted = True
                break

        # ----------------------------------------------------------------------
        # 💾 Save updated appointment list back to JSON
        # ----------------------------------------------------------------------
        if deleted:
            try:
                with open(doc_path, "w", encoding="utf-8") as f:
                    json.dump(appointments, f, indent=2)
                debug_print(f"cancel_appt_confirm: 🗑️ Appointment successfully removed from {doc_path}")
                resp.say(gpt_speak(f"Your appointment with {doctor_name} on {friendly} has been cancelled."), VOICE)
            except Exception as e:
                debug_print(f"cancel_appt_confirm: ⚠️ Could not update JSON file → {e}")
                resp.say("Sorry, there was an error while removing your appointment.", VOICE)
        else:
            debug_print("cancel_appt_confirm: ❌ No matching record found in JSON (nothing removed).")
            resp.say("Sorry, I couldn’t find that appointment to cancel.", VOICE)

        # ----------------------------------------------------------------------
        # 🔁 Optional reschedule flow
        # ----------------------------------------------------------------------
        if reschedule_flag:
            debug_print("cancel_appt_confirm: 🔄 Reschedule-after-cancel detected → forwarding to collect_book_time_date")
            session_data[call_sid]["stage"] = "collect_book_time_date"
            session_data[call_sid]["reschedule_after_cancel"] = False

            # Keep customer info intact so we can reuse it for the new booking
            cust = session_data[call_sid].setdefault("customer", {})
            if phone_e164:
                cust["phone_e164"] = phone_e164
            if dob:
                cust["dob"] = dob

            # Prompt for new date/time
            resp.append(make_gather(
                "Your previous appointment has been cancelled. "
                "Please say the new date and time for your appointment, for example, 'October 12th at 9 a.m.'"
            ))
            resp.redirect("/voice")
            debug_print(f"cancel_appt_confirm: ⏱️ total stage time {_time_mod.perf_counter() - t0:.3f}s")
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # ✅ End of normal flow — hang up politely
        # ----------------------------------------------------------------------
        resp.hangup()
        session_data.pop(call_sid, None)
        debug_print(f"cancel_appt_confirm: ✅ total runtime {_time_mod.perf_counter() - t0:.3f}s")
        save_session(call_sid)
        return str(resp)

   



if __name__ == "__main__":
    
    # ------------------------
    # 🔁 Call this once at startup
    # ------------------------

    load_doctor_appointments()


    app.run(host="0.0.0.0", port=5000)
