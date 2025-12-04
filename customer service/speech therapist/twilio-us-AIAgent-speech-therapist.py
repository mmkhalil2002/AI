#=======
# updated  12/04/2025
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
    #     r"\d+"      → one or more digits, e.g., "7", "1972", "12345"
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
import time as _time_mod
import threading
import traceback
import dateutil.parser as dp
import string, unicodedata as _uni
import random  # local import for clarity
import unicodedata



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
# ----------------------------------------------------------------------
# 🔐 API & SERVICE CREDENTIALS
# ----------------------------------------------------------------------

# OPENAI_API_KEY:
#   • Your secret API key for authenticating requests to OpenAI models (e.g., GPT-4 or GPT-5).
#   • Used for natural language understanding, summarization, and response generation
#     within the voice assistant.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# TWILIO_ACCOUNT_SID:
#   • The unique identifier for your Twilio account.
#   • Used to authenticate API calls to Twilio (voice, SMS, recordings, etc.).
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")

# TWILIO_AUTH_TOKEN:
#   • The authentication token paired with the Account SID.
#   • Grants permission to send or receive calls, messages, and manage Twilio resources.
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

# TWILIO_PHONE_NUMBER:
#   • The Twilio-provisioned phone number assigned to your application (E.164 format).
#   • Used as the caller ID for outgoing calls and as the receiver for incoming calls.
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_NUMBER")

# GOOGLE_CREDENTIALS:
#   • The path to the Google service account JSON credentials file.
#   • Required to access Google Calendar for appointment booking and lookup.
GOOGLE_CREDENTIALS = "credentials.json"


# ================================================================
# 🕒 MAX_SILENT_TIME
# ----------------------------------------------------------------
# FUNCTIONAL PURPOSE:
#   Defines the pause (in milliseconds) inserted between spoken
#   appointment choices in SSML <break> tags.
#
#   This allows the voice assistant to read options clearly,
#   giving callers enough time to process each choice.
#
#   Increasing this value makes pauses longer (slower reading).
#   Decreasing it makes the assistant read options faster.
#
#   Example:
#       MAX_SILENT_TIME = 600   → 0.6-second pause
#       MAX_SILENT_TIME = 1200  → 1.2-second pause
#
#   Used exclusively inside confirm_time_choice() when building
#   the options menu for the user.
# ================================================================
PAUSE_BETWEEN_OPTIONS = 700   # milliseconds


# ----------------------------------------------------------------------
# 🗣️ VOICE INPUT & RECORDING SETTINGS
# ----------------------------------------------------------------------

# SPEECH_INPUT_DURATION:
#   • Maximum time (in seconds) Twilio will wait for the caller to speak before timing out.
#   • If set to "auto", Twilio uses voice activity detection (VAD) to decide when to stop recording.
SPEECH_INPUT_DURATION = os.getenv("SPEECH_INPUT_DURATION", "6")  # keep as string for Twilio

# PAUSE_BETWEEN_DIGITS:
#   • Number of seconds Twilio waits for the next keypad digit (DTMF) input.
#   • After this timeout, the system processes the digits entered so far.
PAUSE_BETWEEN_DIGITS = int(os.getenv("PAUSE_BETWEEN_DIGITS", "7"))

# MAX_RECORD_TIME:
#   • Maximum duration (in seconds) for voicemail or freeform audio recordings.
#   • Used in flows where the caller leaves a message or dictation.
MAX_RECORD_TIME = int(os.getenv("MAX_RECORD_TIME", "60"))


# ----------------------------------------------------------------------
# 🔁 RETRY & LIMIT SETTINGS
# ----------------------------------------------------------------------

# MAX_NUMBER_DR_RETRY:
#   • Maximum number of times to retry retrieving or validating a doctor’s availability.
MAX_NUMBER_DR_RETRY = int(os.getenv("MAX_NUMBER_DR_RETRY", 3))

# MAX_APPT_RETRIEVED_FROM_CALENDER:
#   • Maximum number of appointment events to retrieve from Google Calendar in one query.
MAX_APPT_RETRIEVED_FROM_CALNDER = int(os.getenv("MAX_APPT_RETRIEVED_FROM_CALENDER", 50))

# APPOINTMENT_DURATION_MINUTES:
#   • Default length of each appointment slot (15, 30, 45, or 60 minutes).
#   • Used for building and validating slot availability windows.
APPOINTMENT_DURATION_MINUTES = int(os.getenv("APPOINTMENT_DURATION_MINUTES", 30))

# MAX_TIME_SELECTION_ATTEMPTS:
#   • Maximum number of times the system will prompt the caller to select or confirm
#     an appointment time before ending the session.
MAX_TIME_SELECTION_ATTEMPTS = int(os.getenv("MAX_TIME_SELECTION_ATTEMPTS", 3))

# MAX_SILENCE_RETRIES:
#   • Maximum number of retries allowed when the caller remains silent.
#   • After reaching this limit, the call ends with a polite message.
MAX_SILENCE_RETRIES = int(os.getenv("MAX_SILENCE_RETRIES", 3))

# MAX_GET_PHONE_RETRIES:
#   • Number of times to re-prompt the user to input their phone number if it’s missing or invalid.
MAX_GET_PHONE_RETRIES = int(os.getenv("MAX_GET_PHONE_RETRIES", 3))

# MAX_ADVANCE_MONTHS:
#   • Defines how far into the future (in months) appointment searches can go.
#   • Prevents users from booking unrealistically distant dates.
MAX_ADVANCE_MONTHS = int(os.getenv("MAX_ADVANCE_MONTHS", 6))


# ----------------------------------------------------------------------
# 👤 CUSTOMER MANAGEMENT SETTINGS
# ----------------------------------------------------------------------

# CREATE_NEW_CUSTOMER:
#   • Determines whether new customers can be automatically created during the call flow.
#   • True → allow new registration via phone.
#   • False → restrict to pre-existing patients only.
CREATE_NEW_CUSTOMER = bool(os.getenv("CREATE_NEW_CUSTOMER", True))




# ----------------------------------------------------------------------
# 💾 LOCAL DATABASE SETTINGS
# ----------------------------------------------------------------------

# DB_FOLDER:
#   • Folder location where customer and appointment data are stored locally.
#   • Defaults to "appointment_data".
DB_FOLDER = os.getenv("DB_FOLDER", "appointment_data")

# DB_FILE:
#   • JSON file path containing customer records.
#   • Used as a local cache or fallback to Google Calendar data.
DB_FILE = os.path.join(DB_FOLDER, "customers.json")


# ----------------------------------------------------------------------
# 🌐 GLOBAL TIMEZONE & WORK SCHEDULE
# ----------------------------------------------------------------------

# CLINIC_TZ:
#   • Default time zone used for all appointment times, parsing, and formatting.
#   • Critical for time conversion between Twilio, Google Calendar, and local users.
CLINIC_TZ = os.getenv("CLINIC_TZ", "America/Chicago")

# WORKING_DAYS:
#   • Days of the week the clinic operates.
#   • 0 = Monday, 6 = Sunday.
#   • Example:
#       "0,1,2,3,4" → Monday–Friday
#       "0,1,2,3,5" → Sunday–Thursday
WORKING_DAYS = [int(x) for x in os.getenv("WORKING_DAYS", "0,1,2,3,4").split(",") if x.strip().isdigit()]

# WORKING_HOURS_START / WORKING_HOURS_END:
#   • Clinic’s opening and closing times (24-hour clock).
#   • Example: 8 → 08:00 AM, 17 → 5:00 PM.
WORKING_HOURS_START = int(os.getenv("WORKING_HOURS_START", 8))
WORKING_HOURS_END   = int(os.getenv("WORKING_HOURS_END", 5))

# LUNCH_BREAK_START / LUNCH_BREAK_END:
#   • Defines the clinic’s lunch break window.
#   • Appointments cannot be booked during this period.
#   • Split into hours (H) and minutes (M) for flexible configuration.
LUNCH_BREAK_START = time(
    int(os.getenv("LUNCH_BREAK_START_H", 13)),  # Default: 1 PM
    int(os.getenv("LUNCH_BREAK_START_M", 0))
)
LUNCH_BREAK_END = time(
    int(os.getenv("LUNCH_BREAK_END_H", 14)),    # Default: 2 PM
    int(os.getenv("LUNCH_BREAK_END_M", 0))
)

# NEXT_AVAILABLE_SLOT_OFFSET:
#   • Minimum time (in minutes) between the current moment and the earliest
#     appointment slot offered to a caller.
#   • Ensures adequate preparation time and avoids last-minute bookings.
#   • Example:
#       If now = 8:15 AM and offset = 30, first slot ≥ 8:45 AM.
NEXT_AVAILABLE_SLOT_OFFSET = int(os.getenv("NEXT_AVAILABLE_SLOT_OFFSET", 20))

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



# ----------------------------------------------------------------------
# 🕓 CLINIC OPERATING SCHEDULE & SESSION CONFIGURATION
# ----------------------------------------------------------------------

# WORKING_DAYS:
#   • Specifies which days of the week the clinic operates.
#   • Values are integers (0–6) corresponding to Python’s weekday mapping:
#       0 = Monday, 1 = Tuesday, 2 = Wednesday, 3 = Thursday,
#       4 = Friday, 5 = Saturday, 6 = Sunday
#   • Example:
#       "0,1,2,3,4" → Monday through Friday
#       "0,1,2,3,5" → Sunday through Thursday (common in Middle East)
#   • Loaded dynamically from environment variable "WORKING_DAYS".
#   • Used by appointment scheduling logic to skip weekends or holidays.
WORKING_DAYS = [
    int(x) for x in os.getenv("WORKING_DAYS", "0,1,2,3,4").split(",")
    if x.strip().isdigit()
]

# WORKING_HOURS_START / WORKING_HOURS_END:
#   • Define the daily operating hours of the clinic (24-hour format).
#   • Example:
#       WORKING_HOURS_START = 8  →  Clinic opens at 08:00 AM
#       WORKING_HOURS_END   = 17 →  Clinic closes at 05:00 PM
#   • These hours are enforced by all scheduling and availability logic.
#   • Appointments outside this range are automatically excluded.
WORKING_HOURS_START = int(os.getenv("WORKING_HOURS_START", 8))
WORKING_HOURS_END   = int(os.getenv("WORKING_HOURS_END", 17))

# LUNCH_BREAK_START / LUNCH_BREAK_END:
#   • Define the clinic’s lunch break period when appointments are not allowed.
#   • Configured using hour and minute components for flexibility.
#   • Example:
#       LUNCH_BREAK_START_H = 13, LUNCH_BREAK_START_M = 0  →  1:00 PM
#       LUNCH_BREAK_END_H   = 14, LUNCH_BREAK_END_M   = 0  →  2:00 PM
#   • These values are respected by all scheduling and slot-search routines.
LUNCH_BREAK_START = time(
    int(os.getenv("LUNCH_BREAK_START_H", 13)),  # default 13:00 (1 PM)
    int(os.getenv("LUNCH_BREAK_START_M", 0))
)
LUNCH_BREAK_END = time(
    int(os.getenv("LUNCH_BREAK_END_H", 14)),    # default 14:00 (2 PM)
    int(os.getenv("LUNCH_BREAK_END_M", 0))
)

# SESSION_TIME:
#   • Standard duration (in minutes) of a therapy or appointment session.
#   • Used in conjunction with APPOINTMENT_DURATION_MINUTES to control
#     the slot length for both scheduling and display purposes.
#   • Example: 30 → each booked session lasts 30 minutes.
SESSION_TIME = int(os.getenv("SESSION_TIME", 30))

# ----------------------------------------------------------------------
# ⚙️ RUNTIME BEHAVIOR FLAGS
# ----------------------------------------------------------------------

# USE_GPT:
#   • Enables or disables OpenAI GPT integration for natural language processing.
#   • When False → system uses predefined rule-based or fallback responses.
#   • When True  → system calls GPT models for understanding intent, NLU, etc.
USE_GPT = False

# DEBUG:
#   • Enables verbose console logging for development and troubleshooting.
#   • Should be set to False in production to reduce log noise.
DEBUG = True

# ----------------------------------------------------------------------
# 🌍 COUNTRY CONFIGURATION
# ----------------------------------------------------------------------

# COUNTRY:
#   • Defines the primary country context for the assistant’s behavior.
#   • Used to adjust voice prompts, phone number normalization, date formats,
#     and possibly language preferences.
#   • Example:
#       "US" → Default configuration for United States
#       "EG" → Enables Egypt-specific behaviors (e.g., Arabic prompts, time zones)
#   • Can be changed using environment variable:
#       export COUNTRY=EG
COUNTRY = os.getenv("COUNTRY", "US").upper()   # Default → "US"


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



# ======================================================================
# 🧠 FUNCTIONAL DESCRIPTION — smart_parse_time()
# ======================================================================
# This routine parses a spoken/natural-language date-time string into a
# validated timezone-aware UTC appointment window.
#
# EXTENDED DEBUGGING:
#   ✔ Logs every normalization stage
#   ✔ Logs detection of month/day/hour/minute/AMPM
#   ✔ Logs reasoning behind year rollover & past/future classification
#   ✔ Logs timezone resolution, DST awareness, and fallback paths
#
# ======================================================================

def smart_parse_time(raw: str, tz_offset_hours: int = -5, default_duration_min: int = 30):

    # ------------------------------------------------------------------
    # 🧩 Safe debug wrapper
    # ------------------------------------------------------------------
    def _dbg(msg):
        try:
            debug_print(msg)
        except Exception:
            print(msg)

    # ------------------------------------------------------------------
    # 🚫 Validate input
    # ------------------------------------------------------------------
    if not raw or not str(raw).strip():
        _dbg("[smart_parse_time] ⚠️ Empty input received — aborting")
        return None

    s = str(raw).strip().lower()
    _dbg(f"[smart_parse_time] 🔍 RAW INPUT = '{s}'")

    # ------------------------------------------------------------------
    # 🧹 NORMALIZATION STAGE 1 — Clean STT noise
    # ------------------------------------------------------------------
    _dbg("[smart_parse_time] 🔧 Normalizing STT artifacts")
    before_norm = s

    s = _re.sub(r"o['’]?clock", "", s)
    s = _re.sub(r"\b(a\s*\.?\s*m\.?)\b", "am", s)
    s = _re.sub(r"\b(p\s*\.?\s*m\.?)\b", "pm", s)
    s = _re.sub(r"[^\w\s:]", " ", s)
    s = _re.sub(r"\s+", " ", s).strip()

    # Fix STT "2000 pm" weirdness
    s = _re.sub(r"\b20\s?00\s*pm\b", "2 00 pm", s)
    s = _re.sub(r"\b2000\s*pm\b", "2 00 pm", s)
    s = _re.sub(r"\btwenty hundred\s*pm\b", "2 00 pm", s)

    _dbg(f"[smart_parse_time] 🔄 NORMALIZED = '{s}' (from '{before_norm}')")

    # ------------------------------------------------------------------
    # 🗓 TOKEN EXTRACTION
    # ------------------------------------------------------------------
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
    }

    month = day = None
    hour, minute = 9, 0
    ampm = "am"

    # MONTH detection
    for m in months:
        if m in s:
            month = months[m]
            _dbg(f"[smart_parse_time] 🗓 MONTH DETECTED → {m} ({month})")
            break
    if not month:
        _dbg("[smart_parse_time] 🗓 MONTH missing — will assume current month")

    # TIME detection
    m_time = _re.search(r"\b(\d{1,2})(?:[: ](\d{2}))?\s*(am|pm)?\b", s)
    if m_time:
        hour = int(m_time.group(1))
        minute = int(m_time.group(2) or 0)
        ampm = (m_time.group(3) or "am").lower()
        _dbg(f"[smart_parse_time] ⏰ TIME DETECTED → hour={hour}, minute={minute}, ampm={ampm}")
    else:
        _dbg("[smart_parse_time] ⏰ NO TIME FOUND — default 9:00 AM applied")

    # DAY detection
    m_day = _re.search(r"\b([1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?\b(?=.*\bat\b)", s)
    if m_day:
        day = int(m_day.group(1))
        _dbg(f"[smart_parse_time] 📅 DAY DETECTED → {day}")
    else:
        _dbg("[smart_parse_time] 📅 DAY missing — will assume TODAY")

    # ------------------------------------------------------------------
    # 🌎 TIMEZONE RESOLUTION
    # ------------------------------------------------------------------
    tz_name = globals().get("CLINIC_TZ", "America/Chicago")
    _dbg(f"[smart_parse_time] 🌎 Loading clinic timezone → {tz_name}")

    try:
        tz_local = _pytz.timezone(tz_name)
        _dbg(f"[smart_parse_time] 🌎 Timezone loaded successfully (DST aware: {tz_local.dst(datetime.now()) != timedelta(0)})")
    except Exception:
        _dbg(f"[smart_parse_time] ❗ Invalid TZ '{tz_name}', using fallback offset {tz_offset_hours}")
        tz_local = _pytz.FixedOffset(tz_offset_hours * 60)

    now_local = datetime.now(tz_local)
    _dbg(f"[smart_parse_time] 📌 NOW LOCAL = {now_local.isoformat()}")

    # ------------------------------------------------------------------
    # 📅 Fill missing date parts
    # ------------------------------------------------------------------
    if not month:
        month = now_local.month
    if not day:
        day = now_local.day

    _dbg(f"[smart_parse_time] 🧩 FINAL DATE TOKENS → month={month}, day={day}")

    # ------------------------------------------------------------------
    # 🔄 Normalize AM/PM → 24h
    # ------------------------------------------------------------------
    before_clock = hour
    if ampm == "pm" and hour < 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    _dbg(f"[smart_parse_time] 🕒 12h→24h conversion → {before_clock} {ampm} → {hour:02d}:{minute:02d}")

    # ------------------------------------------------------------------
    # 🧮 Construct localized datetime
    # ------------------------------------------------------------------
    try:
        dt_local = tz_local.localize(datetime(now_local.year, month, day, hour, minute))
        _dbg(f"[smart_parse_time] 🏗️ Local datetime constructed → {dt_local.isoformat()}")
    except Exception as e:
        _dbg(f"[smart_parse_time] ❌ INVALID date components → {e}")
        return None

    # ------------------------------------------------------------------
    # 🔁 YEAR ROLLOVER (handles STT "old month")
    # ------------------------------------------------------------------
    if dt_local < now_local and (now_local.month - month) > 6:
        _dbg("[smart_parse_time] 🔁 YEAR ROLLOVER triggered → adding +1 year")
        dt_local = tz_local.localize(datetime(now_local.year + 1, month, day, hour, minute))
        _dbg(f"[smart_parse_time] 🔁 Rolled datetime → {dt_local.isoformat()}")

    # ------------------------------------------------------------------
    # ⏳ PAST/FUTURE DETECTION
    # ------------------------------------------------------------------
    is_past = dt_local < (now_local - timedelta(minutes=2))

    if is_past:
        _dbg(f"[smart_parse_time] ⏳ PARSED TIME appears in the past → {dt_local.isoformat()}")

        # Special rule: same-day future times must NOT be flagged past
        if dt_local.date() == now_local.date() and dt_local > now_local:
            _dbg("[smart_parse_time] 🔧 SAME-DAY FIX — marking as future instead")
            is_past = False

    # ------------------------------------------------------------------
    # 🚫 Booking horizon limit
    # ------------------------------------------------------------------
    max_months = int(globals().get("MAX_ADVANCE_MONTHS", 6))
    limit_local = now_local + timedelta(days=30 * max_months)

    _dbg(f"[smart_parse_time] 📏 HORIZON LIMIT → {max_months} months → {limit_local.isoformat()}")

    if dt_local > limit_local:
        _dbg("[smart_parse_time] 🚫 Parsed datetime exceeds booking horizon — rejecting")
        return None

    # ------------------------------------------------------------------
    # 🌐 Convert LOCAL → UTC
    # ------------------------------------------------------------------
    dt_utc = dt_local.astimezone(_pytz.UTC)
    dt_end = dt_utc + timedelta(minutes=default_duration_min)

    # ------------------------------------------------------------------
    # 🧑‍🤝‍🧑 Friendly text
    # ------------------------------------------------------------------
    friendly = dt_local.strftime("%A, %B %-d at %-I:%M %p").replace(" 0", " ")

    # ------------------------------------------------------------------
    # 📦 RESULT
    # ------------------------------------------------------------------
    result = {
        "start":    dt_utc.isoformat().replace("+00:00", "Z"),
        "end":      dt_end.isoformat().replace("+00:00", "Z"),
        "friendly": friendly,
        "is_past":  is_past,
    }

    _dbg(f"[smart_parse_time] ✅ SUCCESS → '{raw}' → {friendly} | UTC={result['start']} | past={is_past}")
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
    #import re  # local import to avoid global dependency if not needed

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
    _contains_ssml = bool(_re.search(r"<\s*(break|emphasis|prosody|say-as)\b", prompt, _re.IGNORECASE))

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
"""
this not used any more because it diesnt use google calender any more

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
"""


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
# ----------------------------------------------------------------------
# 🧱 GLOBAL SESSION STRUCTURE — DOCUMENTATION ONLY
# ----------------------------------------------------------------------
# This block shows the expected structure of session_data in memory.
# Each key under session_data corresponds to a unique Twilio CallSid.
# ----------------------------------------------------------------------

# session_data = {
#     "<CallSid>": {                      # Unique Twilio CallSid for each active call
#
#         # --------------------------------------------------------------
#         # 🔹 CORE SESSION STATE
#         # --------------------------------------------------------------
#         "stage": "collect_dob",            # Current conversational stage (e.g., intro, intent, collect_dob)
#         "origin_stage": "book",            # Root flow context ('book', 'cancel', 'reschedule', 'update_cc', 'register')
#         "country": "US",                   # Caller’s country (used for E.164 normalization)
#         "from_e164": "+14694633276",       # Caller’s normalized E.164 number
#         "skip_silence_once": True,         # Skips silence guard for one prompt (temporary flag)
#         "no_input_expected": False,        # Prevents silence handler from re-prompting
#         "retry_booking": 0,                # Number of retries while selecting a valid slot
#         "retry_time": 0,                   # Number of retries for date/time parsing
#
#         # --------------------------------------------------------------
#         # 🔹 CUSTOMER INFORMATION
#         # --------------------------------------------------------------
#         "customer": {
#             "first_name": "Mohamed",       # Captured first name
#             "last_name": "Khalil",         # Captured last name
#             "phone": "4694633276",         # Raw phone input (unformatted)
#             "phone_e164": "+14694633276",  # Normalized E.164 phone number (preferred)
#             "dob": "1985-07-03",           # Date of birth (ISO format)
#             "customer_status": "current",  # One of: 'new', 'pending', 'current', 'unknown'
#
#             # 💳 Credit card information (collected in collect_cc)
#             "cc_number": "4111111111111111",  # Credit card number (unmasked)
#             "cc_exp": "0927",                 # Expiration date (MMYY)
#             "cc_cvv": "123",                  # Security code
#             "cc_name": "Mohamed Khalil",      # Cardholder name
#         },
#
#         # --------------------------------------------------------------
#         # 🔹 BOOKING CONTEXT
#         # --------------------------------------------------------------
#         "booking": {
#             "doctor_name": "Alfred Hitchcock",          # Selected doctor
#             "doctor_id": "cal_alfred_hitchcock@clinic.com",  # Google Calendar ID for the doctor
#             "requested_time": "2025-11-07T15:30:00Z",   # Appointment start time (UTC)
#             "appointment_length": 30,                   # Appointment duration in minutes
#             "appointment_confirmed": True,              # Whether booking was successful
#             "pending_insurance": {                      # Insurance information (optional)
#                 "company": "Blue Cross Blue Shield",     # Insurance provider
#                 "member_id": "123456789",                # Member ID
#             },
#         },
#
#         # --------------------------------------------------------------
#         # 🔹 CANCELLATION / RESCHEDULE CONTEXT
#         # --------------------------------------------------------------
#         "cancel": {
#             "doctor": "Alfred Hitchcock",               # Doctor for the canceled appointment
#             "matching_event": {                         # Details of the event to cancel
#                 "spoken_dt": "October 8th at 9:00 a.m.",# Original spoken date/time phrase
#                 "start": "2025-10-08T14:00:00Z",        # Event start time (UTC)
#                 "end": "2025-10-08T14:30:00Z",          # Event end time (UTC)
#             },
#             "awaiting_input": False,                    # Indicates whether user must confirm
#             "silence_cancel_dt": 0,                     # Silence retry counter for date/time input
#         },
#
#         # --------------------------------------------------------------
#         # 🔹 CREDIT CARD UPDATE CONTEXT
#         # --------------------------------------------------------------
#         "cc_step": 3,                  # Step index in collect_cc (1=card, 2=expiration, 3=cvv)
#         "enforce_dtmf_cc": False,      # Forces keypad input instead of speech (for privacy)
#         "cc_speech_tries": 0,          # Retry counter for speech misrecognition
#
#         # --------------------------------------------------------------
#         # 🔹 DOCTOR MAPPING (for selection menus)
#         # --------------------------------------------------------------
#         "doctor_dtmf_map": {           # Maps DTMF digits to doctor names
#             "1": "Alfred Hitchcock",
#             "2": "Dr. Faten Salim",
#             "3": "Dr. Sarah Osman"
#         },
#         "doctor_name": "Alfred Hitchcock",  # Current doctor in focus
#
#         # --------------------------------------------------------------
#         # 🔹 SILENCE / RETRY COUNTERS
#         # --------------------------------------------------------------
#         "silence_first_name": 0,       # Silence counter for first name prompt
#         "silence_last_name": 0,        # Silence counter for last name prompt
#         "silence_cc": 0,               # Silence counter for collect_cc
#         "silence_cancel_phone": 0,     # Silence counter for cancel phone number
#         "silence_cancel_dt": 0,        # Silence counter for cancel date/time
#
#         # --------------------------------------------------------------
#         # 🔹 META DATA
#         # --------------------------------------------------------------
#         "created_at": "2025-11-07T15:32:18Z",   # Timestamp when session was created
#         "last_updated": "2025-11-07T15:35:12Z"  # Timestamp of last modification
#     }
# }
#
# ----------------------------------------------------------------------
# 🧠 NOTES:
# - Each CallSid entry is unique per caller session.
# - Keys are created dynamically as the user moves through the call flow.
# - Silence counters prevent infinite loops (max 3 attempts).
# - session_data is in-memory and resets when the app restarts.
# ----------------------------------------------------------------------




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
                "dob": "1972-07-03",
                "start_utc": "2025-10-22T16:00:00Z",
                "end_utc": "2025-10-22T16:30:00Z"
            },
            ...
        ]
    """
# ======================================================================
# 🩺 FUNCTION: is_doctor_slot_available()
# ======================================================================
# 🎯 FUNCTIONAL PURPOSE:
#     This routine determines whether a doctor has an AVAILABLE
#     appointment slot at a given time. It is used *only for booking*.
#
#     All business rules are evaluated in **LOCAL CLINIC TIME**:
#        • Working days (Mon–Fri, unless changed)
#        • Working hours (e.g., 08:00 → 17:00 local)
#        • Lunch break exclusion
#        • MAX_ADVANCE_MONTHS future limit
#        • Reject past times
#
#     Overlap detection is done in **UTC**, ensuring correctness across:
#        ✓ DST transitions
#        ✓ Different country rules (US, Egypt)
#        ✓ UTC timestamps stored in JSON
#
# 👌 CANCELLATION DOES NOT USE THIS FUNCTION ANYMORE.
#     Cancellation uses an exact-match JSON lookup instead.
#
# 🧠 RETURN VALUE:
#     True  → Slot is available for booking
#     False → Slot is invalid, forbidden, past, or already booked
# ======================================================================

def is_doctor_slot_available(doctor_name: str, start_iso: str, end_iso: str) -> bool:

    # ------------------------------------------------------------------
    # ⏳ Helper: Convert any ISO string → timezone-aware UTC datetime
    # ------------------------------------------------------------------
    def _as_utc_dt(s: str):
        try:
            s2 = s.replace("Z", "+00:00")            # convert Z → +00:00 for parser
            dt = isoparse(s2)
            if dt.tzinfo is None:                   # naive → treat as UTC
                dt = dt.replace(tzinfo=_pytz.UTC)
            return dt.astimezone(_pytz.UTC)         # ensure UTC normalized
        except Exception as e:
            debug_print(f"[is_doctor_slot_available] ⚠️ bad datetime '{s}': {e}")
            raise

    # ------------------------------------------------------------------
    # ⏳ Normalize input times (UTC)
    # ------------------------------------------------------------------
    try:
        start_dt = _as_utc_dt(start_iso)
        end_dt   = _as_utc_dt(end_iso)
    except Exception:
        debug_print("[is_doctor_slot_available] ⚠️ bad input — cannot parse")
        return False

    # ------------------------------------------------------------------
    # 🧭 Sanity check (must be forward time interval)
    # ------------------------------------------------------------------
    if end_dt <= start_dt:
        debug_print("[is_doctor_slot_available] ⚠️ end ≤ start")
        return False

    # ------------------------------------------------------------------
    # 🌍 Convert UTC → LOCAL for business rules
    # ------------------------------------------------------------------
    tz_name = globals().get("CLINIC_TZ", "America/Chicago")
    try:
        tz_local = _pytz.timezone(tz_name)
    except:
        tz_local = _pytz.timezone("America/Chicago")

    start_local = start_dt.astimezone(tz_local)
    end_local   = end_dt.astimezone(tz_local)

    # ------------------------------------------------------------------
    # 🗓️ Working day validation
    # ------------------------------------------------------------------
    WORKING_DAYS = set(globals().get("WORKING_DAYS", {0, 1, 2, 3, 4}))
    if start_local.weekday() not in WORKING_DAYS:
        debug_print(f"[is_doctor_slot_available] ❌ Non-working day: {start_local.strftime('%A')}")
        return False

    # ------------------------------------------------------------------
    # 🕒 Working hours validation (LOCAL TIME!)
    # ------------------------------------------------------------------
    WSTART = int(globals().get("WORKING_HOURS_START", 8))
    WEND   = int(globals().get("WORKING_HOURS_END", 17))

    start_h = start_local.hour + start_local.minute / 60
    end_h   = end_local.hour   + end_local.minute   / 60

    if start_h < WSTART or end_h > WEND:
        debug_print("[is_doctor_slot_available] ❌ Outside working hours")
        return False

    # ------------------------------------------------------------------
    # 🍽️ Lunch break exclusion (LOCAL TIME)
    # ------------------------------------------------------------------
    LUNCH_START = globals().get("LUNCH_BREAK_START", time(13, 0))
    LUNCH_END   = globals().get("LUNCH_BREAK_END",   time(14, 0))

    s_t = start_local.time()
    e_t = end_local.time()

    if not (e_t <= LUNCH_START or s_t >= LUNCH_END):
        debug_print("[is_doctor_slot_available] ❌ overlaps with lunch break")
        return False

    # ------------------------------------------------------------------
    # 🕒 Past or too-far-in-future check (UTC)
    # ------------------------------------------------------------------
    now_utc = datetime.now(_pytz.UTC)

    if end_dt <= now_utc:
        debug_print("[is_doctor_slot_available] ❌ slot in the past")
        return False

    MAX_ADVANCE_MONTHS = int(globals().get("MAX_ADVANCE_MONTHS", 6))

    # helper to compute dt + n months
    def _add_months(dt, months):
        #import calendar
        y, m = dt.year, dt.month + months
        y += (m - 1) // 12
        m = ((m - 1) % 12) + 1
        d = min(dt.day, calendar.monthrange(y, m)[1])
        return dt.replace(year=y, month=m, day=d)

    limit_end_utc = _add_months(now_utc, MAX_ADVANCE_MONTHS)
    if start_dt > limit_end_utc:
        debug_print("[is_doctor_slot_available] ❌ beyond booking horizon")
        return False

    # ------------------------------------------------------------------
    # 📁 Load doctor’s JSON schedule
    # ------------------------------------------------------------------
    safe_name = _re.sub(r"\s+", "_", doctor_name.strip().lower())
    doc_path = os.path.join("appointment_data", f"{safe_name}.json")
    debug_print(f"[is_doctor_slot_available] 📁 Loading: {doc_path}")

    if not os.path.exists(doc_path):
        debug_print("[is_doctor_slot_available] 🆕 no file — slot free")
        return True

    try:
        with open(doc_path, "r", encoding="utf-8") as f:
            appointments = json.load(f)
    except:
        debug_print("[is_doctor_slot_available] ⚠️ bad JSON — treat as free")
        return True

    # ------------------------------------------------------------------
    # 🔁 OVERLAP CHECK (allow 1-minute tolerance)
    # ------------------------------------------------------------------
    TOLERANCE = timedelta(minutes=1)  # <-- FIX ADDED HERE

    for i, appt in enumerate(appointments, start=1):
        raw_s = appt.get("start_utc") or appt.get("utc_start") or appt.get("time")
        raw_e = appt.get("end_utc")   or appt.get("utc_end")

        if not raw_s or not raw_e:
            continue

        try:
            ap_s = _as_utc_dt(raw_s)
            ap_e = _as_utc_dt(raw_e)
        except:
            continue

        if ap_e <= now_utc:
            continue

        # --------------------------------------------------------------
        # FIXED OVERLAP RULE: allow back-to-back with tolerance
        # --------------------------------------------------------------
        #
        # NO OVERLAP IF:
        #     new_end <= ap_start + tolerance
        #  OR new_start >= ap_end
        #
        # OTHERWISE → overlap
        #
        if (end_dt <= ap_s + TOLERANCE) or (start_dt >= ap_e):
            # no overlap → continue checking others
            continue

        # OTHERWISE: overlap
        debug_print(f"[is_doctor_slot_available] ❌ overlap with appt #{i}")
        return False

    # ------------------------------------------------------------------
    # 🎉 SLOT IS AVAILABLE
    # ------------------------------------------------------------------
    debug_print("[is_doctor_slot_available] ✅ SLOT AVAILABLE")
    return True





# ==========================================================================
# 📌 FUNCTIONAL PURPOSE — get_doctor_next_available_slots()
# --------------------------------------------------------------------------
# This function determines the next available appointment slots for the
# specified doctor, scanning forward through the clinic schedule.
#
# Each candidate slot is evaluated in LOCAL clinic time, then validated
# against the doctor's stored JSON history using is_doctor_slot_available().
#
# The function returns up to `limit` available future appointment times.
#
# --------------------------------------------------------------------------
# 🎯 KEY RESPONSIBILITIES
# --------------------------------------------------------------------------
# • Scan forward chronologically in local time
# • Respect configured working hours (e.g., 8 AM–5 PM)
# • Respect working days (e.g., Monday–Friday)
# • Respect lunch break window (e.g., 1 PM–2 PM)
# • Enforce booking horizon limit (MAX_ADVANCE_MONTHS)
# • Enforce a minimum delay before the first allowed slot
#   (NEXT_AVAILABLE_SLOT_OFFSET)
# • Enforce supported appointment durations: 15, 30, 45, or 60 minutes
# • Ensure all generated slots align perfectly to the scheduling grid
#   (e.g., always 8:00, 8:30, 9:00 — never 8:07)
# • Validate overlap conditions with REAL appointment data
# • Return the first N valid future appointment slots
#
# --------------------------------------------------------------------------
# 🧠 FIXES INCORPORATED
# --------------------------------------------------------------------------
# ✔ Enforce NEXT_AVAILABLE_SLOT_OFFSET and align it to the grid
# ✔ Correct lunch overlap logic to match tolerant edge-overlap rules
# ✔ Remove redundant working-hour boundary checks (previously blocked 8:30)
# ✔ Ensure slot grid always aligns correctly (prevents missing 8:30, 9:00)
# ✔ Normalize timezone handling safely (clinic-local → UTC → local)
# ✔ Prevent "ghost overlap" caused by misaligned request timestamps
#
# ==========================================================================

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
    

    # ----------------------------------------------------------------------
    # 🔧 Load global config values (with safe defaults)
    # ----------------------------------------------------------------------
    MAX_ADVANCE_MONTHS = int(globals().get("MAX_ADVANCE_MONTHS", 6))
    NEXT_AVAILABLE_SLOT_OFFSET = int(globals().get("NEXT_AVAILABLE_SLOT_OFFSET", 30))

    debug_print(f"[get_doctor_next_available_slots] ▶ doctor={doctor_name}, from={from_start_iso}")
    debug_print(f"[get_doctor_next_available_slots] ⚙ MAX_ADVANCE_MONTHS={MAX_ADVANCE_MONTHS}, OFFSET={NEXT_AVAILABLE_SLOT_OFFSET}")

    # ----------------------------------------------------------------------
    # ⏳ APPOINTMENT DURATION
    # ----------------------------------------------------------------------
    if duration_minutes is None:
        # Try APPOINTMENT_DURATION_MINUTES first, fallback to SESSION_TIME
        duration_minutes = int(globals().get(
            "APPOINTMENT_DURATION_MINUTES",
            globals().get("SESSION_TIME", 30)
        ))

    # Only allow standard durations
    if duration_minutes not in (15, 30, 45, 60):
        duration_minutes = 30

    # Slot grid matches duration unless overridden
    if slot_step_minutes is None:
        slot_step_minutes = duration_minutes

    # ----------------------------------------------------------------------
    # 🌍 Load clinic timezone (DST-aware)
    # ----------------------------------------------------------------------
    if tz_name is None:
        tz_name = globals().get("CLINIC_TZ", "America/Chicago")

    try:
        tz_local = _pytz.timezone(tz_name)
    except:
        debug_print("[get_doctor_next_available_slots] ⚠ invalid timezone → fallback Chicago")
        tz_local = _pytz.timezone("America/Chicago")

    # ----------------------------------------------------------------------
    # 🕘 Working hours and working days
    # ----------------------------------------------------------------------
    WSTART = int(globals().get("WORKING_HOURS_START", 8))   # e.g., 8 AM
    WEND   = int(globals().get("WORKING_HOURS_END", 17))    # e.g., 5 PM

    # If not provided → one continuous block
    if not work_hours:
        work_hours = ((WSTART, WEND),)

    WORKING_DAYS = {int(x) for x in globals().get("WORKING_DAYS", {0,1,2,3,4})}
    # 0 = Monday → 4 = Friday by default

    # ----------------------------------------------------------------------
    # 🍽 Lunch break setup
    # ----------------------------------------------------------------------
    def _as_time(val):
        """Convert input like '13:00' or 13 to Python time()."""
        if isinstance(val, time):
            return val
        if not val:
            return None
        try:
            s = str(val)
            h, m = (s.split(":") if ":" in s else (s, "0"))
            return time(int(h), int(m))
        except:
            return None

    LUNCH_START = _as_time(globals().get("LUNCH_BREAK_START"))
    LUNCH_END   = _as_time(globals().get("LUNCH_BREAK_END"))

    # ----------------------------------------------------------------------
    # 🔍 Search horizon (days)
    # ----------------------------------------------------------------------
    if search_days is None:
        search_days = int(globals().get("SEARCH_DAYS", 14))

    # ----------------------------------------------------------------------
    # 🎨 Friendly formatter for human speech
    # ----------------------------------------------------------------------
    def _friendly(dt_local, now_local):
        """Create human-friendly string."""
        try:
            if dt_local.year != now_local.year:
                return dt_local.strftime("%A, %B %-d, %Y at %-I:%M %p")
            return dt_local.strftime("%A, %B %-d at %-I:%M %p")
        except:
            return dt_local.strftime("%A, %B %d at %I:%M %p")

    # ----------------------------------------------------------------------
    # ⏫ Align any datetime to the N-minute appointment grid
    # ----------------------------------------------------------------------
    def _align_up(dt_local, step_min, anchor):
        """
        Examples:
            anchor = 8:00
            dt = 8:07 → 8:30
            dt = 8:31 → 9:00
        """
        dt_local = dt_local.replace(second=0, microsecond=0)

        diff = int((dt_local - anchor).total_seconds() // 60)

        if diff <= 0:
            return anchor  # before opening → snap to wstart

        rem = diff % step_min
        return dt_local if rem == 0 else dt_local + timedelta(minutes=(step_min - rem))

    # ----------------------------------------------------------------------
    # 📅 Add months safely while avoiding overflow (Feb 30 → Feb 28)
    # ----------------------------------------------------------------------
    def _add_months(dt, months):
        #import calendar
        y, m = dt.year, dt.month + months
        y += (m - 1) // 12
        m = ((m - 1) % 12) + 1
        d = min(dt.day, calendar.monthrange(y, m)[1])
        return dt.replace(year=y, month=m, day=d)

    # ----------------------------------------------------------------------
    # 🕒 Convert `from_start_iso` → local time
    # ----------------------------------------------------------------------
    now_utc = datetime.now(_pytz.UTC)
    now_local = now_utc.astimezone(tz_local)

    try:
        req_utc = isoparse(from_start_iso)
        if req_utc.tzinfo is None:
            req_utc = _pytz.UTC.localize(req_utc)
    except:
        req_utc = now_utc

    req_local = req_utc.astimezone(tz_local)

    # ----------------------------------------------------------------------
    # 🕒 ENFORCE FUTURE OFFSET (e.g., no slot allowed <30min from now)
    # ----------------------------------------------------------------------
    min_allowed_local = now_local + timedelta(minutes=NEXT_AVAILABLE_SLOT_OFFSET)

    if req_local < min_allowed_local:
        req_local = min_allowed_local   # enforce delay for safety

    # ----------------------------------------------------------------------
    # ✔ FIX: ALIGN THE ENFORCED TIME TO APPOINTMENT GRID
    # ----------------------------------------------------------------------
    anchor = req_local.replace(hour=WSTART, minute=0, second=0, microsecond=0)
    req_local = _align_up(req_local, slot_step_minutes, anchor)

    # ----------------------------------------------------------------------
    # 📅 Build full search horizon in UTC
    # ----------------------------------------------------------------------
    limit_end_utc = _add_months(now_utc, MAX_ADVANCE_MONTHS)
    search_end_utc = min(now_utc + timedelta(days=search_days), limit_end_utc)

    # ----------------------------------------------------------------------
    # 🚀 Initialization
    # ----------------------------------------------------------------------
    cur_local = req_local
    results = []
    seen = set()

    # ==========================================================================
    # MAIN SEARCH LOOP — scan future days and future slots
    # ==========================================================================
    while cur_local.astimezone(_pytz.UTC) < search_end_utc and len(results) < limit:

        # ------------------------------------------------------------------
        # Skip weekends / closed days
        # ------------------------------------------------------------------
        if cur_local.weekday() not in WORKING_DAYS:
            debug_print("[get_doctor_next_available_slots] 💤 Skipping closed day")
            cur_local = (cur_local + timedelta(days=1)).replace(
                hour=WSTART, minute=0, second=0, microsecond=0
            )
            continue

        # ------------------------------------------------------------------
        # Build today's working intervals
        # ------------------------------------------------------------------
        windows = []
        for ws, we in work_hours:
            wstart = tz_local.localize(datetime(cur_local.year, cur_local.month, cur_local.day, ws, 0))
            wend   = tz_local.localize(datetime(cur_local.year, cur_local.month, cur_local.day, we, 0))
            windows.append((wstart, wend))

        progressed = False

        # ------------------------------------------------------------------
        # Iterate through working windows for today
        # ------------------------------------------------------------------
        for wstart, wend in windows:

            # If before opening → jump to opening
            if cur_local < wstart:
                cur_local = wstart

            # Align cursor to appointment grid
            cur_local = _align_up(cur_local, slot_step_minutes, wstart)

            # --------------------------------------------------------------
            # SLOT-SCANNING LOOP
            # --------------------------------------------------------------
            while cur_local + timedelta(minutes=duration_minutes) <= wend and len(results) < limit:

                # ------------------------------------------------------------------
                # Skip below offset (e.g., less than 30 minutes ahead)
                # ------------------------------------------------------------------
                if cur_local < min_allowed_local:
                    cur_local += timedelta(minutes=slot_step_minutes)
                    continue

                # ------------------------------------------------------------------
                # ✔ FIXED LUNCH-BREAK LOGIC
                # ------------------------------------------------------------------
                if LUNCH_START and LUNCH_END:
                    slot_end_t = (cur_local + timedelta(minutes=duration_minutes)).time()

                    # Overlap rule: only skip if the slot actually intersects lunch
                    if not (slot_end_t <= LUNCH_START or cur_local.time() >= LUNCH_END):
                        debug_print("[get_doctor_next_available_slots] 🍽 lunch skip")
                        # Jump to lunch end, then re-align
                        cur_local = tz_local.localize(
                            datetime.combine(cur_local.date(), LUNCH_END)
                        )
                        cur_local = _align_up(cur_local, slot_step_minutes, wstart)
                        continue

                # ------------------------------------------------------------------
                # FUTURE HORIZON STOP
                # ------------------------------------------------------------------
                if cur_local.astimezone(_pytz.UTC) > limit_end_utc:
                    return results

                # ------------------------------------------------------------------
                # Convert slot to UTC for validation
                # ------------------------------------------------------------------
                start_iso = cur_local.astimezone(_pytz.UTC).isoformat().replace("+00:00", "Z")
                end_iso   = (cur_local + timedelta(minutes=duration_minutes)).astimezone(_pytz.UTC).isoformat().replace("+00:00", "Z")

                # ------------------------------------------------------------------
                # ASK is_doctor_slot_available() if this slot is free
                # ------------------------------------------------------------------
                try:
                    if is_doctor_slot_available(doctor_name, start_iso, end_iso) and start_iso not in seen:
                        seen.add(start_iso)
                        results.append({
                            "start": start_iso,
                            "end": end_iso,
                            "friendly": _friendly(cur_local, now_local),
                            "tz": tz_name,
                        })
                        debug_print(f"[get_doctor_next_available_slots] ✅ Added {results[-1]['friendly']}")
                except Exception as e:
                    debug_print(f"[get_doctor_next_available_slots] ❌ availability check error: {e}")

                # ------------------------------------------------------------------
                # Move to NEXT slot in the grid
                # ------------------------------------------------------------------
                cur_local += timedelta(minutes=slot_step_minutes)
                progressed = True

        # ------------------------------------------------------------------
        # If nothing in this day → jump to next day at opening time
        # ------------------------------------------------------------------
        if not progressed:
            cur_local = (cur_local + timedelta(days=1)).replace(
                hour=WSTART, minute=0, second=0, microsecond=0
            )

    debug_print(f"[get_doctor_next_available_slots] ⏹ Finished with {len(results)} slot(s)")
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


"""
    Retrieve a customer's stored credit card number (cc_number) from customers.json.

    Parameters:
        phone_e164 (str): Customer's phone number in E.164 format (e.g., "+14155552671").
        dob (str): Date of birth in ISO format (YYYY-MM-DD).

    Returns:
        str | None: The customer's stored credit card number if found and valid, otherwise None.

    Behavior:
      ✅ Looks up the customer record by key = "<phone_e164>|<dob>".
      ✅ Ensures that the cc_number is a string of digits (13–19 length typical for credit cards).
      ✅ Logs detailed debug output for traceability and troubleshooting.
"""

def get_customer_cc(phone_e164: str, dob: str) -> Optional[str]:
    
    try:
        # ----------------------------------------------------------------------
        # 🧱 Ensure database (customers.json) is initialized and available
        # ----------------------------------------------------------------------
        init_db()
        data = _load_customers()

        # Build composite key (same schema as used by other customer lookups)
        key = _key(phone_e164, dob)

        # Attempt to retrieve customer record from JSON store
        rec = data.get(key)
        if not rec:
            debug_print(f"get_customer_cc: ❌ no record for key={key}")
            return None

        # ----------------------------------------------------------------------
        # 💳 Extract and validate stored card number
        # ----------------------------------------------------------------------
        cc = rec.get("cc_number")
        if isinstance(cc, str):
            # Remove spaces or hyphens commonly found in stored CC formats
            cc_clean = cc.replace(" ", "").replace("-", "")
            if cc_clean.isdigit() and 13 <= len(cc_clean) <= 19:
                debug_print(f"get_customer_cc: ✅ found cc_number (masked)={_mask(cc_clean)} for {key}")
                return cc_clean
            else:
                debug_print(f"get_customer_cc: ⚠️ invalid cc_number format for {key} → {cc}")
                return None
        else:
            debug_print(f"get_customer_cc: ⚠️ missing or non-string cc_number for {key}")
            return None

    except Exception as e:
        debug_print(f"get_customer_cc: ⚠️ error reading cc_number for {phone_e164}|{dob}: {e}")
        return None

"""
    Update a customer's stored credit card number in customers.json.

    Parameters:
        phone_e164 (str): Customer's phone number in E.164 format (e.g., "+14155552671").
        dob (str): Date of birth in ISO format (YYYY-MM-DD).
        new_cc (str): New credit card number (string of digits 13–19 in length).

    Returns:
        bool: True if the update succeeded, False otherwise.

    Behavior:
      ✅ Validates the new credit card format before updating.
      ✅ Looks up the record by key = "<phone_e164>|<dob>".
      ✅ Persists the updated cc_number field to customers.json.
      ✅ Logs detailed success and error traces for debugging.
"""

def update_customer_cc(phone_e164: str, dob: str, new_cc: str) -> bool:
    
    try:
        # ----------------------------------------------------------------------
        # 🧱 Ensure the customers DB exists and is ready
        # ----------------------------------------------------------------------
        init_db()
        data = _load_customers()
        key = _key(phone_e164, dob)

        # ----------------------------------------------------------------------
        # 🔍 Verify that the record exists before modifying
        # ----------------------------------------------------------------------
        if key not in data:
            debug_print(f"update_customer_cc: ❌ no record found for key={key}")
            return False

        # ----------------------------------------------------------------------
        # 💳 Sanitize and validate the new credit card number
        # ----------------------------------------------------------------------
        cc_clean = str(new_cc).replace(" ", "").replace("-", "")
        if not (cc_clean.isdigit() and 13 <= len(cc_clean) <= 19):
            debug_print(f"update_customer_cc: ⚠️ invalid cc_number format → {new_cc}")
            return False

        # ----------------------------------------------------------------------
        # 💾 Update the record and persist the changes
        # ----------------------------------------------------------------------
        data[key]["cc_number"] = cc_clean
        _save_customers(data)
        debug_print(f"update_customer_cc: ✅ updated cc_number (masked)={_mask(cc_clean)} for {key}")

        return True

    except Exception as e:
        debug_print(f"update_customer_cc: ⚠️ error updating cc_number for {phone_e164}|{dob}: {e}")
        return False


"""
    Retrieve the customer's current status ("new" or "current") from customers.json.

    Behavior:
      ✅ Uses strict E.164-only lookup (no legacy phone fallback).
      ✅ Returns "new" or "current" if found, None if no record exists.
      ✅ Performs a light scan fallback if the exact key is missing.
"""

def get_customer_status(phone: str, dob: str, default_country: str = COUNTRY) -> Optional[str]:
    
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



"""
    Update the customer's credit card info in customers.json by (phone_e164|dob).

    Optimization goals:
      ✅ Strict E.164-only normalization (no legacy fallback).
      ✅ Clearer flow with early returns.
      ✅ Reduced redundant normalization and string ops.
      ✅ Maintains identical behavior and full debug traceability.
"""


def update_cc_info(
    phone: str,
    dob: str,
    *,
    cc_number: Optional[str] = None,
    cc_exp: Optional[str] = None,
    cc_cvv: Optional[str] = None,
    default_country: str = COUNTRY,  # e.g., "US" or "EG"
) -> bool:
   
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

# ================================================================
# 🩺 FUNCTION: cancel_appointment_for_dr_name
# ================================================================
#
# 🎯 PURPOSE:
#     Safely remove ONE appointment record from a doctor’s JSON file.
#     This cancellation requires matching ALL THREE keys:
#         ✓ phone number    (E.164 normalized)
#         ✓ date of birth   (ISO string exactly)
#         ✓ UTC start time  (normalized to canonical UTC ISO)
#
# 🧩 WHAT THIS FUNCTION DOES:
#     1. Normalizes and validates the caller’s phone number into E.164.
#     2. Loads the doctor’s JSON file from disk.
#     3. Converts all stored appointment timestamps to canonical UTC ISO.
#     4. Derives E.164 phone numbers from old or new storage formats.
#     5. Compares caller input → stored appointment records (E.164 + DOB + UTC time).
#     6. Writes back the JSON file WITHOUT the matched appointment.
#
# 📝 INPUTS:
#     doctor_name (str)     → Used to find the doctor’s JSON file.
#     phone (str)           → Raw or E.164; normalized before comparison.
#     dob (str)             → Expected exact DOB string ('YYYY-MM-DD').
#     utc_start (str)       → ISO timestamp representing the start of the appointment.
#     default_country       → Used when normalizing non-E.164 phone inputs.
#
# 🔄 OUTPUT:
#     True  → At least one matching appointment was removed.
#     False → No matching record OR error occurred.
#
# 🛡 SAFETY:
#     • Does NOT throw exceptions outward — always returns False on error.
#     • Supports US + Egypt phone formats automatically.
#     • Ensures old JSON formats (digit-only phone) remain compatible.
#
# ================================================================
def cancel_appointment_for_dr_name(
    doctor_name: str,
    phone: str,
    dob: str,
    utc_start: str,
    *,
    default_country: str = COUNTRY  # fallback country from global constant
) -> bool:

    # ------------------------------------------------------------
    # 🔧 Normalize input phone number → E.164
    # ------------------------------------------------------------
    raw = (phone or "").strip()
    phone_e164 = ""

    # Case 1: Already valid E.164 (starts with + and rest are digits)
    if raw.startswith("+") and raw[1:].replace(" ", "").isdigit():
        phone_e164 = "+" + raw[1:].replace(" ", "")

    else:
        # Case 2: Convert raw phone → E.164 using helper
        try:
            # Primary attempt using default_country (US or EG)
            phone_e164 = normalize_phone_e164(raw, (default_country or "US").upper()) or ""

            # Secondary attempt: try the OTHER supported country
            if not phone_e164:
                alt = "EG" if (default_country or "US").upper() != "EG" else "US"
                phone_e164 = normalize_phone_e164(raw, alt) or ""

        except Exception:
            # If normalization fails completely → phone_e164 remains ""
            phone_e164 = ""

    # Strip DOB and ensure string format
    dob_str = (dob or "").strip()

    # Build the full file path for doctor_name
    full_path = get_doctor_filename(doctor_name)

    debug_print(
        f"cancel_appointment_by_name: doctor='{doctor_name}' "
        f"phone_e164='{phone_e164 or '∅'}' dob='{dob_str or '∅'}' utc='{utc_start or '∅'}'"
    )

    # ------------------------------------------------------------
    # 🛑 Validate minimal requirements before proceeding
    # ------------------------------------------------------------
    if not (os.path.exists(full_path) and phone_e164 and dob_str and utc_start):
        return False

    # ------------------------------------------------------------
    # 📂 Load appointment list from disk
    # ------------------------------------------------------------
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # If the JSON root is not a list → invalid schema
        if not isinstance(data, list):
            return False

    except Exception as e:
        debug_print(f"cancel_appointment_by_name: read error → {e}")
        return False

    # ------------------------------------------------------------
    # 🕓 Normalization helper for stored UTC timestamps
    #     - Convert any ISO or naive timestamp to canonical UTC ISO string
    # ------------------------------------------------------------
    def _to_utc_iso(s: str) -> str:
        dt = dtparser.isoparse(s)

        # If naive timestamp → treat as UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            # Convert any timezone → UTC
            dt = dt.astimezone(timezone.utc)

        # Remove micros for consistent comparison; write as ISO+Z
        return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    # Normalize the target UTC start time (the one we are trying to cancel)
    try:
        target_norm = _to_utc_iso(utc_start)
    except Exception as e:
        debug_print(f"cancel_appointment_by_name: utc parse error → {e}")
        return False

    # ------------------------------------------------------------
    # 📞 Helper to derive E.164 from stored appointment record
    # ------------------------------------------------------------
    def _appt_e164(appt: dict) -> str:

        # 1. New format → explicit phone_e164 field
        pe = (appt.get("phone_e164") or "").strip()
        if pe.startswith("+") and pe[1:].replace(" ", "").isdigit():
            return "+" + pe[1:].replace(" ", "")

        # 2. Old format → "phone" field (may be digits only)
        cand = (appt.get("phone") or "").strip()
        if cand:
            try:
                # Primary
                e164 = normalize_phone_e164(cand, (default_country or "US").upper()) or ""

                # Secondary
                if not e164:
                    alt = "EG" if (default_country or "US").upper() != "EG" else "US"
                    e164 = normalize_phone_e164(cand, alt) or ""

                if e164:
                    return e164

            except Exception:
                pass

        # No usable phone
        return ""

    # ------------------------------------------------------------
    # 🔍 Compare all stored appointments with input and filter out matches
    # ------------------------------------------------------------
    kept = []      # appointments we keep
    removed = 0    # counter for deleted appointments

    for appt in data:

        # Only dictionary objects matter; keep non-dict safely
        if not isinstance(appt, dict):
            kept.append(appt)
            continue

        # Extract normalized E.164 number for this record
        ap_e164 = _appt_e164(appt)

        # Extract stored DOB
        ap_dob  = (appt.get("dob", "") or "").strip()

        # Extract stored start time
        ap_time_raw = (appt.get("time") or appt.get("start") or "").strip()

        # Normalize stored timestamp → canonical UTC ISO
        try:
            ap_time_norm = _to_utc_iso(ap_time_raw) if ap_time_raw else ""
        except Exception:
            # If record is malformed → DO NOT delete it
            kept.append(appt)
            continue

        # Match condition: ALL must match
        if ap_e164 == phone_e164 and ap_dob == dob_str and ap_time_norm == target_norm:
            removed += 1
        else:
            kept.append(appt)

    # ------------------------------------------------------------
    # 🛑 No matching record found
    # ------------------------------------------------------------
    if removed == 0:
        debug_print("cancel_appointment_by_name: no matching record found")
        return False

    # ------------------------------------------------------------
    # 💾 Write updated appointment list back to file
    # ------------------------------------------------------------
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
@app.route("/voice/", methods=["POST"])  # Accepts trailing slash for flexibility
@safe_twiml_route
def voice():
    """
    🎙️ Twilio Voice Webhook Entry Point
    -------------------------------------------------------------
    • This function prepares the environment for your inline
      stage-handling logic (if stage == "intro", elif stage == ...).
    • It handles session loading, country detection, and input capture.
    • After setup, execution flows directly into your existing
      "if stage == ..." chain defined immediately below.
    -------------------------------------------------------------
    """
    global session_data, doctor_names

    # ----------------------------------------------------------------------
    # 🧭 Initialize Twilio Voice Response
    # ----------------------------------------------------------------------
    resp = VoiceResponse()
    debug_print("[voice] ▶ enter voice()")

    # ----------------------------------------------------------------------
    # 🆔 Unique Call Identifier
    # ----------------------------------------------------------------------
    call_sid = request.values.get("CallSid", "")
    debug_print(f"[voice] CallSid={call_sid}")

    # ----------------------------------------------------------------------
    # 💾 Load or initialize session data for this call
    # ----------------------------------------------------------------------
    sd = session_data.get(call_sid, {})
    session_data.setdefault(call_sid, sd)
    debug_print(f"[voice] 🔁 Loaded session for {call_sid}: keys={list(sd.keys())}")
    debug_print(f"[voice] 🩺 doctor_name loaded → {sd.get('doctor_name')}")

    # ----------------------------------------------------------------------
    # 🗣️ Capture Speech / DTMF Input
    # ----------------------------------------------------------------------
    speech_result = (request.values.get("SpeechResult") or "").strip()
    try:
        dtmf_digits = (request.values.get("Digits") or "").strip()
    except Exception:
        dtmf_digits = ""
    debug_print(f"[voice] inputs → speech='{speech_result}' dtmf='{dtmf_digits}'")

    # ----------------------------------------------------------------------
    # 🌍 Determine Caller Country from phone prefix
    # ----------------------------------------------------------------------
    from_number = (request.values.get("From") or "").strip()
    derived_country = COUNTRY  # Default global fallback

    if from_number.startswith("+20"):
        derived_country = "EG"
    elif from_number.startswith("+1"):
        derived_country = "US"

    if "country" not in sd:
        sd["country"] = derived_country
        debug_print(f"[voice] 🌐 country seeded → {derived_country}")
    else:
        debug_print(f"[voice] 🌐 country exists → {sd.get('country')}")

    # ----------------------------------------------------------------------
    # ☎️ Save E.164 Caller Number
    # ----------------------------------------------------------------------
    if from_number.startswith("+"):
        sd["from_e164"] = from_number
        debug_print(f"[voice] from_e164 set → {from_number}")

    # ----------------------------------------------------------------------
    # 🧠 Determine Current Stage (default to 'intro')
    # ----------------------------------------------------------------------
    stage = sd.get("stage", "intro")
    debug_print(f"[voice] 🎯 stage='{stage}'")

    # ----------------------------------------------------------------------
    # 📢 Log Speech Result for debugging trace
    # ----------------------------------------------------------------------
    print(f"📢 [voice] Speech recognized: {speech_result}")

    # ----------------------------------------------------------------------
    # ✅ Continue into stage-handling logic
    # ----------------------------------------------------------------------
    # Do NOT return here — execution will flow into your existing
    #   if stage == "intro": ...
    #   elif stage == "intent": ...
    #   etc.
    # Each stage builds its own TwiML response and ends with:
    #   return str(resp)
    # ----------------------------------------------------------------------

    # ↓ Your existing “if stage == ... elif stage == ...” block starts below











       # 🧩 Stage: INTRO
    # ----------------------------------------------------------------------
    # Functional Description:
    # ----------------------------------------------------------------------
    # • This is the very first conversational stage after Twilio hits /voice.
    # • It welcomes the caller to the clinic and presents the main menu.
    # • The prompt offers both speech and keypad (DTMF) options for navigation.
    # • The <Gather> TwiML block listens for input and posts the result
    #   back to /voice with SpeechResult or Digits.
    # • If the caller stays silent, local retry logic reprompts up to 3 times.
    #   - First time → greeting + full menu
    #   - Second time → “I couldn’t catch that” + full menu
    #   - Third time → transfers to voicemail
    # ----------------------------------------------------------------------
    # Flow Summary:
    # ----------------------------------------------------------------------
    # 1️⃣ Caller dials the clinic number.
    # 2️⃣ Twilio triggers /voice → stage = "intro".
    # 3️⃣ The system greets the caller and asks for intent (book, cancel, etc.).
    # 4️⃣ If no input: retry up to 3 times.
    # 5️⃣ After 3 silences → redirect to voicemail.
    # 6️⃣ On valid speech or DTMF → proceed to stage="intent".
    # ----------------------------------------------------------------------

    if stage == "intro":
        # ------------------------------------------------------------------
        # 🧠 Initialize or update the session for this call
        # ------------------------------------------------------------------
        # Create or access the session dictionary for this CallSid.
        # Preserve any prior values (e.g., phone number, country, doctor).
        sd = session_data.setdefault(call_sid, {})
        sd["stage"] = "intent"   # Next logical stage after intro

        # ------------------------------------------------------------------
        # 🩺 Log diagnostic info to verify session continuity
        # ------------------------------------------------------------------
        debug_print(f"[intro] ▶️ Call SID → {call_sid}")
        debug_print(f"[intro] 🧭 Next stage set to 'intent'")
        debug_print(f"[intro] Current session keys → {list(sd.keys())}")

        # ------------------------------------------------------------------
        # 🔇 Local silence-handling setup
        # ------------------------------------------------------------------
        silence_key = "intro_silence_count"
        silence_count = sd.get(silence_key, 0)
        debug_print(f"[intro] 🔇 Silence attempt #{silence_count}")

        # ------------------------------------------------------------------
        # 🧾 Retrieve user input (speech or keypad)
        # ------------------------------------------------------------------
        raw_speech = (speech_result or "").strip()
        raw_dtmf = (request.values.get("Digits") or "").strip()
        debug_print(f"[intro] 🎧 Received speech='{raw_speech}' dtmf='{raw_dtmf}'")

        # ------------------------------------------------------------------
        # 🎙️ MENU PROMPT — main message text reused for all re-prompts
        # ------------------------------------------------------------------
        menu_text = (
            "Say 'book appointment' or press 1. "
            "Say 'cancel appointment' or press 2. "
            "Say 'new customer' or press 3. "
            "Say 'change appointment' or press 4. "
            "Say 'update credit card' or press 5. "
            "Say 'update pin number' or press 6. "
            "Say 'update insurance information' or press 7. "
            "Say 'leave voicemail' or press 8."
        )

        # ------------------------------------------------------------------
        # 🔇 Handle silence or missing input
        # ------------------------------------------------------------------
        if not raw_speech and not raw_dtmf:
            silence_count += 1
            sd[silence_key] = silence_count
            debug_print(f"[intro] 🤐 No input detected → retry {silence_count}/3")

            # ==============================================================
            # 🚫 3 Silent Attempts → Transfer to Voicemail
            # ==============================================================
            if silence_count >= 3:
                debug_print("[intro] 🚫 Too many silences → redirecting to voicemail")
                sd.pop(silence_key, None)
                sd["stage"] = "voicemail"

                resp.say(
                    gpt_speak("I’m still not hearing anything. Please leave your message after the beep."),
                    VOICE,
                )
                resp.record(
                    max_length=60,                 # Record up to 60 seconds
                    action="/voice",               # Twilio POST target after recording
                    transcribe=True,               # Enable transcription
                    transcribe_callback="/transcription"
                )
                return str(resp)

            # ==============================================================
            # 🔁 1st & 2nd Silence Attempts — Reprompt with adaptive feedback
            # ==============================================================
            if silence_count == 1:
                # First silence: repeat full greeting politely
                reprompt_text = (
                    "Thank you for calling Epic Therapist Clinic. "
                    + menu_text
                )
            else:
                # Second silence: include acknowledgment before menu
                reprompt_text = (
                    "I couldn’t catch that. Please listen again. "
                    + menu_text
                )

            # Create <Gather> prompt for Twilio
            resp.pause(length=1)
            gather = make_gather(
                reprompt_text,
                input="speech dtmf",
                timeout=8,               # Wait 8 seconds
                speech_timeout="auto",   # Detect silence pause
                barge_in=True,           # Allow interruption
                finish_on_key="#",       # '#' ends keypad input
                num_digits=1             # Expect one digit
            )
            resp.append(gather)
            debug_print(f"[intro] 🔁 Re-prompting user after silence (try {silence_count}/3)")
            return str(resp)

        # ------------------------------------------------------------------
        # ✅ Reset silence counter if input received
        # ------------------------------------------------------------------
        if raw_speech or raw_dtmf:
            sd.pop(silence_key, None)

        # ------------------------------------------------------------------
        # 🎙️ Primary greeting (first-time caller)
        # ------------------------------------------------------------------
        # When user first enters this stage (no silence detected yet),
        # they hear a full welcome message followed by the main options.
        prompt = (
            "Thank you for calling Epic Therapist Clinic. "
            + menu_text
        )

        # ------------------------------------------------------------------
        # 🧩 Build <Gather> TwiML to capture caller intent
        # ------------------------------------------------------------------
        gather = make_gather(
            prompt,
            hints="book,cancel,change,reschedule,update,voicemail",
            input="speech dtmf",
            timeout=8,
            speech_timeout="auto",
            barge_in=True,
            finish_on_key="#",
            num_digits=1
        )

        # ------------------------------------------------------------------
        # 📤 Append <Gather> to Twilio response and redirect after
        # ------------------------------------------------------------------
        resp.append(gather)
        resp.redirect("/voice")

        # ------------------------------------------------------------------
        # ✅ Return TwiML back to Twilio for immediate playback
        # ------------------------------------------------------------------
        return str(resp)





    # 🧩 Stage: INTENT
    # ----------------------------------------------------------------------
    # Functional Description:
    # ----------------------------------------------------------------------
    # • This stage follows immediately after the “intro” greeting.
    # • Its purpose is to understand *what the caller wants to do*.
    # • The caller can respond by **saying** a keyword (e.g. “book appointment”)
    #   or **pressing** a corresponding keypad number (e.g. 1–8).
    # • It maps speech and DTMF input to actions such as:
    #     1️⃣ Book an appointment
    #     2️⃣ Cancel an appointment
    #     3️⃣ New customer registration
    #     4️⃣ Reschedule an appointment
    #     5️⃣ Update credit card
    #     6️⃣ Update PIN number
    #     7️⃣ Update insurance info
    #     8️⃣ Leave a voicemail
    # • Local silence handling:
    #   - Retries the prompt up to 3 times if user says nothing.
    #   - After 3 silences → transfers to voicemail as fallback.
    # • This stage then redirects to the relevant sub-stage (e.g. `collect_phone`).
    # ----------------------------------------------------------------------
    # Flow Summary:
    # ----------------------------------------------------------------------
    # 1️⃣ Receive user input (`SpeechResult` or `Digits`) from Twilio.
    # 2️⃣ Detect silence; re-prompt if no input.
    # 3️⃣ Normalize text (lowercase, strip punctuation).
    # 4️⃣ Identify user intent by keyword or keypad mapping.
    # 5️⃣ Update session → set `stage` and `origin_stage`.
    # 6️⃣ Redirect back to `/voice` for the next step.
    # ----------------------------------------------------------------------


    elif stage == "intent":
        # ------------------------------------------------------------------
        # 🧠 Local silence handling setup
        # ------------------------------------------------------------------
        sd = session_data.setdefault(call_sid, {})
        silence_key = "intent_silence_count"
        silence_count = sd.get(silence_key, 0)

        # Extract and normalize user input
        lower = (speech_result or "").lower().strip()
        debug_print(f"[intent] 🎧 speech_result='{lower}' dtmf='{dtmf_digits}'")

        # ------------------------------------------------------------------
        # 🔇 Handle silence locally (no speech, no DTMF)
        # ------------------------------------------------------------------
        if not lower and not dtmf_digits:
            silence_count += 1
            sd[silence_key] = silence_count
            debug_print(f"[intent] 🤐 No input detected → retry {silence_count}/3")

            if silence_count >= 3:
                # After 3 silences → move to voicemail
                debug_print("[intent] 🚫 Too many silences → redirect to voicemail")
                sd.pop(silence_key, None)
                sd["stage"] = "voicemail"

                resp.say(
                    gpt_speak("I’m still not hearing anything. Please leave your message after the beep."),
                    VOICE,
                )
                resp.record(
                    max_length=MAX_RECORD_TIME,
                    action="/voice",
                    transcribe=True,
                    transcribe_callback="/transcription"
                )
                return str(resp)

            # Reprompt if silence count < 3
            prompt_retry = (
                "I didn’t catch that. "
                "Say 'book appointment' or press 1. "
                "Say 'cancel appointment' or press 2. "
                "Say 'new customer' or press 3. "
                "Say 'change appointment' or press 4. "
                "Say 'update credit card' or press 5. "
                "Say 'update PIN number' or press 6. "
                "Say 'update insurance information' or press 7. "
                "Say 'leave voicemail' or press 8."
            )

            gather = make_gather(
                prompt_retry,
                input="speech dtmf",
                timeout=8,
                speech_timeout="auto",
                barge_in=True,
                num_digits=1
            )
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # Reset silence counter if valid input was received
        sd.pop(silence_key, None)

        # ------------------------------------------------------------------
        # 🔢 Handle keypad or spoken numeric input
        # ------------------------------------------------------------------
        choice = None
        if dtmf_digits and dtmf_digits.isdigit():
            choice = dtmf_digits.strip()
        elif lower in {"1", "2", "3", "4", "5", "6", "7", "8"}:
            choice = lower

        # ------------------------------------------------------------------
        # 🗣️ Handle natural language / keyword input
        # ------------------------------------------------------------------
        # If caller spoke words instead of numbers, detect intent by keyword.
        if any(word in lower for word in ["book", "appointment", "schedule"]):
            choice = "1"
        elif any(word in lower for word in ["cancel", "delete", "remove"]):
            choice = "2"
        elif any(word in lower for word in ["new", "user", "register", "customer"]):
            choice = "3"
        elif any(word in lower for word in ["reschedule", "change", "move"]):
            choice = "4"
        elif any(word in lower for word in ["credit", "card", "payment"]):
            choice = "5"
        elif any(word in lower for word in ["pin", "password", "pin number"]):
            choice = "6"
        elif any(word in lower for word in ["insurance", "health", "medical"]):
            choice = "7"
        elif any(word in lower for word in ["voicemail", "message", "record"]):
            choice = "8"

        # ------------------------------------------------------------------
        # ✅ Route user choice (by speech or DTMF)
        # ------------------------------------------------------------------
        if choice:
            # 1️⃣ Book Appointment
            if choice == "1":
                debug_print("[intent] 📅 Booking flow selected")
                session_data[call_sid].update({
                    "stage": "collect_phone",      # Next step: phone verification
                    "origin_stage": "book",
                    "booking": {},
                    "retry_booking": 0,
                    "retry_time": 0
                })
                prompt = "Please say or enter your ten-digit phone number, then press pound."
                gather = make_gather(
                    prompt, input="speech dtmf", timeout=6,
                    speech_timeout="auto", barge_in=True,
                    finish_on_key="#", num_digits=10
                )
                resp.append(gather)
                resp.redirect("/voice")
                return str(resp)

            # 2️⃣ Cancel Appointment
            if choice == "2":
                debug_print("[intent] ❌ Cancel flow selected")
                session_data[call_sid] = {
                    "stage": "collect_phone",
                    "origin_stage": "cancel",
                    "cancel": {},
                    "retry_booking": 0
                }
                prompt = (
                    "Sure, I can help you cancel your appointment. "
                    "Please say or enter the phone number you used when booking, then press pound."
                )
                gather = make_gather(
                    prompt, input="speech dtmf", num_digits=10,
                    finish_on_key="#", timeout=10,
                    speech_timeout="auto", barge_in=True,
                    language="en-US",
                )
                resp.append(gather)
                resp.redirect("/voice")
                save_session(call_sid)
                return str(resp)

            # 3️⃣ New Customer Registration
            if choice == "3":
                debug_print("[intent] 🧾 Registration flow selected")
                session_data[call_sid].update({
                    "stage": "collect_phone",
                    "origin_stage": "register",
                    "booking": {},
                    "retry_booking": 0,
                    "retry_time": 0
                })
                prompt = (
                    "I can help you register. "
                    "Please say or enter your ten-digit phone number, then press pound."
                )
                gather = make_gather(
                    prompt, input="speech dtmf", timeout=6,
                    speech_timeout="auto", barge_in=True,
                    finish_on_key="#", num_digits=10
                )
                resp.append(gather)
                resp.redirect("/voice")
                return str(resp)

            # 4️⃣ Reschedule Appointment
            if choice == "4":
                debug_print("[intent] 🔁 Reschedule flow selected (cancel + rebook)")
                session_data[call_sid] = {
                    "stage": "collect_phone",
                    "origin_stage": "reschedule",
                    "cancel": {},
                    "retry_booking": 0
                }
                prompt = (
                    "Sure, let's reschedule your appointment. "
                    "We'll cancel your current one first. "
                    "Please say or enter your phone number, then press pound."
                )
                gather = make_gather(
                    prompt, input="speech dtmf", num_digits=10,
                    finish_on_key="#", timeout=10,
                    speech_timeout="auto", barge_in=True,
                    language="en-US",
                )
                resp.append(gather)
                resp.redirect("/voice")
                save_session(call_sid)
                return str(resp)

            # 5️⃣ Update Credit Card
            if choice == "5":
                debug_print("[intent] 💳 Credit card update flow selected")
                session_data[call_sid].update({
                    "stage": "collect_first_name",
                    "origin_stage": "update_cc"
                })
                prompt = "You said you want to update your credit card information. Please hold while we process this request."
                gather = make_gather(
                    prompt, 
                    input="speech dtmf",
                    num_digits=10,
                    finish_on_key="#",
                    timeout=10,
                    speech_timeout="auto", 
                    barge_in=True,
                    language="en-US",
                )
                resp.append(gather)
                resp.redirect("/voice")
                save_session(call_sid)
                return str(resp)

            # 6️⃣ Update PIN
            if choice == "6":
                debug_print("[intent] 🔢 PIN update flow selected")
                session_data[call_sid].update({
                    "stage": "collect_first_name",
                    "origin_stage": "update_pin_number"
                })
                
                prompt = ("You said you want to update your PIN number. "
                          "Please hold while we process this request.")
                gather = make_gather(
                        prompt, 
                        input="speech dtmf",
                        num_digits=10,
                        finish_on_key="#",
                        timeout=10,
                        speech_timeout="auto", 
                        barge_in=True,
                        language="en-US",
                    )
                resp.append(gather)
                resp.redirect("/voice")
                return str(resp)
            

            # 7️⃣ Update Insurance
            if choice == "7":
                debug_print("[intent] 🏥 Insurance update flow selected")
                session_data[call_sid]["stage"] = "update_insurance_information"
                resp.say(gpt_speak(
                    "You said you want to update your insurance information. "
                    "This option is not implemented yet. Please call the clinic for assistance."
                ), VOICE)
                return str(resp)

            # 8️⃣ Voicemail
            if choice == "8":
                debug_print("[intent] 📩 Voicemail flow selected")
                session_data[call_sid]["stage"] = "voicemail"
                resp.say(gpt_speak("Please leave your name, phone number, and message after the beep."), VOICE)
                resp.record(
                    max_length=MAX_RECORD_TIME,
                    action="/voice",
                    transcribe=True,
                    transcribe_callback="/transcription"
                )
                return str(resp)

        # ------------------------------------------------------------------
        # 🚫 Handle junk or greeting input (e.g. "hello", "hi", etc.)
        # ------------------------------------------------------------------
        junk_inputs = {
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
            "yo", "test", "yes", "no"
        }
        if not lower or lower in junk_inputs:
            debug_print(f"[intent] ⛔ Ignored input '{lower}' → re-prompting main menu")
            gather = make_gather(
                "Thank you for calling Epic Therapist. "
                "Say 'book appointment' or press 1. "
                "Say 'cancel appointment' or press 2. "
                "Say 'new customer' or press 3. "
                "Say 'change appointment' or press 4. "
                "Say 'update credit card' or press 5. "
                "Say 'update PIN number' or press 6. "
                "Say 'update insurance information' or press 7. "
                "Say 'leave voicemail' or press 8.",
                hints="book,cancel,change,reschedule,update,credit card,pin number,insurance,voicemail",
                num_digits=1
            )
            resp.append(gather)
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


        """
    🧩 Stage: COLLECT_PHONE
    ----------------------------------------------------------------------
    Functional Description:
    ----------------------------------------------------------------------
    • Purpose:
    Collect and validate the caller’s phone number via speech or DTMF.
    This step is shared by booking, cancel, register, and reschedule flows.

    • Flow:
    1️⃣ Play prompt asking for a 10-digit phone number.
    2️⃣ Accept speech (“two one four...”) or keypad input (“2145552671”).
    3️⃣ Convert speech → digits → normalized E.164 number (+14155552671).
    4️⃣ If valid → store in session → advance to next stage (`collect_dob`).
    5️⃣ If silence or invalid input → retry up to 3 times.
    6️⃣ After 3 failures → end politely.

    • Local Silence Handling:
    - Tracks attempts in `silence_collect_phone`.
    - Reprompts politely up to 2 times.
    - On 3rd silence → apologizes and hangs up.
    ----------------------------------------------------------------------
    """

    elif stage == "collect_phone":
        debug_print("[collect_phone] 📍 Stage entered")

        # ------------------------------------------------------------------
        # 💬 Voice messages for reuse and localization
        # ------------------------------------------------------------------
        VOICE_NO_INPUT_MSG = (
            "I didn’t hear your phone number. Please say or enter your ten-digit number, then press pound."
        )
        VOICE_TOO_MANY_SILENCES_MSG = (
            "I'm sorry, I still didn't get your phone number. Please call again later."
        )
        VOICE_INVALID_PHONE_MSG = (
            "That doesn’t sound complete. Please say or enter your ten-digit phone number including area code, then press pound."
        )
        VOICE_TOO_MANY_INVALID_MSG = (
            "I'm sorry, I couldn’t capture your phone number. Please call again later."
        )
        VOICE_ASK_DOB_MSG = (
            "Thanks. What’s your date of birth? You can say it, or enter two digits for month, "
            "two for day, and four for year, then press pound."
        )

        # ------------------------------------------------------------------
        # 🔁 Initialize session safely without overwriting existing keys
        # ------------------------------------------------------------------
        sd = session_data.setdefault(call_sid, {})
        cust = sd.setdefault("customer", {})
        cancel_ctx = sd.setdefault("cancel", {})
        debug_print(f"[collect_phone] session keys before: {list(sd.keys())}")

        # ------------------------------------------------------------------
        # 🌎 Infer caller’s country code for phone normalization
        # ------------------------------------------------------------------
        if "phone_country" not in sd:
            from_country = (request.values.get("FromCountry") or "").upper()
            sd["phone_country"] = from_country or (COUNTRY or "US")
            debug_print(f"[collect_phone] 🌐 phone_country={sd['phone_country']}")

        # ------------------------------------------------------------------
        # 🗣 Capture user input (SpeechResult or DTMF)
        # ------------------------------------------------------------------
        dtmf_digits = (request.values.get("Digits") or "").strip()
        speech_text = (speech_result or "").strip()
        debug_print(f"[collect_phone] 🗣 speech='{speech_text}' 🔢 DTMF='{dtmf_digits}'")

        # ------------------------------------------------------------------
        # 🔇 Local silence handling — no input received
        # ------------------------------------------------------------------
        if not (speech_text or dtmf_digits):
            tries = sd.get("silence_collect_phone", 0) + 1
            sd["silence_collect_phone"] = tries
            debug_print(f"[collect_phone] 🤐 No input (tries={tries}/3)")

            if tries < 3:
                # Reprompt politely
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

            # Too many silences → terminate call politely
            resp.say(gpt_speak(VOICE_TOO_MANY_SILENCES_MSG), VOICE)
            resp.hangup()
            save_session(call_sid)
            session_data.pop(call_sid, None)
            return str(resp)

        # Reset silence counter after valid input
        sd.pop("silence_collect_phone", None)

        # ------------------------------------------------------------------
        # 🔢 Convert spoken numbers to digits
        # ------------------------------------------------------------------
        def _spoken_to_digits(raw: str) -> str:
            """
            Convert spoken number words like “two one four double five”
            into numeric string “21455”.
            Supports “double”/“triple” modifiers.
            """
            if not raw:
                return ""
            # Normalize and split input text
            words = (
                raw.lower()
                .replace("-", " ").replace(",", " ").replace(".", " ")
                .replace("(", " ").replace(")", " ").split()
            )
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
                # Handle "double five" / "triple six"
                if w in ("double", "triple") and i + 1 < len(words):
                    nxt = words[i + 1]
                    if nxt in mapping:
                        out.extend([mapping[nxt]] * (2 if w == "double" else 3))
                        i += 2
                        continue
                # Standard mapping or raw digits
                if w in mapping:
                    out.append(mapping[w])
                else:
                    out.extend([c for c in w if c.isdigit()])
                i += 1
            return "".join(out)

        # Combine inputs: prefer DTMF digits else convert speech
        raw_digits = _re.sub(r"\D", "", dtmf_digits or _spoken_to_digits(speech_text))
        debug_print(f"[collect_phone] 🔍 raw_digits='{raw_digits}'")

        # ------------------------------------------------------------------
        # 🌐 Normalize to E.164 format (e.g., +14155552671)
        # ------------------------------------------------------------------
        country = sd.get("phone_country", (COUNTRY or "US")).upper()
        try:
            phone_e164 = normalize_phone_e164(raw_digits, country)
            debug_print(f"[collect_phone] ✅ normalized → {phone_e164}")
        except Exception as e:
            debug_print(f"[collect_phone] ⚠️ normalize_phone_e164 failed: {e}")
            d = raw_digits
            if country == "US":
                if len(d) == 11 and d.startswith("1"):
                    d = d[1:]
                phone_e164 = f"+1{d}" if len(d) == 10 else ""
            else:
                phone_e164 = ""
            debug_print(f"[collect_phone] ⚙️ fallback normalize → '{phone_e164}'")

        # ------------------------------------------------------------------
        # ❌ Retry for invalid / incomplete phone numbers
        # ------------------------------------------------------------------
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

            # Too many invalid attempts → end call
            resp.say(gpt_speak(VOICE_TOO_MANY_INVALID_MSG), VOICE)
            resp.hangup()
            save_session(call_sid)
            session_data.pop(call_sid, None)
            return str(resp)

        # ------------------------------------------------------------------
        # ✅ Save valid phone number across session contexts
        # ------------------------------------------------------------------
        cust["phone_e164"] = phone_e164
        cust["phone"] = phone_e164
        cancel_ctx["phone_e164"] = phone_e164
        sd["phone_e164"] = phone_e164
        sd["retry_phone"] = 0
        debug_print(f"[collect_phone] 💾 saved phone_e164={phone_e164}")

       

        # ------------------------------------------------------------------
        # 🗓 Proceed normally → next step is DOB collection
        # ------------------------------------------------------------------
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

        # Persist session state for next webhook call
        save_session(call_sid)
        return str(resp)








  
    # ==========================================================================
    # 🎂 Stage: collect_dob — Capture and Validate Customer Date of Birth
    # ==========================================================================
    #
    # FUNCTIONAL OVERVIEW:
    # --------------------
    # • Purpose: To capture and validate the caller's date of birth (DOB)
    #   using speech or keypad (DTMF) input.
    #
    # • Inputs:
    #     - SpeechResult (spoken date, e.g. “July 3rd 1972”)
    #     - Digits (numeric input, e.g. “07031972”)
    #
    # • Flow:
    #     1️⃣ Handle silence with up to 3 polite retries.
    #     2️⃣ Parse DOB using keypad (MMDDYYYY) or speech recognition.
    #     3️⃣ Validate DOB is between 1900–today.
    #     4️⃣ Lookup the customer in customers.json (via phone + DOB).
    #     5️⃣ Route based on origin_stage:
    #         - "register"      → collect_first_name
    #         - "book"          → collect_pin_number
    #         - "cancel"        → collect_pin_number
    #         - "reschedule"    → collect_pin_number
    #         - "update_cc"     → collect_cc
    #         - "update_pin_number" → collect_pin_number
    #     6️⃣ Any unknown condition → polite fallback and hangup.
    #
    # ==========================================================================




    elif stage == "collect_dob":
    
        # Record start time for profiling
        t_stage_start = _time_mod.perf_counter()
        debug_print(f"[collect_dob] 📍 Stage entered at {_time_mod.strftime('%H:%M:%S')}")

        # ----------------------------------------------------------------------
        # 💬 VOICE PROMPTS — centralized for easy localization
        # ----------------------------------------------------------------------
        VOICE_SILENCE_MSG = (
            "Please say your date of birth, for example, 'July 3 1972'. "
            "Or enter two digits for month, two for day, and four for year, then press pound."
        )
        VOICE_SILENCE_FINAL_MSG = (
            "Sorry, I couldn’t get your date of birth. Please call again later."
        )
        VOICE_PARSE_FAIL_MSG = (
            "I didn’t catch your full birth date. Please say it again, for example, 'July 3 1972'. "
            "You can also enter it using your keypad: 2 digits for month, 2 for day, and 4 for year, then press pound."
        )
        VOICE_INVALID_DOB_MSG = (
            "That doesn’t seem like a valid date of birth. "
            "Please enter 2 digits for month, 2 for day, and 4 for year, then press pound."
        )
        VOICE_NOT_FOUND_MSG = (
            "We could not find your record. "
            "You must register first as a new customer with the clinic before booking an appointment."
        )
        VOICE_NEW_CUSTOMER_MSG = (
            "We found your record, but your registration with the clinic is not complete. "
            "Please contact the clinic to finish your registration before booking an appointment. Goodbye!"
        )
        VOICE_PIN_PROMPT_MSG = (
            "Thank you. For security verification, please enter your six digit PIN number now, "
            "followed by the pound key. If you prefer, you can also say each digit slowly."
        )
        VOICE_REGISTER_ROUTE_MSG = (
            "Let's start your registration. Please say or enter your first name and press pound."
        )

        # ----------------------------------------------------------------------
        # 🛡️ Ensure the session dictionary exists
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
        # 🎧 Capture inputs (speech or keypad)
        # ----------------------------------------------------------------------
        dtmf_digits = (request.values.get("Digits") or "").strip()
        speech_text = (speech_result or "").strip()
        debug_print(f"[collect_dob] 🎙️ speech='{speech_text}', 🔢 dtmf='{dtmf_digits}'")

        # ----------------------------------------------------------------------
        # 🔇 Handle silence (no speech or keypad input)
        # ----------------------------------------------------------------------
        if not dtmf_digits and not speech_text:
            tries = sd.get("silence_dob", 0) + 1
            sd["silence_dob"] = tries
            debug_print(f"[collect_dob] 🤐 silence tries={tries}/3")

            # Retry up to 2 times with gentle re-prompts
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
                resp.redirect("/voice")
                return str(resp)

            # After 3 silent attempts → hang up politely
            resp.say(gpt_speak(VOICE_SILENCE_FINAL_MSG), VOICE)
            resp.hangup()
            debug_print(f"[collect_dob] 🧹 clearing session after hangup (call_sid={call_sid})")
            session_data.pop(call_sid, None)
            return str(resp)

        # ✅ Reset silence counter on valid input
        sd.pop("silence_dob", None)

        # ----------------------------------------------------------------------
        # 🧩 Parse DOB from input (DTMF or speech)
        # ----------------------------------------------------------------------
        dob_date = None

        # 🧮 Case 1: Numeric input via DTMF keypad (MMDDYYYY)
        if dtmf_digits:
            d = _re.sub(r"\D", "", dtmf_digits)
            if len(d) >= 8:
                try:
                    mm, dd, yyyy = int(d[0:2]), int(d[2:4]), int(d[4:8])
                    dob_date = date(yyyy, mm, dd)
                    debug_print("[collect_dob] ✅ parsed DOB from keypad")
                except Exception as e:
                    debug_print(f"[collect_dob] ❌ keypad parse error → {e}")

        # 🗣️ Case 2: Parse spoken date (e.g., “July 3rd, 1972”)
        if not dob_date and speech_text:
            try:
                # Clean punctuation and normalize ordinals ("3rd" → "3")
                t = _re.sub(r"[.,;:]+$", "", speech_text)
                t = _re.sub(r"[,\.;:]", " ", t)
                t = _re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", t, flags=_re.IGNORECASE)
                t = _re.sub(r"\s+", " ", t).strip()

                parsed = _dtparse(t, fuzzy=True)
                dob_date = date(parsed.year, parsed.month, parsed.day)
                debug_print("[collect_dob] ✅ parsed DOB from speech")
            except Exception as e:
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
            resp.redirect("/voice")
            return str(resp)

        # ✅ Save DOB in session
        iso_dob = dob_date.strftime("%Y-%m-%d")
        sd["customer"]["dob"] = iso_dob
        sd["cancel"]["dob"] = iso_dob
        debug_print(f"[collect_dob] ✅ Stored DOB → {iso_dob}")

        # ----------------------------------------------------------------------
        # 🔍 Lookup customer in local DB (via phone + DOB)
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
        # 🔀 Route based on origin_stage and customer_status
        # ----------------------------------------------------------------------
        origin_stage = sd.get("origin_stage", "").strip().lower()
        debug_print(f"[collect_dob] 🔁 origin_stage={origin_stage}, found={found}, customer_status={customer_status}")

        # 1️⃣ Registration flow → ask for first name
        if origin_stage == "register":
            sd["stage"] = "collect_first_name"
            g = make_gather(
                VOICE_REGISTER_ROUTE_MSG,
                input="speech dtmf",
                timeout=6,
                speech_timeout="auto",
                barge_in=True,
                finish_on_key="#"
            )
            resp.append(g)
            resp.redirect("/voice")
            debug_print("[collect_dob] 🔁 origin_stage=register → collect_first_name")
            return str(resp)

        # 2️⃣ Booking or Cancel flow → customer not found → must register
        if origin_stage in ("book", "cancel") and not found:
            resp.say(gpt_speak(VOICE_NOT_FOUND_MSG), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            debug_print(f"[collect_dob] ❌ {origin_stage} flow → user not found → ask to register")
            return str(resp)

        # 3️⃣ Booking or Cancel flow → incomplete registration
        if origin_stage in ("book", "cancel") and customer_status == "new":
            resp.say(gpt_speak(VOICE_NEW_CUSTOMER_MSG), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            debug_print(f"[collect_dob] 🟡 {origin_stage} flow → incomplete registration → hangup")
            return str(resp)

        # 4️⃣ Booking, Cancel, or Reschedule → current customer → verify via PIN
        if origin_stage in ("book", "cancel", "reschedule") and customer_status == "current":
            sd["stage"] = "collect_pin_number"
            g = make_gather(
                VOICE_PIN_PROMPT_MSG,
                input="speech dtmf",
                timeout=6,
                speech_timeout="auto",
                barge_in=True,
                finish_on_key="#"
            )
            resp.append(g)
            resp.redirect("/voice")
            debug_print(f"[collect_dob] ✅ {origin_stage} flow → current user → collect_pin_number")
            return str(resp)

        # 5️⃣ Update credit card → skip PIN → go to collect_cc
        if origin_stage == "update_cc" and customer_status == "current":
            sd["stage"] = "collect_cc"
            g = make_gather(
                "Please enter or say your credit card number, then press pound.",
                input="speech dtmf",
                timeout=8,
                speech_timeout="auto",
                barge_in=True,
                finish_on_key="#"
            )
            resp.append(g)
            resp.redirect("/voice")
            debug_print("[collect_dob] ✅ update_cc → current user → collect_cc")
            return str(resp)

        # 6️⃣ Update PIN number → verify identity first
        if origin_stage == "update_pin_number" and customer_status == "current":
            sd["stage"] = "collect_pin_number"
            g = make_gather(
                "Let's verify your identity. Please enter your existing six digit PIN, followed by the pound key.",
                input="speech dtmf",
                timeout=6,
                speech_timeout="auto",
                barge_in=True,
                finish_on_key="#"
            )
            resp.append(g)
            resp.redirect("/voice")
            debug_print("[collect_dob] ✅ update_pin_number → verified customer → collect_pin_number")
            return str(resp)

        # ----------------------------------------------------------------------
        # 🚨 Fallback — unexpected combination
        # ----------------------------------------------------------------------
        resp.say(gpt_speak("I could not determine your registration status. Please call the clinic for assistance."), VOICE)
        resp.hangup()
        debug_print("[collect_dob] ⚠️ fallback → unexpected condition")
        session_data.pop(call_sid, None)
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
            "Now let's collect your insurance information. "
            "You can choose your insurance company by pressing a number on your keypad, "
            "or by saying the company name. I will list them now. "
        )
        MSG_PROMPT_MEMBER_ID = (
            "Please say or enter your insurance member ID or policy number now. "
            "You can include both letters and numbers, then press pound when done."
        )
        MSG_AFTER_SELECTION = (
            "Thank you. You selected {insurance_name}. "
            "Now please say or enter your insurance member ID or policy number. "
            "You can include both letters and numbers, then press pound when done."
        )
        # Make this neutral (no 'confirm your appointment' phrasing)
        MSG_THANK_YOU_NEXT = (
            "Thank you. Your insurance information has been saved. "
            "We will now continue with your registration."
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
        raw_dtmf   = (request.values.get("Digits") or "").strip()
        debug_print(f"collect_insurance_information: speech='{raw_speech}', dtmf='{raw_dtmf}'")

        # Normalized version for fuzzy matching
        spoken_norm = _re.sub(r"[^a-z0-9 ]", " ", (raw_speech or "").lower())
        spoken_norm = _re.sub(r"\s+", " ", spoken_norm).strip()
        debug_print(f"collect_insurance_information: spoken_norm='{spoken_norm}'")

        # ----------------------------------------------------------------------
        # 🏢 Load insurance companies from GLOBAL (configured at top of file)
        # ----------------------------------------------------------------------
        # Expected global declaration somewhere near the top of file:
        # INSURANCE_COMPANIES_LIST = [
        #     name.strip() for name in os.getenv(
        #         "INSURANCE_COMPANIES",
        #         "Blue Cross Blue Shield,Aetna,Cigna,United Healthcare,Humana,Kaiser Permanente"
        #     ).split(",")
        # ]
        global INSURANCE_COMPANIES_LIST
        INSURANCE_COMPANIES_LIST = [
            name.strip() for name in os.getenv(
                "INSURANCE_COMPANIES",
                "Blue Cross Blue Shield,Aetna,Cigna,United Healthcare,Humana,Kaiser Permanente"
            ).split(",")
            if name.strip()
        ]

        # Map: "1" → first company, "2" → second, etc.
        keypad_map = {str(i + 1): name for i, name in enumerate(INSURANCE_COMPANIES_LIST)}
        debug_print(f"collect_insurance_information: keypad_map={keypad_map}")

        # ----------------------------------------------------------------------
        # 🤖 Fuzzy alias map for speech → canonical company name
        # ----------------------------------------------------------------------
        # We handle common mis-hearings (e.g., 'Aetna' → 'if no', 'et na', etc.).
        alias_map = {}
        for name in INSURANCE_COMPANIES_LIST:
            key = name.lower()
            aliases = [key]

            # Special handling for "Aetna" (commonly mis-recognized)
            if "aetna" in key:
                aliases.extend([
                    "aetna",          # correct
                    "etna",           # dropped "a"
                    "et na",          # spaced
                    "edna",           # common mis-hear
                    "if no",          # what STT produced in your log
                    "ifno",
                    "eight na",
                    "eight now",
                ])

            # You can add similar blocks for other companies if STT is bad
            # (e.g., "blue cross blue shield" having "blue shield", etc.)

            alias_map[key] = list({a.strip() for a in aliases if a.strip()})

        debug_print(f"collect_insurance_information: alias_map={alias_map}")

        # ----------------------------------------------------------------------
        # 🧩 Determine current sub-step ("company" or "id")
        # ----------------------------------------------------------------------
        step = sd.get("insurance_step", "company")
        origin_stage = (sd.get("origin_stage") or "").lower()

        # ======================================================================
        # 🧩 STEP 1 — SELECT INSURANCE COMPANY (keypad OR voice)
        # ======================================================================
        if step == "company":
            # --------------------------------------------------------------
            # 1️⃣ Handle keypad (DTMF) input
            # --------------------------------------------------------------
            if raw_dtmf:
                first_digit = next((ch for ch in raw_dtmf if ch in keypad_map), "")
                if first_digit:
                    insurance_name = keypad_map[first_digit]
                    customer["insurance_name"] = insurance_name
                    sd["insurance_step"] = "id"
                    debug_print(f"✅ Selected insurance_name='{insurance_name}' via DTMF '{raw_dtmf}'")

                    g = make_gather(
                        MSG_AFTER_SELECTION.format(insurance_name=insurance_name),
                        input="speech dtmf",
                        timeout=3,
                        speech_timeout="auto",
                        barge_in=False,
                        finish_on_key="#",
                        language="en-US",
                        action="/voice",
                        method="POST",
                    )
                    resp.append(g)
                    return str(resp)

            # --------------------------------------------------------------
            # 2️⃣ Handle voice selection of company name
            # --------------------------------------------------------------
            selected_name = None
            if spoken_norm:
                # First pass: alias map exact/contains match
                for canon_lower, aliases in alias_map.items():
                    for alias in aliases:
                        # match whole phrase or contained as a word group
                        if (spoken_norm == alias) or (f" {alias} " in f" {spoken_norm} "):
                            # Find original display name
                            for display in INSURANCE_COMPANIES_LIST:
                                if display.lower() == canon_lower:
                                    selected_name = display
                                    break
                            if selected_name:
                                break
                    if selected_name:
                        break

                # Second pass (fallback): simple token overlap like before
                if not selected_name:
                    for name in INSURANCE_COMPANIES_LIST:
                        tokens = [t for t in name.lower().split() if len(t) > 2]
                        if any(t in spoken_norm for t in tokens):
                            selected_name = name
                            break

            if selected_name:
                customer["insurance_name"] = selected_name
                sd["insurance_step"] = "id"
                debug_print(f"✅ Selected insurance_name='{selected_name}' via speech='{raw_speech}'")

                g = make_gather(
                    MSG_AFTER_SELECTION.format(insurance_name=selected_name),
                    input="speech dtmf",
                    timeout=8,
                    speech_timeout="auto",
                    barge_in=False,
                    finish_on_key="#",
                    language="en-US",
                    action="/voice",
                    method="POST",
                )
                resp.append(g)
                return str(resp)
            else:
                if raw_speech:
                    debug_print("collect_insurance_information: ❌ speech did not match any company")

            # --------------------------------------------------------------
            # 3️⃣ Silence / invalid input handling (max 3 tries)
            # --------------------------------------------------------------
            if not raw_dtmf and not raw_speech:
                tries = sd.get("insurance_silence_tries", 0) + 1
                sd["insurance_silence_tries"] = tries
                debug_print(f"collect_insurance_information: 🤐 company silence tries={tries}/3")
            else:
                # Speech present but not matched → treat as invalid attempt
                tries = sd.get("insurance_invalid_tries", 0) + 1
                sd["insurance_invalid_tries"] = tries
                debug_print(f"collect_insurance_information: ❌ invalid company selection tries={tries}/3")

            if (sd.get("insurance_silence_tries", 0) >= 3 or
                sd.get("insurance_invalid_tries", 0) >= 3):
                resp.say(gpt_speak(MSG_SILENCE_EXIT), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # --------------------------------------------------------------
            # 4️⃣ Re-prompt with company menu (DTMF + speech)
            # --------------------------------------------------------------
            menu_text = MSG_PROMPT_INSURANCE_COMPANY
            for i, name in enumerate(INSURANCE_COMPANIES_LIST, start=1):
                # “Press 1 or say Blue Cross Blue Shield. Press 2 or say Aetna. ...”
                menu_text += f"For {name}, press {i} or say {name}. "

            g = make_gather(
                menu_text,
                input="speech dtmf",
                timeout=10,
                speech_timeout="auto",
                barge_in=True,
                finish_on_key="#",
                language="en-US",
                action="/voice",
                method="POST",
            )
            resp.append(g)
            return str(resp)

        # ======================================================================
        # 🧩 STEP 2 — COLLECT INSURANCE MEMBER ID / POLICY NUMBER
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
                    resp.say(gpt_speak(MSG_MEMBERID_SILENCE_EXIT), VOICE)
                    resp.hangup()
                    session_data.pop(call_sid, None)
                    return str(resp)

                g = make_gather(
                    MSG_PROMPT_MEMBER_ID,
                    input="speech dtmf",
                    timeout=30,
                    speech_timeout="auto",
                    barge_in=False,
                    finish_on_key="#",
                    language="en-US",
                    action="/voice",
                    method="POST",
                )
                resp.append(g)
                #resp.redirect("/voice")
                return str(resp)

            # --------------------------------------------------------------
            # 🧾 Capture and save insurance member ID / policy number
            # --------------------------------------------------------------
            member_id = (raw_dtmf or raw_speech).strip().upper()
            customer["insurance_member_id"] = member_id
            debug_print(f"✅ Captured insurance_member_id='{member_id}'")

            # Cleanup step state
            sd.pop("insurance_step", None)
            sd.pop("insurance_silence_tries", None)
            sd.pop("insurance_invalid_tries", None)
            sd.pop("insurance_id_silence", None)

            # Set next stage to booking confirmation or registration completion;
            # book_appt_confirm will decide based on customer_status ('register', 'new', 'current').
            sd["stage"] = "book_appt_confirm"

            g = make_gather(
                MSG_THANK_YOU_NEXT,
                input="speech dtmf",
                timeout=4,
                speech_timeout="2",
                barge_in=True,
                language="en-US",
                action="/voice",
                method="POST",
            )
            resp.append(g)
            resp.redirect("/voice")
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

    elif stage == "collect_pin_number":
        # ----------------------------------------------------------------------
        # 🎯 Stage: collect_pin_number
        #
        # PURPOSE:
        #   • Capture and verify caller’s 6-digit PIN via DTMF or speech.
        #   • Determine if caller is booking, canceling, or updating info.
        #   • Ensure origin_stage (“book”, “cancel”, etc.) is retained.
        #
        # FLOW:
        #   collect_phone → collect_dob → collect_pin_number
        #       → collect_dr_info (origin_stage=cancel)
        #
        # ----------------------------------------------------------------------

        debug_print("collect_pin_number: 📍 Stage entered")

        # ----------------------------------------------------------------------
        # 💬 Voice prompts — centralized for localization and clarity
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

        # ✅ Ensure origin_stage persists — do NOT overwrite if already set
        if "origin_stage" not in sd:
            # If cancel structure exists → inherit "cancel", else default to "book"
            sd["origin_stage"] = "cancel" if "cancel" in sd else "book"

        origin_stage = sd.get("origin_stage", "book").strip().lower()
        debug_print(f"collect_pin_number: 🔎 origin_stage={origin_stage}")

        # Retrieve caller identifiers
        phone_e164 = (customer.get("phone_e164") or sd.get("phone_e164") or "").strip()
        dob = (customer.get("dob") or "").strip()

        # Capture inputs from Twilio <Gather>
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
                resp.say(gpt_speak(VOICE_SILENCE_TERMINATE_MSG), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

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

        # Reset silence counter
        sd.pop("silence_pin", None)

        # ======================================================================
        # 🔢 PIN PARSING
        # ======================================================================
        # Extract only numeric digits from the input
        digits = _re.sub(r"\D", "", raw_dtmf or raw_speech)
        debug_print(f"collect_pin_number: normalized digits='{digits}'")

        if len(digits) != 6:
            debug_print("collect_pin_number: ⚠️ invalid PIN length")
            sd["pin_attempts"] = sd.get("pin_attempts", 0) + 1

            if sd["pin_attempts"] >= 3:
                resp.say(gpt_speak(VOICE_TOO_MANY_INVALID_MSG), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

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

            sd.pop("pin_attempts", None)

            # Dynamically decide next stage based on origin_stage
            if origin_stage == "book":
                next_stage = "collect_dr_info"
                msg = VOICE_CORRECT_PIN_BOOK_MSG
            elif origin_stage in ("cancel", "reschedule"):
                next_stage = "collect_dr_info"
                msg = VOICE_CORRECT_PIN_CANCEL_MSG
            elif origin_stage == "update_cc":
                next_stage = "collect_cc"
                msg = VOICE_CORRECT_PIN_CC_MSG
            else:
                next_stage = "intro"
                msg = VOICE_CORRECT_PIN_DEFAULT_MSG

            # ✅ Update session with next stage and skip silence for next round
            sd["stage"] = next_stage
            sd["skip_silence_once"] = True

            # Debug output for full session trace
            debug_print(f"collect_pin_number: 🔀 Transition → origin_stage={origin_stage} → next_stage={next_stage}")
            try:
                debug_print(f"collect_pin_number: 🧾 Session snapshot → {json.dumps(sd, indent=2)}")
            except Exception as e:
                debug_print(f"collect_pin_number: ⚠️ Could not serialize session → {e}")

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
    # 📅 Stage: collect_book_time_date
    # ======================================================================
    # 🎯 FUNCTIONAL PURPOSE:
    #   This stage receives ANY spoken or typed date/time from the caller.
    #   Responsibilities:
    #
    #    1) Detect silence → re-ask or end after retries
    #    2) Parse free-form speech like:
    #          “August 17 at 5 PM”
    #          “Next Friday morning”
    #          “Tomorrow at 3”
    #    3) Convert caller-local time → UTC automatically
    #    4) Check if:
    #          • date is in the past
    #          • outside booking horizon
    #          • not within working hours
    #          • the doctor is unavailable
    #    5) If invalid → suggest 1–3 alternate time slots
    #    6) Store these alternatives in session_data["alts_list"]
    #    7) Redirect to new stage `confirm_time_choice`
    #
    #   This stage DOES NOT ask for confirmation. It only:
    #       ► parses
    #       ► validates
    #       ► proposes alternative slots
    #
    #   Confirmation happens in NEW STAGE: `confirm_time_choice`
    # ======================================================================
    elif stage == "collect_book_time_date":

        # ==================================================================
        # 🗣️ ALL VOICE MESSAGES — MOVED TO TOP (VARIABLES)
        # ==================================================================
        MSG_NEED_DOCTOR = (
            "Before choosing a time, please tell me which doctor you want to see."
        )
        MSG_SILENCE_REPROMPT = (
            "Please say the appointment date and time, for example 'October 10 at 9 A M'."
        )
        MSG_SILENCE_EXIT = (
            "I didn’t hear anything. Please call us back later."
        )
        MSG_PARSE_FAIL = (
            "I could not understand the time. Please say it again."
        )
        MSG_PARSE_FATAL = (
            "Sorry, I could not understand the time. Goodbye."
        )
        MSG_TIME_INVALID = (
            "That time is not available. Here are the next available appointments."
        )
        MSG_TIME_UNAVAILABLE = (
            "That time is not available. Let me suggest alternatives."
        )
        MSG_NO_ALTS = (
            "Sorry, there are no available times."
        )
        MSG_NO_ALTS_SOON = (
            "Sorry, there are no available times soon."
        )

        debug_print(f"[collect_book_time_date] 🗣️ Incoming speech='{speech_result}'")

        # ------------------------------------------------------------------
        # 🧠 LOAD SESSION DATA (create if missing)
        # ------------------------------------------------------------------
        sd = session_data.setdefault(call_sid, {})
        sd.setdefault("stage", "collect_book_time_date")

        # ------------------------------------------------------------------
        # 🩺 ENSURE DOCTOR IS SELECTED BEFORE CHOOSING A TIME
        # ------------------------------------------------------------------
        doctor_name = sd.get("doctor_name")
        if not doctor_name:
            debug_print("[collect_book_time_date] ❗ No doctor selected before time entry")

            # Prompt for doctor name again
            g = make_gather(
                MSG_NEED_DOCTOR,
                input="speech dtmf",
                timeout=8,
                action="/voice"
            )
            resp.append(g)

            # Move session to doctor-collection step
            sd["stage"] = "collect_dr_info"
            save_session(call_sid)
            return str(resp)

        # ------------------------------------------------------------------
        # 🔇 SILENCE HANDLING — NO SPEECH & NO DIGITS
        # ------------------------------------------------------------------
        if not speech_result and not request.values.get("Digits"):

            # Track retry attempts
            sd["silence_retry"] = sd.get("silence_retry", 0) + 1
            debug_print(f"[collect_book_time_date] 🔇 Silence (retry={sd['silence_retry']})")

            # Too many silence attempts → exit
            if sd["silence_retry"] >= 2:
                resp.say(gpt_speak(MSG_SILENCE_EXIT), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Reprompt for date & time
            g = make_gather(
                MSG_SILENCE_REPROMPT,
                input="speech dtmf",
                timeout=8,
                action="/voice"
            )
            resp.append(g)
            save_session(call_sid)
            return str(resp)

        # ------------------------------------------------------------------
        # 🧠 EXTRACT RAW INPUT (speech or DTMF)
        # ------------------------------------------------------------------
        raw = (speech_result or request.values.get("Digits") or "").strip()
        debug_print(f"[collect_book_time_date][parse] RAW='{raw}'")

        # ------------------------------------------------------------------
        # 🧠 TRY PARSING USING smart_parse_time()
        # ------------------------------------------------------------------
        try:
            parsed = smart_parse_time(raw)
        except Exception as e:
            parsed = None
            debug_print(f"[collect_book_time_date] ❌ smart_parse_time raised → {e}")

        # ------------------------------------------------------------------
        # ❌ PARSE FAILED → REPROMPT
        # ------------------------------------------------------------------
        if not parsed:

            sd["retry_time"] = sd.get("retry_time", 0) + 1

            # Too many retries → hang up
            if sd["retry_time"] >= 3:
                resp.say(gpt_speak(MSG_PARSE_FATAL), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Ask again
            g = make_gather(
                MSG_PARSE_FAIL,
                input="speech dtmf",
                timeout=8,
                action="/voice"
            )
            resp.append(g)
            save_session(call_sid)
            return str(resp)

        # ==================================================================
        # 🎉 PARSED SUCCESSFULLY — EXTRACT VALUES
        # ==================================================================
        appointment_start = parsed["start"]      # ISO UTC start
        appointment_end   = parsed["end"]        # ISO UTC end
        friendly          = parsed["friendly"]   # Human-friendly
        is_past           = parsed["is_past"]    # Boolean

        now_utc = _pytz.UTC.localize(_dt.utcnow())
        booking_end_limit = now_utc + timedelta(days=30 * MAX_ADVANCE_MONTHS)

        # ------------------------------------------------------------------
        # 🚫 INVALID RANGE (past OR beyond allowed window)
        # ------------------------------------------------------------------
        if (
            is_past or
            isoparse(appointment_start) <= now_utc or
            isoparse(appointment_start) > booking_end_limit
        ):
            resp.say(gpt_speak(MSG_TIME_INVALID), VOICE)

            # Get next available slots (3 options)
            alts = get_doctor_next_available_slots(
                doctor_name,
                from_start_iso=now_utc.isoformat(),
                limit=3
            )

            if not alts:
                resp.say(gpt_speak(MSG_NO_ALTS_SOON), VOICE)
                resp.hangup()
                return str(resp)

            # ✅ Prepare state for confirm_time_choice (fresh start)
            sd["alts_list"] = alts
            sd["stage"] = "confirm_time_choice"
            sd["confirm_mode"] = False         # not yet asking YES/NO
            sd["alts_spoken"] = False          # alternatives not spoken yet
            sd["silence_retry"] = 0            # reset silence counter
            sd["retry_count"] = 0              # reset confirm retries
            sd["retry_time"] = 0               # reset time-parse retries

            save_session(call_sid)
            resp.redirect("/voice", method="POST")
            return str(resp)

        # ------------------------------------------------------------------
        # ❌ SLOT NOT AVAILABLE — CHECK BY DOCTOR CALENDAR
        # ------------------------------------------------------------------
        if not is_doctor_slot_available(doctor_name, appointment_start, appointment_end):

            resp.say(gpt_speak(MSG_TIME_UNAVAILABLE), VOICE)

            alts = get_doctor_next_available_slots(
                doctor_name,
                from_start_iso=now_utc.isoformat(),
                limit=3
            )

            if not alts:
                resp.say(gpt_speak(MSG_NO_ALTS), VOICE)
                resp.hangup()
                return str(resp)

            # ✅ Prepare state for confirm_time_choice (fresh start)
            sd["alts_list"] = alts
            sd["stage"] = "confirm_time_choice"
            sd["confirm_mode"] = False
            sd["alts_spoken"] = False
            sd["silence_retry"] = 0
            sd["retry_count"] = 0
            sd["retry_time"] = 0

            save_session(call_sid)
            resp.redirect("/voice", method="POST")
            return str(resp)

        # ------------------------------------------------------------------
        # 🟩 SUCCESS — TIME IS PARSED & AVAILABLE
        # ------------------------------------------------------------------
        sd["appointment_time"] = {
            "start": appointment_start,
            "end": appointment_end,
            "friendly": friendly,
        }
        sd["stage"] = "confirm_time_choice"

        # ✅ Reset confirm-time state so confirm_time_choice starts clean
        sd["confirm_mode"] = False
        sd["alts_list"] = []
        sd["alts_spoken"] = False
        sd["silence_retry"] = 0
        sd["retry_count"] = 0
        sd["retry_time"] = 0

        # Redirect to next stage
        save_session(call_sid)
        resp.redirect("/voice", method="POST")
        return str(resp)




   
    # ======================================================================
    # 📌 STAGE: confirm_time_choice  (Unified Final Version)
    # ----------------------------------------------------------------------
    # 🎯 FUNCTIONAL PURPOSE:
    #   This stage performs ALL appointment selection logic:
    #
    #   1️⃣  Presents the 3 available appointment options.
    #   2️⃣  Accepts the user's selection via:
    #          • DTMF (1,2,3)
    #          • Speech (“Option 1”, “Option 2”, etc.)
    #          • Spoken full date (“November 17 at 11:30 AM”)
    #
    #   3️⃣  Once a selection is made → CONFIRMS IT:
    #          “You selected Monday, November 17 at 11:30 AM.
    #           Should I book this appointment? Say yes or no.”
    #
    #   4️⃣  YES → proceed to "book_appt_confirm"
    #   5️⃣  NO → return to menu and allow re-selection
    #
    #   EXTRA BEHAVIOR:
    #      • Handles Twilio fake-silence POST that arrives BEFORE Gather audio.
    #      • Handles real silence with retry limits.
    #      • Handles invalid selections, invalid times, and past times.
    #      • Ensures ONLY ONE spoken message per turn.
    #      • Guarantees consistent voice (no switching between voices).
    #
    #   🔥 THIS IS NOW THE ONLY STAGE NEEDED FOR APPOINTMENT CONFIRMATION.
    # ======================================================================
    elif stage == "confirm_time_choice":

        debug_print("=== ENTER confirm_time_choice (ALTS + SIMPLE YES/NO) ===")

        # --------------------------------------------------------------
        # 🎤 MESSAGE DECLARATIONS
        # --------------------------------------------------------------
        MSG_MENU_INTRO = "The next available appointments are: "
        MSG_MENU_END = (
            "Please say the appointment date and time you prefer. "
            "For example, Monday December first at 9 AM."
        )
        MSG_CONFIRM = (
            "You selected {friendly}. "
            "Please say YES to confirm or NO to cancel."
        )
        MSG_SILENCE_TIME = "I did not hear you. Please say the appointment time again."
        MSG_INVALID_TIME = "I did not understand that time. Please say the appointment time again."
        MSG_PAST = "This time has already passed."
        MSG_RESERVED = "This time is already reserved."
        MSG_GOODBYE = "No problem. Goodbye."
        MSG_TOO_MANY = "Sorry, we were unable to complete your request. Goodbye."

        # Words interpreted as YES / NO
        YES_WORDS = {"yes", "yeah", "yep", "ok", "okay", "sure", "confirm", "book"}
        NO_WORDS = {"no", "nope", "cancel", "stop"}

        # Limits (read from globals with defaults)
        MAX_SILENT = int(globals().get("MAX_SILENT_TIME", 3))
        MAX_RETRIES = int(globals().get("MAX_CONFIRM_RETRIES", 3))

        # --------------------------------------------------------------
        # 📌 LOAD SESSION & INPUT
        # --------------------------------------------------------------
        sd = session_data.setdefault(call_sid, {})

        # flags and counters
        sd.setdefault("silence_retry", 0)        # silence while asking for time or YES/NO
        sd.setdefault("retry_count", 0)          # invalid time retries
        sd.setdefault("confirm_mode", False)     # True → waiting for YES/NO
        sd.setdefault("alts_spoken", False)      # True → we already read out alternatives

        alts = sd.get("alts_list", [])           # alternatives filled by collect_book_time_date
        appt = sd.get("appointment_time")

        raw_speech = (speech_result or "").strip().lower()
        debug_print(f"[confirm_time_choice] speech='{raw_speech}'")

        # 🔢 Also capture DTMF (keys) if available
        raw_dtmf = (digits or "").strip() if "digits" in locals() or "digits" in globals() else ""
        debug_print(f"[confirm_time_choice] dtmf='{raw_dtmf}'")

        # 🧹 Normalize speech text for YES/NO matching (remove punctuation, split into tokens)
        speech_clean = _re.sub(r"[^\w\s]", " ", raw_speech)   # remove .,?! etc.
        speech_clean = _re.sub(r"\s+", " ", speech_clean).strip()
        tokens = speech_clean.split() if speech_clean else []

        # Helper: speak a list of alternative slots and wait for user response
        def _speak_alts_and_wait(alts_list):
            menu_msg = MSG_MENU_INTRO
            for i, slot in enumerate(alts_list, start=1):
                # e.g. "Option 1: Monday, December 1 at 9 AM."
                menu_msg += f"Option {i}: {slot['friendly']}. "
            menu_msg += MSG_MENU_END

            g_local = Gather(input="speech", timeout=8, action="/voice", method="POST")
            g_local.say(menu_msg, voice=VOICE)
            resp.append(g_local)

        # ==============================================================
        # 0️⃣ FIRST ENTRY AFTER A VALID TIME (no alts, not in confirm_mode)
        #    → ASK YES/NO ABOUT EXISTING appointment_time
        # ==============================================================
        if appt and not sd["confirm_mode"] and not alts:
            debug_print("[confirm_time_choice] initial confirm for existing appointment_time")

            sd["confirm_mode"] = True
            sd["silence_retry"] = 0
            sd["retry_count"] = 0

            friendly = appt.get("friendly", "your appointment time")
            g = Gather(input="speech dtmf", timeout=5, action="/voice", method="POST")
            g.say(MSG_CONFIRM.format(friendly=friendly), voice=VOICE)
            resp.append(g)
            return str(resp)

        # ==============================================================
        # 1️⃣ CONFIRMATION MODE — EXPECTING YES / NO
        # ==============================================================
        if sd["confirm_mode"]:
            debug_print("[confirm_time_choice] IN CONFIRMATION MODE")

            # 🔇 Silence while waiting for YES/NO (no speech and no key press)
            if not raw_speech and not raw_dtmf:
                sd["silence_retry"] += 1
                debug_print(f"[confirm_time_choice] confirm silence #{sd['silence_retry']}")

                if sd["silence_retry"] >= MAX_SILENT:
                    resp.say(MSG_TOO_MANY, voice=VOICE)
                    resp.hangup()
                    return str(resp)

                # Accept both speech and keys for YES/NO
                g = Gather(input="speech dtmf", timeout=5, action="/voice", method="POST")
                g.say("Please say YES to confirm or NO to cancel.", voice=VOICE)
                resp.append(g)
                return str(resp)

            # ✅ Detect YES / NO from speech tokens and keys
            is_yes_speech = any(t in YES_WORDS for t in tokens)
            is_no_speech  = any(t in NO_WORDS for t in tokens)
            is_yes_dtmf   = (raw_dtmf == "1")    # key 1 = YES / confirm
            is_no_dtmf    = (raw_dtmf == "2")    # key 2 = NO / cancel

            is_yes = is_yes_speech or is_yes_dtmf
            is_no  = is_no_speech or is_no_dtmf

            # ✅ YES → go to booking stage (new request)
            if is_yes and not is_no:
                debug_print("[confirm_time_choice] YES → book_appt_confirm")
                sd["confirm_mode"] = False
                sd["silence_retry"] = 0
                sd["retry_count"] = 0
                sd["alts_spoken"] = False
                sd["stage"] = "book_appt_confirm"

                save_session(call_sid)
                resp.redirect("/voice", method="POST")
                return str(resp)

            # ❌ NO → cancel and hang up
            if is_no and not is_yes:
                debug_print("[confirm_time_choice] NO → hangup")
                sd["confirm_mode"] = False
                sd["silence_retry"] = 0
                sd["retry_count"] = 0
                sd["alts_spoken"] = False
                resp.say(MSG_GOODBYE, voice=VOICE)
                resp.hangup()
                return str(resp)

            # ❓ Invalid YES/NO → repeat question
            debug_print("[confirm_time_choice] invalid YES/NO → re-prompt")
            g = Gather(input="speech dtmf", timeout=5, action="/voice", method="POST")
            g.say("Please say YES to confirm or NO to cancel.", voice=VOICE)
            resp.append(g)
            return str(resp)

        # ==============================================================
        # 2️⃣ FIRST ENTRY AFTER ALTERNATIVES GENERATED (alts_list)
        #    → READ ALTERNATIVES & ASK FOR A NEW TIME
        # ==============================================================
        if not sd["alts_spoken"] and alts:
            debug_print("[confirm_time_choice] first entry → read alternatives from alts_list")

            sd["alts_spoken"] = True
            sd["silence_retry"] = 0
            sd["retry_count"] = 0

            _speak_alts_and_wait(alts)
            return str(resp)

        # ==============================================================
        # 3️⃣ EXPECTING USER TO SAY A FULL DATE/TIME
        # ==============================================================

        # 🔇 Silence (no speech at all)
        if not raw_speech:
            sd["silence_retry"] += 1
            debug_print(f"[confirm_time_choice] time-input silence #{sd['silence_retry']}")

            if sd["silence_retry"] >= MAX_SILENT:
                resp.say(MSG_TOO_MANY, voice=VOICE)
                resp.hangup()
                return str(resp)

            g = Gather(input="speech", timeout=5, action="/voice", method="POST")
            g.say(MSG_SILENCE_TIME, voice=VOICE)
            resp.append(g)
            return str(resp)

        # --------------------------------------------------------------
        # 4️⃣ USER SPOKE A FULL DATE/TIME → PARSE IT
        # --------------------------------------------------------------
        parsed = smart_parse_time(raw_speech)
        debug_print(f"[confirm_time_choice] PARSED → {parsed}")

        # Not recognized at all
        if not parsed:
            sd["retry_count"] += 1
            debug_print(f"[confirm_time_choice] invalid time, retry_count={sd['retry_count']}")

            if sd["retry_count"] >= MAX_RETRIES:
                resp.say(MSG_TOO_MANY, voice=VOICE)
                resp.hangup()
                return str(resp)

            g = Gather(input="speech", timeout=5, action="/voice", method="POST")
            g.say(MSG_INVALID_TIME, voice=VOICE)
            resp.append(g)
            return str(resp)

        # --------------------------------------------------------------
        # 5️⃣ PAST CHECK → SUGGEST ALTERNATIVES IF POSSIBLE
        # --------------------------------------------------------------
        if parsed.get("is_past"):
            debug_print("[confirm_time_choice] time is in the past → suggest alternatives if possible")

            doctor = sd.get("doctor_name")
            now_utc = _pytz.UTC.localize(_dt.utcnow())

            alts = []
            if doctor:
                try:
                    alts = get_doctor_next_available_slots(
                        doctor,
                        from_start_iso=now_utc.isoformat(),
                        limit=3,
                    )
                    debug_print(f"[confirm_time_choice] past-time alts count={len(alts)}")
                except Exception as e:
                    debug_print(f"[confirm_time_choice] ⚠️ get_doctor_next_available_slots failed → {e}")
                    alts = []

            if not alts:
                # fallback – no alts → simple re-prompt
                g = Gather(input="speech", timeout=5, action="/voice", method="POST")
                g.say(MSG_PAST + " Please say another appointment time.", voice=VOICE)
                resp.append(g)
                return str(resp)

            # ✅ We have alternative slots → store and read them out
            sd["alts_list"] = alts
            sd["confirm_mode"] = False
            sd["alts_spoken"] = True   # we are about to speak them now
            sd["silence_retry"] = 0
            sd["retry_count"] = 0

            _speak_alts_and_wait(alts)
            return str(resp)

        # --------------------------------------------------------------
        # 6️⃣ RESERVED CHECK — SLOT ALREADY BOOKED? → SUGGEST ALTS
        # --------------------------------------------------------------
        doctor = sd.get("doctor_name")
        if doctor and not is_doctor_slot_available(doctor, parsed["start"], parsed["end"]):
            debug_print("[confirm_time_choice] time already reserved → suggest alternatives if possible")

            now_utc = _pytz.UTC.localize(_dt.utcnow())

            alts = []
            try:
                alts = get_doctor_next_available_slots(
                    doctor,
                    from_start_iso=now_utc.isoformat(),
                    limit=3,
                )
                debug_print(f"[confirm_time_choice] reserved-time alts count={len(alts)}")
            except Exception as e:
                debug_print(f"[confirm_time_choice] ⚠️ get_doctor_next_available_slots failed → {e}")
                alts = []

            if not alts:
                # fallback – no alts → simple re-prompt
                g = Gather(input="speech", timeout=5, action="/voice", method="POST")
                g.say(MSG_RESERVED + " Please say another appointment time.", voice=VOICE)
                resp.append(g)
                return str(resp)

            # ✅ We have alternative slots → store and read them out
            sd["alts_list"] = alts
            sd["confirm_mode"] = False
            sd["alts_spoken"] = True
            sd["silence_retry"] = 0
            sd["retry_count"] = 0

            _speak_alts_and_wait(alts)
            return str(resp)

        # --------------------------------------------------------------
        # 7️⃣ VALID & FREE SLOT → REPEAT TIME AND ASK YES/NO
        # --------------------------------------------------------------
        debug_print("[confirm_time_choice] valid, free slot → ask YES/NO")

        sd["appointment_time"] = parsed
        sd["confirm_mode"] = True
        sd["silence_retry"] = 0
        sd["retry_count"] = 0

        # Now we *must* use Gather so Twilio waits for YES/NO (speech or keys)
        g = Gather(input="speech dtmf", timeout=5, action="/voice", method="POST")
        g.say(MSG_CONFIRM.format(friendly=parsed["friendly"]), voice=VOICE)
        resp.append(g)
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
                language="en-GB",   # it was before en-US
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
                        language="en-GB",    # was en-US
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
                language="en-GB",   # wasen-US
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
            language="en-GB",   # was en-US
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
            hints=FOREIGN_NAME_HINTS,
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

            # 🕓 FIX: Give the caller more time to speak the full address
            gather = make_gather(
                PROMPT_RETRY_SILENCE,
                input="speech dtmf",
                language="en-US",
                timeout=15,              # ⏱ total listening time before timeout
                speech_timeout="auto",   # ⏳ automatically waits for pause completion
                barge_in=False,          # prevents premature cutoff mid-sentence
                finish_on_key="#",
                action="/voice", method="POST",
            )
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

            # 🕓 FIX: use extended listening window for retries too
            gather = make_gather(
                PROMPT_INVALID_ADDRESS,
                input="speech dtmf",
                language="en-US",
                timeout=20,
                speech_timeout="auto",
                barge_in=False,
                finish_on_key="#",
                action="/voice", method="POST",
            )
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
        gather = make_gather(
            PROMPT_CONFIRM_NEXT,
            input="speech dtmf",
            language="en-US",
            timeout=20,
            speech_timeout="auto",
            barge_in=True,
            finish_on_key="#",
            action="/voice", method="POST",
        )
        resp.append(gather)
        try:
            from flask import url_for
            resp.redirect(url_for("voice"))
        except Exception:
            resp.redirect("/voice")

        return str(resp)







        # ======================================================================
    # 🧾 Stage: collect_dr_info
    # ----------------------------------------------------------------------
    # 🎯 FUNCTIONAL PURPOSE:
    #   • Capture and identify the doctor with whom the caller wants to book
    #     an appointment, using either keypad (DTMF) input or spoken name.
    #   • Supports fuzzy speech matching to handle variations in pronunciation.
    #   • Handles invalid or junk speech (e.g., “hello”, “ok”) gracefully with retries.
    #
    # 🧩 INPUTS:
    #   • speech_result → Transcribed doctor name (from Twilio Speech-to-Text).
    #   • Digits        → Keypad selection (e.g., “1” for Dr. Smith).
    #   • call_sid      → Unique call identifier for session tracking.
    #
    # 💾 OUTPUTS (saved to session_data[call_sid]):
    #   • doctor_name → Selected doctor’s friendly name.
    #   • stage       → Advances to “collect_book_time_date” upon success.
    #
    # 🔁 FLOW OVERVIEW:
    #   1️⃣ First interaction:
    #       - Builds a keypad map (Press 1 for Dr. X, etc.).
    #       - Plays menu prompt via <Gather> (accepts speech or DTMF).
    #
    #   2️⃣ User response:
    #       - DTMF → Direct doctor selection.
    #       - Speech → Fuzzy matching between spoken tokens and known doctor names.
    #
    #   3️⃣ Retry handling:
    #       - Ignores junk words like “hello”, “ok”, “yes”, etc.
    #       - Allows up to 3 failed attempts before ending politely.
    #
    #   4️⃣ Success:
    #       - Stores selected doctor name in session.
    #       - Prompts for next stage → “collect_book_time_date”.
    #
    # 🧠 SPECIAL BEHAVIOR:
    #   • Handles both keypad and spoken inputs dynamically.
    #   • Maintains persistent doctor list and retry count in session.
    #   • Uses clear, friendly reprompts for speech or DTMF mismatches.
    #
    # ✅ SUMMARY:
    #   This stage reliably collects the intended doctor name by combining
    #   DTMF-based selection with fuzzy speech matching, ensuring smooth and
    #   user-friendly handling of voice or keypad input.
    # ======================================================================


    elif stage == "collect_dr_info":
        # ----------------------------------------------------------------------
        # 💬 VOICE PROMPTS — centralized for easy editing & localization
        # ----------------------------------------------------------------------
        VOICE_BOOK_INTRO_MSG = (
            "Please choose your doctor from the following list. "
            "You may either press the corresponding number on your keypad or say the doctor’s name."
        )
        VOICE_CANCEL_INTRO_MSG = (
            "Your cancellation will be with one of the following doctors. "
            "Please say the doctor's name or press the corresponding number."
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
        VOICE_BOOK_SUCCESS_MSG = (
            "Great, your appointment will be with {doctor_name}. "
            "Please say the appointment date and time, for example, 'October 8 at 9 30 A M'."
        )
        VOICE_CANCEL_SUCCESS_MSG = (
            "Okay, your cancellation will be for {doctor_name}. "
            "Please say the date and time of the appointment you want to cancel."
        )

        # ----------------------------------------------------------------------
        # 🧭 SESSION INITIALIZATION
        # ----------------------------------------------------------------------
        sd = session_data.setdefault(call_sid, {})
        # ⚠️ Preserve previously set origin_stage (cancel/book)
        origin_stage = sd.get("origin_stage", "book")  
        sd.setdefault("retry_booking", 0)
        debug_print(f"[collect_dr_info] 📍 Stage entered (origin={origin_stage})")

        # ----------------------------------------------------------------------
        # 🧹 CLEAN INPUTS (speech + DTMF)
        # ----------------------------------------------------------------------
        _PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        dtmf_digits = (request.values.get("Digits") or "").strip()
        spoken_text = (speech_result or "").strip().lower()
        spoken_clean = spoken_text.translate(str.maketrans('', '', _PUNCT)).strip()
        debug_print(f"[collect_dr_info] 🗣 speech='{spoken_clean}' 🔢 DTMF='{dtmf_digits}'")

        # ----------------------------------------------------------------------
        # 🗂️ INITIALIZE DOCTOR MAP (if not yet built)
        # ----------------------------------------------------------------------
        if "doctor_dtmf_map" not in sd:
            doctor_dtmf_map = {}
            prompt_lines = []

            if isinstance(doctor_names, dict):
                doctor_list = list(doctor_names.values())
            else:
                doctor_list = doctor_names

            for i, friendly in enumerate(doctor_list, start=1):
                doctor_dtmf_map[str(i)] = friendly
                prompt_lines.append(f"Press {i} for {friendly}.")

            sd["doctor_dtmf_map"] = doctor_dtmf_map

            # Pick intro message based on flow type
            intro_msg = VOICE_CANCEL_INTRO_MSG if origin_stage == "cancel" else VOICE_BOOK_INTRO_MSG
            doctor_prompt = f"{intro_msg} " + " ".join(prompt_lines)



            """
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
            """

            # ----------------------------------------------------------------------
            # 🎧 Prompt caller for doctor selection using Twilio <Gather>
            # ----------------------------------------------------------------------
            # The make_gather() helper creates a <Gather> TwiML element that:
            #  • Plays or speaks 'doctor_prompt' text to the caller
            #  • Listens for either speech or keypad (DTMF) input
            #  • Posts the collected input back to your /voice webhook
            # ----------------------------------------------------------------------
            g = make_gather(
                doctor_prompt,         # 🗣️ The spoken prompt (e.g., "Please say or press the number of your doctor.")
                input="speech dtmf",   # 🎙️ Accepts both speech recognition and keypad tones
                timeout=10,            # ⏳ Wait up to 10 seconds for caller to begin responding
                speech_timeout="auto", # 🧠 Automatically end recognition when silence is detected
                barge_in=True,         # 🚪 Allow caller to interrupt the prompt by speaking early
                finish_on_key="#",     # 🔚 End DTMF input collection when '#' is pressed
                num_digits=1           # 🔢 currently only 0-9.
                                       # 🔢 if u want to allow up to 2 digits (e.g., 10, 11, 12, ... 99) change to 2
            )

            # ----------------------------------------------------------------------
            # 📞 Add the <Gather> element to the Twilio <Response>
            # ----------------------------------------------------------------------
            # This appends the generated <Gather> block to your active VoiceResponse
            # so that Twilio executes it immediately.
            # ----------------------------------------------------------------------
            resp.append(g)

            # ----------------------------------------------------------------------
            # 🔁 Redirect Twilio back to /voice after <Gather> completes
            # ----------------------------------------------------------------------
            # Twilio posts the gathered data (SpeechResult / Digits)
            # back to this same webhook (/voice) when:
            #  • the user finishes speaking
            #  • or presses '#'
            #  • or timeout occurs
            # This redirect ensures continuity of your conversation flow.
            # ----------------------------------------------------------------------
            resp.redirect("/voice")

            # ----------------------------------------------------------------------
            # ✅ Return the TwiML response as XML string
            # ----------------------------------------------------------------------
            # Flask returns this TwiML to Twilio; Twilio executes it,
            # speaks the prompt, listens for input, and calls /voice again
            # when input or timeout occurs.
            # ----------------------------------------------------------------------
            return str(resp)


        # ----------------------------------------------------------------------
        # 🧭 RETRIEVE EXISTING DOCTOR MAP
        # ----------------------------------------------------------------------
        doctor_map = sd["doctor_dtmf_map"]
        matched_name = None

        # ----------------------------------------------------------------------
        # 🔢 STEP 1 — DTMF MATCHING
        # ----------------------------------------------------------------------
        if dtmf_digits and dtmf_digits in doctor_map:
            matched_name = doctor_map[dtmf_digits]
            debug_print(f"✅ DTMF matched doctor → {matched_name}")

        # ----------------------------------------------------------------------
        # 🗣️ STEP 2 — SPEECH MATCHING (Partial / Fuzzy)
        # ----------------------------------------------------------------------
        if matched_name is None:
            junk_inputs = {
                "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
                "yo", "test", "1", "yes", "no", "i know", "huh", "what", "okay", "ok",
                "bye", "goodbye", ""
            }

            if not spoken_clean or spoken_clean in junk_inputs or len(spoken_clean) < 3:
                debug_print(f"⏩ Skipping junk doctor input → '{spoken_clean}' (re-prompting)")
                prompt_lines = [f"Press {k} for {v}." for k, v in doctor_map.items()]
                intro_msg = VOICE_CANCEL_INTRO_MSG if origin_stage == "cancel" else VOICE_REPROMPT_MSG
                doctor_prompt = f"{intro_msg} " + " ".join(prompt_lines)
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

            spoken_tokens = set(spoken_clean.split())
            partial_matches = []

            if isinstance(doctor_names, dict):
                doctor_list = list(doctor_names.values())
            else:
                doctor_list = doctor_names

            for friendly in doctor_list:
                friendly_clean = friendly.lower().translate(str.maketrans('', '', _PUNCT)).strip()
                friendly_tokens = set(friendly_clean.split())
                if (
                    spoken_clean in friendly_clean
                    or friendly_clean in spoken_clean
                    or (spoken_tokens & friendly_tokens)
                ):
                    partial_matches.append(friendly)

            if len(partial_matches) == 1:
                matched_name = partial_matches[0]
                debug_print(f"✅ Partial speech match → {matched_name}")
            elif len(partial_matches) > 1:
                debug_print(f"🔍 Multiple doctor matches found → {partial_matches}")
                matched_name = partial_matches[0]

        # ----------------------------------------------------------------------
        # ❌ STEP 3 — HANDLE NO MATCH FOUND
        # ----------------------------------------------------------------------
        if matched_name is None:
            sd["retry_booking"] += 1
            retries = sd["retry_booking"]
            debug_print(f"❌ No doctor match for '{spoken_clean or dtmf_digits}' retry={retries}")

            if retries >= 3:
                resp.say(gpt_speak(VOICE_FINAL_FAIL_MSG), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

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
        sd["doctor_name"] = matched_name

        if origin_stage in ("cancel", "reschedule"):
            sd["stage"] = "collect_cancel_time_date"
            success_msg = VOICE_CANCEL_SUCCESS_MSG.format(doctor_name=matched_name)
        else:
            sd["stage"] = "collect_book_time_date"
            success_msg = VOICE_BOOK_SUCCESS_MSG.format(doctor_name=matched_name)

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
    # 🧭 FUNCTIONAL DESCRIPTION
    # ----------------------------------------------------------------------
    # This stage finalizes the credit card update process after successful
    # PIN verification and collection of card details.
    #
    # Behavior summary:
    #   1. Retrieves customer's E.164 phone number and date of birth.
    #   2. Normalizes phone number if provided in local/national format.
    #   3. Calls update_cc_info(phone_e164, dob, cc_number, cc_exp, cc_cvv)
    #      to persist the credit card update in the database.
    #   4. If missing phone or DOB, the user is redirected to complete
    #      those prerequisites first (collect_phone or collect_dob).
    #   5. Provides a voice confirmation of success or failure to the user.
    #   6. Thanks the customer, confirms that their payment info was updated,
    #      and hangs up gracefully.
    # ----------------------------------------------------------------------


    elif stage == "update_customer_cc":
       

        # ----------------------------------------------------------------------
        # 🧱 Retrieve session and customer context
        # ----------------------------------------------------------------------
        sd   = session_data.get(call_sid, {})
        cust = sd.get("customer", {})

        # Use the country associated with the call session, fallback to global default
        default_country = (sd.get("country") or COUNTRY or "US").upper()

        # ----------------------------------------------------------------------
        # 📞 Extract and normalize phone number
        # ----------------------------------------------------------------------
        # Preference order:
        #   1. customer["phone_e164"]
        #   2. session["phone_e164"]
        #   3. raw customer/session "phone"
        #   4. empty string fallback
        phone_raw = (
            cust.get("phone_e164")   # preferred normalized
            or sd.get("phone_e164")  # fallback normalized
            or cust.get("phone")     # unnormalized; try to fix
            or sd.get("phone")       # unnormalized; try to fix
            or ""
        )
        raw = (phone_raw or "").strip()
        phone_e164 = ""

        # Normalize to E.164 — handles “+1 202 555 0123” and “02025550123” forms
        if raw.startswith("+") and raw[1:].replace(" ", "").isdigit():
            # Already valid +E.164, clean spaces
            phone_e164 = "+" + raw[1:].replace(" ", "")
        else:
            try:
                # Attempt normalization using the default country
                phone_e164 = normalize_phone_e164(raw, default_country) or ""
                if not phone_e164:
                    # Retry using alternate country (US/Egypt flip)
                    alt = "EG" if default_country != "EG" else "US"
                    phone_e164 = normalize_phone_e164(raw, alt) or ""
            except Exception:
                phone_e164 = ""

        phone_to_use = phone_e164  # final E.164 number

        # ----------------------------------------------------------------------
        # 🎂 Retrieve DOB and CC details from session
        # ----------------------------------------------------------------------
        dob_iso   = cust.get("dob") or sd.get("dob_iso") or ""   # e.g., '1986-07-03'
        cc_number = cust.get("cc_number")
        cc_exp    = cust.get("cc_exp") or cust.get("cc_expiration")
        cc_cvv    = cust.get("cc_cvv")

        # ----------------------------------------------------------------------
        # ⚠️ Guard: Require both phone and DOB before updating
        # ----------------------------------------------------------------------
        if not phone_to_use or not dob_iso:
            debug_print("update_customer_cc: ❌ Missing E.164 phone or DOB; redirecting to prerequisite stage")

            # Redirect to collect_phone or collect_dob depending on what’s missing
            sd["stage"] = "collect_phone" if not phone_to_use else "collect_dob"
            prompt = (
                "Before we update your card, please say or enter your phone number, including country code."
                if not phone_to_use else
                "Before we update your card, please say your birth date. "
                "For example, say July 3rd 1956 or enter 2 digits for month, 2 for day, and 4 for year, then press pound."
            )
            resp.append(make_gather(prompt, hints="zero one two three four five six seven eight nine"))
            return str(resp)

        # ----------------------------------------------------------------------
        # 💾 Perform credit card update
        # ----------------------------------------------------------------------
        ok = False
        try:
            # Call the database or API update routine (must be implemented separately)
            result = update_cc_info(
                phone_to_use,   # E.164 number required
                dob_iso,        # verified date of birth
                cc_number=cc_number,
                cc_exp=cc_exp,
                cc_cvv=cc_cvv,
            )

            # Determine if the update succeeded (supports dict or bool return)
            ok = bool(result) if not isinstance(result, dict) else bool(result.get("ok", False))
            debug_print(f"update_customer_cc: ✅ update_cc_info returned ok={ok}")

        except Exception as e:
            ok = False
            debug_print(f"update_customer_cc: 💥 Exception during update_cc_info → {e}")

        # ----------------------------------------------------------------------
        # 🧹 Do NOT mask, clear, or modify stored card fields
        # ----------------------------------------------------------------------
        # We intentionally leave cc_number, cc_exp, and cc_cvv intact in session_data
        # so that confirmation or auditing can use them before session teardown.
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # 🗣️ Final response to caller
        # ----------------------------------------------------------------------
        if ok:
            final_msg = (
                "Thank you. Your credit card information has been securely updated. "
                "We appreciate your time. Have a wonderful day!"
            )
        else:
            final_msg = (
                "I’m sorry, but I couldn’t update your credit card information right now. "
                "Please try again later or contact the clinic directly."
            )

        resp.say(gpt_speak(final_msg), VOICE)

        # ----------------------------------------------------------------------
        # 📴 Graceful call termination
        # ----------------------------------------------------------------------
        # After thanking the customer, we end the call politely.
        # The session will naturally clear on hangup.
        resp.hangup()
        debug_print(f"update_customer_cc: ✅ Completed (ok={ok}); call ending.")
        return str(resp)








     # ======================================================================
    # 🧾 Stage: collect_cc
    # ----------------------------------------------------------------------
    # 🎯 FUNCTIONAL PURPOSE:
    #   • Collects and validates credit card information from the caller
    #     across three sequential steps using either speech or DTMF input.
    #   • Ensures all data passes strict formatting and validation rules,
    #     including Luhn checksum for card numbers and future-date check
    #     for expiration dates.
    #
    # 🧩 INPUTS:
    #   • SpeechResult → Transcribed spoken digits or card info.
    #   • Digits       → Keypad (DTMF) input for numeric entry.
    #   • call_sid     → Unique call identifier for session persistence.
    #
    # 💾 OUTPUTS (stored in session_data[call_sid]["customer"]):
    #   • cc_number → Card number (13–19 digits, Luhn validated).
    #   • cc_exp    → Expiration date (MM/YY format, non-expired).
    #   • cc_cvv    → Security code (3–4 digits).
    #   • cc_name   → Derived from first and last name if available.
    #
    # 🔁 FLOW OVERVIEW:
    #   1️⃣ **Step 1 – Card Number:**
    #        - Accepts 13–19 digits via speech or DTMF.
    #        - Normalizes spoken words (“one two three four”) to digits.
    #        - Validates using the Luhn algorithm.
    #        - On success → advances to Step 2 (Expiration).
    #
    #   2️⃣ **Step 2 – Expiration Date:**
    #        - Accepts MMYY or MMYYYY via DTMF or spoken input.
    #        - Validates month range (1–12) and ensures the date is in the future.
    #        - On success → advances to Step 3 (CVV).
    #
    #   3️⃣ **Step 3 – CVV:**
    #        - Accepts 3–4 digit security code via DTMF or speech.
    #        - On success → stores data and advances to “book_appt_confirm”
    #          or “update_customer_cc” depending on session context.
    #
    # 🧠 SPECIAL BEHAVIOR:
    #   • Graceful silence handling with up to 3 re-prompts per step.
    #   • Automatic fallback to DTMF if speech parsing is uncertain.
    #   • Strips punctuation and converts “double/triple” speech terms correctly.
    #   • Immediate step advancement via “#” key.
    #
    # ⚙️ TECHNICAL DETAILS:
    #   • Uses `_normalize_spoken_digits()` to convert word-based input to digits.
    #   • `_luhn_ok()` performs credit card checksum validation.
    #   • `_mask()` used for secure logging of sensitive card info.
    #   • Twilio `<Gather>` waits for speech or DTMF input; `<Redirect>` triggers next step immediately.
    #
    # ✅ SUMMARY:
    #   This stage ensures secure, flexible, and user-friendly credit card capture
    #   with robust normalization, validation, and retry control across all inputs.
    # ======================================================================
    # ----------------------------------------------------------------------
    # collect_cc - complete stage (robust digit normalization + strict Luhn)
    # ----------------------------------------------------------------------

    elif stage == "collect_cc":
        
        # ----------------------------------------------------------------------
        # 💬 VOICE MESSAGES — centralized for maintainability & localization
        # ----------------------------------------------------------------------
        MSG_SILENCE_EXIT = "I’m still not hearing anything. Please call again later."
        MSG_CARD_PROMPT = "Please enter or say your card number now, then press pound."
        MSG_EXP_PROMPT = "Please enter or say the expiration date, for example, zero nine two seven, then press pound."
        MSG_CVV_PROMPT = "Please enter or say the three or four digit security code, then press pound."
        MSG_INVALID_CARD = "That card number doesn't look right. Please re-enter or say the full card number, then press pound."
        MSG_EXP_INVALID = "That doesn’t look valid. Please enter month and year as M M Y Y, then press pound."
        MSG_EXP_FORMAT = "Please say or enter the expiration date as month and year, for example, zero nine two seven, then press pound."
        MSG_PIN_UPDATED = "Your PIN has been updated successfully. Thank you!"
        MSG_PIN_UPDATE_FAIL = "We couldn’t verify your card, so we can’t update your PIN. Please call the clinic for assistance."
        # 🔁 NOTE:
        #   Previously this prompt told the caller to "say or enter your insurance provider
        #   and policy number". That caused confusion, because the *next* stage
        #   (collect_insurance_information) already:
        #       • lists insurance companies
        #       • then separately asks for member / policy ID.
        #   We now make this a simple transition message only.
        MSG_INSURANCE_PROMPT = (
            "Now let's collect your insurance information. "
            "I will first ask you to choose your insurance company, then I will ask for your member ID."
        )

        # ----------------------------------------------------------------------
        # 🧮 Helper functions
        # ----------------------------------------------------------------------
        def _luhn_ok(pan: str) -> bool:
            # Simple Luhn checksum validator for primary account numbers (PAN)
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
            # Convert spoken numbers ("zero one two", "double three", etc.) to a digits string.
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
            # Unified access to numeric input from DTMF or speech
            if enforce_dtmf:
                return _re.sub(r"\D", "", dtmf or "")
            if dtmf:
                return _re.sub(r"\D", "", dtmf)
            return _re.sub(r"\D", "", _normalize_spoken_digits(speech or ""))

        def _mask(pan: str) -> str:
            # Mask PAN for logs: ************1234
            pan = pan or ""
            if len(pan) <= 4:
                return pan
            return "*" * (len(pan) - 4) + pan[-4:]

        # ----------------------------------------------------------------------
        # 🧱 SESSION INITIALIZATION
        # ----------------------------------------------------------------------
        session_data.setdefault(call_sid, {})
        session_data[call_sid].setdefault("customer", {})
        customer     = session_data[call_sid]["customer"]
        cc_step      = int(session_data[call_sid].get("cc_step", 1))          # 1 = PAN, 2 = EXP, 3 = CVV
        enforce_dm   = bool(session_data[call_sid].get("enforce_dtmf_cc"))
        origin_stage = session_data[call_sid].get("origin_stage", "").lower()

        raw_dtmf   = (request.values.get("Digits") or "").strip()
        raw_speech = (speech_result or "").strip()

        debug_print(f"collect_cc: 📍 step={cc_step}, origin_stage='{origin_stage}', DTMF='{raw_dtmf}', speech='{raw_speech}'")

        # ----------------------------------------------------------------------
        # 🔇 Silence handling (all steps)
        # ----------------------------------------------------------------------
        if not raw_dtmf and not raw_speech:
            tries = session_data[call_sid].get("silence_cc", 0) + 1
            session_data[call_sid]["silence_cc"] = tries
            debug_print(f"collect_cc: 🤐 silence on step {cc_step}; tries={tries}")

            if tries >= 3:
                # After 3 silent attempts → final apology and hangup
                resp.say(gpt_speak(MSG_SILENCE_EXIT), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # Choose prompt based on step (card / expiry / cvv)
            prompt = {
                1: MSG_CARD_PROMPT,
                2: MSG_EXP_PROMPT,
                3: MSG_CVV_PROMPT
            }.get(cc_step, MSG_CARD_PROMPT)

            gather = make_gather(
                prompt,
                input="speech dtmf",
                timeout=30,              # ⬅️ give more time for long card numbers
                speech_timeout="10",     # ⬅️ allow pauses while saying digits
                finish_on_key="#",
                action="/voice",
                barge_in=True,
            )
            resp.append(gather)
            return str(resp)

        # ✅ Reset silence counter on valid input
        session_data[call_sid].pop("silence_cc", None)

        # ======================================================================
        # 🧮 STEP 1 — CARD NUMBER (PAN, 13–19 digits)
        # ======================================================================
        if cc_step == 1:
            # Normalize digits for card number
            pan = _digits_from(raw_dtmf, raw_speech, enforce_dtmf=enforce_dm)
            if len(pan) > 19:
                pan = pan[:19]
            debug_print(f"collect_cc: normalized card digits={pan}")

            # Length check
            if not (13 <= len(pan) <= 19):
                debug_print("collect_cc: ❌ invalid card length")
                gather = make_gather(
                    MSG_INVALID_CARD,
                    input="speech dtmf",
                    timeout=30,              # ⬅️ longer to re-say the whole PAN
                    speech_timeout="10",     # ⬅️ allow natural pauses
                    finish_on_key="#",
                    action="/voice",
                )
                resp.append(gather)
                return str(resp)

            # Luhn check (non-strict failure handling as before)
            if not _luhn_ok(pan):
                debug_print(f"collect_cc: ⚠️ {_mask(pan)} failed Luhn but accepted (non-strict mode).")
            else:
                debug_print(f"collect_cc: ✅ Luhn passed for {_mask(pan)}")

            # ------------------------------------------------------------------
            # 🔀 Branch after card number depending on origin (PIN update flow)
            # ------------------------------------------------------------------
            phone_e164 = (customer.get("phone_e164") or session_data[call_sid].get("phone_e164") or "").strip()
            dob        = (customer.get("dob") or "").strip()

            if origin_stage == "update_pin_number":
                # PIN UPDATE: Skip Expiration + CVV and verify card immediately
                debug_print("collect_cc: ⏭️ Skipping expiration & CVV (PIN update flow)")

                try:
                    stored_cc = get_customer_cc(dob, phone_e164)
                except Exception as e1:
                    debug_print(f"collect_cc: ⚠️ get_customer_cc(dob, phone) failed → {e1} ; trying (phone, dob)")
                    try:
                        stored_cc = get_customer_cc(phone_e164, dob)
                    except Exception as e2:
                        debug_print(f"collect_cc: ⚠️ get_customer_cc(phone, dob) also failed → {e2}")
                        stored_cc = None

                debug_print(f"collect_cc: 💳 stored_cc={_mask(stored_cc)} collected_cc={_mask(pan)}")

                if stored_cc and stored_cc == pan:
                    # Cards match → proceed to update PIN
                    try:
                        new_pin = random.randint(100000, 999999)  # 🔐 existing behavior
                        ok = update_pin_number(phone_e164, dob, new_pin)
                        debug_print(f"collect_cc: ✅ update_pin_number() → {ok}")
                        if ok:
                            resp.say(gpt_speak(MSG_PIN_UPDATED), VOICE)
                        else:
                            resp.say(gpt_speak(MSG_PIN_UPDATE_FAIL), VOICE)
                            resp.hangup()
                            session_data.pop(call_sid, None)
                            return str(resp)
                    except Exception as e:
                        debug_print(f"collect_cc: ⚠️ update_pin_number() exception → {e}")
                        resp.say(gpt_speak(MSG_PIN_UPDATE_FAIL), VOICE)
                        resp.hangup()
                        session_data.pop(call_sid, None)
                        return str(resp)
                else:
                    # 🚫 Card mismatch — authentication failed, end call
                    debug_print("collect_cc: 🚫 Collected CC does not match stored CC → deny PIN update")
                    resp.say(gpt_speak(
                        "Authentication failed. We can’t update your PIN number at this time. "
                        "Please call the clinic for assistance."
                    ), VOICE)
                    resp.hangup()
                    session_data.pop(call_sid, None)
                    return str(resp)

                # After PIN update attempt → back to intro
                session_data[call_sid]["stage"] = "intro"
                resp.redirect("/voice")
                return str(resp)

            # ------------------------------------------------------------------
            # Normal flow: store PAN and move to step 2 (expiration)
            # ------------------------------------------------------------------
            customer["cc_number"] = pan
            session_data[call_sid]["cc_step"] = 2
            debug_print("collect_cc: ➡️ Moving to step 2 (Expiration)")
            resp.redirect("/voice")
            return str(resp)

        # ======================================================================
        # 🗓️ STEP 2 — EXPIRATION (MMYY / MMYYYY)
        # ======================================================================
        if cc_step == 2:
            session_data[call_sid]["no_input_expected"] = True
            # ⬅️ Allow BOTH speech and DTMF for expiration
            digits = _digits_from(raw_dtmf, raw_speech, enforce_dtmf=False)
            debug_print(f"collect_cc: Step 2 digits='{digits}'")

            # Accept 4 or 6 digits:
            #   "0927"   → 09/27
            #   "092027" → 09/2027
            if len(digits) not in (4, 6):
                # Build a <Gather> instruction for Twilio:
                #  - The spoken prompt tells the caller how to give the expiration (examples).
                #  - input="speech dtmf" lets the caller either *speak* the date or *type* it on the keypad.
                #  - timeout=20 means Twilio will wait up to 20 seconds for user input before the gather times out.
                #  - finish_on_key="#" allows the caller to press the pound key to immediately finish typing
                #    instead of waiting for the timeout or for num_digits to be reached.
                #  - action="/voice" tells Twilio to POST the results of this gather back to your /voice
                #    webhook when the gather completes (either by #, by the timeout, or by speech result).
                gather = make_gather(
                    MSG_EXP_FORMAT,
                    input="speech dtmf",
                    timeout=20,          # ⬅️ more generous for spoken expiration
                    speech_timeout="10", # ⬅️ allow pauses while speaking
                    finish_on_key="#",
                    action="/voice",
                )
                # Return the TwiML (string form) immediately so Twilio receives the <Gather>.
                # Important behavior:
                #  - We do NOT call resp.redirect("/voice") here. Returning the TwiML with the <Gather>
                #    causes Twilio to wait for the user's input and then POST back to the 'action' URL.
                #  - After the user types/speaks and the gather finishes, Twilio will call your /voice
                #    webhook again with request parameters such as 'Digits' (for DTMF) and/or
                #    'SpeechResult' (for speech). Your handler should then re-enter this stage and
                #    process the provided digits.
                resp.append(gather)
                return str(resp)

            # Save expiration and move to CVV
            customer["cc_expiration"] = digits
            session_data[call_sid]["cc_step"] = 3
            debug_print(f"collect_cc: ✅ Saved expiration='{digits}' → step 3 (CVV)")
            resp.redirect("/voice")
            return str(resp)

        # ======================================================================
        # 🔐 STEP 3 — CVV (3–4 digits)
        # ======================================================================
        if cc_step == 3:
            session_data[call_sid]["no_input_expected"] = True
            # ⬅️ Allow BOTH speech and DTMF for CVV
            digits = _digits_from(raw_dtmf, raw_speech, enforce_dtmf=False)
            debug_print(f"collect_cc: Step 3 CVV digits='{digits}'")

            # CVV must be 3–4 numeric digits
            if not (3 <= len(digits) <= 4 and digits.isdigit()):
                gather = make_gather(
                    MSG_CVV_PROMPT,
                    input="speech dtmf",
                    timeout=15,          # ⬅️ more time to say CVV
                    speech_timeout="10", # ⬅️ allow a short pause
                    finish_on_key="#",
                    action="/voice",
                )
                resp.append(gather)
                return str(resp)

            # Store CVV and cardholder name
            customer["cc_cvv"] = digits
            if not customer.get("cc_name"):
                customer["cc_name"] = f"{customer.get('first_name','')} {customer.get('last_name','')}".strip()
            debug_print(f"collect_cc: ✅ CVV saved (len={len(digits)}) ; cc_name='{customer.get('cc_name')}'")

            # Finalize for non-PIN-update flows
            session_data[call_sid].pop("no_input_expected", None)
            session_data[call_sid].pop("cc_step", None)
            session_data[call_sid]["cc_speech_tries"] = 0

            # Decide where to go next based on origin_stage
            origin_stage2 = session_data[call_sid].get("origin_stage", "").lower()
            debug_print(f"collect_cc: 🔁 origin_stage after CVV = '{origin_stage2}'")

            if origin_stage2 == "update_cc":
                next_stage = "update_customer_cc"
                session_data[call_sid]["stage"] = next_stage
                session_data[call_sid]["skip_silence_once"] = True
                debug_print(f"collect_cc: ➡️ Auto-advancing to {next_stage} (origin_stage=update_cc)")
                resp.redirect("/voice")
                return str(resp)

            if origin_stage2 in ("book", "reschedule"):
                next_stage = "book_appt_confirm"
                session_data[call_sid]["stage"] = next_stage
                session_data[call_sid]["skip_silence_once"] = True
                debug_print(f"collect_cc: ➡️ Auto-advancing to {next_stage} (origin_stage={origin_stage2})")
                resp.redirect("/voice")
                return str(resp)

            if origin_stage2 == "register":
                # Reset insurance flow state
                session_data[call_sid].pop("insurance_step", None)
                session_data[call_sid].pop("insurance_silence", None)
                session_data[call_sid].pop("insurance_invalid", None)

                # Move to insurance stage
                session_data[call_sid]["stage"] = "collect_insurance_information"

                debug_print("collect_cc: ✅ Transition → collect_insurance_information")

                # IMPORTANT: Do NOT prompt here
                # insurance stage will handle prompting and listing

                resp.redirect("/voice")
                return str(resp)

            # All other / unexpected flows: finish politely and go back to intro
            debug_print(f"collect_cc: ℹ️ CVV captured in non-booking flow (origin_stage={origin_stage2})")
            resp.say(
                gpt_speak(
                    "Thank you. Your card information has been saved. "
                    "You can now continue with the clinic menu."
                ),
                VOICE,
            )
            session_data[call_sid]["stage"] = "intro"
            resp.redirect("/voice")
            return str(resp)

        # ======================================================================
        # 🚨 Fallback — unexpected cc_step value
        # ======================================================================
        debug_print(f"collect_cc: ⚠️ unexpected cc_step={cc_step} → resetting to 1")
        session_data[call_sid]["cc_step"] = 1
        gather = make_gather(
            MSG_CARD_PROMPT,
            input="speech dtmf",
            timeout=30,              # ⬅️ match the other card gathers
            speech_timeout="10",
            finish_on_key="#",
            action="/voice",
            barge_in=True,
        )
        resp.append(gather)
        return str(resp)



   # ======================================================================
    # 🗓️ STAGE: collect_cancel_time_date
    # ======================================================================
    # 🎯 FUNCTIONAL DESCRIPTION:
    #     Captures and parses the spoken or keypad-provided date/time
    #     for the appointment the caller wishes to cancel.
    #
    # 💡 PURPOSE:
    #     1. Ask the caller for appointment date/time.
    #     2. Parse natural language input (e.g., "November third at ten A M").
    #     3. Retry gracefully if no input or invalid date (up to 3 times each).
    #     4. Verify if the given slot exists and is not in the past.
    #     5. Proceed to confirmation or fallback to listing appointments.
    #
    # ⚙️ TECHNICAL FEATURES:
    #     • Uses dateutil.parser for fuzzy recognition of human speech.
    #     • Uses timezone-aware conversion (default: America/Chicago).
    #     • Removes redirect() to prevent premature POST callbacks.
    #     • Supports longer timeout and speech_timeout for natural pauses.
    #
    # 💾 OUTPUTS:
    #     cancel_ctx["matching_event"] = {
    #         "spoken_dt": "<verbatim phrase>",
    #         "start": "<UTC ISO start>",
    #         "end": "<UTC ISO end>"
    #     }
    #     session_data[call_sid]["stage"] = "cancel_appt_confirm"
    #
    # 🔁 NEXT STAGES:
    #     → cancel_appt_confirm (if slot verified)
    #     → cancel_appt_iterate (if invalid or past)
    # ======================================================================

    elif stage == "collect_cancel_time_date":
        debug_print("collect_cancel_time_date: 📍 Stage entered")

        # ------------------------------------------------------------------
        # 💬 Voice prompt strings — centralized for easy editing
        # ------------------------------------------------------------------
        VOICE_PROMPT_INITIAL = (
            "Please say the date and time of the appointment you want to cancel. "
            "For example, say November third at ten A M."
        )
        VOICE_PROMPT_RETRY = (
            "I didn’t catch that. Please say the appointment date and time clearly, "
            "for example, October twenty-first at 3:30 PM."
        )
        VOICE_PROMPT_TOO_MANY_SILENCES = (
            "That doesn’t match any of your appointments. I’ll list your upcoming ones."
        )
        VOICE_PROMPT_PAST_TIME = (
            "That appointment time has already passed. I’ll list your upcoming ones."
        )
        VOICE_PROMPT_SLOT_NOT_FOUND = (
            "That doesn’t match any of your appointments. I’ll list your upcoming ones."
        )

        # ------------------------------------------------------------------
        # 🧱 Retrieve the cancellation context from session (per caller)
        # ------------------------------------------------------------------
        cancel_ctx = session_data[call_sid].setdefault("cancel", {})

        # ------------------------------------------------------------------
        # 🛡 FIRST ENTRY CHECK — If Twilio just redirected here without input
        # ------------------------------------------------------------------
        if not speech_result and not request.values.get("Digits"):
            debug_print("collect_cancel_time_date: 🆕 First entry → play initial prompt")

            # Build gather to capture speech or keypad
            gather = make_gather(
                VOICE_PROMPT_INITIAL,
                input="speech dtmf",
                timeout=25,           # Caller has up to 25 sec to start talking
                speech_timeout="5",   # Allow up to 5 sec pause during speech
                barge_in=True,
                finish_on_key="#",
            )

            # Add gather to response
            resp.append(gather)

            # Twilio will POST to /voice after gather
            resp.redirect("/voice")
            return str(resp)

        # ------------------------------------------------------------------
        # 🎧 Capture raw recognized speech text (if any)
        # ------------------------------------------------------------------
        raw = (speech_result or "").strip()
        debug_print(f"collect_cancel_time_date: 🗣️ Raw speech = '{raw}'")

        # ------------------------------------------------------------------
        # 🔇 Handle silence (no speech recognized)
        # ------------------------------------------------------------------
        if not raw:
            # Increment silence counter
            tries = cancel_ctx.get("silence_cancel_dt", 0) + 1
            cancel_ctx["silence_cancel_dt"] = tries
            debug_print(f"collect_cancel_time_date: 🤐 Silence detected (count={tries})")

            # If 3 consecutive silences → fallback to appointment list stage
            if tries >= 3:
                debug_print("collect_cancel_time_date: 🚫 Silence limit reached → moving to cancel_appt_iterate")
                cancel_ctx.pop("silence_cancel_dt", None)
                cancel_ctx["awaiting_input"] = False
                session_data[call_sid]["stage"] = "cancel_appt_iterate"
                session_data[call_sid]["skip_silence_retry"] = True
                resp.say(gpt_speak(VOICE_PROMPT_TOO_MANY_SILENCES), VOICE)
                resp.redirect("/voice")
                return str(resp)

            # Otherwise → reprompt
            resp.pause(length=1)
            gather = make_gather(
                VOICE_PROMPT_RETRY,
                input="speech dtmf",
                timeout=25,
                speech_timeout="5",
                barge_in=True,
                finish_on_key="#",
            )
            resp.append(gather)
            resp.redirect("/voice")
            return str(resp)

        # If user spoke → reset silence counter
        cancel_ctx.pop("silence_cancel_dt", None)

        # ------------------------------------------------------------------
        # 🧹 Normalize and clean speech artifacts (STT errors)
        # ------------------------------------------------------------------
        day_part, time_part = (None, None)
        try:
            raw_fixed = raw.lower().replace(",", "").strip()

            # Example: "October 1388" → "October 13"
            raw_fixed = _re.sub(r"\b(\d{1,2})\d{2,3}\b", r"\1", raw_fixed)

            # Convert words like "twenty first" → "21"
            ordinal_map = {
                "first":1,"second":2,"third":3,"fourth":4,"fifth":5,"sixth":6,
                "seventh":7,"eighth":8,"ninth":9,"tenth":10,"eleventh":11,
                "twelfth":12,"thirteenth":13,"fourteenth":14,"fifteenth":15,
                "sixteenth":16,"seventeenth":17,"eighteenth":18,"nineteenth":19,
                "twentieth":20,"twenty first":21,"twenty second":22,
                "twenty third":23,"twenty fourth":24,"twenty fifth":25,
                "twenty sixth":26,"twenty seventh":27,"twenty eighth":28,
                "twenty ninth":29,"thirtieth":30,"thirty first":31
            }
            for word, num in ordinal_map.items():
                raw_fixed = _re.sub(rf"\b{word}\b", str(num), raw_fixed)

            # Split into day + time if “at” is present
            if " at " in raw_fixed:
                parts = raw_fixed.split(" at ")
                if len(parts) == 2:
                    day_part = parts[0].strip()
                    time_part = parts[1].strip()
            else:
                # If no time said → default to noon
                day_part = raw_fixed
                time_part = "12:00 pm"

            debug_print(f"collect_cancel_time_date: 📆 Extracted → day='{day_part}', time='{time_part}'")

        except Exception as e:
            debug_print(f"collect_cancel_time_date: ⚠️ Error normalizing text → {e}")

        # ------------------------------------------------------------------
        # 🕒 Parse natural-language datetime into local timezone → convert to UTC
        # ------------------------------------------------------------------
        matched = False
        dt_utc = None
        spoken_phrase = None
        try:
            if day_part:
                # Fix minor errors like "3d" → "3rd"
                day_part_fixed = _re.sub(r"\b(\d{1,2})d\b", r"\1rd", day_part)

                # Build phrase for dateutil.parser
                spoken_phrase = f"{day_part_fixed} at {time_part}"

                # Load clinic timezone
                tz_name = globals().get("CLINIC_TZ", "America/Chicago")
                tz = _pytz.timezone(tz_name)

                # Parse fuzzy natural language
                dt_local = dp.parse(
                    spoken_phrase,
                    fuzzy=True,
                    default=datetime.now(tz)
                )

                # Convert local → UTC
                dt_utc = dt_local.astimezone(_pytz.UTC)

                # Compute 30-minute appointment end
                dt_end = dt_utc + timedelta(minutes=30)

                matched = True
                debug_print(f"collect_cancel_time_date: 🕒 Parsed UTC datetime → {dt_utc}")

        except Exception as e:
            debug_print(f"collect_cancel_time_date: ❌ Failed fuzzy parse → {e}")

        # ------------------------------------------------------------------
        # ❌ Retry on parse failure (3 attempts)
        # ------------------------------------------------------------------
        if not matched:
            retries = cancel_ctx.get("retry_cancel_dt", 0) + 1
            cancel_ctx["retry_cancel_dt"] = retries
            debug_print(f"collect_cancel_time_date: ❌ parse failed → retry {retries}")

            if retries < 3:
                resp.pause(length=1)
                gather = make_gather(
                    VOICE_PROMPT_RETRY,
                    input="speech dtmf",
                    timeout=25,
                    speech_timeout="5",
                    barge_in=True,
                    finish_on_key="#",
                )
                resp.append(gather)
                resp.redirect("/voice")
                return str(resp)

            # Too many parse failures → list all appointments
            debug_print("collect_cancel_time_date: 🚫 Too many parse errors → fallback")
            cancel_ctx.pop("retry_cancel_dt", None)
            session_data[call_sid]["stage"] = "cancel_appt_iterate"
            resp.say(gpt_speak(VOICE_PROMPT_SLOT_NOT_FOUND), VOICE)
            resp.redirect("/voice")
            return str(resp)

        # ------------------------------------------------------------------
        # ⏰ Check that the parsed datetime is not in the past
        # ------------------------------------------------------------------
        now_utc = datetime.utcnow().replace(tzinfo=_pytz.UTC)
        if dt_utc < now_utc:
            debug_print("collect_cancel_time_date: ⏳ Parsed time is in the past → iterate")
            session_data[call_sid]["stage"] = "cancel_appt_iterate"
            resp.say(gpt_speak(VOICE_PROMPT_PAST_TIME), VOICE)
            resp.redirect("/voice")
            return str(resp)

        # ------------------------------------------------------------------
        # 🔍 CHECK IF APPOINTMENT EXISTS IN DOCTOR'S JSON FILE
        # ------------------------------------------------------------------
        doctor_name = session_data[call_sid].get("doctor_name")

        # Compute ISO strings
        start_iso = dt_utc.isoformat().replace("+00:00", "Z")
        end_iso = (dt_utc + timedelta(minutes=30)).isoformat().replace("+00:00", "Z")

        debug_print(f"collect_cancel_time_date: 🔍 Searching JSON for: {doctor_name}, start={start_iso}")

        # Build path to doctor file
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        safe = doctor_name.lower().replace(" ", "_")
        doc_path = os.path.join(BASE_DIR, DB_FOLDER, f"{safe}.json")

        exists = False

        # Load JSON and check existence
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                appts = json.load(f)

            for ap in appts:
                ap_start = ap.get("utc_start", ap.get("start_utc", ""))
                if ap_start == start_iso:
                    exists = True
                    break

        except Exception as e:
            debug_print(f"collect_cancel_time_date: ⚠️ JSON read error → {e}")

        # ------------------------------------------------------------------
        # 🚫 If slot not found → go to listing stage
        # ------------------------------------------------------------------
        if not exists:
            debug_print("collect_cancel_time_date: 🚫 Slot not found in JSON → iterate")
            session_data[call_sid]["stage"] = "cancel_appt_iterate"
            resp.say(gpt_speak(VOICE_PROMPT_SLOT_NOT_FOUND), VOICE)
            resp.redirect("/voice")
            return str(resp)

        # ------------------------------------------------------------------
        # ✅ SUCCESS — Save parsed slot into context and move to confirmation
        # ------------------------------------------------------------------
        cancel_ctx["matching_event"] = {
            "doctor_name": doctor_name,
            "start_utc": start_iso,
            "end_utc": end_iso,
            "friendly": spoken_phrase,
            "phone_e164": cancel_ctx.get("phone_e164", ""),
            "dob": cancel_ctx.get("dob", "")
        }

        session_data[call_sid]["stage"] = "cancel_appt_confirm"

        resp.say(
            gpt_speak(f"You said {day_part} at {time_part}. Let me confirm that appointment."),
            VOICE
        )
        resp.redirect("/voice")
        return str(resp)






    elif stage == "cancel_appt_iterate":
        # ======================================================================
        # 🎯 Stage: cancel_appt_iterate
        #
        # PURPOSE:
        #   • Iterate through the doctor’s JSON appointment file.
        #   • Match by phone number + DOB.
        #   • Present each valid appointment.
        #   • Confirm → send to cancel_appt_confirm.
        #   • Skip → move to next.
        #
        # NOTE:
        #   🚫 This stage MUST NOT use is_doctor_slot_available().
        #   ✔ Cancellation must ONLY check whether the appointment EXISTS.
        # ======================================================================

        t_stage_start = _time_mod.perf_counter()
        debug_print("cancel_appt_iterate: 📍 Stage entered")

        # ----------------------------------------------------------------------
        # 💬 SPEECH MESSAGES
        # ----------------------------------------------------------------------
        VOICE_NO_FILE_MSG = "Sorry, I couldn’t find any appointment records for your doctor."
        VOICE_JSON_ERROR_MSG = "Sorry, there was a problem reading the appointment list."
        VOICE_MISSING_DOCTOR_MSG = "Sorry, I couldn’t identify which doctor your appointment was with."
        VOICE_NO_MATCH_MSG = "I couldn’t find any appointments matching your phone number and date of birth."
        VOICE_LAST_APPT_MSG = "That was the last appointment. Goodbye."
        VOICE_APPT_PROMPT_TEMPLATE = (
            "Appointment with {doctor_name} on {friendly}. "
            "Do you want to cancel this one? Say yes or no. "
            "Press 1 for yes, or 2 for no."
        )

        # ----------------------------------------------------------------------
        # 📌 Retrieve session context
        # ----------------------------------------------------------------------
        cancel_ctx = session_data[call_sid].setdefault("cancel", {})
        doctor = (
            cancel_ctx.get("doctor")
            or session_data[call_sid].get("doctor_name")
            or ""
        ).strip()
        phone_e164 = (cancel_ctx.get("phone_e164") or "").replace("+", "").lstrip("0")
        dob = (cancel_ctx.get("dob") or "").strip()

        debug_print(f"cancel_appt_iterate: inputs → doctor='{doctor}', phone='{phone_e164}', dob='{dob}'")

        # ----------------------------------------------------------------------
        # 🔎 Validate doctor
        # ----------------------------------------------------------------------
        if not doctor:
            debug_print("cancel_appt_iterate: ⚠️ Missing doctor name")
            resp.say(gpt_speak(VOICE_MISSING_DOCTOR_MSG), VOICE)
            resp.hangup()
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # 📂 Get doctor’s JSON path
        # ----------------------------------------------------------------------
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        safe_name = doctor.lower().replace(" ", "_")
        doc_path = os.path.join(BASE_DIR, DB_FOLDER, f"{safe_name}.json")
        debug_print(f"cancel_appt_iterate: 🗂 Appointment file = {doc_path}")

        # ----------------------------------------------------------------------
        # 📥 Load JSON file
        # ----------------------------------------------------------------------
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                appointments = json.load(f)
            debug_print(f"cancel_appt_iterate: 📁 Loaded {len(appointments)} appointments.")
        except FileNotFoundError:
            debug_print("cancel_appt_iterate: ❌ File not found")
            resp.say(gpt_speak(VOICE_NO_FILE_MSG), VOICE)
            resp.hangup()
            save_session(call_sid)
            return str(resp)
        except Exception as e:
            debug_print(f"cancel_appt_iterate: ❌ JSON error → {e}")
            resp.say(gpt_speak(VOICE_JSON_ERROR_MSG), VOICE)
            resp.hangup()
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # 🔍 Match appointments by PHONE + DOB
        # ----------------------------------------------------------------------
        candidates = []
        normalized_phone = _re.sub(r"\D", "", phone_e164)
        normalized_dob = _re.sub(r"[^0-9a-z]+", "", dob.replace("-", "").replace("/", ""))

        for appt in appointments:
            appt_phone = _re.sub(r"\D", "", appt.get("phone_e164", appt.get("phone", "")))
            appt_dob = _re.sub(r"[^0-9a-z]+", "", (appt.get("dob", "") or "").replace("-", "").replace("/", ""))

            phone_match = normalized_phone == appt_phone
            dob_match = not dob or normalized_dob == appt_dob

            if not (phone_match and dob_match):
                continue

            # Friendly date format for TTS
            start_iso = appt.get("utc_start", "")
            try:
                friendly = _dt.fromisoformat(start_iso.replace("Z", "+00:00")).strftime("%A, %B %d at %I:%M %p")
            except Exception:
                friendly = start_iso or "unknown time"

            candidates.append({
                "doctor_name": doctor,
                "start_utc": start_iso,
                "end_utc": appt.get("utc_end", ""),
                "friendly": friendly,
                "phone_e164": phone_e164,
                "dob": dob,
                "index_in_file": appointments.index(appt),
            })

        cancel_ctx["candidates"] = candidates
        cancel_ctx["iter_index"] = 0

        # ----------------------------------------------------------------------
        # 🚫 Nothing found
        # ----------------------------------------------------------------------
        if not candidates:
            resp.say(gpt_speak(VOICE_NO_MATCH_MSG), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            save_session(call_sid)
            return str(resp)

        # ----------------------------------------------------------------------
        # 🎤 Handle confirmation YES/NO
        # ----------------------------------------------------------------------
        dtmf = (request.values.get("Digits") or "").strip()
        utter = (speech_result or "").strip().lower()
        utter = _re.sub(r"[^a-z0-9]+", "", utter)

        YES = {"yes", "yeah", "yep", "confirm", "correct"}
        NO = {"no", "nope", "next"}

        idx = int(cancel_ctx.get("iter_index", 0))
        total = len(candidates)

        if idx >= total:
            resp.say(gpt_speak(VOICE_LAST_APPT_MSG), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            save_session(call_sid)
            return str(resp)

        cand = candidates[idx]

        # YES → Save candidate and go to confirmation
        if utter in YES or dtmf == "1":
            cancel_ctx["matching_event"] = cand
            session_data[call_sid]["stage"] = "cancel_appt_confirm"
            resp.redirect("/voice")
            save_session(call_sid)
            return str(resp)

        # NO → Move forward
        if utter in NO or dtmf == "2":
            idx += 1
            cancel_ctx["iter_index"] = idx

            if idx >= total:
                resp.say(gpt_speak(VOICE_LAST_APPT_MSG), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                save_session(call_sid)
                return str(resp)

            cand = candidates[idx]

        # ----------------------------------------------------------------------
        # 🗣️ Read the appointment aloud
        # ----------------------------------------------------------------------
        say_line = VOICE_APPT_PROMPT_TEMPLATE.format(
            doctor_name=cand["doctor_name"],
            friendly=cand["friendly"],
        )

        gather = make_gather(
            say_line,
            hints="yes no one two",
            input="speech dtmf",
            timeout=5,
            speech_timeout="auto",
            finish_on_key="#",
            action="/voice",
            barge_in=True,
        )
        resp.append(gather)

        save_session(call_sid)
        return str(resp)











    # ======================================================================
    # 🎯 STAGE: book_appt_confirm
    # ======================================================================
    # 🎯 FUNCTIONAL DESCRIPTION:
    #     This is the final confirmation stage of the booking pipeline.
    #     It finalizes either:
    #       (a) A new customer registration (no appointment yet), or
    #       (b) An appointment booking for an existing customer.
    #
    # 💡 PURPOSE:
    #     1. For new customers:
    #        - Save customer info locally (without appointment)
    #        - Instruct caller to verify with clinic manually
    #
    #     2. For current customers:
    #        - Confirm appointment slot availability
    #        - Log the appointment locally
    #        - Send SMS confirmation message
    #
    # ⚙️ TECHNICAL FEATURES:
    #     • Distinguishes between “new” and “current” customers
    #     • Converts UTC → Local timezone for human-readable confirmation
    #     • Handles missing end-time by computing default duration
    #     • Validates slot availability using `is_doctor_slot_available`
    #     • Persists appointment locally via `book_appointment_for_dr_name`
    #     • Sends SMS via Twilio API
    #
    # 💾 OUTPUTS:
    #     • Appointment stored in JSON under doctor’s file
    #     • SMS confirmation sent to customer
    #
    # 🔁 NEXT STAGES:
    #     → Hang up (booking complete)
    #
    # ======================================================================
    
    elif stage == "book_appt_confirm":

        # Start execution timer (for performance diagnostics)
        t_stage_start = _time_mod.perf_counter()
        debug_print("book_appt_confirm: 📍 Stage entered")

        # ------------------------------------------------------------------
        # 💬 VOICE MESSAGES — centralized
        # ------------------------------------------------------------------
        VOICE_REGISTER_MSG = (
            "Thank you {name}. Your registration procedure is completed. "
            "Please contact the clinic to get your PIN number and review your registration info with the clinic front disk"
            "before scheduling an appointment. Goodbye!"
        )

        VOICE_MISSING_APPT_MSG = "Sorry, appointment time is missing. Please try again."
        VOICE_CONFIRMATION_ERROR_MSG = "Sorry, we couldn't confirm the appointment time."
        VOICE_SLOT_TAKEN_MSG = "Sorry, that slot was just taken. Please choose another time."
        VOICE_APPT_CONFIRMED_MSG = (
            "Your appointment with {doctor} has been booked on {time}. "
            "We look forward to seeing you. Goodbye!"
        )

        # ------------------------------------------------------------------
        # 🧩 Retrieve session data
        # ------------------------------------------------------------------
        sd = session_data.get(call_sid, {})
        customer = sd.get("customer", {}) or {}

        origin_stage = (sd.get("origin_stage") or "").strip().lower()
        debug_print(f"book_appt_confirm: 🧭 origin_stage={origin_stage}")

        # ------------------------------------------------------------------
        # 👤 Extract full customer info
        # ------------------------------------------------------------------
        first_name       = (customer.get("first_name") or "").strip()
        last_name        = (customer.get("last_name")  or "").strip()
        customer_address = (customer.get("address")    or "").strip()
        customer_dob     = (customer.get("dob")        or "").strip()
        phone_e164       = (customer.get("phone_e164") or sd.get("phone_e164") or "").strip()
        insurance_name   = (customer.get("insurance_name") or "").strip()
        insurance_member_id = (customer.get("insurance_member_id") or "").strip()

        # ==================================================================
        # ✅ REGISTER FLOW (FORCE NEW STATUS + INSERT)
        # ==================================================================
        if origin_stage == "register":

            debug_print("book_appt_confirm: 🆕 REGISTER FLOW → forcing customer_status='new'")

            customer["customer_status"] = "new"
            sd["customer_status"] = "new"

            try:
                inserted_ok = insert_customer(
                    phone=phone_e164,
                    dob=customer_dob,
                    first_name=first_name,
                    last_name=last_name,
                    address=customer_address,
                    cc_name=f"{first_name} {last_name}".strip(),
                    cc_number="",
                    cc_exp="",
                    cc_cvv="",
                    insurance_name=insurance_name,
                    insurance_member_id=insurance_member_id,
                    customer_status="new",   # ✅ ALWAYS NEW IN REGISTER FLOW
                    pin_number=0,
                )
                debug_print(f"book_appt_confirm: ✅ insert_customer (REGISTER → new) → {inserted_ok}")
            except Exception as e:
                debug_print(f"book_appt_confirm: ❌ insert_customer failed → {e}")

            # Speak registration message and hang up
            msg = VOICE_REGISTER_MSG.format(name=first_name or "there")
            resp.say(gpt_speak(msg), VOICE)
            resp.hangup()

            session_data.pop(call_sid, None)

            debug_print(
                f"book_appt_confirm: ✅ REGISTER FLOW COMPLETE in "
                f"{_time_mod.perf_counter() - t_stage_start:.3f}s"
            )

            return str(resp)

        # ==================================================================
        # ✅ CURRENT CUSTOMER FLOW (BOOKING)
        # ==================================================================
        debug_print("book_appt_confirm: 👤 CURRENT CUSTOMER → proceed with booking")

        doctor_name = sd.get("doctor_name", "the doctor")
        appt = sd.get("appointment_time", {}) or {}
        appointment_start = appt.get("start")
        appointment_end   = appt.get("end")

        # ------------------------------------------------------------------
        # ❌ Missing appointment
        # ------------------------------------------------------------------
        if not appointment_start:
            debug_print("book_appt_confirm: ❌ appointment_start missing")
            resp.say(gpt_speak(VOICE_MISSING_APPT_MSG), VOICE)
            resp.hangup()
            return str(resp)

        # ------------------------------------------------------------------
        # 🕒 Format time in clinic timezone
        # ------------------------------------------------------------------
        tz_name = globals().get("CLINIC_TZ", "America/Chicago")
        try:
            tz = _pytz.timezone(tz_name)
        except Exception:
            tz = _pytz.timezone("America/Chicago")

        try:
            dt_utc   = datetime.fromisoformat(appointment_start.replace("Z", "+00:00"))
            dt_local = dt_utc.astimezone(tz)
            formatted_time = dt_local.strftime("%A, %B %d at %I:%M %p").replace(" 0", " ")
        except Exception as e:
            debug_print(f"book_appt_confirm: time format error → {e}")
            resp.say(gpt_speak(VOICE_CONFIRMATION_ERROR_MSG), VOICE)
            resp.hangup()
            return str(resp)

        # ------------------------------------------------------------------
        # 🕓 Compute end time if missing
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # ✅ Verify slot still available
        # ------------------------------------------------------------------
        try:
            slot_ok = is_doctor_slot_available(
                doctor_name,
                appointment_start,
                appointment_end,
            )
        except Exception as e:
            debug_print(f"book_appt_confirm: ⚠️ slot check failed → {e}")
            slot_ok = False

        if not slot_ok:
            sd["stage"] = "collect_book_time_date"
            resp.append(make_gather(VOICE_SLOT_TAKEN_MSG))
            return str(resp)

        # ------------------------------------------------------------------
        # 💾 Upsert CURRENT customer
        # ------------------------------------------------------------------
        try:
            inserted_ok = insert_customer(
                phone=phone_e164,
                dob=customer_dob,
                first_name=first_name,
                last_name=last_name,
                address=customer_address,
                cc_name=f"{first_name} {last_name}".strip(),
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

        # ------------------------------------------------------------------
        # 🗂️ Save appointment
        # ------------------------------------------------------------------
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
            debug_print(f"book_appt_confirm: ✅ Appointment logged locally")
        except Exception as e:
            debug_print(f"book_appt_confirm: ⚠️ failed logging appointment → {e}")

        # ------------------------------------------------------------------
        # ✅ Confirm appointment
        # ------------------------------------------------------------------
        msg = VOICE_APPT_CONFIRMED_MSG.format(doctor=doctor_name, time=formatted_time)
        resp.say(gpt_speak(msg), VOICE)

        try:
            sms = (
                f"Hi {first_name or 'there'}, your appointment with {doctor_name} "
                f"is confirmed on {formatted_time}. Thank you for choosing Epic Therapist Clinic."
            )
            client.messages.create(body=sms, from_=TWILIO_PHONE_NUMBER, to=phone_e164)
        except Exception as e:
            debug_print(f"book_appt_confirm: ⚠️ SMS failed → {e}")

        resp.hangup()
        session_data.pop(call_sid, None)

        debug_print(
            f"book_appt_confirm: ✅ COMPLETED in "
            f"{_time_mod.perf_counter() - t_stage_start:.3f}s"
        )

        return str(resp)








    # ======================================================================
    # 🧩 STAGE: cancel_appt_confirm
    # ======================================================================
    # 🎯 FUNCTIONAL DESCRIPTION
    # ----------------------------------------------------------------------
    # This stage finalizes the cancellation of an appointment previously
    # selected by the caller during the cancellation flow.
    #
    # It performs the following:
    #   1. Loads the doctor’s local JSON appointment file.
    #   2. Searches for the exact appointment record using:
    #         • start_utc     (mandatory match)
    #         • phone number  (normalized digits)
    #         • date of birth (optional, when provided)
    #   3. Removes the matching entry from the file.
    #   4. Confirms cancellation to the caller.
    #   5. If this cancellation is part of a *reschedule flow*, it immediately
    #      transitions to "collect_book_time_date" so the caller can speak the
    #      new desired appointment time.
    #
    # 💡 IMPORTANT:
    #   • This stage does NOT check working hours, availability, weekends, or
    #     business rules. Cancellation must rely ONLY on JSON storage.
    #   • There is NO dependency on is_doctor_slot_available().
    #   • All comparisons use raw stored UTC timestamps.
    #
    # 📥 INPUTS:
    #   cancel_ctx["matching_event"] = {
    #       "doctor_name", "start_utc", "end_utc",
    #       "friendly", "phone_e164", "dob"
    #   }
    #
    # 📤 OUTPUTS:
    #   • JSON record removed (if matched)
    #   • Spoken confirmation to user
    #   • Optional: transition to booking flow if rescheduling
    #
    # ======================================================================
    elif stage == "cancel_appt_confirm":

        # --------------------------------------------------------------
        # ⏱ Start performance timer (for debugging)
        # --------------------------------------------------------------
        t0 = _time_mod.perf_counter()
        debug_print("cancel_appt_confirm: 📍 Stage entered")

        # --------------------------------------------------------------
        # 🗂 Retrieve the session & cancellation context
        # --------------------------------------------------------------
        sd = session_data.setdefault(call_sid, {})
        cancel_ctx = sd.get("cancel", {})

        # The appointment previously selected during "cancel_appt_iterate"
        cand = cancel_ctx.get("matching_event")

        # Used to determine if we must transition to booking
        origin_stage = sd.get("origin_stage", "").strip().lower()

        # --------------------------------------------------------------
        # 🚫 SAFETY CHECK — ensure appointment exists
        # --------------------------------------------------------------
        if not cand:
            debug_print("cancel_appt_confirm: ⚠️ No appointment selected")
            resp.say(gpt_speak("Sorry, I couldn’t find that appointment to cancel."), VOICE)

            # If caller cancelled as part of reschedule → go to booking
            if origin_stage == "reschedule":
                sd["stage"] = "collect_book_time_date"
                sd["origin_stage"] = ""  # clear origin

                # Ask user to speak new date/time
                gather = make_gather(
                    "Please say the new date and time for your appointment.",
                    input="speech dtmf",
                    timeout=8,
                    speech_timeout="auto",
                    barge_in=True,
                    finish_on_key="#"
                )
                resp.append(gather)
                resp.redirect("/voice")
                return str(resp)

            # Otherwise end call
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)

        # --------------------------------------------------------------
        # 🧩 Extract fields from selected appointment
        # --------------------------------------------------------------
        doctor_name = cand.get("doctor_name", "")
        start_utc   = cand.get("start_utc", "")
        end_utc     = cand.get("end_utc", "")
        friendly    = cand.get("friendly", "")
        phone_e164  = cand.get("phone_e164", "")
        dob         = cand.get("dob", "")

        debug_print(f"cancel_appt_confirm: 🩺 Doctor={doctor_name}, Start={start_utc}")

        # --------------------------------------------------------------
        # 📁 Compute path to doctor’s JSON appointment file
        # --------------------------------------------------------------
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        safe_name = doctor_name.lower().replace(" ", "_")
        file_path = os.path.join(BASE_DIR, DB_FOLDER, f"{safe_name}.json")

        # --------------------------------------------------------------
        # 📂 Load doctor’s appointment list
        # --------------------------------------------------------------
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                appointments = json.load(f)
            debug_print(f"cancel_appt_confirm: 📁 Loaded {len(appointments)} appointments")
        except FileNotFoundError:
            debug_print(f"cancel_appt_confirm: ❌ Missing file for {doctor_name}")
            resp.say(f"I couldn’t find any appointment records for {doctor_name}.", VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)
        except Exception as e:
            debug_print(f"cancel_appt_confirm: ⚠️ Error loading JSON → {e}")
            resp.say("Sorry, something went wrong accessing appointment data.", VOICE)
            resp.hangup()
            return str(resp)

        # --------------------------------------------------------------
        # 🧹 Normalize phone + DOB for comparison
        # --------------------------------------------------------------
        norm_phone = _re.sub(r"\D", "", phone_e164)
        norm_dob   = (dob or "").strip()

        # --------------------------------------------------------------
        # 🗑 SEARCH & DELETE matching appointment
        # --------------------------------------------------------------
        deleted = False

        # Iterate over a COPY to safely remove from original
        for appt in list(appointments):

            # Normalize phone numbers before compare
            appt_phone = _re.sub(r"\D", "", appt.get("phone_e164", appt.get("phone", "")))
            appt_dob   = (appt.get("dob", "") or "").strip()

            # Match start time (stored as start_utc or utc_start)
            start_match = appt.get("start_utc", appt.get("utc_start", "")) == start_utc
            phone_match = appt_phone == norm_phone
            dob_match   = (not norm_dob) or (appt_dob == norm_dob)

            # Final match → delete this record
            if start_match and phone_match and dob_match:
                debug_print(f"cancel_appt_confirm: 🗑️ Removing entry → {appt}")
                appointments.remove(appt)
                deleted = True
                break

        # --------------------------------------------------------------
        # 💾 Write updated JSON file (if deletion happened)
        # --------------------------------------------------------------
        if deleted:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(appointments, f, indent=2)
                debug_print("cancel_appt_confirm: 🗑️ Appointment successfully removed")

                # ------------------------------------------------------
                # 🔄 If reschedule flow → transition to booking
                # ------------------------------------------------------
                if origin_stage == "reschedule":

                    combined_msg = (
                        f"Your appointment with {doctor_name} on {friendly} has been cancelled. "
                        "Let’s book your new appointment now. "
                        "Please say the new date and time."
                    )

                    gather = make_gather(
                        combined_msg,
                        input="speech dtmf",
                        timeout=10,
                        speech_timeout="auto",
                        barge_in=True,
                        finish_on_key="#",
                        action="/voice"
                    )

                    resp.append(gather)
                    resp.redirect("/voice")

                    sd["stage"] = "collect_book_time_date"
                    sd["origin_stage"] = ""
                    session_data[call_sid] = sd

                    return str(resp)

                # ------------------------------------------------------
                # ✔ NORMAL CANCELLATION — not rescheduling
                # ------------------------------------------------------
                msg = (
                    f"Your appointment with {doctor_name} on {friendly} has been cancelled. "
                    "Thank you for calling Epic Therapist Clinic."
                )
                resp.say(gpt_speak(msg), VOICE)
                resp.hangup()

                session_data.pop(call_sid, None)
                return str(resp)

            except Exception as e:
                debug_print(f"cancel_appt_confirm: ❌ failed writing JSON → {e}")
                resp.say("Sorry, there was an error removing your appointment.", VOICE)
                resp.hangup()
                return str(resp)

        # --------------------------------------------------------------
        # ❌ No appointment found → fallback
        # --------------------------------------------------------------
        debug_print("cancel_appt_confirm: ❌ No matching record found")
        resp.say("Sorry, I couldn’t find that appointment to cancel.", VOICE)
        resp.hangup()
        session_data.pop(call_sid, None)
        return str(resp)





if __name__ == "__main__":
    
    # ------------------------
    # 🔁 Call this once at startup
    # ------------------------

    load_doctor_appointments()


    app.run(host="0.0.0.0", port=5000)
