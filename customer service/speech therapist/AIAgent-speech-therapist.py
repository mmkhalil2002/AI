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
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")
GOOGLE_CREDENTIALS = "credentials.json"
SPEECH_INPUT_DURATION = int(os.getenv("SPEECH_INPUT_DURATION", 15))
MAX_RECORD_TIME = int(os.getenv("MAX_RECORD_TIME", 60))
MAX_NUMBER_DR_RETRY = int(os.getenv("MAX_NUMBER_DR_RETRY", 3))
MAX_APPT_RETRIEVED_FROM_CALNDER = int(os.getenv("MAX_APPT_RETRIEVED_FROM_CALENDER", 50))
# 🔧 Appointment duration in minutes (can be 15, 30, 60)
APPOINTMENT_DURATION_MINUTES = 30
USE_GPT = False
VOICE = "Polly.Joanna"
# Load admin numbers and doctor mapping
with open("admin_numbers.txt") as f:
    admin_numbers = [line.strip() for line in f.readlines() if line.strip()]
"""
dr.smith@example.com:primary
dr.jones@example.com:secondary
dr.alex@example.com:backup
"""
with open("doctors.txt") as f:
    dr_google_calendar_ids = dict(line.strip().split(":") for line in f if ":" in line)
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

    # 🧽 Normalize things like "2 30" or "2, 30" → "2:30"
    text = re.sub(r"\b(\d{1,2})[,\s]+(\d{2})\b", r"\1:\2", text)

    # 🧽 Remove ordinal suffixes like "3rd", "22nd" → "3", "22"
    text = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", text, flags=re.IGNORECASE)

    # 🧼 Trim and fix format if it’s just a time
    text = re.sub(r"\s+", " ", text.strip())
    if re.match(r"\d{1,2}:\d{2}$", text):  # e.g. "2:30"
        text = "at " + text

    print(f"🧽 Cleaned time input: {text} (original was: {original_text})")

    # 🧠 Parse using dateparser (default is future)
    dt = dateparser.parse(text, settings={"PREFER_DATES_FROM": "future"})

    # ✅ Override the year to be the current year if parsed
    if dt:
        now = datetime.now()

        # If the spoken text includes a **month name**, we assume user meant *this year*
        if re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", text, re.IGNORECASE):
            dt = dt.replace(year=now.year)

    return dt






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

def extract_doctor_name(speech_text):
    """
    Use ChatGPT (GPT-3.5) to extract the doctor's name from the caller's spoken input.

    Parameters:
        speech_text (str): The full transcribed sentence spoken by the user.

    Returns:
        str: The extracted doctor name as interpreted by the GPT model.
             If GPT is unavailable or uncertain, return the original input as fallback.
    """

    # 🚀 Prompt engineering: We ask the model to extract ONLY the name
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

        # ✂️ Trim and clean up result
        extracted = extracted.replace("Dr.", "").replace("doctor", "").strip()

        return extracted

    except (APIConnectionError, AuthenticationError, RateLimitError) as e:
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
    spoken_day: Optional[str] = None,     # e.g. "Monday" or "July 14"
    spoken_time: Optional[str] = None,    # e.g. "2 PM" or "14 00"
    creds=None
) -> bool:


    """
    Cancel (delete) a Google-Calendar event that belongs to a specific phone
    number according to three fallback rules:

    1.  If BOTH `spoken_day` and `spoken_time` are provided:
        • Delete the FIRST event whose phone appears in summary/description
          AND whose weekday/date matches `spoken_day`
          AND whose clock-time matches `spoken_time`.

    2.  If ONLY `spoken_time` is provided (no `spoken_day`):
        • Delete the FIRST event whose phone matches AND whose clock-time matches.

    3.  If NEITHER day nor time is provided:
        • Delete simply the FIRST future event whose phone matches,
          regardless of date or time.

    Parameters    
    ----------
    calendar_id : str
        The Google Calendar ID for the doctor.
    phone : str
        Normalised phone digits to search for (e.g. `"01012345678"`).
    spoken_day : str | None
        Optional day or date string spoken by the caller
        ("monday", "next tuesday", "july 14", etc.)
    spoken_time : str | None
        Optional time string spoken by the caller
        ("2 pm", "14 00", etc.)
    creds : google.oauth2.credentials.Credentials | None
        Authorised credentials for Google Calendar API.

    Returns
    -------
    bool
        True  → Event found and deleted.
        False → No matching event found.
    """

    # ----- Step 1: Initialise Google Calendar service ------------------------
    calendar = build("calendar", "v3", credentials=creds)

    # ----- Step 2: Fetch up to 50 upcoming events ---------------------------
    now_iso = datetime.utcnow().isoformat() + "Z"
    events = calendar.events().list(
        calendarId=calendar_id,
        timeMin=now_iso,                               # Only future events
        maxResults=MAX_APPT_RETRIEVED_FROM_CALNDER,    # Pull a reasonable window
        singleEvents=True,                             # Expand recurring events
        orderBy="startTime"                            # Chronological order
    ).execute().get("items", [])

    # ----- Helper functions to normalise spoken day/time --------------------
    def norm_day(dt: datetime) -> str:
        # Converts dt → 'monday', 'tuesday', etc.
        return dt.strftime("%A").lower()

    def norm_time(dt: datetime) -> str:
        # Converts dt → 'h am/pm' e.g. '2 pm'
        return dt.strftime("%-I %p").lower()

    # Pre-normalise spoken inputs, if present
    target_day  = spoken_day.lower()  if spoken_day  else None
    target_time = spoken_time.lower() if spoken_time else None

    # ----- Step 3: Iterate through events and find a match ------------------
    for event in events:
        # Event start can be dateTime or date (all-day) — handle both
        start_raw = event["start"].get("dateTime") or event["start"]["date"]
        start_dt  = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))

        summary     = event.get("summary", "")
        description = event.get("description", "")

        # ➊ Phone match is MANDATORY
        if phone not in summary and phone not in description:
            continue  # skip if phone number not present

        # ➋ If caller gave a day, ensure weekday matches
        if target_day and target_day not in norm_day(start_dt):
            continue  # day mismatch → skip

        # ➌ If caller gave a time, ensure hour match (lenient string check)
        if target_time and target_time not in norm_time(start_dt):
            continue  # time mismatch → skip

        # ✅ All required conditions satisfied → delete this event
        calendar.events().delete(
            calendarId=calendar_id,
            eventId=event["id"]
        ).execute()

        return True  # success

    # 🔚 No event matched all required filters
    return False


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
                from_=TWILIO_NUMBER,         # Your Twilio phone number used to send the SMS
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
            timeout=5               # Wait up to 5 seconds for a response before timing out
           )

       # Define a friendly prompt to ask the customer what they want to do
       prompt = "would you like  to book an appointment, cancel an appointment  or leave a message."

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
        #  3. Leave a voicemail
        # ----------------------------------------------------------------------

        lower = speech_result.lower()
        print(f"📢 intent :speech_result: {lower.strip()}")
        # 🚫 Fully ignore 'hello' and similar junk — no response, no retry, no stage change
        junk_inputs = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening", "yo", "test", "1", "yes", "no"}
        if not lower.strip() or lower.strip() in junk_inputs:
            print(f"⛔ Ignored junk input: '{lower}' — re-prompting without response")
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=SPEECH_INPUT_DURATION
            )
            gather.say(gpt_speak(
                "Please tell me if you'd like to book an appointment, cancel one, or leave a message."
            ))
            resp.append(gather)
            return str(resp)

        # ✅ Booking appointment intent
        if any(word in lower for word in ["book", "booking", "appointment", "schedule", "make", "reserve", "meet"]):
            print(f"📅 Intent to book recognized → advancing to 'booking' stage")

            # Start a fresh booking session
            session_data[call_sid] = {
                "stage": "booking",
                "booking": {},            # Dictionary to store booking info
                "retry_booking": 0,       # Retry counter for doctor name
                "retry_time": 0           # Retry counter for time/date
            }

            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=SPEECH_INPUT_DURATION,
                speech_model="phone_call",  # Optimize STT for phone input
                hints=", ".join(googleid_dr_name_map.values())  # Doctor name hints to improve accuracy
            )

            # Prompt user with the list of available doctors
            doctor_list = ", ".join(googleid_dr_name_map.values())
            prompt = (
                f"Great! Let's schedule your appointment. Here is the list of doctors: {doctor_list}. "
                "Please say the name of the doctor you want to book with."
            )
            gather.say(gpt_speak(prompt),VOICE)
            resp.append(gather)
            return str(resp)

        # ✅ Voicemail intent
        elif "message" in lower or "voicemail" in lower:
            print("📩 Intent to leave a message detected → recording voicemail")

            session_data[call_sid]["stage"] = "voicemail"

            # Prompt user to leave a voicemail with details
            resp.say(gpt_speak("Please leave your name, phone number, and message after the beep."),VOICE)

            # Start recording with transcription enabled
            resp.record(
                max_length=MAX_RECORD_TIME,
                action="/voice",              # After recording, return here
                transcribe=True,
                transcribe_callback="/transcription"  # Twilio will POST transcript to this endpoint
            )
            return str(resp)

        # ✅ Cancellation intent
        elif "cancel" in lower or "reschedule" in lower or "change" in lower:
            print("❌ Intent to cancel appointment detected → entering cancellation flow")

            session_data[call_sid] = {
                "stage": "cancel_appointment",  # Start cancel flow
                "cancel": {},                   # Store cancel-related info
                "retry_booking": 0              # Counter to retry if input is unclear
            }

            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=SPEECH_INPUT_DURATION
            )

            prompt = "Sure, I can help you cancel your appointment. Please say the name of the doctor you had booked with."
            gather.say(gpt_speak(prompt),VOICE)
            resp.append(gather)
            return str(resp)

        # ❓ Fallback: unclear or unsupported intent
        else:
            print(f"❓ Unclear intent: '{lower}' → re-prompting for intent choice")
            session_data[call_sid]["stage"] = "intent"

            resp.say(gpt_speak(
                "Sorry, I didn’t catch that. Would you like to book an appointment, cancel one, or leave a message?"
            ),VOICE)
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
                timeout=SPEECH_INPUT_DURATION,
                hints=doctor_list_str
            )
            gather.say(gpt_speak("Please say your first name of the doctor you'd like to book with."),VOICE)
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
                timeout=SPEECH_INPUT_DURATION,
                hints=doctor_list_str
            )
            retry_prompt = (
                f"I couldn't match that to a doctor. Available doctors are: {doctor_list_str}. "
                "Please say your first name again."
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
        from datetime import timedelta
        #from googleapiclient.discovery import build

        # 🔧 Global setting: appointment duration (e.g. 15, 30, 60 minutes)
        APPOINTMENT_DURATION_MINUTES = 30

        # 🆔 The doctor was already selected in previous stage
        doctor_id = session_data[call_sid]["doctor_id"]

        # 🧠 Clean and parse voice input using smart_parse_time (defined outside)
        requested_dt = smart_parse_time(speech_result)
        print(f"spoken date and time: {requested_dt}")

        if not requested_dt:
            # ⚠️ Re-prompt if the time was not understood
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=SPEECH_INPUT_DURATION,
                speech_model="phone_call",
                hints="tomorrow at 2 PM, next Monday at 10 AM, Friday at 12:30"
            )
            gather.say(gpt_speak(
                "Please say the appointment date and time, like 'Tomorrow at 2 PM' or 'Friday at 12:30 in the afternoon'."
            ),VOICE)
            resp.append(gather)
            return str(resp)

        # ✅ Strip seconds and preserve hour:minute for accurate reservation
        event_start = requested_dt.replace(second=0, microsecond=0)
        event_end = event_start + timedelta(minutes=APPOINTMENT_DURATION_MINUTES)
        print(f"📅 Checking slot: {event_start} to {event_end}")

        # 📆 Query Google Calendar for this doctor
        calendar = build("calendar", "v3", credentials=creds)
        events = calendar.events().list(
            calendarId=doctor_id,
            timeMin=event_start.isoformat() + "+00:00",
            timeMax=event_end.isoformat() + "+00:00",
            singleEvents=True
        ).execute()

        if events["items"]:
            # ❌ Slot is taken — re-prompt
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=SPEECH_INPUT_DURATION,
                speech_model="phone_call"
            )
            gather.say(gpt_speak("This time is not available. Please choose another day and time."))
            resp.append(gather)
            return str(resp)

        # ✅ Slot is free — store it and proceed to collect name
        session_data[call_sid]["stage"] = "collect_first_name"
        session_data[call_sid]["appointment_time"] = {
            "start": event_start.isoformat() + "+00:00",
            "end": event_end.isoformat() + "+00:00"
        }

        # 🗓️ Format confirmation time
        friendly_name = googleid_dr_name_map[doctor_id]
        friendly_time = event_start.strftime("%A at %I:%M %p")  # e.g., Thursday at 12:30 PM

        # 🎤 Ask user for name
        gather = Gather(
            input="speech",
            action="/voice",
            method="POST",
            timeout=SPEECH_INPUT_DURATION
        )
        gather.say(gpt_speak(
            f"Your appointment with {friendly_name} is available on {friendly_time}. What is your full name, please?"
        ),VOICE)
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
            gather = Gather(input="speech", action="/voice", method="POST", timeout=SPEECH_INPUT_DURATION)
            gather.say(gpt_speak("I didn't catch that clearly. Please say your first name again."), VOICE)
            resp.append(gather)
            return str(resp)

        # ✅ Save and move to last name
        session_data[call_sid]["customer"] = {"first_name": first_name}
        session_data[call_sid]["stage"] = "collect_last_name"

        gather = Gather(input="speech", action="/voice", method="POST", timeout=SPEECH_INPUT_DURATION)
        gather.say(gpt_speak("Thank you. Now, what is your last name?"), VOICE)
        resp.append(gather)
        return str(resp)

    elif stage == "collect_first_name":
        first = speech_result.strip()
        print(f"👤 collect_first_name: {first}")
        if not first:
            gather = Gather(input="speech", action="/voice", method="POST", timeout=SPEECH_INPUT_DURATION)
            gather.say(gpt_speak("Sorry, I didn't hear your first name. Please say it again."), VOICE)
            resp.append(gather)
            return str(resp)

        session_data[call_sid]["customer"] = {"first_name": first}
        session_data[call_sid]["stage"] = "collect_last_name"

        gather = Gather(input="speech", action="/voice", method="POST", timeout=SPEECH_INPUT_DURATION)
        gather.say(gpt_speak("Thanks. And what's your last name?"), VOICE)
        resp.append(gather)
        return str(resp)

    elif stage == "collect_last_name":
        last = speech_result.strip()
        print(f"👤 collect_last_name: {last}")
        if not last:
            gather = Gather(input="speech", action="/voice", method="POST", timeout=SPEECH_INPUT_DURATION)
            gather.say(gpt_speak("Sorry, I didn't catch your last name. Please repeat it."), VOICE)
            resp.append(gather)
            return str(resp)

        session_data[call_sid]["customer"]["last_name"] = last
        session_data[call_sid]["stage"] = "collect_phone"

        gather = Gather(input="speech", action="/voice", method="POST", timeout=SPEECH_INPUT_DURATION)
        gather.say(gpt_speak("Got it. What is your phone number, please?"), VOICE)
        resp.append(gather)
        return str(resp)


    elif stage == "collect_phone":
        # ----------------------------------------------------------------------
        # 📞 Collect Phone Number
        # ----------------------------------------------------------------------
        phone = speech_result.strip()
        print(f"📱 collect_phone: Collected phone: {phone}")

        if not phone or not any(char.isdigit() for char in phone):
            gather = Gather(input="speech", action="/voice", method="POST", timeout=SPEECH_INPUT_DURATION)
            gather.say(gpt_speak("I didn’t understand your phone number. Please say it again clearly."),VOICE)
            resp.append(gather)
            return str(resp)

        # ✅ Store phone and move to address
        session_data[call_sid]["customer"]["phone"] = phone
        session_data[call_sid]["stage"] = "collect_address"

        gather = Gather(input="speech", action="/voice", method="POST", timeout=SPEECH_INPUT_DURATION)
        gather.say(gpt_speak("Got it. Now, can you please tell me your full address?"),VOICE)
        resp.append(gather)
        return str(resp)

    elif stage == "collect_address":
        # ----------------------------------------------------------------------
        # 🏠 Collect Customer Address and Finalize Appointment
        # ----------------------------------------------------------------------
        address = speech_result.strip()
        print(f"📬 collect_address: Collected address: {address}")

        session_data[call_sid]["customer"]["address"] = address
        session_data[call_sid]["stage"] = "completed"

        customer = session_data[call_sid]["customer"]
        appointment = session_data[call_sid]["appointment_time"]
        doctor_id = session_data[call_sid]["doctor_id"]

        # 🧩 Safely assemble full name from first + last name
        full_name = f"{customer.get('first_name', '')} {customer.get('last_name', '')}".strip()

        # 🛠️ Store full name for backward compatibility (optional)
        session_data[call_sid]["customer"]["name"] = full_name

        # 📅 Create appointment in Google Calendar
        calendar = build("calendar", "v3", credentials=creds)
        event = {
            "summary": f"Appointment for {full_name}",
            "description": f"Name: {full_name}\nPhone: {customer.get('phone')}\nAddress: {address}",
            "start": {"dateTime": appointment["start"], "timeZone": "America/Chicago"},
            "end": {"dateTime": appointment["end"], "timeZone": "America/Chicago"},
        }

        calendar.events().insert(calendarId=doctor_id, body=event).execute()

        # ✅ Confirm to user
        resp.say(gpt_speak(f"Thanks {full_name}. Your appointment has been confirmed. Goodbye!"), VOICE)
        return str(resp)



    elif stage == "confirmed":
        # ------------------------------------------------------------
        # 📍 We're in the final stage of booking: "confirmed"
        # At this point, we already have:
        #   - the selected doctor ID (calendar ID)
        #   - the chosen time (from previous step)
        #   - possibly other metadata like name or phone (optional)
        # So now we simply finalize the confirmation
        # ------------------------------------------------------------

        # 🆔 Get the doctor calendar ID from session
        doctor_id = session_data[call_sid].get("doctor_id")

        # 🧑‍⚕️ Get the friendly doctor name to include in voice prompt
        doctor_name = googleid_dr_name_map.get(doctor_id, "the doctor")

        # 🕐 Get the appointment start time from session to include in SMS/text
        appointment_time = session_data[call_sid].get("appointment_time", {}).get("start")
        formatted_time = ""
        if appointment_time:
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(appointment_time.replace("Z", "").replace("+00:00", ""))
                formatted_time = dt.strftime("%A, %B %d at %I:%M %p")
            except Exception as e:
                print("⚠️ Failed to parse appointment time for SMS:", e)

        # 🧾 (Optional) Extract customer info if available
        customer = session_data[call_sid].get("customer", {})
        customer_name = customer.get("name", "")
        customer_phone = customer.get("phone")

        # 🎤 Compose a confirmation message using GPT for a friendly tone
        confirmation_message = f"Your appointment with {doctor_name} has been successfully booked. We look forward to seeing you. Goodbye!"

        # 🗣️ Say the confirmation message to the caller
        resp.say(gpt_speak(confirmation_message), VOICE)

        # ------------------------------------------------------------
        # 📤 SMS confirmation message to the customer (optional but helpful)
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
                print("📩 SMS sent to customer:", customer_phone)
                print("📤 Twilio SMS SID:", message.sid)
                print("📤 Twilio SMS status:", message.status)
            except Exception as e:
                print("❌ SMS sending failed:", e)

        else:
            print("⚠️ No phone number to send SMS confirmation.")

        # 📞 End the call politely
        resp.hangup()

        # 🧹 Clear the session data so this call session doesn’t persist in memory
        session_data.pop(call_sid, None)

        # 📤 Return the TwiML <Response> to Twilio to execute the hangup and message
        return str(resp)

    
    

    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
