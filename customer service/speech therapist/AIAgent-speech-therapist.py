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
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_NUMBER")
GOOGLE_CREDENTIALS = "credentials.json"
SPEECH_INPUT_DURATION = int(os.getenv("SPEECH_INPUT_DURATION", 5))
MAX_RECORD_TIME = int(os.getenv("MAX_RECORD_TIME", 60))
MAX_NUMBER_DR_RETRY = int(os.getenv("MAX_NUMBER_DR_RETRY", 3))
MAX_APPT_RETRIEVED_FROM_CALNDER = int(os.getenv("MAX_APPT_RETRIEVED_FROM_CALENDER", 50))
# 🔧 Appointment duration in minutes (can be 15, 30, 60)
APPOINTMENT_DURATION_MINUTES = int(os.getenv("APPOINTMENT_DURATION_MINUTES", 30))
# 🌐 Global settings
MAX_TIME_SELECTION_ATTEMPTS = int(os.getenv("MAX_TIME_SELECTION_ATTEMPTS", 3))

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

from datetime import datetime, timedelta
from typing import Tuple
import re

def normalize_date_time(spoken_day: str, spoken_time: str) -> str:
    """
    Normalize spoken input like "3rd of July" → "July 3"
    Removes ordinal suffixes and rearranges if needed.
    """
    # Remove ordinal suffixes (e.g., 1st → 1, 2nd → 2, 3rd → 3, etc.)
    day = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', spoken_day.strip(), flags=re.IGNORECASE)

    # Handle "3 of July", "3rd of July" → "July 3"
    match = re.match(r"(\d+)\s+of\s+([A-Za-z]+)", day, flags=re.IGNORECASE)
    if match:
        day = f"{match.group(2)} {match.group(1)}"

    # Remove commas or prepositions
    day = day.replace(",", "").replace("of", "").strip()

    # Combine and return cleaned format
    return f"{day} {spoken_time}".strip()

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
        "%B %d %I:%M %p",     # July 3 8:30 AM
        "%B %d %H:%M",        # July 3 08:30
        "%A %B %d %I:%M %p",  # Thursday July 3 8:30 AM
        "%A %B %d %H:%M",     # Thursday July 3 08:30
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(combined, fmt)
            break
        except ValueError:
            dt = None

    if not dt:
        raise ValueError(f"Failed to parse time from: '{combined}'")

    start = dt.isoformat()
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
                from_=TWILIO_PHONE_NUMBER,         # Your Twilio phone number used to send the SMS
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
            speech_model="phone_call",  # 🎯 Optimized for voice calls
            bargeIn=True,  
            timeout=5               # Wait up to 5 seconds for a response before timing out
           )

       # Define a friendly prompt to ask the customer what they want to do
       prompt = "would you like  to book an appointment, cancel an appointment, reschedulle an appointment  or leave a message."

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
        #  3. Reschedule an appointment
        #  4. Leave a voicemail
        # ----------------------------------------------------------------------

        lower = speech_result.lower()
        print(f"📢 intent :speech_result: {lower.strip()}")

        # 🚫 Ignore junk or greeting phrases commonly returned by Twilio
        junk_inputs = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening", "yo", "test", "1", "yes", "no"}
        if not lower.strip() or lower.strip() in junk_inputs:
            print(f"⛔ Ignored junk input: '{lower}' — re-prompting without response")
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                speech_model="phone_call",
                bargeIn=True,
                timeout=SPEECH_INPUT_DURATION
            )
            gather.say(gpt_speak("Please tell me if you'd like to book an appointment, cancel one, reschedule, or leave a message."))
            resp.append(gather)
            return str(resp)

        # ✅ Rescheduling intent
        elif any(word in lower for word in ["reschedule", "change", "move"]):
            print("🔁 Intent to reschedule detected → will cancel then rebook")
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
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=SPEECH_INPUT_DURATION,
                speech_model="phone_call",
                bargeIn=True,
                hints=", ".join(doctor_names)
            )
            gather.say(gpt_speak(prompt), VOICE)
            resp.append(gather)
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
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=SPEECH_INPUT_DURATION,
                speech_model="phone_call",
                bargeIn=True,
                hints=", ".join(doctor_names)
            )
            gather.say(gpt_speak(prompt), VOICE)
            resp.append(gather)
            return str(resp)

        # ✅ Booking intent (placed **after** cancel/reschedule to avoid false positives)
        elif any(word in lower for word in ["book", "booking", "schedule", "make", "reserve", "meet"]):
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
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=SPEECH_INPUT_DURATION,
                speech_model="phone_call",
                bargeIn=True,
                hints=", ".join(googleid_dr_name_map.values())
            )
            gather.say(gpt_speak(prompt), VOICE)
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
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                speech_model="phone_call",
                bargeIn=True,
                timeout=SPEECH_INPUT_DURATION
            )
            gather.say(gpt_speak(
                "Sorry, I didn’t catch that. Would you like to book an appointment, cancel one, reschedule, or leave a message?"
            ), VOICE)
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
                            speech_model="phone_call",  # 🎯 Optimized for voice calls
                            bargeIn=True, 
                            timeout=SPEECH_INPUT_DURATION,
                            hints=doctor_list_str
                        )
            gather.say(gpt_speak("Please say the name of the doctor you'd like to book with."),VOICE)
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
                                speech_model="phone_call",  # 🎯 Optimized for voice calls
                                bargeIn=True, 
                                timeout=SPEECH_INPUT_DURATION,
                                hints=doctor_list_str
                        )
            retry_prompt = (
                f"I couldn't match that to a doctor. Available doctors are: {doctor_list_str}. "
                "Please say the doctor name again."
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
                         speech_model="phone_call",  # 🎯 Optimized for voice calls
                         bargeIn=True, 
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
        # 📅 Step 1: Try to extract the date and time from the user's response
        time_info = smart_parse_time(speech_result)

        if not time_info or not isinstance(time_info, tuple) or len(time_info) != 2:
            session_data[call_sid]["retry_time"] = session_data[call_sid].get("retry_time", 0) + 1

            if session_data[call_sid]["retry_time"] >= MAX_TIME_SELECTION_ATTEMPTS:
                resp.say(gpt_speak("Sorry, I still couldn't understand the time. Please try again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=SPEECH_INPUT_DURATION,
                speech_model="phone_call",
                bargeIn=True
            )
            gather.say(gpt_speak("Please say the date and time again, like July 3rd at 9 AM."), VOICE)
            resp.append(gather)
            return str(resp)

        # ✅ Parsed day/time
        spoken_day, spoken_time = time_info
        session_data[call_sid]["spoken_day"] = spoken_day
        session_data[call_sid]["spoken_time"] = spoken_time

        # Build start/end range
        try:
            appointment_start, appointment_end = build_timeslot_range(spoken_day, spoken_time)
        except Exception as e:
            print(f"❌ Failed to build appointment time range: {e}")
            resp.say(gpt_speak("Sorry, I couldn’t understand the time. Let’s try again later."), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)

        # 🧠 Check if the doctor is available
        doctor_id = session_data[call_sid]["doctor_id"]
        if not is_time_slot_available(doctor_id, appointment_start, appointment_end, creds):
            # Offer up to 3 alternative times
            alternatives = suggest_alternative_times(doctor_id, creds, num_options=3)
            session_data[call_sid]["retry_time"] = session_data[call_sid].get("retry_time", 0) + 1

            if not alternatives or session_data[call_sid]["retry_time"] >= MAX_TIME_SELECTION_ATTEMPTS:
                resp.say(gpt_speak("Unfortunately, the doctor isn't available at that time. Please call us again later."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # 💬 Suggest new times and ask again
            alt_text = " or ".join([format_time_for_speech(t) for t in alternatives])
            session_data[call_sid]["suggested_alternatives"] = alternatives

            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=SPEECH_INPUT_DURATION,
                speech_model="phone_call",
                bargeIn=True
            )
            gather.say(gpt_speak(f"That time is not available. Would you like to book on {alt_text}?"), VOICE)
            resp.append(gather)
            return str(resp)

        # 🎯 If available, store and move forward
        session_data[call_sid]["appointment_time"] = {
            "start": appointment_start,
            "end": appointment_end
        }
        session_data[call_sid]["stage"] = "collect_first_name"
        print(f"📆 Appointment scheduled → Start: {appointment_start}, End: {appointment_end}")

        gather = Gather(
            input="speech",
            action="/voice",
            method="POST",
            timeout=SPEECH_INPUT_DURATION,
            speech_model="phone_call",
            bargeIn=True
        )
        gather.say(gpt_speak("Thanks. What is your first name?"), VOICE)
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
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                speech_model="phone_call",
                bargeIn=True,
                timeout=SPEECH_INPUT_DURATION
            )
            gather.say(gpt_speak("I didn't catch that clearly. Please say your first name again."), VOICE)
            resp.append(gather)
            return str(resp)

        # ✅ Save and move to last name
        session_data[call_sid]["customer"] = {"first_name": first_name}
        session_data[call_sid]["stage"] = "collect_last_name"

        gather = Gather(
            input="speech",
            action="/voice",
            method="POST",
            speech_model="phone_call",
            bargeIn=True,
            timeout=SPEECH_INPUT_DURATION
        )
        gather.say(gpt_speak("Thank you. Now, what is your last name?"), VOICE)
        resp.append(gather)
        return str(resp)

    elif stage == "collect_last_name":
        # ----------------------------------------------------------------------
        # 🧍 Stage: Collect Last Name
        # ----------------------------------------------------------------------
        last = speech_result.strip()
        print(f"👤 collect_last_name: {last}")
        if not last:
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                speech_model="phone_call",
                bargeIn=True,
                timeout=SPEECH_INPUT_DURATION
            )
            gather.say(gpt_speak("Sorry, I didn't catch your last name. Please repeat it."), VOICE)
            resp.append(gather)
            return str(resp)

        session_data[call_sid]["customer"]["last_name"] = last
        session_data[call_sid]["stage"] = "collect_phone"

        gather = Gather(
            input="speech",
            action="/voice",
            method="POST",
            speech_model="phone_call",
            bargeIn=True,
            timeout=SPEECH_INPUT_DURATION
        )
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
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                speech_model="phone_call",
                bargeIn=True,
                timeout=SPEECH_INPUT_DURATION
            )
            gather.say(gpt_speak("I didn’t understand your phone number. Please say it again clearly."), VOICE)
            resp.append(gather)
            return str(resp)

        # ✅ Store phone and move to address
        session_data[call_sid]["customer"]["phone"] = phone
        session_data[call_sid]["stage"] = "collect_address"



        gather = Gather(
            input="speech",
            action="/voice",
            method="POST",
            speech_model="phone_call",
            bargeIn=True,
            timeout=SPEECH_INPUT_DURATION
        )
        gather.say(gpt_speak("Got it. tell me your full address"), VOICE)
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





    
    elif stage == "cancel_appointment":
            # ----------------------------------------------------------------------
        # 🔄 This stage handles when a user wants to cancel an appointment
        # and just spoke the doctor’s name (e.g., "Dr. Omar", or "cancel with Dr. Alex")
        # ----------------------------------------------------------------------

        selected_name = speech_result.lower()  # 🎤 Raw speech input from caller (e.g., "cancel with Dr. Omar")
        matched_id = None                      # 🧠 Variable to store matching calendar ID

        # 🔍 Step 1: Try basic match — loop through known doctors and check if the spoken input contains a known name
        for doc_id, friendly_name in googleid_dr_name_map.items():
            if friendly_name.lower() in selected_name:
                matched_id = doc_id
                break

        # 🤖 Step 2: If no match found, call GPT to extract doctor name intelligently
        if not matched_id:
            extracted_name = extract_doctor_name(speech_result)  # 🧠 GPT will try to extract just the doctor name
            if extracted_name:
                for doc_id, friendly_name in googleid_dr_name_map.items():
                    if friendly_name.lower() == extracted_name.lower():
                        matched_id = doc_id
                        break

        # ❌ Step 3: Still no match → handle retries
        if not matched_id:
            # Safely retrieve or initialize retry counter
            retries = session_data[call_sid].get("retry_booking", 0)
            session_data[call_sid]["retry_booking"] = retries + 1

            if retries >= MAX_NUMBER_DR_RETRY:
                # 🛑 Too many failed attempts → end call
                resp.say(gpt_speak(
                    "I'm sorry, I still couldn't match that name with any doctor in our clinic. Please try again later. Goodbye."
                ), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)  # 🧹 Clean up session data
                return str(resp)

            # 🔁 Less than 3 attempts → re-ask with full doctor list
            gather = Gather(
                input="speech", 
                action="/voice",
                method="POST",
                speech_model="phone_call",
                bargeIn=True, 
                timeout=SPEECH_INPUT_DURATION
            )

            # 🧾 Convert the list of available doctors to a comma-separated string
            doctor_list = ", ".join(googleid_dr_name_map.values())
            retry_prompt = (
                f"I didn't recognize that name. Available doctors are: {doctor_list}. "
                "Please say the name again."
            )

            try:
                gather.say(gpt_speak(retry_prompt), VOICE)
            except Exception as e:
                print(f"⚠️ GPT error in retry_prompt fallback: {e}")
                gather.say(retry_prompt, VOICE)

            resp.append(gather)
            return str(resp)

        # ✅ Step 4: Doctor matched — store info and ask for phone number
        session_data[call_sid]["cancel"]["doctor"] = googleid_dr_name_map[matched_id]  # Save friendly name
        session_data[call_sid]["doctor_id"] = matched_id                                # Save Google Calendar ID
        session_data[call_sid]["stage"] = "cancel_appt_by_phone_number"                 # Advance to phone entry

        # 📞 Prompt user for the phone number they used when booking
        gather = Gather(
            input="speech", 
            action="/voice",
            method="POST",
            speech_model="phone_call",
            bargeIn=True, 
            timeout=SPEECH_INPUT_DURATION
        )

        gather.say(gpt_speak("Thanks. What phone number did you use when booking the appointment?"), VOICE)
        resp.append(gather)
        return str(resp)


    elif stage == "cancel_appt_by_phone_number":
        # ----------------------------------------------------------------------
        # 📞 Step 1: Extract the phone number and store it
        # ----------------------------------------------------------------------
        phone = extract_phone_number(speech_result)
        print(f"📱 Extracted phone → {phone}")
        session_data[call_sid]["cancel"]["phone"] = phone

        # ----------------------------------------------------------------------
        # 🗣️ Step 2: Prompt user to now provide the date and time
        # ----------------------------------------------------------------------
        session_data[call_sid]["stage"] = "cancel_appt_get_date"

        gather = Gather(
            input="speech",
            action="/voice",
            method="POST",
            timeout=SPEECH_INPUT_DURATION,
            speech_model="phone_call",
            bargeIn=True
        )
        gather.say(gpt_speak("Thanks. Now, please tell me the date and time of the appointment you want to cancel. For example, say July 3rd at 9 AM."), VOICE)
        resp.append(gather)
        return str(resp)





    elif stage == "cancel_appt_get_date":
        # ----------------------------------------------------------------------
        # 🧠 Step 1: Try extracting spoken date and time using smart_parse_time
        # ----------------------------------------------------------------------
        time_info = smart_parse_time(speech_result)

        if not time_info or not isinstance(time_info, tuple) or len(time_info) != 2:
            # 🛑 Failed to extract → ask the user again
            session_data[call_sid]["retry_time"] = session_data[call_sid].get("retry_time", 0) + 1

            if session_data[call_sid]["retry_time"] >= 3:
                resp.say(gpt_speak("Sorry, I couldn't understand the time. Please try again later. Goodbye."), VOICE)
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # 🔁 Prompt again for date/time
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=SPEECH_INPUT_DURATION,
                speech_model="phone_call",
                bargeIn=True
            )
            gather.say(gpt_speak(
                "Sorry, I didn’t catch that. Please say the date and time of the appointment you want to cancel, like July 3rd at 9 AM."
            ), VOICE)
            resp.append(gather)
            return str(resp)

        # ✅ Extracted values from speech
        spoken_day, spoken_time = time_info
        print(f"📆 Extracted → Day: {spoken_day}, Time: {spoken_time}")
        session_data[call_sid]["cancel"]["day"] = spoken_day
        session_data[call_sid]["cancel"]["time"] = spoken_time

        # ----------------------------------------------------------------------
        # 🔄 Get stored phone number and doctor name
        # ----------------------------------------------------------------------
        phone = session_data[call_sid]["cancel"].get("phone")
        doctor = session_data[call_sid]["cancel"].get("doctor")
        print(f"📱 Using phone → {phone}")
        print(f"👨‍⚕️ Using doctor → {doctor}")

        # ----------------------------------------------------------------------
        # 🔍 Find calendar ID from friendly doctor name
        # ----------------------------------------------------------------------
        calendar_id = None
        for doc_id, friendly in googleid_dr_name_map.items():
            if friendly.lower() == doctor.lower():
                calendar_id = doc_id
                break

        if not calendar_id:
            resp.say(gpt_speak("Sorry, I couldn't find the doctor in our system. Please try again later."), VOICE)
            resp.hangup()
            session_data.pop(call_sid, None)
            return str(resp)

        # ----------------------------------------------------------------------
        # 🗑️ Attempt cancellation and get full event if found
        # ----------------------------------------------------------------------
        canceled_event = cancel_event_by_phone(
            calendar_id=calendar_id,
            phone=phone,
            spoken_day=spoken_day,
            spoken_time=spoken_time,
            creds=creds,
            return_details=True  # ✅ now returns the deleted event object if successful
        )

        if canceled_event:
            from dateutil import parser
            try:
                # 🧾 Format the spoken cancellation time for confirmation
                start = canceled_event.get("start", {})
                start_str = start.get("dateTime", "") or start.get("date", "")
                if start_str:
                    dt = parser.parse(start_str)
                    spoken_time_str = dt.strftime("%B %-d at %-I:%M %p")
                else:
                    spoken_time_str = f"{spoken_day} at {spoken_time}"
            except Exception as e:
                print(f"⚠️ Failed to parse canceled event time: {e}")
                spoken_time_str = f"{spoken_day} at {spoken_time}"

            # 🗣️ Confirm cancellation with friendly voice message
            msg = f"Your appointment with {doctor} on {spoken_time_str} has been cancelled. Thank you!"
            resp.say(gpt_speak(msg), VOICE)
        else:
            # ❌ No matching appointment found
            resp.say(gpt_speak("I'm sorry, I couldn't find an appointment under that phone number and time."), VOICE)

        session_data.pop(call_sid, None)
        return str(resp)



      
    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
