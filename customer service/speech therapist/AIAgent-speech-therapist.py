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

# ✅ OpenAI client initialization
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ Cache dictionary
prompt_cache = {}

def fallback_response(prompt):
    """
    Rule-based fallback if GPT is not available or quota is exceeded.
    Covers greetings, booking, canceling, and voicemail intent using common phrases.
    Also responds to requests mentioning the doctor list.
    """
    prompt_lower = prompt.lower()

    # Greeting-related keywords
    greeting_keywords = ["hello", "hi", "good morning", "good afternoon", "good evening", "hey", "greetings", "salaam", "marhaba"]

    # Booking-related keywords
    booking_keywords = ["book", "make", "schedule", "appointment", "visit", "slot", "reserve"]

    # Canceling-related keywords
    cancel_keywords = ["cancel", "reschedule", "change", "remove", "delete"]

    # Voicemail-related keywords
    voicemail_keywords = ["message", "voicemail", "leave", "say something", "record", "note"]

    # 📋 Doctor listing logic
    if "list of doctors" in prompt_lower or "available doctors" in prompt_lower:
        doctor_names = ", ".join(googleid_dr_name_map.values())
        return f"The available doctors are: {doctor_names}. Please say the name of the doctor you'd like to book with."

    # Rule-based intent matching
    if any(kw in prompt_lower for kw in greeting_keywords):
        return "This is an AI Agen Welcome to Epic Therapist Clinic! Would you like to book an appointment, cancel one, or leave a message?"
    elif any(kw in prompt_lower for kw in booking_keywords):
        return "Sure, I can help you book an appointment. Please tell me the doctor name, here is the doctors list."
    elif any(kw in prompt_lower for kw in cancel_keywords):
        return "Okay, I can help cancel your appointment. Can you please tell me your name and appointment time?"
    elif any(kw in prompt_lower for kw in voicemail_keywords):
        return "Alright, please leave your message after the beep."
    else:
        #return "Sorry, I didn’t understand. Would you like to book, cancel, or leave a message?"
        return prompt

def gpt_speak(prompt):
    """
    Tries to use GPT to answer a prompt. Falls back to rule-based logic on error or quota limits.
    Caches responses to avoid duplicate API calls.
    """
    print(f"📨 Prompt: {prompt}")
    print(f"🔑 Using API Key (first 8 chars): {OPENAI_API_KEY[:8] if OPENAI_API_KEY else 'Not set'}")

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
from googleapiclient.discovery import build


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
       prompt = "Greet the customer cheerfully and ask if they want to book an appointment, cancel an appointment  or leave a message."

       # Use GPT to generate a dynamic and friendly greeting based on the prompt
       gather.say(gpt_speak(prompt))  # This adds spoken text to the <Gather> block
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
        lower = speech_result.lower()
       # Check if the word "message" appears in the user's spoken input (case-insensitive)
        # ✅ Booking appointment intent
        # 👋 Avoid mistaking greetings for intent
        greeting_words = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
        if lower in greeting_words:
            resp.say(gpt_speak("Would you like to book an appointment, cancel one, or leave a message?"))
            session_data[call_sid]["stage"] = "intent"
            return str(resp)

        # ✅ Booking appointment intent
        # Check if the word "book" or related terms appear in the user's spoken input
        if any(word in lower for word in ["book", "booking", "appointment", "schedule", "make", "reserve", "meet"]):

            print(f"will go to booking")
            session_data[call_sid] = {
                "stage": "booking",
                "booking": {},
                "retry_booking": 0,
                "retry_time": 0
                }

            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=SPEECH_INPUT_DURATION,
                speech_model="phone_call",  # Optimize recognition for calls
                hints=", ".join(googleid_dr_name_map.values())  # List of doctor names
                )


            # Prompt with the list of available doctors
            doctor_list = ", ".join(googleid_dr_name_map.values())
            prompt = f"Great! Let's schedule your appointment. Here is the list of doctors: {doctor_list}. Please say the name of the doctor you want to book with."
            gather.say(gpt_speak(prompt))
            resp.append(gather)
            return str(resp)

        elif "message" in lower or "voicemail" in lower:

            # Update the session stage for this call to "voicemail"
            # This is useful to track that the customer chose to leave a message
            session_data[call_sid]["stage"] = "voicemail"

            # Add a spoken prompt to the response telling the customer what to do
            resp.say(gpt_speak("Please leave your name, phone number, and message after the beep."))
            """
             📞 The caller leaves a voice message.

             🔊 Twilio records the audio.

             🧠 Twilio uses speech-to-text to transcribe what was said.

             📬 It then sends a POST request to your /transcription endpoint with this transcription.
            """
            # Start recording the caller's voice message
            # - max_length=60 means the recording will stop after 60 seconds
            # - transcribe=True enables automatic transcription by Twilio
            # - transcribe_callback sets the URL that Twilio will call with the transcribed text
            resp.record(
                max_length=MAX_RECORD_TIME,
                action="/voice",  # After recording, Twilio will call this route
                transcribe=True,
                transcribe_callback="/transcription"
             )

            # Convert the VoiceResponse object to an XML string and return it to Twilio
            return str(resp)

        elif "cancel" in lower or "reschedule" in lower or "change" in lower:
            # 🧠 Start a new session for cancellation
            session_data[call_sid] = {
                "stage": "cancel_appointment",  # Stage to collect doctor's name
                "cancel": {},             # Holds cancel-related info
                "retry_booking": 0         # Retry attempts allowed
                }

            # 🎤 Ask for the doctor's name
            gather = Gather(
                    input="speech",
                    action="/voice",
                    method="POST",
                    timeout=SPEECH_INPUT_DURATION
                )

            prompt = "Sure, I can help you cancel your appointment. Please say the name of the doctor you had booked with."
            gather.say(gpt_speak(prompt))
            resp.append(gather)
            return str(resp)

        # ❓ Fallback: unclear intent
        else:
            resp.say(gpt_speak("Sorry, I didn’t catch that. Would you like to book an appointment, cancel one, or leave a message?"))
            session_data[call_sid]["stage"] = "intent"
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
                session_data[call_sid]["retry_booking"]
                retries = session_data[call_sid]["retry_booking"]

                if retries >= MAX_NUMBER_DR_RETRY:
                    # 🛑 Too many failed attempts → end call
                    resp.say(gpt_speak(
                        "I'm sorry, I still couldn't match that name with any doctor in our clinic. Please try again later. Goodbye."
                    ))
                    resp.hangup()
                    session_data.pop(call_sid, None)  # 🧹 Clean up session data
                    return str(resp)

                # 🔁 Less than 3 attempts → re-ask with full doctor list
                gather = Gather(input="speech", action="/voice", method="POST", timeout=SPEECH_INPUT_DURATION)
                # 🧾 Convert the list of available doctors to a comma-separated string
                """
                    googleid_dr_name_map = {
                    "dr.smith@example.com": "Dr. Smith",
                    "dr.jones@example.com": "Dr. Jones",
                    "dr.alex@example.com": "Dr. Alex",
                    "dr.mariam@example.com": "Dr. Mariam"
                }

                doctor list is 

                Dr. Smith, Dr. Jones, Dr. Alex, Dr. Mariam

                """

                doctor_list = ", ".join(googleid_dr_name_map.values())
                retry_prompt = (
                    f"I didn't recognize that name. Available doctors are: {doctor_list}. "
                    "Please say the name again."
                )
                gather.say(gpt_speak(retry_prompt))
                resp.append(gather)
                return str(resp)

            # ✅ Step 4: Doctor matched — store info and ask for phone number
            session_data[call_sid]["cancel"]["doctor"] = googleid_dr_name_map[matched_id]  # Save friendly name
            session_data[call_sid]["doctor_id"] = matched_id                                # Save Google Calendar ID
            session_data[call_sid]["stage"] = "cancel_appt_by_phone_number"                 # Advance to phone entry

            # 📞 Prompt user for the phone number they used when booking
            gather = Gather(input="speech", action="/voice", method="POST", timeout=SPEECH_INPUT_DURATION)
            gather.say(gpt_speak("Thanks. What phone number did you use when booking the appointment?"))
            resp.append(gather)
            return str(resp)


    elif stage == "cancel_appt_by_phone_number":
            # ----------------------------------------------------------------------
            # 📞 Step 1: Extract the phone number from the caller's spoken response
            # ----------------------------------------------------------------------
            # speech_result is deined in voice procedure
            phone = extract_phone_number(speech_result)  # e.g., "01012345678"

            # ----------------------------------------------------------------------
            # 🧠 Step 2: Retrieve the doctor’s name previously saved in session
            # This was stored during the "cancel_phone" stage
            # ----------------------------------------------------------------------
            doctor = session_data[call_sid]["cancel"].get("doctor")

            # ----------------------------------------------------------------------
            # 🔍 Step 3: Map the doctor name to the corresponding Google Calendar ID
            # This allows us to locate the correct calendar for the cancellation
            # ----------------------------------------------------------------------
            calendar_id = googleid_dr_name_map.get(doctor.lower())

            # ----------------------------------------------------------------------
            # 📅 Step 4: Optionally retrieve day/time spoken earlier (if collected)
            # These can help disambiguate which appointment to cancel
            # ----------------------------------------------------------------------
            spoken_day  = session_data[call_sid]["cancel"].get("day")      # e.g., "Monday" or "July 14"
            spoken_time = session_data[call_sid]["cancel"].get("time")     # e.g., "2 PM" or "14 00"

            # ----------------------------------------------------------------------
            # ❌ Step 5: Safety check – in case doctor was not matched properly
            # This usually shouldn't happen, but we fail gracefully
            # ----------------------------------------------------------------------
            if not calendar_id:
                resp.say(gpt_speak(
                    "Sorry, I couldn't find that doctor in our clinic system. "
                    "Please start over and try again."
                ))
                session_data.pop(call_sid, None)
                return str(resp)

            # ----------------------------------------------------------------------
            # ✅ Step 6: Attempt to cancel the appointment by phone (and optionally day/time)
            # This uses our backend helper cancel_event_by_phone()
            # ----------------------------------------------------------------------
            success = cancel_event_by_phone(
                calendar_id=calendar_id,
                phone=phone,
                spoken_day=spoken_day,
                spoken_time=spoken_time,
                creds=creds
            )

            # ----------------------------------------------------------------------
            # 🎉 Step 7: Respond to user based on result
            # ----------------------------------------------------------------------
            if success:
                # 📢 Success message
                resp.say(gpt_speak(
                    f"Your appointment with {doctor} has been cancelled. Thank you for calling!"
                ))
            else:
                # ⚠️ Failure message — no matching appointment found
                resp.say(gpt_speak(
                    "I'm sorry, I couldn't find any appointment under that phone number. "
                    "Please contact the clinic directly for help."
                ))

            # ----------------------------------------------------------------------
            # 🧹 Step 8: Clear session data after processing
            # ----------------------------------------------------------------------
            session_data.pop(call_sid, None)

            # ----------------------------------------------------------------------
            # 📤 Step 9: Return TwiML to Twilio to speak the result
            # ----------------------------------------------------------------------
            return str(resp)


    
   

    elif stage == "booking":
        # ----------------------------------------------------------------------
        # 📍 Booking flow: the caller has just been asked to name a doctor.
        # Our task here is to identify which doctor they said and, if successful,
        # proceed to ask what time they’d like to book.
        # ----------------------------------------------------------------------

        # ✅ Initialize retry counter for booking stage if not already present
        # This prevents KeyError when incrementing retries below.
        if "retry_booking" not in session_data[call_sid]:
            session_data[call_sid]["retry_booking"] = 0  # 🔁 Used to limit failed attempts

        # 🗣️ 1) Capture the caller’s speech and normalize to lowercase
        spoken_text = speech_result.lower()
        print(f"dr name {spoken_text}")
        spoken_norm = normalize(spoken_text)  # 🧽 Cleaned version for matching
        matched_id = None  # Will hold the Google Calendar ID if matched

        # ------------------------------------------------------------------
        # 🔍 2) FAST MATCH: Try direct or substring match first
        # ------------------------------------------------------------------
        for doc_id, friendly in googleid_dr_name_map.items():
            friendly_norm = normalize(friendly)
            # ✅ Match if full name in speech OR speech in full name
            if friendly_norm in spoken_norm or spoken_norm in friendly_norm:
                matched_id = doc_id
                break

        # ------------------------------------------------------------------
        # 🔁 2b) TOKEN MATCH: Try token-level match if above fails
        # ------------------------------------------------------------------
        if matched_id is None:
            for doc_id, friendly in googleid_dr_name_map.items():
                friendly_norm = normalize(friendly)
                for token in spoken_norm.split():
                    if token in friendly_norm:
                        matched_id = doc_id
                        break
                if matched_id:
                    break

        # ------------------------------------------------------------------
        # 🤖 3) FALLBACK: Use GPT to extract doctor name if still unmatched
        # ------------------------------------------------------------------
        if matched_id is None:
            extracted = extract_doctor_name(speech_result)  # e.g., returns "Dr. Alaa"
            if extracted:
                extracted_norm = normalize(extracted)
                for doc_id, friendly in googleid_dr_name_map.items():
                    friendly_norm = normalize(friendly)
                    # ✅ Match if overlap between extracted name and stored name
                    if extracted_norm in friendly_norm or friendly_norm in extracted_norm:
                        matched_id = doc_id
                        break
                if matched_id is None:
                    # Try token match on extracted name
                    for doc_id, friendly in googleid_dr_name_map.items():
                        friendly_norm = normalize(friendly)
                        for token in extracted_norm.split():
                            if token in friendly_norm:
                                matched_id = doc_id
                                break
                        if matched_id:
                            break

        # ------------------------------------------------------------------
        # ❌ 4) STILL NO MATCH  → handle retries (up to three attempts)
        # ------------------------------------------------------------------
        if matched_id is None:
            session_data[call_sid]["retry_booking"] += 1
            retries = session_data[call_sid]["retry_booking"]

            if retries >= 3:
                # 🚫 Too many failed attempts — end the call
                resp.say(gpt_speak(
                    "I'm sorry, I still couldn't match that name with any doctor in our clinic. "
                    "Please call us again when convenient. Goodbye."
                ))
                resp.hangup()
                session_data.pop(call_sid, None)
                return str(resp)

            # 🔁 Prompt user again with available doctor names
            gather = Gather(input="speech", action="/voice", method="POST", timeout=SPEECH_INPUT_DURATION)
            doctor_list = ", ".join(googleid_dr_name_map.values())
            retry_prompt = (
                f"I didn't recognize that name. Available doctors are: {doctor_list}. "
                "Please say the doctor's name again."
            )
            gather.say(gpt_speak(retry_prompt))
            resp.append(gather)
            return str(resp)

        # ------------------------------------------------------------------
        # ✅ 5) MATCH SUCCESS → store doctor info and proceed to ask time
        # ------------------------------------------------------------------
        print ("doctor found matched id {matched_id} go to time and date")
        session_data[call_sid]["doctor_id"] = matched_id
        session_data[call_sid]["stage"] = "ask_time_date"  # ⏭️ Next stage: ask for time

        # 🗓️ Prompt the user for preferred appointment time
        gather = Gather(input="speech", action="/voice", method="POST", timeout=SPEECH_INPUT_DURATION)
        friendly_name = googleid_dr_name_map[matched_id]
        time_prompt = f"What time would you like to book with {friendly_name}?"
        gather.say(gpt_speak(time_prompt))
        resp.append(gather)
        return str(resp)



    elif stage == "ask_time_date":
        # ----------------------------------------------------------------------
        # 📍 Time & Date Collection Stage:
        # This stage is reached *after* a doctor has already been selected.
        # Our task here is to collect the desired appointment date & time,
        # check for availability, and either confirm or re-prompt.
        # ----------------------------------------------------------------------

        # 🆔 Retrieve the previously selected doctor's calendar ID from session
        doctor_id = session_data[call_sid]["doctor_id"]

        # 🕒 Try to parse the user's spoken time and date using dateparser
        requested_dt = dateparser.parse(speech_result)
        print(f"spoken date and time: {requested_dt}")

        if not requested_dt:
            # ⚠️ If parsing fails, prompt the user again (no need to list doctors again)
            gather = Gather(
                   input="speech",
                   action="/voice",
                   method="POST",
                   timeout=SPEECH_INPUT_DURATION,
                   speech_model="phone_call",  # 📞 Improve transcription for phone calls
                   hints=(
                            "August 5th at 10 AM, July 30 at 3 PM, next Monday at 4 PM, "
                            "Tuesday August 13th at 2 PM, Friday at 9 in the morning, "
                            "September 1st at 11 AM"
                        )  # 🧠 Help Twilio expect full date/time expressions
                    )


            # 🗣️ Improved voice prompt for clearer instructions
            gather.say(gpt_speak(
                 "Please say the appointment date and time. You can say things like "
                 "'August fifth at ten A M', 'July thirtieth at three P M', or 'Monday August twelfth at four in the afternoon'."
                ))
            
            
            resp.append(gather)
            return str(resp)

        # ⏰ Round down to the top of the hour (for 30-minute slots)
        event_start = requested_dt.replace(minute=0, second=0, microsecond=0)
        event_end = event_start + timedelta(minutes=30)

        # 📆 Connect to Google Calendar API
        calendar = build("calendar", "v3", credentials=creds)

        # 🔍 Check for any existing events that overlap with this time
        events = calendar.events().list(
            calendarId=doctor_id,
            timeMin=event_start.isoformat() + "Z",
            timeMax=event_end.isoformat() + "Z",
            singleEvents=True
        ).execute()

        if events["items"]:
            # ❌ Time slot is already taken — ask again (no need to repeat doctor)
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=SPEECH_INPUT_DURATION,
                speech_model="phone_call",
                hints="2 PM, 10 AM, next Tuesday"
            )
            gather.say(gpt_speak("This time is not available. Please choose another day and time."))
            resp.append(gather)
            return str(resp)

        # ✅ No conflict — save and confirm booking
        session_data[call_sid]["stage"] = "confirmed"

        # 📋 Create the calendar event
        event = {
            "summary": f"Appointment for {call_sid}",
            "start": {"dateTime": event_start.isoformat(), "timeZone": "UTC"},
            "end": {"dateTime": event_end.isoformat(), "timeZone": "UTC"},
        }

        calendar.events().insert(calendarId=doctor_id, body=event).execute()

        # 🔊 Confirm the appointment to the caller
        friendly_time = event_start.strftime("%A at %I %p")
        friendly_name = googleid_dr_name_map[doctor_id]
        resp.say(gpt_speak(f"Your appointment with {friendly_name} is confirmed on {friendly_time}. Thank you!"))

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

        # 🧾 (Optional) Extract previously stored event start time if you saved it,
        # but here we'll just tell the user the confirmation message again.
        # You could store event_start in session if needed for more precision.

        # 🧑‍⚕️ Get the friendly doctor name to include in voice prompt
        doctor_name = googleid_dr_name_map.get(doctor_id, "the doctor")

        # 🎤 Compose a confirmation message using GPT for a friendly tone
        confirmation_message = f"Your appointment with {doctor_name} has been successfully booked. We look forward to seeing you. Goodbye!"

        # 🗣️ Say the confirmation message to the caller
        resp.say(gpt_speak(confirmation_message))

        # 📞 End the call politely
        resp.hangup()

        # 🧹 Clear the session data so this call session doesn’t persist in memory
        session_data.pop(call_sid, None)

        # 📤 Return the TwiML <Response> to Twilio to execute the hangup and message
        return str(resp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
