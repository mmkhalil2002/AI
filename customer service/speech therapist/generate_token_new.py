# -------------------------------------------------------------
# 📜 Script: generate_token.py
# 🧠 Purpose: Authenticate with Google Calendar API and save token
# 📌 Note: This script is intended to be executed ONLY ONCE during setup.
#          It generates 'token.pkl' which is reused in production.
# -------------------------------------------------------------

import os                                  # For checking file existence
import pickle                              # For saving/loading token credentials
from google_auth_oauthlib.flow import InstalledAppFlow  # To initiate OAuth flow
from google.auth.transport.requests import Request       # For refreshing tokens

# Define the scope of access required — full Google Calendar access
SCOPES = ['https://www.googleapis.com/auth/calendar']

def main():
    creds = None
    token_path = "token.pkl"                          # File to save token
    credentials_path = "credentials.json"             # Google OAuth client file

    # ✅ Step 1: Try to load existing token
    if os.path.exists(token_path):
        with open(token_path, "rb") as token_file:
            creds = pickle.load(token_file)

    # ✅ Step 2: If no valid token, perform authentication flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # 🔄 Automatically refresh expired token if refresh token is present
            creds.refresh(Request())
        else:
            # 🌐 Perform initial OAuth flow to get new credentials
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            try:
                # Try browser-based flow (ideal on Windows or desktop)
                creds = flow.run_local_server(port=0)
            except Exception:
                # 💡 Fallback for headless servers (e.g., Ubuntu VM)
                print("⚠️ GUI browser not available. Falling back to manual console auth.")
                auth_url, _ = flow.authorization_url(prompt='consent')
                print(f"👉 Please go to this URL: {auth_url}")
                code = input("🔑 Paste the authorization code here: ")
                creds = flow.fetch_token(code=code)

        # ✅ Step 3: Save credentials to disk so this script never needs to be rerun
        with open(token_path, "wb") as token_file:
            pickle.dump(creds, token_file)

    print("✅ Token saved to token.pkl — run this script only ONCE during setup.")

if __name__ == "__main__":
    main()
