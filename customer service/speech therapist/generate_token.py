from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle, os

CRED_FILE = "credentials.json"
TOKEN_FILE = "token.pkl"
SCOPES = ["https://www.googleapis.com/auth/calendar"]

def main():
    if os.path.exists(TOKEN_FILE):
        print("token.pkl already exists → skip.")
        return
    flow = InstalledAppFlow.from_client_secrets_file(CRED_FILE, SCOPES)
    creds = flow.run_console()              # URL → browser → paste code
    with open(TOKEN_FILE, "wb") as f:
        pickle.dump(creds, f)
    print("✅ token.pkl saved.")

if __name__ == "__main__":  # mohamed
    main()
