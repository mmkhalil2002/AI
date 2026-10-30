Run: py -3 generate_patient_data.py (Windows) or python3 generate_patient_data.py (Ubuntu).
Uses .env BASE_INPUT_DIR and BASE_OUTPUT_DIR relative to .env location.
The script prompts for first name, last name, and DOB (six digits: MMDDYY), then uses firstname-lastname-MMDDYY as the patient ID.
Same patient + date: adds only new files by SHA-256, never overwrites; different date: new date folder.
Internet X-rays/EKGs use Wikimedia Commons; blood tests are synthetic examples, not downloaded patient data.
