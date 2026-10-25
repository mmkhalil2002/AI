MedGemma Patient Data Importer - updated for the supplied .env

Place generate_patient_directories.py and .env together in med_data/.

.env content:
  BASE_INPUT_DIR= .\
  BASE_OUTPUT_DIR= ..\python

Both paths are resolved relative to med_data/ (the directory holding .env).
BASE_INPUT_DIR is the default external-device input directory. Press Enter at
its prompt to use this default, or enter another directory such as D:\
BASE_OUTPUT_DIR is the destination root.

Generated directory layout:
  ../python/patient_records/<patient-id>/<YYYY-MM-DD>/xrays/
  ../python/patient_records/<patient-id>/<YYYY-MM-DD>/ekgs/
  ../python/patient_records/<patient-id>/<YYYY-MM-DD>/blood_tests/

Run from Windows CMD:
  py -3 generate_patient_directories.py

Run from Ubuntu:
  python3 generate_patient_directories.py

External source subfolders accepted: xray/ (or xrays/), ekg/ (or ekgs/),
and blood test/ (or blood_tests/). Existing files are never overwritten.
Educational X-ray and EKG images are downloaded from Wikimedia Commons.
Blood-test JSON records are synthetic and generated locally, not downloaded.
Educational samples must not be used for patient diagnosis.
No extra Python packages are required. Bytecode caching is disabled in the script.
