"""Load Ollama settings from .env and make real MedGemma inference requests."""

# os reads settings; Path locates .env; dotenv loads key/value pairs.
import os
from pathlib import Path
import requests
from dotenv import load_dotenv

# Read the .env next to this file even when launched from another directory.
load_dotenv(Path(__file__).resolve().parent / '.env')

# OLLAMA_URL is a base URL (e.g. http://10.0.81.107:11434), not an API path.
OLLAMA_BASE_URL = os.environ['OLLAMA_URL'].rstrip('/')
OLLAMA_GENERATE_URL = f'{OLLAMA_BASE_URL}/api/generate'

# Read the exact Ollama model tag and generation parameters from .env.
MEDGEMMA_MODEL = os.environ['MEDGEMMA_MODEL']
GENERATE_TIMEOUT = int(os.getenv('GENERATE_TIMEOUT', '1800'))
NUM_PREDICT = int(os.getenv('NUM_PREDICT', '2048'))
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.2'))
REPEAT_PENALTY = float(os.getenv('REPEAT_PENALTY', '1.2'))
REPEAT_LAST_N = int(os.getenv('REPEAT_LAST_N', '256'))


# ============================================================================
# ROUTINE: generate(prompt, images=None)
# PURPOSE: Submit text and optional complete Base64 images to Ollama/MedGemma.
# INPUTS: prompt (str), images (optional list of Base64 image strings).
# OUTPUT: MedGemma's response text; raises for HTTP/format errors.
# STEPS: Build request -> attach images -> POST -> check status -> parse JSON.
# CONFIG: Ollama IP, model, timeout, and generation options come from .env.
# PRIVACY: If Ollama runs on another host, images/data travel over the network.
# ============================================================================
def generate(prompt: str, images: list[str] | None = None) -> str:
    """Call the configured Ollama model and return its generated text."""
    # 1. Build the Ollama request. stream=False asks for one JSON response.
    #    'options' applies the settings defined in the user's .env file.
    payload = {
        'model': MEDGEMMA_MODEL,
        'prompt': prompt,
        'stream': False,
        'options': {
            'num_predict': NUM_PREDICT,
            'temperature': TEMPERATURE,
            'repeat_penalty': REPEAT_PENALTY,
            'repeat_last_n': REPEAT_LAST_N,
        },
    }

    # 2. Only image tools include the 'images' field. Blood-test calls do not.
    if images is not None:
        payload['images'] = images

    # 3. POST to the Ollama IP/port specified in .env, using its timeout.
    #    This is the actual network connection to MedGemma via Ollama.
    response = requests.post(
        OLLAMA_GENERATE_URL, json=payload, timeout=GENERATE_TIMEOUT
    )

    # 4. Surface HTTP errors (e.g., unavailable model or unreachable server).
    response.raise_for_status()

    # 5. Parse Ollama's JSON body into a Python dictionary.
    result = response.json()

    # 6. Validate that a text response exists before returning it.
    if not isinstance(result.get('response'), str):
        raise RuntimeError('Ollama response does not contain generated text')

    # 7. Return only the text; the MCP server sends it back to the client.
    return result['response']
