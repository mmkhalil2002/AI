"""Desktop GUI client for a MedGemma model served by Ollama.

Run with: python medgemma_client_commented.py

This program displays a Tkinter window on a computer. It is not an Android APK.
Ollama and the model run on another computer (or on this same computer).
Only the question and answer travel over HTTP.
"""

# json converts Python dictionaries to the format Ollama expects and parses its reply.
import json

# A background thread lets the GUI remain responsive while Ollama generates text.
import threading

# tkinter supplies the window, labels, buttons, and multiline text widgets.
import tkinter as tk
from tkinter import scrolledtext

# urllib is included with Python; no additional HTTP package is required.
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# Replace this IP if the address of the Windows computer running Ollama changes.
# The phone/computer running this client must be able to reach this address.
# Port 11434 is Ollama's standard port; /api/generate accepts a text prompt.
MEDGEMMA_URL = "http://10.0.81.181:11434/api/generate"

# Use the exact model name shown by `ollama list` on the Ollama computer.
MODEL_NAME = "medgemma1.5"


def send_question(question: str) -> str:
    """Send a prompt to Ollama and return the model's answer as plain text.

    This function performs blocking network I/O, so call it from a background
    thread rather than directly from a Tkinter button callback.
    """

    # Prepend the same educational instruction used in the Kotlin version.
    # The user's question follows after a blank line.
    complete_prompt = (
        "This is for educational demonstration only. "
        "Provide a clear educational explanation. "
        "Do not provide a personal diagnosis or treatment plan.\n\n"
        + question
    )

    # Ollama expects a JSON object containing the model and prompt.
    # stream=False requests one complete response, which is easier to parse
    # than a sequence of partial JSON messages.
    request_data = {
        "model": MODEL_NAME,
        "prompt": complete_prompt,
        "stream": False,
        "options": {
            "num_predict": 300,  # Approximate maximum number of output tokens.
            "temperature": 0.1,  # Lower values tend to make answers steadier.
        },
    }

    # Convert the dictionary to UTF-8 JSON bytes for the HTTP request body.
    # The Content-Type header tells Ollama how to interpret those bytes.
    request = Request(
        MEDGEMMA_URL,
        data=json.dumps(request_data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        # Wait for Ollama and read its complete JSON response. CPU inference
        # can be slow, so the socket timeout is set to five minutes.
        # The with block closes the response after it has been read.
        with urlopen(request, timeout=300) as connection:
            response_body = connection.read().decode("utf-8")

    except HTTPError as error:
        # The server responded, but returned an error code such as 404 or 500.
        # Include Ollama's response body to help identify the problem.
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama returned HTTP {error.code}: {details}"
        ) from error

    except URLError as error:
        # No usable HTTP response was received. Possible causes include an
        # unreachable IP address, firewall rules, or Ollama not listening.
        raise RuntimeError(
            f"Could not connect to Ollama: {error.reason}"
        ) from error

    # Ollama /api/generate returns a JSON object with a "response" field.
    # Extract that field and remove whitespace around the generated answer.
    response_json = json.loads(response_body)
    answer = response_json.get("response", "").strip()

    # An empty answer is reported as an error so the GUI does not silently
    # show a blank response box after a seemingly successful request.
    if not answer:
        raise RuntimeError("MedGemma returned an empty response.")

    return answer


def show_result(status: str, message: str) -> None:
    """Display a finished answer or error in the Tkinter window.

    Tkinter widgets must be changed from the GUI's main thread. The worker
    schedules this function with window.after rather than calling it itself.
    """
    status_label.config(text=status)
    response_box.delete("1.0", tk.END)  # Clear all previous response text.
    response_box.insert(tk.END, message)
    send_button.config(state=tk.NORMAL)  # Allow the next question.


def process_question(question: str) -> None:
    """Run one Ollama request in a background thread."""
    try:
        answer = send_question(question)

        # Schedule the screen update on the main Tkinter event loop.
        window.after(0, show_result, "Completed", answer)

    except Exception as error:
        # Display connection errors, malformed JSON, and other failures.
        # Use a default argument to capture the error text before this
        # exception handler finishes and its local `error` is cleared.
        message = f"Error: {error}"
        window.after(0, show_result, "Request failed", message)


def on_send() -> None:
    """Read the user's question when Send to MedGemma is pressed."""
    # Tkinter text positions begin at "1.0" (line 1, character 0).
    # Strip removes trailing newline and spaces inserted by the text box.
    question = prompt_box.get("1.0", tk.END).strip()

    # Avoid sending an empty request to Ollama.
    if not question:
        status_label.config(text="Please enter a question.")
        return

    # Immediately show progress and prevent repeated clicks during this call.
    status_label.config(text="MedGemma is processing...")
    response_box.delete("1.0", tk.END)
    send_button.config(state=tk.DISABLED)

    # A daemon thread performs the slow request without freezing the window.
    # `args` passes the question to process_question; start begins the thread.
    threading.Thread(
        target=process_question,
        args=(question,),
        daemon=True,
    ).start()


# Create the desktop window and choose its title and initial size in pixels.
window = tk.Tk()
window.title("MedGemma Client")
window.geometry("700x550")

# Input label and multiline question box. pack arranges widgets vertically.
tk.Label(window, text="Enter your question:").pack(
    anchor="w", padx=12, pady=(12, 4)
)
prompt_box = scrolledtext.ScrolledText(window, height=6, wrap=tk.WORD)
prompt_box.pack(fill=tk.X, padx=12)

# Pressing this button invokes on_send(). It is disabled while a request runs.
send_button = tk.Button(window, text="Send to MedGemma", command=on_send)
send_button.pack(pady=10)

# The status label shows Ready, processing, completed, or an error state.
status_label = tk.Label(window, text="Ready")
status_label.pack(anchor="w", padx=12)

# The lower text box displays the generated answer or an error message.
tk.Label(window, text="MedGemma response:").pack(
    anchor="w", padx=12, pady=(12, 4)
)
response_box = scrolledtext.ScrolledText(window, wrap=tk.WORD)
response_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

# Start Tkinter's event loop so the window stays open and handles clicks.
window.mainloop()