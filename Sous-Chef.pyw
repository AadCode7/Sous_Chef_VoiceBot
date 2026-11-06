"""Install once:
    pip install soundcard numpy faster-whisper groq"""

import ctypes
import io
import os
import queue
import sys
import threading
import time
import traceback
import warnings
import wave
import tkinter as tk
from ctypes import wintypes
from tkinter import messagebox, simpledialog

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "sous_chef_log.txt")
KEY_PATH = os.path.join(HERE, "groq_key.txt")
IS_WIN = sys.platform == "win32"

# pythonw has no console, so stdout/stderr are None. Some libraries (model
# download progress bars, etc.) crash when they try to print. Send to a log.
if sys.stdout is None or sys.stderr is None:
    _log = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = _log
    if sys.stderr is None:
        sys.stderr = _log

# Windows loopback sends no data during silence; soundcard warns about the gap.
# It is harmless, so hide it.
warnings.filterwarnings("ignore", message="data discontinuity in recording")

try:
    import numpy as np
    import soundcard as sc
    from faster_whisper import WhisperModel
    from groq import Groq
except ImportError as e:
    _r = tk.Tk()
    _r.withdraw()
    messagebox.showerror(
        "Missing package",
        f"{e}\n\nInstall with:\npip install soundcard numpy faster-whisper groq",
    )
    sys.exit(1)

# ---------------- settings ----------------
STT_BACKEND = "local"                # "local" = faster-whisper on your PC (private)
                                     # "groq"  = Groq's hosted Whisper (free tier, very
                                     #           fast, but the chef's audio is sent online)
STT_MODEL = "small.en"               # local only. base.en = faster, medium.en = more accurate
STT_DEVICE = "cpu"                   # "cpu" always works. "cuda" needs the NVIDIA CUDA
                                     # libraries installed (that was the cublas error)
GROQ_STT_MODEL = "whisper-large-v3-turbo"   # groq backend only

LLM_MAX_TOKENS = 10000 

LLM_MODELS = [
    "openai/gpt-oss-20b",                        # main: fastest
    "openai/gpt-oss-120b",                     # backup 1: smarter
    "qwen/qwen3.8-27b",   # backup 2
]
COOLDOWN_SEC = 60 

SAMPLE_RATE = 16000
BLOCK_SEC = 0.1
SILENCE_RMS = 0.01                   # raise if background noise triggers speech
PAUSE_SEC = 0.7                      # pause that ends a chunk
MAX_CHUNK_SEC = 12                   # force a flush during long monologues
MIN_CHUNK_SEC = 0.6                  # ignore blips shorter than this
CONTEXT_SENTENCES = 4

WDA_EXCLUDEFROMCAPTURE = 0x11        # Windows 10 version 2004 and later

BG, PANEL, FG, MUTED = "#1e1e1e", "#252526", "#f2f2f2", "#9aa0a6"

CULINARY_PROMPT = (
    "In this data science, machine learning, and AI technical interview, we will discuss "
    "LLMs, RAG-Retrieval Augmented Generation, RLHF, MLOps, PyTorch, scikit-learn, and TensorFlow. Let's cover "
    "MCP-Model Context Protocol, gradient descent, bias-variance tradeoff, LangChain, Hugging Face, NumPy, pandas, "
    "The candidate explains their deep learning architecture, API endpoints, and Docker containers."
)
SYSTEM_PROMPT = (
    " You are a 5 year experienced data scientist with AI/ML expertise. Try to be concise and clear. "
    " Provide important parts of answer to questions asked. Provide complete but simple code in python ONLY IF ASKED! "
    " Underline or Bold the key points them. Avoid unnecessary details and focus on the important points if answer is too long. "
    " Never add tables, equations, fillers, sources, or ending summary and dont introduce the answer by saying 'Here is the ...' "    
)


# ---------------- hide window from screen capture ----------------
def exclude_from_capture(root):
    """Returns True if the window is now excluded from screen capture."""
    if not IS_WIN:
        return False
    user32 = ctypes.windll.user32
    user32.GetParent.argtypes = [wintypes.HWND]
    user32.GetParent.restype = wintypes.HWND
    user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
    root.update()  # make sure the native window exists
    hwnd = user32.GetParent(root.winfo_id())
    return bool(user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))


# ---------------- workers (all run off the UI thread) ----------------
_model = None


def get_model():
    global _model
    if _model is None:
        _model = WhisperModel(STT_MODEL, device=STT_DEVICE,
                              compute_type="int8" if STT_DEVICE == "cpu" else "float16")
    return _model


def capture_worker(device_id, chunk_q, ui, stop):
    """Reads system audio, cuts it into speech chunks at pauses."""
    try:
        mic = sc.get_microphone(id=device_id, include_loopback=True)
        block = int(SAMPLE_RATE * BLOCK_SEC)
        pause_blocks = int(PAUSE_SEC / BLOCK_SEC)
        max_blocks = int(MAX_CHUNK_SEC / BLOCK_SEC)
        min_blocks = int(MIN_CHUNK_SEC / BLOCK_SEC)
        buf, speaking, quiet = [], False, 0

        with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as rec:
            while not stop.is_set():
                data = rec.record(numframes=block)[:, 0].astype(np.float32)
                if np.sqrt(np.mean(data ** 2)) > SILENCE_RMS:
                    speaking, quiet = True, 0
                    buf.append(data)
                elif speaking:
                    buf.append(data)
                    quiet += 1

                if speaking and (quiet >= pause_blocks or len(buf) >= max_blocks):
                    if len(buf) >= min_blocks:
                        chunk_q.put(np.concatenate(buf))
                    buf, speaking, quiet = [], False, 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        ui.put(("status", f"Audio error: {e}"))
    finally:
        stop.set()


def to_wav_bytes(audio):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def transcribe_local(model, audio):
    segs, _ = model.transcribe(
        audio, language="en", beam_size=1,
        vad_filter=True, initial_prompt=CULINARY_PROMPT,
    )
    return " ".join(s.text.strip() for s in segs).strip()


def transcribe_groq(client, audio):
    result = client.audio.transcriptions.create(
        file=("chunk.wav", to_wav_bytes(audio)),
        model=GROQ_STT_MODEL,
        language="en",
        prompt=CULINARY_PROMPT,
        temperature=0.0,
    )
    return (result.text or "").strip()


def transcribe_worker(client, chunk_q, llm_q, ui, stop):
    """Turns audio chunks into text, forwards text to the UI and the LLM."""
    try:
        if STT_BACKEND == "groq":
            if client is None:
                ui.put(("status", "Groq speech-to-text needs an API key"))
                return
            transcribe = lambda a: transcribe_groq(client, a)
        else:
            ui.put(("status", "Loading speech model (first run downloads it)..."))
            model = get_model()
            transcribe = lambda a: transcribe_local(model, a)

        ui.put(("status", "Listening..."))
        while not stop.is_set():
            try:
                audio = chunk_q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                text = transcribe(audio)
            except Exception as e:
                # one bad chunk must not end the session
                traceback.print_exc()
                ui.put(("status", f"Transcription hiccup (still listening): {e}"))
                continue
            if text:
                ui.put(("transcript", text))
                llm_q.put(text)
    except Exception as e:
        traceback.print_exc()
        ui.put(("status", f"Speech-to-text error: {e}"))
    finally:
        stop.set()


def session(client, device_id, llm_q, ui, stop):
    chunk_q = queue.Queue()
    threads = [
        threading.Thread(target=capture_worker, args=(device_id, chunk_q, ui, stop), daemon=True),
        threading.Thread(target=transcribe_worker, args=(client, chunk_q, llm_q, ui, stop), daemon=True),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ui.put(("stopped", None))


_cooldown = {}   # model name -> time until which we skip it


def stream_one(client, model, messages, ui, state):
    """Streams one model's answer into the UI.
    Returns True only if it produced text AND finished normally.
    Raises on API errors (rate limit, model removed, timeout...)."""
    stream = client.chat.completions.create(
        model=model, stream=True, temperature=0.3,
        max_tokens=LLM_MAX_TOKENS, messages=messages,
    )
    got = False
    for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.delta.content:
            got = True
            state["shown"] = True
            ui.put(("token", choice.delta.content))
        if choice.finish_reason == "length":
            return False          # ran out of tokens mid-answer
    return got                    # False if the reply was empty


def ask_with_fallback(client, messages, ui):
    now = time.time()
    # skip models that errored recently, unless that would leave none
    models = [m for m in LLM_MODELS if _cooldown.get(m, 0) <= now] or list(LLM_MODELS)
    state = {"shown": False}
    last_error = None
    for i, model in enumerate(models):
        if i > 0:
            ui.put(("status", f"Switching to backup model: {model}"))
            if state["shown"]:
                ui.put(("token", f"\n[switching to {model}...]\n"))
        try:
            if stream_one(client, model, messages, ui, state):
                return
        except Exception as e:
            traceback.print_exc()
            last_error = e
            _cooldown[model] = time.time() + COOLDOWN_SEC
    if state["shown"]:
        ui.put(("token", " ...[incomplete: every model failed or hit its limit]"))
    else:
        ui.put(("token", f"[All models failed: {last_error or 'empty response'}]"))


def llm_worker(client, in_q, ui):
    history = []
    while True:
        text = in_q.get()
        if text is None:
            break
        # If the chef kept talking while we were answering, merge everything
        # that piled up into one request so the answer never lags behind.
        while True:
            try:
                more = in_q.get_nowait()
            except queue.Empty:
                break
            if more is None:
                return
            text += " " + more
        context = " ".join(history[-CONTEXT_SENTENCES:])
        history.append(text)
        ui.put(("sentence", None))
        if client is None:
            ui.put(("token", "[No Groq API key set. Restart the app and enter one.]"))
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Earlier: {context}\nLatest: {text}"},
        ]
        try:
            ask_with_fallback(client, messages, ui)
        except Exception as e:
            traceback.print_exc()
            ui.put(("token", f"[LLM error: {e}]"))


# ---------------- API key ----------------
def get_api_key(root):
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    if os.path.exists(KEY_PATH):
        with open(KEY_PATH, encoding="utf-8") as f:
            key = f.read().strip()
        if key:
            return key
    key = simpledialog.askstring(
        "Groq API key",
        "Paste your free Groq API key\n(get one at console.groq.com/keys):",
        parent=root, show="*",
    )
    key = (key or "").strip()
    if key:
        with open(KEY_PATH, "w", encoding="utf-8") as f:
            f.write(key)
    return key or None


# ---------------- window ----------------
class App:
    def __init__(self, root, client):
        self.root = root
        self.client = client
        self.ui = queue.Queue()
        self.llm_q = queue.Queue()
        self.stop = None
        self.running = False
        self.size = 14
        threading.Thread(target=llm_worker, args=(client, self.llm_q, self.ui), daemon=True).start()

        root.configure(bg=BG)

        # top bar
        bar = tk.Frame(root, bg=BG)
        bar.pack(fill="x", padx=8, pady=6)
        tk.Label(bar, text="Audio source:", bg=BG, fg=FG).pack(side="left")

        self.devices = {}
        try:
            for d in sc.all_microphones(include_loopback=True):
                self.devices[d.name] = d.id
        except Exception:
            pass
        names = list(self.devices) or ["(no audio devices found)"]
        default = names[0]
        try:
            speaker = sc.default_speaker().name
            if speaker in self.devices:
                default = speaker
        except Exception:
            pass
        self.device_var = tk.StringVar(value=default)
        menu = tk.OptionMenu(bar, self.device_var, *names)
        menu.config(bg="#333333", fg=FG, activebackground="#444444", activeforeground=FG,
                    highlightthickness=0, borderwidth=0, width=38, anchor="w")
        menu["menu"].config(bg="#333333", fg=FG)
        menu.pack(side="left", padx=6)

        self.btn = tk.Button(bar, text="Start", command=self.toggle, bg="#2d6a4f", fg="white",
                             activebackground="#40916c", activeforeground="white",
                             relief="flat", padx=16, borderwidth=0)
        self.btn.pack(side="left")
        self.status = tk.Label(bar, text="Idle", bg=BG, fg=MUTED)
        self.status.pack(side="right")

        # two panels
        pw = tk.PanedWindow(root, orient="horizontal", bg=BG, sashwidth=6, borderwidth=0)
        pw.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.raw = self._panel(pw, "Chef said (live transcript)")
        self.simple = self._panel(pw, "In simple terms")

        root.bind("<Control-plus>", lambda e: self.resize(2))
        root.bind("<Control-equal>", lambda e: self.resize(2))
        root.bind("<Control-minus>", lambda e: self.resize(-2))
        root.bind("<Escape>", lambda e: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.poll()

    def _panel(self, pw, title):
            frame = tk.Frame(pw, bg=BG)
            tk.Label(frame, text=title, bg=BG, fg=MUTED, anchor="w").pack(fill="x")
            body = tk.Frame(frame, bg=BG)
            body.pack(fill="both", expand=True)
            scroll = tk.Scrollbar(body, orient="vertical")
            text = tk.Text(body, wrap="word", font=("Segoe UI", self.size), padx=10, pady=10,
                           bg=PANEL, fg=FG, insertbackground="white", borderwidth=0,
                           state="disabled", yscrollcommand=scroll.set)
            scroll.config(command=text.yview)
            scroll.pack(side="right", fill="y")
            text.pack(side="left", fill="both", expand=True)
            text.tag_configure("marker", foreground="#74c69d")
            pw.add(frame, stretch="always", minsize=200)
            return text

    @staticmethod
    def _append(widget, s, tag=None):
            # Follow new text only if the reader is already at the bottom. If they
            # scrolled up to re-read something, leave the view where it is.
            at_bottom = widget.yview()[1] >= 0.99
            widget.configure(state="normal")
            if tag:
                widget.insert("end", s, tag)
            else:
                widget.insert("end", s)
            widget.configure(state="disabled")
            if at_bottom:
                widget.see("end")

    def resize(self, delta):
        self.size = max(8, min(40, self.size + delta))
        for w in (self.raw, self.simple):
            w.configure(font=("Segoe UI", self.size))

    def toggle(self):
        if self.running:
            self.stop.set()
            self.btn.config(state="disabled", text="Stopping...")
            return
        device_id = self.devices.get(self.device_var.get())
        if device_id is None:
            self.status.config(text="No audio device selected")
            return
        self.stop = threading.Event()
        self.running = True
        self.btn.config(text="Stop", bg="#9d2b2b", activebackground="#c0392b")
        threading.Thread(target=session,
                         args=(self.client, device_id, self.llm_q, self.ui, self.stop),
                         daemon=True).start()

    def poll(self):
            try:
                while True:
                    kind, val = self.ui.get_nowait()
                    if kind == "status":
                        self.status.config(text=val)
                    elif kind == "transcript":
                        self._append(self.raw, val + "\n\n")
                    elif kind == "sentence":
                        first = not self.simple.get("1.0", "end").strip()
                        self._append(self.simple, "" if first else "\n\n")
                        self._append(self.simple, "▸ ", "marker")
                    elif kind == "token":
                        self._append(self.simple, val)
                    elif kind == "stopped":
                        self.running = False
                        self.btn.config(text="Start", state="normal", bg="#2d6a4f",
                                        activebackground="#40916c")
                        if self.status.cget("text") in ("Listening...", "Stopping..."):
                            self.status.config(text="Stopped")
            except queue.Empty:
                pass
            self.root.after(40, self.poll)

    def close(self):
        if self.stop:
            self.stop.set()
        self.root.destroy()


def main():
    root = tk.Tk()
    root.title("Sous-Chef")
    root.geometry("960x600+80+80")
    root.attributes("-topmost", True)       # stays above the meeting window
    if IS_WIN:
        root.attributes("-toolwindow", True)  # no taskbar button, hidden from Alt+Tab
    root.attributes("-alpha", 0.95)

    key = "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"
    # short timeout + no SDK retries, so a stuck or rate-limited model fails
    # fast and the next model in LLM_MODELS takes over
    client = Groq(api_key=key, timeout=20.0, max_retries=0) if key else None
    App(root, client)

    if IS_WIN and not exclude_from_capture(root):
        root.title("Sous-Chef (WARNING: capture exclusion failed, do NOT share screen)")
    elif not IS_WIN:
        root.title("Sous-Chef (capture exclusion is Windows-only, do NOT share screen)")

    root.mainloop()


if __name__ == "__main__":
    main()
