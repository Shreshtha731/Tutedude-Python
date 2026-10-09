import queue
import shutil
import subprocess
import threading

# Vocabulary spoken translations (cleans formatting & supports Hindi toggle)
SPOKEN_PHRASES = {
    "hello": {"en": "Hello", "hi": "Namaste"},
    "thank_you": {"en": "Thank you", "hi": "Dhanyavaad"},
    "yes": {"en": "Yes", "hi": "Haan"},
    "no": {"en": "No", "hi": "Nahin"},
    "help": {"en": "Help", "hi": "Madad"},
}


class TextToSpeechEngine:

  def __init__(self, target_lang="en"):
    self.speech_queue = queue.Queue()  #
    self.target_lang = target_lang
    self.running = True

    # Detect if we have access to Windows PowerShell from WSL
    self.has_powershell = shutil.which("powershell.exe") is not None

    # Dedicated background daemon thread[cite: 1, 17]
    self.worker_thread = threading.Thread(target=self._worker, daemon=True)
    self.worker_thread.start()
    print(
        f"[✓] Audio Engine initialized (Host Bridge:"
        f" {'Active' if self.has_powershell else 'Native Linux'})"
    )

  def set_language(self, lang_code):
    if lang_code in ["en", "hi"]:
      self.target_lang = lang_code
      print(f"[*] Audio Language: {lang_code.upper()}")

  def _worker(self):
    while self.running:
      try:
        token = self.speech_queue.get(timeout=0.1)
        if token is None:
          break

        # Map token to spoken phrase
        phrase_entry = SPOKEN_PHRASES.get(token.lower(), {})
        spoken_text = phrase_entry.get(
            self.target_lang, token.replace("_", " ")
        )

        print(f"\n🔊 SPEAKING ALOUD: \"{spoken_text}\"")

        # Route through Windows Speech Engine via WSL host bridge
        if self.has_powershell:
          ps_cmd = (
              f"Add-Type -AssemblyName System.Speech; $s = New-Object"
              f" System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = 1;"
              f' $s.Speak("{spoken_text}");'
          )
          subprocess.run(
              ["powershell.exe", "-Command", ps_cmd],
              stdout=subprocess.DEVNULL,
              stderr=subprocess.DEVNULL,
          )
        else:
          # Native Linux fallback
          subprocess.run(
              ["espeak", spoken_text],
              stdout=subprocess.DEVNULL,
              stderr=subprocess.DEVNULL,
          )

        self.speech_queue.task_done()
      except queue.Empty:
        continue
      except Exception as e:
        print(f"[!] Audio synthesis error: {e}")

  def speak(self, text):
    """Enqueues confirmed word to be spoken aloud[cite: 1, 17]."""
    if text:
      self.speech_queue.put(text)

  def stop(self):
    self.running = False
    self.speech_queue.put(None)