"""
Text-to-Speech module. Speaks on a background thread via a queue so it
never blocks the real-time video loop.
"""
import queue
import threading
import config as cfg


class TTSModule:
    def __init__(self, engine=cfg.TTS_ENGINE, rate=cfg.TTS_RATE, volume=cfg.TTS_VOLUME):
        self.engine_name = engine
        self.rate = rate
        self.volume = volume
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def speak(self, text):
        """Non-blocking: enqueue text to be spoken."""
        if text and text.strip():
            self._queue.put(text.strip())

    def stop(self):
        self._stop.set()
        self._queue.put(None)

    def _worker(self):
        if self.engine_name == "pyttsx3":
            self._run_pyttsx3()
        else:
            self._run_gtts()

    def _run_pyttsx3(self):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)
        while not self._stop.is_set():
            text = self._queue.get()
            if text is None:
                break
            engine.say(text)
            engine.runAndWait()

    def _run_gtts(self):
        import os
        import tempfile
        from gtts import gTTS
        try:
            import playsound
        except ImportError:
            playsound = None

        while not self._stop.is_set():
            text = self._queue.get()
            if text is None:
                break
            tmp_path = None
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                tmp_path = tmp.name
                tmp.close()
                gTTS(text=text, lang="en").save(tmp_path)
                if playsound:
                    playsound.playsound(tmp_path)
                else:
                    os.system(f"mpg123 {tmp_path}" if os.name != "nt" else f"start {tmp_path}")
            finally:
                if tmp_path:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
