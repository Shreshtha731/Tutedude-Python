import queue
import threading
import speech_recognition as sr


class AudioInputTranslator:

  def __init__(self):
    self.recognizer = sr.Recognizer()
    self.text_queue = queue.Queue()
    self.latest_speech = "Listening for incoming voice..."
    self.running = True

    self.thread = threading.Thread(target=self._listen_loop, daemon=True)
    self.thread.start()

  def _listen_loop(self):
    try:
      mic = sr.Microphone()
    except Exception as e:
      self.latest_speech = "No mic detected."
      return

    with mic as source:
      self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
      while self.running:
        try:
          audio = self.recognizer.listen(source, phrase_time_limit=4)
          text = self.recognizer.recognize_google(audio)
          self.latest_speech = text
        except sr.UnknownValueError:
          pass
        except Exception:
          pass

  def get_latest_transcription(self):
    return self.latest_speech

  def stop(self):
    self.running = False