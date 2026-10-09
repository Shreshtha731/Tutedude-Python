"""
Configuration settings for the Sign Language Recognition - UI & Output Engine
(Member 4 deliverable: display, TTS, buffering/stabilization)
"""

# ---- Camera settings ----
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 720

# ---- Stabilization settings ----
BUFFER_SIZE = 15                 # number of recent frames to look at for majority vote
CONFIRM_THRESHOLD = 0.75         # fraction of buffer that must agree to confirm a sign
MIN_CONFIDENCE = 0.55            # per-frame confidence floor to even count a prediction
COOLDOWN_FRAMES = 20             # frames to wait after confirming before accepting a new sign
IDLE_LABEL = "idle"              # label used when no hand / no confident prediction

# ---- Special control gestures (map these to whatever your model outputs) ----
SPACE_GESTURE = "space"
DELETE_GESTURE = "delete"
CLEAR_GESTURE = "clear"
SPEAK_GESTURE = "speak"

# ---- TTS settings ----
TTS_ENGINE = "pyttsx3"              # "pyttsx3" (offline) or "gtts" (online, needs internet)
TTS_RATE = 165                      # words per minute, pyttsx3 only
TTS_VOLUME = 1.0

# ---- HUD appearance (BGR colors, since OpenCV) ----
HUD_FONT_SCALE = 0.8
HUD_BG_ALPHA = 0.55
COLOR_TEXT = (255, 255, 255)
COLOR_ACCENT = (60, 200, 255)
COLOR_CONFIRM = (80, 220, 120)
COLOR_LOW_CONF = (60, 60, 220)
COLOR_BAR_BG = (50, 50, 50)
