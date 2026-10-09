import os

# ==========================================
# SYSTEM & FILE PATH CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "MP_Data")
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
ACTIONS_PATH = os.path.join(BASE_DIR, "actions.json")

# ==========================================
# ACTION VOCABULARY
# ==========================================
# Active gesture classes configured for live inference
ACTIONS = ["hello", "thank_you", "yes", "no", "help", "ok"]

# ==========================================
# HARDWARE & CAMERA INGESTION SETTINGS
# ==========================================
CAMERA_INDEX = 0
CAPTURE_WIDTH = 1280  # Upgraded to 720p HD[cite: 18]
CAPTURE_HEIGHT = 720  #[cite: 18]
TARGET_FPS = 30  # Target frame acquisition rate[cite: 1, 7]

# ==========================================
# FEATURE & SEQUENCE DIMENSIONS
# ==========================================
SEQUENCE_LENGTH = 30  # 30-frame temporal window for Bi-LSTM[cite: 1, 7]
FEATURES_PER_FRAME = 126  # 42 landmarks * 3 coordinates (x, y, z)[cite: 1, 7]
NUM_LANDMARKS_PER_HAND = 21

# ==========================================
# TEMPORAL STABILIZATION & GATING THRESHOLDS
# ==========================================
PREDICTION_CONFIDENCE_THRESHOLD = 0.60  # Softmax confidence gate for responsive triggering
CONSENSUS_THRESHOLD = 0.65  # Majority vote agreement across rolling buffer[cite: 1]
STABILIZER_BUFFER_SIZE = 15  # Sliding prediction FIFO buffer depth[cite: 1]
COOLDOWN_FRAMES = 15  # Refractory period preventing duplicate speech vocalization[cite: 1]

# ==========================================
# AUDIO & TEXT-TO-SPEECH (TTS) PARAMETERS
# ==========================================
TTS_RATE = 160  # Words per minute
TTS_VOLUME = 1.0  # Master synthesis volume (0.0 to 1.0)
DEFAULT_AUDIO_LANG = "en"  # Active vocalization language ('en' or 'hi')