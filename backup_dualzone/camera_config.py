"""
Camera, Hand Tracking, Gesture, and Drawing Configuration for Drawing Monster Defense.
Optimized for Tracking Stability, Responsiveness, Low Latency, and Debounced Control.
"""

# Laptop Camera Server URL
CAMERA_IP = "192.168.1.172"
CAMERA_PORT = 5000
STREAM_ENDPOINT = "video"
CAMERA_URL = f"http://{CAMERA_IP}:{CAMERA_PORT}/{STREAM_ENDPOINT}"

# MediaPipe Hands Parameters (Optimized from diagnosis to minimize dropouts)
MAX_NUM_HANDS = 2
STATIC_IMAGE_MODE = False
MODEL_COMPLEXITY = 1
MIN_DETECTION_CONFIDENCE = 0.5   # 0.5 allows robust re-acquisition without false positives
MIN_TRACKING_CONFIDENCE = 0.5    # 0.5 prevents premature tracking drop during rapid finger movement

# Image & Handedness Settings
FLIP_HORIZONTAL = False          # Laptop server already flips horizontally
INVERT_HANDEDNESS = False        # MediaPipe raw matches mirrored view

# Stream Optimization Settings
USE_THREADED_CAMERA = True       # Decouples network frame grabbing to eliminate buffer lag
CAMERA_BUFFER_SIZE = 1           # Minimizes OpenCV internal MJPEG buffer queue

# Landmark Smoothing (One Euro Filter)
USE_LANDMARK_SMOOTHING = True
ONE_EURO_MIN_CUTOFF = 1.2        # Minimum cutoff frequency (Hz) at zero speed (eliminates jitter)
ONE_EURO_BETA = 0.015            # Speed coefficient: dynamically raises cutoff during fast motion (zero lag)
ONE_EURO_DCUTOFF = 1.0           # Derivative cutoff frequency (Hz)
FILTER_RESET_TIMEOUT = 0.25      # Seconds of tracking loss before filter resets

# Landmark Definitions
LANDMARK_WRIST = 0
LANDMARK_THUMB_TIP = 4
LANDMARK_INDEX_MCP = 5
LANDMARK_INDEX_PIP = 6
LANDMARK_INDEX_FINGER_TIP = 8
LANDMARK_MIDDLE_MCP = 9
LANDMARK_MIDDLE_PIP = 10
LANDMARK_MIDDLE_TIP = 12
LANDMARK_RING_MCP = 13
LANDMARK_RING_PIP = 14
LANDMARK_RING_TIP = 16
LANDMARK_PINKY_MCP = 17
LANDMARK_PINKY_PIP = 18
LANDMARK_PINKY_TIP = 20

# Two-Hand Join Gesture Parameters (Toggle Mode)
JOIN_HOLD_DURATION = 0.7         # Seconds to hold join gesture to toggle (0.5–1.0s)
JOIN_DISTANCE_THRESHOLD = 65     # Pixels: Distance between index tips/palms to detect join
RELEASE_DISTANCE_THRESHOLD = 115 # Pixels: Distance required to consider hands separated (hysteresis)
GESTURE_COOLDOWN = 0.4           # Minimum seconds after a toggle before next evaluation
SCALE_NORMALIZED_CHECK = True    # Use relative hand size to make join detection distance-invariant

# Drawing Control: Pointing vs Fist Detection
FIST_CONFIRM_FRAMES = 2          # Consecutive frames required to confirm fist (anti-flicker)
POINTING_CONFIRM_FRAMES = 2      # Consecutive frames required to confirm pointing

# Drawing Continuity & Gap Recovery
MAX_BRIDGE_GAP_TIME = 0.20       # Seconds: Maximum tracking loss duration eligible for bridge interpolation
MAX_BRIDGE_GAP_DIST = 80         # Pixels: Maximum distance eligible for bridge interpolation
MIN_STROKE_POINT_DIST = 2        # Pixels: Minimum movement to record new trajectory point
DRAW_STROKE_THICKNESS = 4

# Visualization Settings
WINDOW_NAME = "Drawing Monster Defense - Hand Tracking & Air Drawing"
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Colors (BGR)
COLOR_LEFT_HAND = (255, 120, 0)       # Cyan / Blue
COLOR_RIGHT_HAND = (0, 165, 255)     # Orange
COLOR_INDEX_TIP = (0, 255, 0)        # Bright Green
COLOR_DRAW_LEFT = (255, 220, 0)      # Cyan / Light Yellow-Blue
COLOR_DRAW_RIGHT = (0, 150, 255)     # Amber / Orange
COLOR_JOIN_ACTIVE = (0, 255, 255)    # Yellow for gesture hold indicator
COLOR_FIST_PAUSED = (120, 120, 120)  # Grey for paused hand
COLOR_BG_PANEL = (20, 20, 20)        # Dark grey for HUD
COLOR_TEXT = (255, 255, 255)         # White
COLOR_NOT_DETECTED = (140, 140, 140) # Grey
