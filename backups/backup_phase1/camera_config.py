"""
Camera, Hand Tracking, Gesture, and Drawing Configuration for Drawing Monster Defense.
Edit the CAMERA_URL here when the Laptop Camera Server IP changes.
"""

# Laptop Camera Server URL
# Default IP of the laptop camera server
CAMERA_IP = "192.168.1.172"
CAMERA_PORT = 5000
STREAM_ENDPOINT = "video"
CAMERA_URL = f"http://{CAMERA_IP}:{CAMERA_PORT}/{STREAM_ENDPOINT}"

# MediaPipe Hands Parameters
MAX_NUM_HANDS = 2
STATIC_IMAGE_MODE = False
MODEL_COMPLEXITY = 1
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

# Image & Handedness Settings
# Laptop camera server already applies cv2.flip(frame, 1) before streaming.
# Set FLIP_HORIZONTAL = True only if input is unflipped.
FLIP_HORIZONTAL = False

# MediaPipe calculates handedness assuming a mirrored front camera.
# If Left/Right labels appear inverted during testing, set this to True.
INVERT_HANDEDNESS = False

# Landmark Definitions
LANDMARK_WRIST = 0
LANDMARK_THUMB_TIP = 4
LANDMARK_INDEX_FINGER_TIP = 8
LANDMARK_MIDDLE_MCP = 9
LANDMARK_MIDDLE_TIP = 12

# Two-Hand Join Gesture Parameters
JOIN_HOLD_DURATION = 0.7          # Seconds to hold join gesture to toggle (0.5–1.0s)
JOIN_DISTANCE_THRESHOLD = 65      # Pixels: Distance between index tips/palms to detect join
RELEASE_DISTANCE_THRESHOLD = 115  # Pixels: Distance required to consider hands separated (hysteresis)
GESTURE_COOLDOWN = 0.4            # Minimum seconds after a toggle before next evaluation
SCALE_NORMALIZED_CHECK = True     # Use relative hand size to make join detection distance-invariant

# Air Drawing Settings
DRAW_STROKE_THICKNESS = 4
MIN_STROKE_POINT_DIST = 3         # Minimum pixel movement to record new trajectory point
COLOR_DRAW_LEFT = (255, 220, 0)   # Cyan / Light Cyan-Yellow (BGR)
COLOR_DRAW_RIGHT = (0, 150, 255)  # Orange / Amber (BGR)
COLOR_JOIN_ACTIVE = (0, 255, 255) # Bright Yellow for gesture hold indicator

# Visualization Settings
WINDOW_NAME = "Drawing Monster Defense - Hand Tracking & Air Drawing"
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Colors (BGR)
COLOR_LEFT_HAND = (255, 120, 0)       # Blue-ish / Cyan
COLOR_RIGHT_HAND = (0, 165, 255)     # Orange
COLOR_INDEX_TIP = (0, 255, 0)        # Bright Green
COLOR_BG_PANEL = (20, 20, 20)        # Dark grey for HUD
COLOR_TEXT = (255, 255, 255)         # White
COLOR_NOT_DETECTED = (128, 128, 128) # Grey
