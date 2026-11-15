"""
Camera, Hand Tracking, Gesture, and Drawing Configuration for Drawing Monster Defense.
Optimized for Tracking Stability, 3D Index Finger Recognition across all directions,
Low Latency, and Dual-Zone Air Drawing.
"""

# Laptop Camera Server URL
CAMERA_IP = "192.168.1.172"
CAMERA_PORT = 5000
STREAM_ENDPOINT = "video"
CAMERA_URL = f"http://{CAMERA_IP}:{CAMERA_PORT}/{STREAM_ENDPOINT}"

# Automatic Camera Detection & Fallback Configuration
AUTO_CAMERA_DETECTION = True          # Automatically prioritize Local Webcam -> Fallback to IP Camera
LOCAL_CAMERA_INDICES = [0, 1]         # Device indices to scan for local webcams (Primary source)
LOCAL_CAMERA_TIMEOUT = 1.5            # Seconds timeout to probe local webcam
NETWORK_STREAM_TIMEOUT = 1.0          # Seconds timeout for network stream reachability check
CAMERA_FAILURE_THRESHOLD = 15         # Consecutive dropped frames before triggering reconnect/fallback
CAMERA_RECONNECT_COOLDOWN = 3.0       # Minimum seconds between source switches to prevent thrashing
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# MediaPipe Hands Parameters
# 0.45 confidence ensures rapid tracking lock during fast motion without dropping to palm detection
MAX_NUM_HANDS = 2
STATIC_IMAGE_MODE = False
MODEL_COMPLEXITY = 1
MIN_DETECTION_CONFIDENCE = 0.45   # Rapid re-acquisition on hand entry/recovery
MIN_TRACKING_CONFIDENCE = 0.45    # Prevents tracking drop during fast finger movement & motion blur

# Image & Handedness Settings
FLIP_HORIZONTAL = False          # Laptop server already flips horizontally
INVERT_HANDEDNESS = False        # MediaPipe raw matches mirrored view

# Stream Optimization Settings
USE_THREADED_CAMERA = True       # Decouples network frame grabbing to eliminate buffer lag
CAMERA_BUFFER_SIZE = 1           # Minimizes OpenCV internal MJPEG buffer queue

# Landmark Smoothing (One Euro Filter)
USE_LANDMARK_SMOOTHING = True
ONE_EURO_MIN_CUTOFF = 1.2        # Minimum cutoff frequency (Hz) at zero speed (eliminates jitter)
ONE_EURO_BETA = 0.028            # Dynamically raises cutoff during fast motion for ZERO lag tracking
ONE_EURO_DCUTOFF = 1.0           # Derivative cutoff frequency (Hz)
FILTER_RESET_TIMEOUT = 0.25      # Seconds of tracking loss before filter resets

# Landmark Definitions
LANDMARK_WRIST = 0
LANDMARK_THUMB_CMC = 1
LANDMARK_THUMB_MCP = 2
LANDMARK_THUMB_IP = 3
LANDMARK_THUMB_TIP = 4
LANDMARK_INDEX_MCP = 5
LANDMARK_INDEX_PIP = 6
LANDMARK_INDEX_DIP = 7
LANDMARK_INDEX_FINGER_TIP = 8
LANDMARK_MIDDLE_MCP = 9
LANDMARK_MIDDLE_PIP = 10
LANDMARK_MIDDLE_DIP = 11
LANDMARK_MIDDLE_TIP = 12
LANDMARK_RING_MCP = 13
LANDMARK_RING_PIP = 14
LANDMARK_RING_DIP = 15
LANDMARK_RING_TIP = 16
LANDMARK_PINKY_MCP = 17
LANDMARK_PINKY_PIP = 18
LANDMARK_PINKY_DIP = 19
LANDMARK_PINKY_TIP = 20

# Image Quality Assessment & Adaptive Enhancement Thresholds
QUALITY_DARK_THRESHOLD = 60.0          # Mean luminance below which image is classified as TOO_DARK
QUALITY_BRIGHT_THRESHOLD = 195.0       # Mean luminance above which image is classified as TOO_BRIGHT
QUALITY_LOW_CONTRAST_THRESHOLD = 32.0  # Luminance standard deviation below which contrast is poor / hazy
QUALITY_BLUR_SHARPNESS_THRESHOLD = 50.0 # Laplacian variance below which frame is motion-blurred
QUALITY_NOISE_THRESHOLD = 8.0          # High-frequency noise standard deviation threshold
CLAHE_CLIP_LIMIT = 2.0                 # Contrast limiting factor for CLAHE local equalization
CLAHE_GRID_SIZE = (8, 8)               # Spatial grid for adaptive histogram equalization
DARK_GAMMA = 0.65                      # Precomputed Gamma LUT exponent for shadow recovery
BRIGHT_GAMMA = 1.35                    # Precomputed Gamma LUT exponent for highlight attenuation

# Confidence-Aware Tracking & Safe Drawing Thresholds
CONFIDENCE_STABLE_MIN = 0.60           # Threshold for TRACKING_STABLE (normal drawing permitted)
CONFIDENCE_UNCERTAIN_MIN = 0.35        # Threshold for TRACKING_UNCERTAIN (drawing temporarily paused)
TRACKING_RECOVERY_CONFIRM_FRAMES = 2   # Consecutive stable frames required before resuming new stroke

# 3D Index Finger Pose Recognition Thresholds
INDEX_STRAIGHTNESS_MIN = 0.76    # 3D Chord-to-segment ratio for straight index finger
INDEX_PIP_ANGLE_MIN_COS = 0.65   # Cosine of PIP joint angle (straight >= 0.65, ~49 deg)
INDEX_PALM_EXTENSION_RATIO = 1.20 # Index tip must be at least 1.2x further from palm center than middle tip
FIST_MAX_STRAIGHTNESS = 0.55     # Straightness below which finger is considered curled
OPEN_HAND_MIDDLE_EXT = 0.75      # Middle finger straightness above which palm is considered open

# Two-Hand Join Gesture Parameters (Toggle Mode)
JOIN_HOLD_DURATION = 0.7         # Seconds to hold join gesture to toggle (0.5–1.0s)
JOIN_DISTANCE_THRESHOLD = 65     # Pixels: Distance between index tips/palms to detect join
RELEASE_DISTANCE_THRESHOLD = 115 # Pixels: Distance required to consider hands separated (hysteresis)
GESTURE_COOLDOWN = 0.4           # Minimum seconds after a toggle before next evaluation
SCALE_NORMALIZED_CHECK = True    # Use relative hand size to make join detection distance-invariant

# Drawing Control: Pointing vs Fist Detection
FIST_CONFIRM_FRAMES = 2          # Consecutive frames required to confirm fist (anti-flicker)
POINTING_CONFIRM_FRAMES = 1      # Quick 1-frame responsiveness when pointing is confirmed

# Drawing Continuity & Gap Recovery
MAX_BRIDGE_GAP_TIME = 0.22       # Seconds: Maximum tracking loss duration eligible for bridge interpolation
MAX_BRIDGE_GAP_DIST = 100        # Pixels: Maximum distance eligible for bridge interpolation during fast strokes
MIN_STROKE_POINT_DIST = 2        # Pixels: Minimum movement to record new trajectory point
DRAW_STROKE_THICKNESS = 4

# Dual-Zone Air Drawing Settings
ZONE_SPLIT_RATIO = 0.5           # Vertical split at 50%
COLOR_ZONE_DIVIDER = (0, 220, 255) # Yellow / Amber for center line
COLOR_ZONE_OUT = (0, 0, 255)       # Red indicator when hand crosses out of assigned zone
COLOR_ZONE_LEFT_TAG = (255, 180, 0) # Cyan accent
COLOR_ZONE_RIGHT_TAG = (0, 165, 255) # Orange accent

# Visualization & HUD Settings
WINDOW_NAME = "Drawing Monster Defense - Hand Tracking & Air Drawing"
SHOW_DEBUG_HUD = False            # Set False to hide the HUD debug panel on screen (press 'D' to toggle)
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
