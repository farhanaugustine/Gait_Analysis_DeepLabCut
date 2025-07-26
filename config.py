# config.py
import os
import numpy as np

# --- Central Output Directory ---
RESULTS_DIR = 'results'

# =============================================================================
# SECTION 1: FILE PATHS
# =============================================================================
INPUT_VIDEO_PATH = r"LL1-1_BLKS.mp4"
INPUT_CSV_PATH = r"LL1-1_BLKS_superanimal_topviewmouse_fasterrcnn_resnet50_fpn_v2_resnet_50.csv"
ROI_CONFIG_PATH = 'roi_config.json'

# Output files
OUTPUT_CSV_PATH = os.path.join(RESULTS_DIR, 'final_analysis_data.csv')
GAIT_ANALYSIS_PATH = os.path.join(RESULTS_DIR, 'gait_analysis_summary.csv')
ANALYSIS_SUMMARY_PATH = os.path.join(RESULTS_DIR, 'analysis_summary.json')
OUTPUT_VIDEO_PATH = os.path.join(RESULTS_DIR, 'behavior_analysis_output.mp4')

# =============================================================================
# SECTION 2: DATASET CONFIGURATION
# =============================================================================
KEYPOINT_ORDER = [
    'Nose', 'LeftEar', 'RightEar', 'LeftEarTip', 'RightEarTip', 'LeftEye', 'RightEye',
    'Neck', 'MidBack', 'MouseCenter', 'MidBackend', 'MidBackend2', 'MidBackend3',
    'TailBase', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'LeftShoulder',
    'LeftMidside', 'LeftHip', 'RightShoulder', 'RightMidside', 'RightHip',
    'TailEnd', 'HeadMidpoint'
]

# =============================================================================
# SECTION 3: GAIT ANALYSIS PARAMETERS
# =============================================================================
GAIT_PAWS = ['LeftShoulder', 'RightShoulder', 'LeftHip', 'RightHip']
PAW_ORDER_HILDEBRAND = ['LeftShoulder', 'RightShoulder', 'LeftHip', 'RightHip']
PAW_SPEED_THRESHOLD_PX_PER_FRAME = 5
STRIDE_REFERENCE_PAW = 'LeftHip'
PAW_PLOT_COLORS = {
    'LeftShoulder': (255, 100, 100),
    'RightShoulder': (100, 100, 255),
    'LeftHip': (255, 255, 100),
    'RightHip': (100, 255, 100)
}

# =============================================================================
# SECTION 4: SKELETON & POSE
# =============================================================================
SKELETON_CONNECTIONS = [
    ("Nose", "HeadMidpoint"), ("LeftEye", "HeadMidpoint"), ("RightEye", "HeadMidpoint"),
    ("LeftEar", "HeadMidpoint"), ("RightEar", "HeadMidpoint"), ("LeftEar", "LeftEarTip"),
    ("RightEar", "RightEarTip"), ("HeadMidpoint", "Neck"), ("Neck", "MidBack"),
    ("MidBack", "MouseCenter"), ("MouseCenter", "MidBackend"), ("MidBackend", "MidBackend2"),
    ("MidBackend2", "MidBackend3"), ("MidBackend3", "TailBase"), ("Neck", "LeftShoulder"),
    ("Neck", "RightShoulder"), ("LeftShoulder", "LeftMidside"), ("LeftMidside", "LeftHip"),
    ("RightShoulder", "RightMidside"), ("RightMidside", "RightHip"), ("LeftHip", "TailBase"),
    ("RightHip", "TailBase"), ("TailBase", "Tail1"), ("Tail1", "Tail2"), ("Tail2", "Tail3"),
    ("Tail3", "Tail4"), ("Tail4", "Tail5"), ("Tail5", "TailEnd")
]
ELONGATION_CONNECTION = ("Nose", "TailBase")
BODY_ANGLE_CONNECTION = ("Neck", "Nose")

# =============================================================================
# SECTION 5: DISPLAY & DRAWING SETTINGS
# =============================================================================
BEHAVIOR_COLORS = { "stance": (100, 255, 100), "swing": (255, 100, 100) }
SKELETON_COLOR = (255, 255, 255)
KEYPOINT_COLOR = (0, 0, 255)
KEYPOINT_RADIUS = 3
RESIZED_VIDEO_WIDTH = 500
DASHBOARD_WIDTH = 680
GRAPH_WINDOW_SECONDS = 5
HILDEBRAND_WINDOW_SECONDS = 4
MAX_LIST_ITEMS = 4

# =============================================================================
# SECTION 6: ANALYSIS PARAMETERS
# =============================================================================
DETECTION_CONF_THRESHOLD = 0.6