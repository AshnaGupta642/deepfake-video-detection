import os

# Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset Paths
DATASET_PATHS = {
    "SDFVD_REAL": os.path.join(PROJECT_ROOT, "SDFVD", "videos_real"),
    "SDFVD_FAKE": os.path.join(PROJECT_ROOT, "SDFVD", "videos_fake"),
    "UADFV_REAL": os.path.join(PROJECT_ROOT, "UADFV", "real"),
    "UADFV_FAKE": os.path.join(PROJECT_ROOT, "UADFV", "fake")
}
# Number of random frames to extract
NUM_FRAMES = 30

# Output folder
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "extracted_frames")

# Supported video formats
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")