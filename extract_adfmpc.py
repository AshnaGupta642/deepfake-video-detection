import os
import cv2
import torch
import numpy as np

from motion_phase.regions import regions



from motion_phase.landmarks import FaceLandmarks
from motion_phase.optical_flow import OpticalFlow
from motion_phase.motion_tracker import MotionTracker
from motion_phase.ad_fmpc import ADFMPC
from motion_phase.feature_vector import AD_FMPC_Vector
from motion_phase.regions import regions

# Initialize all modules

landmark_detector = FaceLandmarks()

optical_flow = OpticalFlow()

tracker = MotionTracker(window_size=5)

adfmpc = ADFMPC()

vectorizer = AD_FMPC_Vector()

FACE_ROOT = "faces"

dataset = "SDFVD_FAKE"

video = "vs1"

video_path = os.path.join(FACE_ROOT, dataset, video)

frame_names = sorted(os.listdir(video_path))

frames = []
all_features = []
all_labels = []
all_video_names = []
datasets = [

    ("SDFVD_FAKE", 1),

    ("SDFVD_REAL", 0),

    ("UADFV_FAKE", 1),

    ("UADFV_REAL", 0)

]

def extract_region_centers(landmarks):

    region_points = {}

    for region_name, indices in regions.items():

        pts = landmarks[indices]

        center = np.mean(pts, axis=0)

        region_points[region_name] = center

    return region_points


def extract_region_centers(landmarks):

    region_points = {}

    for region_name, indices in regions.items():

        pts = landmarks[indices]

        center = np.mean(pts, axis=0)

        region_points[region_name] = center

    return region_points

def process_video(video_path):

    frame_names = sorted(os.listdir(video_path))

    frames = []

    for frame in frame_names:

        img = cv2.imread(os.path.join(video_path, frame))

        if img is not None:
            frames.append(img)

    if len(frames) < 6:
        return None

    history = None

    vectors = []

    for i in range(1, len(frames)):

        landmarks1 = landmark_detector.detect(frames[i-1])
        landmarks2 = landmark_detector.detect(frames[i])

        if landmarks1 is None or landmarks2 is None:
            continue

        regions1 = extract_region_centers(landmarks1)
        regions2 = extract_region_centers(landmarks2)

        motion = optical_flow.calculate_motion(regions1, regions2)

        history = tracker.update(motion)

        # Need at least 5 motions
        if i < 5:
            continue

        features = adfmpc.extract_features(history)

        vector = vectorizer.flatten_features(features)

        vectors.append(vector)

    if len(vectors) == 0:
        return None

    vectors = np.stack(vectors)

    return np.mean(vectors, axis=0)

for frame_name in frame_names:

    path = os.path.join(video_path, frame_name)

    img = cv2.imread(path)

    if img is None:
        continue

    frames.append(img)

print("Frames Loaded :", len(frames))

print("\nDetecting landmarks...\n")

print("\nRegion Centers\n")

landmarks = landmark_detector.detect(frames[0])

region_centers = extract_region_centers(landmarks)

for region, center in region_centers.items():

    print(region, "->", center)

history = None

for i in range(1, 6):

    print(f"\nProcessing Frame Pair {i} -> {i+1}")

    landmarks1 = landmark_detector.detect(frames[i-1])
    landmarks2 = landmark_detector.detect(frames[i])

    if landmarks1 is None or landmarks2 is None:
        continue

    regions1 = extract_region_centers(landmarks1)
    regions2 = extract_region_centers(landmarks2)

    motion = optical_flow.calculate_motion(regions1, regions2)

    history = tracker.update(motion)

print("\nMotion History Created!")

print(history.keys())

print("\nHistory Length")

for region in history:

    print(region, len(history[region]))

print("\nExtracting AD-FMPC Features...\n")

features = adfmpc.extract_features(history)

print(features)
for dataset_name, label in datasets:

    dataset_path = os.path.join("faces", dataset_name)

    print(f"\nProcessing {dataset_name}")

    videos = sorted(os.listdir(dataset_path))

    for video in videos:

        video_path = os.path.join(dataset_path, video)

        print(video)

        feature = process_video(video_path)

        if feature is None:
            print("Skipped")
            continue

        all_features.append(feature)

        all_labels.append(label)

        all_video_names.append(video)

features = torch.tensor(
    np.array(all_features),
    dtype=torch.float32
)

labels = torch.tensor(
    all_labels,
    dtype=torch.long
)
os.makedirs("features", exist_ok=True)
torch.save(
    features,
    "features/ADFMPC_features.pt"
)

torch.save(
    labels,
    "features/ADFMPC_labels.pt"
)

torch.save(
    all_video_names,
    "features/ADFMPC_video_names.pt"
)
print("\nExtraction Finished")

print("Videos :", len(all_video_names))

print("Features Shape :", features.shape)

print("Labels Shape :", labels.shape)
print(video.shape)

print(video)