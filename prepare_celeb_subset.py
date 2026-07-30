import os
import shutil
import random


# Source folders
fake_source = "Celeb-DF-v2/Celeb-synthesis"

real_sources = [
    "Celeb-DF-v2/Celeb-real",
    "Celeb-DF-v2/YouTube-real"
]


# Destination folders
fake_dest = "Celeb_subset/FAKE"
real_dest = "Celeb_subset/REAL"


os.makedirs(fake_dest, exist_ok=True)
os.makedirs(real_dest, exist_ok=True)


# Number of videos
FAKE_COUNT = 500
REAL_COUNT = 700


# -------------------------
# Copy Fake videos
# -------------------------

fake_videos = os.listdir(fake_source)

random.shuffle(fake_videos)

fake_videos = fake_videos[:FAKE_COUNT]


for video in fake_videos:

    src = os.path.join(fake_source, video)
    dst = os.path.join(fake_dest, video)

    shutil.copy2(src, dst)


print("Fake videos copied:", len(fake_videos))


# -------------------------
# Copy Real videos
# -------------------------

real_videos = []

for folder in real_sources:

    videos = os.listdir(folder)

    for v in videos:
        real_videos.append(
            os.path.join(folder, v)
        )


random.shuffle(real_videos)

real_videos = real_videos[:REAL_COUNT]


for video_path in real_videos:

    video_name = os.path.basename(video_path)

    dst = os.path.join(real_dest, video_name)

    shutil.copy2(video_path, dst)


print("Real videos copied:", len(real_videos))


print("\nSubset preparation completed!")