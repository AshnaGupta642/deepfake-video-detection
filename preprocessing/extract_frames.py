import os
import cv2
import random
from tqdm import tqdm

from config import (
    DATASET_PATHS,
    VIDEO_EXTENSIONS,
    NUM_FRAMES,
    OUTPUT_FOLDER
)

from utils import get_video_files


def extract_random_frames(video_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < NUM_FRAMES:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = sorted(random.sample(range(total_frames), NUM_FRAMES))

    count = 0

    for idx in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        success, frame = cap.read()

        if success:

            filename = f"frame_{count+1:03d}.jpg"

            cv2.imwrite(
                os.path.join(output_dir, filename),
                frame
            )

            count += 1

    cap.release()


def process_dataset(dataset_name, folder):

    videos = get_video_files(folder, VIDEO_EXTENSIONS)

    print(f"\nProcessing {dataset_name}")

    for video in tqdm(videos):

        video_name = os.path.splitext(os.path.basename(video))[0]

        output_dir = os.path.join(
            OUTPUT_FOLDER,
            dataset_name,
            video_name
        )

        extract_random_frames(video, output_dir)


def main():

    for dataset_name, folder in DATASET_PATHS.items():

        process_dataset(dataset_name, folder)


if __name__ == "__main__":
    main()