import cv2
import os

from config import DATASET_PATHS, VIDEO_EXTENSIONS
from utils import get_video_files


def load_dataset(dataset_name, folder):

    videos = get_video_files(folder, VIDEO_EXTENSIONS)

    print("=" * 60)
    print(dataset_name)
    print(f"Total Videos : {len(videos)}")
    print("=" * 60)

    for video_path in videos:

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"[ERROR] Cannot open : {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = total_frames / fps if fps != 0 else 0

        print(f"Video      : {os.path.basename(video_path)}")
        print(f"Frames     : {total_frames}")
        print(f"FPS        : {fps:.2f}")
        print(f"Resolution : {width} x {height}")
        print(f"Duration   : {duration:.2f} sec")
        print("-" * 60)

        cap.release()


def main():

    for name, path in DATASET_PATHS.items():

        load_dataset(name, path)


if __name__ == "__main__":
    main()