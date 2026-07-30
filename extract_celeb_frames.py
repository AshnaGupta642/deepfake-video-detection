import os
import cv2
from tqdm import tqdm


def extract_frames(video_folder, output_folder, num_frames=10):

    os.makedirs(output_folder, exist_ok=True)

    videos = os.listdir(video_folder)

    for video in tqdm(videos):

        video_path = os.path.join(video_folder, video)

        video_name = os.path.splitext(video)[0]

        save_path = os.path.join(output_folder, video_name)

        os.makedirs(save_path, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            continue

        # 10 frames uniformly select
        frame_indices = [
            int(i * total_frames / num_frames)
            for i in range(num_frames)
        ]

        current_frame = 0
        saved = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if current_frame in frame_indices:

                frame_name = f"frame_{saved:03d}.jpg"

                cv2.imwrite(
                    os.path.join(save_path, frame_name),
                    frame
                )

                saved += 1

            current_frame += 1

            if saved == num_frames:
                break

        cap.release()


# Fake
extract_frames(
    "Celeb_subset/FAKE",
    "extracted_frames/CELEBDF_FAKE",
    num_frames=10
)


# Real
extract_frames(
    "Celeb_subset/REAL",
    "extracted_frames/CELEBDF_REAL",
    num_frames=10
)


print("Frame Extraction Completed!")