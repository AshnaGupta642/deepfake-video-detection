import os

def get_video_files(folder_path, extensions):
    """
    Returns a list of video file paths from a folder.
    """

    videos = []

    for file in os.listdir(folder_path):
        if file.lower().endswith(extensions):
            videos.append(os.path.join(folder_path, file))

    return sorted(videos)