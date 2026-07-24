import cv2
from mtcnn import MTCNN


class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()

    def detect_faces(self, image):
        """
        Detect all faces in an image.
        Returns a list of face bounding boxes.
        """

        results = self.detector.detect_faces(image)

        faces = []

        for result in results:
            x, y, width, height = result["box"]

            # Sometimes MTCNN may return negative x or y
            x = max(0, x)
            y = max(0, y)

            faces.append((x, y, width, height))

        return faces