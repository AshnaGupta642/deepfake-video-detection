import cv2
from landmarks import FaceLandmarks

img = cv2.imread(r"C:\Users\ASHNA GUPTA\deepfake-video-detection\faces\SDFVD_FAKE\vs1\frame_026.jpg")

detector = FaceLandmarks()

points = detector.detect(img)

print(points.shape)