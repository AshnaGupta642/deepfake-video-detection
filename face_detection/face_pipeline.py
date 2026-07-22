import os
import cv2
from face_detection.mtcnn_detector import FaceDetector
from face_detection.preprocess import crop_face
from face_detection.resize_faces import resize_face


# Input folder: Member 1 ke extracted frames
INPUT_DIR = "extracted_frames"

# Output folder: Member 2 ke processed faces
OUTPUT_DIR = "faces"


detector = FaceDetector()


for root, dirs, files in os.walk(INPUT_DIR):

    for file in files:

        # Sirf image files process karo
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # Input image ka complete path
        input_path = os.path.join(root, file)

        # Image read karo
        image = cv2.imread(input_path)

        if image is None:
            print(f"Image load nahi hui: {input_path}")
            continue

        # MTCNN ke liye BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Faces detect karo
        boxes = detector.detect_faces(image_rgb)

        if len(boxes) == 0:
            print(f"Face nahi mila: {input_path}")
            continue

        # Pehla detected face use kar rahe hain
        face = crop_face(image, boxes[0])

        # 112 x 112 resize
        face = resize_face(face)

        # Same folder structure output mein maintain karna
        relative_path = os.path.relpath(input_path, INPUT_DIR)
        output_path = os.path.join(OUTPUT_DIR, relative_path)

        # Output folder create karo
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Face save karo
        cv2.imwrite(output_path, face)

        print(f"Processed: {input_path}")