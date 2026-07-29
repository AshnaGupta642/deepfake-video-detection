import cv2
import face_alignment
import torch


class FaceLandmarks:

    def __init__(self):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=device
        )

        # Disable torch compile if available
        try:
            torch._dynamo.config.disable = True
        except:
            pass


    def detect(self, image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            landmarks = self.fa.get_landmarks(image)

        if landmarks is None:
            return None

        return landmarks[0]