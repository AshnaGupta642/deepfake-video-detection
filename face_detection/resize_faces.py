import cv2


def resize_face(face, size=(112, 112)):
    """
    Resize cropped face to 112 x 112 pixels.
    """

    resized_face = cv2.resize(face, size)

    return resized_face