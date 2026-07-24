import cv2


def crop_face(image, box):
    """
    Crop a face from an image using bounding box.
    box = (x, y, width, height)
    """

    x, y, width, height = box

    face = image[y:y + height, x:x + width]

    return face