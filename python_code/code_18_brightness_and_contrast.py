"""Adjust contrast with alpha and brightness with beta."""
import cv2
from code_04_image_helpers import example_path, load_image, show_images


def adjust_image(image, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


if __name__ == '__main__':
    image = load_image(example_path())
    show_images({
        'Original': image,
        'Low contrast': adjust_image(image, alpha=0.8),
        'High contrast': adjust_image(image, alpha=1.8),
        'Brighter': adjust_image(image, beta=30),
    })
