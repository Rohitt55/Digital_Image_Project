"""Resize an image to 100 by 100 pixels."""
import cv2
from code_04_image_helpers import example_path, load_image, show_images


def resize_image(image, width=100, height=100):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    image = load_image(example_path())
    show_images({'Original': image, 'Resized': resize_image(image)})
