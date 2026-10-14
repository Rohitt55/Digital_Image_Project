"""Improve local grayscale contrast with CLAHE."""
import cv2
from code_04_image_helpers import example_path, load_image, show_images


def apply_clahe(image, clip_limit=2.0, tile_size=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(image)


if __name__ == '__main__':
    image = load_image(example_path(), grayscale=True)
    show_images({'Original': image, 'CLAHE': apply_clahe(image)}, histograms=True)
