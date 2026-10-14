"""Stretch grayscale intensities from their current range to 0–255."""
import numpy as np
from code_04_image_helpers import example_path, load_image, show_images


def stretch_contrast(image):
    values = image.astype(float)
    minimum = values.min()
    maximum = values.max()
    if maximum == minimum:
        return image.copy()  # A constant image has no range to stretch.
    return ((values - minimum) * 255 / (maximum - minimum)).astype(np.uint8)


if __name__ == '__main__':
    image = load_image(example_path(), grayscale=True)
    show_images({'Original': image, 'Stretched': stretch_contrast(image)}, histograms=True)
