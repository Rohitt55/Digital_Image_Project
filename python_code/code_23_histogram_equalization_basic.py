"""Equalize a small image using mapping = round(255 × CDF)."""
import numpy as np
from code_04_image_helpers import sample_pixels, show_images


def equalize_histogram(image):
    histogram = np.bincount(image.ravel(), minlength=256)
    probability = histogram / image.size
    cdf = probability.cumsum()
    mapping = np.round(255 * cdf).astype(np.uint8)
    return mapping[image], mapping, cdf


if __name__ == '__main__':
    image = sample_pixels()
    equalized, mapping, cdf = equalize_histogram(image)
    for intensity in np.unique(image):
        print(f'{intensity} -> {mapping[intensity]}, CDF: {cdf[intensity]:.4f}')
    print('Original:\n', image, '\nEqualized:\n', equalized)
    show_images({'Original': image, 'Equalized': equalized}, histograms=True, pixel_values=True)
