"""Equalize using (CDF − CDFmin) / (pixel count − CDFmin)."""
import numpy as np
from code_04_image_helpers import sample_pixels, show_images


def equalize_histogram(image):
    histogram = np.bincount(image.ravel(), minlength=256)
    cdf = histogram.cumsum()
    minimum_cdf = cdf[cdf > 0][0]
    if image.size == minimum_cdf:
        return image.copy(), np.arange(256, dtype=np.uint8), cdf
    mapping = np.round(255 * (cdf - minimum_cdf) / (image.size - minimum_cdf))
    mapping = np.clip(mapping, 0, 255).astype(np.uint8)
    return mapping[image], mapping, cdf


if __name__ == '__main__':
    image = sample_pixels()
    equalized, mapping, cdf = equalize_histogram(image)
    for intensity in np.unique(image):
        print(f'{intensity} -> {mapping[intensity]}, cumulative count: {cdf[intensity]}')
    print('Original:\n', image, '\nEqualized:\n', equalized)
    show_images({'Original': image, 'Equalized': equalized}, histograms=True, pixel_values=True)
