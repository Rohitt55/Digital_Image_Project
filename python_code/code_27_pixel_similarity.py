"""Compare grayscale pixels with an intensity tolerance."""
import argparse
import cv2
import numpy as np
from code_04_image_helpers import load_image, show_images


def compare_pixels(source, reference, tolerance=5):
    height, width = source.shape
    aligned = cv2.resize(reference, (width, height))
    # Signed integers prevent uint8 subtraction from wrapping around.
    difference = np.abs(source.astype(np.int16) - aligned.astype(np.int16))
    similar_pixels = int(np.count_nonzero(difference <= tolerance))
    return aligned, {
        'Similar pixels': similar_pixels,
        'Different pixels': source.size - similar_pixels,
        'Total pixels': source.size,
        'Tolerance': tolerance,
        'Similarity (%)': 100 * similar_pixels / source.size,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='Source image path')
    parser.add_argument('reference', help='Reference image path')
    parser.add_argument('--tolerance', type=int, default=5, choices=range(256), metavar='0..255')
    args = parser.parse_args()
    source = load_image(args.source, grayscale=True)
    reference = load_image(args.reference, grayscale=True)
    aligned, measurements = compare_pixels(source, reference, args.tolerance)
    for name, value in measurements.items():
        print(f'{name}: {value}')
    show_images({'Source': source, 'Reference (resized)': aligned}, histograms=True)
