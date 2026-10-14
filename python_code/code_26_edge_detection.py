"""Compare Sobel, Prewitt, Roberts and Canny edges."""
import cv2
import numpy as np
from code_04_image_helpers import example_path, load_image, show_images


def gradient_edges(image, horizontal_kernel, vertical_kernel):
    horizontal = cv2.filter2D(image, cv2.CV_32F, np.array(horizontal_kernel, np.float32))
    vertical = cv2.filter2D(image, cv2.CV_32F, np.array(vertical_kernel, np.float32))
    magnitude = cv2.magnitude(horizontal, vertical)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def detect_edges(image):
    return {
        'Original': image,
        'Sobel': gradient_edges(image, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        'Prewitt': gradient_edges(image, [[-1, 0, 1]] * 3,
                                  [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
        'Roberts': gradient_edges(image, [[1, 0], [0, -1]], [[0, 1], [-1, 0]]),
        'Canny': cv2.Canny(image, 100, 200),
    }


if __name__ == '__main__':
    image = load_image(example_path(), grayscale=True)
    show_images(detect_edges(image))
