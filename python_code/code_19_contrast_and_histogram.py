"""Compare contrast adjustment, equalization and CLAHE."""
import cv2
from code_04_image_helpers import example_path, load_image, show_images


def enhance_images(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
    return {
        'Original': image,
        'Low contrast': cv2.convertScaleAbs(image, alpha=0.5),
        'High contrast': cv2.convertScaleAbs(image, alpha=1.8),
        'Grayscale': gray,
        'Equalized': cv2.equalizeHist(gray),
        'CLAHE': clahe.apply(gray),
    }


if __name__ == '__main__':
    show_images(enhance_images(load_image(example_path())), histograms=True)
