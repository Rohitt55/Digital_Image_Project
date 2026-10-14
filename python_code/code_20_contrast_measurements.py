"""Measure grayscale range, Michelson contrast and RMS contrast."""
from code_04_image_helpers import example_path, load_image, show_images


def measure_contrast(image):
    values = image.astype(float)
    minimum = float(values.min())
    maximum = float(values.max())
    intensity_sum = maximum + minimum
    return {
        'Minimum': minimum,
        'Maximum': maximum,
        'Mean': float(values.mean()),
        'Range contrast': maximum - minimum,
        'Michelson contrast': (maximum - minimum) / intensity_sum if intensity_sum else 0,
        'RMS contrast': float(values.std()),
    }


if __name__ == '__main__':
    image = load_image(example_path(), grayscale=True)
    for name, value in measure_contrast(image).items():
        print(f'{name}: {value:.4f}')
    show_images({'Grayscale': image}, histograms=True)
