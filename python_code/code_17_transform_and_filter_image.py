"""Apply smoothing filters and basic image transformations."""
from pathlib import Path
import cv2
from code_04_image_helpers import example_path, load_image, show_images


def filter_images(image):
    return {
        'Original': image,
        'Average': cv2.blur(image, (5, 5)),
        'Gaussian': cv2.GaussianBlur(image, (5, 5), 0),
        'Median': cv2.medianBlur(image, 5),
        'Bilateral': cv2.bilateralFilter(image, 9, 75, 75),
    }


def transform_images(image):
    height, width = image.shape[:2]
    # Crop the central half so the example works with small images too.
    crop = image[height // 4:max(height // 4 + 1, 3 * height // 4),
                 width // 4:max(width // 4 + 1, 3 * width // 4)]
    return {
        'Cropped': crop,
        'Resized': cv2.resize(image, (300, 300)),
        'Rotated': cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        'Horizontal flip': cv2.flip(image, 1),
        'Vertical flip': cv2.flip(image, 0),
        'Both flips': cv2.flip(image, -1),
    }


if __name__ == '__main__':
    image = load_image(example_path())
    results = transform_images(image)
    show_images(filter_images(image))
    show_images({'Original': image, **results})
    output_folder = Path('results')
    output_folder.mkdir(exist_ok=True)
    for title, result in results.items():
        filename = title.lower().replace(' ', '_') + '.png'
        if not cv2.imwrite(str(output_folder / filename), result):
            raise ValueError(f'Could not save {filename}')
    print('Saved transformed images in:', output_folder.resolve())
