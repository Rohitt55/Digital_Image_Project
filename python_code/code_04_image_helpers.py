import cv2
import numpy as np


def example_path():
    """Use a command-line image path, or image.jpg in the current folder."""
    import argparse

    parser = argparse.ArgumentParser(description='Run this image-processing example.')
    parser.add_argument('image', nargs='?', default='image.jpg', help='Input image path')
    return parser.parse_args().image


def load_image(path, grayscale=False):
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return image


def show_images(images, histograms=False, pixel_values=False):
    """Display a title-to-image dictionary with optional grayscale histograms."""
    import matplotlib.pyplot as plt

    columns = 2 if histograms else min(3, len(images))
    rows = len(images) if histograms else (len(images) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.5 * rows), squeeze=False)
    for index, (title, image) in enumerate(images.items()):
        axis = axes.flat[index * 2 if histograms else index]
        display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        axis.imshow(display, cmap='gray', vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis('off')
        if pixel_values and image.ndim == 2 and image.size <= 256:
            for row, column in np.ndindex(image.shape):
                value = int(image[row, column])
                color = 'black' if value > 127 else 'white'
                axis.text(column, row, str(value), ha='center', va='center', color=color)
        if histograms:
            values = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            histogram_axis = axes.flat[index * 2 + 1]
            histogram_axis.hist(values.ravel(), bins=256, range=(0, 256))
            histogram_axis.set_title(title + ' histogram')
            histogram_axis.set_xlabel('Intensity')
            histogram_axis.set_ylabel('Pixel count')
    for axis in list(axes.flat)[len(images) * (2 if histograms else 1):]:
        axis.axis('off')
    figure.tight_layout()
    plt.show()


def sample_pixels():
    """Small grayscale matrix used by both equalization examples."""
    return np.array([
        [52, 55, 61, 59, 79, 61, 76, 61],
        [62, 59, 55, 104, 94, 85, 59, 71],
        [63, 65, 66, 113, 144, 104, 63, 72],
        [64, 70, 70, 126, 154, 109, 71, 69],
        [67, 73, 68, 106, 122, 88, 68, 68],
        [68, 79, 60, 70, 77, 66, 58, 75],
        [69, 85, 64, 58, 55, 61, 65, 83],
        [70, 87, 69, 68, 65, 73, 78, 90],
    ], dtype=np.uint8)


def bounded(image, maximum=512):
    h, w = image.shape[:2]
    if max(h, w) <= maximum:
        return image.copy()
    scale = maximum / max(h, w)
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def label_image(labels):
    """Stable colors with label zero shown as background."""
    values = labels.astype(np.uint32)
    colors = np.stack((values * 73 % 255, values * 151 % 255, values * 199 % 255), axis=-1).astype(np.uint8)
    colors[labels == 0] = 0
    return colors


def u8(values):
    return np.clip(np.round(values), 0, 255).astype(np.uint8)


def reference_gray(reference, shape):
    from code_02_image_operations import gray
    if reference is None:
        raise ValueError('Upload a reference image for this operation.')
    return cv2.resize(gray(reference), (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
