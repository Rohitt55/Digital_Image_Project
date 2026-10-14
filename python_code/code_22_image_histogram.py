"""Count the occurrences of each intensity in a small image."""
import numpy as np
import matplotlib.pyplot as plt


def image_histogram(image, levels=256):
    return np.bincount(image.ravel(), minlength=levels)


if __name__ == '__main__':
    image = np.array([
        [2, 3, 4, 4, 6], [1, 2, 4, 5, 6], [1, 1, 5, 6, 6],
        [0, 1, 3, 3, 4], [0, 1, 2, 3, 4],
    ], dtype=np.uint8)
    histogram = image_histogram(image, levels=7)
    for intensity, count in enumerate(histogram):
        print(f'Intensity {intensity}: {count} pixels')
    figure, axes = plt.subplots(1, 2)
    axes[0].imshow(image, cmap='gray', vmin=0, vmax=6)
    axes[0].set_title('Grayscale image')
    axes[1].bar(range(7), histogram)
    axes[1].set(title='Histogram', xlabel='Intensity', ylabel='Pixel count')
    figure.tight_layout()
    plt.show()
