"""Matplotlib figures and Python-style printed output for the web lab."""
import base64
import io
import os
from pathlib import Path
import tempfile
import threading

import cv2
import numpy as np

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'pixellab-matplotlib'))

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from code_02_image_operations import gray

_plot_lock = threading.Lock()


def preview(image):
    """Bound figure resolution without changing the full-image measurements."""
    h, w = image.shape[:2]
    if max(h, w) > 960:
        scale = 960 / max(h, w)
        image = cv2.resize(
            image,
            (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image


def draw_image(axis, image, title):
    axis.imshow(preview(image), cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    axis.set_title(title, fontsize=12)
    axis.set_xticks([])
    axis.set_yticks([])
    if image.ndim == 2 and max(image.shape) <= 16:
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                value = int(image[y, x])
                axis.text(
                    x,
                    y,
                    str(value),
                    ha='center',
                    va='center',
                    fontsize=6,
                    color='black' if value > 127 else 'white'
                )


def matrix_text(image, label):
    values = gray(image)
    if max(values.shape) <= 16:
        heading = f'{label} grayscale pixel matrix (actual {values.shape[0]} x {values.shape[1]}):'
    else:
        values = cv2.resize(values, (8, 8), interpolation=cv2.INTER_AREA)
        heading = f'{label} grayscale pixel sample (resampled to 8 x 8 using area interpolation):'
    return heading + '\n' + np.array2string(values, max_line_width=100, threshold=256)


def make_output(original, result, before, after, operation, params, details):
    """Return a downloadable PNG figure and text derived from actual backend results."""
    with _plot_lock:
        figure = Figure(figsize=(14, 8), dpi=140, facecolor='white')
        axes = figure.subplots(2, 3)
        figure.suptitle(f'Digital Image Processing — {operation}', fontsize=16)
        for row, (image, stats, label, color) in enumerate([
            (original, before, 'Original', '#526ba3'),
            (result, after, 'Processed', '#277454')
        ]):
            draw_image(axes[row, 0], image, f"{label} image ({stats['width']} × {stats['height']})")
            histogram = axes[row, 1]
            histogram.bar(np.arange(256), stats['histogram'], width=1, color=color)
            histogram.set_title(f'{label} grayscale histogram', fontsize=12)
            histogram.set_xlabel('Pixel intensity')
            histogram.set_ylabel('Frequency (pixel count)')
            histogram.set_xlim(0, 255)
            histogram.set_ylim(bottom=0)
            histogram.ticklabel_format(axis='y', style='sci', scilimits=(0, 4))
            cdf = axes[row, 2]
            cdf.plot(np.arange(256), stats['cdf'], color=color, linewidth=1.8)
            cdf.set_title(f'{label} cumulative distribution (CDF)', fontsize=12)
            cdf.set_xlabel('Pixel intensity')
            cdf.set_ylabel('Cumulative probability')
            cdf.set_xlim(0, 255)
            cdf.set_ylim(0, 1.02)
            for axis in (histogram, cdf):
                axis.grid(axis='y', alpha=0.2)
                axis.set_axisbelow(True)
                axis.tick_params(labelsize=8)
        figure.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5, w_pad=2.5)
        buffer = io.BytesIO()
        FigureCanvasAgg(figure).print_png(buffer)
        figure.clear()
    figure_data = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
    lines = [
        f'>>> Operation: {operation}',
        f"Parameters: {params or 'Default settings'}",
        '',
        'IMAGE STATISTICS (grayscale luminance)',
        f"{'Metric':<22}{'Original':>16}{'Processed':>16}",
        '-' * 54
    ]
    for label, key in [
        ('Width', 'width'),
        ('Height', 'height'),
        ('Channels', 'channels'),
        ('Minimum intensity', 'min'),
        ('Maximum intensity', 'max'),
        ('Mean intensity', 'mean'),
        ('RMS contrast', 'rms_contrast'),
        ('Michelson contrast', 'michelson')
    ]:
        lines.append(f'{label:<22}{str(before[key]):>16}{str(after[key]):>16}')
    lines.extend(['', matrix_text(original, 'Original'), '', matrix_text(result, 'Processed')])
    for label, stats in [('Original', before), ('Processed', after)]:
        lines.extend([
            '',
            f'{label} histogram (intensities 0–255, pixel counts):',
            np.array2string(np.array(stats['histogram']), max_line_width=100, threshold=256),
            f'{label} CDF (intensities 0–255):',
            np.array2string(np.array(stats['cdf']), precision=4, max_line_width=100, threshold=256)
        ])
    lines.extend(['', 'OPERATION RESULT'])
    for key, value in details.items():
        if key not in ('pixel_matrix', 'download'):
            lines.append(f'{key}: {value}')
    lines.extend(['', 'Process finished successfully.'])
    return dict(figure=figure_data, console='\n'.join(lines))
