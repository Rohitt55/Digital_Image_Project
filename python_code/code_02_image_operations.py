"""Backend algorithms adapted from the supplied labs, with validated user parameters.

OpenCV images remain BGR internally. Scalar calculations use floating point to
avoid the uint8 overflow present in some of the original demonstration scripts.
"""
import inspect
import math
import time
from pathlib import Path

import cv2
import numpy as np

OPERATIONS = {}


def field(key, label, default, low, high, step=1):
    return dict(key=key, label=label, default=default, min=low, max=high, step=step, type='number')


def choice(key, label, default, options):
    return dict(key=key, label=label, default=default, options=options, type='select')


def operation(key, name, group, explanation, formula, source, fields=(), requires_reference=False):

    def register(function):
        OPERATIONS[key] = dict(
            id=key,
            name=name,
            group=group,
            explanation=explanation,
            formula=formula,
            source=source,
            fields=list(fields),
            function=function,
            requires_reference=requires_reference
        )
        return function
    return register


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def normalize(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def statistics(image):
    values = gray(image)
    lo, hi = (int(values.min()), int(values.max()))
    histogram = np.bincount(values.ravel(), minlength=256)
    return dict(
        width=image.shape[1],
        height=image.shape[0],
        channels=1 if image.ndim == 2 else 3,
        min=lo,
        max=hi,
        mean=round(float(values.mean()), 3),
        rms_contrast=round(float(values.astype(float).std()), 3),
        michelson=round((hi - lo) / (hi + lo), 4) if hi + lo else 0,
        histogram=histogram.tolist(),
        cdf=(histogram.cumsum() / values.size).tolist()
    )


def describe(key):
    item = OPERATIONS[key]
    function = item['function']
    implementation = Path(inspect.getsourcefile(function)).resolve()
    code_root = Path(__file__).resolve().parent
    names = {'cv2': 'OpenCV', 'numpy': 'NumPy', 'scipy': 'SciPy', 'skimage': 'scikit-image', 'pywt': 'PyWavelets'}
    libraries = set()
    for value in function.__globals__.values():
        if inspect.ismodule(value):
            module_name = value.__name__.split('.')[0]
            if module_name in names:
                libraries.add(names[module_name])
    return {
        **{k: v for k, v in item.items() if k != 'function'},
        'code': inspect.getsource(function),
        'implementation_file': 'python_code/' + implementation.relative_to(code_root).as_posix(),
        'module_libraries': sorted(libraries),
        'code_type': 'Python image-processing algorithm / educational demo'
    }


def execute(key, image, params, reference=None):
    spec = OPERATIONS[key]
    clean = {}
    for item in spec['fields']:
        value = params.get(item['key'], item['default'])
        if item['type'] == 'select':
            if value not in item['options']:
                raise ValueError('Invalid ' + item['label'] + '.')
        elif item['type'] == 'text':
            if not isinstance(value, str) or len(value) > 512:
                raise ValueError(item['label'] + ' must be text of at most 512 characters.')
        else:
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(item['label'] + ' must be a number.') from exc
            if not math.isfinite(value) or not item['min'] <= value <= item['max']:
                raise ValueError(f"{item['label']} must be between {item['min']} and {item['max']}.")
            if item['step'] == 1:
                if not value.is_integer():
                    raise ValueError(item['label'] + ' must be a whole number.')
                value = int(value)
        clean[item['key']] = value
    if 'kernel' in clean and clean['kernel'] % 2 == 0:
        raise ValueError('Kernel size must be an odd number (3, 5, 7, ...).')
    start = time.perf_counter()
    output = spec['function'](image, clean, reference)
    result, extra = output if isinstance(output, tuple) else (output, {})
    extra['processing_ms'] = round((time.perf_counter() - start) * 1000, 2)
    return (result, extra)


@operation(
    'grayscale',
    'Grayscale',
    'Color & intensity',
    'Convert color into luminance using weighted RGB channels.',
    'Y = 0.299R + 0.587G + 0.114B',
    'code_19_contrast_and_histogram.py'
)
def grayscale(image, params, reference):
    return gray(image)


@operation(
    'negative',
    'Image negative',
    'Color & intensity',
    'Invert each intensity to highlight darker structures.',
    's = 255 - r',
    'Chap2 Image Enhancement.pdf'
)
def negative(image, params, reference):
    return 255 - image


@operation(
    'contrast',
    'Brightness & contrast',
    'Color & intensity',
    'Scale intensity and add a brightness offset. Saturate values at the valid range.',
    's = clip(alpha × r + beta, 0, 255)',
    'code_18_brightness_and_contrast.py',
    [field('alpha', 'Contrast (alpha)', 1.3, 0, 4, .1), field('beta', 'Brightness (beta)', 0, -255, 255)]
)
def contrast(image, params, reference):
    return np.clip(image.astype(np.float32) * params['alpha'] + params['beta'], 0, 255).astype(np.uint8)


@operation(
    'gamma',
    'Gamma correction',
    'Color & intensity',
    'A gamma below 1 brightens midtones; above 1 darkens them.',
    's = 255 × (r / 255)^gamma',
    'Chap2 Image Enhancement.pdf',
    [field('gamma', 'Gamma', .7, .1, 5, .1)]
)
def gamma(image, params, reference):
    table = np.round(255 * (np.arange(256) / 255) ** params['gamma']).astype(np.uint8)
    return cv2.LUT(image, table)


@operation(
    'quantization',
    'Intensity quantization',
    'Color & intensity',
    'Reduce grayscale or per-channel RGB intensity levels to demonstrate digital image quantization.',
    'q = floor(r × levels / 256); s = round(q × 255 / (levels - 1))',
    'DIP-Lab1.pdf',
    [
        choice('bits', 'Bit depth', '3', ['1', '2', '3', '4', '5', '6', '7', '8']),
        choice('domain','Quantization domain','grayscale',['grayscale','RGB channels'])
    ]
)
def quantization(image, params, reference):
    levels = 2 ** int(params['bits'])
    indices = np.floor((gray(image) if params['domain'] == 'grayscale' else image).astype(float) * levels / 256)
    return (np.round(indices * 255 / (levels - 1)).astype(np.uint8), {'intensity_levels': levels})


@operation(
    'color_channel',
    'Color space channels',
    'Color & intensity',
    'Inspect an individual RGB, HSV or Lab channel as a grayscale map. HSV hue is scaled from 0–179 to 0–255 for display.',
    'Convert color space → select channel → display intensity',
    'DIP-Lab1.pdf / Chap3-Color Image Processing.pdf',
    [choice(
        'channel',
        'Channel',
        'HSV hue',
        ['Red', 'Green', 'Blue', 'HSV hue', 'HSV saturation', 'HSV value', 'Lab L', 'Lab a', 'Lab b']
    )]
)
def color_channel(image, params, reference):
    channel = params['channel']
    if channel in ['Red', 'Green', 'Blue']:
        return image[:, :, {'Red': 2, 'Green': 1, 'Blue': 0}[channel]]
    if channel.startswith('HSV'):
        values = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        result = values[:, :, {'HSV hue': 0, 'HSV saturation': 1, 'HSV value': 2}[channel]]
        return np.round(result.astype(float) * 255 / 179).astype(np.uint8) if channel == 'HSV hue' else result
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, {'Lab L': 0, 'Lab a': 1, 'Lab b': 2}[channel]]


@operation(
    'pixel_matrix',
    'Digital pixel matrix',
    'Geometry',
    'Sample the grayscale image into a small matrix and inspect each pixel value.',
    'Digital image = spatial sampling + intensity quantization',
    'DIP-Lab1.pdf',
    [field('size', 'Matrix size', 8, 2, 16)]
)
def pixel_matrix(image, params, reference):
    sampled = cv2.resize(gray(image), (params['size'], params['size']), interpolation=cv2.INTER_AREA)
    return (sampled, {'pixel_matrix': sampled.tolist()})


@operation(
    'stretch',
    'Contrast stretching',
    'Histograms',
    'Map the darkest and brightest values to 0 and 255. Constant images stay unchanged.',
    's = (r - min) / (max - min) × 255',
    'code_21_contrast_stretching.py'
)
def stretch(image, params, reference):
    lo, hi = (float(image.min()), float(image.max()))
    return image.copy() if hi == lo else np.round((image.astype(float) - lo) / (hi - lo) * 255).astype(np.uint8)


@operation(
    'histogram',
    'Histogram & contrast analysis',
    'Histograms',
    'Inspect grayscale frequency, cumulative distribution and contrast statistics.',
    'p(r) = count(r) / N; CDF(r) = sum p(0…r)',
    'code_22_image_histogram.py / code_20_contrast_measurements.py'
)
def histogram(image, params, reference):
    return image.copy()


@operation(
    'equalize_basic',
    'Basic histogram equalization',
    'Histograms',
    'Apply the cumulative probability mapping from the basic lab.',
    's = round(255 × CDF(r))',
    'code_23_histogram_equalization_basic.py'
)
def equalize_basic(image, params, reference):
    values = gray(image)
    hist = np.bincount(values.ravel(), minlength=256)
    mapping = np.round(255 * hist.cumsum() / values.size).astype(np.uint8)
    return mapping[values]


@operation(
    'equalize',
    'Normalized equalization',
    'Histograms',
    'Subtract the first nonzero CDF to distribute grayscale values over the full range.',
    's = round((CDF - CDFmin) / (N - CDFmin) × 255)',
    'code_24_histogram_equalization_normalized.py'
)
def equalize(image, params, reference):
    values = gray(image)
    cdf = np.bincount(values.ravel(), minlength=256).cumsum()
    first = cdf[cdf > 0][0]
    if values.size == first:
        return values.copy()
    mapping = np.clip(np.round((cdf - first) / (values.size - first) * 255), 0, 255).astype(np.uint8)
    return mapping[values]


@operation(
    'clahe',
    'Adaptive equalization · CLAHE',
    'Histograms',
    'Enhance local grayscale contrast while limiting noise amplification.',
    'Local histogram equalization with a clip limit',
    'code_25_grayscale_clahe.py',
    [field('clip', 'Clip limit', 2, .1, 40, .1), field('grid', 'Tile grid size', 8, 2, 32)]
)
def clahe(image, params, reference):
    return cv2.createCLAHE(
        clipLimit=params['clip'],
        tileGridSize=(params['grid'], params['grid'])
    ).apply(gray(image))


@operation(
    'resize',
    'Resize / sampling',
    'Geometry',
    'Choose output dimensions and an interpolation method.',
    'New image = resample(image, width, height)',
    'code_16_resize_image.py',
    [
        field('width', 'Width (px)', 400, 1, 4096),
        field('height', 'Height (px)', 300, 1, 4096),
        choice('interpolation', 'Interpolation', 'area', ['area', 'nearest', 'linear', 'cubic'])
    ]
)
def resize(image, params, reference):
    if params['width'] * params['height'] > 12000000:
        raise ValueError('Output must contain at most 12 million pixels.')
    methods = dict(area=cv2.INTER_AREA, nearest=cv2.INTER_NEAREST, linear=cv2.INTER_LINEAR, cubic=cv2.INTER_CUBIC)
    return cv2.resize(image, (params['width'], params['height']), interpolation=methods[params['interpolation']])


@operation(
    'crop',
    'Crop image',
    'Geometry',
    'Specify a rectangle as percentages of the original dimensions.',
    'output = image[y : y+h, x : x+w]',
    'code_17_transform_and_filter_image.py',
    [
        field('x', 'Left (%)', 10, 0, 99),
        field('y', 'Top (%)', 10, 0, 99),
        field('width', 'Width (%)', 80, 1, 100),
        field('height', 'Height (%)', 80, 1, 100)
    ]
)
def crop(image, params, reference):
    if params['x'] + params['width'] > 100 or params['y'] + params['height'] > 100:
        raise ValueError('The crop rectangle must fit within the image (left + width and top + height ≤ 100).')
    h, w = image.shape[:2]
    x, y = (int(w * params['x'] / 100), int(h * params['y'] / 100))
    right = min(w, max(x + 1, int(w * (params['x'] + params['width']) / 100)))
    bottom = min(h, max(y + 1, int(h * (params['y'] + params['height']) / 100)))
    return image[y:bottom, x:right].copy()


@operation(
    'rotate',
    'Rotate image',
    'Geometry',
    'Rotate in exact quarter turns without interpolation.',
    '90°, 180° or 270° clockwise rotation',
    'code_17_transform_and_filter_image.py',
    [choice('angle', 'Clockwise angle', '90', ['90', '180', '270'])]
)
def rotate(image, params, reference):
    rotations = {'90': cv2.ROTATE_90_CLOCKWISE, '180': cv2.ROTATE_180, '270': cv2.ROTATE_90_COUNTERCLOCKWISE}
    return cv2.rotate(image, rotations[params['angle']])


@operation(
    'flip',
    'Flip image',
    'Geometry',
    'Reflect pixels horizontally, vertically or along both axes.',
    'Horizontal: f(x,y) → f(W-1-x,y)',
    'code_17_transform_and_filter_image.py',
    [choice('axis', 'Direction', 'horizontal', ['horizontal', 'vertical', 'both'])]
)
def flip(image, params, reference):
    return cv2.flip(image, {'horizontal': 1, 'vertical': 0, 'both': -1}[params['axis']])


KERNEL = [field('kernel', 'Kernel size (odd)', 5, 3, 31)]


@operation(
    'mean',
    'Average filter',
    'Filters & restoration',
    'Smooth the image by averaging neighboring pixels.',
    's(x,y) = mean of k × k neighborhood',
    'code_17_transform_and_filter_image.py',
    KERNEL
)
def mean(image, params, reference):
    return cv2.blur(image, (params['kernel'], params['kernel']))


@operation(
    'gaussian',
    'Gaussian filter',
    'Filters & restoration',
    'Smooth using a Gaussian weighted neighborhood.',
    'G(x,y) ∝ exp(-(x²+y²) / (2σ²))',
    'code_17_transform_and_filter_image.py',
    KERNEL
)
def gaussian(image, params, reference):
    return cv2.GaussianBlur(image, (params['kernel'], params['kernel']), 0)


@operation(
    'median',
    'Median filter',
    'Filters & restoration',
    'Reduce impulse noise by selecting the neighborhood median.',
    's(x,y) = median of k × k neighborhood',
    'code_17_transform_and_filter_image.py',
    KERNEL
)
def median(image, params, reference):
    return cv2.medianBlur(image, params['kernel'])


@operation(
    'bilateral',
    'Bilateral filter',
    'Filters & restoration',
    'Smooth similar colors while preserving strong boundaries.',
    'Weight = spatial Gaussian × intensity Gaussian',
    'code_17_transform_and_filter_image.py',
    [field('diameter', 'Neighborhood diameter', 9, 3, 15), field('sigma', 'Color / space sigma', 75, 1, 150)]
)
def bilateral(image, params, reference):
    return cv2.bilateralFilter(image, params['diameter'], params['sigma'], params['sigma'])


@operation(
    'sharpen',
    'Unsharp masking',
    'Filters & restoration',
    'Add a scaled high frequency residual to sharpen detail.',
    's = r + amount × (r - Gaussian(r))',
    'Chap2 Image Enhancement.pdf',
    [field('amount', 'Sharpening amount', 1, .1, 5, .1)]
)
def sharpen(image, params, reference):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return np.clip(
        image.astype(float) + params['amount'] * (image.astype(float) - blurred),
        0,
        255
    ).astype(np.uint8)


@operation(
    'sobel',
    'Sobel edges',
    'Edges & segmentation',
    'Find horizontal and vertical gradients and combine their magnitudes.',
    'Magnitude = sqrt(Gx² + Gy²)',
    'code_26_edge_detection.py'
)
def sobel(image, params, reference):
    values = gray(image)
    sx = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3)
    return normalize(cv2.magnitude(sx, sy))


@operation(
    'prewitt',
    'Prewitt edges',
    'Edges & segmentation',
    'Estimate gradients using uniform 3 × 3 derivative masks.',
    'Gx = [-1,0,1; -1,0,1; -1,0,1]; Gy = Gxᵀ',
    'code_26_edge_detection.py'
)
def prewitt(image, params, reference):
    kernel = np.array([[-1, 0, 1]] * 3, np.float32)
    values = gray(image)
    return normalize(cv2.magnitude(
        cv2.filter2D(values, cv2.CV_32F, kernel),
        cv2.filter2D(values, cv2.CV_32F, kernel.T)
    ))


@operation(
    'roberts',
    'Roberts edges',
    'Edges & segmentation',
    'Detect diagonal changes using small 2 × 2 masks.',
    'Gx = [1,0;0,-1]; Gy = [0,1;-1,0]',
    'code_26_edge_detection.py'
)
def roberts(image, params, reference):
    values = gray(image)
    gx = cv2.filter2D(values, cv2.CV_32F, np.array([[1, 0], [0, -1]], np.float32))
    gy = cv2.filter2D(values, cv2.CV_32F, np.array([[0, 1], [-1, 0]], np.float32))
    return normalize(cv2.magnitude(gx, gy))


@operation(
    'canny',
    'Canny edges',
    'Edges & segmentation',
    'Find thin edges with Gaussian smoothing and hysteresis thresholds.',
    'Smooth → gradients → suppression → hysteresis',
    'code_26_edge_detection.py',
    [field('low', 'Lower threshold', 100, 0, 255), field('high', 'Upper threshold', 200, 0, 255)]
)
def canny(image, params, reference):
    if params['low'] >= params['high']:
        raise ValueError('The lower threshold must be smaller than the upper threshold.')
    return cv2.Canny(cv2.GaussianBlur(gray(image), (5, 5), 0), params['low'], params['high'])


@operation(
    'threshold',
    'Binary threshold',
    'Edges & segmentation',
    'Separate pixels into foreground and background using one cutoff.',
    's = 255 if r > T, otherwise 0',
    'Chap5-Image Segmentation.pdf',
    [field('threshold', 'Threshold', 127, 0, 255)]
)
def threshold(image, params, reference):
    return cv2.threshold(gray(image), params['threshold'], 255, cv2.THRESH_BINARY)[1]


@operation(
    'otsu',
    'Otsu segmentation',
    'Edges & segmentation',
    'Choose the threshold that minimizes within-class variance automatically.',
    'T* = argmin within-class variance',
    'Chap5-Image Segmentation.pdf'
)
def otsu(image, params, reference):
    cutoff, output = cv2.threshold(gray(image), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (output, {'selected_threshold': cutoff})


@operation(
    'adaptive',
    'Adaptive threshold',
    'Edges & segmentation',
    'Compute a local threshold to handle uneven illumination.',
    'T(x,y) = Gaussian weighted neighborhood mean - C',
    'Chap5-Image Segmentation.pdf',
    [field('kernel', 'Block size (odd)', 11, 3, 51), field('constant', 'Subtracted constant', 2, -30, 30)]
)
def adaptive(image, params, reference):
    return cv2.adaptiveThreshold(
        gray(image),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        params['kernel'],
        params['constant']
    )


@operation(
    'morphology',
    'Morphological processing',
    'Morphology',
    'Apply a structuring element to a thresholded binary image.',
    'Opening = erode then dilate; closing = dilate then erode',
    'Chap8-Morphological Processing.pdf',
    [
        choice('method', 'Operation', 'opening', ['erosion', 'dilation', 'opening', 'closing', 'gradient']),
        field('kernel', 'Kernel size (odd)', 5, 3, 31),
        field('iterations', 'Iterations', 1, 1, 5),
        field('threshold', 'Binary threshold', 127, 0, 255)
    ]
)
def morphology(image, params, reference):
    binary = cv2.threshold(gray(image), params['threshold'], 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((params['kernel'], params['kernel']), np.uint8)
    methods = dict(
        erosion=cv2.MORPH_ERODE,
        dilation=cv2.MORPH_DILATE,
        opening=cv2.MORPH_OPEN,
        closing=cv2.MORPH_CLOSE,
        gradient=cv2.MORPH_GRADIENT
    )
    return cv2.morphologyEx(binary, methods[params['method']], kernel, iterations=params['iterations'])


@operation(
    'fft',
    'FFT / DFT spectrum',
    'Transforms & comparison',
    'Use FFT to compute the 2D DFT, center zero frequency, then display log magnitude.',
    'Spectrum = log(1 + |fftshift(fft2(image))|)',
    'Student Task-FFT-DFT-IDFT.pdf'
)
def fft(image, params, reference):
    spectrum = np.fft.fftshift(np.fft.fft2(gray(image)))
    return (
        normalize(np.log1p(np.abs(spectrum))),
        {'note': 'FFT computes the DFT efficiently. Output shows normalized log magnitude, not the complex coefficients.'}
    )


@operation(
    'idft',
    'DFT → IDFT reconstruction',
    'Transforms & comparison',
    'Compute a forward DFT and its inverse to reconstruct the grayscale input.',
    'Reconstruction = real(ifft2(fft2(image)))',
    'Student Task-FFT-DFT-IDFT.pdf'
)
def idft(image, params, reference):
    values = gray(image)
    reconstructed = np.fft.ifft2(np.fft.fft2(values)).real
    error = float(np.mean((reconstructed - values.astype(float)) ** 2))
    return (np.clip(np.round(reconstructed), 0, 255).astype(np.uint8), {'reconstruction_mse': error})


@operation(
    'frequency_filter',
    'Frequency filtering',
    'Transforms & comparison',
    'Filter the centered DFT using a circular low-pass or high-pass mask.',
    'Output = IDFT(DFT(image) × frequency mask)',
    'Chap4-Image Transforms.pdf',
    [
        choice('mode', 'Filter type', 'low-pass', ['low-pass', 'high-pass']),
        field('radius', 'Cutoff (% of smaller dimension)', 10, 1, 50)
    ]
)
def frequency_filter(image, params, reference):
    values = gray(image)
    h, w = values.shape
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - w // 2) ** 2 + (yy - h // 2) ** 2 <= (min(h, w) * params['radius'] / 100) ** 2
    if params['mode'] == 'high-pass':
        mask = ~mask
    spectrum = np.fft.fftshift(np.fft.fft2(values))
    output = np.fft.ifft2(np.fft.ifftshift(spectrum * mask)).real
    return normalize(output) if params['mode'] == 'high-pass' else np.clip(np.round(output), 0, 255).astype(np.uint8)


@operation(
    'matching',
    'Pixel similarity',
    'Transforms & comparison',
    'Compare aligned grayscale pixels with a reference, resized to the source dimensions. This measures pixel agreement, not object recognition.',
    'Similarity = count(|source - reference| ≤ tolerance) / N × 100',
    'code_27_pixel_similarity.py',
    [field('tolerance', 'Intensity tolerance', 5, 0, 255)]
)
def matching(image, params, reference):
    if reference is None:
        raise ValueError('Upload a reference image for pixel similarity.')
    source = gray(image)
    target = cv2.resize(gray(reference), (source.shape[1], source.shape[0]))
    difference = np.abs(source.astype(np.int16) - target.astype(np.int16))
    similar = int(np.count_nonzero(difference <= params['tolerance']))
    return (difference.astype(np.uint8), {
        'similar_pixels': similar,
        'different_pixels': int(source.size - similar),
        'similarity_percent': round(100 * similar / source.size, 3),
        'note': 'Output is the absolute grayscale difference; black pixels are identical.'
    })


@operation(
    'compression',
    'JPEG compression',
    'Transforms & comparison',
    'Encode JPEG at the selected quality, then compare the decoded result. Download saves the actual JPEG.',
    'Lossy JPEG encoding with adjustable quality',
    'Chap7-Image Compression.pdf',
    [field('quality', 'JPEG quality', 80, 1, 100)]
)
def compression(image, params, reference):
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, params['quality']])
    if not ok:
        raise ValueError('JPEG compression failed.')
    result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    # Preserve the exact JPEG bytes; report statistics on its decoded pixels.
    import base64
    payload = 'data:image/jpeg;base64,' + base64.b64encode(encoded).decode('ascii')
    mse = float(np.mean((image.astype(float) - result.astype(float)) ** 2))
    return result, {
        'jpeg_bytes': int(encoded.size),
        'raw_pixel_bytes': int(image.nbytes),
        'raw_to_jpeg_ratio': round(image.nbytes / encoded.size, 2),
        'mse': round(mse, 5),
        'psnr_db': round(10 * math.log10(255 ** 2 / mse), 3) if mse else 'infinity',
        'download': payload
    }
