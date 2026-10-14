import math
from collections import deque

import cv2
import numpy as np
from scipy import ndimage
from skimage.filters.rank import entropy as local_entropy
from skimage.morphology import disk

from code_02_image_operations import operation, field, choice, gray, normalize
from code_04_image_helpers import bounded, label_image, reference_gray, u8

SOURCE = 'Chap1 Digital Image Fundamentals.pdf'
GROUP = 'Fundamentals'


@operation(
    'entropy',
    'Entropy & image information',
    GROUP,
    'Calculate grayscale Shannon entropy, variance, energy and the intensity probability distribution.',
    'H = -Σ p(r) log₂ p(r)',
    SOURCE
)
def entropy(image, params, reference):
    values = gray(image)
    probabilities = np.bincount(values.ravel(), minlength=256) / values.size
    nonzero = probabilities[probabilities > 0]
    return (image.copy(), {
        'entropy_bits_per_pixel': float(-np.sum(nonzero * np.log2(nonzero))),
        'variance': float(values.astype(float).var()),
        'intensity_energy': float(np.sum(probabilities ** 2)),
        'note': 'Entropy measures intensity uncertainty; it is not a measure of semantic content.'
    })


@operation(
    'local_entropy',
    'Local entropy map',
    GROUP,
    'Map the information in a disk-shaped neighborhood to reveal texture.',
    'H(x,y) = -Σ local p(r) log₂ local p(r)',
    SOURCE,
    [field('radius', 'Neighborhood radius', 3, 1, 15)]
)
def entropy_map(image, params, reference):
    result = local_entropy(gray(image), disk(params['radius']))
    return (normalize(result), {
        'maximum_local_entropy_bits': float(result.max()),
        'note': 'Display is scaled to 0–255; entropy measurements are in bits.'
    })


@operation(
    'quality',
    'Image quality · MSE / PSNR / SSIM',
    GROUP,
    'Compare the image with a reference using error and structural similarity metrics. The reference is resized to align dimensions.',
    'MSE = mean((I-R)²); PSNR = 10 log₁₀(255²/MSE)',
    SOURCE,
    requires_reference=True
)
def quality(image, params, reference):
    from skimage.metrics import structural_similarity
    values = gray(image)
    target = reference_gray(reference, values.shape)
    difference = values.astype(float) - target.astype(float)
    mse = float(np.mean(difference ** 2))
    side = min(values.shape)
    ssim = float(structural_similarity(
        values,
        target,
        data_range=255,
        win_size=min(7, side if side % 2 else side - 1)
    )) if side >= 3 else None
    return (np.abs(difference).astype(np.uint8), {
        'mse': mse,
        'psnr_db': float(10 * math.log10(255 ** 2 / mse)) if mse else 'infinity',
        'ssim': ssim,
        'note': 'Output is the absolute grayscale difference. SSIM needs dimensions of at least 3 pixels.'
    })


@operation('connectivity', 'Pixel connectivity & components', GROUP, 'Label connected foreground regions using 4-, 8- or mixed adjacency. Mixed adjacency excludes diagonal links when a shared orthogonal neighbor is foreground.', 'N4, N8 and m-adjacency on a binary pixel set', SOURCE,
           [
               choice('adjacency', 'Adjacency', '8', ['4', '8', 'mixed']),
               field('threshold', 'Foreground threshold', 127, 0, 255)
           ])
def connectivity(image, params, reference):
    values = gray(bounded(image)) > params['threshold']
    if params['adjacency'] != 'mixed':
        labels, count = ndimage.label(
            values,
            ndimage.generate_binary_structure(2, 1 if params['adjacency'] == '4' else 2)
        )
    else:
        h, w = values.shape
        labels = np.zeros((h, w), np.int32)
        count = 0
        for y, x in zip(*np.nonzero(values)):
            if labels[y, x]:
                continue
            count += 1
            labels[y, x] = count
            queue = deque([(y, x)])
            while queue:
                yy, xx = queue.popleft()
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ny, nx = (yy + dy, xx + dx)
                    if not (0 <= ny < h and 0 <= nx < w) or not values[ny, nx] or labels[ny, nx]:
                        continue
                    if dy and dx and (values[yy, nx] or values[ny, xx]):
                        continue
                    labels[ny, nx] = count
                    queue.append((ny, nx))
    return (label_image(labels), {
        'connected_components': int(count),
        'adjacency': params['adjacency'],
        'foreground_pixels': int(values.sum()),
        'note': 'Color labels distinguish components; analysis is limited to a 512-pixel longest side.'
    })


@operation('pixel_distances', 'Pixel distances & shortest path', GROUP, 'Mark two pixels and measure Euclidean, city-block and chessboard distances. Display a shortest 4-connected path.', 'De = √(dx²+dy²); D4 = |dx|+|dy|; D8 = max(|dx|,|dy|)', SOURCE,
           [
               field('x1','Start x (%)',20,0,100),
               field('y1','Start y (%)',20,0,100),
               field('x2','End x (%)',80,0,100),
               field('y2','End y (%)',80,0,100)
           ])
def pixel_distances(image, params, reference):
    output = image.copy()
    h, w = image.shape[:2]
    a = (round((w - 1) * params['x1'] / 100), round((h - 1) * params['y1'] / 100))
    b = (round((w - 1) * params['x2'] / 100), round((h - 1) * params['y2'] / 100))
    dx, dy = (abs(a[0] - b[0]), abs(a[1] - b[1]))
    cv2.line(output, a, (b[0], a[1]), (0, 220, 255), 2)
    cv2.line(output, (b[0], a[1]), b, (0, 220, 255), 2)
    for point in [a, b]:
        cv2.circle(output, point, 5, (0, 0, 255), -1)
    return (output, {
        'euclidean_distance': float(math.hypot(dx, dy)),
        'city_block_distance': dx + dy,
        'chessboard_distance': max(dx, dy),
        'note': 'The illustrated path has no obstacles; foreground-constrained paths are a separate graph-search problem.'
    })


@operation(
    'lloyd_quantization',
    'Lloyd–Max / K-means quantization',
    GROUP,
    'Optimize grayscale levels by repeatedly assigning intensities to the nearest center and updating each center by its histogram-weighted mean.',
    'centroid = Σ intensity × frequency / Σ frequency',
    SOURCE,
    [field('levels', 'Quantization levels', 8, 2, 32),field('iterations','Maximum iterations',30,1,100)]
)
def lloyd_quantization(image, params, reference):
    values = gray(image)
    hist = np.bincount(values.ravel(), minlength=256).astype(float)
    centers = np.linspace(int(values.min()), int(values.max()), params['levels'])
    levels = np.arange(256)
    for iteration in range(params['iterations']):
        labels = np.abs(levels[:, None] - centers).argmin(axis=1)
        updated = centers.copy()
        for k in range(params['levels']):
            mask = labels == k
            if hist[mask].sum():
                updated[k] = np.average(levels[mask], weights=hist[mask])
        if np.allclose(updated, centers):
            break
        centers = updated
    labels = np.abs(levels[:, None] - centers).argmin(axis=1)
    output = u8(centers[labels[values]])
    return (output, {
        'centers': centers.round(3).tolist(),
        'iterations': iteration + 1,
        'mse': float(np.mean((values.astype(float) - output) ** 2))
    })


@operation('image_arithmetic', 'Image arithmetic & logic', GROUP, 'Combine two aligned images with addition, subtraction, averaging, multiplication, division or binary logic.', 'g = f op reference; arithmetic uses saturation', SOURCE,
           [choice(
               'method',
               'Operation',
               'average',
               ['add','subtract','average','multiply','divide','AND','OR','XOR']
           )], requires_reference=True)
def image_arithmetic(image, params, reference):
    if reference is None:
        raise ValueError('Upload a reference image.')
    target = cv2.resize(reference, (image.shape[1], image.shape[0]))
    a, b = (image.astype(float), target.astype(float))
    method = params['method']
    if method == 'add':
        result = u8(a + b)
    elif method == 'subtract':
        result = u8(a - b)
    elif method == 'average':
        result = u8((a + b) / 2)
    elif method == 'multiply':
        result = u8(a * b / 255)
    elif method == 'divide':
        result = u8(a / np.maximum(b, 1) * 127)
    else:
        result = {'AND': cv2.bitwise_and, 'OR': cv2.bitwise_or, 'XOR': cv2.bitwise_xor}[method](image, target)
    return result


@operation(
    'image_formation',
    'Illumination & reflectance model',
    GROUP,
    'Simulate a spatial illumination field multiplied by image reflectance.',
    'f(x,y) = i(x,y) × r(x,y)',
    SOURCE,
    [
        field('illumination','Illumination strength',.7,.1,2,.1),
        choice('pattern','Illumination field','radial',['radial','horizontal','uniform'])
    ]
)
def image_formation(image, params, reference):
    h, w = image.shape[:2]
    y, x = np.mgrid[:h, :w]
    if params['pattern'] == 'radial':
        light = 0.25 + 0.75 * np.exp(-(((x - w / 2) / max(w / 2, 1)) ** 2 + ((y - h / 2) / max(h / 2, 1)) ** 2))
    elif params['pattern'] == 'horizontal':
        light = np.broadcast_to(np.linspace(0.25, 1, w), (h, w))
    else:
        light = np.ones((h, w))
    return (
        u8(image.astype(float) * light[:, :, None] * params['illumination']),
        {'note': 'A teaching simulation: the uploaded image is treated as reflectance, not calibrated physical sensor data.'}
    )


@operation(
    'indexed_palette',
    'Indexed palette / color quantization',
    GROUP,
    'Quantize an RGB image into a palette and reconstruct from its index array.',
    'RGB image → palette indices → palette lookup',
    SOURCE,
    [field('colors','Palette colors',16,2,256)]
)
def indexed_palette(image, params, reference):
    from PIL import Image
    rgb = cv2.cvtColor(bounded(image, 512), cv2.COLOR_BGR2RGB)
    indexed = Image.fromarray(rgb).quantize(colors=params['colors'], method=Image.Quantize.MEDIANCUT)
    palette = np.array(indexed.getpalette(), np.uint8).reshape(-1, 3)
    indices = np.asarray(indexed)
    result = cv2.cvtColor(palette[indices], cv2.COLOR_RGB2BGR)
    return (result, {
        'used_palette_colors': int(len(np.unique(indices))),
        'index_bits_required': int(np.ceil(np.log2(params['colors']))),
        'palette_rgb': palette[:params['colors']].tolist(),
        'note': 'Palette indices are categorical color labels, not grayscale intensities.'
    })


@operation(
    'feature_vectors',
    'Row / column projection feature vectors',
    GROUP,
    'Threshold a binary shape and sum foreground pixels along rows and columns.',
    'row[y]=Σx binary[y,x]; column[x]=Σy binary[y,x]',
    SOURCE,
    [field('threshold','Binary threshold',127,0,255),choice('polarity','Foreground','bright',['bright','dark'])]
)
def feature_vectors(image, params, reference):
    values = gray(bounded(image, 128))
    binary = values > params['threshold'] if params['polarity'] == 'bright' else values <= params['threshold']
    return (binary.astype(np.uint8) * 255, {
        'row_projection': binary.sum(axis=1).tolist(),
        'column_projection': binary.sum(axis=0).tolist(),
        'foreground_area': int(binary.sum())
    })


DIGITS={
    '0':['01110','10001','10011','10101','11001','10001','01110'],
    '1':['00100','01100','00100','00100','00100','00100','01110'],
    '2':['01110','10001','00001','00010','00100','01000','11111'],
    '3':['11110','00001','00001','01110','00001','00001','11110'],
    '4':['00010','00110','01010','10010','11111','00010','00010'],
    '5':['11111','10000','10000','11110','00001','00001','11110'],
    '6':['01110','10000','10000','11110','10001','10001','01110'],
    '7':['11111','00001','00010','00100','01000','01000','01000'],
    '8':['01110','10001','10001','01110','10001','10001','01110'],
    '9':['01110','10001','10001','01111','00001','00001','01110']}


@operation('digit_rules','Rule-based toy digit classification',GROUP,'Compare row/column feature vectors with ten fixed 7×5 glyph templates. This is a classroom classifier, not OCR.', 'Decision rule: choose minimum squared projection-feature distance',SOURCE,
           [
               choice('sample','Input glyph','7',['uploaded image','0','1','2','3','4','5','6','7','8','9']),
               field('threshold','Uploaded foreground threshold',127,0,255)
           ])
def digit_rules(image, params, reference):
    templates = {
        key: np.array([[int(value) for value in row] for row in rows], np.uint8) for key,
        rows in DIGITS.items()
    }
    if params['sample'] == 'uploaded image':
        glyph = (cv2.resize(gray(image), (5, 7), interpolation=cv2.INTER_AREA) > params['threshold']).astype(np.uint8)
    else:
        glyph = templates[params['sample']]
    vector = np.r_[glyph.sum(axis=1), glyph.sum(axis=0)].astype(float)
    scores = {
        key: float(np.sum((vector - np.r_[value.sum(axis=1), value.sum(axis=0)]) ** 2)) for key,
        value in templates.items()
    }
    return (glyph * 255, {
        'predicted_template_digit': min(scores, key=scores.get),
        'rule_distances': scores,
        'pixel_matrix': (glyph * 255).tolist(),
        'note': 'Fixed-template projection rules only; uploaded glyphs must be isolated and have bright foreground. This does not run Tesseract or a learned OCR model.'
    })


@operation('gaussian_classifier','Parametric Gaussian pixel classifier',GROUP,'Classify grayscale pixels by comparing two Gaussian likelihoods with priors.', 'posterior(class) ∝ prior × Gaussian(intensity; mean, sigma)',SOURCE,
           [
               field('mean0','Background mean',50,0,255),
               field('mean1','Foreground mean',190,0,255),
               field('sigma0','Background sigma',25,1,100),
               field('sigma1','Foreground sigma',25,1,100),
               field('prior','Foreground prior',.5,.01,.99,.01)
           ])
def gaussian_classifier(image, params, reference):
    values = gray(image).astype(float)
    log0 = np.log(1 - params['prior']) - np.log(params['sigma0']) - 0.5 * ((values - params['mean0']) / params['sigma0']) ** 2
    log1 = np.log(params['prior']) - np.log(params['sigma1']) - 0.5 * ((values - params['mean1']) / params['sigma1']) ** 2
    return ((log1 > log0).astype(np.uint8) * 255, {
        'foreground_pixels': int(np.count_nonzero(log1 > log0)),
        'note': 'Classes are intensity distributions entered by the user, not trained semantic labels.'
    })


@operation('pupil_demo','Pupil centroid / gaze geometry demo',GROUP,'Detect a dark approximately circular candidate in a cropped eye image and mark its centroid. This is not a calibrated gaze estimator.', 'centroid=(M10/M00,M01/M00); circularity=4π area/perimeter²',SOURCE,
           [field('threshold','Dark-region threshold',60,0,255),field('minimum','Minimum area',10,1,10000)])
def pupil_demo(image, params, reference):
    source = bounded(image, 512)
    binary = (gray(source) < params['threshold']).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / perimeter ** 2 if perimeter else 0
        if params['minimum'] <= area < source.shape[0] * source.shape[1] * 0.5 and circularity > 0.3:
            moment = cv2.moments(contour)
            if moment['m00']:
                candidates.append((area * circularity, contour, moment))
    details = {
        'candidate_found': bool(candidates),
        'note': 'Use a cropped eye image. A dark circular blob is only a geometric candidate; pupil identity and real gaze direction are not guaranteed.'
    }
    if candidates:
        _, contour, moment = max(candidates, key=lambda item: item[0])
        x = moment['m10'] / moment['m00']
        y = moment['m01'] / moment['m00']
        cv2.drawContours(source, [contour], -1, (0, 255, 0), 1)
        cv2.drawMarker(source, (round(x), round(y)), (0, 0, 255))
        details.update(
            centroid_x=x,
            centroid_y=y,
            normalized_eye_offset=[
                (x - source.shape[1] / 2) / max(source.shape[1] / 2, 1),
                (y - source.shape[0] / 2) / max(source.shape[0] / 2, 1)
            ]
        )
    return (source, details)


@operation('shortest_path','Shortest foreground path · 4 / 8 / mixed',GROUP,'Use breadth-first search to find the shortest path between two pixels constrained to thresholded foreground.', 'BFS minimizes the number of unit-cost adjacency steps',SOURCE,
           [
               choice('adjacency','Adjacency','8',['4','8','mixed']),
               field('threshold','Foreground threshold',127,0,255),
               field('x1','Start x (%)',20,0,100),
               field('y1','Start y (%)',20,0,100),
               field('x2','End x (%)',80,0,100),
               field('y2','End y (%)',80,0,100),
               field('minkowski','Minkowski exponent',3,1,10)
           ])
def shortest_path(image, params, reference):
    source = bounded(image, 256)
    binary = gray(source) > params['threshold']
    h, w = binary.shape
    start = (round((h - 1) * params['y1'] / 100), round((w - 1) * params['x1'] / 100))
    end = (round((h - 1) * params['y2'] / 100), round((w - 1) * params['x2'] / 100))
    previous = {start: None}
    queue = deque([start]) if binary[start] and binary[end] else deque()
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)] + ([] if params['adjacency'] == '4' else [(-1, -1), (-1, 1), (1, -1), (1, 1)])
    found = False
    while queue:
        y, x = queue.popleft()
        if (y, x) == end:
            found = True
            break
        for dy, dx in neighbors:
            ny, nx = (y + dy, x + dx)
            if not (0 <= ny < h and 0 <= nx < w) or not binary[ny, nx] or (ny, nx) in previous:
                continue
            if params['adjacency'] == 'mixed' and dy and dx and (binary[y, nx] or binary[ny, x]):
                continue
            previous[ny, nx] = (y, x)
            queue.append((ny, nx))
    path = []
    if found:
        point = end
        while point is not None:
            path.append(point)
            point = previous[point]
        for y, x in path:
            source[y, x] = (0, 0, 255)
    dy, dx = (abs(start[0] - end[0]), abs(start[1] - end[1]))
    m = params['minkowski']
    return (source, {
        'path_found': found,
        'path_steps': len(path) - 1 if found else None,
        'minkowski_distance': float((dx ** m + dy ** m) ** (1 / m)),
        'note': 'BFS treats diagonal steps as unit cost. Endpoints must be foreground; coordinates refer to the 256-pixel preview.'
    })


@operation(
    'fc_demo',
    'Neural fully connected layer arithmetic demo',
    GROUP,
    'Extract a small fixed intensity vector and apply fixed matrix weights, ReLU and softmax to demonstrate a neural layer.',
    'logits=W x+b; ReLU=max(0,x); softmax=exp(logits)/Σ exp(logits)',
    SOURCE,
    [field('seed','Fixed weight seed',42,0,9999)]
)
def fc_demo(image, params, reference):
    sample = cv2.resize(gray(image), (4, 4), interpolation=cv2.INTER_AREA).astype(float) / 255
    rng = np.random.default_rng(params['seed'])
    weights = rng.normal(0, 0.3, (3, 16))
    bias = np.array([0.1, 0, -0.1])
    hidden = np.maximum(sample.ravel(), 0)
    logits = weights @ hidden + bias
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    return (u8(sample * 255), {
        'input_features': sample.ravel().round(4).tolist(),
        'fixed_weights': weights.round(4).tolist(),
        'logits': logits.tolist(),
        'softmax_probabilities': probabilities.tolist(),
        'note': 'Random fixed weights are untrained. These output indices have no semantic class labels and do not implement trained GazeNet/CNN/DeepFace recognition.'
    })
