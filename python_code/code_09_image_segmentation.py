import cv2
import numpy as np
from scipy import ndimage
from skimage import feature, segmentation

from code_02_image_operations import operation, field, choice, gray, normalize
from code_04_image_helpers import bounded, label_image, reference_gray, u8

SOURCE='Chap5-Image Segmentation.pdf'
GROUP='Edges & segmentation'


@operation(
    'segmentation_metrics',
    'Dice / IoU / accuracy / precision / recall',
    GROUP,
    'Compare binary foreground masks from the input and reference image.',
    'Dice=2TP/(2TP+FP+FN); IoU=TP/(TP+FP+FN)',
    SOURCE,
    [field('threshold','Mask threshold',127,0,255)],
    requires_reference=True
)
def segmentation_metrics(image, params, reference):
    values = gray(image)
    target = reference_gray(reference, values.shape)
    source_mask = values > params['threshold']
    target_mask = target > params['threshold']
    tp = int(np.count_nonzero(source_mask & target_mask))
    fp = int(np.count_nonzero(source_mask & ~target_mask))
    fn = int(np.count_nonzero(~source_mask & target_mask))
    tn = int(np.count_nonzero(~source_mask & ~target_mask))

    def divide(numerator, denominator):
        return numerator / denominator if denominator else 1.0

    output = np.zeros((*source_mask.shape, 3), np.uint8)
    output[source_mask & target_mask] = (70, 190, 60)
    output[source_mask & ~target_mask] = (0, 0, 255)
    output[~source_mask & target_mask] = (255, 100, 0)
    return output, {
        'true_positive': tp, 'false_positive': fp, 'false_negative': fn, 'true_negative': tn,
        'dice': divide(2 * tp, 2 * tp + fp + fn), 'iou': divide(tp, tp + fp + fn),
        'pixel_accuracy': divide(tp + tn, source_mask.size),
        'precision': divide(tp, tp + fp), 'recall': divide(tp, tp + fn),
        'note': 'Green = overlap; red = input-only; blue = reference-only. Empty denominators use the perfect-empty-mask convention.',
    }


@operation(
    'isodata',
    'Iterative global threshold · ISODATA',
    GROUP,
    'Iteratively update a threshold using the mean intensity of the two classes.',
    'Tnext=(mean below T + mean above T)/2',
    SOURCE,
    [field('initial','Initial threshold',127,0,255)]
)
def isodata(image, params, reference):
    values = gray(image)
    cutoff = float(params['initial'])
    for iteration in range(100):
        low = values[values <= cutoff]
        high = values[values > cutoff]
        if not low.size or not high.size:
            break
        updated = (float(low.mean()) + float(high.mean())) / 2
        if abs(updated - cutoff) < 0.1:
            cutoff = updated
            break
        cutoff = updated
    return ((values > cutoff).astype(np.uint8) * 255, {'selected_threshold': cutoff, 'iterations': iteration + 1})


@operation(
    'adaptive_mean',
    'Adaptive mean threshold',
    GROUP,
    'Use the local mean to determine a threshold for each pixel.',
    'T(x,y)=local mean-C',
    SOURCE,
    [field('kernel','Block size (odd)',11,3,51),field('constant','Subtracted constant',2,-30,30)]
)
def adaptive_mean(image, params, reference):
    return cv2.adaptiveThreshold(
        gray(image),
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        params['kernel'],
        params['constant']
    )


@operation('lbp_lpg','Local binary patterns / gradient patterns',GROUP,'Calculate circular neighbor comparisons on grayscale intensity or Sobel gradient magnitude.', 'LBP=Σ 2ᵖ [neighbor ≥ center]; LPG applies comparisons to gradient magnitude',SOURCE,
           [
               choice('mode','Pattern','LBP',['LBP','LPG']),
               field('radius','Radius',1,1,5),
               choice('method','Encoding','uniform',['uniform','default'])
           ])
def lbp_lpg(image, params, reference):
    values = gray(image)
    if params['mode'] == 'LPG':
        horizontal = cv2.Sobel(values, cv2.CV_32F, 1, 0)
        vertical = cv2.Sobel(values, cv2.CV_32F, 0, 1)
        values = normalize(cv2.magnitude(horizontal, vertical))
    pattern = feature.local_binary_pattern(values, 8, params['radius'], params['method'])
    counts = np.bincount(pattern.astype(int).ravel(), minlength=10 if params['method'] == 'uniform' else 256)
    return (normalize(pattern), {
        'texture_histogram': counts.tolist(),
        'note': 'The pattern map is normalized for display. This computes texture features; it does not train an object classifier.'
    })


@operation('point_line','Point / directional line detection',GROUP,'Use second-derivative masks for isolated points or horizontal, vertical and diagonal lines.', 'response = correlation(image, selected detection mask)',SOURCE,
           [
               choice(
                   'direction',
                   'Detector',
                   'horizontal',
                   ['point','horizontal','vertical','45 degrees','135 degrees']
               ),
               field('threshold','Response threshold',30,0,255)
           ])
def point_line(image, params, reference):
    kernels = {
        'point': [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
        'horizontal': [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],
        'vertical': [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]],
        '45 degrees': [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]],
        '135 degrees': [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]
    }
    response = np.abs(cv2.filter2D(gray(image), cv2.CV_32F, np.array(kernels[params['direction']], np.float32)))
    return (response > params['threshold']).astype(np.uint8) * 255


@operation('hough_lines','Hough lines / lane detection pipeline',GROUP,'Find edges and line segments using the probabilistic Hough transform. An optional lower-half ROI illustrates lane detection.', 'ρ=x cos θ+y sin θ',SOURCE,
           [
               choice('mode','Region','whole image',['whole image','lane ROI']),
               field('votes','Hough vote threshold',30,5,150),
               field('length','Minimum line length',30,5,300),
               field('gap','Maximum gap',10,0,100)
           ])
def hough_lines(image, params, reference):
    output = bounded(image, 768)
    edges = cv2.Canny(gray(output), 80, 160)
    if params['mode'] == 'lane ROI':
        h, w = edges.shape
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [np.array([[0, h - 1], [w // 2, h // 2], [w - 1, h - 1]], np.int32)], 255)
        edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        params['votes'],
        minLineLength=params['length'],
        maxLineGap=params['gap']
    )
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 80), 2)
    return (output, {
        'line_segments': 0 if lines is None else len(lines),
        'note': 'Lane ROI is a teaching pipeline; arbitrary image lines are not certified road-lane predictions.'
    })


@operation('log_dog','LoG / DoG / gradient threshold',GROUP,'Use Gaussian smoothing with Laplacian, a difference of Gaussian scales or thresholded gradient magnitude.', 'LoG=∇²(Gσ*f); DoG=Gσ1*f-Gσ2*f',SOURCE,
           [
               choice('mode','Detector','LoG',['LoG','DoG','gradient threshold']),
               choice('display','Response','magnitude',['magnitude','zero crossings']),
               field('sigma','Gaussian sigma',1.5,.5,5,.1),
               field('threshold','Threshold',20,0,255)
           ])
def log_dog(image, params, reference):
    values = gray(image).astype(np.float32)
    smooth = cv2.GaussianBlur(values, (0, 0), params['sigma'])
    if params['mode'] == 'LoG':
        response = cv2.Laplacian(smooth, cv2.CV_32F)
    elif params['mode'] == 'DoG':
        response = smooth - cv2.GaussianBlur(values, (0, 0), params['sigma'] * 1.6)
    else:
        response = cv2.magnitude(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
        return (response > params['threshold']).astype(np.uint8) * 255
    if params['display'] == 'zero crossings':
        low = ndimage.minimum_filter(response, size=3)
        high = ndimage.maximum_filter(response, size=3)
        return ((low < 0) & (high > 0) & (high - low > params['threshold'])).astype(np.uint8) * 255
    return normalize(np.abs(response))


@operation(
    'gradient_direction',
    'Gradient magnitude / orientation / linking',
    GROUP,
    'Inspect edge strength and orientation, or link strong edges to weak neighbors using hysteresis.',
    'θ=atan2(Gy,Gx); magnitude=√(Gx²+Gy²)',
    SOURCE,
    [
        choice('mode','Output','orientation',['magnitude','orientation','hysteresis']),
        field('low','Lower gradient threshold',40,0,255),
        field('high','Upper gradient threshold',100,0,255)
    ]
)
def gradient_direction(image, params, reference):
    from skimage.filters import apply_hysteresis_threshold
    values = gray(image)
    x = cv2.Sobel(values, cv2.CV_32F, 1, 0)
    y = cv2.Sobel(values, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(x, y)
    if params['mode'] == 'magnitude':
        return normalize(magnitude)
    if params['mode'] == 'hysteresis':
        if params['low'] >= params['high']:
            raise ValueError('Lower threshold must be below upper threshold.')
        return apply_hysteresis_threshold(magnitude, params['low'], params['high']).astype(np.uint8) * 255
    return u8((np.arctan2(y, x) + np.pi) * 255 / (2 * np.pi))


@operation('region_growing','Seeded region growing',GROUP,'Grow the connected region whose intensity is within the tolerance of the seed pixel.', '|I(pixel)-I(seed)| ≤ tolerance, with connected flood traversal',SOURCE,
           [
               field('x','Seed x (%)',50,0,100),
               field('y','Seed y (%)',50,0,100),
               field('tolerance','Intensity tolerance',20,0,255),
               choice('connectivity','Connectivity','4',['4','8'])
           ])
def region_growing(image, params, reference):
    values = gray(bounded(image))
    h, w = values.shape
    seed = (round((h - 1) * params['y'] / 100), round((w - 1) * params['x'] / 100))
    region = segmentation.flood(
        values,
        seed,
        tolerance=params['tolerance'],
        connectivity=1 if params['connectivity'] == '4' else 2
    )
    return (region.astype(np.uint8) * 255, {
        'seed_intensity': int(values[seed]),
        'region_pixels': int(region.sum()),
        'note': 'Region growing uses a maximum side of 512 pixels and a fixed seed-intensity criterion.'
    })


@operation(
    'kmeans_segmentation',
    'Feature-based K-means segmentation',
    GROUP,
    'Cluster pixels by color, optionally including spatial position.',
    'min Σ ||feature-centroid||²',
    SOURCE,
    [
        field('clusters','Clusters',4,2,10),
        field('spatial','Spatial feature weight',0,0,100),
        field('iterations','Iterations',20,1,50)
    ]
)
def kmeans_segmentation(image, params, reference):
    source = bounded(image, 256)
    h, w = source.shape[:2]
    features = source.reshape(-1, 3).astype(float)
    if params['spatial']:
        y, x = np.mgrid[:h, :w]
        features = np.column_stack((
            features,
            x.ravel() / max(w - 1, 1) * params['spatial'],
            y.ravel() / max(h - 1, 1) * params['spatial']
        ))
    rng = np.random.default_rng(42)
    centers = features[rng.choice(len(features), params['clusters'], replace=len(features) < params['clusters'])]
    for iteration in range(params['iterations']):
        distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        updated = centers.copy()
        for cluster in range(params['clusters']):
            members = features[labels == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
        if np.allclose(updated, centers):
            break
        centers = updated
    return (u8(centers[labels, :3].reshape(h, w, 3)), {
        'clusters_used': int(len(np.unique(labels))),
        'iterations': iteration + 1,
        'note': 'Clustering is limited to a 256-pixel longest side; it assigns color/position groups, not semantic classes.'
    })


@operation(
    'fuzzy_cmeans',
    'Fuzzy C-means segmentation',
    GROUP,
    'Learn soft grayscale memberships using intensity histogram weights, then display the strongest-membership cluster.',
    'uik ∝ |xi-ck|^(-2/(m-1)); ck=Σ uik^m xi / Σ uik^m',
    SOURCE,
    [
        field('clusters','Clusters',3,2,8),
        field('fuzziness','Fuzziness m',2,1.1,4,.1),
        field('iterations','Iterations',30,1,100)
    ]
)
def fuzzy_cmeans(image, params, reference):
    values = gray(image)
    hist = np.bincount(values.ravel(), minlength=256).astype(float)
    levels = np.arange(256.0)
    centers = np.linspace(values.min(), values.max(), params['clusters'])
    for iteration in range(params['iterations']):
        distance = np.maximum(np.abs(levels[:, None] - centers), 1e-06)
        weights = distance ** (-2 / (params['fuzziness'] - 1))
        memberships = weights / weights.sum(axis=1, keepdims=True)
        powered = memberships ** params['fuzziness'] * hist[:, None]
        updated = (powered * levels[:, None]).sum(axis=0) / np.maximum(powered.sum(axis=0), 1e-12)
        if np.allclose(updated, centers, atol=0.01):
            centers = updated
            break
        centers = updated
    labels = np.abs(levels[:, None] - centers).argmin(axis=1)
    return (u8(centers[labels[values]]), {
        'cluster_centers': centers.tolist(),
        'iterations': iteration + 1,
        'mean_max_membership': float(np.average(memberships.max(axis=1), weights=hist))
    })


@operation(
    'watershed',
    'Marker-controlled watershed',
    GROUP,
    'Segment thresholded foreground using distance-transform peaks as watershed seeds.',
    'Flood negative distance from local-maxima markers',
    SOURCE,
    [field('threshold','Foreground threshold',127,0,255),field('distance','Minimum seed spacing',8,1,50)]
)
def watershed(image, params, reference):
    source = bounded(image)
    binary = gray(source) > params['threshold']
    distance = ndimage.distance_transform_edt(binary)
    coordinates = feature.peak_local_max(
        distance,
        min_distance=params['distance'],
        labels=binary,
        exclude_border=False
    )
    seeds = np.zeros(binary.shape, bool)
    if coordinates.size:
        seeds[tuple(coordinates.T)] = True
    markers, _ = ndimage.label(seeds)
    labels = segmentation.watershed(-distance, markers, mask=binary)
    return (label_image(labels), {
        'segments': int(labels.max()),
        'note': 'This foreground-distance watershed is limited to a 512-pixel side; threshold polarity matters.'
    })


@operation(
    'split_merge',
    'Quadtree splitting / adjacent region merging',
    GROUP,
    'Split regions by their intensity range, then optionally merge adjacent regions by their mean difference.',
    'Split if max-min > T; merge if |meanA-meanB| ≤ Tmerge',
    SOURCE,
    [
        field('threshold','Split range threshold',40,0,255),
        field('minimum','Minimum tile size',8,2,32),
        choice('mode','Output','split and merge',['split only','split and merge']),
        field('merge','Merge mean tolerance',15,0,100)
    ]
)
def split_merge(image, params, reference):
    values = gray(bounded(image, 256))
    h, w = values.shape
    labels = np.zeros((h, w), np.int32)
    count = 0

    def split(y0, y1, x0, x1):
        nonlocal count
        region = values[y0:y1, x0:x1]
        if int(region.max()) - int(region.min()) <= params['threshold'] or min(region.shape) <= params['minimum']:
            count += 1
            labels[y0:y1, x0:x1] = count
            return
        ym = (y0 + y1) // 2
        xm = (x0 + x1) // 2
        for a, b, c, d in [(y0, ym, x0, xm), (y0, ym, xm, x1), (ym, y1, x0, xm), (ym, y1, xm, x1)]:
            split(a, b, c, d)
    split(0, h, 0, w)
    initial = count
    if params['mode'] == 'split and merge':
        parent = np.arange(count + 1)
        sums = np.bincount(labels.ravel(), weights=values.ravel(), minlength=count + 1)
        sizes = np.bincount(labels.ravel(), minlength=count + 1)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        horizontal_pairs = np.column_stack((labels[:, :-1].ravel(), labels[:, 1:].ravel()))
        vertical_pairs = np.column_stack((labels[:-1, :].ravel(), labels[1:, :].ravel()))
        pairs = np.unique(np.vstack((horizontal_pairs, vertical_pairs)), axis=0)
        for _ in range(5):
            changes = 0
            for a, b in pairs:
                a, b = (find(a), find(b))
                if a != b and abs(sums[a] / sizes[a] - sums[b] / sizes[b]) <= params['merge']:
                    parent[b] = a
                    sums[a] += sums[b]
                    sizes[a] += sizes[b]
                    changes += 1
            if not changes:
                break
        lookup = np.array([find(i) for i in range(count + 1)])
        labels = lookup[labels]
    return (label_image(labels), {
        'split_regions': initial,
        'output_regions': int(len(np.unique(labels))),
        'note': 'A bounded 256-pixel quadtree demo with at most five adjacent-merge passes.'
    })


@operation('border_tracing','Graph-search boundaries / contour approximation',GROUP,'Extract foreground contours, trace their boundary order, and optionally approximate with polygons.', 'Boundary traversal → Douglas–Peucker polygon approximation',SOURCE+' / Chap8-Morphological Processing.pdf',
           [
               field('threshold','Foreground threshold',127,0,255),
               field('epsilon','Polygon tolerance (% perimeter)',1,0,10,.1),
               choice('mode','Output','contours',['contours','polygons'])
           ])
def border_tracing(image, params, reference):
    output = bounded(image)
    binary = (gray(output) > params['threshold']).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if params['mode'] == 'polygons':
        contours = [cv2.approxPolyDP(c, params['epsilon'] / 100 * cv2.arcLength(c, True), True) for c in contours]
    cv2.drawContours(output, contours, -1, (0, 220, 30), 1)
    return (output, {
        'contours': len(contours),
        'boundary_points': sum((len(c) for c in contours)),
        'note': 'Contours preserve boundary traversal order; epsilon controls polygon simplification.'
    })


@operation(
    'active_contour',
    'Active contour · snake',
    GROUP,
    'Evolve an elliptical contour toward intensity edges using internal smoothness constraints.',
    'Minimize internal contour energy + image edge energy',
    SOURCE,
    [
        field('iterations','Maximum iterations',60,5,150),
        field('alpha','Continuity weight',.01,.001,1,.001),
        field('beta','Smoothness weight',1,.1,10,.1)
    ]
)
def active_contour(image, params, reference):
    output = bounded(image, 256)
    h, w = output.shape[:2]
    if min(h, w) < 8:
        raise ValueError('Use an image at least 8 pixels on each side for a snake.')
    t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    initial = np.column_stack((h / 2 + 0.35 * h * np.sin(t), w / 2 + 0.35 * w * np.cos(t)))
    values = cv2.GaussianBlur(gray(output), (5, 5), 1).astype(float) / 255
    contour = segmentation.active_contour(
        values,
        initial,
        alpha=params['alpha'],
        beta=params['beta'],
        gamma=0.01,
        max_num_iter=params['iterations']
    )
    points = np.column_stack((contour[:, 1], contour[:, 0])).astype(np.int32)
    cv2.polylines(output, [points], True, (0, 255, 40), 2)
    return (output, {
        'contour_points': len(contour),
        'note': 'Snake initialization is a central ellipse on a 256-pixel preview; results depend on object position and parameters.'
    })


@operation(
    'level_set',
    'Level-set segmentation · Chan–Vese',
    GROUP,
    'Evolve a region boundary to separate two intensity distributions without requiring strong edges.',
    'Minimize region fitting error + boundary regularity',
    SOURCE,
    [field('iterations','Iterations',40,1,100),field('smoothing','Morphological smoothing',2,1,5)]
)
def level_set(image, params, reference):
    values = gray(bounded(image, 256)).astype(float) / 255
    result = segmentation.morphological_chan_vese(
        values,
        params['iterations'],
        init_level_set='checkerboard',
        smoothing=params['smoothing']
    )
    return (
        result.astype(np.uint8) * 255,
        {'note': 'Morphological Chan–Vese is a level-set implementation on a maximum 256-pixel preview.'}
    )
