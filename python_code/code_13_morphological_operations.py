import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology

from code_02_image_operations import operation,field,choice,gray,normalize
from code_04_image_helpers import bounded,reference_gray

SOURCE='Chap8-Morphological Processing.pdf'
GROUP='Morphology'


@operation('morphology_full','Binary / grayscale morphology & top-hats',GROUP,'Choose erosion, dilation, opening, closing, gradient, white top-hat or black top-hat with a selected structuring element.', 'White top-hat=f-opening(f); black top-hat=closing(f)-f',SOURCE,
           [
               choice('domain','Domain','binary',['binary','grayscale']),
               choice(
                   'method',
                   'Method',
                   'white top-hat',
                   ['erosion','dilation','opening','closing','gradient','white top-hat','black top-hat']
               ),
               choice('shape','Structuring element','ellipse',['rectangle','ellipse','cross']),
               field('kernel','Kernel size (odd)',5,3,31),
               field('iterations','Iterations',1,1,5),
               field('threshold','Binary threshold',127,0,255)
           ])
def morphology_full(image, params, reference):
    values = gray(image)
    if params['domain'] == 'binary':
        values = (values > params['threshold']).astype(np.uint8) * 255
    shapes = {'rectangle': cv2.MORPH_RECT, 'ellipse': cv2.MORPH_ELLIPSE, 'cross': cv2.MORPH_CROSS}
    element = cv2.getStructuringElement(shapes[params['shape']], (params['kernel'], params['kernel']))
    methods = {
        'erosion': cv2.MORPH_ERODE,
        'dilation': cv2.MORPH_DILATE,
        'opening': cv2.MORPH_OPEN,
        'closing': cv2.MORPH_CLOSE,
        'gradient': cv2.MORPH_GRADIENT,
        'white top-hat': cv2.MORPH_TOPHAT,
        'black top-hat': cv2.MORPH_BLACKHAT
    }
    return cv2.morphologyEx(values, methods[params['method']], element, iterations=params['iterations'])


@operation('binary_logic','Binary AND / OR / XOR / NAND / NOR',GROUP,'Threshold two aligned images, then apply Boolean set operations.', 'AND=intersection; OR=union; XOR=symmetric difference',SOURCE,
           [
               choice('method','Boolean operation','AND',['AND','OR','XOR','NAND','NOR']),
               field('threshold','Binary threshold',127,0,255)
           ],requires_reference=True)
def binary_logic(image, params, reference):
    values = gray(image)
    target = reference_gray(reference, values.shape)
    a = values > params['threshold']
    b = target > params['threshold']
    method = params['method']
    if method == 'AND':
        result = a & b
    elif method == 'OR':
        result = a | b
    elif method == 'XOR':
        result = a ^ b
    elif method == 'NAND':
        result = ~(a & b)
    else:
        result = ~(a | b)
    return result.astype(np.uint8) * 255


@operation(
    'binary_not',
    'Binary NOT / set complement',
    GROUP,
    'Complement a thresholded binary foreground mask.',
    'NOT A = 1-A',
    SOURCE,
    [field('threshold','Foreground threshold',127,0,255)]
)
def binary_not(image, params, reference):
    return (gray(image) <= params['threshold']).astype(np.uint8) * 255


@operation('hit_miss','Hit-or-miss ternary template',GROUP,'Find exact binary neighborhood patterns. In the template, 1 requires foreground, -1 requires background and 0 is ignored.', 'A⊛B=(A erosion Bforeground) ∩ (Acomplement erosion Bbackground)',SOURCE,
           [
               dict(
                   key='template',
                   label='Nine ternary values (1, -1, 0)',
                   default='0,1,0,-1,1,1,-1,-1,0',
                   type='text'
               ),
               field('threshold','Binary threshold',127,0,255)
           ])
def hit_miss(image, params, reference):
    try:
        values = [int(value.strip()) for value in params['template'].split(',')]
    except ValueError as exc:
        raise ValueError('Template needs nine integers from -1, 0 and 1.') from exc
    if len(values) != 9 or any((value not in [-1, 0, 1] for value in values)) or (not any(values)):
        raise ValueError('Template needs nine ternary values and at least one constraint.')
    binary = (gray(image) > params['threshold']).astype(np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_HITMISS, np.array(values, np.int8).reshape(3, 3)) * 255


@operation('skeleton','Skeletonization / thinning / pruning',GROUP,'Extract a one-pixel-wide skeleton, medial axis or iterative morphological skeleton. Pruning removes endpoint pixels for a selected number of steps.', 'Skeleton = union of eroded sets minus their openings',SOURCE,
           [
               choice('method','Method','thinning',['thinning','medial axis','morphological skeleton','pruning']),
               field('threshold','Foreground threshold',127,0,255),
               field('iterations','Pruning steps',3,1,20)
           ])
def skeleton(image, params, reference):
    binary = gray(bounded(image, 512)) > params['threshold']
    if params['method'] == 'medial axis':
        result = morphology.medial_axis(binary, rng=42)
    elif params['method'] == 'morphological skeleton':
        values = binary.astype(np.uint8) * 255
        result = np.zeros_like(binary)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        for _ in range(max(binary.shape) + 1):
            eroded = cv2.erode(values, element, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            opened = cv2.dilate(eroded, element)
            result |= values > opened
            values = eroded
            if not np.any(values):
                break
    else:
        result = morphology.skeletonize(binary)
        if params['method'] == 'pruning':
            kernel = np.ones((3, 3), np.uint8)
            kernel[1, 1] = 0
            for _ in range(params['iterations']):
                neighbors = ndimage.convolve(result.astype(np.uint8), kernel, mode='constant')
                result[result & (neighbors == 1)] = False
    return (result.astype(np.uint8) * 255, {
        'skeleton_pixels': int(result.sum()),
        'note': 'Analysis is limited to a 512-pixel side. Endpoint pruning can shorten true branches as well as remove spurs.'
    })


@operation('distance_boundary','Distance / digital depth / boundaries / hole filling',GROUP,'Inspect binary object interiors with a distance transform or extract their boundaries and fill holes.', 'Depth(pixel) = distance to nearest background; boundary=A-erosion(A)',SOURCE+' / Chap1 Digital Image Fundamentals.pdf',
           [
               choice('method','Operation','distance transform',['distance transform','boundary','fill holes']),
               field('threshold','Foreground threshold',127,0,255)
           ])
def distance_boundary(image, params, reference):
    binary = gray(image) > params['threshold']
    if params['method'] == 'distance transform':
        distance = ndimage.distance_transform_edt(binary)
        return (normalize(distance), {
            'maximum_digital_depth': float(distance.max()),
            'note': 'The depth map is normalized for display; distances are in image pixels.'
        })
    if params['method'] == 'fill holes':
        return ndimage.binary_fill_holes(binary).astype(np.uint8) * 255
    return (binary & ~ndimage.binary_erosion(binary)).astype(np.uint8) * 255


@operation(
    'morph_reconstruction',
    'Morphological reconstruction',
    GROUP,
    'Reconstruct by geodesic dilation from an eroded marker under the original mask.',
    'marker ≤ reconstructed ≤ mask',
    SOURCE,
    [
        field('kernel','Marker erosion size (odd)',5,3,31),
        choice('domain','Domain','grayscale',['grayscale','binary']),
        field('threshold','Binary threshold',127,0,255)
    ]
)
def morph_reconstruction(image, params, reference):
    values = gray(image)
    if params['domain'] == 'binary':
        values = (values > params['threshold']).astype(np.uint8) * 255
    marker = cv2.erode(values, np.ones((params['kernel'], params['kernel']), np.uint8))
    return morphology.reconstruction(marker, values, method='dilation').astype(np.uint8)
