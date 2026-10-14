import cv2
import numpy as np

from code_02_image_operations import operation, field, choice, gray, normalize
from code_04_image_helpers import u8

SOURCE='Chap3-Color Image Processing.pdf'
GROUP='Color & intensity'


def channels(image, space):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(float) / 255
    if space == 'RGB':
        return (rgb, ['R', 'G', 'B'])
    if space == 'CMY':
        return (1 - rgb, ['C', 'M', 'Y'])
    if space == 'CMYK':
        k = 1 - rgb.max(axis=2)
        cmy = (1 - rgb - k[:, :, None]) / np.maximum(1 - k[:, :, None], 1e-12)
        return (np.dstack((cmy, k)), ['C', 'M', 'Y', 'K'])
    if space == 'HSI':
        r, g, b = (rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2])
        theta = np.arccos(np.clip(
            0.5 * (r - g + (r - b)) / np.maximum(np.sqrt((r - g) ** 2 + (r - b) * (g - b)), 1e-12),
            -1,
            1
        ))
        hue = np.where(b <= g, theta, 2 * np.pi - theta) / (2 * np.pi)
        intensity = rgb.mean(axis=2)
        saturation = 1 - rgb.min(axis=2) / np.maximum(intensity, 1e-12)
        saturation[intensity == 0] = 0
        hue[saturation < 1e-10] = 0
        return (np.dstack((hue, saturation, intensity)), ['H', 'S', 'I'])
    conversions = {
        'HSV': cv2.COLOR_BGR2HSV,
        'Lab': cv2.COLOR_BGR2LAB,
        'XYZ': cv2.COLOR_BGR2XYZ,
        'YCbCr': cv2.COLOR_BGR2YCrCb,
        'YUV': cv2.COLOR_BGR2YUV
    }
    result = cv2.cvtColor(image, conversions[space]).astype(float) / 255
    if space == 'HSV':
        result[:, :, 0] *= 255 / 179
    names = {
        'HSV': ['H', 'S', 'V'],
        'Lab': ['L', 'a', 'b'],
        'XYZ': ['X', 'Y', 'Z'],
        'YCbCr': ['Y', 'Cr', 'Cb'],
        'YUV': ['Y', 'U', 'V']
    }
    return (result, names[space])


def hsi_rgb(hsi):
    h, s, i = (hsi[:, :, 0] * 2 * np.pi, hsi[:, :, 1], hsi[:, :, 2])
    rgb = np.empty(hsi.shape, float)
    for sector in range(3):
        mask = (h >= sector * 2 * np.pi / 3) & (h < (sector + 1) * 2 * np.pi / 3)
        angle = h[mask] - sector * 2 * np.pi / 3
        low = i[mask] * (1 - s[mask])
        high = i[mask] * (1 + s[mask] * np.cos(angle) / np.maximum(np.cos(np.pi / 3 - angle), 1e-12))
        middle = 3 * i[mask] - low - high
        values = np.stack((high, middle, low), axis=-1)
        rgb[mask] = np.roll(values, sector, axis=-1)
    return cv2.cvtColor(u8(rgb * 255), cv2.COLOR_RGB2BGR)


@operation('color_models','RGB / CMY / CMYK / HSI / XYZ / YUV models',GROUP,'Inspect color-model channels or convert to the selected model and back. Channel maps are grayscale displays, not natural RGB colors.', 'CMY=1-RGB; K=1-max(RGB); I=(R+G+B)/3',SOURCE,
           [
               choice('space','Color model','CMYK',['RGB','CMY','CMYK','HSI','HSV','Lab','XYZ','YCbCr','YUV']),
               choice('mode','Output','channel',['channel','round-trip']),
               field('channel','Channel number',1,1,4)
           ])
def color_models(image, params, reference):
    values, names = channels(image, params['space'])
    if params['mode'] == 'channel':
        if params['channel'] > len(names):
            raise ValueError(f"{params['space']} has only {len(names)} channels.")
        index = params['channel'] - 1
        return (u8(values[:, :, index] * 255), {
            'channel': names[index],
            'channel_means_0_to_1': values.mean(axis=(0, 1)).round(5).tolist(),
            'note': 'OpenCV YCbCr channels are ordered Y, Cr, Cb. HSV hue is scaled to the display range.'
        })
    space = params['space']
    if space == 'RGB':
        result = image.copy()
    elif space == 'CMY':
        result = cv2.cvtColor(u8((1 - values) * 255), cv2.COLOR_RGB2BGR)
    elif space == 'CMYK':
        result = cv2.cvtColor(u8((1 - values[:, :, :3]) * (1 - values[:, :, 3:]) * 255), cv2.COLOR_RGB2BGR)
    elif space == 'HSI':
        result = hsi_rgb(values)
    else:
        encoded = values.copy() * 255
        if space == 'HSV':
            encoded[:, :, 0] *= 179 / 255
        inverse = {
            'HSV': cv2.COLOR_HSV2BGR,
            'Lab': cv2.COLOR_LAB2BGR,
            'XYZ': cv2.COLOR_XYZ2BGR,
            'YCbCr': cv2.COLOR_YCrCb2BGR,
            'YUV': cv2.COLOR_YUV2BGR
        }
        result = cv2.cvtColor(u8(encoded), inverse[space])
    return (result, {
        'round_trip_mse': float(np.mean((result.astype(float) - image) ** 2)),
        'note': '8-bit model conversion can introduce rounding and gamut clipping; HSI/CMYK use floating-point formulas.'
    })


@operation(
    'chromaticity',
    'CIE XYZ chromaticity',
    GROUP,
    'Separate chromaticity from intensity using normalized XYZ coordinates.',
    'x=X/(X+Y+Z); y=Y/(X+Y+Z); z=1-x-y',
    SOURCE,
    [choice('coordinate','Coordinate','x',['x','y','z'])]
)
def chromaticity(image, params, reference):
    from skimage.color import rgb2xyz
    xyz = rgb2xyz(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(float) / 255)
    coordinates = xyz / np.maximum(xyz.sum(axis=2, keepdims=True), 1e-12)
    index = ['x', 'y', 'z'].index(params['coordinate'])
    return (u8(coordinates[:, :, index] * 255), {
        'mean_xyz_chromaticity': coordinates.mean(axis=(0, 1)).tolist(),
        'note': 'Black pixels have zero XYZ sum and are represented by zeros; this is not a full spectral CIE horseshoe plot.'
    })


@operation(
    'pseudocolor',
    'Pseudocolor / density slicing',
    GROUP,
    'Map grayscale intensities to colors or discrete intensity bands.',
    'RGB = color lookup table(gray intensity)',
    SOURCE,
    [
        choice('map','Color map','viridis',['viridis','jet','hot','rainbow','density slices']),
        field('levels','Density levels',8,2,32)
    ]
)
def pseudocolor(image, params, reference):
    values = gray(image)
    maps = {
        'viridis': cv2.COLORMAP_VIRIDIS,
        'jet': cv2.COLORMAP_JET,
        'hot': cv2.COLORMAP_HOT,
        'rainbow': cv2.COLORMAP_RAINBOW
    }
    if params['map'] == 'density slices':
        values = u8(np.floor(values.astype(float) * params['levels'] / 256) * 255 / (params['levels'] - 1))
        return cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
    return cv2.applyColorMap(values, maps[params['map']])


@operation('color_slicing','Color slicing · cube / sphere',GROUP,'Keep colors near a target RGB value and suppress the others.', 'Cube: max|RGB-target| ≤ radius; sphere: ||RGB-target||₂ ≤ radius',SOURCE,
           [
               choice('shape','Color neighborhood','sphere',['sphere','cube']),
               field('red','Target red',200,0,255),
               field('green','Target green',100,0,255),
               field('blue','Target blue',50,0,255),
               field('radius','Color radius',80,1,255),
               choice('background','Other pixels','grayscale',['grayscale','black'])
           ])
def color_slicing(image, params, reference):
    distance = image.astype(float) - np.array([params['blue'], params['green'], params['red']])
    mask = np.linalg.norm(distance, axis=2) <= params['radius'] if params['shape'] == 'sphere' else np.abs(distance).max(axis=2) <= params['radius']
    result = cv2.cvtColor(gray(image), cv2.COLOR_GRAY2BGR) if params['background'] == 'grayscale' else np.zeros_like(image)
    result[mask] = image[mask]
    return (result, {'selected_pixels': int(mask.sum())})


@operation(
    'white_balance',
    'White balance / color correction',
    GROUP,
    'Estimate channel gains from channel averages, maxima or Minkowski means.',
    'balanced channel = channel × target / channel statistic',
    SOURCE,
    [
        choice('method','Method','gray world',['gray world','white patch','shades of gray']),
        field('power','Shades-of-gray power',6,1,10)
    ]
)
def white_balance(image, params, reference):
    values = image.astype(float)
    if params['method'] == 'gray world':
        estimate = values.mean(axis=(0, 1))
    elif params['method'] == 'white patch':
        estimate = values.max(axis=(0, 1))
    else:
        estimate = np.mean((values / 255) ** params['power'], axis=(0, 1)) ** (1 / params['power']) * 255
    target = float(estimate.mean())
    gain = np.divide(target, estimate, out=np.ones(3), where=estimate > 0)
    return (u8(values * gain), {'bgr_gains': gain.tolist()})


@operation('hsv_adjust','Hue / saturation / value adjustment',GROUP,'Edit hue, saturation and brightness in HSV space.', 'H′=(H+shift) mod 360; S′=gain×S; V′=gain×V',SOURCE,
           [
               field('hue','Hue shift (degrees)',30,-180,180),
               field('saturation','Saturation gain',1.2,0,3,.1),
               field('value','Value gain',1,0,3,.1)
           ])
def hsv_adjust(image, params, reference):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(float)
    hsv[:, :, 0] = (hsv[:, :, 0] + params['hue'] / 2) % 180
    hsv[:, :, 1] *= params['saturation']
    hsv[:, :, 2] *= params['value']
    return cv2.cvtColor(u8(hsv), cv2.COLOR_HSV2BGR)


@operation('color_processing','Color filtering / edges / morphology',GROUP,'Process RGB channels independently, or process luminance only in Lab to preserve chromaticity.', 'Per-channel processing versus luminance-only processing',SOURCE,
           [
               choice('domain','Domain','Lab luminance',['RGB channels','Lab luminance']),
               choice(
                   'method',
                   'Operation',
                   'Gaussian',
                   ['Gaussian','mean','median','Sobel','erosion','dilation','opening','closing']
               ),
               field('kernel','Kernel size (odd)',3,3,15)
           ])
def color_processing(image, params, reference):
    values = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) if params['domain'] == 'Lab luminance' else image.copy()
    for index in [0] if params['domain'] == 'Lab luminance' else range(3):
        plane = values[:, :, index]
        k = params['kernel']
        if params['method'] == 'Gaussian':
            plane = cv2.GaussianBlur(plane, (k, k), 0)
        elif params['method'] == 'mean':
            plane = cv2.blur(plane, (k, k))
        elif params['method'] == 'median':
            plane = cv2.medianBlur(plane, k)
        elif params['method'] == 'Sobel':
            plane = normalize(cv2.magnitude(
                cv2.Sobel(plane, cv2.CV_32F, 1, 0),
                cv2.Sobel(plane, cv2.CV_32F, 0, 1)
            ))
        else:
            plane = cv2.morphologyEx(plane, {
                'erosion': cv2.MORPH_ERODE,
                'dilation': cv2.MORPH_DILATE,
                'opening': cv2.MORPH_OPEN,
                'closing': cv2.MORPH_CLOSE
            }[params['method']], np.ones((k, k), np.uint8))
        values[:, :, index] = plane
    return cv2.cvtColor(values, cv2.COLOR_LAB2BGR) if params['domain'] == 'Lab luminance' else values


@operation(
    'chroma_subsampling',
    'Chroma subsampling · 4:4:4 / 4:2:2 / 4:2:0',
    GROUP,
    'Reduce chroma resolution and reconstruct a displayable image while keeping luminance unchanged.',
    '4:2:0 stores chroma at half width and half height',
    SOURCE,
    [choice('sampling','Chroma sampling','4:2:0',['4:4:4','4:2:2','4:2:0'])]
)
def chroma_subsampling(image, params, reference):
    values = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    h, w = values.shape[:2]
    ch = h if params['sampling'] != '4:2:0' else max(1, (h + 1) // 2)
    cw = w if params['sampling'] == '4:4:4' else max(1, (w + 1) // 2)
    for channel in [1, 2]:
        values[:, :, channel] = cv2.resize(
            cv2.resize(values[:, :, channel], (cw, ch), interpolation=cv2.INTER_AREA),
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )
    return (cv2.cvtColor(values, cv2.COLOR_YCrCb2BGR), {
        'luma_samples': h * w,
        'chroma_samples_per_channel': ch * cw,
        'note': 'Display reconstructs chroma by interpolation. No JPEG bitstream is produced by this operation.'
    })


@operation(
    'color_equalize',
    'Color histogram / adaptive equalization',
    GROUP,
    'Equalize RGB channels separately or equalize Lab luminance while preserving a and b channels.',
    'Per-channel CDF mapping versus luminance-only mapping',
    SOURCE,
    [
        choice('domain','Domain','Lab luminance',['RGB channels','Lab luminance']),
        choice('method','Method','equalization',['equalization','CLAHE']),
        field('clip','CLAHE clip limit',2,.1,40,.1)
    ]
)
def color_equalize(image, params, reference):
    values = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) if params['domain'] == 'Lab luminance' else image.copy()
    for channel in [0] if params['domain'] == 'Lab luminance' else range(3):
        plane = values[:, :, channel]
        values[:, :, channel] = cv2.equalizeHist(plane) if params['method'] == 'equalization' else cv2.createCLAHE(clipLimit=params['clip'], tileGridSize=(8, 8)).apply(plane)
    return (
        cv2.cvtColor(values, cv2.COLOR_LAB2BGR) if params['domain'] == 'Lab luminance' else values,
        {'note': 'Independent RGB equalization may change color balance; luminance-only mode preserves chromatic components more closely.'}
    )


@operation(
    'color_segmentation',
    'HSV / Lab color segmentation',
    GROUP,
    'Select pixels by a range in one color-space channel.',
    'foreground if lower ≤ selected channel ≤ upper',
    SOURCE,
    [
        choice('space','Color space','HSV',['HSV','Lab']),
        field('channel','Channel number',1,1,3),
        field('lower','Lower threshold (0–255)',30,0,255),
        field('upper','Upper threshold (0–255)',150,0,255)
    ]
)
def color_segmentation(image, params, reference):
    if params['lower'] > params['upper']:
        raise ValueError('Lower threshold must not exceed upper threshold.')
    values, names = channels(image, params['space'])
    plane = values[:, :, params['channel'] - 1] * 255
    return (((plane >= params['lower']) & (plane <= params['upper'])).astype(np.uint8) * 255, {
        'channel': names[params['channel'] - 1],
        'note': 'HSV hue uses a display-normalized 0–255 range. This is color-range segmentation, not object classification.'
    })


@operation(
    'color_sharpen',
    'Color Laplacian / tonal correction',
    GROUP,
    'Apply Laplacian sharpening to all color channels or lift shadows while reducing highlights with a smooth tone curve.',
    'sharpen=f-amount×Laplacian(f); tone curve=piecewise interpolation',
    SOURCE,
    [choice('mode','Operation','Laplacian',['Laplacian','tonal correction']),field('amount','Amount',.3,.1,2,.1)]
)
def color_sharpen(image, params, reference):
    if params['mode'] == 'Laplacian':
        return u8(image.astype(float) - params['amount'] * cv2.Laplacian(image, cv2.CV_32F, ksize=3))
    table = u8(np.interp(
        np.arange(256),
        [0, 64, 192, 255],
        [0, min(160, 64 + 30 * params['amount']), max(120, 192 - 20 * params['amount']), 255]
    ))
    return cv2.LUT(image, table)
