import cv2
import numpy as np
from scipy import ndimage

from code_02_image_operations import operation,field,choice,gray,normalize
from code_04_image_helpers import bounded,u8,reference_gray


@operation('concept_lab','Acquisition / perception / memory / FPS / linearity demos','Fundamentals','Run small educational experiments for topics that are not ordinary image filters.', 'Models illustrate concepts; they do not reproduce physical cameras, eyes or trained networks.','Chap1 Digital Image Fundamentals.pdf / Chap2 Image Enhancement.pdf / Chap4-Image Transforms.pdf',
           [
               choice(
                   'demo',
                   'Concept experiment',
                   'sampling and ADC',
                   [
                       'sampling and ADC',
                       'eye image formation',
                       'simultaneous contrast',
                       'memory and resolution',
                       'FPS motion samples',
                       'linearity test',
                       '3D connectivity',
                       'processing pipeline'
                   ]
               ),
               field('samples','Sampling side / FPS',16,2,128),
               field('bits','Bit depth',4,1,8)
           ])
def concept_lab(image, params, reference):
    demo = params['demo']
    values = bounded(image, 512)
    if demo == 'sampling and ADC':
        sample = cv2.resize(gray(values), (params['samples'], params['samples']), interpolation=cv2.INTER_AREA)
        levels = 2 ** params['bits']
        result = u8(np.floor(sample.astype(float) * levels / 256) * 255 / (levels - 1))
        return (result, {
            'pixel_matrix': result.tolist() if params['samples'] <= 16 else 'Matrix omitted above 16×16',
            'quantization_levels': levels,
            'note': 'Sampling models spatial sensors; quantization models an ideal ADC. This does not simulate a specific camera response.'
        })
    if demo == 'eye image formation':
        return (
            cv2.flip(values, -1),
            {'note': 'A pinhole/lens image is inverted at the sensing plane. Human vision involves further neural interpretation; this inversion alone is not an eye simulation.'}
        )
    if demo == 'simultaneous contrast':
        output = np.full((256, 512, 3), 40, np.uint8)
        output[:, 256:] = 215
        output[80:176, 80:176] = 128
        output[80:176, 336:432] = 128
        return (output, {
            'patch_intensity_left': 128,
            'patch_intensity_right': 128,
            'note': 'Both patches have exactly the same intensity but may appear different because of their surrounds.'
        })
    if demo == 'memory and resolution':
        h, w = image.shape[:2]
        return (values, {
            'width': w,
            'height': h,
            'megapixels': w * h / 1000000.0,
            '8_bit_grayscale_bytes': w * h,
            '24_bit_rgb_bytes': w * h * 3,
            'selected_bit_depth_grayscale_bytes': int(np.ceil(w * h * params['bits'] / 8)),
            'printed_width_inches_at_300_ppi': w / 300,
            'note': 'Raw uncompressed storage calculations omit headers, padding and metadata. DPI is a printer-dot property; PPI describes image pixels per inch.'
        })
    if demo == 'FPS motion samples':
        canvas = np.full((160, 640, 3), 245, np.uint8)
        for frame in range(8):
            x = 40 + frame * 80
            y = 80
            position = frame / max(params['samples'], 1)
            cv2.circle(canvas, (x, round(y + 30 * np.sin(2 * np.pi * position))), 8, (60, 140, 40), -1)
            cv2.putText(canvas, f'{position:.2f}s', (x - 27, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (40, 40, 40), 1)
        return (canvas, {
            'frames_per_second': params['samples'],
            'time_between_frames': 1 / params['samples'],
            'note': 'Eight synthetic motion samples; this is a temporal sampling illustration, not an uploaded video.'
        })
    if demo == 'linearity test':
        a = gray(values).astype(float)
        b = np.flip(a, axis=1)
        alpha = 0.3
        beta = 0.7
        combined = alpha * a + beta * b
        blurred_combined = cv2.GaussianBlur(combined, (5, 5), 0)
        blurred_a = cv2.GaussianBlur(a, (5, 5), 0)
        blurred_b = cv2.GaussianBlur(b, (5, 5), 0)
        linear_error = float(np.max(np.abs(blurred_combined - (alpha * blurred_a + beta * blurred_b))))
        median_combined = ndimage.median_filter(combined, size=3)
        median_a = ndimage.median_filter(a, size=3)
        median_b = ndimage.median_filter(b, size=3)
        nonlinear_error = float(np.max(np.abs(median_combined - (alpha * median_a + beta * median_b))))
        return (u8(blurred_combined), {
            'gaussian_linearity_error': linear_error,
            'median_linearity_error': nonlinear_error,
            'note': 'Linearity is tested before uint8 clipping. This uses an image and its reflection as two input signals.'
        })
    if demo == '3D connectivity':
        volume = np.zeros((7, 7, 7), bool)
        volume[2, 2, 2] = True
        volume[3, 3, 3] = True
        volume[4, 4, 4] = True
        counts = {
            str(neighbors): int(ndimage.label(volume, ndimage.generate_binary_structure(3, rank))[1]) for neighbors,
            rank in [(6, 1), (18, 2), (26, 3)]
        }
        return (cv2.resize(
            volume.max(axis=0).astype(np.uint8) * 255,
            (210, 210),
            interpolation=cv2.INTER_NEAREST
        ), {
            'synthetic_3d_components': counts,
            'note': 'Three diagonal foreground voxels in a synthetic 7³ volume; output is a maximum-intensity projection. In a 2D rectangular grid, standard neighborhoods are 4/8; a six-neighbor case needs a different lattice.'
        })
    result = cv2.Canny(cv2.GaussianBlur(gray(values), (5, 5), 0), 80, 160)
    return (result, {
        'stages': [
            'upload/acquisition',
            '8-bit representation',
            'grayscale conversion',
            'Gaussian preprocessing',
            'edge extraction',
            'display/analysis'
        ],
        'note': 'A concrete image-processing pipeline; cognition and semantic recognition are distinct higher-level tasks.'
    })


@operation(
    'spectral_demo',
    'Spectral tristimulus integral demo',
    'Color & intensity',
    'Integrate a synthetic spectrum against three clearly labeled Gaussian sensitivity functions.',
    'X,Y,Z = Σ spectrum(λ) × sensitivity(λ) Δλ',
    'Chap3-Color Image Processing.pdf',
    [field('peak','Spectrum peak wavelength (nm)',550,400,700),field('width','Spectrum width (nm)',30,5,100)]
)
def spectral_demo(image, params, reference):
    wavelengths = np.linspace(380, 780, 401)
    spectrum = np.exp(-0.5 * ((wavelengths - params['peak']) / params['width']) ** 2)
    sensitivities = np.array([
        np.exp(-0.5 * ((wavelengths - center) / width) ** 2) for center,
        width in [(600, 45), (550, 35), (450, 25)]
    ])
    values = np.sum(sensitivities * spectrum, axis=1)
    canvas = np.full((280, 620, 3), 250, np.uint8)
    for curve, color in zip([spectrum, *sensitivities], [(20, 20, 20), (0, 0, 220), (0, 160, 0), (220, 0, 0)]):
        points = np.column_stack((40 + (wavelengths - 380) / 400 * 540, 240 - curve * 190)).astype(np.int32)
        cv2.polylines(canvas, [points], False, color, 2)
    cv2.putText(
        canvas,
        'Synthetic spectrum + sensitivity functions',
        (30, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (30, 30, 30),
        1
    )
    return (canvas, {
        'synthetic_tristimulus_integrals': values.tolist(),
        'normalized_coefficients': (values / max(values.sum(), 1e-12)).tolist(),
        'note': 'These Gaussian sensitivities are teaching functions, not official CIE standard-observer tables. Use CIE XYZ chromaticity for standard RGB-to-XYZ conversion.'
    })


@operation(
    'registration_fusion',
    'Translation registration / MRI–CT fusion demo',
    'Filters & restoration',
    'Estimate translation between two grayscale images by phase correlation and overlay the registered reference.',
    'shift from Fourier cross-power spectrum; fusion = weighted overlay',
    'Chap6-Image Restoration.pdf',
    [field('weight','Input fusion weight',.5,0,1,.1)],
    requires_reference=True
)
def registration_fusion(image, params, reference):
    if reference is None:
        raise ValueError('Upload a reference image.')
    values = bounded(image, 512)
    target = cv2.resize(reference, (values.shape[1], values.shape[0]))
    a = gray(values).astype(np.float32)
    b = gray(target).astype(np.float32)
    shift, response = cv2.phaseCorrelate(b, a)
    dx, dy = shift if response > 0.01 else (0, 0)
    aligned = cv2.warpAffine(
        target,
        np.float32([[1, 0, dx], [0, 1, dy]]),
        (values.shape[1], values.shape[0]),
        borderMode=cv2.BORDER_CONSTANT
    )
    return (cv2.addWeighted(values, params['weight'], aligned, 1 - params['weight'], 0), {
        'shift_dx': float(dx),
        'shift_dy': float(dy),
        'phase_response': float(response),
        'note': 'Translation-only generic image fusion; this is not a clinically validated deformable MRI–CT registration system.'
    })


@operation(
    'rician_noise',
    'MRI magnitude / Rician noise demo',
    'Filters & restoration',
    'Add independent Gaussian noise to two synthetic quadrature components, then form the magnitude image.',
    'magnitude = sqrt((signal+nreal)² + nimag²)',
    'Chap6-Image Restoration.pdf',
    [field('sigma','Quadrature noise sigma',15,1,80),field('seed','Random seed',42,0,9999)]
)
def rician_noise(image, params, reference):
    values = gray(image).astype(float)
    rng = np.random.default_rng(params['seed'])
    real = values + rng.normal(0, params['sigma'], values.shape)
    imag = rng.normal(0, params['sigma'], values.shape)
    return (
        u8(np.hypot(real, imag)),
        {'note': 'Synthetic magnitude-noise model; no real MRI quadrature measurements are supplied.'}
    )
