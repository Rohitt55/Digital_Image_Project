import cv2
import numpy as np
import pywt
from scipy import fft, linalg, signal

from code_02_image_operations import operation, field, choice, gray, normalize
from code_04_image_helpers import bounded, u8

SOURCE='Chap4-Image Transforms.pdf'
GROUP='Transforms & comparison'


@operation('affine','Translation / rotation / shear / affine', 'Geometry','Apply a 2D affine transform about the image center, including scale, shear and translation. Output keeps the input canvas.', 'p′ = A p + t in homogeneous coordinates',SOURCE,
           [
               field('angle','Angle (degrees)',15,-180,180),
               field('scale','Scale',1,.1,3,.1),
               field('shear','Horizontal shear',0,-1,1,.1),
               field('x','Horizontal translation (%)',0,-100,100),
               field('y','Vertical translation (%)',0,-100,100)
           ])
def affine(image, params, reference):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), params['angle'], params['scale'])
    homogeneous = np.vstack((matrix, [0, 0, 1]))
    shear = np.array([[1, params['shear'], -params['shear'] * (h - 1) / 2], [0, 1, 0], [0, 0, 1]])
    homogeneous = shear @ homogeneous
    homogeneous[:2, 2] += np.array([w * params['x'] / 100, h * params['y'] / 100])
    return (
        cv2.warpAffine(image, homogeneous[:2], (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT),
        {'affine_matrix': homogeneous.round(4).tolist()}
    )


@operation(
    'pyramid',
    'Gaussian / Laplacian image pyramid',
    'Geometry',
    'Build a multiresolution pyramid and display its levels as a montage.',
    'Gaussian: blur+downsample; Laplacian: Gk-expand(Gk+1)',
    SOURCE,
    [choice('type','Pyramid','Gaussian',['Gaussian','Laplacian']),field('levels','Levels',3,1,5)]
)
def pyramid(image, params, reference):
    current = bounded(image, 768)
    gaussian = [current]
    for _ in range(params['levels']):
        if min(current.shape[:2]) < 2:
            break
        current = cv2.pyrDown(current)
        gaussian.append(current)
    levels = gaussian
    if params['type'] == 'Laplacian':
        levels = [normalize(gaussian[i].astype(float) - cv2.pyrUp(
            gaussian[i + 1],
            dstsize=(gaussian[i].shape[1], gaussian[i].shape[0])
        )) for i in range(len(gaussian) - 1)] + [gaussian[-1]]
    h, w = gaussian[0].shape[:2]
    canvas = np.zeros((h, sum((level.shape[1] for level in levels)), 3), np.uint8)
    x = 0
    for level in levels:
        canvas[:level.shape[0], x:x + level.shape[1]] = level
        x += level.shape[1]
    return (canvas, {
        'level_shapes': [[a.shape[0], a.shape[1]] for a in levels],
        'note': 'Pyramid construction uses a preview with a maximum side of 768 pixels.'
    })


@operation(
    'sigmoid',
    'Sigmoid / nonlinear contrast stretching',
    'Color & intensity',
    'Apply a smooth S-shaped transfer function centered at a chosen intensity.',
    's = 255 / (1 + exp(-gain(r/255-midpoint)))',
    SOURCE,
    [field('gain','Sigmoid gain',10,1,30),field('midpoint','Midpoint (0–1)',.5,.1,.9,.05)]
)
def sigmoid(image, params, reference):
    return u8(255 / (1 + np.exp(-params['gain'] * (image.astype(float) / 255 - params['midpoint']))))


@operation(
    'fourier_components',
    'Fourier magnitude / phase / real / imaginary',
    GROUP,
    'Inspect the components of the centered complex 2D Fourier transform.',
    'F = magnitude × exp(j phase)',
    SOURCE,
    [choice('component','Component','phase',['magnitude','phase','real','imaginary'])]
)
def fourier_components(image, params, reference):
    spectrum = np.fft.fftshift(np.fft.fft2(gray(image)))
    if params['component'] == 'magnitude':
        result = normalize(np.log1p(np.abs(spectrum)))
    elif params['component'] == 'phase':
        result = u8((np.angle(spectrum) + np.pi) * 255 / (2 * np.pi))
    else:
        result = normalize(spectrum.real if params['component'] == 'real' else spectrum.imag)
    return (
        result,
        {'note': 'This is a coefficient visualization, not a spatial-domain processed photo. Phase maps −π…π to 0…255.'}
    )


@operation(
    'convolution_theorem',
    'Convolution theorem & zero padding',
    GROUP,
    'Compare linear spatial convolution and zero-padded FFT convolution using the same kernel.',
    'F(f*k) = F(f)F(k); padding prevents circular wraparound',
    SOURCE,
    [field('kernel','Mean kernel size (odd)',5,3,31)]
)
def convolution_theorem(image, params, reference):
    values = gray(bounded(image, 512)).astype(float)
    kernel = np.ones((params['kernel'], params['kernel'])) / params['kernel'] ** 2
    spatial = signal.convolve2d(values, kernel, mode='same', boundary='fill')
    frequency = signal.fftconvolve(values, kernel, mode='same')
    return (u8(frequency), {
        'maximum_spatial_fft_error': float(np.max(np.abs(spatial - frequency))),
        'note': 'Both methods use zero padding and the same output support; analysis is limited to a 512-pixel side.'
    })


@operation(
    'dct',
    'Discrete cosine transform / inverse DCT',
    GROUP,
    'Compute an orthonormal 2D DCT or independent 1D row DCTs, then optionally reconstruct from low-frequency coefficients.',
    'F=DCT(f); reconstruction=IDCT(F)',
    SOURCE,
    [
        choice('dimensions','Transform dimensions','2D',['2D','1D rows']),
        choice('mode','Output','spectrum',['spectrum','reconstruction']),
        field('retain','Retain low frequencies (%)',50,1,100)
    ]
)
def dct(image, params, reference):
    values = gray(bounded(image, 1024)).astype(float)
    coefficients = fft.dctn(values, type=2, norm='ortho') if params['dimensions'] == '2D' else fft.dct(values, type=2, axis=1, norm='ortho')
    if params['mode'] == 'spectrum':
        return (
            normalize(np.log1p(np.abs(coefficients))),
            {'note': 'Log absolute DCT coefficients are normalized for display.'}
        )
    h, w = values.shape
    rows = h if params['dimensions'] == '1D rows' else max(1, round(h * params['retain'] / 100))
    cols = max(1, round(w * params['retain'] / 100))
    retained = np.zeros_like(coefficients)
    retained[:rows, :cols] = coefficients[:rows, :cols]
    result = fft.idctn(retained, type=2, norm='ortho') if params['dimensions'] == '2D' else fft.idct(retained, type=2, axis=1, norm='ortho')
    return (u8(result), {
        'retained_coefficients': rows * cols,
        'total_coefficients': h * w,
        'reconstruction_mse': float(np.mean((values - result) ** 2))
    })


@operation('wavelet','Haar / db4 / db8 wavelet transform',GROUP,'Compute multilevel DWT subbands, reconstruct after thresholding or retain only the approximation band.', 'DWT → approximation/detail subbands → inverse DWT', 'Chap2 Image Enhancement.pdf / '+SOURCE,
           [
               choice('wavelet','Wavelet','haar',['haar','db4','db8']),
               choice('mode','Output','subbands',['subbands','reconstruction','approximation only']),
               field('level','Decomposition level',2,1,4),
               field('threshold','Detail soft threshold',0,0,100)
           ])
def wavelet(image, params, reference):
    values = gray(bounded(image, 1024)).astype(float)
    maximum = pywt.dwtn_max_level(values.shape, params['wavelet'])
    level = min(params['level'], max(1, maximum))
    coefficients = pywt.wavedec2(values, params['wavelet'], level=level, mode='periodization')
    if params['mode'] == 'subbands':
        array, _ = pywt.coeffs_to_array(coefficients)
        return (normalize(np.log1p(np.abs(array))), {
            'decomposition_level': level,
            'note': 'Subband montage is normalized log magnitude. The wavelet boundary mode is periodization.'
        })
    filtered = [coefficients[0]]
    for detail in coefficients[1:]:
        bands = []
        for band in detail:
            if params['mode'] == 'approximation only':
                band = np.zeros_like(band)
            elif params['threshold']:
                band = pywt.threshold(band, params['threshold'], mode='soft')
            bands.append(band)
        filtered.append(tuple(bands))
    result = pywt.waverec2(filtered, params['wavelet'], mode='periodization')[:values.shape[0], :values.shape[1]]
    return (u8(result), {
        'decomposition_level': level,
        'reconstruction_mse': float(np.mean((values - result) ** 2))
    })


@operation('walsh_hadamard','Walsh / Hadamard transform & inverse',GROUP,'Transform a power-of-two grayscale preview with natural Hadamard or sequency-ordered Walsh basis. Reconstruct after retaining coefficients.', 'F = H f Hᵀ / N; Walsh reorders basis by sign changes',SOURCE,
           [
               choice('basis','Basis','Walsh sequency',['Walsh sequency','Hadamard natural']),
               choice('mode','Output','spectrum',['spectrum','reconstruction']),
               choice('size','Preview size','64',['16','32','64','128']),
               field('retain','Retained coefficient rows/cols (%)',100,1,100)
           ])
def walsh_hadamard(image, params, reference):
    size = int(params['size'])
    values = cv2.resize(gray(image), (size, size), interpolation=cv2.INTER_AREA).astype(float)
    basis = linalg.hadamard(size).astype(float) / np.sqrt(size)
    if params['basis'] == 'Walsh sequency':
        basis = basis[np.argsort(np.count_nonzero(np.diff(basis, axis=1), axis=1), kind='stable')]
    coefficients = basis @ values @ basis.T
    if params['mode'] == 'spectrum':
        return (
            normalize(np.log1p(np.abs(coefficients))),
            {'note': f'Input is resampled to {size}×{size} for the power-of-two transform.'}
        )
    keep = max(1, round(size * params['retain'] / 100))
    coefficients[keep:, :] = 0
    coefficients[:, keep:] = 0
    result = basis.T @ coefficients @ basis
    return (u8(result), {
        'reconstruction_mse': float(np.mean((result - values) ** 2)),
        'retained_coefficients': keep * keep,
        'note': f'MSE is against the {size}×{size} resampled grayscale input.'
    })


@operation('pca_klt','PCA / KLT / Hotelling block reconstruction',GROUP,'Learn an orthonormal basis from image patches and reconstruct using the leading principal components.', 'Centered patches → SVD → retain eigenvectors → inverse projection',SOURCE,
           [choice('block','Patch size','4',['4','8','16']),field('components','Retained components',4,1,256)])
def pca_klt(image, params, reference):
    values = gray(bounded(image, 256)).astype(float)
    block = int(params['block'])
    h, w = values.shape
    pad_h = -h % block
    pad_w = -w % block
    padded = np.pad(values, ((0, pad_h), (0, pad_w)), mode='edge')
    ph, pw = padded.shape
    patches = padded.reshape(ph // block, block, pw // block, block).transpose(0, 2, 1, 3).reshape(-1, block * block)
    mean = patches.mean(axis=0)
    centered = patches - mean
    _, singular, basis = np.linalg.svd(centered, full_matrices=False)
    keep = min(params['components'], len(basis))
    projected = centered @ basis[:keep].T
    reconstructed = projected @ basis[:keep] + mean
    result = reconstructed.reshape(ph // block, pw // block, block, block).transpose(0, 2, 1, 3).reshape(ph, pw)[:h, :w]
    energy = singular ** 2
    ratio = float(energy[:keep].sum() / energy.sum()) if energy.sum() else 1.0
    return (u8(result), {
        'retained_components': keep,
        'explained_variance_ratio': ratio,
        'reconstruction_mse': float(np.mean((values - result) ** 2)),
        'note': 'Basis is learned from patches of this image, limited to a 256-pixel longest side.'
    })


@operation('signal_demo','Impulse / sampling / Fourier basis demo',GROUP,'Generate a synthetic intensity signal to inspect impulse shifting, aliasing, Fourier series and low/high-pass response.', 'Nyquist: sampling frequency ≥ 2 × highest signal frequency',SOURCE,
           [
               choice(
                   'mode',
                   'Demonstration',
                   'aliasing',
                   ['aliasing','Fourier square-wave series','shifted impulse','1D low-pass','1D high-pass']
               ),
               field('frequency','Signal cycles',12,1,100),
               field('samples','Sampling points',32,4,256),
               field('terms','Fourier terms',5,1,30)
           ])
def signal_demo(image, params, reference):
    x = np.arange(512) / 512
    if params['mode'] == 'shifted impulse':
        values = np.zeros(512)
        values[params['frequency'] * 5 % 512] = 1
    elif params['mode'] == 'Fourier square-wave series':
        values = sum((np.sin(2 * np.pi * (2 * k + 1) * params['frequency'] * x) / (2 * k + 1) for k in range(params['terms']))) * 4 / np.pi
    else:
        values = np.sin(2 * np.pi * params['frequency'] * x)
        if params['mode'] == 'aliasing':
            indices = np.linspace(0, 511, params['samples'], endpoint=False).astype(int)
            values = np.interp(np.arange(512), indices, values[indices])
        elif params['mode'] == '1D low-pass':
            values = np.convolve(values, np.ones(15) / 15, mode='same')
        else:
            values -= np.convolve(values, np.ones(15) / 15, mode='same')
    strip = np.broadcast_to(u8(127.5 + 127.5 * values), (160, 512)).copy()
    return (strip, {
        'signal_cycles': params['frequency'],
        'sample_count': params['samples'],
        'nyquist_condition_met': params['samples'] >= 2 * params['frequency'],
        'note': 'Output is a synthetic 1D signal rendered as vertical intensity stripes; the uploaded photo is not used.'
    })


@operation(
    'points_3d',
    '3D scaling / rotation / translation demo',
    'Geometry',
    'Transform the vertices of a synthetic 3D cube and project its edges onto a 2D image.',
    'p′ = R(scale × p) + translation',
    SOURCE,
    [
        field('angle','Y-axis rotation (degrees)',30,-180,180),
        field('scale','Cube scale',1,.1,2,.1),
        field('translation','X translation',0,-1,1,.1)
    ]
)
def points_3d(image, params, reference):
    points = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]], float)
    angle = np.deg2rad(params['angle'])
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    transformed = points * params['scale'] @ rotation.T + np.array([params['translation'], 0, 0])
    output = np.full((420, 520, 3), 245, np.uint8)
    projection = np.column_stack((
        transformed[:, 0] + 0.3 * transformed[:, 2],
        transformed[:, 1] + 0.2 * transformed[:, 2]
    )) * 80 + np.array([260, 210])
    for a in range(8):
        for b in range(a + 1, 8):
            if np.count_nonzero(points[a] != points[b]) == 1:
                cv2.line(
                    output,
                    tuple(projection[a].astype(int)),
                    tuple(projection[b].astype(int)),
                    (70, 130, 45),
                    2
                )
    return (output, {
        'transformed_vertices': transformed.round(3).tolist(),
        'note': 'This is a synthetic point-geometry demonstration, not a reconstructed 3D medical volume.'
    })
