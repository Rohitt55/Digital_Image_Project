import cv2
import numpy as np
from scipy import ndimage
from skimage import exposure
from skimage.morphology import skeletonize

from code_02_image_operations import operation, field, choice, gray, normalize
from code_04_image_helpers import bounded, u8, reference_gray

SOURCE = 'Chap2 Image Enhancement.pdf'
GROUP = 'Filters & restoration'


@operation('log_transform', 'Log / inverse-log transformation', 'Color & intensity', 'Expand dark intensities with a logarithm, or emphasize bright intensities with its inverse.', 'log: s = 255 log(1+r)/log(256); inverse: s = 255(expm1(a r/255))/expm1(a)', SOURCE,
           [
               choice('mode','Transform','log',['log','inverse-log']),
               field('strength','Inverse-log strength',3,.1,8,.1)
           ])
def log_transform(image, params, reference):
    values = image.astype(float)
    return u8(255 * np.log1p(values) / np.log(256)) if params['mode'] == 'log' else u8(255 * np.expm1(params['strength'] * values / 255) / np.expm1(params['strength']))


@operation('piecewise', 'Piecewise intensity transformation', 'Color & intensity', 'Choose two control points for contrast stretching, intensity slicing or binary slicing.', 'Piecewise linear mapping through (0,0), (r1,s1), (r2,s2), (255,255)', SOURCE,
           [
               choice('mode','Mode','stretch',['stretch','slice-keep-background','slice-binary']),
               field('r1','Lower input',60,1,253),
               field('r2','Upper input',180,2,254),
               field('s1','Lower output',20,0,255),
               field('s2','Upper output',230,0,255)
           ])
def piecewise(image, params, reference):
    if params['r1'] >= params['r2']:
        raise ValueError('Lower input must be less than upper input.')
    values = gray(image)
    if params['mode'] == 'stretch':
        return u8(np.interp(values, [0, params['r1'], params['r2'], 255], [0, params['s1'], params['s2'], 255]))
    mask = (values >= params['r1']) & (values <= params['r2'])
    return np.where(mask, 255, values if params['mode'] == 'slice-keep-background' else 0).astype(np.uint8)


@operation('bit_planes', 'Bit-plane slicing', 'Color & intensity', 'Inspect one bit of each grayscale pixel or reconstruct the highest selected bits.', 'Plane b = ((r >> b) & 1) × 255', SOURCE,
           [
               choice('mode','Display','single plane',['single plane','reconstruct high bits']),
               field('bit','Bit / lowest retained bit',7,0,7)
           ])
def bit_planes(image, params, reference):
    values = gray(image)
    if params['mode'] == 'single plane':
        return (values >> params['bit'] & 1) * 255
    return values >> params['bit'] << params['bit']


@operation(
    'ahe',
    'Adaptive histogram equalization · AHE',
    'Histograms',
    'Equalize local histograms without contrast clipping. Compare with CLAHE to see noise amplification.',
    'Local CDF mapping without histogram clipping',
    SOURCE,
    [field('kernel','Local window size (odd)',31,3,127)]
)
def ahe(image, params, reference):
    from skimage.filters.rank import equalize
    values = gray(image)
    return equalize(values, footprint=np.ones((params['kernel'], params['kernel']), np.uint8))


@operation(
    'histogram_specification',
    'Histogram specification / matching',
    'Histograms',
    'Map the source grayscale CDF to the nearest reference grayscale CDF.',
    's = CDFreference⁻¹(CDFsource(r))',
    SOURCE,
    requires_reference=True
)
def histogram_specification(image, params, reference):
    if reference is None:
        raise ValueError('Upload a reference image.')
    return u8(exposure.match_histograms(gray(image), gray(reference)))


@operation('order_filters', 'Min / max / midpoint / alpha-trimmed filters', GROUP, 'Use order statistics to remove impulse noise or estimate neighborhood extremes.', 'Min, max, (min+max)/2, or trimmed neighborhood mean', SOURCE,
           [
               choice('method','Filter','midpoint',['minimum','maximum','midpoint','alpha-trimmed']),
               field('kernel','Kernel size (odd)',3,3,9),
               field('trim','Values discarded from each end',1,0,10)
           ])
def order_filters(image,params,reference):
    values=gray(image)
    low=ndimage.minimum_filter(values,size=params['kernel'],mode='reflect')
    high=ndimage.maximum_filter(values,size=params['kernel'],mode='reflect')
    if params['method']=='minimum': return low
    if params['method']=='maximum': return high
    if params['method']=='midpoint': return u8((low.astype(float)+high)/2)
    if 2*params['trim']>=params['kernel']**2: raise ValueError('Trim must leave at least one neighborhood value.')
    pad=params['kernel']//2
    padded=np.pad(values,pad,mode='symmetric')
    windows=np.lib.stride_tricks.sliding_window_view(padded,(params['kernel'],params['kernel']))
    # Process rows to bound the temporary sorted-neighborhood allocation.
    result=np.empty(values.shape,np.uint8)
    trim=params['trim']
    for y in range(values.shape[0]):
        ordered=np.sort(windows[y].reshape(values.shape[1],-1),axis=-1)
        result[y]=u8(ordered[:,trim:ordered.shape[1]-trim if trim else None].mean(axis=-1))
    return result


@operation('convolution', 'Custom convolution / correlation · FIR', GROUP, 'Apply a 3×3 numeric kernel. Convolution flips the kernel; correlation keeps its orientation. No code is evaluated.', 'Correlation: Σ k(a,b) f(x+a,y+b); convolution flips k', SOURCE,
           [
               choice('mode','Operation','correlation',['correlation','convolution']),
               dict(
                   key='coefficients',
                   label='Nine kernel values (comma separated)',
                   default='0,-1,0,-1,5,-1,0,-1,0',
                   type='text'
               ),
               choice('display','Display','clip',['clip','normalize'])
           ])
def convolution(image, params, reference):
    try:
        values = np.array([float(x.strip()) for x in params['coefficients'].split(',')], np.float32)
    except ValueError as exc:
        raise ValueError('Kernel must contain nine numeric values.') from exc
    if values.size != 9 or not np.all(np.isfinite(values)) or np.any(np.abs(values) > 100):
        raise ValueError('Kernel needs nine finite values between -100 and 100.')
    kernel = values.reshape(3, 3)
    if params['mode'] == 'convolution':
        kernel = kernel[::-1, ::-1]
    result = cv2.filter2D(gray(image), cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
    return normalize(result) if params['display'] == 'normalize' else u8(result)


@operation('laplacian', 'Laplacian sharpening / edges', GROUP, 'Use the second spatial derivative for edge response or sharpening.', '∇²f = ∂²f/∂x² + ∂²f/∂y²; sharpen = f - amount × ∇²f', SOURCE,
           [
               choice('mode','Output','sharpen',['sharpen','edge magnitude']),
               field('amount','Laplacian amount',.5,.1,3,.1)
           ])
def laplacian(image, params, reference):
    values = gray(image).astype(np.float32)
    response = cv2.Laplacian(values, cv2.CV_32F, ksize=3)
    return normalize(np.abs(response)) if params['mode'] == 'edge magnitude' else u8(values - params['amount'] * response)


@operation('ensemble_average', 'Ensemble averaging · noise reduction', GROUP, 'Generate independent noisy observations and average them with a fixed random seed.', 'Var(mean of N independent noise samples) = σ²/N', SOURCE,
           [
               field('frames','Number of observations',8,2,64),
               field('sigma','Noise standard deviation',20,1,80),
               field('seed','Random seed',42,0,9999)
           ])
def ensemble_average(image, params, reference):
    rng = np.random.default_rng(params['seed'])
    total = np.zeros(image.shape, float)
    for _ in range(params['frames']):
        total += image.astype(float) + rng.normal(0, params['sigma'], image.shape)
    result = u8(total / params['frames'])
    return (result, {
        'theoretical_noise_std': float(params['sigma'] / np.sqrt(params['frames'])),
        'observed_residual_std': float((result.astype(float) - image).std()),
        'note': 'Synthetic noise is added before averaging; the uploaded image is the known clean target.'
    })


@operation('frequency_filters', 'Ideal / Butterworth / Gaussian frequency filters', 'Transforms & comparison', 'Choose low-pass, high-pass, band-pass or band-reject transfer functions in the centered DFT.', 'G(u,v) = H(u,v) F(u,v); output = IDFT(G)', SOURCE,
           [
               choice('family','Transfer function','Gaussian',['Ideal','Butterworth','Gaussian']),
               choice('mode','Pass type','low-pass',['low-pass','high-pass','band-pass','band-reject']),
               field('cutoff','Lower cutoff (%)',10,1,45),
               field('upper','Upper cutoff (%)',25,2,50),
               field('order','Butterworth order',2,1,10)
           ])
def frequency_filters(image, params, reference):
    values = gray(image)
    h, w = values.shape
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - w // 2) ** 2 + (y - h // 2) ** 2)

    def low(radius):
        radius = min(h, w) * radius / 100
        if params['family'] == 'Ideal':
            return (distance <= radius).astype(float)
        if params['family'] == 'Butterworth':
            return 1 / (1 + (distance / radius) ** (2 * params['order']))
        return np.exp(-distance ** 2 / (2 * radius ** 2))
    transfer = low(params['cutoff'])
    if params['mode'] == 'high-pass':
        transfer = 1 - transfer
    elif params['mode'] in ['band-pass', 'band-reject']:
        if params['cutoff'] >= params['upper']:
            raise ValueError('Lower cutoff must be below upper cutoff.')
        lower = min(h, w) * params['cutoff'] / 100
        upper = min(h, w) * params['upper'] / 100
        center = (lower + upper) / 2
        width = upper - lower
        if params['family'] == 'Ideal':
            reject = ((distance < lower) | (distance > upper)).astype(float)
        elif params['family'] == 'Butterworth':
            denominator = distance ** 2 - center ** 2
            ratio = np.divide(
                distance * width,
                denominator,
                out=np.full_like(distance, 1000000000000.0),
                where=np.abs(denominator) > 1e-12
            )
            reject = 1 / (1 + np.abs(ratio) ** (2 * params['order']))
        else:
            ratio = (distance ** 2 - center ** 2) / np.maximum(distance * width, 1e-12)
            reject = 1 - np.exp(-ratio ** 2)
        transfer = reject if params['mode'] == 'band-reject' else 1 - reject
    result = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(values)) * transfer)).real
    return (
        normalize(np.abs(result)) if params['mode'] in ['high-pass', 'band-pass'] else u8(result),
        {'note': 'High-pass and band-pass results show normalized response magnitude; low-pass/reject retain the intensity scale.'}
    )


@operation('homomorphic', 'Homomorphic illumination correction', GROUP, 'Separate illumination and reflectance approximately using log intensity and a high-frequency emphasis filter.', 'g = exp(IDFT(H × DFT(log(1+f)))) - 1', SOURCE,
           [
               field('low_gain','Low-frequency gain',.5,.1,1,.1),
               field('high_gain','High-frequency gain',1.5,1,3,.1),
               field('cutoff','Cutoff (%)',10,1,50)
           ])
def homomorphic(image, params, reference):
    values = np.log1p(gray(image).astype(float))
    h, w = values.shape
    y, x = np.ogrid[:h, :w]
    distance = (x - w // 2) ** 2 + (y - h // 2) ** 2
    transfer = params['low_gain'] + (params['high_gain'] - params['low_gain']) * (1 - np.exp(-distance / (2 * (min(h, w) * params['cutoff'] / 100) ** 2)))
    result = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(values)) * transfer)).real
    return normalize(np.expm1(np.clip(result, -20, 20)))


@operation(
    'fingerprint_pipeline',
    'Fingerprint enhancement pipeline',
    GROUP,
    'Demonstrate contrast enhancement, denoising, local binarization and ridge thinning.',
    'CLAHE → Gaussian → adaptive threshold → skeleton',
    SOURCE
)
def fingerprint_pipeline(image, params, reference):
    values = gray(bounded(image))
    enhanced = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(values)
    smooth = cv2.GaussianBlur(enhanced, (3, 3), 0)
    binary = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
    return (
        skeletonize(binary > 0).astype(np.uint8) * 255,
        {'note': 'This is an enhancement/thinning demonstration, not fingerprint identity recognition.'}
    )


@operation(
    'separable_filter',
    'Separable / weighted-mean kernel filtering',
    GROUP,
    'Compare full 2D filtering with horizontal-then-vertical filtering for a separable kernel.',
    'K2D = ky kxᵀ; full correlation equals two 1D passes',
    SOURCE,
    [
        choice('type','Kernel','Gaussian',['Gaussian','box','weighted binomial']),
        field('kernel','Kernel size (odd)',5,3,15)
    ]
)
def separable_filter(image, params, reference):
    from scipy.special import comb
    size = params['kernel']
    if params['type'] == 'Gaussian':
        vector = cv2.getGaussianKernel(size, 0).ravel().astype(np.float32)
    elif params['type'] == 'box':
        vector = np.ones(size, np.float32) / size
    else:
        vector = np.array([comb(size - 1, i) for i in range(size)], np.float32)
        vector /= vector.sum()
    values = gray(image).astype(np.float32)
    separated = cv2.sepFilter2D(values, cv2.CV_32F, vector, vector)
    full = cv2.filter2D(values, cv2.CV_32F, np.outer(vector, vector))
    return (u8(separated), {
        'maximum_full_separable_error': float(np.max(np.abs(full - separated))),
        'kernel_vector': vector.tolist()
    })


@operation(
    'fingerprint_orientation',
    'Fingerprint ridge orientation / minutiae demo',
    GROUP,
    'Estimate local ridge orientation from a smoothed gradient structure tensor or mark skeleton endpoints and branch candidates.',
    'orientation = ½ atan2(2GxGy,Gx²-Gy²)+π/2',
    SOURCE,
    [choice('mode','Output','orientation',['orientation','minutiae candidates'])]
)
def fingerprint_orientation(image, params, reference):
    values = gray(bounded(image, 512))
    x = cv2.Sobel(values, cv2.CV_32F, 1, 0)
    y = cv2.Sobel(values, cv2.CV_32F, 0, 1)
    a = cv2.GaussianBlur(2 * x * y, (15, 15), 3)
    b = cv2.GaussianBlur(x * x - y * y, (15, 15), 3)
    angle = 0.5 * np.arctan2(a, b) + np.pi / 2
    if params['mode'] == 'orientation':
        return (
            u8(angle % np.pi / np.pi * 255),
            {'note': 'Normalized ridge orientation map; it does not compare fingerprint identities.'}
        )
    binary = skeletonize(cv2.adaptiveThreshold(
        values,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        3
    ) > 0)
    neighbors = ndimage.convolve(
        binary.astype(np.uint8),
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8),
        mode='constant'
    )
    endpoints = binary & (neighbors == 1)
    branches = binary & (neighbors >= 3)
    output = cv2.cvtColor(binary.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    output[endpoints] = (0, 0, 255)
    output[branches] = (0, 255, 0)
    return (output, {
        'endpoint_candidates': int(endpoints.sum()),
        'branch_candidates': int(branches.sum()),
        'note': 'Raw neighbor-count candidates can include artifacts; no fingerprint identification is performed.'
    })
