import cv2
import numpy as np
from scipy import ndimage, stats
from skimage import feature, measure, restoration, transform

from code_02_image_operations import operation, field, choice, gray, normalize
from code_04_image_helpers import bounded, u8

SOURCE='Chap6-Image Restoration.pdf'
GROUP='Filters & restoration'


@operation('noise_models','Gaussian / uniform / exponential / gamma / Rayleigh / Poisson / speckle / impulse noise',GROUP,'Generate seeded samples from eight noise families. Additive distributions are centered to avoid an unintended brightness shift.', 'g=f+n (additive); g=f+f n (speckle); Poisson models photon counts',SOURCE,
           [
               choice(
                   'model',
                   'Noise model',
                   'Gaussian',
                   ['Gaussian','Uniform','Exponential','Gamma','Rayleigh','Poisson','Speckle','Salt and pepper']
               ),
               field('sigma','Noise standard deviation',20,1,100),
               field('amount','Impulse fraction',.05,.001,.5,.001),
               field('peak','Poisson peak photon count',30,1,255),
               field('seed','Random seed',42,0,9999)
           ])
def noise_models(image, params, reference):
    values = image.astype(float)
    rng = np.random.default_rng(params['seed'])
    sigma = params['sigma']
    model = params['model']
    if model == 'Gaussian':
        noise = rng.normal(0, sigma, image.shape)
    elif model == 'Uniform':
        noise = rng.uniform(-np.sqrt(3) * sigma, np.sqrt(3) * sigma, image.shape)
    elif model == 'Exponential':
        noise = rng.exponential(sigma, image.shape) - sigma
    elif model == 'Gamma':
        noise = rng.gamma(2, sigma / np.sqrt(2), image.shape) - np.sqrt(2) * sigma
    elif model == 'Rayleigh':
        scale = sigma / np.sqrt((4 - np.pi) / 2)
        noise = rng.rayleigh(scale, image.shape) - scale * np.sqrt(np.pi / 2)
    elif model == 'Poisson':
        noise = rng.poisson(values / 255 * params['peak']) / params['peak'] * 255 - values
    elif model == 'Speckle':
        noise = values * rng.normal(0, sigma / 255, image.shape)
    else:
        selector = rng.random(image.shape[:2])
        result = image.copy()
        result[selector < params['amount'] / 2] = 0
        result[(selector >= params['amount'] / 2) & (selector < params['amount'])] = 255
        noise = result.astype(float) - values
    result = u8(values + noise)
    signal_power = float(np.mean(values ** 2))
    noise_power = float(np.mean(noise ** 2))
    return (result, {
        'unclipped_noise_mean': float(noise.mean()),
        'unclipped_noise_std': float(noise.std()),
        'snr_db': float(10 * np.log10(signal_power / noise_power)) if signal_power and noise_power else 'undefined',
        'note': 'Random seed makes this synthetic degradation repeatable. Noise statistics are measured before intensity clipping.'
    })


@operation('probability_models','Gaussian / half-normal / exponential PDF & CDF','Fundamentals','Draw analytical probability density and cumulative distribution curves for a chosen distribution.', 'CDF(x)=∫ PDF(t)dt; distributions are normalized probability models','Chap1 Digital Image Fundamentals.pdf / '+SOURCE,
           [
               choice(
                   'distribution',
                   'Distribution',
                   'Gaussian',
                   ['Gaussian','Standard normal','Half-normal','Exponential','Gamma','Rayleigh','Uniform']
               ),
               field('scale','Scale',1,.1,5,.1)
           ])
def probability_models(image, params, reference):
    model = params['distribution']
    scale = params['scale']
    x = np.linspace(-4 * scale, 6 * scale, 512)
    distributions = {
        'Gaussian': stats.norm(scale=scale),
        'Standard normal': stats.norm(),
        'Half-normal': stats.halfnorm(scale=scale),
        'Exponential': stats.expon(scale=scale),
        'Gamma': stats.gamma(a=2, scale=scale),
        'Rayleigh': stats.rayleigh(scale=scale),
        'Uniform': stats.uniform(loc=-scale, scale=2 * scale)
    }
    distribution = distributions[model]
    pdf = distribution.pdf(x)
    cdf = distribution.cdf(x)
    canvas = np.full((360, 600, 3), 250, np.uint8)
    for index, (values, title, color) in enumerate([
        (pdf, 'Probability density', (180, 90, 30)),
        (cdf, 'Cumulative probability', (50, 145, 55))
    ]):
        top = 30 + index * 165
        cv2.putText(canvas, title, (45, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
        maximum = max(float(values.max()), 1e-12)
        points = np.column_stack((np.linspace(45, 570, 512), top + 130 - values / maximum * 110)).astype(np.int32)
        cv2.polylines(canvas, [points], False, color, 2)
        cv2.line(canvas, (45, top + 130), (570, top + 130), (130, 130, 130), 1)
    return (canvas, {
        'distribution_mean': float(distribution.mean()),
        'distribution_variance': float(distribution.var()),
        'note': 'Synthetic analytical curves; the uploaded image is not used. Each curve has its own vertical display scale.'
    })


@operation('mean_restoration','Arithmetic / geometric / harmonic / contraharmonic mean',GROUP,'Apply neighborhood mean estimators suited to different noise distributions.', 'Geometric=exp(mean(log I)); harmonic=1/mean(1/I); contra=Σ I^(Q+1)/Σ I^Q',SOURCE,
           [
               choice('method','Estimator','geometric',['arithmetic','geometric','harmonic','contraharmonic']),
               field('kernel','Kernel size (odd)',3,3,15),
               field('q','Contraharmonic order Q',1.5,-2,2,.1)
           ])
def mean_restoration(image, params, reference):
    values = gray(image).astype(float)
    safe = np.maximum(values, 1e-06)
    size = params['kernel']
    if params['method'] == 'arithmetic':
        result = ndimage.uniform_filter(values, size=size, mode='reflect')
    elif params['method'] == 'geometric':
        result = np.exp(ndimage.uniform_filter(np.log(safe), size=size, mode='reflect'))
    elif params['method'] == 'harmonic':
        result = 1 / ndimage.uniform_filter(1 / safe, size=size, mode='reflect')
    else:
        result = ndimage.uniform_filter(safe ** (params['q'] + 1), size=size, mode='reflect') / np.maximum(
            ndimage.uniform_filter(safe ** params['q'], size=size, mode='reflect'),
            1e-12
        )
    return (
        u8(result),
        {'note': 'Zero intensities use a small positive epsilon for logarithms and negative powers.'}
    )


def psf(kernel, kind):
    if kind == 'motion':
        response = np.zeros((kernel, kernel))
        response[kernel // 2, :] = 1 / kernel
        return response
    g = cv2.getGaussianKernel(kernel, max(kernel / 6, 0.5))
    return g @ g.T


def otf(response, shape):
    kernel = np.zeros(shape, float)
    if response.shape[0] > shape[0] or response.shape[1] > shape[1]:
        raise ValueError('Blur kernel must fit inside the image.')
    kernel[:response.shape[0], :response.shape[1]] = response
    kernel = np.roll(kernel, (-(response.shape[0] // 2), -(response.shape[1] // 2)), axis=(0, 1))
    return np.fft.fft2(kernel)


BLUR_FIELDS=[
    choice('blur','Point-spread function','Gaussian',['Gaussian','motion']),
    field('kernel','Blur kernel size (odd)',5,3,31)
]


@operation(
    'degradation',
    'Image degradation · blur + noise',
    GROUP,
    'Apply a known blur transfer function and seeded Gaussian noise.',
    'G(u,v)=H(u,v)F(u,v)+N(u,v)',
    SOURCE,
    BLUR_FIELDS+[field('noise','Noise sigma',5,0,40),field('seed','Random seed',42,0,9999)]
)
def degradation(image, params, reference):
    values = gray(image).astype(float)
    transfer = otf(psf(params['kernel'], params['blur']), values.shape)
    result = np.fft.ifft2(np.fft.fft2(values) * transfer).real + np.random.default_rng(params['seed']).normal(0, params['noise'], values.shape)
    return (
        u8(result),
        {'note': 'Circular convolution with a known PSF is used so restoration assumptions are reproducible.'}
    )


@operation('inverse_wiener','Inverse / Wiener / minimum-MSE deconvolution',GROUP,'Restore using a known PSF. Demo mode first degrades the input; uploaded-degraded mode treats the input as the measured observation.', 'Inverse=G/H; Wiener=H*G/(|H|²+K)',SOURCE,BLUR_FIELDS+
           [
               choice('method','Restoration','Wiener',['inverse','Wiener']),
               choice(
                   'input_mode',
                   'Input interpretation',
                   'degrade then restore',
                   ['degrade then restore','uploaded degraded image']
               ),
               field('noise','Demo Gaussian noise sigma',3,0,40),
               field('regularization','Wiener K / inverse cutoff',.01,.0001,.5,.0001)
           ])
def inverse_wiener(image, params, reference):
    values = gray(bounded(image, 1024)).astype(float)
    transfer = otf(psf(params['kernel'], params['blur']), values.shape)
    observed = values
    if params['input_mode'] == 'degrade then restore':
        observed = np.fft.ifft2(np.fft.fft2(values) * transfer).real + np.random.default_rng(42).normal(0, params['noise'], values.shape)
    if params['method'] == 'inverse':
        inverse = np.divide(
            1,
            transfer,
            out=np.zeros_like(transfer),
            where=np.abs(transfer) >= params['regularization']
        )
    else:
        inverse = transfer.conj() / (np.abs(transfer) ** 2 + params['regularization'])
    restored = np.fft.ifft2(np.fft.fft2(observed) * inverse).real
    details = {'note': 'Known-PSF circular-convolution demo, limited to a 1024-pixel side. Wiener K approximates the noise-to-signal power ratio.'}
    if params['input_mode'] == 'degrade then restore':
        details.update(
            degraded_mse=float(np.mean((observed - values) ** 2)),
            restored_mse=float(np.mean((restored - values) ** 2))
        )
    return (u8(restored), details)


@operation(
    'adaptive_wiener',
    'Adaptive local Wiener denoising',
    GROUP,
    'Estimate local variance and suppress noise relative to the estimated noise power.',
    'output=local mean + max(var-noise,0)/max(var,ε) × (input-mean)',
    SOURCE,
    [field('kernel','Kernel size (odd)',5,3,31),field('noise','Noise variance (0 = estimate)',0,0,10000)]
)
def adaptive_wiener(image, params, reference):
    values = gray(image).astype(float)
    mean = ndimage.uniform_filter(values, size=params['kernel'])
    variance = np.maximum(ndimage.uniform_filter(values ** 2, size=params['kernel']) - mean ** 2, 0)
    noise = params['noise'] or float(variance.mean())
    gain = np.maximum(variance - noise, 0) / np.maximum(variance, 1e-12)
    return (u8(mean + gain * (values - mean)), {'estimated_noise_variance': noise})


@operation(
    'diffusion',
    'Anisotropic diffusion · Perona–Malik',
    GROUP,
    'Iteratively smooth weak gradients while preserving stronger edges.',
    'Inext=I+dt Σ conductance(gradient) × gradient',
    SOURCE,
    [
        field('iterations','Iterations',15,1,100),
        field('kappa','Edge sensitivity',20,1,100),
        field('time','Time step',.2,.01,.24,.01)
    ]
)
def diffusion(image, params, reference):
    values = gray(bounded(image, 512)).astype(float)
    for _ in range(params['iterations']):
        padded = np.pad(values, 1, mode='edge')
        directions = [
            padded[:-2, 1:-1] - values,
            padded[2:, 1:-1] - values,
            padded[1:-1, :-2] - values,
            padded[1:-1, 2:] - values
        ]
        values += params['time'] * sum((np.exp(-(d / params['kappa']) ** 2) * d for d in directions))
    return u8(values)


@operation(
    'nonlocal_means',
    'Nonlocal means denoising',
    GROUP,
    'Denoise using patch similarity rather than only immediate neighbors.',
    'output pixel = weighted mean of pixels with similar patches',
    SOURCE,
    [field('strength','Filter strength',10,1,40)]
)
def nonlocal_means(image, params, reference):
    values = gray(bounded(image, 512))
    return cv2.fastNlMeansDenoising(values, None, params['strength'], 7, 21)


@operation(
    'wavelet_denoise',
    'Wavelet noise / speckle denoising',
    GROUP,
    'Shrink noisy wavelet detail coefficients. Optional log-domain processing handles multiplicative speckle.',
    'wavelet thresholding; speckle: log → denoise → exp',
    SOURCE,
    [
        choice('noise','Noise interpretation','additive',['additive','speckle']),
        choice('wavelet','Wavelet','db4',['haar','db4','db8']),
        field('sigma','Noise sigma (intensity units)',15,1,80)
    ]
)
def wavelet_denoise(image, params, reference):
    values = gray(bounded(image, 1024)).astype(float) / 255
    if params['noise'] == 'speckle':
        log = np.log(np.maximum(values, 1 / 255))
        result = np.exp(restoration.denoise_wavelet(
            log,
            sigma=params['sigma'] / 255,
            wavelet=params['wavelet'],
            method='BayesShrink',
            rescale_sigma=True
        ))
    else:
        result = restoration.denoise_wavelet(
            values,
            sigma=params['sigma'] / 255,
            wavelet=params['wavelet'],
            method='BayesShrink',
            rescale_sigma=True
        )
    return u8(result * 255)


@operation(
    'glcm',
    'GLCM texture features',
    GROUP,
    'Quantize grayscale levels and calculate co-occurrence texture descriptors.',
    'P(i,j) counts gray-level pairs at a distance and angle',
    SOURCE,
    [
        field('distance','Pixel distance',1,1,10),
        choice('angle','Pair angle','0',['0','45','90','135']),
        choice('levels','Quantization levels','16',['8','16','32','64'])
    ]
)
def glcm(image, params, reference):
    values = gray(bounded(image, 512))
    levels = int(params['levels'])
    quantized = (values.astype(np.uint16) * levels // 256).astype(np.uint8)
    matrix = feature.graycomatrix(
        quantized,
        [params['distance']],
        [np.deg2rad(int(params['angle']))],
        levels=levels,
        symmetric=True,
        normed=True
    )
    details = {key: float(feature.graycoprops(matrix, key)[0, 0]) for key in [
        'contrast',
        'dissimilarity',
        'homogeneity',
        'energy',
        'correlation',
        'ASM'
    ]}
    details['note'] = 'The displayed co-occurrence matrix is normalized for visibility; descriptors use the normalized probabilities.'
    return (normalize(matrix[:, :, 0, 0]), details)


@operation(
    'region_features',
    'Shape / region feature extraction',
    GROUP,
    'Measure binary connected regions: area, perimeter, centroid, eccentricity and bounding box.',
    'Shape descriptors from connected foreground labels',
    SOURCE,
    [field('threshold','Foreground threshold',127,0,255),field('minimum','Minimum region area',10,1,10000)]
)
def region_features(image, params, reference):
    source = bounded(image)
    labels = measure.label(gray(source) > params['threshold'])
    properties = measure.regionprops(labels)
    output = source.copy()
    rows = []
    for region in properties:
        if region.area < params['minimum']:
            continue
        y0, x0, y1, x1 = region.bbox
        cv2.rectangle(output, (x0, y0), (x1 - 1, y1 - 1), (0, 230, 20), 1)
        if len(rows) < 30:
            rows.append({
                'area': int(region.area),
                'perimeter': float(region.perimeter),
                'centroid': list(region.centroid),
                'eccentricity': float(region.eccentricity),
                'bbox': list(region.bbox)
            })
    return (output, {
        'regions': len([r for r in properties if r.area >= params['minimum']]),
        'first_30_regions': rows,
        'note': 'Feature extraction is limited to a 512-pixel side; areas refer to that analysis resolution.'
    })


@operation('ct_reconstruction','CT Radon / backprojection / iterative reconstruction',GROUP,'Treat the grayscale image as a synthetic attenuation phantom, project it, then inspect or reconstruct the sinogram.', 'Sinogram=Radon(phantom); inverse by unfiltered, filtered or SART reconstruction',SOURCE,
           [
               choice(
                   'mode',
                   'Output',
                   'filtered backprojection',
                   ['sinogram','unfiltered backprojection','filtered backprojection','SART']
               ),
               field('angles','Projection angles',60,10,180),
               choice('filter','FBP filter','ramp',['ramp','shepp-logan','cosine','hamming','hann']),
               field('iterations','SART iterations',2,1,5)
           ])
def ct_reconstruction(image, params, reference):
    values = gray(cv2.resize(bounded(image, 128), (128, 128), interpolation=cv2.INTER_AREA)).astype(float) / 255
    theta = np.linspace(0, 180, params['angles'], endpoint=False)
    sinogram = transform.radon(values, theta=theta, circle=False)
    if params['mode'] == 'sinogram':
        return (
            normalize(sinogram),
            {'note': 'Columns are projection angles; rows are detector positions. This is a synthetic phantom, not clinical CT data.'}
        )
    if params['mode'] == 'SART':
        result = None
        for _ in range(params['iterations']):
            result = transform.iradon_sart(sinogram, theta=theta, image=result)
        center = (result.shape[0] - 128) // 2
        result = result[center:center + 128, center:center + 128]
    else:
        result = transform.iradon(
            sinogram,
            theta=theta,
            filter_name=None if params['mode'] == 'unfiltered backprojection' else params['filter'],
            circle=False,
            output_size=128
        )
    return (u8(result * 255), {
        'reconstruction_mse': float(np.mean((values - result) ** 2)),
        'note': '128×128 synthetic CT demonstration; it does not process calibrated clinical Hounsfield units.'
    })


@operation('mri_kspace','MRI k-space sampling / reconstruction',GROUP,'Generate synthetic k-space from grayscale and reconstruct after keeping a central band or uniformly spaced lines.', 'k-space=FFT2(image); reconstruction=|IFFT2(mask × k-space)|',SOURCE,
           [
               choice('mode','Output','reconstruction',['k-space magnitude','reconstruction']),
               choice('sampling','Sampling pattern','central band',['central band','every nth line']),
               field('retain','Central band retained (%)',50,1,100),
               field('stride','Line stride',2,1,8)
           ])
def mri_kspace(image, params, reference):
    values = gray(bounded(image, 512))
    spectrum = np.fft.fftshift(np.fft.fft2(values))
    mask = np.zeros(values.shape, bool)
    if params['sampling'] == 'central band':
        h = values.shape[0]
        keep = max(1, round(h * params['retain'] / 100))
        start = (h - keep) // 2
        mask[start:start + keep] = True
    else:
        mask[::params['stride']] = True
    masked = spectrum * mask
    if params['mode'] == 'k-space magnitude':
        return (
            normalize(np.log1p(np.abs(masked))),
            {'note': 'Synthetic k-space log magnitude; the upload is treated as the desired image, not acquired MRI measurements.'}
        )
    result = np.abs(np.fft.ifft2(np.fft.ifftshift(masked)))
    return (u8(result), {
        'retained_samples': int(mask.sum()),
        'total_samples': int(mask.size),
        'reconstruction_mse': float(np.mean((result - values) ** 2)),
        'note': 'Synthetic MRI sampling demo; missing k-space lines can cause blur or aliasing.'
    })


@operation(
    'window_level',
    'CT window / level intensity mapping',
    GROUP,
    'Demonstrate window/level display on 8-bit intensities. The upload has no calibrated Hounsfield units.',
    'display=clip((intensity-(level-width/2))/width,0,1) × 255',
    SOURCE,
    [field('level','Window center (0–255)',127,0,255),field('width','Window width',120,1,510)]
)
def window_level(image, params, reference):
    return (
        u8((gray(image).astype(float) - (params['level'] - params['width'] / 2)) / params['width'] * 255),
        {'note': 'Educational 8-bit windowing, not DICOM/Hounsfield-unit windowing.'}
    )


@operation(
    'phase_unwrap',
    'Phase wrapping / unwrapping demo',
    GROUP,
    'Map grayscale to a synthetic smooth phase field, wrap into −π…π and unwrap with a spatial algorithm.',
    'wrapped=angle(exp(j phase)); unwrap recovers continuous phase up to 2π offsets',
    SOURCE,
    [field('cycles','Phase range in cycles',3,1,8),choice('mode','Output','unwrapped',['wrapped','unwrapped'])]
)
def phase_unwrap(image, params, reference):
    values = cv2.GaussianBlur(gray(bounded(image, 256)).astype(float), (9, 9), 2) / 255 * 2 * np.pi * params['cycles']
    wrapped = np.angle(np.exp(1j * values))
    unwrapped = restoration.unwrap_phase(wrapped)
    return (
        normalize(wrapped if params['mode'] == 'wrapped' else unwrapped),
        {'note': 'Synthetic phase derived from intensity; this is not measured MRI phase and the displayed phase is normalized.'}
    )


@operation(
    'bias_correction',
    'Smooth multiplicative bias-field correction',
    GROUP,
    'Estimate a smooth illumination/bias field and divide it from the image. This is a simple homomorphic-style estimator, not N4.',
    'corrected = image / smooth(image) × mean(smooth(image))',
    SOURCE,
    [field('sigma','Bias field smoothing sigma',30,5,100)]
)
def bias_correction(image, params, reference):
    values = gray(image).astype(float)
    bias = ndimage.gaussian_filter(values, params['sigma'])
    result = values / np.maximum(bias, 1) * max(float(bias.mean()), 1)
    return (
        u8(result),
        {'note': 'Assumes slow spatial bias variation. It does not implement clinical N4 bias correction.'}
    )
