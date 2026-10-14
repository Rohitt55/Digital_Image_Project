"""Build the reviewed chapter catalog from the source-page topic inventories."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python_code'))
from code_02_image_operations import OPERATIONS
import code_03_operation_loader

ALIAS={
 'digitization_demo':'concept_lab','eye_formation_demo':'concept_lab','perception_demo':'concept_lab',
 'gazenet_fc_demo':'fc_demo','image_metrics':'concept_lab','fps_demo':'concept_lab','pipeline_demo':'concept_lab',
 'palette':'indexed_palette','features':'feature_vectors','digit_classifier':'digit_rules','shape_features':'region_features',
 'distribution_demo':'probability_models','noise':'noise_models','parametric_classifier':'gaussian_classifier',
 'distance':'shortest_path','digital_depth':'distance_boundary','connectivity_demo':'connectivity','components':'connectivity',
 'connectivity_3d_demo':'concept_lab','linearity_demo':'concept_lab','illumination':'image_formation','sampling':'resize',
 'kmeans_quantization':'lloyd_quantization','dark_region':'threshold','histogram_match':'histogram_specification',
 'custom_kernel':'convolution','kernel_support_demo':'convolution','lti_demo':'concept_lab','fir_demo':'convolution',
 'filter_taxonomy_demo':'order_filters','derivatives':'gradient_direction','rank_filter':'order_filters',
 'signal_filter_demo':'signal_demo','frequency_filter':'frequency_filters','fingerprint_demo':'fingerprint_orientation',
 'fingerprint_enhance':'fingerprint_pipeline','frequency_emphasis':'homomorphic','color_fundamentals_demo':'spectral_demo',
 'color_space':'color_models','hue_saturation':'hsv_adjust','color_segment':'color_segmentation','color_filter':'color_processing',
 'color_edges':'color_processing','chroma_subsample':'chroma_subsampling','color_morphology':'morphology_full',
 'density_slice':'pseudocolor','color_transform_demo':'color_models','color_slice':'color_slicing','color_tone':'color_sharpen',
 'color_laplacian':'color_sharpen','transform_intro_demo':'fourier_components','spatial_operations_demo':'concept_lab',
 'translate':'affine','rotate_affine':'affine','point3d_demo':'points_3d','identity':'histogram','piecewise_stretch':'piecewise',
 'intensity_slice':'piecewise','bit_plane':'bit_planes','impulse_demo':'signal_demo','sampling_demo':'signal_demo',
 'fourier_series_demo':'signal_demo','transform_kernel_demo':'separable_filter','walsh':'walsh_hadamard',
 'hadamard':'walsh_hadamard','fwht_demo':'walsh_hadamard','klt':'pca_klt','energy_compaction_demo':'pca_klt'}

MAP={
 'Dice coefficient':'segmentation_metrics','Intersection over union (Jaccard)':'segmentation_metrics','Pixel accuracy':'segmentation_metrics','Precision':'segmentation_metrics','Recall':'segmentation_metrics',
 'Manual global thresholding':'threshold','Iterative global thresholding (ISODATA)':'isodata','Adaptive mean thresholding':'adaptive_mean','Lane detection pipeline':'hough_lines','Adaptive Gaussian thresholding':'adaptive',
 'Local binary pattern (LBP)':'lbp_lpg','Local patterns of gradients (LPG)':'lbp_lpg','Connected-component object counting':'connectivity','Otsu optimum global thresholding':'otsu','Point detection':'point_line','Directional line detection':'point_line','Hough line transform':'hough_lines','Edge linking and hysteresis':'gradient_direction','Sobel edge detection':'sobel','Prewitt edge detection':'prewitt','Roberts cross edge detection':'roberts','Canny edge detection':'canny','Laplacian edge detection':'laplacian','Laplacian of Gaussian / Marr-Hildreth':'log_dog','Difference of Gaussians':'log_dog','Gradient with thresholding':'log_dog','Seeded region growing':'region_growing','Feature-based K-means clustering':'kmeans_segmentation','Graph-search border tracing':'border_tracing','Adjacent region merging':'split_merge','Quadtree region splitting and merging':'split_merge','Watershed segmentation':'watershed','Fuzzy C-means':'fuzzy_cmeans','Active contours (snakes)':'active_contour','Level-set segmentation':'level_set',
 'Image degradation model':'degradation','Gaussian noise':'noise_models','Uniform noise':'noise_models','Exponential noise':'noise_models','Gamma (Erlang) noise':'noise_models','Rayleigh noise':'noise_models','Poisson photon noise':'noise_models','Speckle noise':'noise_models','Salt-and-pepper impulse noise':'noise_models','Noise statistics and signal-to-noise ratio':'noise_models','Median noise filter':'median','Arithmetic mean filter':'mean_restoration','Geometric mean filter':'mean_restoration','Harmonic mean filter':'mean_restoration','Contraharmonic mean filter':'mean_restoration','Inverse filtering':'inverse_wiener','Minimum-MSE Wiener deconvolution':'inverse_wiener','Radon transform / CT sinogram':'ct_reconstruction','Filtered backprojection / inverse Radon':'ct_reconstruction','MRI formation and k-space reconstruction':'mri_kspace','CT window/level mapping':'window_level','Shape/region feature extraction':'region_features','Adaptive local Wiener filtering':'adaptive_wiener','Anisotropic diffusion':'diffusion','Nonlocal means denoising':'nonlocal_means','GLCM texture features':'glcm','Wavelet speckle denoising':'wavelet_denoise',
 'Compression ratio':'lossless_coding','Relative redundancy / efficiency':'lossless_coding','Bit rate':'lossless_coding','Mean squared error':'quality','Peak signal-to-noise ratio':'quality','Predictive coding / DPCM':'predictive_coding','Run-length encoding':'lossless_coding','Huffman coding':'lossless_coding','Entropy calculation':'entropy','Arithmetic coding':'lossless_coding','Golomb coding':'golomb_coding','Rice coding (power-of-two Golomb)':'golomb_coding','Order-k exponential-Golomb coding':'golomb_coding','LZW dictionary coding':'lossless_coding','DCT transform coding (1D and 2D)':'dct','JPEG DCT/quantization pipeline':'jpeg_pipeline','Zigzag coefficient ordering':'jpeg_pipeline','MPEG I/P frames and motion-compensated residuals':'motion_coding','H.263 DCT residual quantization':'motion_coding','Vector quantization':'vector_quantization',
 'Binary logical AND':'binary_logic','Binary logical OR':'binary_logic','Binary logical XOR':'binary_logic','Binary logical NOT':'binary_not','Binary logical NAND':'binary_logic','Binary logical NOR':'binary_logic','Binary erosion':'morphology_full','Grayscale erosion':'morphology_full','Binary dilation':'morphology_full','Grayscale dilation':'morphology_full','Binary and grayscale opening':'morphology_full','Binary and grayscale closing':'morphology_full','Hit-or-miss transform':'hit_miss','Morphological gradient':'morphology_full','White top-hat':'morphology_full','Black top-hat':'morphology_full','Morphological skeletonization / thinning':'skeleton','Skeleton pruning':'skeleton',
 # These are explicitly concept lessons, with related experiments rather than claimed implementations.
 'Segmentation fundamentals and processing workflow':'concept_lab','Segmentation method taxonomy':'threshold','Vehicle count from sensor pulses':'concept_lab','Gradient edge detection and direction':'gradient_direction','Bone fracture edge demonstration':'sobel','Nonmaximum suppression':'canny','Edge operator comparison':'sobel','Four/eight connectivity and region stopping':'region_growing',
 'CNN segmentation':'fc_demo','Fully convolutional networks (FCN)':'fc_demo','U-Net segmentation':'fc_demo','Mask R-CNN instance segmentation':'fc_demo','Additive versus multiplicative noise':'noise_models','X-ray, CT, MRI modality comparison':'ct_reconstruction','X-ray attenuation image formation':'image_formation','Fourier slice theorem':'ct_reconstruction','ART algebraic reconstruction':'ct_reconstruction','SIRT simultaneous iterative reconstruction':'ct_reconstruction','MLEM reconstruction':'ct_reconstruction','CT stroke worked example':'window_level','MRI tumor processing worked example':'level_set','Ultrasound processing workflow':'wavelet_denoise','MRI bias-field correction':'bias_correction','MRI phase unwrapping':'phase_unwrap','MRI Rician noise reduction':'rician_noise','Registration and MRI-CT fusion':'registration_fusion','Motion and metal artifact correction':'inverse_wiener','3D volume reconstruction/visualization':'points_3d','Classification algorithms (SVM, random forest, k-NN, CNN)':'digit_rules',
 'Compression fundamentals and storage budget':'concept_lab','Encoder/decoder model and lossless/lossy taxonomy':'lossless_coding','Coding redundancy and variable-length coding':'lossless_coding','Compression standards, formats and containers':'jpeg_pipeline','Morphology foundations':'morphology_full','Structuring elements, reflection and translation':'morphology_full','Image representation and description':'border_tracing','Lung CT transfer learning worked example':'fc_demo','DeepFace expression workflow worked example':'fc_demo'}

CHAPTERS={
    1:'Digital Image Fundamentals',
    2:'Image Enhancement',
    3:'Color Image Processing',
    4:'Image Transforms',
    5:'Image Segmentation',
    6:'Image Restoration',
    7:'Image Compression',
    8:'Morphological Processing'
}

LESSONS={
 'Human eye structure and image formation':'The cornea and lens focus incoming light onto the retina. Cones support color perception and fine detail, while rods are more sensitive in dim conditions. The optical image is inverted; neural processing is much richer than rotating a photo. The related experiment illustrates inversion only.',
 'Visual perception':'Perceived brightness depends on surrounding intensities, adaptation and contrast. In the simultaneous-contrast experiment both central patches have the same numeric value. Their apparent difference demonstrates why intensity measurements and subjective perception can differ.',
 'CNN GazeNet feature fusion and FC demo':'A gaze network can combine learned eye/face features and geometric information before fully connected prediction layers. Its weights must be trained with labeled data. The related demo only calculates a fixed untrained matrix layer, ReLU and softmax; its outputs are not gaze predictions.',
 'CNN segmentation':'A CNN learns local feature detectors from labeled training examples. Segmentation requires a prediction at each pixel, rather than one label for the whole image. The fixed fully connected demo illustrates arithmetic only; no trained segmentation model or weights are included.',
 'Fully convolutional networks (FCN)':'An FCN uses convolutional prediction and upsampling to produce dense pixel labels. Unlike a classifier with a fixed final vector, its spatial output retains image structure. The linked neural arithmetic demo is a basic building block, not an FCN implementation.',
 'U-Net segmentation':'U-Net combines a contracting encoder and expanding decoder with skip connections that preserve spatial detail. It is trained against pixel masks, often using Dice or cross-entropy losses. This catalog explains the architecture; it does not provide pretrained U-Net weights. Classical segmentation experiments and mask metrics are available separately.',
 'Mask R-CNN instance segmentation':'Mask R-CNN detects object regions and predicts a separate mask for each instance. This differs from assigning one semantic class to every pixel. A pretrained detector needs model weights, dependencies and data; no trained Mask R-CNN is bundled. Watershed illustrates separating classical instances without semantic labels.',
 'ART algebraic reconstruction':'ART represents projection measurements as a linear system Ax=b and sequentially updates the image using individual measurement residuals. It is sensitive to noise and relaxation settings. The available iterative CT experiment runs SART from scikit-image, a related algorithm; it does not claim to be sequential ART.',
 'SIRT simultaneous iterative reconstruction':'SIRT updates an image from a collection of projection residuals simultaneously, usually with sensitivity normalization. It differs from ART and SART in update organization. The related CT demo provides filtered backprojection and SART. Full SIRT is described here rather than mislabeled as SART.',
 'MLEM reconstruction':'Maximum-likelihood expectation-maximization uses a forward model and a statistical measurement model, commonly Poisson counts. A typical multiplicative update combines a backprojected measurement-to-prediction ratio with sensitivity normalization. The current CT demo illustrates projection and reconstruction geometry; it does not implement a calibrated MLEM scanner model.',
 'MRI bias-field correction':'A slowly varying multiplicative field can make identical tissues appear to have different intensities. Bias correction estimates that field and compensates for it. The runnable example divides by a smoothed intensity field; it is a simple teaching estimator, not the N4 clinical algorithm.',
 'MRI phase unwrapping':'Measured phase is modulo 2π. Spatial unwrapping seeks a continuous field consistent with wrapped observations. The experiment constructs synthetic phase from grayscale intensity, wraps it, then uses spatial phase unwrapping. It does not interpret the input as acquired MRI phase.',
 'Registration and MRI-CT fusion':'Multimodal registration aligns images before fusion. A clinical system may need affine/deformable transforms, physical coordinates and modality-aware metrics. The runnable example estimates translation using phase correlation and overlays the reference; it is a generic two-image demo.',
 'Motion and metal artifact correction':'Motion and metal alter measurement physics and can create structured reconstruction artifacts. Real correction needs the relevant acquisition model and often projection data. Known-PSF deconvolution illustrates restoration principles but does not remove arbitrary clinical motion or metal artifacts.',
 'Classification algorithms (SVM, random forest, k-NN, CNN)':'SVM learns decision boundaries, random forests combine decision trees, k-NN votes among nearby feature vectors, and CNNs learn image features. Training and evaluation require labeled data and held-out examples. The included digit demo uses explicit fixed template rules; it is not a trained SVM, forest or CNN.',
 'Lung CT transfer learning worked example':'Transfer learning starts from a pretrained model, adapts it to labeled CT data, and evaluates on separate patients. Preprocessing, leakage control and dataset labels affect the result. This topic is an architecture/workflow lesson; no trained diagnostic model is bundled.',
 'DeepFace expression workflow worked example':'A pretrained expression workflow detects and aligns a face, applies a trained model and interprets its output labels. The application contains a lesson and an untrained layer-arithmetic demonstration. It does not run DeepFace or claim to infer emotions from faces.',
 'Compression standards, formats and containers':'A coding algorithm, a file format and a multimedia container are different layers. The slides survey CCITT/JBIG/JPEG/JPEG-LS/JPEG2000, BMP/GIF/PNG/TIFF/PDF, and video families including DV, H.261–H.264, MPEG, AVS, HDV, M-JPEG, QuickTime and VC-1. The app emits actual JPEG plus clearly named educational coding exports; it does not implement every standardized codec/container.',
 '3D volume reconstruction/visualization':'A medical volume contains multiple spatially calibrated slices or projections. Rendering and reconstruction require a stack and voxel spacing. The linked experiment transforms a synthetic cube; it demonstrates 3D geometry and does not reconstruct a volume from one uploaded image.',
 'Fast Walsh-Hadamard butterfly algorithm':'The fast transform recursively combines pairs as a+b and a−b, reducing computation to O(N log N). Walsh sequency ordering reorders rows by sign changes; natural Hadamard order differs. The Walsh/Hadamard image experiment computes the mathematical transform with orthonormal matrices; the butterfly algorithm is explained here.'}


def preset(topic, op):
    text = topic.lower()
    params = {}
    if op == 'concept_lab':
        if any((x in text for x in ['dpi', 'memory', 'resolution', 'storage budget'])):
            params = {'demo': 'memory and resolution'}
        elif 'eye structure' in text:
            params = {'demo': 'eye image formation'}
        elif 'perception' in text:
            params = {'demo': 'simultaneous contrast'}
        elif 'frame rate' in text or 'sensor pulses' in text:
            params = {'demo': 'FPS motion samples'}
        elif 'linear' in text or 'lti' in text:
            params = {'demo': 'linearity test'}
        elif '3d' in text:
            params = {'demo': '3D connectivity'}
        elif 'pipeline' in text or 'workflow' in text:
            params = {'demo': 'processing pipeline'}
    if op == 'frequency_filters':
        if 'butterworth' in text:
            params['family'] = 'Butterworth'
        elif 'gaussian' in text:
            params['family'] = 'Gaussian'
        else:
            params['family'] = 'Ideal'
        if 'band' in text:
            params['mode'] = 'band-reject' if 'bandreject' in text else 'band-pass'
    if op == 'color_models':
        for name in ['CMYK', 'CMY', 'HSI', 'HSV', 'XYZ', 'YCbCr', 'YUV', 'Lab']:
            if name.lower() in text or (name == 'Lab' and 'cielab' in text):
                params['space'] = name
                break
        if 'to/from' in text or 'reconstruction' in text:
            params['mode'] = 'round-trip'
    if op == 'color_processing':
        params['domain'] = 'RGB channels'
        if 'edge' in text:
            params['method'] = 'Sobel'
        elif 'vector mean' in text:
            params['method'] = 'mean'
        else:
            params['method'] = 'Gaussian'
    if op == 'pseudocolor' and 'slicing' in text:
        params['map'] = 'density slices'
    if op == 'color_sharpen' and 'tonal' in text:
        params['mode'] = 'tonal correction'
    if op == 'walsh_hadamard':
        params['basis'] = 'Hadamard natural' if 'natural' in text else 'Walsh sequency'
    if op == 'signal_demo':
        if 'impulse' in text:
            params['mode'] = 'shifted impulse'
        elif 'fourier series' in text:
            params['mode'] = 'Fourier square-wave series'
        elif 'low/high' in text:
            params['mode'] = '1D low-pass'
        else:
            params['mode'] = 'aliasing'
    if op == 'log_dog':
        if 'difference' in text:
            params['mode'] = 'DoG'
        elif 'gradient' in text:
            params['mode'] = 'gradient threshold'
        else:
            params['mode'] = 'LoG'
    if op == 'laplacian' and 'edge' in text:
        params['mode'] = 'edge magnitude'
    if op == 'gradient_direction' and 'linking' in text:
        params['mode'] = 'hysteresis'
    if op == 'point_line' and 'point' in text:
        params['direction'] = 'point'
    if op == 'hough_lines' and 'lane' in text:
        params['mode'] = 'lane ROI'
    if op == 'lbp_lpg':
        params['mode'] = 'LPG' if 'gradients' in text else 'LBP'
    if op == 'noise_models':
        for name, needle in [
            ('Gaussian', 'gaussian'),
            ('Uniform', 'uniform'),
            ('Exponential', 'exponential'),
            ('Gamma', 'gamma'),
            ('Rayleigh', 'rayleigh'),
            ('Poisson', 'poisson'),
            ('Speckle', 'speckle'),
            ('Salt and pepper', 'salt')
        ]:
            if needle in text:
                params['model'] = name
                break
    if op == 'mean_restoration':
        if 'contraharmonic' in text:
            params['method'] = 'contraharmonic'
        elif 'harmonic' in text:
            params['method'] = 'harmonic'
        elif 'geometric' in text:
            params['method'] = 'geometric'
        else:
            params['method'] = 'arithmetic'
    if op == 'inverse_wiener':
        params['method'] = 'inverse' if text == 'inverse filtering' else 'Wiener'
    if op == 'ct_reconstruction':
        if 'radon transform' in text:
            params['mode'] = 'sinogram'
        elif text.startswith(('art ', 'sirt ', 'mlem ')):
            params['mode'] = 'SART'
        else:
            params['mode'] = 'filtered backprojection'
    if op == 'wavelet_denoise':
        params['noise'] = 'speckle'
    if op == 'lossless_coding':
        if 'run-length' in text:
            params['codec'] = 'RLE'
        elif 'arithmetic' in text:
            params['codec'] = 'Arithmetic'
        elif 'lzw' in text:
            params['codec'] = 'LZW'
        else:
            params['codec'] = 'Huffman'
    if op == 'golomb_coding':
        if 'exponential' in text:
            params['method'] = 'Exp-Golomb'
        elif 'rice' in text:
            params['method'] = 'Rice'
        else:
            params['method'] = 'Golomb'
    if op == 'jpeg_pipeline' and 'zigzag' in text:
        params['display'] = 'zigzag scan'
    if op == 'motion_coding' and 'h.263' in text:
        params['mode'] = 'H.263 DCT residual'
    if op == 'morphology_full':
        for name in ['erosion', 'dilation', 'opening', 'closing', 'gradient', 'white top-hat', 'black top-hat']:
            if name in text:
                params['method'] = name
        params['domain'] = 'grayscale' if text.startswith('grayscale') else 'binary'
    if op == 'binary_logic':
        params['method'] = topic.split()[-1]
    if op == 'skeleton' and 'pruning' in text:
        params['method'] = 'pruning'
    if op == 'wavelet' and 'daubechies' in text:
        params['wavelet'] = 'db4'
    if op == 'color_sharpen' and 'laplacian' in text:
        params['mode'] = 'Laplacian'
    if topic == 'Identity and negative intensity transforms':
        return {}
    return params


def build():
    first = json.loads((ROOT / 'data/source_topics_1_4.json').read_text(encoding='utf8'))
    second = json.loads((ROOT / 'data/source_topics_5_8.json').read_text(encoding='utf8'))['topics']
    chapters = [dict(id=i, title=title) for i, title in CHAPTERS.items()]
    topics = []
    for index, item in enumerate(first + second):
        topic = item['topic']
        candidate = item.get('operation_candidate')
        op = ALIAS.get(candidate, candidate) if candidate else MAP.get(topic)
        if op not in OPERATIONS:
            raise ValueError(f'Unmapped topic: {topic} ({op})')
        params = preset(topic, op)
        spec = OPERATIONS[op]
        allowed = {field['key']: field for field in spec['fields']}
        for key, value in params.items():
            if key not in allowed or (allowed[key]['type'] == 'select' and value not in allowed[key]['options']):
                raise ValueError(f'Invalid preset for {topic}: {key}={value}')
        topic_data = {key: item[key] for key in [
            'chapter',
            'topic',
            'source',
            'pages',
            'kind',
            'explanation',
            'formula'
        ]}
        topic_data.update(
            id=f'course-{index + 1}',
            lesson=LESSONS.get(topic, item['explanation']),
            operations=[dict(id=op, name=spec['name'], params=params)],
            coverage='Runnable algorithm' if item['kind'] == 'algorithm' else 'Concept lesson + related experiment',
            scope='This experiment implements the classical technique or an explicitly labeled bounded teaching model.' if item['kind'] == 'algorithm' else 'The related experiment illustrates a building block. It is not a claim that the complete physical, clinical, standardized or trained system is implemented.'
        )
        # Broad topics can link to several related algorithms.
        extras = {
            'Chromaticity/trichromatic coefficients and spectral tristimulus': ['spectral_demo'],
            'First and second partial derivatives': ['laplacian'],
            'Mean, box and weighted mean smoothing': ['separable_filter'],
            'Fingerprint enhancement and orientation demonstration': ['fingerprint_orientation'],
            'Piecewise-linear and sigmoid/power-law contrast stretching': ['sigmoid'],
            'Uniform grayscale and color quantization': ['indexed_palette'],
            'Color morphology/object analysis': ['color_processing', 'region_features'],
            'CNN segmentation': ['segmentation_metrics'],
            'U-Net segmentation': ['level_set', 'segmentation_metrics'],
            'Mask R-CNN instance segmentation': ['watershed'],
            'X-ray dark-region threshold example': ['binary_not'],
            'Fourier spectrum, magnitude, phase and frequency centering': ['fourier_components']
        }
        for extra in extras.get(topic, []):
            topic_data['operations'].append(dict(id=extra, name=OPERATIONS[extra]['name'], params={}))
        topics.append(topic_data)
    for chapter in chapters:
        members = [t for t in topics if t['chapter'] == chapter['id']]
        chapter.update(
            topic_count=len(members),
            algorithm_topics=sum((t['kind'] == 'algorithm' for t in members)),
            concept_topics=sum((t['kind'] == 'concept' for t in members)),
            source=members[0]['source']
        )
    document = dict(
        chapters=chapters,
        topics=topics,
        scope='Distinct taught topics from the eight uploaded chapter PDFs. Runnable algorithms share parameterized operations; physical systems, neural architectures, clinical workflows and codec standards are transparent lessons with related teaching experiments.'
    )
    (ROOT / 'data/course_topics.json').write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding='utf8'
    )
    print(f'Catalog: {len(topics)} topics; {len(OPERATIONS)} runnable operations; every topic mapped.')


if __name__ == '__main__':
    build()
