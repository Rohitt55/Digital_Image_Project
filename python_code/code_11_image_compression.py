import base64
import json

import cv2
import numpy as np
from scipy import fft

from code_02_image_operations import operation, field, choice, gray, normalize
from code_04_image_helpers import bounded, u8
import code_12_lossless_codecs as codecs

SOURCE='Chap7-Image Compression.pdf'
GROUP='Compression'


def artifact(method, shape, payload, metadata):
    document = {
        'format': 'PixelLab teaching codec v1',
        'method': method,
        'shape': list(shape),
        'metadata': metadata,
        'payload_base64': base64.b64encode(payload).decode('ascii')
    }
    encoded = json.dumps(document, separators=(',', ':')).encode('utf8')
    return ({
        'name': f"PixelLab-{method.lower().replace(' ', '-')}.json",
        'mime': 'application/json',
        'data': 'data:application/json;base64,' + base64.b64encode(encoded).decode('ascii')
    }, len(encoded))


@operation('lossless_coding','RLE / Huffman / arithmetic / LZW coding',GROUP,'Encode and independently decode a bounded grayscale image. Report payload size and exported file size separately.', 'Lossless decode(encode(pixels)) = pixels',SOURCE,
           [
               choice('codec','Coding method','Huffman',['RLE','Huffman','Arithmetic','LZW']),
               choice('size','Maximum analysis side','128',['32','64','128'])
           ])
def lossless_coding(image, params, reference):
    values = gray(bounded(image, int(params['size'])))
    pixels = values.ravel().tolist()
    method = params['codec']
    if method == 'RLE':
        payload, metadata = codecs.rle_encode(pixels)
        decoded = codecs.rle_decode(payload)
    elif method == 'Huffman':
        payload, metadata = codecs.huffman_encode(pixels)
        decoded = codecs.huffman_decode(payload, metadata)
    elif method == 'Arithmetic':
        payload, metadata = codecs.arithmetic_encode(pixels)
        decoded = codecs.arithmetic_decode(payload, metadata)
    else:
        payload, metadata = codecs.lzw_encode(pixels)
        decoded = codecs.lzw_decode(payload)
    output = np.array(decoded, np.uint8).reshape(values.shape)
    download, size = artifact(method, values.shape, payload, metadata)
    return (output, {
        'raw_grayscale_bytes': int(values.size),
        'coded_payload_bytes': len(payload),
        'exported_file_bytes': size,
        'payload_compression_ratio': values.size / max(1, len(payload)),
        'file_compression_ratio': values.size / size,
        'payload_bits_per_pixel': len(payload) * 8 / values.size,
        'relative_payload_redundancy': 1 - len(payload) / values.size,
        'lossless_round_trip': bool(np.array_equal(output, values)),
        'artifact': download,
        'note': 'Losslessness is relative to the bounded grayscale analysis image. Export includes dimensions, code metadata and base64 payload; metadata overhead can exceed any compression saving. LZW uses fixed 16-bit codes.'
    })


@operation('golomb_coding','Golomb / Rice / exponential-Golomb coding',GROUP,'Encode nonnegative intensity symbols with unary/truncated-binary or Exp-Golomb codes and decode them back.', 'Golomb: q=floor(n/m), r=n mod m; Rice uses m=2^k',SOURCE,
           [
               choice('method','Code family','Rice',['Golomb','Rice','Exp-Golomb']),
               field('m','Golomb divisor m',16,1,128),
               field('k','Rice / Exp-Golomb order k',4,0,7)
           ])
def golomb_coding(image, params, reference):
    values = gray(bounded(image, 64))
    payload, metadata = codecs.golomb_encode(values.ravel().tolist(), params['method'], params['m'], params['k'])
    decoded = codecs.golomb_decode(payload, metadata)
    output = np.array(decoded, np.uint8).reshape(values.shape)
    download, size = artifact(params['method'], values.shape, payload, metadata)
    return (output, {
        'coded_bits': metadata['bit_count'],
        'coded_payload_bytes': len(payload),
        'raw_bytes': int(values.size),
        'exported_file_bytes': size,
        'lossless_round_trip': bool(np.array_equal(output, values)),
        'artifact': download,
        'note': '64-pixel maximum-side grayscale teaching stream. These codes are efficient for small residual symbols; raw photo intensities may expand.'
    })


@operation(
    'predictive_coding',
    'Predictive coding / DPCM',
    GROUP,
    'Predict each pixel from the reconstructed previous raster pixel and quantize the residual. Unit quantization reconstructs exactly.',
    'residual=round((current-prediction)/step); reconstructed=prediction+step×residual',
    SOURCE,
    [field('step','Residual quantization step',1,1,50)]
)
def predictive_coding(image, params, reference):
    values = gray(bounded(image, 128))
    previous = 0
    codes = []
    decoded = []
    for value in values.ravel():
        code = round((int(value) - previous) / params['step'])
        previous = int(np.clip(previous + code * params['step'], 0, 255))
        codes.append(code)
        decoded.append(previous)
    output = np.array(decoded, np.uint8).reshape(values.shape)
    payload = np.array(codes, dtype='>i2').tobytes()
    download, size = artifact('DPCM', values.shape, payload, {
        'step': params['step'],
        'dtype': 'big-endian int16',
        'predictor': 'previous reconstructed raster pixel'
    })
    return (output, {
        'reconstruction_mse': float(np.mean((values.astype(float) - output) ** 2)),
        'residual_entropy_bits': float(-sum((prob * np.log2(prob) for prob in np.unique(codes, return_counts=True)[1] / len(codes)))),
        'artifact': download,
        'exported_file_bytes': size,
        'note': 'DPCM changes the distribution; fixed 16-bit residual storage alone is not an entropy-compressed bitstream.'
    })


def zigzag(block):
    indices = []
    for diagonal in range(2 * block.shape[0] - 1):
        pairs = [(y, diagonal - y) for y in range(block.shape[0]) if 0 <= diagonal - y < block.shape[1]]
        indices.extend(pairs[::-1] if diagonal % 2 == 0 else pairs)
    return [int(block[y, x]) for y, x in indices]


JPEG_TABLE=np.array(
    [
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ],
    float
)


@operation(
    'jpeg_pipeline',
    'JPEG / DCT quantization / zigzag stages',
    GROUP,
    'Run block DCT, quantization, zigzag scan and inverse DCT on grayscale. This exposes JPEG building blocks; the JSON export is a teaching format.',
    '8×8 DCT → quantize → zigzag → inverse quantize → IDCT',
    SOURCE,
    [
        field('quality','Quantization quality',50,1,100),
        choice(
            'display',
            'Output',
            'reconstruction',
            ['reconstruction','first block coefficients','first block quantized coefficients','zigzag scan']
        )
    ]
)
def jpeg_pipeline(image, params, reference):
    values = gray(bounded(image, 128))
    h, w = values.shape
    padded = np.pad(values, ((0, -h % 8), (0, -w % 8)), mode='edge').astype(float)
    output = np.zeros_like(padded)
    quality = params['quality']
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    table = np.maximum(1, np.floor((JPEG_TABLE * scale + 50) / 100))
    blocks = []
    first = None
    for y in range(0, padded.shape[0], 8):
        for x in range(0, padded.shape[1], 8):
            coefficients = fft.dctn(padded[y:y + 8, x:x + 8] - 128, norm='ortho')
            quantized = np.round(coefficients / table).astype(np.int16)
            if first is None:
                first = (coefficients, quantized)
            output[y:y + 8, x:x + 8] = fft.idctn(quantized * table, norm='ortho') + 128
            blocks.append(zigzag(quantized))
    result = u8(output[:h, :w])
    payload = np.array(blocks, dtype='>i2').tobytes()
    download, size = artifact('DCT-teaching', values.shape, payload, {
        'quantization_table': table.astype(int).tolist(),
        'padded_shape': list(padded.shape),
        'ordering': 'block raster, each block zigzag',
        'dtype': 'big-endian int16'
    })
    details = {
        'first_block_zigzag': zigzag(first[1]),
        'nonzero_quantized_coefficients': int(np.count_nonzero(blocks)),
        'reconstruction_mse': float(np.mean((values.astype(float) - result) ** 2)),
        'artifact': download,
        'exported_file_bytes': size,
        'note': 'This is a grayscale JPEG-stage demo, not a standards-compliant JPEG bitstream. Use JPEG compression for an actual .jpg download.'
    }
    if params['display'] == 'first block coefficients':
        result = normalize(np.log1p(np.abs(first[0])))
    elif params['display'] == 'first block quantized coefficients':
        result = normalize(np.log1p(np.abs(first[1])))
    elif params['display'] == 'zigzag scan':
        result = normalize(np.array(details['first_block_zigzag']).reshape(8, 8))
    return (result, details)


@operation(
    'vector_quantization',
    'Vector quantization · image blocks',
    GROUP,
    'Learn a codebook of 4×4 grayscale blocks and store one codebook index per block.',
    'block → nearest codeword; decoder replaces index with codeword',
    SOURCE,
    [field('codewords','Codebook size',16,2,64),field('iterations','Training iterations',15,1,40)]
)
def vector_quantization(image, params, reference):
    values = gray(bounded(image, 128))
    h, w = values.shape
    padded = np.pad(values, ((0, -h % 4), (0, -w % 4)), mode='edge')
    ph, pw = padded.shape
    blocks = padded.reshape(ph // 4, 4, pw // 4, 4).transpose(0, 2, 1, 3).reshape(-1, 16).astype(float)
    count = min(params['codewords'], len(blocks))
    rng = np.random.default_rng(42)
    centers = blocks[rng.choice(len(blocks), count, replace=False)]
    for _ in range(params['iterations']):
        labels = ((blocks[:, None] - centers[None]) ** 2).sum(axis=2).argmin(axis=1)
        updated = np.array([blocks[labels == k].mean(axis=0) if np.any(labels == k) else centers[k] for k in range(count)])
        if np.allclose(updated, centers):
            break
        centers = updated
    centers = u8(centers)
    labels = ((blocks[:, None] - centers[None].astype(float)) ** 2).sum(axis=2).argmin(axis=1)
    result = centers[labels].reshape(ph // 4, pw // 4, 4, 4).transpose(0, 2, 1, 3).reshape(ph, pw)[:h, :w]
    payload = labels.astype(np.uint8).tobytes()
    download, size = artifact('VQ', values.shape, payload, {
        'codebook': centers.tolist(),
        'block_size': 4,
        'padded_shape': [ph, pw]
    })
    return (result, {
        'codewords': count,
        'reconstruction_mse': float(np.mean((values.astype(float) - result) ** 2)),
        'index_bytes': len(payload),
        'codebook_bytes': int(centers.nbytes),
        'artifact': download,
        'exported_file_bytes': size,
        'note': 'Compression must account for both indices and codebook. Analysis uses a 128-pixel maximum side.'
    })


@operation('motion_coding','MPEG I/P-frame / H.263 residual demo',GROUP,'Use the reference as the previous frame, estimate global motion and quantize the motion-compensated residual. No standards-compliant video file is produced.', 'P frame = motion prediction(previous frame) + decoded residual',SOURCE,
           [
               choice(
                   'mode',
                   'Coding stage',
                   'MPEG residual',
                   ['I frame','MPEG residual','H.263 DCT residual','reconstructed frame']
               ),
               field('step','Residual quantization step',8,1,64)
           ],requires_reference=True)
def motion_coding(image, params, reference):
    if reference is None:
        raise ValueError('Upload the previous frame as the reference image.')
    current = gray(bounded(image, 256))
    previous = cv2.resize(gray(reference), (current.shape[1], current.shape[0])).astype(np.float32)
    shift, response = cv2.phaseCorrelate(previous, current.astype(np.float32))
    dx, dy = shift if response > 0.01 else (0, 0)
    prediction = cv2.warpAffine(
        previous,
        np.float32([[1, 0, dx], [0, 1, dy]]),
        (current.shape[1], current.shape[0]),
        borderMode=cv2.BORDER_REFLECT
    )
    residual = current.astype(float) - prediction
    if params['mode'] == 'I frame':
        ok, data = cv2.imencode('.jpg', current, [cv2.IMWRITE_JPEG_QUALITY, 80])
        result = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        return (result, {
            'jpeg_intra_frame_bytes': len(data),
            'note': 'I-frame illustration uses JPEG intra-image coding, not an MPEG container.'
        })
    if params['mode'] == 'H.263 DCT residual':
        h, w = residual.shape
        padded = np.pad(residual, ((0, -h % 8), (0, -w % 8)), mode='edge')
        decoded = np.zeros_like(padded)
        for y in range(0, padded.shape[0], 8):
            for x in range(0, padded.shape[1], 8):
                coefficients = fft.dctn(padded[y:y + 8, x:x + 8], norm='ortho')
                decoded[y:y + 8, x:x + 8] = fft.idctn(
                    np.round(coefficients / params['step']) * params['step'],
                    norm='ortho'
                )
        decoded = decoded[:h, :w]
    else:
        decoded = np.round(residual / params['step']) * params['step']
    reconstruction = u8(prediction + decoded)
    result = reconstruction if params['mode'] == 'reconstructed frame' else normalize(decoded)
    return (result, {
        'motion_dx': float(dx),
        'motion_dy': float(dy),
        'motion_response': float(response),
        'reconstruction_mse': float(np.mean((current.astype(float) - reconstruction) ** 2)),
        'residual_variance': float(residual.var()),
        'note': 'Two-frame global-motion teaching demo. It does not implement MPEG/H.263 syntax, block motion search or a playable video container.'
    })
