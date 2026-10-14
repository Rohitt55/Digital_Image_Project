import base64
import io
import json
import unittest

import cv2
import numpy as np

from code_01_web_app import app, COURSE
from code_02_image_operations import OPERATIONS, execute, gray
import code_12_lossless_codecs as codecs


class ChapterTests(unittest.TestCase):

    def setUp(self):
        self.image = np.random.default_rng(42).integers(0, 256, (48, 64, 3), dtype=np.uint8)

    def test_all_selectable_algorithm_variants(self):
        for key, spec in OPERATIONS.items():
            for field in spec['fields']:
                if field['type'] != 'select':
                    continue
                for option in field['options']:
                    with self.subTest(operation=key, field=field['key'], value=option):
                        result, details = execute(key, self.image, {field['key']: option}, self.image)
                        self.assertEqual(result.dtype, np.uint8)
                        self.assertIn(result.ndim, [2, 3])
                        self.assertGreater(result.size, 0)
                        json.dumps(details, allow_nan=False)

    def test_course_topics_have_valid_source_and_presets(self):
        self.assertEqual(len(COURSE['chapters']), 8)
        self.assertEqual(len(COURSE['topics']), 264)
        for topic in COURSE['topics']:
            self.assertTrue(topic['lesson'])
            self.assertTrue(topic['pages'])
            for item in topic['operations']:
                with self.subTest(topic=topic['topic'], operation=item['id']):
                    self.assertIn(item['id'], OPERATIONS)
                    output, _ = execute(item['id'], self.image, item['params'], self.image)
                    self.assertGreater(output.size, 0)
        with app.test_client() as client:
            self.assertEqual(len(client.get('/api/course').get_json()['topics']), 264)
            with client.get('/api/course/source/1') as pdf:
                self.assertEqual(pdf.status_code, 200)
                self.assertTrue(pdf.data.startswith(b'%PDF'))
            self.assertEqual(client.get('/api/course/source/99').status_code, 404)

    def test_lossless_coders_decode_difficult_sequences(self):
        sequences = [
            [0],
            [255] * 100,
            list(range(256)),
            [0, 255] * 100,
            np.random.default_rng(4).integers(0, 256, 1024).tolist()
        ]
        for sequence in sequences:
            for name in ['RLE', 'Huffman', 'Arithmetic', 'LZW']:
                with self.subTest(codec=name, length=len(sequence)):
                    if name == 'RLE':
                        payload, _ = codecs.rle_encode(sequence)
                        decoded = codecs.rle_decode(payload)
                    elif name == 'Huffman':
                        payload, metadata = codecs.huffman_encode(sequence)
                        decoded = codecs.huffman_decode(payload, metadata)
                    elif name == 'Arithmetic':
                        payload, metadata = codecs.arithmetic_encode(sequence)
                        decoded = codecs.arithmetic_decode(payload, metadata)
                    else:
                        payload, _ = codecs.lzw_encode(sequence)
                        decoded = codecs.lzw_decode(payload)
                    self.assertEqual(decoded, sequence)
            for method in ['Golomb', 'Rice', 'Exp-Golomb']:
                for divisor in [1, 3, 16, 127]:
                    for order in [0, 4, 7]:
                        payload, metadata = codecs.golomb_encode(sequence, method, divisor, order)
                        self.assertEqual(codecs.golomb_decode(payload, metadata), sequence)

    def test_exported_arithmetic_file_is_independently_decodable(self):
        with app.test_client() as client:
            response = client.post('/api/process', data={
                'operation': 'lossless_coding',
                'params': json.dumps({'codec': 'Arithmetic'}),
                'image': (io.BytesIO(cv2.imencode('.png', self.image)[1].tobytes()), 'input.png')
            })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        document = json.loads(base64.b64decode(data['artifact']['data'].split(',')[1]))
        decoded = np.array(
            codecs.arithmetic_decode(base64.b64decode(document['payload_base64']), document['metadata']),
            np.uint8
        ).reshape(document['shape'])
        np.testing.assert_array_equal(decoded, gray(self.image))
        self.assertTrue(data['details']['lossless_round_trip'])
        self.assertEqual(
            len(base64.b64decode(data['artifact']['data'].split(',')[1])),
            data['details']['exported_file_bytes']
        )

    def test_transform_reconstruction_and_metrics(self):
        for key, params in [
            ('dct', {'mode': 'reconstruction', 'retain': 100}),
            ('wavelet', {'mode': 'reconstruction', 'threshold': 0}),
            ('pca_klt', {'components': 256}),
            ('predictive_coding', {'step': 1})
        ]:
            with self.subTest(operation=key):
                output, details = execute(key, self.image, params)
                np.testing.assert_array_equal(output, gray(self.image))
                self.assertLess(details['reconstruction_mse'], 1e-15)
        _, details = execute('segmentation_metrics', self.image, {}, self.image)
        for key in ['dice', 'iou', 'pixel_accuracy', 'precision', 'recall']:
            self.assertEqual(details[key], 1)
        _, details = execute('quality', self.image, {}, self.image)
        self.assertEqual(details['mse'], 0)
        self.assertEqual(details['ssim'], 1)

    def test_morphology_mathematical_properties(self):
        values = gray(self.image)
        opening, _ = execute('morphology_full', self.image, {'domain': 'grayscale', 'method': 'opening'})
        closing, _ = execute('morphology_full', self.image, {'domain': 'grayscale', 'method': 'closing'})
        white, _ = execute('morphology_full', self.image, {'domain': 'grayscale', 'method': 'white top-hat'})
        black, _ = execute('morphology_full', self.image, {'domain': 'grayscale', 'method': 'black top-hat'})
        self.assertTrue(np.all(opening <= values))
        self.assertTrue(np.all(closing >= values))
        np.testing.assert_array_equal(white, values - opening)
        np.testing.assert_array_equal(black, closing - values)


if __name__=='__main__': unittest.main()
