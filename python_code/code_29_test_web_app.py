import base64
import io
import json
import unittest

import cv2
import numpy as np
from PIL import Image

from code_01_web_app import app
from code_02_image_operations import OPERATIONS, execute, gray, statistics


def png(image):
    return cv2.imencode('.png', image)[1].tobytes()


def decode(data_url):
    return cv2.imdecode(np.frombuffer(base64.b64decode(data_url.split(',')[1]), np.uint8), cv2.IMREAD_UNCHANGED)


class ImageAppTests(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        y, x = np.mgrid[:48, :64]
        self.image = np.stack((x * 4 % 256, y * 5 % 256, (x + y) * 2 % 256), axis=-1).astype(np.uint8)

    def request_operation(self, operation, params=None, image=None, reference=None):
        data = dict(
            operation=operation,
            params=json.dumps(params or {}),
            image=(io.BytesIO(png(self.image if image is None else image)), 'test.png')
        )
        if reference is not None:
            data['reference'] = (io.BytesIO(png(reference)), 'reference.png')
        return self.client.post('/api/process', data=data, content_type='multipart/form-data')

    def test_every_operation_via_api(self):
        for operation in OPERATIONS:
            with self.subTest(operation=operation):
                response = self.request_operation(
                    operation,
                    reference=self.image if operation == 'matching' or OPERATIONS[operation]['requires_reference'] else None
                )
                self.assertEqual(response.status_code, 200, response.get_json())
                data = response.get_json()
                image = decode(data['result'])
                self.assertIsNotNone(image)
                self.assertEqual(image.shape[:2], (data['after']['height'], data['after']['width']))
                self.assertEqual(sum(data['after']['histogram']), image.shape[0] * image.shape[1])
                self.assertEqual(statistics(image), data['after'])

    def test_index_and_catalog(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'PixelLab', response.data)
        for path in ['/static/app.js', '/static/style.css']:
            with self.client.get(path) as asset:
                self.assertEqual(asset.status_code, 200)
        operations = self.client.get('/api/operations').get_json()
        self.assertEqual(len(operations), len(OPERATIONS))
        self.assertGreaterEqual(len(operations), 130)
        self.assertTrue(all(('def ' in operation['code'] for operation in operations)))
        self.assertEqual(self.client.get('/api/health').get_json()['status'], 'ok')
        demo = decode(self.client.get('/api/demo').get_json()['image'])
        self.assertEqual(demo.shape, (480, 720, 3))

    def test_known_pixel_math(self):
        values = np.array([[0, 50, 128, 200, 255]], dtype=np.uint8)
        image = cv2.cvtColor(values, cv2.COLOR_GRAY2BGR)
        result, _ = execute('contrast', image, dict(alpha=1, beta=-100))
        np.testing.assert_array_equal(gray(result), [[0, 0, 28, 100, 155]])
        result, _ = execute('negative', image, {})
        np.testing.assert_array_equal(gray(result), [[255, 205, 127, 55, 0]])
        result, _ = execute('quantization', image, dict(bits='1'))
        np.testing.assert_array_equal(result, [[0, 0, 255, 255, 255]])

    def test_constant_images_are_stable(self):
        for value in [0, 128, 255]:
            image = np.full((8, 8, 3), value, np.uint8)
            for key in ['stretch', 'equalize']:
                result, _ = execute(key, image, {})
                self.assertTrue(np.all(result == value))
            stats = statistics(image)
            self.assertEqual(stats['rms_contrast'], 0)
            self.assertEqual(stats['michelson'], 0)

    def test_normalized_equalization_known_distribution(self):
        image = np.array([[0, 0, 128, 255]], np.uint8)
        result, _ = execute('equalize', image, {})
        np.testing.assert_array_equal(result, [[0, 0, 128, 255]])
        result, _ = execute('equalize_basic', image, {})
        np.testing.assert_array_equal(result, [[128, 128, 191, 255]])

    def test_inverse_transform_and_identity_comparison(self):
        result, details = execute('idft', self.image, {})
        np.testing.assert_array_equal(result, gray(self.image))
        self.assertLess(details['reconstruction_mse'], 1e-20)
        result, details = execute('matching', self.image, {}, self.image)
        self.assertEqual(details['similarity_percent'], 100)
        self.assertFalse(np.any(result))

    def test_geometry(self):
        result = decode(self.request_operation('resize', {'width': 25, 'height': 15}).get_json()['result'])
        self.assertEqual(result.shape[:2], (15, 25))
        result, _ = execute('rotate', self.image, {'angle': '90'})
        np.testing.assert_array_equal(result, np.rot90(self.image, k=-1))
        result, _ = execute('crop', self.image, {'x': 0, 'y': 0, 'width': 50, 'height': 50})
        np.testing.assert_array_equal(result, self.image[:24, :32])

    def test_parameter_errors(self):
        cases = [
            ('unknown', {}),
            ('gamma', {'gamma': 0}),
            ('gamma', {'gamma': 'NaN'}),
            ('resize', {'width': 1.5}),
            ('resize', {'width': 4096, 'height': 4096}),
            ('median', {'kernel': 4}),
            ('crop', {'x': 90, 'width': 30}),
            ('canny', {'low': 220, 'high': 100}),
            ('flip', {'axis': 'bad'}),
            ('matching', {})
        ]
        for key, params in cases:
            with self.subTest(key=key, params=params):
                response = self.request_operation(key, params)
                self.assertEqual(response.status_code, 400)
                self.assertIn('error', response.get_json())

    def test_upload_and_json_errors(self):
        response = self.client.post('/api/process', data={'operation': 'grayscale'})
        self.assertEqual(response.status_code, 400)
        for params in ['{broken', '[]']:
            response = self.client.post('/api/process', data={
                'operation': 'grayscale',
                'params': params,
                'image': (io.BytesIO(png(self.image)), 'test.png')
            })
            self.assertEqual(response.status_code, 400)
        response = self.client.post('/api/process', data={
            'operation': 'grayscale',
            'image': (io.BytesIO(b'not an image'), 'fake.png')
        })
        self.assertEqual(response.status_code, 400)
        old_limit = app.config['MAX_CONTENT_LENGTH']
        try:
            app.config['MAX_CONTENT_LENGTH'] = 100
            response = self.request_operation('grayscale')
            self.assertEqual(response.status_code, 413)
        finally:
            app.config['MAX_CONTENT_LENGTH'] = old_limit

    def test_tiff_is_converted_for_browser(self):
        buffer = io.BytesIO()
        Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)).save(buffer, format='TIFF')
        buffer.seek(0)
        response = self.client.post(
            '/api/process',
            data={'operation': 'grayscale', 'image': (buffer, 'test.tiff')}
        )
        self.assertEqual(response.status_code, 200)
        np.testing.assert_array_equal(decode(response.get_json()['original']), self.image)

    def test_jpeg_bytes_match_reported_result(self):
        response = self.request_operation('compression', {'quality': 35})
        data = response.get_json()
        self.assertTrue(data['result'].startswith('data:image/jpeg;'))
        payload = base64.b64decode(data['result'].split(',')[1])
        self.assertEqual(len(payload), data['details']['jpeg_bytes'])
        expected = cv2.imencode('.jpg', self.image, [cv2.IMWRITE_JPEG_QUALITY, 35])[1].tobytes()
        self.assertEqual(payload, expected)
        self.assertGreater(data['details']['mse'], 0)

    def test_matplotlib_figure_and_console_output(self):
        response = self.client.post('/api/process', data={
            'operation': 'pixel_matrix',
            'params': json.dumps({'size': 8}),
            'lab_output': 'true',
            'image': (io.BytesIO(png(self.image)), 'test.png')
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        figure = decode(data['lab']['figure'])
        self.assertEqual(figure.shape[:2], (1120, 1960))
        self.assertGreater(float(figure.std()), 10)
        console = data['lab']['console']
        self.assertIn('Operation: Digital pixel matrix', console)
        self.assertIn('Original grayscale pixel sample (resampled to 8 x 8', console)
        self.assertIn('Processed grayscale pixel matrix (actual 8 x 8)', console)
        self.assertIn('Process finished successfully.', console)
        self.assertIn(np.array2string(decode(data['result']), max_line_width=100, threshold=256), console)


if __name__ == '__main__':
    unittest.main()
