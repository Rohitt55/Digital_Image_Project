"""Local web interface for the supplied CSE 4245 image-processing labs."""
import base64
import io
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

from code_02_image_operations import OPERATIONS, execute, describe, statistics
import code_03_operation_loader  # Registers the additional chapter algorithms.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
app = Flask(__name__, template_folder=str(PROJECT_ROOT / 'templates'),
            static_folder=str(PROJECT_ROOT / 'static'), static_url_path='/static')
COURSE = json.loads((PROJECT_ROOT / 'data/course_topics.json').read_text(encoding='utf8'))
app.config['MAX_CONTENT_LENGTH'] = 24 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 12_000_000


def read_image(upload):
    if upload is None or not upload.filename:
        raise ValueError('Please upload an image first.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(upload.stream) as image:
                if image.width * image.height > 12000000:
                    raise ValueError('Please use an image with at most 12 million pixels.')
                rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('Cannot read this image. Try PNG, JPG, BMP, WebP or TIFF.') from exc


def encode(image, extension='.png', options=None):
    ok, data = cv2.imencode(extension, image, options or [])
    if not ok:
        raise ValueError('Could not encode the processed image.')
    mime = 'image/jpeg' if extension == '.jpg' else 'image/png'
    return 'data:' + mime + ';base64,' + base64.b64encode(data).decode('ascii')


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/api/operations')
def operations():
    groups = [
        'Fundamentals',
        'Color & intensity',
        'Histograms',
        'Geometry',
        'Filters & restoration',
        'Edges & segmentation',
        'Morphology',
        'Transforms & comparison',
        'Compression',
        'Features & recognition'
    ]
    keys = sorted(
        OPERATIONS,
        key=lambda key: groups.index(OPERATIONS[key]['group']) if OPERATIONS[key]['group'] in groups else len(groups)
    )
    result = []
    fallback = {
        'Fundamentals': 1,
        'Color & intensity': 3,
        'Histograms': 2,
        'Geometry': 4,
        'Filters & restoration': 6,
        'Edges & segmentation': 5,
        'Morphology': 8,
        'Transforms & comparison': 4,
        'Compression': 7
    }
    for key in keys:
        item = describe(key)
        chapters = {topic['chapter'] for topic in COURSE['topics'] if any((op['id'] == key for op in topic['operations']))}
        item['chapters'] = sorted(chapters or {fallback.get(item['group'], 1)})
        result.append(item)
    return jsonify(result)


@app.get('/api/course')
def course():
    return jsonify(COURSE)


@app.get('/api/chapters')
def chapters():
    return jsonify([{'id': item['id'], 'title': item['title']} for item in COURSE['chapters']])


@app.get('/api/course/source/<int:chapter>')
def course_source(chapter):
    entry = next((item for item in COURSE['chapters'] if item['id'] == chapter), None)
    if entry is None:
        return (jsonify(error='Unknown chapter.'), 404)
    path = PROJECT_ROOT / 'pdfs' / entry['source']
    if not path.is_file():
        return (jsonify(error='The original chapter PDF is missing from the pdfs folder.'), 404)
    return send_file(path, mimetype='application/pdf')


@app.get('/api/health')
def health():
    return jsonify(status='ok', operations=len(OPERATIONS), course_topics=len(COURSE['topics']))


@app.get('/api/demo')
def demo():
    """A repeatable test scene with color, edges, texture and uneven lighting."""
    y, x = np.mgrid[:480, :720]
    image = np.zeros((480, 720, 3), np.uint8)
    image[:, :, 0] = (40 + x * 0.16 + y * 0.1).astype(np.uint8)
    image[:, :, 1] = (60 + x * 0.12).astype(np.uint8)
    image[:, :, 2] = (95 + y * 0.18).astype(np.uint8)
    cv2.rectangle(image, (80, 95), (295, 345), (100, 190, 225), -1)
    cv2.circle(image, (490, 235), 125, (180, 205, 85), -1)
    for i in range(8):
        cv2.line(image, (90 + i * 25, 110), (90 + i * 25, 330), (45, 65, 85), 3)
    cv2.putText(image, 'DIP LAB', (230, 435), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (240, 245, 250), 3)
    return jsonify(image=encode(image), statistics=statistics(image))


@app.post('/api/process')
def process():
    try:
        operation = request.form.get('operation', '')
        if operation not in OPERATIONS:
            raise ValueError('Choose a valid processing operation.')
        params = json.loads(request.form.get('params', '{}'))
        if not isinstance(params, dict):
            raise ValueError('Parameters must be a JSON object.')
        image = read_image(request.files.get('image'))
        reference = read_image(request.files.get('reference')) if request.files.get('reference') else None
        if (OPERATIONS[operation]['requires_reference'] or operation == 'matching') and reference is None:
            raise ValueError('Upload a reference image for this operation.')
        result, details = execute(operation, image, params, reference)
        output = details.pop('download') if operation == 'compression' else encode(result)
        artifact = details.pop('artifact', None)
        before, after = (statistics(image), statistics(result))
        lab = None
        if request.form.get('lab_output') == 'true':
            from code_15_plot_results import make_output
            lab = make_output(image, result, before, after, describe(operation)['name'], params, details)
        return jsonify(
            original=encode(image),
            result=output,
            before=before,
            after=after,
            details=details,
            lab=lab,
            artifact=artifact,
            operation=describe(operation)['name']
        )
    except (ValueError, TypeError, OverflowError, json.JSONDecodeError) as exc:
        return (jsonify(error=str(exc)), 400)
    except cv2.error:
        return (
            jsonify(error='These settings cannot be applied to this image. Try a larger image or smaller kernel.'),
            400
        )


@app.errorhandler(RequestEntityTooLarge)
def too_large(error):
    return (jsonify(error='Upload is too large. The total request limit is 24 MB.'), 413)


if __name__ == '__main__':
    import argparse
    import threading
    import time
    import urllib.error
    import urllib.request
    import webbrowser

    parser = argparse.ArgumentParser(description='Start the local PixelLab backend.')
    parser.add_argument('--open-browser', action='store_true', help='Open the app after the backend is ready.')
    args = parser.parse_args()

    def open_when_ready():
        for _ in range(60):
            try:
                with urllib.request.urlopen('http://127.0.0.1:5000/api/health', timeout=1) as response:
                    if json.load(response).get('status') == 'ok':
                        webbrowser.open('http://127.0.0.1:5000')
                        return
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(0.25)
        print('Backend is not ready. Check the error above before refreshing your browser.', flush=True)

    if args.open_browser:
        threading.Thread(target=open_when_ready, daemon=True).start()
    app.run(host='127.0.0.1', port=5000, debug=False)
