# Python code guide — কোন কোড কী কাজে লাগে

Project-এর সব Python source file সরাসরি এই ফোল্ডারে আছে। App, algorithms, মূল উদাহরণ, tests ও catalog builder একই জায়গায় রাখা হয়েছে।

সব file-এ `code_01_` থেকে `code_30_` পর্যন্ত ক্রমিক নম্বর আছে। এরপর কাজ অনুযায়ী lowercase `snake_case` নাম লেখা। নামের মধ্যে lab/chapter prefix নেই এবং সব Python source একই ফোল্ডারে আছে। পুরোনো ও নতুন নামের mapping: [file_name_changes.json](file_name_changes.json)।

## Folder layout

```text
python_code/
├── code_01_web_app.py               Flask backend / API
├── code_02_image_operations.py      Basic algorithms + operation registry
├── code_15_plot_results.py          Matplotlib plots + console output
├── code_03_operation_loader.py      Algorithm registration
├── code_05_image_fundamentals.py    Pixel and intensity analysis
├── code_06_image_enhancement.py     Enhancement and filtering
├── code_07_color_processing.py      Color operations
├── code_08_image_transforms.py      Geometry and transforms
├── code_09_image_segmentation.py    Segmentation and edges
├── code_10_image_restoration.py     Noise and restoration
├── code_11_image_compression.py     Compression workflows
├── code_13_morphological_operations.py
├── code_04_image_helpers.py         Shared helpers
├── code_12_lossless_codecs.py       Encoders and decoders
├── code_14_learning_demos.py        Concept experiments
├── code_30_build_topic_catalog.py   Topic catalog builder
├── code_29_test_web_app.py          API tests
├── code_28_test_algorithms.py       Algorithm tests
├── code_16_resize_image.py          Resize example
├── code_26_edge_detection.py        Edge detection example
└── ...                      Other examples
```

Project root-এ `start.bat`, `requirements.txt`, `templates/`, `static/` ও `data/` থাকবে। সব chapter, lab, task ও course outline PDF **`pdfs/`** ফোল্ডারে আছে। `.venv/`-এ installed third-party packages থাকে; সেটি project source code-এর ফোল্ডার নয়।

## কোন ধরনের কোড ব্যবহার হয়েছে

| Code type | ব্যবহারের জায়গা | কাজ |
| --- | --- | --- |
| Python backend / REST API | `code_01_web_app.py` | Browser-এর image ও settings গ্রহণ করে, algorithm চালায় এবং JSON result দেয় |
| Function-based algorithm code | `code_02_image_operations.py` ও সংশ্লিষ্ট algorithm file | প্রতিটি operation আলাদা function-এ লেখা; image array input নিয়ে image ও measurements return করে |
| Decorator / registry pattern | `code_02_image_operations.py`-এর `@operation(...)` | Operation-এর নাম, group, formula, parameter ও reference এক জায়গায় register করে |
| Numerical / matrix processing | NumPy, SciPy | Pixel math, histogram, probability, transforms, filtering ও linear algebra |
| Computer vision / image processing | OpenCV, scikit-image | Color conversion, resize, edges, segmentation, restoration ও morphology |
| Wavelet processing | PyWavelets | Haar, db4, db8 transforms ও reconstruction |
| Scientific plotting | Matplotlib, `code_15_plot_results.py` | Image, histogram ও CDF-এর figure তৈরি এবং browser-এর জন্য PNG encode করে |
| Image decoding / orientation | Pillow | JPG/PNG/TIFF ইত্যাদি পড়ে এবং EXIF orientation ঠিক করে |
| Coding / decoding algorithms | `code_12_lossless_codecs.py` | RLE, Huffman, arithmetic, Golomb/Rice/Exp-Golomb ও LZW bitstream encode/decode |
| Automated test code | `code_29_test_web_app.py`, `code_28_test_algorithms.py` | API, pixel math, variants, presets, codec round-trip ও algorithm properties যাচাই |
| Data preparation / catalog generation | `code_30_build_topic_catalog.py` | PDF topic inventory থেকে source-linked chapter catalog তৈরি করে |

Frontend-এ **HTML** দিয়ে structure, **CSS** দিয়ে design এবং **JavaScript** দিয়ে controls/API calls ব্যবহার হয়েছে। এগুলো root-এর `templates/` ও `static/`-এ আছে। ছবির algorithm Python backend-এ চলে।

## App-এর প্রধান ফাইল

| Python file | বিস্তারিত দায়িত্ব |
| --- | --- |
| `code_01_web_app.py` | Flask routes, upload validation, image I/O, course PDF serving, algorithm execution এবং result response |
| `code_02_image_operations.py` | Basic operations, parameter validation, grayscale statistics, decorator registry এবং operation code information |
| `code_15_plot_results.py` | ছয়টি Matplotlib plot, PNG figure এবং terminal-এর মতো text/matrix output |

প্রতিটি operation-এর **Algorithm & code** tab-এ actual Python file path, code type, module-এ ব্যবহৃত libraries এবং function-এর code দেখা যায়। নতুন কাজের code খুঁজতে এটিই সবচেয়ে সহজ উপায়।

## Algorithm files

| File | কী ধরনের কাজ |
| --- | --- |
| `code_03_operation_loader.py` | Algorithm file-গুলো import করে operation register করে |
| `code_04_image_helpers.py` | Shared helpers: bounded previews, uint8 conversion, region label colors ও aligned references |
| `code_05_image_fundamentals.py` | Entropy, connectivity, shortest paths, quantization, palette, features, probability classifier ও basic educational demos |
| `code_06_image_enhancement.py` | Intensity transforms, histogram matching/AHE, custom kernels, order filters, frequency enhancement ও fingerprint processing |
| `code_07_color_processing.py` | Color models/conversions, chromaticity, pseudocolor, balancing, segmentation ও color filtering |
| `code_08_image_transforms.py` | Affine geometry, pyramids, FFT components, DCT, wavelets, Walsh/Hadamard, PCA ও signal/3D demos |
| `code_09_image_segmentation.py` | Thresholding, edges, texture, Hough, growing/clustering, watershed, contours, snakes/level sets ও mask metrics |
| `code_10_image_restoration.py` | Noise models, filters, inverse/Wiener restoration, texture/shape measurements ও synthetic CT/MRI experiments |
| `code_11_image_compression.py` | Image coding workflows, compression measurements, export generation, DPCM, DCT stages, VQ ও two-frame demonstrations |
| `code_12_lossless_codecs.py` | Independently decodable lossless teaching codec implementations |
| `code_13_morphological_operations.py` | Binary/grayscale morphology, Boolean masks, hit-or-miss, skeletons, top-hats, distance maps ও reconstruction |
| `code_14_learning_demos.py` | Acquisition/perception, memory/FPS/linearity, synthetic 3D connectivity, spectral, registration ও Rician-noise teaching models |

Algorithm file-এর `@operation(...)`-এ explanation ও formula আছে। Function-এর input `image` হলো OpenCV image, `params` হলো validated parameter dictionary এবং `reference` হলো optional দ্বিতীয় image। Output হলো image অথবা `(image, details)`।

কোড পড়ার সময় প্রথমে decorator-এর নিচের function দেখুন। সেখানে input প্রস্তুত করা, algorithm চালানো এবং result return করার ধাপ আছে। এক লাইনে একাধিক statement রাখা হয়নি। ছবি load ও plot করার সাধারণ কাজ `code_04_image_helpers.py`-এ আছে, যাতে example-গুলো ছোট থাকে। RLE/Huffman/LZW-এর মতো coding algorithm নিজস্ব encode/decode ধাপ দিয়ে লেখা হয়েছে।

## মূল উদাহরণগুলো

| Original Python file | Topic / code type |
| --- | --- |
| `code_16_resize_image.py` | Image resizing / downsampling — OpenCV lab script |
| `code_17_transform_and_filter_image.py` | Crop, resize, rotate, flip ও smoothing filters — OpenCV lab script |
| `code_19_contrast_and_histogram.py` | Contrast, grayscale, histogram ও equalization — OpenCV/Matplotlib lab script |
| `code_18_brightness_and_contrast.py` | Alpha/beta brightness ও contrast adjustment — OpenCV lab script |
| `code_20_contrast_measurements.py` | Contrast measurements ও histogram — NumPy/Matplotlib lab script |
| `code_22_image_histogram.py` | একটি ছোট matrix-এর histogram — NumPy/Matplotlib example |
| `code_21_contrast_stretching.py` | Minimum/maximum দিয়ে contrast stretching — NumPy/OpenCV example |
| `code_23_histogram_equalization_basic.py` | CDF-based basic histogram equalization — NumPy/Matplotlib example |
| `code_24_histogram_equalization_normalized.py` | CDFmin-normalized equalization — NumPy/Matplotlib example |
| `code_25_grayscale_clahe.py` | Grayscale adaptive contrast enhancement — OpenCV CLAHE example |
| `code_26_edge_detection.py` | Sobel, Prewitt, Roberts ও Canny — OpenCV edge detection example |
| `code_27_pixel_similarity.py` | Reference-এর সঙ্গে grayscale pixel comparison — OpenCV/NumPy example |

Example file-গুলোর algorithm ছোট function-এ আছে। সরাসরি চালালে `if __name__ == '__main__':` অংশ ছবি পড়ে ও ফল দেখায়; import করলে শুধু function পাওয়া যায়। `code_04_image_helpers.py`-এর `load_image()` ছবি পড়ে এবং `show_images()` ছবি, histogram ও ছোট matrix-এর pixel value দেখায়। OpenCV-এর BGR ছবি plot করার আগে RGB-তে বদলানো হয়।

ছবি লাগে এমন example-এ command-এর শেষে নিজের ছবির path দিন। না দিলে বর্তমান folder-এর `image.jpg` ব্যবহার হয়। Matrix example `code_22`, `code_23`, `code_24`-এ আলাদা ছবি লাগে না।

```powershell
.\.venv\Scripts\python.exe python_code\code_26_edge_detection.py "C:\Images\photo.jpg"
.\.venv\Scripts\python.exe python_code\code_27_pixel_similarity.py "C:\Images\photo.jpg" "C:\Images\reference.jpg" --tolerance 5
```

`code_17_transform_and_filter_image.py` চালালে transformed ছবি বর্তমান folder-এর `results/`-এ save হয়। Web app-এ uploaded image ব্যবহার হয়।

## চালানো ও যাচাই

Project root থেকে সহজ উপায়:

```powershell
.\start.bat
```

Backend সরাসরি চালাতে:

```powershell
.\.venv\Scripts\python.exe python_code\code_01_web_app.py
```

Tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s python_code -p "code_*_test*.py" -v
```

Catalog rebuild:

```powershell
.\.venv\Scripts\python.exe python_code/code_30_build_topic_catalog.py
```

কোড সরানোর পরে চালু থাকা server **Ctrl+C** দিয়ে বন্ধ করে `start.bat` আবার চালান। Browser-এ **Ctrl+F5** চাপুন।
