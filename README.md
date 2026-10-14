# PixelLab — Digital Image Processing

আপনার দেওয়া আটটি chapter PDF, Python lab code, DIP-Lab1, DIP-Lab2 ও FFT/DFT/IDFT task-এর ভিত্তিতে তৈরি local web app। এখন **১৩৪টি runnable operation** এবং **২৬৪টি source-page-linked topic** আছে। ছবি upload করে operation ও parameter বাছুন; Python backend কাজ করবে। Browser-এ ছবি, histogram/CDF, statistics, Python code এবং terminal-এর মতো ফল দেখুন।

## চালানোর নিয়ম

এই কম্পিউটারে Python environment ও প্রয়োজনীয় package প্রস্তুত করা আছে। **`start.bat`-এ double-click করুন**, অথবা PowerShell-এ **`.\start.bat`** চালান। Backend ready হলে browser-এ **http://127.0.0.1:5000** খুলবে। Terminal window খোলা রাখবেন; বন্ধ করতে Ctrl+C চাপুন। `ERR_CONNECTION_REFUSED` দেখলে launcher আবার চালিয়ে page refresh করুন।

PowerShell থেকে সরাসরি:

```powershell
.\.venv\Scripts\python.exe python_code\code_01_web_app.py
```

অন্য কম্পিউটারে source নিতে গেলে `.python`, `.venv`, `.uv-cache` copy করবেন না। Python 3.12 install করে internet সংযোগসহ `start.bat` চালান; প্রথমবার packages install হবে। Manual setup:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe python_code\code_01_web_app.py
```

## ব্যবহার

1. **Choose image** দিয়ে ছবি দিন অথবা **Try a sample** চাপুন।
2. বাম পাশ থেকে operation বাছুন; প্রয়োজন হলে parameter দিন।
3. **Run operation** চাপুন। Original ও processed image পাশাপাশি দেখুন।
4. **Image insights**-এ grayscale histogram, mean intensity ও RMS contrast দেখুন।
   **Lab output** tab-এ Python lab code-এর মতো Original/Processed image, histogram ও CDF-এর ছয়টি Matplotlib plot এবং terminal-এর মতো printed result দেখুন। **Save figure** দিয়ে পুরো figure PNG হিসেবে download করুন; figure-এ click করলে full size খুলবে। ছোট image-এর actual pixel matrix দেখায়; বড় image-এর ক্ষেত্রে console-এ স্পষ্টভাবে চিহ্নিত 8×8 resampled sample দেখায়।
5. **Algorithm & code**-এ formula, course reference ও backend function দেখুন।
6. **Download** দিয়ে PNG/JPEG result save করুন। **Use as next input** দিয়ে পরের operation চালান। **Reset to uploaded image** দিয়ে মূল ছবিতে ফিরুন।
7. বাম পাশের **CHAPTER** dropdown দিয়ে operation তালিকা ছোট করুন। Search-এ নাম বা কাজ লিখে operation খুঁজুন। প্রতিটির ব্যাখ্যা, formula ও source code **Algorithm & code** tab-এ দেখা যায়।
8. Coding operation-এ **Download encoded file** দিয়ে self-describing educational JSON bitstream save করুন। Result দেখানোর আগে backend সেই stream স্বাধীন decoder দিয়ে decode করে মিলিয়ে দেখে। File-এ shape, coding metadata এবং base64 payload থাকে।

UI-তে তিনটি step আপনার অগ্রগতি দেখায়: **Add an image → Choose & run → Compare & save**। **Common operations** থেকে Grayscale, CLAHE, Resize, Canny ও Gaussian blur বাছতে পারেন। প্রতিটি parameter-এর range/default hint আছে; **Reset settings** দিয়ে default-এ ফিরুন। ছবির কোণার **⤢** button বড় preview খোলে; Escape বা Close দিয়ে বন্ধ করুন। Mobile-এ **Browse operations** দিয়ে search ও chapter menu খুলুন। Search operation-এর নামের পাশাপাশি তার explanation-ও খোঁজে; **Clear filters** দিয়ে সব operation ফিরিয়ে আনুন। Tab-গুলোর মধ্যে keyboard-এর Left/Right/Home/End দিয়ে চলা যায়।

Pixel similarity, quality metrics, histogram specification, arithmetic/Boolean combination, registration এবং two-frame coding-এ দ্বিতীয় reference image লাগবে। Reference প্রয়োজন হলে upload field নিজে দেখা যাবে। Pixel similarity aligned grayscale pixel তুলনা করে; JPEG compression প্রকৃত JPEG দেয়। History বর্তমান page session-এ সর্বশেষ ১২টি experiment রাখে।

## Chapter coverage

| Chapter | Catalog topics | Runnable-topic entries | Concept lessons |
| --- | ---: | ---: | ---: |
| 1. Digital Image Fundamentals | 36 | 20 | 16 |
| 2. Image Enhancement | 35 | 27 | 8 |
| 3. Color Image Processing | 26 | 23 | 3 |
| 4. Image Transforms | 29 | 20 | 9 |
| 5. Image Segmentation | 47 | 35 | 12 |
| 6. Image Restoration | 44 | 27 | 17 |
| 7. Image Compression | 24 | 20 | 4 |
| 8. Morphological Processing | 23 | 18 | 5 |
| **Total** | **264** | **190** | **74** |

একই parameterized operation-এর মাধ্যমে একাধিক related topic চলে; তাই ১৯০টি algorithm-topic entry এবং ১৩৪টি operation-এর count আলাদা। নিচে কয়েকটি বড় group:

- Fundamentals: entropy/local entropy, quality metrics, 4/8/m connectivity, shortest paths, palette/quantization, Lloyd–Max, projection features, Gaussian classifier, pupil centroid, fixed-glyph rule classification ও acquisition/perception demos.
- Enhancement: AHE/CLAHE, histogram specification, custom convolution/correlation, separable kernels, derivatives, order filters, ensemble averaging, ideal/Butterworth/Gaussian pass/reject filters, homomorphic correction ও fingerprint orientation/thinning.
- Color: RGB, CMY/CMYK, HSI/HSV, Lab, XYZ, YCbCr/YUV conversions/round-trips, chromaticity, spectral teaching integrals, white balance, color segmentation, color filtering, equalization, tonal correction, pseudocolor ও chroma subsampling.
- Transforms: affine geometry, pyramids, bit planes, FFT components, DCT/IDCT, Haar/db4/db8 wavelets, Walsh/Hadamard, PCA/KLT, convolution theorem এবং synthetic signal/3D point demos.
- Segmentation: all basic edge operators, LoG/DoG, ISODATA, adaptive mean/Gaussian, Hough/lane pipeline, LBP/LPG, region growing, K-means/FCM, split/merge, watershed, snakes/level sets, contours এবং Dice/IoU/precision/recall.
- Restoration: eight noise families, four mean estimators, inverse/Wiener deconvolution, adaptive Wiener, diffusion, NLM, wavelet denoising, GLCM/shape features, synthetic CT Radon/FBP/SART, MRI k-space, windowing, Rician/phase/bias এবং translation-fusion demos.
- Compression: RLE, Huffman, 32-bit arithmetic coding, LZW, Golomb/Rice/Exp-Golomb encode/decode, DPCM, VQ, block DCT/quantization/zigzag এবং two-frame MPEG/H.263 building-block demos.
- Morphology: binary/grayscale erosion/dilation/opening/closing/gradient, Boolean masks, hit-or-miss, both top-hats, skeleton/thinning/pruning, distance/boundary/hole filling এবং reconstruction.

**Coverage-এর অর্থ:** classical algorithm-গুলো চালানো যায়। Physics, human vision, neural-network architecture, medical workflow ও standardized codec survey-র topic-এ lesson এবং related teaching experiment আছে। Trained GazeNet/U-Net/Mask R-CNN/DeepFace, clinical diagnosis, full ART/SIRT/MLEM scanners বা standards-compliant MPEG/H.263/JPEG2000 codec আছে বলে দাবি করা হয়নি। তাদের lesson-এ কী বোঝানো হয়েছে এবং demo কী করে তা স্পষ্ট লেখা আছে।

কিছু algorithm সময় ও memory নিয়ন্ত্রণে grayscale preview-তে চলে (সাধারণত longest side 128–1024); output notes-এ analysis resolution উল্লেখ থাকে। Coding losslessness ওই grayscale analysis image-এর তুলনায়, original full-color upload-এর তুলনায় নয়। Spectrum/pattern/label map photo নয়; সংশ্লিষ্ট display explanation পড়ুন। FFT efficiently DFT গণনা করে; spectrum display complex coefficients-এর normalized log magnitude।

## Backend কীভাবে কাজ করে

সব project Python source **`python_code/`**-এ আছে। কোন file ও কোন type-এর code কী কাজে লাগে তার বাংলা বিবরণ: **[Python code guide](python_code/README.md)**। **Algorithm & code** tab-এ actual implementation path ও module libraries দেখা যায়।

`Browser → POST /api/process → validation → python_code algorithms → OpenCV/NumPy → JSON result → browser`

| File | কাজ |
| --- | --- |
| `python_code/code_01_web_app.py` | Flask routes, uploads, image decoding/encoding, API errors |
| `python_code/code_02_image_operations.py` | Operation registry, parameter validation, processing functions, histogram/statistics |
| `python_code/code_03_operation_loader.py` | একই ফোল্ডারের algorithm file import করে operation register করে |
| `data/course_topics.json` | ২৬৪টি topic, source PDF pages, lesson, operation ও parameter preset |
| `data/source_topics_1_4.json`, `data/source_topics_5_8.json` | Reviewed PDF topic inventories |
| `pdfs/` | সব chapter, lab, task, class-test ও course outline PDF |
| `python_code/code_30_build_topic_catalog.py` | Source inventories থেকে catalog rebuild ও mapping validation |
| `python_code/code_15_plot_results.py` | Matplotlib figure, pixel matrix ও Python-style console output |
| `templates/index.html` | App interface |
| `static/app.js` | Upload, controls, API calls, graph, download, session history |
| `static/style.css` | Desktop/mobile responsive design |
| `static/usability.css`, `static/plain.css` | সহজ controls ও plain academic lab layout |
| `python_code/code_29_test_web_app.py` | API, pixel math, invalid inputs, compression and reconstruction verification |
| `python_code/code_28_test_algorithms.py` | Algorithm variants, topic presets, codec round-trips, transforms ও morphology properties |
| `start.bat` | Windows launcher |

ছোট example scripts-ও `python_code/`-এ রাখা হয়েছে। ছবির path command-line-এ দেওয়া যায়; web app uploaded image ব্যবহার করে। Example-এ ছোট function ও shared load/plot helper আছে। চালানোর নিয়ম [Python code guide](python_code/README.md)-এ আছে। Backend internally BGR রাখে এবং display-এর জন্য PNG/JPEG encode করে; contrast calculation-এ floating point ব্যবহার করে uint8 overflow এড়ানো হয়েছে। Negative brightness offset-এ clipping ব্যবহার করা হয়েছে যাতে অন্ধকার করার সময় absolute value নিয়ে উল্টো উজ্জ্বল না হয়। Constant image stretching/equalization-এ division by zero এড়ানো হয়েছে।

Upload-এর মোট limit 24 MB; input ও resize output সর্বোচ্চ 12 million pixels। Image orientation EXIF অনুযায়ী ঠিক করা হয়; input RGB 8-bit-এ convert হয়। Transparency, animation, TIFF-এর অতিরিক্ত page ও original metadata রাখা হয় না। Histogram/statistics grayscale luminance-এর ওপর গণনা করা হয়। Images server-এ স্থায়ীভাবে save হয় না।

## Teacher demo

**Try a sample → Grayscale → Normalized equalization → Canny edges** দিয়ে আলাদা ফল দেখান। পরের operation একই result-এ চালাতে **Use as next input** চাপুন। **Digital pixel matrix**-এ 8×8 sampling দেখান। **DFT → IDFT reconstruction**-এ reconstruction error দেখান। প্রতিটির **Algorithm & code** tab খুলে formula ও function ব্যাখ্যা করুন। নিজের name/student ID presentation-এ যোগ করুন।

সব chapter-এর operation দেখাতে sidebar-এর **CHAPTER** dropdown ও search ব্যবহার করুন। Compression-এর Huffman/Arithmetic/LZW-তে `lossless_round_trip: True` এবং actual export size দেখান। Wavelet/PCA-তে reconstruction, segmentation-এ mask metrics, morphology-তে top-hat/skeleton এবং restoration-এ synthetic noise/known-PSF demo দেখান।

## Verification

এই কম্পিউটারে **১৮টি automated test** পাস করেছে। সব **১৩৪টি operation** API দিয়ে, chapter algorithms-এর dropdown variant-গুলো এবং সব **২৬৪টি topic-এর preset** execute করে পরীক্ষা করা হয়েছে। Codec decode-এ constant/alternating/random sequences, complete export decoding, transform reconstruction, mask metrics ও top-hat properties যাচাই করা হয়েছে। Chrome browser-এ chapter/topic filters, source PDF links, preset loading, custom kernel validation, encoded-file download, reference metrics, wavelet/watershed, Matplotlib/console, result chaining এবং 390px mobile layout পাস করেছে।

App update করার পরে চালু থাকা terminal-এ **Ctrl+C** চাপুন, তারপর **`.\start.bat`** আবার চালান এবং browser-এ **Ctrl+F5** দিয়ে refresh করুন। পুরোনো server চালু থাকলে নতুন Python code বা cached template load নাও হতে পারে।

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s python_code -p "code_*_test*.py" -v
node --check static/app.js
```

Catalog rebuild:

```powershell
.\.venv\Scripts\python.exe python_code/code_30_build_topic_catalog.py
```

Algorithm-library references: [scikit-image restoration](https://scikit-image.org/docs/stable/api/skimage.restoration.html), [segmentation](https://scikit-image.org/docs/stable/api/skimage.segmentation.html), [transforms](https://scikit-image.org/docs/stable/api/skimage.transform.html), [PyWavelets DWT/IDWT](https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html).

Server আপনার কম্পিউটারের `127.0.0.1` address-এ চলে। App চালানোর পর image processing-এর জন্য internet লাগে না। এটি local class demonstration-এর জন্য প্রস্তুত; public hosting configuration অন্তর্ভুক্ত নয়।
