import os
import numpy as np
import cv2
import pydicom
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

def get_brightness(img):
    return np.mean(img)

def get_rms_contrast(img):
    return np.std(img)

def get_michelson_contrast(img):
    i_min, i_max = np.min(img), np.max(img)
    return (i_max - i_min) / (i_max + i_min + 1e-5)

def get_laplacian_variance(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def get_tenengrad(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    return np.sqrt(gx**2 + gy**2).sum()

def estimate_noise(img):
    h, w = img.shape
    central_region = img[h//4:3*h//4, w//4:3*w//4]
    blurred = cv2.GaussianBlur(central_region, (5, 5), 0)
    residual = central_region.astype(np.float32) - blurred.astype(np.float32)
    return np.std(residual)

def static_preprocessing(img):
    equalized = cv2.equalizeHist(img)
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    return cv2.filter2D(equalized, -1, kernel)

def adaptive_preprocessing(img, brightness, contrast, sharpness, noise):
    processed = img.copy()
    if contrast < 40 or get_michelson_contrast(img) < 0.8:
        processed = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(processed)
    if sharpness < 100 or get_tenengrad(processed) < 500:
        processed = cv2.filter2D(processed, -1, np.array([[0,-1,0], [-1,5,-1], [0,-1,0]]))
    if noise > 5:
        processed = cv2.fastNlMeansDenoising(processed, h=10)
    if brightness < 100 or brightness > 200:
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
    return processed

def evaluate(original, processed):
    return {
        'SSIM': ssim(original, processed),
        'PSNR': psnr(original, processed)
    }

def save_adaptive_output(adaptive_img, fname):
    os.makedirs("processed", exist_ok=True)
    cv2.imwrite(os.path.join("processed", f"{fname}.png"), adaptive_img)

def auto_label(brightness, contrast, sharpness, noise):
    if brightness < 80:
        return 'underexposed'
    elif brightness > 200:
        return 'overexposed'
    elif contrast < 30:
        return 'low_contrast'
    elif sharpness < 50 or noise > 6:
        return 'blurry'
    return 'good'

def train_quality_classifier(X, y):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("\nModel Performance (Random Forest Classifier):")
    print(classification_report(y_test, preds))
    return clf

def classify_quality(clf, metrics):
    return clf.predict([metrics])[0]

def process_folder(input_dir, clf=None, collect_labels=False):
    dcm_files = glob(os.path.join(input_dir, "*.dcm"))
    jpg_files = glob(os.path.join(input_dir, "*.png"))
    all_files = dcm_files + jpg_files

    results, X, y = [], [], []

    for file_path in all_files:
        if file_path.endswith(".dcm"):
            ds = pydicom.dcmread(file_path)
            img = ds.pixel_array.astype(np.uint8)
        else:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

        brightness = get_brightness(img)
        contrast = get_rms_contrast(img)
        sharpness = get_laplacian_variance(img)
        noise = estimate_noise(img)

        static_img = static_preprocessing(img)
        adaptive_img = adaptive_preprocessing(img, brightness, contrast, sharpness, noise)

        static_eval = evaluate(img, static_img)
        adapt_eval = evaluate(img, adaptive_img)

        filename = os.path.splitext(os.path.basename(file_path))[0]
        save_adaptive_output(adaptive_img, filename)

        label = auto_label(brightness, contrast, sharpness, noise) if collect_labels else classify_quality(clf, [brightness, contrast, sharpness, noise]) if clf else 'unlabeled'
        if collect_labels:
            X.append([brightness, contrast, sharpness, noise])
            y.append(label)

        results.append({
            "File": filename,
            "Brightness": brightness,
            "RMS_Contrast": contrast,
            "Sharpness": sharpness,
            "Noise": noise,
            "QualityLabel": label,
            "SSIM_Static": static_eval["SSIM"],
            "PSNR_Static": static_eval["PSNR"],
            "SSIM_Adaptive": adapt_eval["SSIM"],
            "PSNR_Adaptive": adapt_eval["PSNR"]
        })

    return results, X, y

if __name__ == "__main__":
    # Step 1: Set path to your input images folder
    input_folder = "Dataset"

    # Step 2: Auto-label dataset using image quality rules
    results, X, y = process_folder(input_folder, collect_labels=True)

    # Step 3: Train classifier on extracted features
    clf = train_quality_classifier(X, y)

    # Step 4: Rerun with model for prediction-based routing
    final_results, _, _ = process_folder(input_folder, clf=clf)

    # Step 5: Print final results
    for res in final_results:
        print(res)
