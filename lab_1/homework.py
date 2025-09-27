import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error, structural_similarity as ssim


# 1
def load_image(path: str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


# 2
def plot_histogram(image: np.ndarray, title: str = "Гистограмма") -> None:
    plt.figure()
    plt.title(title)
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()


# 3
def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


# 4
def compare_images(img1: np.ndarray, img2: np.ndarray, label: str) -> None:
    mse_val = mean_squared_error(img1, img2)
    ssim_val = ssim(img1, img2)
    print(f"[{label}] MSE: {mse_val:.4f}, SSIM: {ssim_val:.4f}")


# 5
def histogram_equalization(image: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(image)


# 6
def thresholding_tests(image: np.ndarray) -> None:
    thresholds = [
        ("BINARY", cv2.THRESH_BINARY),
        ("BINARY_INV", cv2.THRESH_BINARY_INV),
        ("TRUNC", cv2.THRESH_TRUNC),
        ("TOZERO", cv2.THRESH_TOZERO),
        ("TOZERO_INV", cv2.THRESH_TOZERO_INV),
    ]

    plt.figure(figsize=(10, 6))
    for i, (name, t_type) in enumerate(thresholds, start=1):
        _, th = cv2.threshold(image, 127, 255, t_type)
        plt.subplot(2, 3, i)
        plt.imshow(th, cmap="gray")
        plt.title(name)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def main() -> None:
    # 1
    img = load_image("sar_1_gray.jpg")

    # 2
    plot_histogram(img, "Гистограмма исходного изображения")

    # 3
    gamma_low = gamma_correction(img, gamma=0.5)
    gamma_high = gamma_correction(img, gamma=1.5)

    # 4
    compare_images(img, gamma_low, "Gamma < 1")
    compare_images(img, gamma_high, "Gamma > 1")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Исходное")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Gamma < 1")
    plt.imshow(gamma_low, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Gamma > 1")
    plt.imshow(gamma_high, cmap="gray")
    plt.axis("off")
    plt.show()

    # 5
    eq_img = histogram_equalization(img)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Исходное")
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Equalized")
    plt.imshow(eq_img, cmap="gray")
    plt.axis("off")
    plt.show()

    # 6
    thresholding_tests(img)


if __name__ == "__main__":
    main()
