import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim



def load_image(path: str, grayscale: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imread(path, flag)


def evaluate(original: np.ndarray, processed: np.ndarray, label: str) -> None:
    """Считает PSNR и SSIM между изображениями."""
    if len(original.shape) == 3:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray, proc_gray = original, processed

    psnr_val = psnr(orig_gray, proc_gray)
    ssim_val = ssim(orig_gray, proc_gray)
    print(f"{label}: PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}")


def visualize(original: np.ndarray, noisy: np.ndarray, results: dict, title: str) -> None:
    """Визуализация фильтрации."""
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.title("Оригинал")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB) if len(original.shape) == 3 else original, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("Зашумленное")
    plt.imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB) if len(noisy.shape) == 3 else noisy, cmap="gray")
    plt.axis("off")

    for i, (name, img) in enumerate(results.items(), start=3):
        plt.subplot(2, 3, i)
        plt.title(name)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap="gray")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def add_constant_noise(image: np.ndarray, value: int = 30) -> np.ndarray:
    """Постоянный шум (добавляем ко всем пикселям)."""
    noisy = image.astype(np.float32) + value
    return np.clip(noisy, 0, 255).astype(np.uint8)

# 1
def add_gaussian_noise(image: np.ndarray, mean: float = 0, var: float = 20) -> np.ndarray:
    """Гауссовский шум."""
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = cv2.add(image.astype(np.float32), gauss)
    return np.clip(noisy, 0, 255).astype(np.uint8)


# 2
def apply_filters(noisy: np.ndarray) -> dict:
    """Фильтрация: медианный, гауссовский, билатеральный, нелокальные средние."""
    return {
        "Медианный": cv2.medianBlur(noisy, 5),
        "Гауссов": cv2.GaussianBlur(noisy, (5, 5), 1.5),
        "Билатеральный": cv2.bilateralFilter(noisy, 9, 75, 75),
        "Нелокальные средние": (
            cv2.fastNlMeansDenoisingColored(noisy, None, 10, 10, 7, 21)
            if len(noisy.shape) == 3
            else cv2.fastNlMeansDenoising(noisy, None, 10, 7, 21)
        )
    }


# 3
def experiment(original: np.ndarray, noisy: np.ndarray, noise_name: str) -> None:
    """Прогон фильтров + метрики."""
    results = apply_filters(noisy)
    print(f"\n==== {noise_name} ====")
    for name, img in results.items():
        evaluate(original, img, name)
    visualize(original, noisy, results, f"{noise_name} - фильтрация")


def main():
    img_color = load_image("img.jpg")
    img_gray = load_image("sar_1.jpg", grayscale=True)

    for original, label in [(img_color, "Цветное изображение"), (img_gray, "SAR (градации серого)")]:
        print(f"\n==== {label} ====")

        noisy_gauss = add_gaussian_noise(original)
        experiment(original, noisy_gauss, f"{label} - Гауссовский шум")

        noisy_const = add_constant_noise(original, value=30)
        experiment(original, noisy_const, f"{label} - Постоянный шум")


if __name__ == "__main__":
    main()
