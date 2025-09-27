import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def load_image(path: str) -> np.ndarray:

    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


# 1
def detect_longest_line(image: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]], float]:
    """Находит линии на изображении с помощью преобразования Хафа и выделяет самую длинную."""
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                            threshold=100, minLineLength=80, maxLineGap=10)

    img_lines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    longest_line = None
    max_len = 0.0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > max_len:
                max_len = length
                longest_line = (x1, y1, x2, y2)
            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 1)

        if longest_line:
            x1, y1, x2, y2 = longest_line
            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 3)

    return img_lines, longest_line, max_len


# 2
def apply_binarization_methods(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Применяет разные методы бинаризации к изображению:
    - Простая пороговая
    - Otsu
    - Адаптивная
    """

    _, binary_simple = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    _, binary_otsu = cv2.threshold(image, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_adapt = cv2.adaptiveThreshold(image, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 35, 10)

    return binary_simple, binary_otsu, binary_adapt


def visualize_results(original: np.ndarray,
                      edges: np.ndarray,
                      img_lines: np.ndarray,
                      binary_simple: np.ndarray,
                      binary_otsu: np.ndarray,
                      binary_adapt: np.ndarray) -> None:
    """
    Визуализирует результаты анализа.
    """

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.title("Исходное изображение")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("Контуры (Canny)")
    plt.imshow(edges, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title("Линии Хафа (Пункт 1)")
    plt.imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title("Пороговая (Пункт 2)")
    plt.imshow(binary_simple, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title("Otsu (Пункт 2)")
    plt.imshow(binary_otsu, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.title("Адаптивная (Пункт 2)")
    plt.imshow(binary_adapt, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    img = load_image("sar_3.jpg")

    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    # 1
    img_lines, longest_line, line_len = detect_longest_line(img)
    print(f"Самая длинная линия: {longest_line}, длина={line_len:.2f} px")

    # 2
    binary_simple, binary_otsu, binary_adapt = apply_binarization_methods(img)

    visualize_results(img, edges, img_lines, binary_simple, binary_otsu, binary_adapt)


if __name__ == "__main__":
    main()
