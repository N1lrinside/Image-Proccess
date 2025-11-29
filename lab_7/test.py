import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from collections import Counter
import os

# === ШАГ 1: загрузка изображения и перевод в монохромное ===
image = cv.imread("sar_1_gray.jpg")             # загружаем
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # в серый
_, mono = cv.threshold(image_gray, 128, 255, cv.THRESH_BINARY)  # бинаризация

print(">>> Шаг 1: изображение конвертировано в монохромное")

# Сохранение монохромного
mono.tofile("mono.bin")
np.savetxt("mono.txt", mono, fmt="%d")
print("Сохранено в файлы mono.bin и mono.txt")

# Визуализация этапа 1
plt.figure(figsize=(12,7))
plt.subplot(1,3,1); plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB)); plt.title("Исходное")
plt.axis("off")
plt.subplot(1,3,2); plt.imshow(image_gray, cmap="gray"); plt.title("Оттенки серого")
plt.axis("off")
plt.subplot(1,3,3); plt.imshow(mono, cmap="gray"); plt.title("Монохромное")
plt.axis("off")
plt.show()


# ========= Шаг 2. Вейвлет-преобразование Хаара (другой вариант реализации) =========

def haar2d(data):
    data = data.astype(np.float32)
    h, w = data.shape
    h -= h % 2
    w -= w % 2
    data = data[:h, :w]

    row_pass = np.zeros_like(data)
    for r in range(h):
        for c in range(0, w, 2):
            row_pass[r, c//2] = (data[r, c] + data[r, c+1]) / 2
            row_pass[r, c//2 + w//2] = (data[r, c] - data[r, c+1]) / 2

    col_pass = np.zeros_like(row_pass)
    for c in range(w):
        for r in range(0, h, 2):
            col_pass[r//2, c] = (row_pass[r, c] + row_pass[r+1, c]) / 2
            col_pass[r//2 + h//2, c] = (row_pass[r, c] - row_pass[r+1, c]) / 2

    return (
        col_pass[:h//2, :w//2],
        col_pass[:h//2, w//2:],
        col_pass[h//2:, :w//2],
        col_pass[h//2:, w//2:],
        col_pass
    )


LL, LH, HL, HH, full = haar2d(image_gray)

plt.figure(figsize=(13, 8))
plt.subplot(2, 3, 1); plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB)); plt.title("Исходное"); plt.axis("off")
plt.subplot(2, 3, 2); plt.imshow(LL, cmap="gray"); plt.title("LL"); plt.axis("off")
plt.subplot(2, 3, 3); plt.imshow(full, cmap="jet"); plt.title("Полное вейвлет-преобразование Хаара"); plt.axis("off")
plt.subplot(2, 3, 4); plt.imshow(LH, cmap="coolwarm"); plt.title("LH"); plt.axis("off")
plt.subplot(2, 3, 5); plt.imshow(HL, cmap="coolwarm"); plt.title("HL"); plt.axis("off")
plt.subplot(2, 3, 6); plt.imshow(HH, cmap="coolwarm"); plt.title("HH"); plt.axis("off")
plt.show()


# ========= Шаг 3. Квантование высокочастотных компонент =========

def quantize_block(arr, q_levels = 4):
    mn, mx = np.min(arr), np.max(arr)
    scale = (mx - mn) / (q_levels - 1)
    quant = np.round((arr - mn) / scale).astype(int)
    return quant, mn, scale

Q_LH, base_LH, step_LH = quantize_block(LH)
Q_HL, base_HL, step_HL = quantize_block(HL)
Q_HH, base_HH, step_HH = quantize_block(HH)

LH_restored = base_LH + Q_LH * step_LH
HL_restored = base_HL + Q_HL * step_HL
HH_restored = base_HH + Q_HH * step_HH

plt.figure(figsize=(18, 10))

plt.subplot(2, 3, 1)
plt.imshow(LH, cmap="coolwarm")
plt.title("LH до квантования")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(HL, cmap="coolwarm")
plt.title("HL до квантования")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(HH, cmap="coolwarm")
plt.title("HH до квантования")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(LH_restored, cmap="coolwarm")
plt.title("LH после квантования")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(HL_restored, cmap="coolwarm")
plt.title("HL после квантования")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(HH_restored, cmap="coolwarm")
plt.title("HH после квантования")
plt.axis("off")

plt.tight_layout()
plt.show()

print("[INFO] Компоненты квантованы")


# ========= Шаг 4. RLE-сжатие =========

def rle_counter(data):
    flat = data.flatten()
    return [(int(v), int(c)) for v, c in Counter(flat).items()]


rle_LH = rle_counter(Q_LH)
rle_HL = rle_counter(Q_HL)
rle_HH = rle_counter(Q_HH)

with open("haar_rle.txt", "w") as f:
    f.write("LL\n")
    np.savetxt(f, LL.astype(int), fmt="%d")
    f.write("\nLH\n")
    [f.write(f"{v} {c}\n") for v, c in rle_LH]
    f.write("\nHL\n")
    [f.write(f"{v} {c}\n") for v, c in rle_HL]
    f.write("\nHH\n")
    [f.write(f"{v} {c}\n") for v, c in rle_HH]

print("Результат записан в haar_rle.txt")

orig_sz = len(open("mono.txt").read().encode("utf-8"))
comp_sz = len(open("haar_rle.txt").read().encode("utf-8"))

print(f"Исходный текстовый файл: {orig_sz} байт")
print(f"После Хаара + квантования + RLE: {comp_sz} байт")
print(f"Сжатие: {orig_sz/comp_sz:.2f} раз")