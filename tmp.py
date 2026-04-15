import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import json
from datetime import datetime

# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

# Глобальный seed для воспроизводимости
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)


# =============================================================================
# УТИЛИТЫ
# =============================================================================

def load_image(path: str) -> np.ndarray:
    """Загрузка изображения"""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(image: np.ndarray, path: str):
    """Сохранение изображения"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr)


def apply_transform(image: np.ndarray, transform: A.Compose) -> np.ndarray:
    """Применение трансформации к изображению"""
    transformed = transform(image=image)
    return transformed['image']


# =============================================================================
# 1. ГЕОМЕТРИЧЕСКИЕ ТРАНСФОРМАЦИИ
# =============================================================================

def create_shear_transform(
    shear: float = 0,
    border_value: int = 0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Сдвиг (shear) изображения.
    
    Args:
        shear: Угол сдвига в градусах (0-45)
        border_value: Значение заполнения границ
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с трансформацией shear
    """
    return A.Compose([
        A.Affine(
            scale=1.0,
            translate_percent=0,
            rotate=0,
            shear=(shear, shear),
            p=1,
            cval=border_value
        )
    ], seed=seed)


def create_perspective_transform(
    scale: float = 0.05,
    pad_value: int = 0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Перспективные искажения.
    
    Args:
        scale: Коэффициент перспективного искажения (0.0-0.5)
        pad_value: Значение заполнения границ
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с перспективной трансформацией
    """
    return A.Compose([
        A.Perspective(
            scale=(scale, scale),
            p=1,
            pad_val=pad_value
        )
    ], seed=seed)


def create_elastic_transform(
    alpha: float = 1,
    sigma: float = 50,
    border_value: int = 0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Эластичные деформации.
    
    Args:
        alpha: Коэффициент эластичности (0-10)
        sigma: Стандартное отклонение для гауссова ядра (30-100)
        border_value: Значение заполнения границ
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с эластичной трансформацией
    """
    return A.Compose([
        A.ElasticTransform(
            alpha=alpha,
            sigma=sigma,
            alpha_affine=0,
            p=1,
            border_mode=cv2.BORDER_CONSTANT,
            value=border_value
        )
    ], seed=seed)


def create_grid_distortion_transform(
    num_steps: int = 5,
    distort_limit: float = 0.1,
    border_value: int = 0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Искажение сетки.
    
    Args:
        num_steps: Количество шагов сетки (1-10)
        distort_limit: Лимит искажения (0.0-0.5)
        border_value: Значение заполнения границ
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с grid distortion
    """
    return A.Compose([
        A.GridDistortion(
            num_steps=num_steps,
            distort_limit=(distort_limit, distort_limit),
            p=1,
            border_mode=cv2.BORDER_CONSTANT,
            value=border_value
        )
    ], seed=seed)


def create_optical_distortion_transform(
    distort_limit: float = 0.1,
    shift_limit: float = 0,
    border_value: int = 0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Оптические искажения.
    
    Args:
        distort_limit: Лимит искажения (0.0-1.0)
        shift_limit: Лимит сдвига (0.0-0.5)
        border_value: Значение заполнения границ
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с оптической дисторсией
    """
    return A.Compose([
        A.OpticalDistortion(
            distort_limit=(distort_limit, distort_limit),
            shift_limit=(shift_limit, shift_limit),
            p=1,
            border_mode=cv2.BORDER_CONSTANT,
            value=border_value
        )
    ], seed=seed)


def create_shift_scale_rotate_transform(
    shift_limit: float = 0.1,
    scale_limit: float = 0.1,
    rotate_limit: float = 15,
    border_value: int = 0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Комбинированная трансформация: сдвиг + масштаб + поворот.
    
    Args:
        shift_limit: Лимит сдвига (0.0-0.5)
        scale_limit: Лимит масштабирования (0.0-0.5)
        rotate_limit: Лимит поворота в градусах (0-180)
        border_value: Значение заполнения границ
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с комбинированной геометрической трансформацией
    """
    return A.Compose([
        A.ShiftScaleRotate(
            shift_limit=(shift_limit, shift_limit),
            scale_limit=(scale_limit, scale_limit),
            rotate_limit=(rotate_limit, rotate_limit),
            p=1,
            border_mode=cv2.BORDER_CONSTANT,
            value=border_value
        )
    ], seed=seed)


# =============================================================================
# 2. ЦВЕТОВЫЕ ТРАНСФОРМАЦИИ
# =============================================================================
def create_brightness_contrast_transform(
    brightness: float = 0.0,
    contrast: float = 0.0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Комбинированное изменение яркости и контраста.
    
    Args:
        brightness: Коэффициент яркости (-1.0 до 1.0)
        contrast: Коэффициент контраста (-1.0 до 1.0)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с трансформацией яркости и контраста
    """
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=(brightness, brightness),
            contrast_limit=(contrast, contrast),
            p=1
        )
    ], seed=seed)


def create_hsv_transform(
    hue_shift: int = 0,
    sat_shift: float = 0.0,
    val_shift: float = 0.0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Комбинированная HSV трансформация.
    
    Args:
        hue_shift: Сдвиг тона (0-180)
        sat_shift: Сдвиг насыщенности (-1.0 до 1.0)
        val_shift: Сдвиг значения (-1.0 до 1.0)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с HSV трансформацией
    """
    return A.Compose([
        A.HueSaturationValue(
            hue_shift_limit=(hue_shift, hue_shift),
            sat_shift_limit=(int(sat_shift * 255), int(sat_shift * 255)),
            val_shift_limit=(int(val_shift * 255), int(val_shift * 255)),
            p=1
        )
    ], seed=seed)


def create_rgb_shift_transform(
    r_shift: int = 0,
    g_shift: int = 0,
    b_shift: int = 0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Сдвиг RGB каналов.
    
    Args:
        r_shift: Сдвиг красного канала (0-100)
        g_shift: Сдвиг зелёного канала (0-100)
        b_shift: Сдвиг синего канала (0-100)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с RGB shift
    """
    return A.Compose([
        A.RGBShift(
            r_shift_limit=(r_shift, r_shift),
            g_shift_limit=(g_shift, g_shift),
            b_shift_limit=(b_shift, b_shift),
            p=1
        )
    ], seed=seed)


def create_gamma_transform(
    gamma: float = 1.0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Гамма-коррекция.
    
    Args:
        gamma: Коэффициент гаммы (0.3-3.0)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с гамма-коррекцией
    """
    return A.Compose([
        A.RandomGamma(
            gamma_limit=(gamma, gamma),
            p=1
        )
    ], seed=seed)


def create_clahe_transform(
    clip_limit: float = 4.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        clip_limit: Лимит отсечения (1.0-10.0)
        tile_grid_size: Размер сетки для гистограммы
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с CLAHE
    """
    return A.Compose([
        A.CLAHE(
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
            p=1
        )
    ], seed=seed)


def create_solarize_transform(
    threshold: int = 128,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Соляризация.
    
    Args:
        threshold: Порог инверсии (0-255)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с соляризацией
    """
    return A.Compose([
        A.Solarize(
            threshold=threshold,
            p=1
        )
    ], seed=seed)


def create_posterize_transform(
    num_bits: int = 4,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Постеризация.
    
    Args:
        num_bits: Количество бит на канал (1-8)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с постеризацией
    """
    return A.Compose([
        A.Posterize(
            num_bits=num_bits,
            p=1
        )
    ], seed=seed)


def create_equalize_transform(
    by_channels: bool = True,
    mode: str = 'cv',
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Эквализация гистограммы.
    
    Args:
        by_channels: Применять к каждому каналу отдельно
        mode: Режим эквализации ('cv' или 'pil')
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с эквализацией
    """
    return A.Compose([
        A.Equalize(
            p=1,
            mode=mode,
            by_channels=by_channels
        )
    ], seed=seed)


def create_invert_transform(
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Инверсия цветов.
    
    Args:
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с инверсией
    """
    return A.Compose([
        A.InvertImg(p=1)
    ], seed=seed)


def create_to_gray_transform(
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Перевод в оттенки серого.
    
    Args:
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с конвертацией в gray
    """
    return A.Compose([
        A.ToGray(p=1)
    ], seed=seed)


def create_channel_shuffle_transform(
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Перемешивание каналов.
    
    Args:
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с channel shuffle
    """
    return A.Compose([
        A.ChannelShuffle(p=1)
    ], seed=seed)


def create_to_sepia_transform(
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Эффект сепии.
    
    Args:
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с сепией
    """
    return A.Compose([
        A.ToSepia(p=1)
    ], seed=seed)


# =============================================================================
# 3. ТРАНСФОРМАЦИИ РАЗМЫВАНИЯ И РЕЗКОСТИ
# =============================================================================

def create_blur_transform(
    kernel_size: int = 5,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Размытие (Blur).
    
    Args:
        kernel_size: Размер ядра размытия (нечётное число, 3-21)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с blur
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return A.Compose([
        A.Blur(
            blur_limit=(kernel_size, kernel_size),
            p=1
        )
    ], seed=seed)


def create_gaussian_blur_transform(
    kernel_size: int = 5,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Гауссово размытие.
    
    Args:
        kernel_size: Размер ядра (нечётное число, 3-21)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с gaussian blur
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return A.Compose([
        A.GaussianBlur(
            blur_limit=(kernel_size, kernel_size),
            p=1
        )
    ], seed=seed)


def create_median_blur_transform(
    kernel_size: int = 5,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Медианное размытие.
    
    Args:
        kernel_size: Размер ядра (нечётное число, 3-21)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с median blur
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return A.Compose([
        A.MedianBlur(
            blur_limit=(kernel_size, kernel_size),
            p=1
        )
    ], seed=seed)


def create_motion_blur_transform(
    kernel_size: int = 5,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Размытие в движении.
    
    Args:
        kernel_size: Размер ядра (нечётное число, 3-31)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с motion blur
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return A.Compose([
        A.MotionBlur(
            blur_limit=(kernel_size, kernel_size),
            p=1
        )
    ], seed=seed)


def create_sharpen_transform(
    alpha: float = 0.5,
    lightness: float = 1.0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Повышение резкости.
    
    Args:
        alpha: Сила эффекта (0.0-1.0)
        lightness: Яркость (0.0-2.0)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с sharpen
    """
    return A.Compose([
        A.Sharpen(
            alpha=(alpha, alpha),
            lightness=(lightness, lightness),
            p=1
        )
    ], seed=seed)


def create_unsharp_mask_transform(
    alpha: float = 0.5,
    blur_limit: int = 5,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Маска нерезкости.
    
    Args:
        alpha: Сила эффекта (0.0-2.0)
        blur_limit: Лимит размытия (нечётное число, 3-21)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с unsharp mask
    """
    if blur_limit % 2 == 0:
        blur_limit += 1
    return A.Compose([
        A.UnsharpMask(
            alpha=(alpha, alpha),
            blur_limit=(blur_limit, blur_limit),
            p=1
        )
    ], seed=seed)


def create_emboss_transform(
    alpha: float = 0.5,
    strength: float = 0.5,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Эффект тиснения (Emboss).
    
    Args:
        alpha: Смешивание с оригиналом (0.0-1.0)
        strength: Сила эффекта (0.0-1.0)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с emboss
    """
    return A.Compose([
        A.Emboss(
            alpha=(alpha, alpha),
            strength=(strength, strength),
            p=1
        )
    ], seed=seed)


# =============================================================================
# 4. ТРАНСФОРМАЦИИ ШУМА
# =============================================================================

def create_gauss_noise_transform(
    var_limit: float = 50.0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Гауссов шум.
    
    Args:
        var_limit: Дисперсия шума (0.0-500.0)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с gaussian noise
    """
    return A.Compose([
        A.GaussNoise(
            var_limit=(var_limit, var_limit),
            p=1
        )
    ], seed=seed)


def create_multiplicative_noise_transform(
    multiplier: float = 1.0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Мультипликативный шум.
    
    Args:
        multiplier: Множитель шума (0.5-1.5)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с multiplicative noise
    """
    return A.Compose([
        A.MultiplicativeNoise(
            multiplier=(multiplier, multiplier),
            p=1
        )
    ], seed=seed)


def create_iso_noise_transform(
    intensity: float = 0.1,
    color_shift: float = 0.01,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    ISO шум (цветной шум).
    
    Args:
        intensity: Интенсивность шума (0.0-0.5)
        color_shift: Сдвиг цвета (0.0-0.1)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с iso noise
    """
    return A.Compose([
        A.IsoNoise(
            color_shift=(color_shift, color_shift),
            intensity=(intensity, intensity),
            p=1
        )
    ], seed=seed)


# =============================================================================
# 5. ТРАНСФОРМАЦИИ ВЫРЕЗАНИЯ (DROPOUT)
# =============================================================================

def create_coarse_dropout_transform(
    max_holes: int = 5,
    max_height: int = 32,
    max_width: int = 32,
    fill_value: int = 0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Coarse Dropout (Cutout).
    
    Args:
        max_holes: Максимальное количество отверстий (1-20)
        max_height: Максимальная высота отверстия (8-128)
        max_width: Максимальная ширина отверстия (8-128)
        fill_value: Значение заполнения
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с coarse dropout
    """
    return A.Compose([
        A.CoarseDropout(
            max_holes=max_holes,
            max_height=max_height,
            max_width=max_width,
            fill_value=fill_value,
            p=1
        )
    ], seed=seed)


def create_grid_dropout_transform(
    ratio: float = 0.1,
    unit_size_min: int = 10,
    unit_size_max: int = 10,
    fill_value: int = 0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Grid Dropout.
    
    Args:
        ratio: Доля вырезаемых областей (0.0-0.5)
        unit_size_min: Минимальный размер ячейки
        unit_size_max: Максимальный размер ячейки
        fill_value: Значение заполнения
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с grid dropout
    """
    return A.Compose([
        A.GridDropout(
            ratio=ratio,
            unit_size_min=unit_size_min,
            unit_size_max=unit_size_max,
            fill_value=fill_value,
            p=1
        )
    ], seed=seed)


# =============================================================================
# 6. ТРАНСФОРМАЦИИ СЖАТИЯ И АРТЕФАКТОВ
# =============================================================================

def create_jpeg_compression_transform(
    quality: int = 75,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    JPEG сжатие.
    
    Args:
        quality: Качество JPEG (10-100)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с JPEG компрессией
    """
    return A.Compose([
        A.ImageCompression(
            quality_lower=quality,
            quality_upper=quality,
            compression_type=0,  # JPEG
            p=1
        )
    ], seed=seed)


def create_downscale_transform(
    scale: float = 0.5,
    interpolation: int = cv2.INTER_NEAREST,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Понижение разрешения.
    
    Args:
        scale: Коэффициент масштабирования (0.1-1.0)
        interpolation: Метод интерполяции
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с downscale
    """
    return A.Compose([
        A.Downscale(
            scale_min=scale,
            scale_max=scale,
            interpolation=interpolation,
            p=1
        )
    ], seed=seed)


def create_pixel_dropout_transform(
    dropout_prob: float = 0.1,
    per_channel: bool = False,
    drop_value: int = 0,
    mask_drop_value: int = 0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Pixel Dropout.
    
    Args:
        dropout_prob: Вероятность дропаута пикселя (0.0-1.0)
        per_channel: Применять к каждому каналу отдельно
        drop_value: Значение для дропнутых пикселей
        mask_drop_value: Значение для дропнутых пикселей маски
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с pixel dropout
    """
    return A.Compose([
        A.PixelDropout(
            dropout_prob=dropout_prob,
            per_channel=per_channel,
            drop_value=drop_value,
            mask_drop_value=mask_drop_value,
            p=1
        )
    ], seed=seed)


# =============================================================================
# 7. ПОГОДНЫЕ И АТМОСФЕРНЫЕ ЭФФЕКТЫ
# =============================================================================

def create_rain_transform(
    slant: int = 0,
    drop_length: int = 20,
    drop_width: int = 1,
    blur_val: int = 2,
    brightness_coefficient: float = 0.7,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Эффект дождя.
    
    Args:
        slant: Наклон дождя (-30 до 30)
        drop_length: Длина капли (10-50)
        drop_width: Ширина капли (1-5)
        blur_val: Размытие (1-10)
        brightness_coefficient: Коэффициент яркости (0.5-1.0)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с эффектом дождя
    """
    return A.Compose([
        A.RandomRain(
            slant_lower=slant,
            slant_upper=slant,
            drop_length=drop_length,
            drop_width=drop_width,
            blur_val=blur_val,
            brightness_coefficient=brightness_coefficient,
            p=1
        )
    ], seed=seed)


def create_snow_transform(
    snow_point: float = 0.1,
    brightness_coeff: float = 2.5,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Эффект снега.
    
    Args:
        snow_point: Плотность снега (0.0-0.5)
        brightness_coeff: Коэффициент яркости (2.0-3.0)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с эффектом снега
    """
    return A.Compose([
        A.RandomSnow(
            snow_point_lower=snow_point,
            snow_point_upper=snow_point,
            brightness_coeff=brightness_coeff,
            p=1
        )
    ], seed=seed)


def create_fog_transform(
    fog_coef: float = 0.3,
    alpha_coef: float = 0.08,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Эффект тумана.
    
    Args:
        fog_coef: Коэффициент тумана (0.0-1.0)
        alpha_coef: Альфа-коэффициент (0.01-0.3)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с эффектом тумана
    """
    return A.Compose([
        A.RandomFog(
            fog_coef_lower=fog_coef,
            fog_coef_upper=fog_coef,
            alpha_coef=alpha_coef,
            p=1
        )
    ], seed=seed)


def create_shadow_transform(
    num_shadows: int = 3,
    shadow_dimension: int = 5,
    shadow_roi: Tuple[float, float, float, float] = (0, 0.5, 1, 1),
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Тени.
    
    Args:
        num_shadows: Количество теней (1-10)
        shadow_dimension: Размерность тени
        shadow_roi: Область интереса для теней (x_min, y_min, x_max, y_max)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с эффектом теней
    """
    return A.Compose([
        A.RandomShadow(
            num_shadows_lower=num_shadows,
            num_shadows_upper=num_shadows,
            shadow_dimension=shadow_dimension,
            shadow_roi=shadow_roi,
            p=1
        )
    ], seed=seed)


def create_sun_flare_transform(
    num_flare_circles: int = 10,
    src_radius: int = 100,
    flare_roi: Tuple[float, float, float, float] = (0, 0, 0.5, 0.5),
    angle_lower: float = 0,
    angle_upper: float = 0.5,
    src_color: Tuple[int, int, int] = (255, 255, 255),
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Солнечные блики.
    
    Args:
        num_flare_circles: Количество кругов бликов (1-20)
        src_radius: Радиус источника (50-400)
        flare_roi: Область бликов (x_min, y_min, x_max, y_max)
        angle_lower: Нижний угол (0-1)
        angle_upper: Верхний угол (0-1)
        src_color: Цвет источника (R, G, B)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с солнечными бликами
    """
    return A.Compose([
        A.RandomSunFlare(
            flare_roi=flare_roi,
            angle_lower=angle_lower,
            angle_upper=angle_upper,
            num_flare_circles_lower=num_flare_circles,
            num_flare_circles_upper=num_flare_circles,
            src_radius=src_radius,
            src_color=src_color,
            p=1
        )
    ], seed=seed)


def create_rainbow_transform(
    overlap: float = 0.5,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Эффект радуги (через ToneCurve).
    
    Args:
        overlap: Перекрытие кривых (0.0-1.0)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с эффектом радуги
    """
    return A.Compose([
        A.RandomToneCurve(
            scale=overlap,
            p=1
        )
    ], seed=seed)


def create_spatter_transform(
    mean: float = 0.65,
    std: float = 0.3,
    gauss_sigma: float = 2,
    cutout_threshold: float = 0.68,
    intensity: float = 0.6,
    mode: str = 'rain',
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Эффект брызг (дождь/грязь).
    
    Args:
        mean: Среднее значение (0.0-1.0)
        std: Стандартное отклонение (0.0-1.0)
        gauss_sigma: Сигма гаусса (1-10)
        cutout_threshold: Порог вырезания (0.0-1.0)
        intensity: Интенсивность (0.0-1.0)
        mode: Режим ('rain' или 'mud')
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с эффектом брызг
    """
    return A.Compose([
        A.Spatter(
            mean=mean,
            std=std,
            gauss_sigma=gauss_sigma,
            cutout_threshold=cutout_threshold,
            intensity=intensity,
            mode=mode,
            p=1
        )
    ], seed=seed)


def create_chromatic_aberration_transform(
    r_shift: float = 0.0,
    g_shift: float = 0.0,
    b_shift: float = 0.0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Хроматическая аберрация.
    
    Args:
        r_shift: Сдвиг красного канала (-0.1 до 0.1)
        g_shift: Сдвиг зелёного канала (-0.1 до 0.1)
        b_shift: Сдвиг синего канала (-0.1 до 0.1)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с хроматической аберрацией
    """
    return A.Compose([
        A.ChromaticAberration(
            primary_distortion_red=r_shift,
            secondary_distortion_red=r_shift,
            primary_distortion_blue=b_shift,
            secondary_distortion_blue=b_shift,
            interpolation=cv2.INTER_LINEAR,
            p=1
        )
    ], seed=seed)


def create_defocus_transform(
    radius: int = 5,
    alias_blur: float = 0.1,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Эффект дефокуса.
    
    Args:
        radius: Радиус дефокуса (1-20)
        alias_blur: Размытие алиасинга (0.0-0.5)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с дефокусом
    """
    return A.Compose([
        A.Defocus(
            radius=(radius, radius),
            alias_blur=(alias_blur, alias_blur),
            p=1
        )
    ], seed=seed)


def create_zoom_blur_transform(
    max_factor: float = 1.5,
    step_factor: float = 0.02,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Zoom Blur.
    
    Args:
        max_factor: Максимальный фактор зума (1.0-3.0)
        step_factor: Шаг фактора (0.01-0.1)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с zoom blur
    """
    return A.Compose([
        A.ZoomBlur(
            max_factor=(max_factor, max_factor),
            step_factor=(step_factor, step_factor),
            p=1
        )
    ], seed=seed)

# =============================================================================
# 9. ПРОДВИНУТЫЕ ТРАНСФОРМАЦИИ
# =============================================================================

def create_morphological_transform(
    scale: int = 3,
    operation: str = 'dilation',
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Морфологические операции.
    
    Args:
        scale: Размер ядра (1-10)
        operation: Тип операции ('dilation', 'erosion', 'opening', 'closing')
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с морфологической трансформацией
    """
    return A.Compose([
        A.Morphological(
            scale=(scale, scale),
            operation=operation,
            p=1
        )
    ], seed=seed)


def create_planckian_jitter_transform(
    mode: str = 'blackbody',
    selected_temperature: int = 5000,
    sampling_method: str = 'uniform',
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Planckian Jitter (изменение цветовой температуры).
    
    Args:
        mode: Режим ('blackbody' или 'cied')
        selected_temperature: Выбранная температура (3000-25000)
        sampling_method: Метод сэмплирования
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с planckian jitter
    """
    return A.Compose([
        A.PlanckianJitter(
            mode=mode,
            selected_temperature=selected_temperature,
            sampling_method=sampling_method,
            p=1
        )
    ], seed=seed)


def create_shot_noise_transform(
    scale: float = 0.1,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Shot Noise (фотонный шум).
    
    Args:
        scale: Масштаб шума (0.0-1.0)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с shot noise
    """
    return A.Compose([
        A.ShotNoise(
            scale_range=(scale, scale),
            p=1
        )
    ], seed=seed)

# =============================================================================
# 10. УТИЛИТЫ ДЛЯ OPTUNA ИНТЕГРАЦИИ
# =============================================================================

# Словарь всех трансформаций для удобного доступа
TRANSFORM_REGISTRY = {
    # Геометрические
    'rotate': create_rotate_transform,
    'shift': create_shift_transform,
    'scale': create_scale_transform,
    'affine_rotate': create_affine_rotate_transform,
    'shear': create_shear_transform,
    'perspective': create_perspective_transform,
    'elastic': create_elastic_transform,
    'grid_distortion': create_grid_distortion_transform,
    'optical_distortion': create_optical_distortion_transform,
    'shift_scale_rotate': create_shift_scale_rotate_transform,
    
    # Цветовые
    'brightness': create_brightness_transform,
    'contrast': create_contrast_transform,
    'brightness_contrast': create_brightness_contrast_transform,
    'hue': create_hue_transform,
    'saturation': create_saturation_transform,
    'value': create_value_transform,
    'hsv': create_hsv_transform,
    'rgb_shift': create_rgb_shift_transform,
    'gamma': create_gamma_transform,
    'clahe': create_clahe_transform,
    'solarize': create_solarize_transform,
    'posterize': create_posterize_transform,
    'equalize': create_equalize_transform,
    'invert': create_invert_transform,
    'to_gray': create_to_gray_transform,
    'channel_shuffle': create_channel_shuffle_transform,
    'to_sepia': create_to_sepia_transform,
    
    # Размытие и резкость
    'blur': create_blur_transform,
    'gaussian_blur': create_gaussian_blur_transform,
    'median_blur': create_median_blur_transform,
    'motion_blur': create_motion_blur_transform,
    'sharpen': create_sharpen_transform,
    'unsharp_mask': create_unsharp_mask_transform,
    'emboss': create_emboss_transform,
    
    # Шум
    'gauss_noise': create_gauss_noise_transform,
    'multiplicative_noise': create_multiplicative_noise_transform,
    'iso_noise': create_iso_noise_transform,
    
    # Dropout
    'coarse_dropout': create_coarse_dropout_transform,
    'cutout': create_cutout_transform,
    'grid_dropout': create_grid_dropout_transform,
    'mask_dropout': create_mask_dropout_transform,
    'xy_masking': create_xy_masking_transform,
    
    # Сжатие
    'jpeg_compression': create_jpeg_compression_transform,
    'webp_compression': create_webp_compression_transform,
    'downscale': create_downscale_transform,
    'pixel_dropout': create_pixel_dropout_transform,
    
    # Погодные эффекты
    'rain': create_rain_transform,
    'snow': create_snow_transform,
    'fog': create_fog_transform,
    'shadow': create_shadow_transform,
    'sun_flare': create_sun_flare_transform,
    'rainbow': create_rainbow_transform,
    'spatter': create_spatter_transform,
    'chromatic_aberration': create_chromatic_aberration_transform,
    'defocus': create_defocus_transform,
    'zoom_blur': create_zoom_blur_transform,
    
    # Отражения и повороты
    'horizontal_flip': create_horizontal_flip_transform,
    'vertical_flip': create_vertical_flip_transform,
    'random_rotate90': create_random_rotate90_transform,
    'transpose': create_transpose_transform,
    'd4': create_d4_transform,
    
    # Продвинутые
    'advanced_blur': create_advanced_blur_transform,
    'morphological': create_morphological_transform,
    'planckian_jitter': create_planckian_jitter_transform,
    'shot_noise': create_shot_noise_transform,
    'ringing_overshoot': create_ringing_overshoot_transform,
}


def get_transform_params(transform_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Получить параметры для оптимизации через Optuna для данной трансформации.
    
    Args:
        transform_name: Название трансформации из TRANSFORM_REGISTRY
    
    Returns:
        Словарь с параметрами и их диапазонами для Optuna
    """
    param_ranges = {
        'rotate': {
            'angle': {'type': 'float', 'low': 0, 'high': 180, 'step': 1},
        },
        'shift': {
            'shift_limit': {'type': 'float', 'low': 0.0, 'high': 0.5, 'step': 0.01},
        },
        'scale': {
            'scale': {'type': 'float', 'low': 0.5, 'high': 2.0, 'step': 0.05},
        },
        'shear': {
            'shear': {'type': 'float', 'low': 0, 'high': 45, 'step': 1},
        },
        'perspective': {
            'scale': {'type': 'float', 'low': 0.0, 'high': 0.5, 'step': 0.01},
        },
        'elastic': {
            'alpha': {'type': 'float', 'low': 0, 'high': 10, 'step': 0.5},
            'sigma': {'type': 'float', 'low': 30, 'high': 100, 'step': 5},
        },
        'grid_distortion': {
            'num_steps': {'type': 'int', 'low': 1, 'high': 10, 'step': 1},
            'distort_limit': {'type': 'float', 'low': 0.0, 'high': 0.5, 'step': 0.01},
        },
        'optical_distortion': {
            'distort_limit': {'type': 'float', 'low': 0.0, 'high': 1.0, 'step': 0.05},
            'shift_limit': {'type': 'float', 'low': 0.0, 'high': 0.5, 'step': 0.01},
        },
        'shift_scale_rotate': {
            'shift_limit': {'type': 'float', 'low': 0.0, 'high': 0.5, 'step': 0.01},
            'scale_limit': {'type': 'float', 'low': 0.0, 'high': 0.5, 'step': 0.01},
            'rotate_limit': {'type': 'float', 'low': 0, 'high': 180, 'step': 1},
        },
        'brightness': {
            'brightness': {'type': 'float', 'low': -1.0, 'high': 1.0, 'step': 0.05},
        },
        'contrast': {
            'contrast': {'type': 'float', 'low': -1.0, 'high': 1.0, 'step': 0.05},
        },
        'brightness_contrast': {
            'brightness': {'type': 'float', 'low': -1.0, 'high': 1.0, 'step': 0.05},
            'contrast': {'type': 'float', 'low': -1.0, 'high': 1.0, 'step': 0.05},
        },
        'hue': {
            'hue_shift': {'type': 'int', 'low': 0, 'high': 180, 'step': 1},
        },
        'saturation': {
            'saturation': {'type': 'float', 'low': -1.0, 'high': 1.0, 'step': 0.05},
        },
        'hsv': {
            'hue_shift': {'type': 'int', 'low': 0, 'high': 180, 'step': 1},
            'sat_shift': {'type': 'float', 'low': -1.0, 'high': 1.0, 'step': 0.05},
            'val_shift': {'type': 'float', 'low': -1.0, 'high': 1.0, 'step': 0.05},
        },
        'rgb_shift': {
            'r_shift': {'type': 'int', 'low': 0, 'high': 100, 'step': 1},
            'g_shift': {'type': 'int', 'low': 0, 'high': 100, 'step': 1},
            'b_shift': {'type': 'int', 'low': 0, 'high': 100, 'step': 1},
        },
        'gamma': {
            'gamma': {'type': 'float', 'low': 0.3, 'high': 3.0, 'step': 0.1},
        },
        'clahe': {
            'clip_limit': {'type': 'float', 'low': 1.0, 'high': 10.0, 'step': 0.5},
        },
        'solarize': {
            'threshold': {'type': 'int', 'low': 0, 'high': 255, 'step': 5},
        },
        'posterize': {
            'num_bits': {'type': 'int', 'low': 1, 'high': 8, 'step': 1},
        },
        'blur': {
            'kernel_size': {'type': 'int', 'low': 3, 'high': 21, 'step': 2},
        },
        'gaussian_blur': {
            'kernel_size': {'type': 'int', 'low': 3, 'high': 21, 'step': 2},
        },
        'median_blur': {
            'kernel_size': {'type': 'int', 'low': 3, 'high': 21, 'step': 2},
        },
        'motion_blur': {
            'kernel_size': {'type': 'int', 'low': 3, 'high': 31, 'step': 2},
        },
        'sharpen': {
            'alpha': {'type': 'float', 'low': 0.0, 'high': 1.0, 'step': 0.05},
            'lightness': {'type': 'float', 'low': 0.0, 'high': 2.0, 'step': 0.1},
        },
        'unsharp_mask': {
            'alpha': {'type': 'float', 'low': 0.0, 'high': 2.0, 'step': 0.1},
            'blur_limit': {'type': 'int', 'low': 3, 'high': 21, 'step': 2},
        },
        'gauss_noise': {
            'var_limit': {'type': 'float', 'low': 0.0, 'high': 500.0, 'step': 10.0},
        },
        'multiplicative_noise': {
            'multiplier': {'type': 'float', 'low': 0.5, 'high': 1.5, 'step': 0.01},
        },
        'iso_noise': {
            'intensity': {'type': 'float', 'low': 0.0, 'high': 0.5, 'step': 0.01},
            'color_shift': {'type': 'float', 'low': 0.0, 'high': 0.1, 'step': 0.001},
        },
        'coarse_dropout': {
            'max_holes': {'type': 'int', 'low': 1, 'high': 20, 'step': 1},
            'max_height': {'type': 'int', 'low': 8, 'high': 128, 'step': 8},
            'max_width': {'type': 'int', 'low': 8, 'high': 128, 'step': 8},
        },
        'cutout': {
            'num_holes': {'type': 'int', 'low': 1, 'high': 10, 'step': 1},
            'max_h_size': {'type': 'int', 'low': 8, 'high': 128, 'step': 8},
            'max_w_size': {'type': 'int', 'low': 8, 'high': 128, 'step': 8},
        },
        'grid_dropout': {
            'ratio': {'type': 'float', 'low': 0.0, 'high': 0.5, 'step': 0.01},
        },
        'jpeg_compression': {
            'quality': {'type': 'int', 'low': 10, 'high': 100, 'step': 5},
        },
        'downscale': {
            'scale': {'type': 'float', 'low': 0.1, 'high': 1.0, 'step': 0.05},
        },
        'rain': {
            'slant': {'type': 'int', 'low': -30, 'high': 30, 'step': 1},
            'drop_length': {'type': 'int', 'low': 10, 'high': 50, 'step': 5},
            'brightness_coefficient': {'type': 'float', 'low': 0.5, 'high': 1.0, 'step': 0.05},
        },
        'snow': {
            'snow_point': {'type': 'float', 'low': 0.0, 'high': 0.5, 'step': 0.01},
            'brightness_coeff': {'type': 'float', 'low': 2.0, 'high': 3.0, 'step': 0.1},
        },
        'fog': {
            'fog_coef': {'type': 'float', 'low': 0.0, 'high': 1.0, 'step': 0.05},
            'alpha_coef': {'type': 'float', 'low': 0.01, 'high': 0.3, 'step': 0.01},
        },
        'shadow': {
            'num_shadows': {'type': 'int', 'low': 1, 'high': 10, 'step': 1},
        },
        'sun_flare': {
            'num_flare_circles': {'type': 'int', 'low': 1, 'high': 20, 'step': 1},
            'src_radius': {'type': 'int', 'low': 50, 'high': 400, 'step': 25},
        },
        'spatter': {
            'mean': {'type': 'float', 'low': 0.0, 'high': 1.0, 'step': 0.05},
            'std': {'type': 'float', 'low': 0.0, 'high': 1.0, 'step': 0.05},
            'intensity': {'type': 'float', 'low': 0.0, 'high': 1.0, 'step': 0.05},
        },
        'defocus': {
            'radius': {'type': 'int', 'low': 1, 'high': 20, 'step': 1},
            'alias_blur': {'type': 'float', 'low': 0.0, 'high': 0.5, 'step': 0.05},
        },
        'zoom_blur': {
            'max_factor': {'type': 'float', 'low': 1.0, 'high': 3.0, 'step': 0.1},
            'step_factor': {'type': 'float', 'low': 0.01, 'high': 0.1, 'step': 0.01},
        },
        'advanced_blur': {
            'blur_limit': {'type': 'int', 'low': 3, 'high': 21, 'step': 2},
            'noise_limit': {'type': 'float', 'low': 0.0, 'high': 1.0, 'step': 0.05},
        },
        'shot_noise': {
            'scale': {'type': 'float', 'low': 0.0, 'high': 1.0, 'step': 0.05},
        },
    }
    
    return param_ranges.get(transform_name, {})


# =============================================================================
# 11. ПРИМЕР ИНТЕГРАЦИИ С OPTUNA
# =============================================================================

def create_optuna_objective(
    image: np.ndarray,
    transform_name: str,
    metric_func: callable,
    target_value: Optional[float] = None
):
    """
    Создание objective функции для Optuna.
    
    Args:
        image: Исходное изображение
        transform_name: Название трансформации из TRANSFORM_REGISTRY
        metric_func: Функция для оценки качества аугментации
        target_value: Целевое значение метрики (если есть)
    
    Returns:
        Objective функция для Optuna
    """
    def objective(trial):
        # Получаем параметры для данной трансформации
        param_ranges = get_transform_params(transform_name)
        
        # Собираем параметры из trial
        params = {}
        for param_name, param_config in param_ranges.items():
            if param_config['type'] == 'int':
                if 'step' in param_config and param_config['step'] > 1:
                    # Дискретные значения с шагом
                    values = list(range(
                        param_config['low'],
                        param_config['high'] + 1,
                        param_config['step']
                    ))
                    params[param_name] = trial.choice(f'{transform_name}_{param_name}', values)
                else:
                    params[param_name] = trial.int(
                        f'{transform_name}_{param_name}',
                        param_config['low'],
                        param_config['high']
                    )
            elif param_config['type'] == 'float':
                if 'step' in param_config and param_config['step'] > 0:
                    # Дискретные float значения
                    values = np.arange(
                        param_config['low'],
                        param_config['high'] + param_config['step'],
                        param_config['step']
                    ).tolist()
                    params[param_name] = trial.choice(f'{transform_name}_{param_name}', values)
                else:
                    params[param_name] = trial.float(
                        f'{transform_name}_{param_name}',
                        param_config['low'],
                        param_config['high']
                    )
        
        # Создаём трансформацию с параметрами из Optuna
        transform_func = TRANSFORM_REGISTRY.get(transform_name)
        if transform_func is None:
            raise ValueError(f"Transform '{transform_name}' not found in registry")
        
        transform = transform_func(**params, seed=GLOBAL_SEED)
        
        # Применяем трансформацию
        augmented_image = apply_transform(image, transform)
        
        # Вычисляем метрику
        metric_value = metric_func(image, augmented_image, **params)
        
        return metric_value
    
    return objective


def run_optuna_optimization(
    image: np.ndarray,
    transform_name: str,
    metric_func: callable,
    n_trials: int = 100,
    direction: str = 'maximize',
    study_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Запуск оптимизации параметров трансформации через Optuna.
    
    Args:
        image: Исходное изображение
        transform_name: Название трансформации
        metric_func: Функция метрики для оптимизации
        n_trials: Количество итераций
        direction: Направление оптимизации ('maximize' или 'minimize')
        study_name: Имя исследования
    
    Returns:
        Словарь с лучшими параметрами и результатами
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("Please install optuna: pip install optuna")
    
    # Создаём objective функцию
    objective = create_optuna_objective(image, transform_name, metric_func)
    
    # Создаём исследование
    study = optuna.create_study(
        direction=direction,
        study_name=study_name or f'{transform_name}_optimization',
        load_if_exists=True
    )
    
    # Запускаем оптимизацию
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Собираем результаты
    results = {
        'transform_name': transform_name,
        'best_params': study.best_params,
        'best_value': study.best_value,
        'n_trials': n_trials,
        'direction': direction,
        'trials': [
            {
                'params': trial.params,
                'value': trial.value,
                'number': trial.number
            }
            for trial in study.trials
        ]
    }
    
    # Извлекаем чистые параметры (без префиксов)
    clean_params = {}
    for key, value in study.best_params.items():
        param_name = key.replace(f'{transform_name}_', '')
        clean_params[param_name] = value
    
    results['clean_best_params'] = clean_params
    
    return results


def visualize_optuna_results(
    study: Any,
    transform_name: str,
    save_path: Optional[str] = None
):
    """
    Визуализация результатов оптимизации Optuna.
    
    Args:
        study: Optuna study объект
        transform_name: Название трансформации
        save_path: Путь для сохранения графиков
    """
    try:
        import optuna.visualization as vis
    except ImportError:
        raise ImportError("Please install optuna: pip install optuna")
    
    # Создаём графики
    fig_importance = vis.plot_param_importance(study)
    fig_slice = vis.plot_slice(study)
    fig_parallel = vis.plot_parallel_coordinate(study)
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig_importance.write_image(os.path.join(save_path, f'{transform_name}_importance.png'))
        fig_slice.write_image(os.path.join(save_path, f'{transform_name}_slice.png'))
        fig_parallel.write_image(os.path.join(save_path, f'{transform_name}_parallel.png'))
        print(f'✓ Графики сохранены в: {save_path}')
    
    return {
        'importance': fig_importance,
        'slice': fig_slice,
        'parallel': fig_parallel
    }


# =============================================================================
# 12. ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =============================================================================

if __name__ == '__main__':
    # ==================== НАСТРОЙКИ ====================
    IMAGE_PATH = 'your_image.jpg'  # Укажите путь к вашему изображению
    OUTPUT_DIR = 'augmentation_optuna'
    
    print('=' * 60)
    print('ALBUMENTATIONS + OPTUNA ИНТЕГРАЦИЯ')
    print('=' * 60)
    
    # ==================== ЗАГРУЗКА ИЗОБРАЖЕНИЯ ====================
    image = load_image(IMAGE_PATH)
    print(f'✓ Изображение загружено: {IMAGE_PATH}')
    print(f'✓ Размер: {image.shape}')
    
    # ==================== ПРИМЕР 1: СОЗДАНИЕ ОДНОЙ ТРАНСФОРМАЦИИ ====================
    print('\n' + '=' * 60)
    print('ПРИМЕР 1: Создание отдельных трансформаций')
    print('=' * 60)
    
    # Создаём трансформации с конкретными параметрами
    rotate_transform = create_rotate_transform(angle=45, seed=GLOBAL_SEED)
    brightness_transform = create_brightness_transform(brightness=0.3, seed=GLOBAL_SEED)
    blur_transform = create_gaussian_blur_transform(kernel_size=7, seed=GLOBAL_SEED)
    
    # Применяем и сохраняем
    rotated = apply_transform(image, rotate_transform)
    brightened = apply_transform(image, brightness_transform)
    blurred = apply_transform(image, blur_transform)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_image(rotated, os.path.join(OUTPUT_DIR, 'rotated_45deg.jpg'))
    save_image(brightened, os.path.join(OUTPUT_DIR, 'brightness_0.3.jpg'))
    save_image(blurred, os.path.join(OUTPUT_DIR, 'gauss_blur_7.jpg'))
    
    print('✓ Трансформации применены и сохранены')
    
    # ==================== ПРИМЕР 2: ИСПОЛЬЗОВАНИЕ РЕЕСТРА ====================
    print('\n' + '=' * 60)
    print('ПРИМЕР 2: Использование TRANSFORM_REGISTRY')
    print('=' * 60)
    
    # Получаем функцию трансформации из реестра
    transform_func = TRANSFORM_REGISTRY.get('shear')
    if transform_func:
        shear_transform = transform_func(shear=15, seed=GLOBAL_SEED)
        sheared = apply_transform(image, shear_transform)
        save_image(sheared, os.path.join(OUTPUT_DIR, 'shear_15deg.jpg'))
        print('✓ Shear трансформация применена')
    
    # ==================== ПРИМЕР 3: ПОЛУЧЕНИЕ ПАРАМЕТРОВ ДЛЯ OPTUNA ====================
    print('\n' + '=' * 60)
    print('ПРИМЕР 3: Параметры для Optuna')
    print('=' * 60)
    
    # Получаем диапазоны параметров для оптимизации
    param_ranges = get_transform_params('rotate')
    print(f'Параметры для rotate: {param_ranges}')
    
    param_ranges = get_transform_params('brightness_contrast')
    print(f'Параметры для brightness_contrast: {param_ranges}')
    
    # ==================== ПРИМЕР 4: ВИЗУАЛИЗАЦИЯ НЕСКОЛЬКИХ ТРАНСФОРМАЦИЙ ====================
    print('\n' + '=' * 60)
    print('ПРИМЕР 4: Визуализация нескольких трансформаций')
    print('=' * 60)
    
    test_transforms = {
        'rotate_30': create_rotate_transform(angle=30),
        'brightness_0.5': create_brightness_transform(brightness=0.5),
        'contrast_-0.3': create_contrast_transform(contrast=-0.3),
        'blur_9': create_gaussian_blur_transform(kernel_size=9),
        'noise_100': create_gauss_noise_transform(var_limit=100),
        'jpeg_50': create_jpeg_compression_transform(quality=50),
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    axes[0].imshow(image)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    for i, (name, transform) in enumerate(test_transforms.items()):
        augmented = apply_transform(image, transform)
        axes[i + 1].imshow(augmented)
        axes[i + 1].set_title(name, fontsize=10)
        axes[i + 1].axis('off')
    
    axes[-1].axis('off')
    plt.suptitle('Примеры аугментаций', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'examples_grid.png'), dpi=150)
    plt.show()
    
    print(f'✓ Сетка сохранена в: {OUTPUT_DIR}/examples_grid.png')
    
    # ==================== ПРИМЕР 5: ШАБЛОН ДЛЯ OPTUNA ====================
    print('\n' + '=' * 60)
    print('ПРИМЕР 5: Шаблон для Optuna оптимизации')
    print('=' * 60)
    
    # Пример метрики (замените на вашу)
    def example_metric(original: np.ndarray, augmented: np.ndarray, **params) -> float:
        """
        Пример функции метрики для оптимизации.
        Замените на вашу реальную метрику (например, accuracy модели).
        """
        # Пример: MSE между оригиналом и аугментированным изображением
        mse = np.mean((original.astype(float) - augmented.astype(float)) ** 2)
        return -mse  # Отрицательное MSE для максимизации
    
    # Шаблон для запуска оптимизации (раскомментируйте для использования)
    """
    results = run_optuna_optimization(
        image=image,
        transform_name='rotate',
        metric_func=example_metric,
        n_trials=50,
        direction='maximize',
        study_name='rotate_optimization'
    )
    
    print(f'Лучшие параметры: {results["clean_best_params"]}')
    print(f'Лучшее значение: {results["best_value"]}')
    
    # Визуализация результатов
    # visualize_optuna_results(study, 'rotate', save_path=OUTPUT_DIR)
    """
    
    print('\nДля запуска Optuna оптимизации:')
    print('1. Установите: pip install optuna')
    print('2. Реализуйте вашу функцию метрики')
    print('3. Раскомментируйте код в Примере 5')
    print('4. Запустите скрипт')
    
    # ==================== ВЫВОД СПИСКА ВСЕХ ТРАНСФОРМАЦИЙ ====================
    print('\n' + '=' * 60)
    print('ДОСТУПНЫЕ ТРАНСФОРМАЦИИ')
    print('=' * 60)
    
    for i, transform_name in enumerate(TRANSFORM_REGISTRY.keys(), 1):
        print(f'{i:3d}. {transform_name}')
    
    print(f'\nВсего трансформаций: {len(TRANSFORM_REGISTRY)}')
    
    print('\n' + '=' * 60)
    print('ГОТОВО!')
    print('=' * 60)
    print(f'Результаты сохранены в: {OUTPUT_DIR}/')