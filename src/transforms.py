import albumentations as A
import cv2


GLOBAL_SEED = 42


def create_shear_transform(
    shear: tuple[float, float] | float | dict[str, float | tuple[float, float]] = (0.0, 0.0),
    border_value: int = 0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Сдвиг (shear) изображения.
    
    Args:
        shear: Угол сдвига в градусах [-360, 360]
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
            shear=shear,
            p=1,
            fill=border_value
        )
    ], seed=seed)


def create_perspective_transform(
    scale: tuple[float, float] | float = (0.05, 0.1),
    pad_value: int = 0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Перспективные искажения.
    
    Args:
        scale: Коэффициент перспективного искажения [0, scale]
        pad_value: Значение заполнения границ
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с перспективной трансформацией
    """
    return A.Compose([
        A.Perspective(
            scale=scale,
            p=1,
            fill=pad_value
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
        alpha: Коэффициент эластичности [any]
        sigma: Стандартное отклонение для гауссова ядра [any]
        border_value: Значение заполнения границ
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с эластичной трансформацией
    """
    return A.Compose([
        A.ElasticTransform(
            alpha=alpha,
            sigma=sigma,
            p=1,
            border_mode=cv2.BORDER_CONSTANT,
            fill=border_value
        )
    ], seed=seed)


def create_grid_distortion_transform(
    num_steps: int = 5,
    distort_limit: tuple[float, float] | float = (-0.3, 0.3),
    border_value: int = 0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Искажение сетки.
    
    Args:
        num_steps: Количество шагов сетки [1, num_steps]
        distort_limit: Лимит искажения [-1, 1]
        border_value: Значение заполнения границ
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с grid distortion
    """
    return A.Compose([
        A.GridDistortion(
            num_steps=num_steps,
            distort_limit=distort_limit,
            p=1,
            border_mode=cv2.BORDER_CONSTANT,
            fill=border_value
        )
    ], seed=seed)


def create_optical_distortion_transform(
    distort_limit: tuple[float, float] | float = (-0.05, 0.05),
    border_value: int = 0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Оптические искажения.
    
    Args:
        distort_limit: Лимит искажения [any]
        border_value: Значение заполнения границ
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с оптической дисторсией
    """
    return A.Compose([
        A.OpticalDistortion(
            distort_limit=distort_limit,
            p=1,
            border_mode=cv2.BORDER_CONSTANT,
            fill=border_value
        )
    ], seed=seed)


def create_shift_scale_rotate_transform(
    shift_limit: tuple[float, float] | float = (-0.0625, 0.0625),
    scale_limit: tuple[float, float] | float = (-0.1, 0.1),
    rotate_limit: tuple[float, float] | float = (-45, 45),
    border_value: int = 0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Комбинированная трансформация: сдвиг + масштаб + поворот.
    
    Args:
        shift_limit: Лимит сдвига [-1, 1]
        scale_limit: Лимит масштабирования [any]
        rotate_limit: Лимит поворота в градусах [any]
        border_value: Значение заполнения границ
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с комбинированной геометрической трансформацией
    """
    return A.Compose([
        A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotate_limit,
            p=1,
            border_mode=cv2.BORDER_CONSTANT,
            fill=border_value
        )
    ], seed=seed)


def create_brightness_contrast_transform(
    brightness_limit: tuple[float, float] | float = (-0.2, 0.2),
    contrast_limit: tuple[float, float] | float = (-0.2, 0.2),
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Комбинированное изменение яркости и контраста.
    
    Args:
        brightness: Коэффициент яркости [-1, 1]
        contrast: Коэффициент контраста [-1, 1]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с трансформацией яркости и контраста
    """
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=1
        )
    ], seed=seed)


def create_hsv_transform(
    hue_shift_limit: tuple[float, float] | float = (-20, 20),
    sat_shift_limit: tuple[float, float] | float = (-30, 30),
    val_shift_limit: tuple[float, float] | float = (-20, 20),
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Комбинированная HSV трансформация.
    
    Args:
        hue_shift: Сдвиг тона [-180, 180]
        sat_shift: Сдвиг насыщенности [-255, 255]
        val_shift: Сдвиг значения [-255, 255]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с HSV трансформацией
    """
    return A.Compose([
        A.HueSaturationValue(
            hue_shift_limit=hue_shift_limit,
            sat_shift_limit=sat_shift_limit,
            val_shift_limit=val_shift_limit,
            p=1
        )
    ], seed=seed)


def create_rgb_shift_transform(
    r_shift_limit: tuple[float, float] | float = (-20, 20),
    g_shift_limit: tuple[float, float] | float = (-20, 20),
    b_shift_limit: tuple[float, float] | float = (-20, 20),
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Сдвиг RGB каналов.
    
    Args:
        r_shift: Сдвиг красного канала [0, 255]
        g_shift: Сдвиг зелёного канала [0, 255]
        b_shift: Сдвиг синего канала [0, 255]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с RGB shift
    """
    return A.Compose([
        A.RGBShift(
            r_shift_limit=r_shift_limit,
            g_shift_limit=g_shift_limit,
            b_shift_limit=b_shift_limit,
            p=1
        )
    ], seed=seed)


def create_gamma_transform(
    gamma_limit: tuple[float, float] | float = (80, 120),
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Гамма-коррекция.
    
    Args:
        gamma: Коэффициент гаммы [1, gamma_limit]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с гамма-коррекцией
    """
    return A.Compose([
        A.RandomGamma(
            gamma_limit=gamma_limit,
            p=1
        )
    ], seed=seed)


def create_clahe_transform(
    clip_limit: tuple[float, float] | float = (1.0, 4.0),
    tile_grid_size: tuple[int, int] = (8, 8),
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        clip_limit: Лимит отсечения [1, clip_limit]
        tile_grid_size: Размер сетки для гистограммы [any]
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
    threshold: float = 0.5,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Соляризация.
    
    Args:
        threshold: Диапазон порогового значения [0, 1]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с соляризацией
    """
    return A.Compose([
        A.Solarize(
            threshold_range=(threshold, threshold),
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
        num_bits: Количество бит на канал [1, 7]
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


def create_blur_transform(
    blur_limit: tuple[int, int] | int = (3, 7),
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Размытие (Blur).
    
    Args:
        blur_limit: Размер ядра размытия [3, blur_limit]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с blur
    """
    return A.Compose([
        A.Blur(
            blur_limit=blur_limit,
            p=1
        )
    ], seed=seed)


def create_gaussian_blur_transform(
    blur_limit: tuple[int, int] | int = 0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Гауссово размытие.
    
    Args:
        blur_limit: Размер ядра [0, blur_limit]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с gaussian blur
    """
    return A.Compose([
        A.GaussianBlur(
            blur_limit=blur_limit,
            p=1
        )
    ], seed=seed)


def create_median_blur_transform(
    blur_limit: tuple[int, int] | int = (3, 7),
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Медианное размытие.
    
    Args:
        kernel_size: Размер ядра [3, blur_limit]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с median blur
    """
    return A.Compose([
        A.MedianBlur(
            blur_limit=blur_limit,
            p=1
        )
    ], seed=seed)


def create_motion_blur_transform(
    blur_limit: tuple[int, int] | int = (3, 7),
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Размытие в движении.
    
    Args:
        kernel_size: Размер ядра [3, blur_limit]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с motion blur
    """
    return A.Compose([
        A.MotionBlur(
            blur_limit=blur_limit,
            p=1
        )
    ], seed=seed)


def create_sharpen_transform(
    alpha: tuple[float, float] = (0.2, 0.5),
    lightness: tuple[float, float] = (0.5, 1.0),
    method: str = 'kernel',
    kernel_size: int = 5,
    sigma: float = 1.0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Повышение резкости.
    
    Args:
        alpha: Сила эффекта [0, 1]
        lightness: Яркость [0, lightness]
        method: 'kernel' | 'gaussian'
        kernel_size: [1, kernel_size]
        sigma: [0, sigma]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с sharpen
    """
    return A.Compose([
        A.Sharpen(
            alpha=alpha,
            lightness=lightness,
            method=method,
            kernel_size=kernel_size,
            sigma=sigma,
            p=1
        )
    ], seed=seed)


def create_unsharp_mask_transform(
    alpha: tuple[float, float] | float = (0.2, 0.5),
    sigma_limit: tuple[float, float] | float = 0.0,
    blur_limit: tuple[int, int] | int = (3, 7),
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Маска нерезкости.
    
    Args:
        alpha: Сила эффекта [0, 1]
        sigma_limit: [0, sigma_limit]
        blur_limit: Лимит размытия (0, blur_limit)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с unsharp mask
    """
    return A.Compose([
        A.UnsharpMask(
            alpha=alpha,
            blur_limit=blur_limit,
            sigma_limit=sigma_limit,
            p=1
        )
    ], seed=seed)


def create_emboss_transform(
    alpha: tuple[float, float] = (0.2, 0.5),
    strength: tuple[float, float] = (0.2, 0.7),
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Эффект тиснения (Emboss).
    
    Args:
        alpha: [0, 1]
        strength: Сила эффекта [0, 1]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с emboss
    """
    return A.Compose([
        A.Emboss(
            alpha=alpha,
            strength=strength,
            p=1
        )
    ], seed=seed)


def create_gauss_noise_transform(
    std_range: tuple[float, float] = (0.2, 0.44),
    mean_range: tuple[float, float] = (0.0, 0.0),
    per_channel: bool = False,
    noise_scale_factor: float = 1,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Гауссов шум.
    
    Args:
        std_range:  [0, 1]
        mean_range: [-1, 1]
        per_channel: bool
        noise_scale_factor: (0, 1]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с gaussian noise
    """
    return A.Compose([
        A.GaussNoise(
            std_range=std_range,
            mean_range=mean_range,
            per_channel=per_channel,
            noise_scale_factor=noise_scale_factor,
            p=1
        )
    ], seed=seed)


def create_multiplicative_noise_transform(
    multiplier: tuple[float, float] = (0.9, 1.1),
    per_channel: bool = False,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Мультипликативный шум.
    
    Args:
        multiplier: Множитель шума [0, multiplier]
        per_channel: bool
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с multiplicative noise
    """
    return A.Compose([
        A.MultiplicativeNoise(
            multiplier=multiplier,
            per_channel=per_channel,
            p=1
        )
    ], seed=seed)


def create_iso_noise_transform(
    color_shift: tuple[float, float] = (0.01, 0.05),
    intensity: tuple[float, float] = (0.1, 0.5),
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    ISO шум (цветной шум).
    
    Args:
        intensity: Интенсивность шума [0, intensity]
        color_shift: Сдвиг цвета [0, 1]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с iso noise
    """
    return A.Compose([
        A.ISONoise(
            color_shift=color_shift,
            intensity=intensity,
            p=1
        )
    ], seed=seed)


def create_coarse_dropout_transform(
    num_holes_range: tuple[int, int] = (1, 2),
    hole_height_range: tuple[float, float] | tuple[int, int] = (0.1, 0.2),
    hole_width_range: tuple[float, float] | tuple[int, int] = (0.1, 0.2),
    fill_value: int = 0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Coarse Dropout (Cutout).
    
    Args:
        num_holes_range: Максимальное количество отверстий [0, max_holes]
        hole_height_range: Максимальная высота отверстия [0, 1]
        hole_width_range: Максимальная ширина отверстия [0, 1]
        fill_value: Значение заполнения
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с coarse dropout
    """
    return A.Compose([
        A.CoarseDropout(
            num_holes_range=num_holes_range,
            hole_height_range=hole_height_range,
            hole_width_range=hole_width_range,
            fill=fill_value,
            p=1
        )
    ], seed=seed)


def create_grid_dropout_transform(
    ratio: float = 0.1,
    unit_size_range: tuple[int, int] = (5, 15),
    fill_value: int = 0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Grid Dropout.
    
    Args:
        ratio: Доля вырезаемых областей [0, 1]
        unit_size_range: Размер ячейки [2, unit_size_range]
        fill_value: Значение заполнения
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с grid dropout
    """
    return A.Compose([
        A.GridDropout(
            ratio=ratio,
            unit_size_range=unit_size_range,
            fill=fill_value,
            p=1
        )
    ], seed=seed)


def create_compression_transform(
    compression_type: str = 'jpeg',
    quality_range: tuple[int, int] = (99, 100),
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    JPEG сжатие.
    
    Args:
        compression_type: 'jpeg' | 'webp'
        quality_range: Качество JPEG [1, 100]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с JPEG компрессией
    """
    return A.Compose([
        A.ImageCompression(
            quality_range=quality_range,
            compression_type=compression_type,  # JPEG
            p=1
        )
    ], seed=seed)


def create_downscale_transform(
    scale_range: tuple[float, float] = (0.25, 0.25),
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Понижение разрешения.
    
    Args:
        scale_range: Коэффициент масштабирования [0, 1]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с downscale
    """
    return A.Compose([
        A.Downscale(
            scale_range=scale_range,
            p=1
        )
    ], seed=seed)


def create_pixel_dropout_transform(
    dropout_prob: float = 0.1,
    per_channel: bool = False,
    drop_value: int = 0,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Pixel Dropout.
    
    Args:
        dropout_prob: Вероятность дропаута пикселя [0, 1]
        per_channel: Применять к каждому каналу отдельно bool
        drop_value: Значение для дропнутых пикселей [0, 1]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с pixel dropout
    """
    return A.Compose([
        A.PixelDropout(
            dropout_prob=dropout_prob,
            per_channel=per_channel,
            drop_value=drop_value,
            p=1
        )
    ], seed=seed)


def create_rain_transform(
    rain_type: str = 'drizzle',
    slant_range: tuple[float, float] = (-10, 10),
    drop_length: int | None = None,
    drop_width: int = 1,
    blur_value: int = 7,
    brightness_coefficient: float = 0.7,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Эффект дождя.
    
    Args:
        rain_type: 'drizzle' | 'heavy' | 'torrential'
        slant: Наклон дождя [-180, 180]
        drop_length: Длина капли [1, drop_length]
        drop_width: Ширина капли [1, drop_width]
        blur_value: Размытие [1, blur_value]
        brightness_coefficient: Коэффициент яркости (0, 1]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с эффектом дождя
    """
    return A.Compose([
        A.RandomRain(
            rain_type=rain_type,
            slant_range=slant_range,
            drop_length=drop_length,
            drop_width=drop_width,
            blur_value=blur_value,
            brightness_coefficient=brightness_coefficient,
            p=1
        )
    ], seed=seed)


def create_snow_transform(
    snow_point_range: tuple[float, float] = (0.1, 0.3),
    brightness_coeff: float = 2.5,
    method='bleach',
    seed: int = GLOBAL_SEED,
) -> A.Compose:
    """
    Эффект снега.
    
    Args:
        snow_point_range: Плотность снега (0, 1)
        brightness_coeff: Коэффициент яркости (0, brightness_coeff]
        method: 'bleach' | 'texture'
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с эффектом снега
    """
    return A.Compose([
        A.RandomSnow(
            snow_point_range=snow_point_range,
            brightness_coeff=brightness_coeff,
            method=method,
            p=1
        )
    ], seed=seed)


def create_fog_transform(
    alpha_coef: float = 0.08,
    fog_coef_range: tuple[float, float] = (0.3, 1),
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Эффект тумана.
    
    Args:
        fog_coef: Коэффициент тумана [0, 1]
        alpha_coef: Альфа-коэффициент [0, 1]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с эффектом тумана
    """
    return A.Compose([
        A.RandomFog(
            fog_coef_range=fog_coef_range,
            alpha_coef=alpha_coef,
            p=1
        )
    ], seed=seed)


def create_shadow_transform(
    num_shadows_limit: tuple[int, int] = (1, 2),
    shadow_dimension: int = 5,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Тени.
    
    Args:
        num_shadows_limit: Количество теней [1, inf)
        shadow_dimension: Размерность тени [0, inf)
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с эффектом теней
    """
    return A.Compose([
        A.RandomShadow(
            num_shadows_limit=num_shadows_limit,
            shadow_dimension=shadow_dimension,
            p=1
        )
    ], seed=seed)


def create_sun_flare_transform(
    num_flare_circles_range: tuple[int, int] = (6, 10),
    src_radius: int = 400,
    src_color: tuple[int, ...] = (255, 255, 255),
    angle_range: tuple[float, float] = (0, 1),
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Солнечные блики.
    
    Args:
        num_flare_circles_range: Количество кругов бликов [1, inf)
        src_radius: Радиус источника [1, inf)
        angle_range: угол [0, 1]
        src_color: Цвет источника (R, G, B) [0 [255, 255, 255]]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с солнечными бликами
    """
    return A.Compose([
        A.RandomSunFlare(
            angle_range=angle_range,
            num_flare_circles_range=num_flare_circles_range,
            src_radius=src_radius,
            src_color=src_color,
            p=1
        )
    ], seed=seed)


def create_rainbow_transform(
    scale: float = 0.1,
    per_channel: bool = False,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Эффект радуги (через ToneCurve).
    
    Args:
        scale: Перекрытие кривых [0, 1]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с эффектом радуги
    """
    return A.Compose([
        A.RandomToneCurve(
            scale=scale,
            per_channel=per_channel,
            p=1
        )
    ], seed=seed)


def create_spatter_transform(
    mean: tuple[float, float] | float = (0.65, 0.65),
    std: tuple[float, float] | float = (0.3, 0.3),
    gauss_sigma: tuple[float, float] | float = (2, 2),
    cutout_threshold: tuple[float, float] | float = (0.68, 0.68),
    intensity: tuple[float, float] | float = (0.6, 0.6),
    mode: str = 'rain',
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Эффект брызг (дождь/грязь).
    
    Args:
        mean: Среднее значение [0, mean]
        std: Стандартное отклонение [0, std]
        gauss_sigma: Сигма гаусса [0, gauss_sigma]
        cutout_threshold: Порог вырезания [0, cutout_threshold]
        intensity: Интенсивность [0, intensity]
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
    primary_distortion_limit: tuple[float, float] | float = (-0.02, 0.02),
    secondary_distortion_limit: tuple[float, float] | float = (-0.05, 0.05),
    mode: str = 'green_purple',
    interpolation: 0 | 6 | 1 | 2 | 3 | 4 | 5 = 1,
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Хроматическая аберрация.
    
    Args:
        primary_distortion_limit: (-inf, inf)
        secondary_distortion_limit: (-inf, inf)
        interpolation: 'green_purple' | 'red_blue' | 'random'
        interpolation: [0, 6]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с хроматической аберрацией
    """
    return A.Compose([
        A.ChromaticAberration(
            primary_distortion_limit=primary_distortion_limit,
            secondary_distortion_limit=secondary_distortion_limit,
            mode=mode,
            interpolation=interpolation,
            p=1
        )
    ], seed=seed)


def create_defocus_transform(
    radius: tuple[int, int] | int = (3, 10),
    alias_blur: tuple[float, float] | float = (0.1, 0.5),
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Эффект дефокуса.
    
    Args:
        radius: Радиус дефокуса [1, radius]
        alias_blur: Размытие алиасинга [0, alias_blur]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с дефокусом
    """
    return A.Compose([
        A.Defocus(
            radius=radius,
            alias_blur=alias_blur,
            p=1
        )
    ], seed=seed)


def create_zoom_blur_transform(
    max_factor: tuple[float, float] | float = (1, 1.31),
    step_factor: tuple[float, float] | float = (0.01, 0.03),
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Zoom Blur.
    
    Args:
        max_factor: Максимальный фактор зума [1, max_factor]
        step_factor: Шаг фактора [0, step_factor]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с zoom blur
    """
    return A.Compose([
        A.ZoomBlur(
            max_factor=max_factor,
            step_factor=step_factor,
            p=1
        )
    ], seed=seed)


def create_morphological_transform(
    scale: tuple[int, int] | int = (2, 3),
    operation: str = 'dilation',
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Морфологические операции.
    
    Args:
        scale: Размер ядра [1, inf]
        operation: Тип операции ('dilation', 'erosion')
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с морфологической трансформацией
    """
    return A.Compose([
        A.Morphological(
            scale=scale,
            operation=operation,
            p=1
        )
    ], seed=seed)


def create_planckian_jitter_transform(
    mode: str = 'blackbody',
    temperature_limit: tuple[int, int] =  (3000, 15000),
    sampling_method: str = 'uniform',
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Planckian Jitter (изменение цветовой температуры).
    
    Args:
        mode: Режим ('blackbody' или 'cied')
        selected_temperature: "blackbody" mode: [3000K, 15000K]. "cied" mode: [4000K, 15000K]
        sampling_method: 'uniform' | 'gaussian'
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с planckian jitter
    """
    return A.Compose([
        A.PlanckianJitter(
            mode=mode,
            temperature_limit=temperature_limit,
            sampling_method=sampling_method,
            p=1
        )
    ], seed=seed)


def create_shot_noise_transform(
    scale_range: tuple[float, float] = (0.1, 0.3),
    seed: int = GLOBAL_SEED
) -> A.Compose:
    """
    Shot Noise (фотонный шум).
    
    Args:
        scale_range: Масштаб шума [0, inf]
        seed: Seed для воспроизводимости
    
    Returns:
        A.Compose с shot noise
    """
    return A.Compose([
        A.ShotNoise(
            scale_range=scale_range,
            p=1
        )
    ], seed=seed)