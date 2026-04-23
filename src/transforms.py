from typing import Any, Union, Tuple, Dict, Optional
import albumentations as A
import cv2
from base import BaseTransform
from enums import DataType


GLOBAL_SEED = 42


class ArgRange:
    """"""
    def __init__(
        self,
        values: list[int | float | str],
        data_type: DataType,
        is_tuple: bool,
    ):
        self.values = values
        self.data_type = data_type
        self.is_tuple = is_tuple



class ShearTransform(BaseTransform):
    def __init__(
        self,
        shear: Union[Tuple[float, float], float, Dict[str, Any]] = (0.0, 0.0), 
        fill: int = 0,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.shear = shear
        self.fill = fill
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.Affine(
                scale=1.0,
                translate_percent=0,
                rotate=0,
                shear=self.shear,
                p=1,
                fill=self.fill
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'shear': [-360., 360.],
            'fill': [0, 255]
        }


class PerspectiveTransform(BaseTransform):
    def __init__(
        self,
        scale: Union[Tuple[float, float], float] = (0.05, 0.1), 
        fill: int = 0,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.scale = scale
        self.fill = fill
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.Perspective(
                scale=self.scale,
                p=1,
                fill=self.fill
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'scale': [0., float('inf')],
            'fill': [0, 255]
        }


class ElasticTransform(BaseTransform):
    def __init__(
        self,
        alpha: float = 1,
        sigma: float = 50, 
        fill: int = 0,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.fill = fill
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.ElasticTransform(
                alpha=self.alpha,
                sigma=self.sigma,
                p=1,
                border_mode=cv2.BORDER_CONSTANT,
                fill=self.fill
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'alpha': [0., float('inf')],
            'sigma': [0., float('inf')],
            'fill': [0, 255]
        }


class GridDistortionTransform(BaseTransform):
    def __init__(
        self,
        num_steps: int = 5, 
        distort_limit: Union[Tuple[float, float], float] = (-0.3, 0.3), 
        fill: int = 0,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.num_steps = num_steps
        self.distort_limit = distort_limit
        self.fill = fill
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.GridDistortion(
                num_steps=self.num_steps,
                distort_limit=self.distort_limit,
                p=1,
                border_mode=cv2.BORDER_CONSTANT,
                fill=self.fill
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'num_steps': [1, float('inf')],
            'distort_limit': [-1., 1.],
            'fill': [0, 255]
        }


class OpticalDistortionTransform(BaseTransform):
    def __init__(
        self,
        distort_limit: Union[Tuple[float, float], float] = (-0.05, 0.05), 
        fill: int = 0,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.distort_limit = distort_limit
        self.fill = fill
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.OpticalDistortion(
                distort_limit=self.distort_limit,
                p=1,
                border_mode=cv2.BORDER_CONSTANT,
                fill=self.fill
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'distort_limit': [float('-inf'), float('inf')],
            'fill': [0, 255]
        }


class ShiftScaleRotateTransform(BaseTransform):
    def __init__(
        self, 
        shift_limit: Union[Tuple[float, float], float] = (-0.0625, 0.0625),
        scale_limit: Union[Tuple[float, float], float] = (-0.1, 0.1),
        rotate_limit: Union[Tuple[float, float], float] = (-45, 45),
        fill: int = 0,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.fill = fill
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=self.shift_limit,
                scale_limit=self.scale_limit,
                rotate_limit=self.rotate_limit,
                p=1,
                border_mode=cv2.BORDER_CONSTANT,
                fill=self.fill
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'shift_limit': [-1., 1.],
            'scale_limit': [float('-inf'), float('inf')],
            'rotate_limit': [-360., 360.],
            'fill': [0, 255]
        }


class BrightnessContrastTransform(BaseTransform):
    def __init__(
        self, 
        brightness_limit: Union[Tuple[float, float], float] = (-0.2, 0.2),
        contrast_limit: Union[Tuple[float, float], float] = (-0.2, 0.2),
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=self.brightness_limit,
                contrast_limit=self.contrast_limit,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'brightness_limit': [-1., 1.],
            'contrast_limit': [-1., 1.]
        }


class HSVTransform(BaseTransform):
    def __init__(
        self, 
        hue_shift_limit: Union[Tuple[float, float], float] = (-20, 20),
        sat_shift_limit: Union[Tuple[float, float], float] = (-30, 30),
        val_shift_limit: Union[Tuple[float, float], float] = (-20, 20),
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.HueSaturationValue(
                hue_shift_limit=self.hue_shift_limit,
                sat_shift_limit=self.sat_shift_limit,
                val_shift_limit=self.val_shift_limit,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'hue_shift_limit': [-180., 180.],
            'sat_shift_limit': [-255., 255.],
            'val_shift_limit': [-255., 255.]
        }


class RGBShiftTransform(BaseTransform):
    def __init__(
        self, 
        r_shift_limit: Union[Tuple[float, float], float] = (-20, 20),
        g_shift_limit: Union[Tuple[float, float], float] = (-20, 20),
        b_shift_limit: Union[Tuple[float, float], float] = (-20, 20),
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.r_shift_limit = r_shift_limit
        self.g_shift_limit = g_shift_limit
        self.b_shift_limit = b_shift_limit
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.RGBShift(
                r_shift_limit=self.r_shift_limit,
                g_shift_limit=self.g_shift_limit,
                b_shift_limit=self.b_shift_limit,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'r_shift_limit': [-255., 255.],
            'g_shift_limit': [-255., 255.],
            'b_shift_limit': [-255., 255.]
        }


class GammaTransform(BaseTransform):
    def __init__(
        self, 
        gamma_limit: Union[Tuple[float, float], float] = (80, 120),
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.gamma_limit = gamma_limit
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.RandomGamma(
                gamma_limit=self.gamma_limit,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'gamma_limit': [0.1, float('inf')]
        }


class CLAHETransform(BaseTransform):
    def __init__(
        self, 
        clip_limit: Union[Tuple[float, float], float] = (1.0, 4.0),
        tile_grid_size: Tuple[int, int] = (8, 8),
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.CLAHE(
                clip_limit=self.clip_limit,
                tile_grid_size=self.tile_grid_size,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'clip_limit': [1., float('inf')],
            'tile_grid_size': [1, 100]
        }
    

class SolarizeTransform(BaseTransform):
    def __init__(
        self,
        threshold_range: tuple[float, float] = (0.5, 0.5),
        seed: int = GLOBAL_SEED,
    ):
        super().__init__()
        self.threshold_range = threshold_range
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.Solarize(
                threshold_range=self.threshold_range,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'threshold_range': [0., 1.]
        }


class PosterizeTransform(BaseTransform):
    def __init__(
        self,
        num_bits: int = 4,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.num_bits = num_bits
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.Posterize(
                num_bits=self.num_bits,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'num_bits': [1, 7]
        }


class EqualizeTransform(BaseTransform):
    def __init__(
        self,
        by_channels: bool = True,
        mode: str = 'cv',
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.by_channels = by_channels
        self.mode = mode
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.Equalize(
                p=1,
                mode=self.mode,
                by_channels=self.by_channels
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'by_channels': [False, True],
            'mode': ['cv', 'pil']
        }


class InvertTransform(BaseTransform):
    def __init__(
        self,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.InvertImg(p=1)
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {}


class ToGrayTransform(BaseTransform):
    def __init__(
        self,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.ToGray(p=1)
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {}


class ChannelShuffleTransform(BaseTransform):
    def __init__(
        self,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.ChannelShuffle(p=1)
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {}
    

class ToSepiaTransform(BaseTransform):
    def __init__(
        self,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.ToSepia(p=1)
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {}


class BlurTransform(BaseTransform):
    def __init__(
        self,
        blur_limit: Union[Tuple[int, int], int] = (3, 7),
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.blur_limit = blur_limit
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.Blur(
                blur_limit=self.blur_limit,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'blur_limit': [3, float('inf')]
        }


class GaussianBlurTransform(BaseTransform):
    def __init__(
        self,
        blur_limit: Union[Tuple[int, int], int] = 0,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.blur_limit = blur_limit
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.GaussianBlur(
                blur_limit=self.blur_limit,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'blur_limit': [0, float('inf')]
        }


class MedianBlurTransform(BaseTransform):
    def __init__(
        self,
        blur_limit: Union[Tuple[int, int], int] = (3, 7),
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.blur_limit = blur_limit
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.MedianBlur(
                blur_limit=self.blur_limit,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'blur_limit': [3, float('inf')]
        }


class MotionBlurTransform(BaseTransform):
    def __init__(
        self,
        blur_limit: Union[Tuple[int, int], int] = (3, 7),
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.blur_limit = blur_limit
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.MotionBlur(
                blur_limit=self.blur_limit,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'blur_limit': [3, float('inf')]
        }


class SharpenTransform(BaseTransform):
    def __init__(
        self, 
        alpha: Tuple[float, float] = (0.2, 0.5),
        lightness: Tuple[float, float] = (0.5, 1.0),
        method: str = 'kernel',
        kernel_size: int = 5,
        sigma: float = 1.0,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.alpha = alpha
        self.lightness = lightness
        self.method = method
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.Sharpen(
                alpha=self.alpha,
                lightness=self.lightness,
                method=self.method,
                kernel_size=self.kernel_size,
                sigma=self.sigma,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'alpha': [0., 1.],
            'lightness': [0., float('inf')],
            'method': ['kernel', 'gaussian'],
            'kernel_size': [1, float('inf')],
            'sigma': [0., float('inf')]
        }


class UnsharpMaskTransform(BaseTransform):
    def __init__(
        self, 
        alpha: Union[Tuple[float, float], float] = (0.2, 0.5),
        sigma_limit: Union[Tuple[float, float], float] = 0.0,
        blur_limit: Union[Tuple[int, int], int] = (3, 7),
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.alpha = alpha
        self.sigma_limit = sigma_limit
        self.blur_limit = blur_limit
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.UnsharpMask(
                alpha=self.alpha,
                blur_limit=self.blur_limit,
                sigma_limit=self.sigma_limit,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'alpha': [0., 1.],
            'sigma_limit': [0., float('inf')],
            'blur_limit': [0, float('inf')]
        }


class EmbossTransform(BaseTransform):
    def __init__(
        self, 
        alpha: Tuple[float, float] = (0.2, 0.5),
        strength: Tuple[float, float] = (0.2, 0.7),
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.alpha = alpha
        self.strength = strength
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.Emboss(
                alpha=self.alpha,
                strength=self.strength,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'alpha': [0., 1.],
            'strength': [0., 1.]
        }


class GaussNoiseTransform(BaseTransform):
    def __init__(
        self, 
        std_range: Tuple[float, float] = (0.2, 0.44),
        mean_range: Tuple[float, float] = (0.0, 0.0),
        per_channel: bool = False,
        noise_scale_factor: float = 1,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.std_range = std_range
        self.mean_range = mean_range
        self.per_channel = per_channel
        self.noise_scale_factor = noise_scale_factor
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.GaussNoise(
                std_range=self.std_range,
                mean_range=self.mean_range,
                per_channel=self.per_channel,
                noise_scale_factor=self.noise_scale_factor,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'std_range': [0., 1.],
            'mean_range': [-1., 1.],
            'per_channel': [False, True],
            'noise_scale_factor': [0., 1.]
        }


class MultiplicativeNoiseTransform(BaseTransform):
    def __init__(
        self, 
        multiplier: Tuple[float, float] = (0.9, 1.1),
        per_channel: bool = False,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.multiplier = multiplier
        self.per_channel = per_channel
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.MultiplicativeNoise(
                multiplier=self.multiplier,
                per_channel=self.per_channel,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'multiplier': [0., float('inf')],
            'per_channel': [False, True]
        }


class ISONoiseTransform(BaseTransform):
    def __init__(
        self, 
        color_shift: Tuple[float, float] = (0.01, 0.05),
        intensity: Tuple[float, float] = (0.1, 0.5),
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.color_shift = color_shift
        self.intensity = intensity
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.ISONoise(
                color_shift=self.color_shift,
                intensity=self.intensity,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'intensity': [0., float('inf')],
            'color_shift': [0., 1.]
        }


class CoarseDropoutTransform(BaseTransform):
    def __init__(
        self, 
        num_holes_range: Tuple[int, int] = (1, 2),
        hole_height_range: Union[Tuple[float, float], Tuple[int, int]] = (0.1, 0.2),
        hole_width_range: Union[Tuple[float, float], Tuple[int, int]] = (0.1, 0.2),
        fill: int = 0,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.num_holes_range = num_holes_range
        self.hole_height_range = hole_height_range
        self.hole_width_range = hole_width_range
        self.fill = fill
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.CoarseDropout(
                num_holes_range=self.num_holes_range,
                hole_height_range=self.hole_height_range,
                hole_width_range=self.hole_width_range,
                fill=self.fill,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'num_holes_range': [0, float('inf')],
            'hole_height_range': [0., 1.],
            'hole_width_range': [0., 1.],
            'fill': [0, 255]
        }


class GridDropoutTransform(BaseTransform):
    def __init__(
        self, 
        ratio: float = 0.1,
        unit_size_range: Tuple[int, int] = (5, 15),
        fill: int = 0,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.ratio = ratio
        self.unit_size_range = unit_size_range
        self.fill = fill
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.GridDropout(
                ratio=self.ratio,
                unit_size_range=self.unit_size_range,
                fill=self.fill,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'ratio': [0., 1.],
            'unit_size_range': [2, float('inf')],
            'fill': [0, 255]
        }


class CompressionTransform(BaseTransform):
    def __init__(
        self, 
        compression_type: str = 'jpeg',
        quality_range: Tuple[int, int] = (99, 100),
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.compression_type = compression_type
        self.quality_range = quality_range
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.ImageCompression(
                quality_range=self.quality_range,
                compression_type=self.compression_type,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'compression_type': ['jpeg', 'webp'],
            'quality_range': [1, 100]
        }


class DownscaleTransform(BaseTransform):
    def __init__(
        self, 
        scale_range: Tuple[float, float] = (0.25, 0.25),
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.scale_range = scale_range
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.Downscale(
                scale_range=self.scale_range,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'scale_range': [0., 1.]
        }


class PixelDropoutTransform(BaseTransform):
    def __init__(
        self, 
        dropout_prob: float = 0.1,
        per_channel: bool = False,
        drop_value: int = 0,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.per_channel = per_channel
        self.drop_value = drop_value
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.PixelDropout(
                dropout_prob=self.dropout_prob,
                per_channel=self.per_channel,
                drop_value=self.drop_value,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'dropout_prob': [0., 1.],
            'per_channel': [False, True],
            'drop_value': [0, 1]  # [0, 1] inf float or [0, 255] if int
        }


class RainTransform(BaseTransform):
    def __init__(
        self, 
        rain_type: str = 'drizzle',
        slant_range: Tuple[float, float] = (-10, 10),
        drop_length: Optional[int] = None,
        drop_width: int = 1,
        blur_value: int = 7,
        brightness_coefficient: float = 0.7,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.rain_type = rain_type
        self.slant_range = slant_range
        self.drop_length = drop_length
        self.drop_width = drop_width
        self.blur_value = blur_value
        self.brightness_coefficient = brightness_coefficient
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.RandomRain(
                rain_type=self.rain_type,
                slant_range=self.slant_range,
                drop_length=self.drop_length,
                drop_width=self.drop_width,
                blur_value=self.blur_value,
                brightness_coefficient=self.brightness_coefficient,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'rain_type': ['drizzle', 'heavy', 'torrential'],
            'slant_range': [-180., 180.],
            'drop_length': [1, float('inf')],
            'drop_width': [1, float('inf')],
            'blur_value': [1, float('inf')],
            'brightness_coefficient': [0., 1.]
        }


class SnowTransform(BaseTransform):
    def __init__(
        self, 
        snow_point_range: Tuple[float, float] = (0.1, 0.3),
        brightness_coeff: float = 2.5,
        method: str = 'bleach',
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.snow_point_range = snow_point_range
        self.brightness_coeff = brightness_coeff
        self.method = method
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.RandomSnow(
                snow_point_range=self.snow_point_range,
                brightness_coeff=self.brightness_coeff,
                method=self.method,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'snow_point_range': [0., 1.],
            'brightness_coeff': [0., float('inf')],
            'method': ['bleach', 'texture']
        }


class FogTransform(BaseTransform):
    def __init__(
        self, 
        alpha_coef: float = 0.08,
        fog_coef_range: Tuple[float, float] = (0.3, 1),
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.alpha_coef = alpha_coef
        self.fog_coef_range = fog_coef_range
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.RandomFog(
                fog_coef_range=self.fog_coef_range,
                alpha_coef=self.alpha_coef,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'fog_coef_range': [0., 1.],
            'alpha_coef': [0., 1.]
        }


class ShadowTransform(BaseTransform):
    def __init__(
        self, 
        num_shadows_limit: Tuple[int, int] = (1, 2),
        shadow_dimension: int = 5,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.num_shadows_limit = num_shadows_limit
        self.shadow_dimension = shadow_dimension
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.RandomShadow(
                num_shadows_limit=self.num_shadows_limit,
                shadow_dimension=self.shadow_dimension,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'num_shadows_limit': [1, float('inf')],
            'shadow_dimension': [0, float('inf')]
        }


class SunFlareTransform(BaseTransform):
    def __init__(
        self, 
        num_flare_circles_range: Tuple[int, int] = (6, 10),
        src_radius: int = 400,
        src_color: Tuple[int, ...] = (255, 255, 255),
        angle_range: Tuple[float, float] = (0, 1),
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.num_flare_circles_range = num_flare_circles_range
        self.src_radius = src_radius
        self.src_color = src_color
        self.angle_range = angle_range
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.RandomSunFlare(
                angle_range=self.angle_range,
                num_flare_circles_range=self.num_flare_circles_range,
                src_radius=self.src_radius,
                src_color=self.src_color,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'num_flare_circles_range': [1, float('inf')],
            'src_radius': [1, float('inf')],
            'src_color': [0, 255],
            'angle_range': [0., 1.],
        }


class RainbowTransform(BaseTransform):
    def __init__(
        self, 
        scale: float = 0.1,
        per_channel: bool = False,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.scale = scale
        self.per_channel = per_channel
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.RandomToneCurve(
                scale=self.scale,
                per_channel=self.per_channel,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'scale': [0., 1.],
            'per_channel': [False, True]
        }


class SpatterTransform(BaseTransform):
    def __init__(
        self, 
        mean: Union[Tuple[float, float], float] = (0.65, 0.65),
        std: Union[Tuple[float, float], float] = (0.3, 0.3),
        gauss_sigma: Union[Tuple[float, float], float] = (2, 2),
        cutout_threshold: Union[Tuple[float, float], float] = (0.68, 0.68),
        intensity: Union[Tuple[float, float], float] = (0.6, 0.6),
        mode: str = 'rain',
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.mean = mean
        self.std = std
        self.gauss_sigma = gauss_sigma
        self.cutout_threshold = cutout_threshold
        self.intensity = intensity
        self.mode = mode
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.Spatter(
                mean=self.mean,
                std=self.std,
                gauss_sigma=self.gauss_sigma,
                cutout_threshold=self.cutout_threshold,
                intensity=self.intensity,
                mode=self.mode,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'mean': [0., float('inf')],
            'std': [0., float('inf')],
            'gauss_sigma': [0., float('inf')],
            'cutout_threshold': [0., float('inf')],
            'intensity': [0., float('inf')],
            'mode': ['rain', 'mud']
        }


class ChromaticAberrationTransform(BaseTransform):
    def __init__(
        self, 
        primary_distortion_limit: Union[Tuple[float, float], float] = (-0.02, 0.02),
        secondary_distortion_limit: Union[Tuple[float, float], float] = (-0.05, 0.05),
        mode: str = 'green_purple',
        interpolation: int = 1,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.primary_distortion_limit = primary_distortion_limit
        self.secondary_distortion_limit = secondary_distortion_limit
        self.mode = mode
        self.interpolation = interpolation
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.ChromaticAberration(
                primary_distortion_limit=self.primary_distortion_limit,
                secondary_distortion_limit=self.secondary_distortion_limit,
                mode=self.mode,
                interpolation=self.interpolation,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'primary_distortion_limit': [float('-inf'), float('inf')],
            'secondary_distortion_limit': [float('-inf'), float('inf')],
            'mode': ['green_purple', 'red_blue', 'random'],
            'interpolation': [0, 6]
        }


class DefocusTransform(BaseTransform):
    def __init__(
        self, 
        radius: Union[Tuple[int, int], int] = (3, 10),
        alias_blur: Union[Tuple[float, float], float] = (0.1, 0.5),
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.radius = radius
        self.alias_blur = alias_blur
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.Defocus(
                radius=self.radius,
                alias_blur=self.alias_blur,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'radius': [1, float('inf')],
            'alias_blur': [0., float('inf')]
        }


class ZoomBlurTransform(BaseTransform):
    def __init__(
        self, 
        max_factor: Union[Tuple[float, float], float] = (1, 1.31),
        step_factor: Union[Tuple[float, float], float] = (0.01, 0.03),
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.max_factor = max_factor
        self.step_factor = step_factor
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.ZoomBlur(
                max_factor=self.max_factor,
                step_factor=self.step_factor,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'max_factor': [1., float('inf')],
            'step_factor': [0., float('inf')]
        }


class MorphologicalTransform(BaseTransform):
    def __init__(
        self, 
        scale: Union[Tuple[int, int], int] = (2, 3),
        operation: str = 'dilation',
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.scale = scale
        self.operation = operation
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.Morphological(
                scale=self.scale,
                operation=self.operation,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'scale': [1, float('inf')],
            'operation': ['dilation', 'erosion']
        }


class PlanckianJitterTransform(BaseTransform):
    def __init__(
        self, 
        mode: str = 'blackbody',
        temperature_limit: Tuple[int, int] = (3000, 15000),
        sampling_method: str = 'uniform',
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.mode = mode
        self.temperature_limit = temperature_limit
        self.sampling_method = sampling_method
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.PlanckianJitter(
                mode=self.mode,
                temperature_limit=self.temperature_limit,
                sampling_method=self.sampling_method,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'mode': ['blackbody', 'cied'],
            'temperature_limit': [3000, 15000],
            'sampling_method': ['uniform', 'gaussian']
        }


class ShotNoiseTransform(BaseTransform):
    def __init__(
        self, 
        scale_range: Tuple[float, float] = (0.1, 0.3),
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.scale_range = scale_range
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.ShotNoise(
                scale_range=self.scale_range,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'scale_range': [0., float('inf')]
        }
