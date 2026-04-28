from typing import Any, Literal
import albumentations as A
import cv2

from ..base import BaseTransform
from ..utils.enums import DataType, Inf
from ..utils.dataclasses import ArgRange


GLOBAL_SEED = 42


class ShearTransform(BaseTransform):
    def __init__(
        self,
        shear: tuple[float, float] | float | dict[str, float | tuple[float, float]] = (0, 0),
        fill: tuple[float, ...] | float = 0,
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
            'shear': ArgRange(values=[-360., 360.], data_type=DataType.FLOAT, is_tuple=True),
            'fill': ArgRange(values=[0, 255], data_type=DataType.INT)
        }


class PerspectiveTransform(BaseTransform):
    def __init__(
        self,
        scale: tuple[float, float] | float = (0.05, 0.1), 
        fill: tuple[float, ...] | float = 0,
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
    def get_ranges() -> dict[str, ArgRange]:
        return {
            'scale': ArgRange(values=[0., Inf.TINY.value], data_type=DataType.FLOAT, is_tuple=True),
            'fill': ArgRange(values=[0, 255], data_type=DataType.INT)
        }


class ElasticTransform(BaseTransform):
    def __init__(
        self,
        alpha: float = 1,
        sigma: float = 50, 
        fill: tuple[float, ...] | float = 0,
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
            'alpha': ArgRange(values=[0., Inf.BIG.value], data_type=DataType.FLOAT),
            'sigma': ArgRange(values=[0., Inf.BIG.value], data_type=DataType.FLOAT),
            'fill': ArgRange(values=[0, 255], data_type=DataType.INT)
        }


class GridDistortionTransform(BaseTransform):
    def __init__(
        self,
        num_steps: int = 5, 
        distort_limit: tuple[float, float] | float = (-0.3, 0.3),
        fill: tuple[float, ...] | float = 0,
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
            'num_steps': ArgRange(values=[1, Inf.MEDIUM.value], data_type=DataType.INT),
            'distort_limit': ArgRange(values=[-1., 1.], data_type=DataType.FLOAT, is_tuple=True),
            'fill': ArgRange(values=[0, 255], data_type=DataType.INT)
        }


class OpticalDistortionTransform(BaseTransform):
    def __init__(
        self,
        distort_limit: tuple[float, float] | float = (-0.05, 0.05), 
        fill: tuple[float, ...] | float = 0,
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
            'distort_limit': ArgRange(values=[-Inf.TINY.value, Inf.TINY.value], data_type=DataType.FLOAT, is_tuple=True),
            'fill': ArgRange(values=[0, 255], data_type=DataType.INT)
        }


class ShiftScaleRotateTransform(BaseTransform):
    def __init__(
        self, 
        shift_limit: tuple[float, float] | float = (-0.0625, 0.0625),
        scale_limit: tuple[float, float] | float = (-0.1, 0.1),
        rotate_limit: tuple[float, float] | float = (-45, 45),
        fill: tuple[float, ...] | float = 0,
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
            'shift_limit': ArgRange(values=[-1., 1.], data_type=DataType.FLOAT, is_tuple=True),
            'scale_limit': ArgRange(values=[0, Inf.TINY.value], data_type=DataType.FLOAT, is_tuple=True),
            'rotate_limit': ArgRange(values=[-360., 360.], data_type=DataType.FLOAT, is_tuple=True),
            'fill': ArgRange(values=[0, 255], data_type=DataType.INT)
        }


class BrightnessContrastTransform(BaseTransform):
    def __init__(
        self, 
        brightness_limit: tuple[float, float] | float = (-0.2, 0.2),
        contrast_limit: tuple[float, float] | float = (-0.2, 0.2),
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
            'brightness_limit': ArgRange(values=[-1., 1.], data_type=DataType.FLOAT, is_tuple=True),
            'contrast_limit': ArgRange(values=[-1., 1.], data_type=DataType.FLOAT, is_tuple=True)
        }


class HSVTransform(BaseTransform):
    def __init__(
        self, 
        hue_shift_limit: tuple[float, float] | float = (-20, 20),
        sat_shift_limit: tuple[float, float] | float = (-30, 30),
        val_shift_limit: tuple[float, float] | float = (-20, 20),
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
            'hue_shift_limit': ArgRange(values=[-180., 180.], data_type=DataType.FLOAT, is_tuple=True),
            'sat_shift_limit': ArgRange(values=[-255., 255.], data_type=DataType.FLOAT, is_tuple=True),
            'val_shift_limit': ArgRange(values=[-255., 255.], data_type=DataType.FLOAT, is_tuple=True)
        }


class RGBShiftTransform(BaseTransform):
    def __init__(
        self, 
        r_shift_limit: tuple[float, float] | float = (-20, 20),
        g_shift_limit: tuple[float, float] | float = (-20, 20),
        b_shift_limit: tuple[float, float] | float = (-20, 20),
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
            'r_shift_limit': ArgRange(values=[-255., 255.], data_type=DataType.FLOAT, is_tuple=True),
            'g_shift_limit': ArgRange(values=[-255., 255.], data_type=DataType.FLOAT, is_tuple=True),
            'b_shift_limit': ArgRange(values=[-255., 255.], data_type=DataType.FLOAT, is_tuple=True),
        }


class GammaTransform(BaseTransform):
    def __init__(
        self, 
        gamma_limit: tuple[float, float] | float = (80, 120),
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
            'gamma_limit': ArgRange(values=[1., Inf.MEDIUM.value], data_type=DataType.FLOAT, is_tuple=True)
        }


class CLAHETransform(BaseTransform):
    def __init__(
        self, 
        clip_limit: tuple[float, float] | float = 4,
        tile_grid_size: tuple[int, int] = (8, 8),
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
            'clip_limit': ArgRange(values=[1., Inf.SMALL.value], data_type=DataType.FLOAT, is_tuple=True),
            'tile_grid_size': ArgRange(values=[1, 100], data_type=DataType.INT, is_tuple=True),
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
            'threshold_range': ArgRange(values=[0., 1.], data_type=DataType.FLOAT, is_tuple=True)
        }


class PosterizeTransform(BaseTransform):
    def __init__(
        self,
        num_bits: int | tuple[int, int] | list[tuple[int, int]] = 4,
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
            'num_bits': ArgRange(values=[1, 7], data_type=DataType.INT)
        }


class EqualizeTransform(BaseTransform):
    def __init__(
        self,
        mode: Literal['cv', 'pil'] = "cv",
        by_channels: bool = True,
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
            'by_channels': ArgRange(values=[False, True], data_type=DataType.BOOL),
            'mode': ArgRange(values=['cv', 'pil'], data_type=DataType.STR)
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
        blur_limit: tuple[int, int] | int = (3, 7),
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
            'blur_limit': ArgRange(values=[3, Inf.MEDIUM.value], data_type=DataType.INT, is_tuple=True)
        }


class GaussianBlurTransform(BaseTransform):
    def __init__(
        self,
        blur_limit: tuple[int, int] | int = 0,
        sigma_limit: tuple[float, float] | float = (0.5, 3),
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.blur_limit = blur_limit
        self.sigma_limit = sigma_limit
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.GaussianBlur(
                blur_limit=self.blur_limit,
                sigma_limit=self.sigma_limit,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'blur_limit': ArgRange(values=[0, Inf.BIG.value], data_type=DataType.INT),
            'sigma_limit': ArgRange(values=[0., Inf.BIG.value], data_type=DataType.FLOAT, is_tuple=True)
        }


class MedianBlurTransform(BaseTransform):
    def __init__(
        self,
        blur_limit: tuple[int, int] | int = (3, 7),
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
            'blur_limit': ArgRange(values=[3, Inf.MEDIUM.value], data_type=DataType.INT, is_tuple=True)
        }


class MotionBlurTransform(BaseTransform):
    def __init__(
        self,
        blur_limit: tuple[int, int] | int = (3, 7),
        allow_shifted: bool = True,
        angle_range: tuple[float, float] = (0, 360),
        direction_range: tuple[float, float] = (-1, 1),
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.blur_limit = blur_limit
        self.allow_shifted = allow_shifted
        self.angle_range = angle_range
        self.direction_range = direction_range
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.MotionBlur(
                blur_limit=self.blur_limit,
                allow_shifted=self.allow_shifted,
                angle_range=self.angle_range,
                direction_range=self.direction_range,
                p=1,
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'blur_limit': ArgRange(values=[3, Inf.MEDIUM.value], data_type=DataType.INT, is_tuple=True),
            'allow_shifted': ArgRange(values=[True, False], data_type=DataType.BOOL),
            'angle_range': ArgRange(values=[0., 360.], data_type=DataType.FLOAT, is_tuple=True),
            'direction_range': ArgRange(values=[-1., 1.], data_type=DataType.FLOAT, is_tuple=True),
        }


class SharpenTransform(BaseTransform):
    def __init__(
        self, 
        alpha: tuple[float, float] = (0.2, 0.5),
        lightness: tuple[float, float] = (0.5, 1),
        method: Literal['kernel', 'gaussian'] = "kernel",
        kernel_size: int = 5,
        sigma: float = 1,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.alpha = alpha
        self.lightness = lightness
        self.method = method
        self.kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
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
            'alpha': ArgRange(values=[0., 1.], data_type=DataType.FLOAT, is_tuple=True),
            'lightness': ArgRange(values=[0., Inf.TINY.value], data_type=DataType.FLOAT, is_tuple=True),
            'method': ArgRange(values=['kernel', 'gaussian'], data_type=DataType.STR),
            'kernel_size': ArgRange(values=[3, Inf.SMALL.value], data_type=DataType.INT),
            'sigma': ArgRange(values=[0., Inf.TINY.value], data_type=DataType.FLOAT)
        }


class UnsharpMaskTransform(BaseTransform):
    def __init__(
        self, 
        blur_limit: tuple[int, int] | int = (3, 7),
        sigma_limit: tuple[float, float] | float = 0,
        alpha: tuple[float, float] | float = (0.2, 0.5),
        threshold: int = 10,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.alpha = alpha
        self.sigma_limit = sigma_limit
        self.blur_limit = blur_limit
        self.threshold = threshold
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.UnsharpMask(
                alpha=self.alpha,
                blur_limit=self.blur_limit,
                sigma_limit=self.sigma_limit,
                threshold=self.threshold,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'alpha': ArgRange(values=[0., 1.], data_type=DataType.FLOAT, is_tuple=True),
            'sigma_limit': ArgRange(values=[0., Inf.SMALL.value], data_type=DataType.FLOAT),
            'blur_limit': ArgRange(values=[0, Inf.BIG.value], data_type=DataType.INT, is_tuple=True),
            'threshold': ArgRange(values=[0, 255], data_type=DataType.INT)
        }


class EmbossTransform(BaseTransform):
    def __init__(
        self, 
        alpha: tuple[float, float] = (0.2, 0.5),
        strength: tuple[float, float] = (0.2, 0.7),
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
            'alpha': ArgRange(values=[0., 1.], data_type=DataType.FLOAT, is_tuple=True),
            'strength': ArgRange(values=[0., 1.], data_type=DataType.FLOAT, is_tuple=True)
        }


class GaussNoiseTransform(BaseTransform):
    def __init__(
        self, 
        std_range: tuple[float, float] = (0.2, 0.44),
        mean_range: tuple[float, float] = (0, 0),
        per_channel: bool = True,
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
            'std_range': ArgRange(values=[0., 1.], data_type=DataType.FLOAT, is_tuple=True),
            'mean_range': ArgRange(values=[-1., 1.], data_type=DataType.FLOAT, is_tuple=True),
            'per_channel': ArgRange(values=[False, True], data_type=DataType.BOOL),
            'noise_scale_factor': ArgRange(values=[0., 1.], data_type=DataType.FLOAT)
        }


class MultiplicativeNoiseTransform(BaseTransform):
    def __init__(
        self, 
        multiplier: tuple[float, float] | float = (0.9, 1.1),
        per_channel: bool = False,
        elementwise: bool = False,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.multiplier = multiplier
        self.per_channel = per_channel
        self.elementwise = elementwise
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.MultiplicativeNoise(
                multiplier=self.multiplier,
                per_channel=self.per_channel,
                elementwise=self.elementwise,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'multiplier': ArgRange(values=[0., Inf.TINY.value], data_type=DataType.FLOAT, is_tuple=True),
            'per_channel': ArgRange(values=[False, True], data_type=DataType.BOOL),
            'elementwise': ArgRange(values=[False, True], data_type=DataType.BOOL)
        }


class ISONoiseTransform(BaseTransform):
    def __init__(
        self, 
        color_shift: tuple[float, float] = (0.01, 0.05),
        intensity: tuple[float, float] = (0.1, 0.5),
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
            'intensity': ArgRange(values=[0., Inf.TINY.value], data_type=DataType.FLOAT, is_tuple=True),
            'color_shift': ArgRange(values=[0., 1.], data_type=DataType.FLOAT, is_tuple=True),
        }


class CoarseDropoutTransform(BaseTransform):
    def __init__(
        self, 
        num_holes_range: tuple[int, int] = (1, 2),
        hole_height_range: tuple[float, float] | tuple[int, int] = (0.1, 0.2),
        hole_width_range: tuple[float, float] | tuple[int, int] = (0.1, 0.2),
        fill: tuple[float, ...] | float | Literal['random', 'random_uniform', 'inpaint_telea', 'inpaint_ns'] = 0,
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
            'num_holes_range': ArgRange(values=[0, Inf.LARGE.value], data_type=DataType.INT, is_tuple=True),
            'hole_height_range': ArgRange(values=[0., 1.], data_type=DataType.FLOAT, is_tuple=True),
            'hole_width_range': ArgRange(values=[0., 1.], data_type=DataType.FLOAT, is_tuple=True),
            'fill': ArgRange(values=[0, 255], data_type=DataType.INT)
        }


class GridDropoutTransform(BaseTransform):
    def __init__(
        self, 
        ratio: float = 0.1,
        unit_size_range: tuple[int, int] = (5, 15),
        fill: tuple[float, ...] | float | Literal['random', 'random_uniform', 'inpaint_telea', 'inpaint_ns'] = 0,
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
            'ratio': ArgRange(values=[0., 1.], data_type=DataType.FLOAT),
            'unit_size_range': ArgRange(values=[2, Inf.MEDIUM.value], data_type=DataType.INT, is_tuple=True),
            'fill': ArgRange(values=[0, 255], data_type=DataType.INT)
        }


class CompressionTransform(BaseTransform):
    def __init__(
        self, 
        compression_type: Literal['jpeg', 'webp'] = "jpeg",
        quality_range: tuple[int, int] = (99, 100),
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
            'compression_type': ArgRange(values=['jpeg', 'webp'], data_type=DataType.STR),
            'quality_range': ArgRange(values=[1, 100], data_type=DataType.INT, is_tuple=True)
        }


class DownscaleTransform(BaseTransform):
    def __init__(
        self, 
        scale_range: tuple[float, float] = (0.25, 0.25),
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
            'scale_range': ArgRange(values=[0., 1.], data_type=DataType.FLOAT, is_tuple=True)
        }


class PixelDropoutTransform(BaseTransform):
    def __init__(
        self, 
        dropout_prob: float = 0.1,
        per_channel: bool = False,
        seed: int = GLOBAL_SEED
    ):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.per_channel = per_channel
        self.seed = seed

    def transform(self, img: Any) -> Any:
        transform_pipeline = A.Compose([
            A.PixelDropout(
                dropout_prob=self.dropout_prob,
                per_channel=self.per_channel,
                p=1
            )
        ], seed=self.seed)
        return transform_pipeline(image=img)['image']

    @staticmethod
    def get_ranges() -> dict[str, list]:
        return {
            'dropout_prob': ArgRange(values=[0., 1.], data_type=DataType.FLOAT),
            'per_channel': ArgRange(values=[False, True], data_type=DataType.BOOL),
        }


class RainTransform(BaseTransform):
    def __init__(
        self, 
        slant_range: tuple[float, float] = (-10, 10),
        drop_length: int | None = None,
        drop_width: int = 1,
        blur_value: int = 7,
        brightness_coefficient: float = 0.7,
        rain_type: Literal['drizzle', 'heavy', 'torrential', 'default'] = "default",
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
            'rain_type': ArgRange(values=['drizzle', 'heavy', 'torrential'], data_type=DataType.STR),
            'slant_range': ArgRange(values=[-45., 45.], data_type=DataType.FLOAT, is_tuple=True),
            'drop_length': ArgRange(values=[1, Inf.SMALL.value], data_type=DataType.INT),
            'drop_width': ArgRange(values=[1, Inf.SMALL.value], data_type=DataType.INT),
            'blur_value': ArgRange(values=[1, Inf.SMALL.value], data_type=DataType.INT),
            'brightness_coefficient': ArgRange(values=[0., 1.], data_type=DataType.FLOAT)
        }


class SnowTransform(BaseTransform):
    def __init__(
        self, 
        brightness_coeff: float = 2.5,
        snow_point_range: tuple[float, float] = (0.1, 0.3),
        method: Literal['bleach', 'texture'] = "bleach",
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
            'snow_point_range': ArgRange(values=[0., 1.], data_type=DataType.FLOAT, is_tuple=True),
            'brightness_coeff': ArgRange(values=[0., Inf.SMALL.value], data_type=DataType.FLOAT),
            'method': ArgRange(values=['bleach', 'texture'], data_type=DataType.STR)
        }


class FogTransform(BaseTransform):
    def __init__(
        self, 
        alpha_coef: float = 0.08,
        fog_coef_range: tuple[float, float] = (0.3, 1),
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
            'fog_coef_range': ArgRange(values=[0., 1.], data_type=DataType.FLOAT, is_tuple=True),
            'alpha_coef': ArgRange(values=[0., 1.], data_type=DataType.FLOAT)
        }


class ShadowTransform(BaseTransform):
    def __init__(
        self, 
        num_shadows_limit: tuple[int, int] = (1, 2),
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
            'num_shadows_limit': ArgRange(values=[1, Inf.SMALL.value], data_type=DataType.INT, is_tuple=True),
            'shadow_dimension': ArgRange(values=[3, Inf.SMALL.value], data_type=DataType.INT)
        }


class SunFlareTransform(BaseTransform):
    def __init__(
        self, 
        src_radius: int = 400,
        src_color: tuple[int, ...] = (255, 255, 255),
        angle_range: tuple[float, float] = (0, 1),
        num_flare_circles_range: tuple[int, int] = (6, 10),
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
            'num_flare_circles_range': ArgRange(values=[1, Inf.MEDIUM.value], data_type=DataType.INT, is_tuple=True),
            'src_radius': ArgRange(values=[1, Inf.LARGE.value], data_type=DataType.INT),
            'src_color': ArgRange(values=[0, 255], data_type=DataType.INT, is_tuple=True),
            'angle_range': ArgRange(values=[0., 1.], data_type=DataType.FLOAT, is_tuple=True),
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
            'scale': ArgRange(values=[0., 1.], data_type=DataType.FLOAT),
            'per_channel': ArgRange(values=[False, True], data_type=DataType.BOOL)
        }


class SpatterTransform(BaseTransform):
    def __init__(
        self, 
        mean: tuple[float, float] | float = (0.65, 0.65),
        std: tuple[float, float] | float = (0.3, 0.3),
        gauss_sigma: tuple[float, float] | float = (2, 2),
        cutout_threshold: tuple[float, float] | float = (0.68, 0.68),
        intensity: tuple[float, float] | float = (0.6, 0.6),
        mode: Literal['rain', 'mud'] = "rain",
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
            'mean': ArgRange(values=[0., 1.], data_type=DataType.FLOAT, is_tuple=True),
            'std': ArgRange(values=[0., 1.], data_type=DataType.FLOAT, is_tuple=True),
            'gauss_sigma': ArgRange(values=[0., Inf.SMALL.value], data_type=DataType.FLOAT, is_tuple=True),
            'cutout_threshold': ArgRange(values=[0., 1.], data_type=DataType.FLOAT, is_tuple=True),
            'intensity': ArgRange(values=[0., 1.], data_type=DataType.FLOAT, is_tuple=True),
            'mode': ArgRange(values=['rain', 'mud'], data_type=DataType.STR)
        }


class ChromaticAberrationTransform(BaseTransform):
    def __init__(
        self, 
        primary_distortion_limit: tuple[float, float] | float = (-0.02, 0.02),
        secondary_distortion_limit: tuple[float, float] | float = (-0.05, 0.05),
        mode: Literal['green_purple', 'red_blue', 'random'] = "green_purple",
        interpolation: Any = cv2.INTER_LINEAR,
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
            'primary_distortion_limit': ArgRange(values=[-Inf.TINY.value, Inf.TINY.value], data_type=DataType.FLOAT, is_tuple=True),
            'secondary_distortion_limit': ArgRange(values=[-Inf.TINY.value, Inf.TINY.value], data_type=DataType.FLOAT, is_tuple=True),
            'mode': ArgRange(values=['green_purple', 'red_blue', 'random'], data_type=DataType.STR),
            'interpolation': ArgRange(values=[cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4], data_type=DataType.INT)
        }


class DefocusTransform(BaseTransform):
    def __init__(
        self, 
        radius: tuple[int, int] | int = (3, 10),
        alias_blur: tuple[float, float] | float = (0.1, 0.5),
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
            'radius': ArgRange(values=[1, Inf.SMALL.value], data_type=DataType.INT, is_tuple=True),
            'alias_blur': ArgRange(values=[0., Inf.TINY.value], data_type=DataType.FLOAT, is_tuple=True)
        }


class ZoomBlurTransform(BaseTransform):
    def __init__(
        self, 
        max_factor: tuple[float, float] | float = (1, 1.31),
        step_factor: tuple[float, float] | float = (0.01, 0.03),
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
            'max_factor': ArgRange(values=[1., Inf.SMALL.value], data_type=DataType.FLOAT, is_tuple=True),
            'step_factor': ArgRange(values=[0., Inf.TINY.value], data_type=DataType.FLOAT, is_tuple=True),
        }


class MorphologicalTransform(BaseTransform):
    def __init__(
        self, 
        scale: tuple[int, int] | int = (2, 3),
        operation: Literal['erosion', 'dilation'] = "dilation",
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
            'scale': ArgRange(values=[1, Inf.SMALL.value], data_type=DataType.INT, is_tuple=True),
            'operation': ArgRange(values=['dilation', 'erosion'], data_type=DataType.STR)
        }


class ShotNoiseTransform(BaseTransform):
    def __init__(
        self, 
        scale_range: tuple[float, float] = (0.1, 0.3),
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
            'scale_range': ArgRange(values=[0., Inf.TINY.value], data_type=DataType.FLOAT, is_tuple=True)
        }
