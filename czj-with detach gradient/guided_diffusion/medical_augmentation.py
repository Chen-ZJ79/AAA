"""
医学图像数据增强模块
支持多模态MRI图像的一致性变换
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torchvision import transforms
import cv2
from scipy.ndimage import rotate, zoom, shift
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


class MedicalImageAugmentation:
    """
    医学图像数据增强类
    支持多模态MRI图像的一致性变换
    """
    
    def __init__(self, 
                 rotation_range=15,
                 translation_range=0.1,
                 scale_range=0.1,
                 flip_prob=0.5,
                 noise_prob=0.3,
                 noise_std=0.05,
                 elastic_prob=0.3,
                 elastic_alpha=1000,
                 elastic_sigma=30,
                 brightness_range=0.1,
                 contrast_range=0.1):
        """
        初始化数据增强参数
        
        Args:
            rotation_range: 旋转角度范围（度）
            translation_range: 平移范围（相对于图像尺寸的比例）
            scale_range: 缩放范围
            flip_prob: 翻转概率
            noise_prob: 添加噪声的概率
            noise_std: 噪声标准差
            elastic_prob: 弹性变换概率
            elastic_alpha: 弹性变换强度
            elastic_sigma: 弹性变换平滑度
            brightness_range: 亮度调整范围
            contrast_range: 对比度调整范围
        """
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.elastic_prob = elastic_prob
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def random_rotation(self, image, mask, angle=None):
        """
        随机旋转（保持多模态一致性）
        
        Args:
            image: 输入图像 [C, H, W]
            mask: 分割掩码 [H, W]
            angle: 指定角度，如果为None则随机生成
            
        Returns:
            rotated_image, rotated_mask
        """
        if angle is None:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
        
        # 转换为numpy进行旋转
        if isinstance(image, torch.Tensor):
            image_np = image.numpy()
            mask_np = mask.numpy()
        else:
            image_np = image
            mask_np = mask
        
        # 对每个通道进行旋转
        rotated_channels = []
        for c in range(image_np.shape[0]):
            rotated_channel = rotate(image_np[c], angle, reshape=False, order=1)
            rotated_channels.append(rotated_channel)
        
        rotated_image = np.stack(rotated_channels, axis=0)
        rotated_mask = rotate(mask_np, angle, reshape=False, order=0)  # 最近邻插值
        
        # 转换回tensor
        if isinstance(image, torch.Tensor):
            rotated_image = torch.from_numpy(rotated_image).float()
            rotated_mask = torch.from_numpy(rotated_mask).long()
        
        return rotated_image, rotated_mask
    
    def random_translation(self, image, mask, shift_x=None, shift_y=None):
        """
        随机平移（保持多模态一致性）
        
        Args:
            image: 输入图像 [C, H, W]
            mask: 分割掩码 [H, W]
            shift_x, shift_y: 指定平移量，如果为None则随机生成
            
        Returns:
            translated_image, translated_mask
        """
        h, w = image.shape[1], image.shape[2]
        
        if shift_x is None:
            shift_x = random.uniform(-self.translation_range * w, self.translation_range * w)
        if shift_y is None:
            shift_y = random.uniform(-self.translation_range * h, self.translation_range * h)
        
        # 转换为numpy进行平移
        if isinstance(image, torch.Tensor):
            image_np = image.numpy()
            mask_np = mask.numpy()
        else:
            image_np = image
            mask_np = mask
        
        # 对每个通道进行平移
        translated_channels = []
        for c in range(image_np.shape[0]):
            translated_channel = shift(image_np[c], [shift_y, shift_x], order=1)
            translated_channels.append(translated_channel)
        
        translated_image = np.stack(translated_channels, axis=0)
        translated_mask = shift(mask_np, [shift_y, shift_x], order=0)  # 最近邻插值
        
        # 转换回tensor
        if isinstance(image, torch.Tensor):
            translated_image = torch.from_numpy(translated_image).float()
            translated_mask = torch.from_numpy(translated_mask).long()
        
        return translated_image, translated_mask
    
    def random_scale(self, image, mask, scale_factor=None):
        """
        随机缩放（保持多模态一致性）
        
        Args:
            image: 输入图像 [C, H, W]
            mask: 分割掩码 [H, W]
            scale_factor: 指定缩放因子，如果为None则随机生成
            
        Returns:
            scaled_image, scaled_mask
        """
        if scale_factor is None:
            scale_factor = random.uniform(1 - self.scale_range, 1 + self.scale_range)
        
        # 转换为numpy进行缩放
        if isinstance(image, torch.Tensor):
            image_np = image.numpy()
            mask_np = mask.numpy()
        else:
            image_np = image
            mask_np = mask
        
        # 对每个通道进行缩放
        scaled_channels = []
        for c in range(image_np.shape[0]):
            scaled_channel = zoom(image_np[c], scale_factor, order=1)
            # 如果缩放后尺寸不匹配，进行裁剪或填充
            if scaled_channel.shape != image_np[c].shape:
                scaled_channel = self._resize_to_match(scaled_channel, image_np[c].shape)
            scaled_channels.append(scaled_channel)
        
        scaled_image = np.stack(scaled_channels, axis=0)
        scaled_mask = zoom(mask_np, scale_factor, order=0)  # 最近邻插值
        if scaled_mask.shape != mask_np.shape:
            scaled_mask = self._resize_to_match(scaled_mask, mask_np.shape)
        
        # 转换回tensor
        if isinstance(image, torch.Tensor):
            scaled_image = torch.from_numpy(scaled_image).float()
            scaled_mask = torch.from_numpy(scaled_mask).long()
        
        return scaled_image, scaled_mask
    
    def random_flip(self, image, mask, flip_h=None, flip_v=None):
        """
        随机翻转（保持多模态一致性）
        
        Args:
            image: 输入图像 [C, H, W]
            mask: 分割掩码 [H, W]
            flip_h, flip_v: 指定翻转方向，如果为None则随机决定
            
        Returns:
            flipped_image, flipped_mask
        """
        if flip_h is None:
            flip_h = random.random() < self.flip_prob
        if flip_v is None:
            flip_v = random.random() < self.flip_prob
        
        flipped_image = image.clone() if isinstance(image, torch.Tensor) else image.copy()
        flipped_mask = mask.clone() if isinstance(mask, torch.Tensor) else mask.copy()
        
        if flip_h:
            flipped_image = torch.flip(flipped_image, dims=[2]) if isinstance(flipped_image, torch.Tensor) else np.flip(flipped_image, axis=2)
            flipped_mask = torch.flip(flipped_mask, dims=[1]) if isinstance(flipped_mask, torch.Tensor) else np.flip(flipped_mask, axis=1)
        
        if flip_v:
            flipped_image = torch.flip(flipped_image, dims=[1]) if isinstance(flipped_image, torch.Tensor) else np.flip(flipped_image, axis=1)
            flipped_mask = torch.flip(flipped_mask, dims=[0]) if isinstance(flipped_mask, torch.Tensor) else np.flip(flipped_mask, axis=0)
        
        return flipped_image, flipped_mask
    
    def add_noise(self, image, noise_std=None):
        """
        添加高斯噪声
        
        Args:
            image: 输入图像 [C, H, W]
            noise_std: 噪声标准差，如果为None则使用默认值
            
        Returns:
            noisy_image
        """
        if noise_std is None:
            noise_std = self.noise_std
        
        if isinstance(image, torch.Tensor):
            noise = torch.randn_like(image) * noise_std
            noisy_image = image + noise
        else:
            noise = np.random.normal(0, noise_std, image.shape)
            noisy_image = image + noise
        
        return noisy_image
    
    def elastic_transform(self, image, mask, alpha=None, sigma=None):
        """
        弹性变换（保持多模态一致性）
        
        Args:
            image: 输入图像 [C, H, W]
            mask: 分割掩码 [H, W]
            alpha: 弹性变换强度
            sigma: 弹性变换平滑度
            
        Returns:
            transformed_image, transformed_mask
        """
        if alpha is None:
            alpha = self.elastic_alpha
        if sigma is None:
            sigma = self.elastic_sigma
        
        # 转换为numpy进行弹性变换
        if isinstance(image, torch.Tensor):
            image_np = image.numpy()
            mask_np = mask.numpy()
        else:
            image_np = image
            mask_np = mask
        
        # 生成随机位移场
        h, w = image_np.shape[1], image_np.shape[2]
        dx = gaussian_filter((np.random.random((h, w)) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.random((h, w)) * 2 - 1), sigma) * alpha
        
        # 创建坐标网格
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        # 对每个通道进行弹性变换
        transformed_channels = []
        for c in range(image_np.shape[0]):
            transformed_channel = map_coordinates(image_np[c], indices, order=1, mode='reflect')
            transformed_channel = transformed_channel.reshape((h, w))
            transformed_channels.append(transformed_channel)
        
        transformed_image = np.stack(transformed_channels, axis=0)
        transformed_mask = map_coordinates(mask_np, indices, order=0, mode='reflect')
        transformed_mask = transformed_mask.reshape((h, w))
        
        # 转换回tensor
        if isinstance(image, torch.Tensor):
            transformed_image = torch.from_numpy(transformed_image).float()
            transformed_mask = torch.from_numpy(transformed_mask).long()
        
        return transformed_image, transformed_mask
    
    def adjust_brightness_contrast(self, image, brightness_factor=None, contrast_factor=None):
        """
        调整亮度和对比度
        
        Args:
            image: 输入图像 [C, H, W]
            brightness_factor: 亮度调整因子
            contrast_factor: 对比度调整因子
            
        Returns:
            adjusted_image
        """
        if brightness_factor is None:
            brightness_factor = random.uniform(-self.brightness_range, self.brightness_range)
        if contrast_factor is None:
            contrast_factor = random.uniform(1 - self.contrast_range, 1 + self.contrast_range)
        
        adjusted_image = image.clone() if isinstance(image, torch.Tensor) else image.copy()
        
        # 调整亮度
        adjusted_image = adjusted_image + brightness_factor
        
        # 调整对比度
        mean_val = adjusted_image.mean()
        adjusted_image = (adjusted_image - mean_val) * contrast_factor + mean_val
        
        # 确保值在合理范围内
        adjusted_image = torch.clamp(adjusted_image, 0, 1) if isinstance(adjusted_image, torch.Tensor) else np.clip(adjusted_image, 0, 1)
        
        return adjusted_image
    
    def _resize_to_match(self, array, target_shape):
        """
        调整数组尺寸以匹配目标形状
        """
        if array.shape == target_shape:
            return array
        
        # 计算裁剪或填充
        h_diff = array.shape[0] - target_shape[0]
        w_diff = array.shape[1] - target_shape[1]
        
        if h_diff > 0:
            start_h = h_diff // 2
            array = array[start_h:start_h + target_shape[0], :]
        elif h_diff < 0:
            pad_h = (-h_diff) // 2
            array = np.pad(array, ((pad_h, pad_h), (0, 0)), mode='constant')
        
        if w_diff > 0:
            start_w = w_diff // 2
            array = array[:, start_w:start_w + target_shape[1]]
        elif w_diff < 0:
            pad_w = (-w_diff) // 2
            array = np.pad(array, ((0, 0), (pad_w, pad_w)), mode='constant')
        
        return array
    
    def __call__(self, image, mask, grade_label=None):
        """
        执行完整的数据增强流程
        
        Args:
            image: 输入图像 [C, H, W]
            mask: 分割掩码 [H, W]
            grade_label: 病理分级标签（不会被变换）
            
        Returns:
            augmented_image, augmented_mask, grade_label
        """
        # 随机旋转
        if random.random() < 0.7:  # 70%概率应用旋转
            image, mask = self.random_rotation(image, mask)
        
        # 随机平移
        if random.random() < 0.7:  # 70%概率应用平移
            image, mask = self.random_translation(image, mask)
        
        # 随机缩放
        if random.random() < 0.5:  # 50%概率应用缩放
            image, mask = self.random_scale(image, mask)
        
        # 随机翻转
        if random.random() < self.flip_prob:
            image, mask = self.random_flip(image, mask)
        
        # 添加噪声
        if random.random() < self.noise_prob:
            image = self.add_noise(image)
        
        # 弹性变换
        if random.random() < self.elastic_prob:
            image, mask = self.elastic_transform(image, mask)
        
        # 调整亮度和对比度
        if random.random() < 0.3:  # 30%概率调整亮度对比度
            image = self.adjust_brightness_contrast(image)
        
        return image, mask, grade_label


class MedicalImageAugmentationPipeline:
    """
    医学图像数据增强流水线
    支持训练时的随机增强和验证时的确定性变换
    """
    
    def __init__(self, 
                 is_training=True,
                 augmentation_config=None):
        """
        初始化增强流水线
        
        Args:
            is_training: 是否为训练模式
            augmentation_config: 增强配置字典
        """
        self.is_training = is_training
        
        if augmentation_config is None:
            augmentation_config = {
                'rotation_range': 15,
                'translation_range': 0.1,
                'scale_range': 0.1,
                'flip_prob': 0.5,
                'noise_prob': 0.3,
                'noise_std': 0.05,
                'elastic_prob': 0.3,
                'elastic_alpha': 1000,
                'elastic_sigma': 30,
                'brightness_range': 0.1,
                'contrast_range': 0.1
            }
        
        if is_training:
            self.augmentation = MedicalImageAugmentation(**augmentation_config)
        else:
            # 验证时不进行增强
            self.augmentation = None
    
    def __call__(self, image, mask, grade_label=None):
        """
        执行数据增强
        
        Args:
            image: 输入图像 [C, H, W]
            mask: 分割掩码 [H, W]
            grade_label: 病理分级标签
            
        Returns:
            augmented_image, augmented_mask, grade_label
        """
        if self.is_training and self.augmentation is not None:
            return self.augmentation(image, mask, grade_label)
        else:
            return image, mask, grade_label


def create_augmentation_pipeline(is_training=True, config_name='default'):
    """
    创建数据增强流水线的工厂函数
    
    Args:
        is_training: 是否为训练模式
        config_name: 配置名称
        
    Returns:
        MedicalImageAugmentationPipeline
    """
    configs = {
        'default': {
            'rotation_range': 15,
            'translation_range': 0.1,
            'scale_range': 0.1,
            'flip_prob': 0.5,
            'noise_prob': 0.3,
            'noise_std': 0.05,
            'elastic_prob': 0.3,
            'elastic_alpha': 1000,
            'elastic_sigma': 30,
            'brightness_range': 0.1,
            'contrast_range': 0.1
        },
        'light': {
            'rotation_range': 10,
            'translation_range': 0.05,
            'scale_range': 0.05,
            'flip_prob': 0.3,
            'noise_prob': 0.2,
            'noise_std': 0.03,
            'elastic_prob': 0.2,
            'elastic_alpha': 500,
            'elastic_sigma': 20,
            'brightness_range': 0.05,
            'contrast_range': 0.05
        },
        'heavy': {
            'rotation_range': 30,
            'translation_range': 0.2,
            'scale_range': 0.2,
            'flip_prob': 0.7,
            'noise_prob': 0.5,
            'noise_std': 0.1,
            'elastic_prob': 0.5,
            'elastic_alpha': 2000,
            'elastic_sigma': 50,
            'brightness_range': 0.2,
            'contrast_range': 0.2
        }
    }
    
    config = configs.get(config_name, configs['default'])
    return MedicalImageAugmentationPipeline(is_training=is_training, augmentation_config=config)


if __name__ == "__main__":
    # 测试数据增强
    print("测试医学图像数据增强模块...")

    test_image = torch.randn(4, 224, 224)  # 4个模态
    test_mask = torch.randint(0, 4, (224, 224))  # 分割掩码
    test_grade = torch.tensor(1)  # HGG

    aug_pipeline = create_augmentation_pipeline(is_training=True, config_name='default')

    aug_image, aug_mask, aug_grade = aug_pipeline(test_image, test_mask, test_grade)
    
    print(f"原始图像形状: {test_image.shape}")
    print(f"增强后图像形状: {aug_image.shape}")
    print(f"原始掩码形状: {test_mask.shape}")
    print(f"增强后掩码形状: {aug_mask.shape}")
    print(f"病理分级标签: {aug_grade}")
    print("数据增强测试完成！")









