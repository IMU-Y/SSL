import os
import numpy as np
from skimage import transform
from skimage import io
from typing import Tuple, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_single_image(args: Tuple) -> None:
    """处理单张图像的数据增强
    
    Args:
        args: (图像, 标注, 索引, 保存路径, 方向, 翻转方式)
    """
    img, gt, idx, aug_dir_val, orient, flip = args
    
    try:
        # 旋转
        img_o = transform.rotate(img, orient)
        gt_o = transform.rotate(gt, orient)
        
        # 翻转
        if flip == 0:
            img_o_f, gt_o_f = img_o, gt_o
        elif flip == 1 and (orient == 90 or orient == 0):
            img_o_f, gt_o_f = np.fliplr(img_o), np.fliplr(gt_o)
        elif flip == 2 and (orient == 90 or orient == 0):
            img_o_f, gt_o_f = np.flipud(img_o), np.flipud(gt_o)
        else:
            return None
            
        # 创建保存路径
        img_save_dir = Path(aug_dir_val) / 'images'
        gt_save_dir = Path(aug_dir_val) / 'gt' 
        
        img_save_dir.mkdir(parents=True, exist_ok=True)
        gt_save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存图像
        img_o_f = (img_o_f * 255).astype(np.uint8)
        gt_o_f = (gt_o_f * 255).astype(np.uint8)
        
        img_path = img_save_dir / f'{idx+4124}.png'
        gt_path = gt_save_dir / f'{idx+4124}.png'
        
        io.imsave(img_path, img_o_f)
        io.imsave(gt_path, gt_o_f)
        
        return (f'semi_mamba/images/{4124+idx}.png',
                f'semi_mamba/gt/{4124+idx}.png')
                
    except Exception as e:
        logger.error(f"处理图像 {idx} 时出错: {e}")
        return None

def augment_dataset(data_type: str = 'val') -> None:
    """数据增强主函数
    
    Args:
        data_type: 'train' 或 'val'
    """
    root_dir = Path('Mamba-UNet/data/ACDC')
    
    # 加载数据
    try:
        images_set = np.load(Path('dataset') / 'images' / data_type / 'dcm.npy')
        gts_set = np.load(Path('dataset') / 'gt' / data_type / 'nii.npy')
    except Exception as e:
        logger.error(f"加载数据集出错: {e}")
        return

    orients = [0]
    flips = [0]
    aug_dir = root_dir / 'data'
    
    # 准备并行处理的参数
    tasks = []
    for i in range(images_set.shape[0]):
        img = images_set[i].astype(np.uint8)
        gt = gts_set[i].astype(np.uint8)
        for o in orients:
            for f in flips:
                tasks.append((img, gt, i, aug_dir, o, f))
    
    # 并行处理图像
    results = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for result in executor.map(process_single_image, tasks):
            if result:
                results.append(result)
    
    # 写入配对信息
    with open(root_dir / f'{data_type}_pair.lst', 'w') as f:
        for img_path, gt_path in results:
            f.write(f'{img_path} {gt_path}\n')

if __name__ == '__main__':
    augment_dataset('test')
    logger.info('数据增强完成')
