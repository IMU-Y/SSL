import cv2
import numpy as np
import os
from PIL import Image

def convert_to_custom_color(input_path, output_path):
    # 读取灰度图
    gray_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if gray_img is None:
        print(f"无法读取图片: {input_path}")
        return
    
    # 创建自定义颜色映射
    # 0: 黑色(背景)
    # 1: 红色(正常组织)
    # 2: 黄色(钙化斑块)
    # 3: 绿色(纤维斑块)
    # 4: 蓝色(脂质斑块)
    colors = np.zeros((256, 3), dtype=np.uint8)
    
    # 定义颜色映射
    color_map = {
        0: [0, 0, 0],      # 黑色 - 背景
        1: [255, 0, 0],    # 红色 - 正常组织
        2: [255, 255, 0],  # 黄色 - 钙化斑块
        3: [0, 255, 0],    # 绿色 - 纤维斑块
        4: [0, 0, 255]     # 蓝色 - 脂质斑块
    }
    
    # 设置颜色映射
    for i in range(256):
        if i in color_map:
            colors[i] = color_map[i]
    
    # 保存为PNG格式，包含调色板
    pil_image = Image.fromarray(gray_img, mode='L')
    pil_image.putpalette(colors.ravel())
    pil_image.save(output_path, format='PNG')

def process_directory(input_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历输入目录中的所有PNG文件
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.png'):
                # 构建输入和输出文件路径
                rel_path = os.path.relpath(root, input_dir)
                input_path = os.path.join(root, file)
                output_subdir = os.path.join(output_dir, rel_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_path = os.path.join(output_subdir, file)
                
                # 转换图片
                convert_to_custom_color(input_path, output_path)

                print(f"处理完成: {input_path} -> {output_path}")

# 使用示例
input_directory = "./model/ACDC/Semi_Mamba_UNet_3/mambaunet_predictions"  # 输入目录
output_directory = "./model/ACDC/Semi_Mamba_UNet_3/mambaunet_predictions_color"  # 输出目录
process_directory(input_directory, output_directory)
