import os
import h5py
import numpy as np

def convert_npy_to_h5(image_npy_dir, label_npy_dir, output_dir):
    """
    将NPY格式的图像和标签转换为HDF5格式
    Args:
        image_npy_dir: 包含图像npy文件的目录
        label_npy_dir: 包含标签npy文件的目录
        output_dir: 输出h5文件的目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有npy文件
    image_files = sorted([f for f in os.listdir(image_npy_dir) if f.endswith('.npy')])
    label_files = sorted([f for f in os.listdir(label_npy_dir) if f.endswith('.npy')])
    
    # 确保图像和标签文件一一对应
    assert len(image_files) == len(label_files), "图像和标签数量不匹配！"
    
    # 处理每对图像和标签
    for img_file, label_file in zip(image_files, label_files):
        # 读取npy文件
        images = np.load(os.path.join(image_npy_dir, img_file))  # (N, H, W, C)
        labels = np.load(os.path.join(label_npy_dir, label_file))  # 假设标签也是4D的
        
        print(f"图像形状: {images.shape}, 标签形状: {labels.shape}")
        
        # 处理每个切片
        for slice_idx in range(images.shape[0]):
            # 获取单个切片
            image = images[slice_idx]  # (H, W, C)
            label = labels[slice_idx]  # 获取对应的标签切片
            
            # 创建h5文件名
            h5_filename = os.path.join(output_dir, 
                                     f"{img_file.replace('.npy', '')}_slice_{slice_idx:04d}.h5")
            
            # 创建h5文件并写入数据
            with h5py.File(h5_filename, 'w') as h5f:
                h5f.create_dataset('image', data=image)  # 存储单个切片
                h5f.create_dataset('label', data=label)  # 存储对应的标签
            
            if slice_idx % 100 == 0:  # 每100个切片打印一次进度
                print(f"处理 {img_file} 的第 {slice_idx}/{images.shape[0]} 个切片")

def create_dataset_list(output_dir, train_ratio=1):
    """
    创建训练集和验证集的列表文件
    Args:
        output_dir: h5文件所在的目录
        train_ratio: 训练集比例
    """
    # 获取所有h5文件
    h5_files = sorted([f.replace('.h5', '') for f in os.listdir(output_dir) if f.endswith('.h5')])
    
    # 随机打乱文件列表
    np.random.shuffle(h5_files)
    
    # 划分训练集和验证集
    train_size = int(len(h5_files) * train_ratio)
    train_files = h5_files[:train_size]
    val_files = h5_files[train_size:]
    
    # 写入训练集列表
    with open(os.path.join(os.path.dirname(output_dir), 'train_slices.list'), 'w') as f:
        f.write('\n'.join(train_files))
    
    # 写入验证集列表
    with open(os.path.join(os.path.dirname(output_dir), 'val.list'), 'w') as f:
        f.write('\n'.join(val_files))
    
    print(f"创建了训练集列表（{len(train_files)}个样本）和验证集列表（{len(val_files)}个样本）")

# 使用示例
if __name__ == "__main__":
    # 设置路径
    image_npy_dir = './dataset/images/train'  # npy格式图像所在目录
    label_npy_dir = './dataset/gt/train'  # npy格式标签所在目录
    output_base_dir = './dataset/h5/train'         # 输出基础目录
    
    # 创建输出目录结构
    output_data_dir = os.path.join(output_base_dir, 'data/slices')
    os.makedirs(output_data_dir, exist_ok=True)
    
    # 转换文件
    convert_npy_to_h5(image_npy_dir, label_npy_dir, output_data_dir)
    
    # 创建数据集列表
    create_dataset_list(output_data_dir)