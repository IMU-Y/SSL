import argparse
import os
import cv2
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int, default=5,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')


def calculate_metric_percase(pred, gt):
    """
    计算每个类别的Dice系数
    pred: 预测结果，特定类别的二值图（0或1）
    gt: 真实标签，特定类别的二值图（0或1）
    """
    pred = pred.astype(np.int32)
    gt = gt.astype(np.int32)
    
    # 计算Dice系数
    intersection = np.sum(pred * gt)
    pred_sum = np.sum(pred)
    gt_sum = np.sum(gt)
    
    # 修改边界情况处理
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    elif pred_sum == 0 or gt_sum == 0:
        return 0.0
    
    dice = (2.0 * intersection) / (pred_sum + gt_sum)
    return dice


def visualize_results(pred, case, test_save_path):
    """
    只保存预测结果的可视化，不添加文字标记
    """
    # 创建预测结果的可视化图像
    vis_pred = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    
    # 定义颜色映射
    colors = [
        [255, 0, 0],    # 红色 - 正常斑块
        [0, 255, 0],    # 绿色 - 纤维斑块
        [0, 0, 255],    # 蓝色 - 脂质斑块
        [255, 255, 0],  # 黄色 - 钙化斑块
    ]
    
    # 为预测结果上色
    for i in range(1, 5):  # 1-4类（跳过背景0）
        mask_pred = pred == i
        vis_pred[mask_pred] = colors[i-1]
    
    # 保存结果（转换为BGR格式）
    vis_pred = cv2.cvtColor(vis_pred, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(test_save_path, f"{case}.png"), vis_pred)


def test_single_volume(case, net, test_save_path, FLAGS):
    image_path = os.path.join(FLAGS.root_path, "data/images", f"{case}.png")
    label_path = os.path.join(FLAGS.root_path, "data/gt", f"{case}.png")

    # 读取图像并打印信息
    image = cv2.imread(image_path)
    # print(f"\nDebug - Original image shape: {image.shape}, dtype: {image.dtype}, range: [{image.min()}, {image.max()}]")
    
    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 归一化
    image = image.astype(np.float32) / 255.0

    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    
    original_image = image.copy()

    # 调整图像大小
    x, y = image.shape[:2]
    image = zoom(image, (384 / x, 384 / y, 1), order=0)
    
    # 转换为tensor
    input = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().cuda()
    
    # 预测
    net.eval()
    with torch.no_grad():
        out_main = net(input)
        softmax_out = torch.softmax(out_main, dim=1)
        
        # 添加调试信息
        # print("\nDebug softmax probabilities:")
        for i in range(FLAGS.num_classes):
            prob = softmax_out[0, i]
            # print(f"Class {i} max prob: {prob.max().item():.4f}")
        
        out = torch.argmax(softmax_out, dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = zoom(out, (x / 384, y / 384), order=0)
        print(f"Prediction unique values: {np.unique(pred)}")  # 检查预测值

    # 计算指标
    metrics = []
    class_names = ['背景', '正常斑块', '纤维斑块', '脂质斑块', '钙化斑块']
    for i in range(FLAGS.num_classes):
        metric = calculate_metric_percase(pred == i, label == i)
        metrics.append(metric)

    # 可视化预测结果
    visualize_results(pred, case, test_save_path)  # 只传递预测结果

    return metrics


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/{}_{}/{}".format(        
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    test_save_path = "../model/{}_{}/{}_predictions/".format(        
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    
    net = net_factory(net_type=FLAGS.model, in_chns=3, class_num=FLAGS.num_classes)
    
    # 添加模型加载调试
    save_mode_path = os.path.join(snapshot_path, '{}_best_model1.pth'.format(FLAGS.model))
    
    state_dict = torch.load(save_mode_path)
    
    net.load_state_dict(torch.load(save_mode_path))
    net.eval()

    # 修改指标累加部分
    class_names = ['背景', '正常斑块', '纤维斑块', '脂质斑块', '钙化斑块']
    total_metrics = [0.0] * FLAGS.num_classes  # 5个类别的累加器
    
    print("\n开始测试...")
    for case in tqdm(image_list):
        metrics = test_single_volume(case, net, test_save_path, FLAGS)
        for i in range(len(metrics)):
            total_metrics[i] += np.asarray(metrics[i])
    
    # 打印每个类别的详细指标
    print("\n分割结果：")
    print("-" * 50)
    print("类别\t\tDice系数")
    print("-" * 50)
    for i, name in enumerate(class_names):
        avg_metric = total_metrics[i] / len(image_list)
        print(f"{name}\t\t{avg_metric:.4f}")
    print("-" * 50)
    
    # 计算并打印总体平均值
    avg_metrics = [total / len(image_list) for total in total_metrics]
    overall_dice = sum(avg_metrics) / len(avg_metrics)
    print(f"\n总体平均Dice系数: {overall_dice:.4f}")
    
    # 保存结果到文本文件
    with open(os.path.join(test_save_path, 'results.txt'), 'w', encoding='utf-8') as f:
        f.write("分割结果：\n")
        f.write("-" * 50 + "\n")
        f.write("类别\t\tDice系数\n")
        f.write("-" * 50 + "\n")
        for i, name in enumerate(class_names):
            avg_metric = total_metrics[i] / len(image_list)
            f.write(f"{name}\t\t{avg_metric:.4f}\n")
        f.write("-" * 50 + "\n")
        f.write(f"\n总体平均Dice系数: {overall_dice:.4f}\n")
    
    return avg_metrics


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metrics = Inference(FLAGS)
    print('Average metrics per class:', metrics)
    if isinstance(metrics, (list, np.ndarray)):
        print('Overall Average Dice:', np.mean(metrics))
    else:
        print('Single metric:', metrics)
