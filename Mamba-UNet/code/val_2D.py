import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 or gt.sum() > 0:
        return 0.0, 0.0
    else:
        return 1.0, 0.0


def test_single_volume(image, label, net, classes, patch_size=[384, 384]):
    image = image.squeeze(0).cpu().detach().numpy()  # [1,3,H,W] -> [3,H,W]
    label = label.squeeze(0).cpu().detach().numpy()  # [1,H,W] -> [H,W]
    
    # 调整图像大小
    x, y = image.shape[1:]  # 注意这里取shape[1:]因为是[3,H,W]格式
    image = zoom(image, (1, patch_size[0] / x, patch_size[1] / y), order=0)  # 保持通道数不变
    
    # 转换为tensor
    input = torch.from_numpy(image).unsqueeze(0).float().cuda()  # [3,H,W] -> [1,3,H,W]
    
    net.eval()
    with torch.no_grad():
        out = net(input)
        out = torch.softmax(out, dim=1)
        out = torch.argmax(out, dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(pred == i, label == i))
    
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
