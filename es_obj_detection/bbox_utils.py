#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, sys, pickle, getopt, math
sys.path.append(os.path.abspath('..'))
import torch
from torch import nn, optim
from torchvision import transforms
from es_pytorch_onnx.common_torch import *

def MultiBoxPrior(feature_map, sizes=[0.75, 0.5], ratios=[1, 2, 0.5]):
    """
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        feature_map: torch tensor, Shape: [N, C, H, W].
        sizes: List of sizes (0~1) of generated MultiBoxPriores.
        ratios: List of aspect ratios (non-negative) of generated MultiBoxPriores.
    Returns:
        anchors of shape (1, num_anchors, 4). 由于batch里每个都一样, 所以第一维为1
    """
    pairs = [] # pair of (size, sqrt(ration))

    for r in ratios:
        pairs.append([sizes[0], torch.sqrt(r)])
    for s in sizes[1:]:
        pairs.append([s, torch.sqrt(ratios[0])])

    pairs = torch.tensor(pairs, device=device)

    ss1 = pairs[:, 0] * pairs[:, 1] # size * sqrt(ration)
    ss2 = pairs[:, 0] / pairs[:, 1] # size / sqrt(ration)

    base_anchors = torch.stack([-ss1, -ss2, ss1, ss2], dim=1) / 2

    h, w = feature_map.shape[-2:]
    shifts_x = torch.arange(0, w, device=device) // w
    shifts_y = torch.arange(0, h, device=device) // h
    shift_x, shift_y = torch.meshgrid(shifts_x, shifts_y)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

    anchors = shifts.reshape((-1, 1, 4)) + base_anchors.reshape((1, -1, 4))
    return anchors.view(1, -1, 4)

def compute_intersection(set_1, set_2):
    """
    计算anchor之间的交集
    Args:
        set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
        set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
    Returns:
        intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, shape: (n1, n2)
    """
    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

def compute_jaccard(set_1, set_2):
    """
    计算anchor之间的Jaccard系数(IoU)
    Args:
        set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
        set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
    Returns:
        Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, shape: (n1, n2)
    """
    # Find intersections
    intersection = compute_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)

def assign_anchor(bb, anchor, jaccard_threshold=0.5):
    """
    # 按照「9.4.1. 生成多个锚框」图9.3所讲为每个anchor分配真实的bb, anchor表示成归一化(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        bb: 真实边界框(bounding box), shape:（nb, 4）
        anchor: 待分配的anchor, shape:（na, 4）
        jaccard_threshold: 预先设定的阈值
    Returns:
        assigned_idx: shape: (na, ), 每个anchor分配的真实bb对应的索引, 若未分配任何bb则为-1
    """
    na = anchor.shape[0]
    nb = bb.shape[0]
    # print("na: ", na)
    # print("nb: ", nb)
    # jaccard = compute_jaccard(anchor, bb).detach().cpu().numpy() # shape: (na, nb)
    jaccard = compute_jaccard(anchor, bb) # shape: (na, nb)
    assigned_idx = torch.ones(na, dtype=torch.long, device=device) * -1  # 初始全为-1
    # 先为每个bb分配一个anchor(不要求满足jaccard_threshold)
    # print('jaccard.shape: ', jaccard.shape, ' jaccard.device: ', jaccard.device)
    jaccard_cp = jaccard.clone()

    jdx_arr = torch.tensor(range(nb), device=device)
    idx_arr = torch.argmax(jaccard_cp[:, jdx_arr], dim=0)
    assigned_idx[idx_arr] = jdx_arr
    jaccard_cp[idx_arr, :] = float("-inf") # 赋值为负无穷, 相当于去掉这一行
    '''
    for j in range(nb):
        i = torch.argmax(jaccard_cp[:, j])
        assigned_idx[i] = j
        jaccard_cp[i, :] = float("-inf") # 赋值为负无穷, 相当于去掉这一行
    '''

    # 处理还未被分配的anchor, 要求满足jaccard_threshold
    idx_mask = assigned_idx == -1
    idx_arr = torch.tensor(range(na), device=device)
    j_arr = torch.argmax(jaccard[idx_arr[idx_mask], :], dim=1)
    # print('j_arr: ', j_arr)
    threshold_mask = jaccard[idx_arr[idx_mask], j_arr] >= jaccard_threshold
    assigned_idx[idx_mask][threshold_mask] = j_arr[threshold_mask]
    '''
    print('matched_idx_arr.shape: ', matched_idx_arr.shape)
    print('jaccard.shape: ', jaccard.shape)
    print('j_arr.shape: ', j_arr.shape)
    print('threshold_mask.shape: ', threshold_mask.shape)
    print('assigned_idx.shape: ', assigned_idx.shape)
    print('assigned_idx[idx_mask][threshold_mask].shape: ', assigned_idx[idx_mask][threshold_mask].shape)
    print('j_arr[threshold_mask].shape: ', j_arr[threshold_mask].shape)
    '''
    '''
    for i in range(na):
        if assigned_idx[i] == -1:
            j = torch.argmax(jaccard[i, :])
            # print("j: ", j)
            if jaccard[i, j] >= jaccard_threshold:
                assigned_idx[i] = j
    '''

    return assigned_idx

def xy_to_cxcy(xy):
    """
    将(x_min, y_min, x_max, y_max)形式的anchor转换成(center_x, center_y, w, h)形式的.
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
    Args:
        xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    Returns:
        bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def MultiBoxTarget_one(anc, lab, eps=1e-6):
    """
    MultiBoxTarget函数的辅助函数, 处理batch中的一个
    Args:
        anc: shape of (锚框总数, 4)
        lab: shape of (真实锚框数, 5), 5代表[类别标签, 四个坐标值]
        eps: 一个极小值, 防止log0
    Returns:
        offset: (锚框总数*4, )
        bbox_mask: (锚框总数*4, ), 0代表背景, 1代表非背景
        cls_labels: (锚框总数, 4), 0代表背景
    """
    an = anc.shape[0]
    func_start = time.time()
    assigned_idx = assign_anchor(lab[:, 1:], anc) # (锚框总数, )
    # print("assign_anchor func time: %.4f sec" %(time.time() - func_start))
    bbox_mask = ((assigned_idx >= 0).float().unsqueeze(-1)).repeat(1, 4) # (锚框总数, 4)

    cls_labels = torch.zeros(an, dtype=torch.long, device=device) # 0表示背景
    assigned_bb = torch.zeros((an, 4), dtype=torch.float32, device=device) # 所有anchor对应的bb坐标
    # parallel acc
    idx_mask = assigned_idx >= 0 # 即非背景
    idx_arr = torch.tensor(range(an), device=device)
    cls_labels[idx_arr[idx_mask]] = lab[assigned_idx[idx_mask], 0].long() + 1 # 注意要加一
    assigned_bb[idx_arr[idx_mask], :] = lab[assigned_idx[idx_mask], 1:]
    '''
    original code:

    print('tmp_idx.shape: ', tmp_idx.shape, ' tmp_idx.device: ', tmp_idx.device)
    for i in range(an):
        bb_idx = assigned_idx[i]
        if bb_idx >= 0: # 即非背景
            cls_labels[i] = lab[bb_idx, 0].long().item() + 1 # 注意要加一
            assigned_bb[i, :] = lab[bb_idx, 1:]
    '''
    center_anc = xy_to_cxcy(anc) # (center_x, center_y, w, h)
    center_assigned_bb = xy_to_cxcy(assigned_bb)

    offset_xy = 10.0 * (center_assigned_bb[:, :2] - center_anc[:, :2]) / center_anc[:, 2:]
    offset_wh = 5.0 * torch.log(eps + center_assigned_bb[:, 2:] / center_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], dim = 1) * bbox_mask # (锚框总数, 4)

    return offset.view(-1), bbox_mask.view(-1), cls_labels


def MultiBoxTarget(anchor, label_list):
    """
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成归一化(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        anchor: torch tensor, 输入的锚框, 一般是通过MultiBoxPrior生成, shape:（1，锚框总数，4）
        label_list: 真实标签, shape为(bn, 每张图片最多的真实锚框数, 5)
               第二维中，如果给定图片没有这么多锚框, 可以先用-1填充空白, 最后一维中的元素为[类别标签, 四个坐标值]
    Returns:
        列表, [bbox_offset, bbox_mask, cls_labels]
        bbox_offset: 每个锚框的标注偏移量，形状为(bn，锚框总数*4)
        bbox_mask: 形状同bbox_offset, 每个锚框的掩码, 一一对应上面的偏移量, 负类锚框(背景)对应的掩码均为0, 正类锚框的掩码均为1
        cls_labels: 每个锚框的标注类别, 其中0表示为背景, 形状为(bn，锚框总数)
    """
    # assert len(anchor.shape) == 3 and len(label.shape) == 3
    bn = len(label_list)
    batch_offset = []
    batch_mask = []
    batch_cls_labels = []
    # print('bn: ', bn)
    for b in range(bn):
        label_list[b] = label_list[b].to(device)
        offset, bbox_mask, cls_labels = MultiBoxTarget_one(anchor[0, :, :], label_list[b])

        batch_offset.append(offset)
        batch_mask.append(bbox_mask)
        batch_cls_labels.append(cls_labels)

    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    cls_labels = torch.stack(batch_cls_labels)

    return [bbox_offset, bbox_mask, cls_labels]