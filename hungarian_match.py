# -*- coding: utf-8 -*-
"""
Time    : 2022/5/12 19:20
Author  : cong
"""
from scipy.optimize import linear_sum_assignment
import numpy as np


def iou_batch(bb_test, bb_gt):
    """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o


iou_threshold = 0.3
dets = np.load('dets.npy')
tracks = np.load('trks.npy')
# 计算iou相似度矩阵
iou_matrix = iou_batch(dets, tracks)

if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    # axis=1以后就是将一个矩阵的每一行向量相加,0:列相加
    # 每行的元素相加， 求出所有行的max为1，所有列max为1，
    # 如果大于0.3的位置恰好一一对应，可直接得到匹配结果，否则利用匈牙利算法进行匹配
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        # np.where(a) 找出符合条件的元素索引
        print('*'*20)
        matched_indices = np.stack(np.where(a), axis=1) # ndarray
    else:
        x,y = linear_sum_assignment(-iou_matrix) # tuple
        matched_indices = np.array(list(zip(x, y)))

unmatched_detections = []
for d, det in enumerate(dets):
    # matched_indices[:, 0]：array([0, 1, 2])
    if d not in matched_indices[:, 0]:
        unmatched_detections.append(d)
unmatched_trackers = []
for t, trk in enumerate(tracks):
    # matched_indices[:, 1]：array([2, 1, 0])
    if t not in matched_indices[:, 1]:
        unmatched_trackers.append(t)

# filter out matched with low IOU
matches = []
for m in matched_indices:
    if iou_matrix[m[0], m[1]] < iou_threshold:
        unmatched_detections.append(m[0])
        unmatched_trackers.append(m[1])
    else:
        matches.append(m.reshape(1, 2))
if len(matches) == 0:
    matches = np.empty((0, 2), dtype=int)
else:
    matches1 = np.concatenate(matches, axis=0)



