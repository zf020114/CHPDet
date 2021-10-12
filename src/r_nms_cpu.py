# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 10:15:36 2019

@author: admin
"""

#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

def nms_rotate_cpu(boxes, scores, iou_threshold, max_output_size):
    keep = []#保留框的结果集合
    order = scores.argsort()[::-1]#对检测结果得分进行降序排序
    num = boxes.shape[0]#获取检测框的个数

    suppressed = np.zeros((num), dtype=np.int)
    for _i in range(num):
        if len(keep) >= max_output_size:#若当前保留框集合中的个数大于max_output_size时，直接返回
            break

        i = order[_i]
        if suppressed[i] == 1:#对于抑制的检测框直接跳过
            continue
        keep.append(i)#保留当前框的索引
        # (midx,midy),(width,height), angle)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4]) 
#        r1 = ((boxes[i, 1], boxes[i, 0]), (boxes[i, 3], boxes[i, 2]), boxes[i, 4]) #根据box信息组合成opencv中的旋转bbox
#        print("r1:{}".format(r1))
        area_r1 = boxes[i, 2] * boxes[i, 3]#计算当前检测框的面积
        for _j in range(_i + 1, num):#对剩余的而进行遍历
            j = order[_j]
            if suppressed[i] == 1:
                continue
            r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
            area_r2 = boxes[j, 2] * boxes[j, 3]
            inter = 0.0

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]#求两个旋转矩形的交集，并返回相交的点集合
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)#求点集的凸边形
                int_area = cv2.contourArea(order_pts)#计算当前点集合组成的凸边形的面积
                inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + 0.0000001)

            if inter >= iou_threshold:#对大于设定阈值的检测框进行滤除
                suppressed[j] = 1

    return np.array(keep, np.int64)


if __name__ == '__main__':
    img = cv2.imread('E:/ship/0.jpg')
    boxes = np.array([[50, 40, 100, 100, 0],
                      [60, 50, 100, 100, 0],
                      [50, 30, 100, 100, -45.],
                      [200, 190, 100, 100, 0.]])
    
    scores = np.array([0.99, 0.88, 0.66, 0.77])
    keep = nms_rotate_cpu(boxes,scores,0.7, 5)
 