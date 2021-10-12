#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:39:49 2020

@author: zf
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os, sys
import cv2
import numpy as np
import shutil
import sys
sys.path.append('/home/zf/0tools')
from DataFunction import get_file_paths_recursive,read_rotate_xml,rotate_rect2cv,rotate_rect2cv_np
# from r_nms_cpu import nms_rotate_cpus
from All_Class_NAME_LABEL import NAME_LABEL_MAP_HRSC, NAME_LABEL_MAP_USnavy_20
#sys.path.append('/media/zf/E/Dataset/DOTA_devkit')
sys.path.append('/home/zf/s2anet_rep/DOTA_devkit')
from dota15 import data_v15_evl_1

def nms_rotate_cpu(boxes, scores, iou_threshold, max_output_size):
    keep = []#保留框的结果集合
    order = scores.argsort()[::-1]#对检测结果得分进行降序排序
    num = boxes.shape[0]#获取检测框的个数
    suppressed = np.zeros((num), dtype=np.int)
    angle_det = np.zeros((num))
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
                angle_det[j]=np.abs( r1 [2]-r2[2])
    return np.array(keep, np.int64),angle_det

def get_label_name_map(NAME_LABEL_MAP):
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict

def eval_rotatexml(GT_xml_dir,   det_xml_dir, NAME_LABEL_MAP,
                   file_ext='.xml', ovthresh=0.5):
    # GT_xml_path 是需要转换为txt的xml文件
    # txt_dir_h是将要写入的xml转换为txt的文件路径
    # annopath 是GT的txt文件
    # 读取原图路径
   
    LABEl_NAME_MAP = get_label_name_map(NAME_LABEL_MAP)
    file_paths = get_file_paths_recursive(GT_xml_dir, '.xml')
    angle_det_all=[]
    for count, xml_path in enumerate(file_paths):
        img_size,gsd,imagesource,gtbox_label,extra=read_rotate_xml(xml_path,NAME_LABEL_MAP)
        det_xml=xml_path.replace(GT_xml_dir,det_xml_dir)
        try:
            img_size,gsd,imagesource,detbox_label,extra=read_rotate_xml(det_xml,NAME_LABEL_MAP)
        except :
            continue
        cvrboxes=[]
        for box in gtbox_label:
            cvrboxes.append(rotate_rect2cv_np(box))
        for box in detbox_label:
            cvrboxes.append(rotate_rect2cv_np(box))
        cvrboxes = np.array(cvrboxes)
        score=np.ones((len(gtbox_label)))
        score_det=np.array(extra)
        score=np.hstack((score,score_det))
        # gtbox_label=np.array(gtbox_label+detbox_label)
        if len(cvrboxes)>0:##斜框NMS
            keep ,angle_det= nms_rotate_cpu(cvrboxes, score, ovthresh,
                                  200)  #这里可以改
            angle_det=np.array(angle_det)
            inx=angle_det>0
            angle_det=angle_det[inx]
            angle_det=angle_det.tolist()
            # assert(len(angle_det)==len(gtbox_label))
            print(len(angle_det))
            angle_det_all+=angle_det
    
    print(len(angle_det_all))
    angle_det_all=np.array(angle_det_all)
    mean_angle=np.mean(angle_det_all)
    print('Mean of  angle is {}'.format(mean_angle))


if __name__ == '__main__':
    #这是手动计算同样一个rotatebox旋转5度iou
    #和不同的检测算法角度预测精度的程序，实验结果表明没有明显的优势
    GT_xml_dir = '/home/zf/Dataset/USnavy_test_gt/train/rotatexml' 
    det_xml_dir = '/home/zf/Dataset/USnavy_test_gt/6center_DLA1024_rotatexml_merge'
    NAME_LABEL_MAP = NAME_LABEL_MAP_USnavy_20
    eval_rotatexml(GT_xml_dir,   det_xml_dir, NAME_LABEL_MAP, ovthresh=0.5)
