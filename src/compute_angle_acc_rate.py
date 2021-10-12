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
    num_det_all=[]
    num_tp_all=[]
    num_gt_all=[]
    for count, xml_path in enumerate(file_paths):
        img_size,gsd,imagesource,gtbox_label,extra=read_rotate_xml(xml_path,NAME_LABEL_MAP)
        # num_gt = np.ones((len(gtbox_label)))
        num_det = np.zeros((len(gtbox_label)))
        num_tp = np.zeros((len(gtbox_label)))
        det_xml=xml_path.replace(GT_xml_dir,det_xml_dir)
        try:
            img_size,gsd,imagesource,detbox_label,extra=read_rotate_xml(det_xml,NAME_LABEL_MAP)
        except :
            continue
        if len(detbox_label)>0:
            for i,box in enumerate( gtbox_label):
                gt_cvrbox=rotate_rect2cv_np(box)
                gt_center=np.array([box[0],box[1]])
                gt_centers=np.repeat(gt_center[None], len(detbox_label), axis=0)
                det_centers=np.array(detbox_label)[:,0:2]
                diff=det_centers-gt_centers
                dist=np.sqrt(np.square(diff[:,0])+np.square(diff[:,1]),)
                index=np.argmin(dist)
                det_gt_box=detbox_label[index]
                det_cvrbox=rotate_rect2cv_np(det_gt_box)
                
                r1 = ((gt_cvrbox[0], gt_cvrbox[1]), (gt_cvrbox[2], gt_cvrbox[3]), gt_cvrbox[4]) 

                area_r1 = gt_cvrbox[2] * gt_cvrbox[3]#计算当前检测框的面积
                r2 = ((det_cvrbox[0], det_cvrbox[1]), (det_cvrbox[ 2], det_cvrbox[3]), det_cvrbox[4])
                area_r2 = det_cvrbox[2] * det_cvrbox[3]
                inter = 0.0
                int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]#求两个旋转矩形的交集，并返回相交的点集合
                if int_pts is not None:
                    order_pts = cv2.convexHull(int_pts, returnPoints=True)#求点集的凸边形
                    int_area = cv2.contourArea(order_pts)#计算当前点集合组成的凸边形的面积
                    inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + 0.0000001)
                if inter >= ovthresh:#对大于设定阈值的检测框进行滤除
                    num_det[i]=1
                    # angle_diff=np.min([np.abs(gtbox_label[i][4]-detbox_label[int(index)][4]),
                    #                    np.abs(gtbox_label[i][4]-detbox_label[int(index)][4]+np.pi*2),
                    #                    np.abs(gtbox_label[i][4]-detbox_label[int(index)][4]-np.pi*2),
                    #                   np.abs(gtbox_label[i][4]-detbox_label[int(index)][4]+np.pi),
                    #                    np.abs(gtbox_label[i][4]-detbox_label[int(index)][4]-np.pi)])
                    angle_diff=np.min([np.abs(gtbox_label[i][4]-detbox_label[int(index)][4]),
                                       np.abs(gtbox_label[i][4]-detbox_label[int(index)][4]+np.pi*2),
                                       np.abs(gtbox_label[i][4]-detbox_label[int(index)][4]-np.pi*2)])
                    if angle_diff<np.pi/18:
                        num_tp[i]=1
                    else:
                        print('xmlpath {}'.format(xml_path))
                        print('gt index {}'.format(i))
                        print('det index {}'.format(index))
                        
            num_det=num_det.tolist()
            num_tp=num_tp.tolist()
            # num_gt=num_gt.tolist()
            # assert(len(angle_det)==len(gtbox_label))
            print(len(num_det))
            num_det_all+=num_det
            num_tp_all+=num_tp
    
    print(len(num_det_all))
    num_det_all=np.array(num_det_all)
    mean_det=np.mean(num_det_all)
    num_tp_all=np.array(num_tp_all)
    mean_tp=np.mean(num_tp_all)
    print('Mean of  det is {}'.format(mean_det))
    print('Mean of  gt is {}'.format(mean_tp))
    print('bow acc rate {}'.format(mean_tp/mean_det))
if __name__ == '__main__':
    #这是手动计算同样一个rotatebox旋转5度iou
    #和不同的检测算法角度预测精度的程序，实验结果表明没有明显的优势
    GT_xml_dir = '/home/zf/Dataset/USnavy_test_gt/train/rotatexml' 
    # det_xml_dir =  '/home/zf/Dataset/USnavy_test_gt/CHE_Hourglass512'
    det_xml_dir = '/home/zf/Dataset/USnavy_test_gt/6center_DLA1024_rotatexml_merge'
    # det_xml_dir = '/home/zf/Dataset/USnavy_test_gt/5s2a_show_rotatexml_merge'
    NAME_LABEL_MAP = NAME_LABEL_MAP_USnavy_20
    eval_rotatexml(GT_xml_dir,   det_xml_dir, NAME_LABEL_MAP, ovthresh=0.5)
