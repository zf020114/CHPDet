#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 23:00:51 2020

@author: zf
"""
# import pycocotools.coco as coco
# from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt


import cv2
from PIL import Image, ImageDraw
import numpy as np
from scipy import stats as st
import sys
sys.path.append('/home/zf/0tools')
from py_cpu_nms import py_cpu_nms
from r_nms_cpu import nms_rotate_cpu
from DataFunction import write_rotate_xml, read_rotate_xml, rotate_rect2cv_np
from All_Class_NAME_LABEL import NAME_LABEL_MAP_DOTA10
from Rotatexml2DotaTxT import eval_rotatexml

def get_label_name_map(NAME_LABEL_MAP):
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict

def get_file_paths_recursive(folder=None, file_ext=None):
    """ Get the absolute path of all files in given folder recursively
    :param folder:
    :param file_ext:
    :return:
    """
    file_list = []
    if folder is None:
        return file_list
    file_list = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith(file_ext)]
    return file_list

def id2name(coco):
    classes=dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']]=cls['name']
    return classes

def show_rotate_box(src_img,rotateboxes,name=None):
    cx, cy, w,h,Angle=rotateboxes[:,0], rotateboxes[:,1], rotateboxes[:,2], rotateboxes[:,3], rotateboxes[:,4]
    p_rotate=[]
    for i in range(rotateboxes.shape[0]):
        RotateMatrix=np.array([
                              [np.cos(Angle[i]),-np.sin(Angle[i])],
                              [np.sin(Angle[i]),np.cos(Angle[i])]])
        rhead,r1,r2,r3,r4=np.transpose([0,-h/2]),np.transpose([-w[i]/2,-h[i]/2]),np.transpose([w[i]/2,-h[i]/2]),np.transpose([w[i]/2,h[i]/2]),np.transpose([-w[i]/2,h[i]/2])
        rhead=np.transpose(np.dot(RotateMatrix, rhead))+[cx[i],cy[i]]
        p1=np.transpose(np.dot(RotateMatrix, r1))+[cx[i],cy[i]]
        p2=np.transpose(np.dot(RotateMatrix, r2))+[cx[i],cy[i]]
        p3=np.transpose(np.dot(RotateMatrix, r3))+[cx[i],cy[i]]
        p4=np.transpose(np.dot(RotateMatrix, r4))+[cx[i],cy[i]]
        p_rotate_=np.int32(np.vstack((p1,p2,p3,p4)))
        p_rotate.append(p_rotate_)
    cv2.polylines(src_img,np.array(p_rotate),True,(0,255,255))
    # if name==None:
    #     cv2.imwrite('1.jpg',src_img)
    # else:
    #     cv2.imwrite('{}.jpg'.format(name),src_img)
    cv2.imshow('rotate_box',src_img.astype('uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def py_cpu_label_nms(dets, thresh, max_output_size,LABEL_NAME_MAP,NAME_LONG_MAP,std_scale):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 6]
    label = dets[:, 5]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        if len(keep) >= max_output_size:
            break
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # find the overlap bbox
        inds1 = np.where(ovr > thresh)[0]+1
        # inds1=np.concatenate(inds1,np.expand_dims(i, 0))
        inds1=np.concatenate((order[inds1],np.expand_dims(i, 0)),axis=0)
        #get label index
        label_refine = label[inds1]
        #get accture length
        h_acc=y2[inds1]-y1[inds1]
        #initial long_label and p
        long_label=np.zeros_like(label_refine)
        p=np.zeros_like(label_refine)
        for j in range(label_refine.size):
            
            long_label[j]=NAME_LONG_MAP[LABEL_NAME_MAP[label_refine[j]]]
            p[j]=2* st.norm.cdf(-(np.abs(h_acc[j] - long_label[j]*2) / (std_scale*2*long_label[j])))
            
        label_refine = label[inds1]
        inds_other=np.where(label_refine == 24)[0]
        if  inds_other.size>0 and inds_other.size != long_label.size:
            p[inds_other]=1-np.max(p)
        scores[inds1]=scores[inds1]*p
        
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        if len(keep) >= max_output_size:
            break
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
        
    return np.array(keep, np.int64),scores

def merge_rotatexml(xmldir, merge_xmldir, NAME_LABEL_MAP,NAME_LONG_MAP,
                    std_scale, score_threshold):
    LABEL_NAME_MAP = get_label_name_map(NAME_LABEL_MAP)

    if not os.path.exists(merge_xmldir):
        os.mkdir(merge_xmldir)
    imgs_path=get_file_paths_recursive(xmldir, '.xml')
    name_det={}
    for num,img_path in enumerate(imgs_path,0):
        str_num=img_path.replace('.xml','').strip().split('__')
        img_name=str_num[0]
        scale=np.float(str_num[1])
        ww_=np.int(str_num[2])
        hh_=np.int(str_num[3])
        img_size,gsd,imagesource,gtbox,extra=read_rotate_xml(img_path,NAME_LABEL_MAP)
        gtbox=np.array(gtbox)
        if len(gtbox)>0:
            gtbox[:,6]=extra
            inx=gtbox[:,6]>score_threshold
            rotateboxes=gtbox[inx]
            gtbox[:, 0]=(gtbox[:, 0]+ww_)/scale
            gtbox[:, 1]=(gtbox[:, 1]+hh_)/scale
            gtbox[:, 2]/=scale
            gtbox[:, 3]/=scale
            if not img_name in name_det :
                name_det[img_name]= gtbox
            else:
                name_det[img_name]= np.vstack((name_det[img_name],gtbox))
#        else:
#            print('  ')
# # #将所有检测结果综合起来 #正框mns
    for img_name, rotateboxes in name_det.items():
        if std_scale==0:
            print('no nms')
        else:
            inx, scores = py_cpu_label_nms(
                dets=np.array(rotateboxes, np.float32),
                thresh=nms_threshold,
                max_output_size=500,
                LABEL_NAME_MAP=LABEL_NAME_MAP_USnavy_20,
                NAME_LONG_MAP=NAME_LONG_MAP_USnavy,
                std_scale=std_scale)
            rotateboxes = rotateboxes[inx]
            # result[:, 4]=scores[inx]
            # inx=result[:, 4]>score_threshold_2
            # result=result[inx]
        cvrboxes=[]
        for box in rotateboxes:
            cvrboxes.append(rotate_rect2cv_np(box))
        cvrboxes = np.array(cvrboxes)
        if len(cvrboxes)>0:##斜框NMS
            keep = nms_rotate_cpu(cvrboxes, rotateboxes[:, 6], rnms_threshold,
                                  4000)  #这里可以改
            rotateboxes=rotateboxes[keep]
        # keep=[]#去掉过 尺寸异常的目标
        # for i in range(rotateboxes.shape[0]):
        #     box=rotateboxes[i,:]
        #     actual_long=box[3]*scale
        #     standad_long = NAME_LONG_MAP[LABEL_NAME_MAP[box[5]]]
        #     STD_long = NAME_LONG_MAP[LABEL_NAME_MAP[box[5]]]
        #     STD_long *= std_scale
        #     if np.abs(actual_long-standad_long)/standad_long < STD_long *1.2 and box[2]>16:
        #         keep.append(i)
        #     else:
        #         print('{}  hh:{}  ww:{}'.format(img_name,hh_,ww_))
        #         print('{} th label {} is wrong,long is {} normal long is {} width is {}'.format(i+1,LABEl_NAME_MAP[box[5]],actual_long,standad_long,box[3]))
        # rotateboxes=rotateboxes[keep]
        #保存检测结果 为比赛格式
        # image_dir,image_name=os.path.split(img_path)
        # gt_box=rotateboxes[:,0:5]
        # label=rotateboxes[:,6]
        write_rotate_xml(merge_xmldir,'{}.jpg'.format(img_name),[1024 ,1024,3],0.5,'USnavy',rotateboxes,LABEL_NAME_MAP,rotateboxes[:,6])#size,gsd,imagesource

if __name__ == '__main__':
    #Store annotations and train2014/val2014/... in this folder
    xmldir ='/home/zf/Dataset/DOTA_800_aug/val_det_xml1024'
    #the path you want to save your results for coco to voc
    merge_xmldir = '/home/zf/Dataset/DOTA_800_aug/val_det_merge_xml1024'
    nms_threshold=0.9#0.8
    rnms_threshold=0.15#0.1
    score_threshold=0.02
    std_scale=0
    IOU_thresh=0.5
    NAME_LABEL_MAP =  NAME_LABEL_MAP_DOTA10
    NAME_LONG_MAP = NAME_LABEL_MAP_DOTA10
    merge_rotatexml(xmldir, merge_xmldir, NAME_LABEL_MAP, NAME_LONG_MAP, std_scale,  score_threshold)

    txt_dir_h = '/home/zf/Dataset/DOTA_800_aug/val_det_merge_xmltxt'
    annopath = '/media/zf/F/Dataset_ori/Dota/val/labelTxt/' 
    file_ext = '.xml'
    flag = 'test'
    eval_rotatexml(merge_xmldir, txt_dir_h, annopath, NAME_LABEL_MAP, file_ext,  flag, IOU_thresh)