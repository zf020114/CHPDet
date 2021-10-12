#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 23:00:51 2020

@author: zf
"""
# import pycocotools.coco as coco
from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import numpy as np
import sys
sys.path.append('/home/zf/0tools')
from py_cpu_nms import py_cpu_nms, py_cpu_label_nms
from r_nms_cpu import nms_rotate_cpu
from DataFunction import write_rotate_xml
from All_Class_NAME_LABEL import NAME_LABEL_MAP_USnavy_20,NAME_LONG_MAP_USnavy,NAME_STD_MAP_USnavy,LABEL_NAME_MAP_USnavy_20
from Rotatexml2DotaTxT import eval_rotatexml


def get_label_name_map(NAME_LABEL_MAP):
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict

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
    cv2.imshow('rotate_box',src_img.astype('uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def result2rotatebox_cvbox(coco,dataset,img,classes,cls_id,show=True):
 
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    anns = coco.loadAnns(annIds)
    result_=np.zeros((0,7))
    for ann in anns:
        class_name=classes[ann['category_id']]
        bbox=ann['bbox']
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[2] + bbox[0])
        ymax = int(bbox[3] + bbox[1])
        ang=bbox[4]
        score=ann['score']
        if score>score_threshold:
            res_array=np.array([xmin,ymin,xmax,ymax,ang,score,NAME_LABEL_MAP[class_name]]).T
            result_=np.vstack((result_,res_array))
    result = result_   
    cx, cy, w,h=(result[:,0]+result[:,2])/2 ,(result[:,1]+result[:,3])/2 ,(result[:,2]-result[:,0]) ,(result[:,3]-result[:,1])
    ang=result[:,4]/180*np.pi
    center=np.vstack((cx,cy)).T#中心坐标
    rotateboxes=np.vstack((cx, cy, w,h,ang,result[:,6],result[:,5])).T
    p_cv=[]
    cvboxes=np.zeros_like(rotateboxes)
    for i in range(rotateboxes.shape[0]):
        angle_cv=rotateboxes[i,4]/np.pi*180
        if angle_cv>0:
            angle_cv=angle_cv
            cv_h=rotateboxes[i,3]
            cv_w=rotateboxes[i,2]
        else:
            angle_cv=angle_cv+90
            cv_h=rotateboxes[i,2]
            cv_w=rotateboxes[i,3]
        cvboxes[i,:]=[rotateboxes[i,0],rotateboxes[i,1],cv_w,cv_h,angle_cv,result[i,6],result[i,5]]
    return cvboxes,rotateboxes


def merge_json2rotatexml(gtFile, annFile, merge_xmldir, NAME_LABEL_MAP,
                         NAME_LONG_MAP, NAME_STD_MAP, std_scale,
                         score_threshold):
    classes_names = []
    for name, label in NAME_LABEL_MAP.items():
        classes_names.append(name)
    if not os.path.exists(merge_xmldir):
        os.mkdir(merge_xmldir)
    LABEl_NAME_MAP = get_label_name_map(NAME_LABEL_MAP)
    #COCO API for initializing annotated data
    coco = COCO(gtFile)
    # coco_o=COCO(gtFile)
    coco_res=coco.loadRes(annFile)
    #show all classes in coco
    classes = id2name(coco)
    print(classes)
    classes_ids = coco.getCatIds(catNms=classes_names)
    print(classes_ids)
    #Get ID number of this class
    img_ids=coco.getImgIds()
    print('len of dataset:{}'.format(len(img_ids)))
    # imgIds=img_ids[0:10]
    img_name_ori=''
    result=np.zeros((0,8))
    name_det={}
    for imgId in tqdm(img_ids):
        img = coco_res.loadImgs(imgId)[0]
        # Img_fullname='%s/%s/%s'%(dataDir,dataset,img['file_name'])
        filename = img['file_name']
        str_num=filename.replace('.jpg','').strip().split('__')
        img_name=str_num[0]
        scale=np.float(str_num[1])
        ww_=np.int(str_num[2])
        hh_=np.int(str_num[3])
        cvboxes,rotateboxes=result2rotatebox_cvbox(coco_res, 'val2017', img, classes,classes_ids,show=True)
        xml_dir='/home/zf/CenterNet/exp/ctdet/CenR_usnavy512_dla_2x/rotatexml'
        write_rotate_xml(xml_dir, img["file_name"],
                         [1024, 1024, 3], 0.5, 'USnavy', rotateboxes,
                         LABEl_NAME_MAP, rotateboxes[:,6])  #size,gsd,imagesource
        # img_dir='/home/zf/CenterNet/data/coco/val2017'
        # src_img=cv2.imread(os.path.join(img_dir,img["file_name"]))
        # show_rotate_box(src_img,rotateboxes)
    #     result_[:, 0:4]=(result_[:, 0:4]+ww_)/scale
    #     if not img_name in name_det :
    #         name_det[img_name]= result_
    #     else:
    #         name_det[img_name]= np.vstack((name_det[img_name],result_))
    # # # # #将所有检测结果综合起来 #正框mns
    # for img_name, result in name_det.items():
    #     # if std_scale==0:
    #     #     inx = py_cpu_nms(dets=np.array(result, np.float32),thresh=nms_threshold,max_output_size=500)
    #     #     result=result[inx]
    #     # else:
    #     #     inx ,scores=  py_cpu_label_nms(dets=np.array(result, np.float32),thresh=nms_threshold,max_output_size=500,
    #     #                         LABEL_NAME_MAP=LABEL_NAME_MAP_USnavy_20,
    #     #                         NAME_LONG_MAP=NAME_LONG_MAP_USnavy,std_scale=std_scale)
    #     #     result=result[inx]
    #     #     result[:, 4]=scores[inx]

    #     cvrboxes,rotateboxes=result2rotatebox_cvbox(img,result)
    #     #斜框NMS
    #     if cvrboxes.size>0:
    #         keep = nms_rotate_cpu(cvrboxes,cvrboxes[:,6],rnms_threshold, 200)   #这里可以改
    #         rotateboxes=rotateboxes[keep]
    #     # #去掉过小的目标
    #     # keep=[]
    #     # for i in range(rotateboxes.shape[0]):
    #     #     box=rotateboxes[i,:]
    #     #     actual_long=box[3]*scale
    #     #     standad_long=NAME_LONG_MAP[LABEl_NAME_MAP[box[5]]]
    #     #     STD_long=NAME_STD_MAP[LABEl_NAME_MAP[box[5]]]
    #     #     if np.abs(actual_long-standad_long)/standad_long < STD_long *1.2 and box[2]>16:
    #     #         keep.append(i)
    #     #     else:
    #     #         print('{}  hh:{}  ww:{}'.format(img_name,hh_,ww_))
    #     #         print('{} th label {} is wrong,long is {} normal long is {} width is {}'.format(i+1,LABEl_NAME_MAP[box[5]],actual_long,standad_long,box[3]))
    #     # rotateboxes=rotateboxes[keep]
    #     #保存检测结果 为比赛格式
    #     # image_dir,image_name=os.path.split(img_path)
    #     # gt_box=rotateboxes[:,0:5]
    #     # label=rotateboxes[:,6]
    #     write_rotate_xml(merge_xmldir, '{}.jpg'.format(img_name),
    #                      [1024, 1024, 3], 0.5, 'USnavy', rotateboxes,
    #                      LABEl_NAME_MAP, rotateboxes[:,
                                                    #  6])  #size,gsd,imagesource

if __name__ == '__main__':
    #Store annotations and train2014/val2014/... in this folder
    annFile = '/home/zf/CenterNet/exp/ctdet/CenR_usnavy512_dla_2x/results.json'
    gtFile='/home/zf/Dataset/US_Navy_train_square/annotations/person_keypoints_val2017.json'
    #the path you want to save your results for coco to voc
    merge_xmldir = '/home/zf/Dataset/USnavy_test_gt/CenR_DLA34'
    txt_dir_h = '/media/zf/E/Dataset/US_Navy_train_square/eval/0/'
    annopath = '/media/zf/E/Dataset/US_Navy_train_square/eval/GT20_TxT/'  #GT_angle0_TxT#GT20_TxT
    nms_threshold=0.9#0.8
    rnms_threshold=0.15#0.1
    score_threshold=0.1
    std_scale=0.4
    score_threshold_2=0.05
    NAME_LABEL_MAP =  NAME_LABEL_MAP_USnavy_20
    NAME_LONG_MAP = NAME_LONG_MAP_USnavy
    NAME_STD_MAP = NAME_STD_MAP_USnavy
    merge_json2rotatexml(gtFile,annFile, merge_xmldir, NAME_LABEL_MAP, NAME_LONG_MAP, NAME_STD_MAP, std_scale,  score_threshold)
    eval_rotatexml(merge_xmldir,
                   txt_dir_h,
                   annopath,
                   NAME_LABEL_MAP=NAME_LABEL_MAP,
                   file_ext='.xml',
                   flag='test')
