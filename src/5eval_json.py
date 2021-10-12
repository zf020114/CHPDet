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
    # if name==None:
    #     cv2.imwrite('1.jpg',src_img)
    # else:
    #     cv2.imwrite('{}.jpg'.format(name),src_img)
    cv2.imshow('rotate_box',src_img.astype('uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def result2rotatebox_cvbox(src_img,result):
    cx, cy, w,h=(result[:,0]+result[:,2])/2 ,(result[:,1]+result[:,3])/2 ,(result[:,2]-result[:,0]) ,(result[:,3]-result[:,1])
    head=result[:,5:7]#头部坐标
    center=np.vstack((cx,cy)).T#中心坐标
    det=head-center
    Angle=np.zeros_like(cx)
    for i in range(det.shape[0]):
        if det[i,0]==0:
            if det[i,1]>0:
                Angle[i]=np.pi/2
            else:
                Angle[i]=-np.pi/2
        elif det[i,0]<0:
            Angle[i]=np.arctan(det[i,1]/det[i,0])+np.pi*3/2
        else:
            Angle[i]=np.arctan(det[i,1]/det[i,0])+np.pi/2
    rotateboxes=np.vstack((cx, cy, w,h,Angle,result[:,7],result[:,4])).T
    center_head=np.sqrt(det[:,0]*det[:,0]+det[:,1]*det[:,1])
    # ratio_h_head=np.array(([2*center_head/h, h/center_head/2])).min(axis=0)
    # rotateboxes[:,6]*=ratio_h_head*ratio_h_head
    #这里可以将超过范围的点去掉
    # keep=center_head>h/3
    # rotateboxes=rotateboxes[keep]
    # show_rotate_box(src_img,rotateboxes)
    #   在opencv中，坐标系原点在左上角，相对于x轴，逆时针旋转角度为负，顺时针旋转角度为正。所以，θ∈（-90度，0]。
    #rotatexml  start normal y  shunp_cv
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
        cvboxes[i,:]=[rotateboxes[i,0],rotateboxes[i,1],cv_w,cv_h,angle_cv,result[i,7],result[i,4]]
    return cvboxes,rotateboxes

def showimg_kepoint(coco,dataset,img,classes,cls_id,show=True):
    global dataDir
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    anns = coco.loadAnns(annIds)
    objs = []
    result_=np.zeros((0,8))
    for ann in anns:
        class_name=classes[ann['category_id']]
        bbox=ann['bbox']
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[2] + bbox[0])
        ymax = int(bbox[3] + bbox[1])
        obj = [class_name, xmin, ymin, xmax, ymax]
        objs.append(obj)
        keypoints=ann['keypoints']
        keypoint=keypoints[0:2]
        score=ann['score']
        if score>score_threshold:
            res_array=np.array([xmin,ymin,xmax,ymax,score,keypoint[0],keypoint[1],NAME_LABEL_MAP[class_name]]).T
            result_=np.vstack((result_,res_array))
    return result_


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
    coco_res=coco.loadRes(annFile)
    classes = id2name(coco)
    print(classes)
    classes_ids = coco.getCatIds(catNms=classes_names)
    print(classes_ids)
    #Get ID number of this class
    img_ids=coco.getImgIds()
    print('len of dataset:{}'.format(len(img_ids)))
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
        result_=showimg_kepoint(coco_res, 'val2017', img, classes,classes_ids,show=True)
        inx=result_[:, 4]>score_threshold
        result_=result_[inx]
   
        cvboxes,rotateboxes=result2rotatebox_cvbox(img,result_)
        # img=cv2.imread(Img_fullname)
        # show_rotate_box(img,rotateboxes)
        result_[:, 0]=(result_[:, 0]+ww_)/scale
        result_[:, 2]=(result_[:, 2]+ww_)/scale
        result_[:, 5]=(result_[:, 5]+ww_)/scale
        result_[:, 1]=(result_[:, 1]+hh_)/scale
        result_[:, 3]=(result_[:, 3]+hh_)/scale
        result_[:, 6]=(result_[:, 6]+hh_)/scale
        if not img_name in name_det :
            name_det[img_name]= result_
        else:
            name_det[img_name]= np.vstack((name_det[img_name],result_))
    # # #将所有检测结果综合起来 #正框mns
    for img_name, result in name_det.items():
        if std_scale==0:
            inx = py_cpu_nms(dets=np.array(result, np.float32),thresh=nms_threshold,max_output_size=500)
            result=result[inx]
        else:
            inx ,scores=  py_cpu_label_nms(dets=np.array(result, np.float32),thresh=nms_threshold,max_output_size=500,
                                LABEL_NAME_MAP=LABEL_NAME_MAP,
                                NAME_LONG_MAP=NAME_LONG_MAP,std_scale=std_scale)
            result=result[inx]
            result[:, 4]=scores[inx]
     
        cvrboxes,rotateboxes=result2rotatebox_cvbox(img,result)
        #斜框NMS
        if cvrboxes.size>0:
            keep = nms_rotate_cpu(cvrboxes,cvrboxes[:,6],rnms_threshold, 200)   #这里可以改
            rotateboxes=rotateboxes[keep]
        # #去掉过小的目标
        # keep=[]
        # for i in range(rotateboxes.shape[0]):
        #     box=rotateboxes[i,:]
        #     actual_long=box[3]*scale
        #     standad_long=NAME_LONG_MAP[LABEl_NAME_MAP[box[5]]]
        #     STD_long=NAME_STD_MAP[LABEl_NAME_MAP[box[5]]]
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
        write_rotate_xml(merge_xmldir, '{}.jpg'.format(img_name),
                         [1024, 1024, 3], 0.5, 'USnavy', rotateboxes,
                         LABEl_NAME_MAP, rotateboxes[:,
                                                     6])  #size,gsd,imagesource

if __name__ == '__main__':
    #Store annotations and train2014/val2014/... in this folder
    # annFile = '/media/zf/F/Dataset/USnavy_test_gt/json文件/resultsDLA_512_1_10.json' #0.9 0.15 0.05 map82.30 加上去掉过小的目标后是82。55 加上1.3约束82。80
    # annFile = '/media/zf/E/CenterNet/exp/multi_pose/dla_USnavy512_RGuass1_12raduis_arf/results.json'#map 81.34
    # annFile = '/media/zf/E/CenterNet/exp/multi_pose/dla_USnavy512_RGuass_scale1/results.json' #0.8 0.1 0.1  map 80.88
    # annFile = '/media/zf/F/Dataset/USnavy_test_gt/json文件/dla_1x.json'#map80.39 80.52
    # annFile = '/media/zf/F/Dataset/USnavy_test_gt/json文件/dla_USnavy512_RGuass1_12raduis.json'#map 80.59
    # annFile = '/media/zf/F/Dataset/USnavy_test_gt/json文件/dla_USnavy512_RGuass1_12raduis_arf.json'#map #map81.37 82.10
    # annFile = '/media/zf/E/CenterNet/exp/multi_pose/dla_USnavy512_DLA1x_FL/results.json'#map #map81.36
    # annFile = '/media/zf/F/Dataset/USnavy_test_gt/json文件/dla_USnavy512_RGuass_scale1.json'#map 80.86
    # annFile = '/media/zf/F/Dataset/USnavy_test_gt/json文件/results_Hourglass_512.json'#map 82.46 82.42
    # annFile = "/media/zf/F/Dataset/USnavy_test_gt/json文件/CHP_DLA_34_ARF_RGUSS_roatate.json"# 0.9  0.15  0.03   0     83.09
    
    # annFile = "/media/zf/E/Dataset/USnavy_test_gt/json文件/dla_USnavy512_guass.json" #map: 0.8661307458688858
    # annFile = "/media/zf/E/Dataset/USnavy_test_gt/json文件/FGSD_DLA34_ARF_Rguass_20.json"#map: 0.861307352764251
    # annFile = "/media/zf/F/Dataset/USnavy_test_gt/json文件/dla_USnavy512_RGuass1_12raduis_addconv_20.json"#map: 0.8307403617505426 map: 0.8357533770407948
    # annFile = "/media/zf/E/Dataset/USnavy_test_gt/json文件/dla_1x_USnavy.json"#map: 0.8838930423970701
    # annFile = "/media/zf/Z/0/hg_3x_Usnavy_512/results.json"#这是20类的HG104  map: 0.8508702824464331
        # annFile ="/media/zf/E/CenterNet/exp/multi_pose/hg_3x_Usnavy_512/results.json"#map: 0.8508702824464331
    # annFile = "/media/zf/E/CenterNet/exp/multi_pose/dla_USnavy512_RGuass1_12raduis_arf_20_FL/results.json" #map: 0.8437783437896875
    # annFile="/media/zf/F/Dataset/USnavy_test_gt/json文件/dla_USnavy512_addconv_20.json"#map: 0.8265993090632107
    # annFile = "/media/zf/F/Dataset/USnavy_test_gt/json文件/dla_USnavy512_20.json"#map: 0.8296416651842302
    annFile = "/media/zf/Z/0/hg_arf_3x_Usnavy_512.json"
    gtFile='/media/zf/E/Dataset/US_Navy_train_square/annotations/person_keypoints_val2017.json'
    #the path you want to save your results for coco to voc
    merge_xmldir = '/media/zf/E/Dataset/US_Navy_train_square/FGSDxml'
    txt_dir_h = '/media/zf/E/Dataset/US_Navy_train_square/eval/0/'
    annopath = '/media/zf/E/Dataset/US_Navy_train_square/eval/GT20_TxT/'  #GT_angle0_TxT#GT20_TxT
    nms_threshold=0.9#0.8
    rnms_threshold=0.15#0.1
    score_threshold=0.05
    std_scale=0
    NAME_LABEL_MAP ={
        '航母': 1,
        '黄蜂级': 2,
        '塔瓦拉级': 3,
        '奥斯汀级': 4,
        '惠特贝岛级': 5,
        '圣安东尼奥级': 6,
        '新港级': 7,
        '提康德罗加级': 8,
        '阿利·伯克级': 9,
        '佩里级': 10,
        '刘易斯和克拉克级': 11,
        '供应级': 12,
        '凯泽级': 13,
        '霍普级': 14,
        '仁慈级': 15,
        '自由级': 16,
        '独立级': 17,
        '复仇者级': 18,
        '潜艇':19,
        '其他':20
        }
    NAME_LONG_MAP = {
        '航母': 330,
        '黄蜂级': 253,
        '塔瓦拉级': 253,
        '奥斯汀级': 173,
        '惠特贝岛级': 185,
        '圣安东尼奥级': 208,
        '新港级': 168,
        '提康德罗加级': 172,
        '阿利·伯克级': 154,
        '佩里级': 135,
        '刘易斯和克拉克级': 210,
        '供应级': 229,
        '凯泽级': 206,
        '霍普级': 290,
        '仁慈级': 272,
        '自由级': 120,
        '独立级': 127,
        '复仇者级': 68,
        '潜艇':140,
        '其他':200
        }
    normal=0.10
    NAME_STD_MAP = {
            '航母': 0.1,
            '黄蜂级': normal,
            '塔瓦拉级': normal,
            '奥斯汀级': normal,
            '惠特贝岛级': normal,
            '圣安东尼奥级': normal,
            '新港级': normal,
            '提康德罗加级': normal,
            '阿利·伯克级': normal,
            '佩里级': normal+0.01,
            '刘易斯和克拉克级': normal,
            '供应级': normal-0.01,
            '凯泽级': normal-0.01,
            '霍普级': normal,
            '仁慈级': normal,
            '自由级': normal*2,
            '独立级': normal*2,
            '复仇者级': normal*2,
            '潜艇':0.75,
            '其他':1
            }
    def get_label_name_map(NAME_LABEL_MAP):
        reverse_dict = {}
        for name, label in NAME_LABEL_MAP.items():
            reverse_dict[label] = name
        return reverse_dict
    LABEL_NAME_MAP=get_label_name_map(NAME_LABEL_MAP)
    merge_json2rotatexml(gtFile,annFile, merge_xmldir, NAME_LABEL_MAP, NAME_LONG_MAP, NAME_STD_MAP, std_scale,  score_threshold)
    eval_rotatexml(merge_xmldir,
                   txt_dir_h,
                   annopath,
                   NAME_LABEL_MAP=NAME_LABEL_MAP,
                   file_ext='.xml',
                   flag='test',
                   ovthresh=0.5)
