#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:39:49 2020

@author: zf
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os,sys
import cv2
import numpy as np
from DataFunction import get_file_paths_recursive,read_rotate_xml,rotate_rect2cv
import shutil
from All_Class_NAME_LABEL import NAME_LABEL_MAP_HRSC,NAME_LABEL_MAP_USnavy_20#,NAME_LABEL_MAP_ICDAR
from DOTA_devkit.dota15 import data_v15_evl_1
# GT_xml_path='/media/zf/E/Dataset/USnavy_test_gt/'#'E:/Standard_data_set/HRSC2016/Test/Annotations/'
# txt_dir_h='/media/zf/E/Dataset/US_Navy_train_square/eval/GT_angle0_TxT'

GT_xml_path='/media/zf/E/Dataset/US_Navy_train_square/eval/Testxml_dla512/'
txt_dir_h='/media/zf/E/Dataset/US_Navy_train_square/eval/Testxml_dla512_20txt/'

# GT_xml_path='/media/zf/E/Dataset_txt/ICDAR15_aug1024/eval/Testxml_hg1024/'
# txt_dir_h='/media/zf/E/Dataset_txt/ICDAR15_aug1024/eval/Testxml_hg1024_txt/'


detpath = txt_dir_h#'/media/zf/E/Dataset/US_Navy_train_square/eval/Testxml_dla512_20txt/'
annopath ='/media/zf/E/Dataset/US_Navy_train_square/eval/GT20_TxT/'#GT_angle0_TxT#GT20_TxT
imagesetfile = '/media/zf/E/Dataset/US_Navy_train_square/eval/testlist.txt'

NAME_LABEL_MAP =  NAME_LABEL_MAP_USnavy_20 #NAME_LABEL_MAP_ICDAR
def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict
LABEl_NAME_MAP = get_label_name_map()

file_paths = get_file_paths_recursive(GT_xml_path, '.xml') 
if  os.path.isdir(txt_dir_h):
    shutil.rmtree(txt_dir_h,True)
os.makedirs(txt_dir_h)



for count, xml_path in enumerate(file_paths):   
    img_size,gsd,imagesource,gtbox_label,extra=read_rotate_xml(xml_path,NAME_LABEL_MAP)
      # eval txt
    CLASS_DOTA = NAME_LABEL_MAP.keys()

    # Task2 #
    write_handle_h = {}
    for sub_class in CLASS_DOTA:
        if sub_class == 'back_ground':
            continue
        write_handle_h[sub_class] = open(os.path.join(txt_dir_h, 'Task1_%s.txt' % sub_class), 'a+')

    for i, rbox in enumerate(gtbox_label):
        # rbox[4]=0
        rbox_cv=rotate_rect2cv(rbox)
        rect_box = cv2.boxPoints(rbox_cv)
        #xml_path.split('/')[-1].split('.')[0]
        command = '%s %.3f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n' % (os.path.splitext(os.path.split(xml_path)[1])[0],
                                                      np.float(extra[i]),
                                                      rect_box[0][0], rect_box[0][1], rect_box[1][0], rect_box[1][1],
                                                      rect_box[2][0], rect_box[2][1], rect_box[3][0], rect_box[3][1])
        write_handle_h[LABEl_NAME_MAP[rbox[5]]].write(command)

    for sub_class in CLASS_DOTA:
        if sub_class == 'back_ground':
            continue
        write_handle_h[sub_class].close()
data_v15_evl_1(detpath,annopath,imagesetfile)

#     ### GT
# for count, xml_path in enumerate(file_paths):   
#     img_size,gsd,imagesource,gtbox_label,extra=read_rotate_xml(xml_path,NAME_LABEL_MAP)
#       # eval txt
#     CLASS_DOTA = NAME_LABEL_MAP.keys()

#     # Task1 #
#     write_handle_h = open(os.path.join(txt_dir_h, 'Task1_gt_{}.txt'.format(os.path.splitext(os.path.split(xml_path)[1])[0])), 'w')#Task1_gt_

#     for i, rbox in enumerate(gtbox_label):
#         rbox[4]=0
#         rbox_cv=rotate_rect2cv(rbox)
#         rect_box = cv2.boxPoints(rbox_cv)
        
#         command = '%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %s 0\n' % (
#                                                       rect_box[0][0], rect_box[0][1], rect_box[1][0], rect_box[1][1],
#                                                       rect_box[2][0], rect_box[2][1], rect_box[3][0], rect_box[3][1],
#                                                       LABEl_NAME_MAP[rbox[5]]
#                                                       )
#         write_handle_h.write(command)
#     write_handle_h.close()



    # #GT ICDAR Tiou
# for count, xml_path in enumerate(file_paths):   
#     img_size,gsd,imagesource,gtbox_label,extra=read_rotate_xml(xml_path,NAME_LABEL_MAP)
#       # eval txt
#     CLASS_DOTA = NAME_LABEL_MAP.keys()

#     # Task1 #
#     write_handle_h = open(os.path.join(txt_dir_h, '{}.txt'.format(os.path.splitext(os.path.split(xml_path)[1])[0])), 'w')#Task1_gt_

#     for i, rbox in enumerate(gtbox_label):
#         rbox_cv=rotate_rect2cv(rbox)
#         rect_box = cv2.boxPoints(rbox_cv)
        
#         command = '%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%s\n' % (
#                                                       rect_box[0][0], rect_box[0][1], rect_box[1][0], rect_box[1][1],
#                                                       rect_box[2][0], rect_box[2][1], rect_box[3][0], rect_box[3][1],
#                                                       LABEl_NAME_MAP[rbox[5]]
#                                                       )
        
#         write_handle_h.write(command)
#     write_handle_h.close()
    
    
# #         # #GT det Tiou
# for count, xml_path in enumerate(file_paths):   
#     img_size,gsd,imagesource,gtbox_label,extra=read_rotate_xml(xml_path,NAME_LABEL_MAP)
#       # eval txt
#     CLASS_DOTA = NAME_LABEL_MAP.keys()

#     # Task1 #
#     write_handle_h = open(os.path.join(txt_dir_h, '{}.txt'.format(os.path.splitext(os.path.split(xml_path)[1])[0])), 'w')#Task1_gt_

#     for i, rbox in enumerate(gtbox_label):
#         rbox_cv=rotate_rect2cv(rbox)
#         rect_box = cv2.boxPoints(rbox_cv)
        
#         command = '{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.2f},{}\n'.format(
#                                                       rect_box[0][0], rect_box[0][1], rect_box[1][0], rect_box[1][1],
#                                                       rect_box[2][0], rect_box[2][1], rect_box[3][0], rect_box[3][1],
#                                                       float(extra[i]),LABEl_NAME_MAP[rbox[5]]
#                                                       )
#         # command = '%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.2f,%s\n' % (
#         #                                               rect_box[0][0], rect_box[0][1], rect_box[1][0], rect_box[1][1],
#         #                                               rect_box[2][0], rect_box[2][1], rect_box[3][0], rect_box[3][1],
#         #                                               float(extra[i]),LABEl_NAME_MAP[rbox[5]]
#         #                                               )
        
#         write_handle_h.write(command)
#     write_handle_h.close()