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
import sys
sys.path.append('/home/zf/0tools')
from DataFunction import get_file_paths_recursive,read_rotate_xml,rotate_rect2cv,read_VOC_xml
import shutil
sys.path.append('/media/zf/E/Dataset/DOTA_devkit')
import os.path as osp


def get_label_name_map(NAME_LABEL_MAP):
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict

def generate_file_list(img_dir,output_txt,file_ext='.txt'):
    #读取原图路径
    # img_dir=os.path.split(img_dir)[0]
    imgs_path = get_file_paths_recursive(img_dir, file_ext)
    f = open(output_txt, "w",encoding='utf-8')
    for num,img_path in enumerate(imgs_path,0):
        obj='{}\n'.format(os.path.splitext(os.path.split(img_path)[1])[0])
        f.write(obj)
    f.close()
    print('Generate {} down!'.format(output_txt))
    
def rotated_box_to_poly_np(rrects):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    polys = []
    for rrect in rrects:
        x_ctr, y_ctr, width, height, angle = rrect[:5]
        tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
        rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        poly = R.dot(rect)
        x0, x1, x2, x3 = poly[0, :4] + x_ctr
        y0, y1, y2, y3 = poly[1, :4] + y_ctr
        poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
        polys.append(poly)
    # polys = np.array(polys)
    # polys = get_best_begin_point(polys)
    return polys


NAME_LABEL_MAP = {
    'Roundabout': 1,
    'Intersection': 2,
    'Bridge': 3,
    'Tennis Court': 4,
    'Basketball Court': 5,
    'Football Field': 6,
    'Baseball Field': 7,
    'Liquid Cargo Ship': 8,
    'Passenger Ship': 9,
    'Dry Cargo Ship': 10,
    'Motorboat': 11,
    'Fishing Boat': 12,
    'Engineering Ship': 13,
    'Warship': 14,
    'Tugboat': 15,
    'other-ship': 16,
    'Cargo Truck': 17,
    'Small Car': 18,
    'Dump Truck': 19,
    'Tractor': 20,
    'Bus': 21,
    'Trailer': 22,
    'Truck Tractor': 23,
    'Van': 24,
    'Excavator': 25,
    'other-vehicle': 26,
    'Boeing787': 27,
    'Boeing777': 28,
    'A350': 29,
    'A330': 30,
    'Boeing747': 31,
    'A321': 32,
    'ARJ21': 33,
    'Boeing737': 34,
    'A220': 35,
    'C919': 36,
    'other-airplane': 37
}
NAME_LABEL_MAP_eval = {
    'Roundabout': 1,
    'Intersection': 2,
    'Bridge': 3,
    'Tennis-Court': 4,
    'Basketball-Court': 5,
    'Football-Field': 6,
    'Baseball-Field': 7,
    'Liquid-Cargo-Ship': 8,
    'Passenger-Ship': 9,
    'Dry-Cargo-Ship': 10,
    'Motorboat': 11,
    'Fishing-Boat': 12,
    'Engineering-Ship': 13,
    'Warship': 14,
    'Tugboat': 15,
    'other-ship': 16,
    'Cargo-Truck': 17,
    'Small-Car': 18,
    'Dump-Truck': 19,
    'Tractor': 20,
    'Bus': 21,
    'Trailer': 22,
    'Truck-Tractor': 23,
    'Van': 24,
    'Excavator': 25,
    'other-vehicle': 26,
    'Boeing787': 27,
    'Boeing777': 28,
    'A350': 29,
    'A330': 30,
    'Boeing747': 31,
    'A321': 32,
    'ARJ21': 33,
    'Boeing737': 34,
    'A220': 35,
    'C919': 36,
    'other-airplane': 37
    }

GT_xml_path='/media/zf/U/2021ZKXT_aug/annotations/valrotatexml'
txt_dir_h='/media/zf/U/2021ZKXT_aug/annotations/GT'

imagesetfile=osp.join(osp.dirname(GT_xml_path), 'gt_list.txt')
generate_file_list(GT_xml_path,imagesetfile,file_ext='.xml')
# LABEl_NAME_MAP = get_label_name_map(NAME_LABEL_MAP)
LABEl_NAME_MAP_eval=get_label_name_map(NAME_LABEL_MAP_eval)
file_paths = get_file_paths_recursive(GT_xml_path, '.xml') 
if  os.path.isdir(txt_dir_h):
    shutil.rmtree(txt_dir_h,True)
os.makedirs(txt_dir_h)

    ### GT
for count, xml_path in enumerate(file_paths):   
    img_size,gsd,imagesource,gtbox_label,extra=read_rotate_xml(xml_path,NAME_LABEL_MAP)
      # eval txt
    CLASS_DOTA = NAME_LABEL_MAP.keys()
    # Task1 #
    write_handle_h = open(os.path.join(txt_dir_h, '{}.txt'.format(os.path.splitext(os.path.split(xml_path)[1])[0])), 'w')#Task1_gt_
    # gtbox_label=np.array(gtbox_label)
    
    ploys=rotated_box_to_poly_np(gtbox_label)
    
    for i, rect_box in enumerate(ploys):
        # rbox[4]=0
        # rbox_cv=rotate_rect2cv(rbox)
        # rect_box = cv2.boxPoints(rbox_cv)
        # xmin,ymin,xmax,ymax=np.min(rect_box[:,0]),np.min(rect_box[:,1]),np.max(rect_box[:,0]),np.max(rect_box[:,1])

        # command = '%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %s 0\n' % (
        #                                               xmin, ymin, xmax, ymin,
        #                                               xmax, ymax, xmin, ymax,
        #                                               LABEl_NAME_MAP[rbox[5]]
        #                                               )
        # command = '%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %s 0\n' % (
        #                                       rect_box[0][0], rect_box[0][1], rect_box[1][0], rect_box[1][1],
        #                                       rect_box[2][0], rect_box[2][1], rect_box[3][0], rect_box[3][1],
        #                                       LABEl_NAME_MAP[rbox[5]]
        #                                       )
        command = '%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %s 0\n' % (
                                              rect_box[0], rect_box[1], rect_box[2], rect_box[3],
                                              rect_box[4], rect_box[5], rect_box[6], rect_box[7],
                                              LABEl_NAME_MAP_eval[gtbox_label[i][5]]
                                              )
        write_handle_h.write(command)
    write_handle_h.close()


