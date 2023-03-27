# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 18:22:36 2019
版本10.31
@author: admin
"""
import os
import cv2
import numpy as np
import DataFunction
from timeit import default_timer as timer
from tqdm import tqdm
from All_Class_NAME_LABEL import NAME_LABEL_MAP_USnavy,NAME_LABEL_MAP_HRSC,NAME_LABEL_MAP_DOTA15
#voc路径
#data_dir='/media/zf/E/Dataset/USnavy_test_gt'
#img_dir = data_dir#os.path.join(data_dir,'train2017')
#xml_dir = data_dir#os.path.join(data_dir,'rotatexml')
#outputfolder='/media/zf/E/Dataset/US_Navy_train_square'#os.path.join(data_dir,'crop_dir')#r'E:\Dataset\US_Navy_test_aug'
#voc路径
img_dir =  '/media/zf/F/Dataset_ori/Dota/val/images'#r'E:\Dataset\HRSC2016FullDataSet\AllImages'
xml_dir = img_dir
outputfolder='/media/zf/E/Dataset/DOTA_800_aug/val2017/'
output_VOC_folder=''#存储新的VOCanno位置
file_ext='.png'

[w_crop,h_crop]=[800,800]  #这是要切割的图像尺寸  第一个是宽，第二个是高
overlap_ratio=1/5
outrange_ratio=1/6
ratios=[1]#设置在切割过程中缩放的倍数

NAME_LABEL_MAP = NAME_LABEL_MAP_DOTA15 

def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict
LABEl_NAME_MAP = get_label_name_map()


#新建输出文件夹
if not os.path.isdir(os.path.join(outputfolder,'val2017')):
    os.makedirs(os.path.join(outputfolder,'val2017'))
if not os.path.isdir(os.path.join(outputfolder,'rotatexml_val')):
    os.makedirs(os.path.join(outputfolder,'rotatexml_val'))

#读取原图全路径  
imgs_path = DataFunction.get_file_paths_recursive(img_dir, file_ext) 
#旋转角的大小，整数表示逆时针旋转
imgs_total_num=len(imgs_path)
for img_path in tqdm(imgs_path):#enumerate(imgs_path,0):
# for num,img_path in enumerate(imgs_path,0):
    start = timer()
    #一、读取图像并获取基本参数
    # img_path=imgs_path[num]
    img = cv2.imread(img_path)

    img_size=img.shape#高，宽 ，通道
    img_num=1
    #二、读取标注文件
    [floder,name]=os.path.split(img_path)
    xml_path=os.path.join(xml_dir,os.path.splitext(name)[0]+'.xml')
    img_size,gsd,imagesource,gtbox,extra=DataFunction.read_rotate_xml(xml_path,NAME_LABEL_MAP)
    
#    gsd=0.5
    if len(gtbox)>0:
        DataFunction.crop_img_rotatexml_val(ratios,overlap_ratio,outrange_ratio,h_crop,w_crop,img_path,outputfolder,output_VOC_folder,img_num,img_size,img,gsd,imagesource,gtbox,LABEl_NAME_MAP)
    else :
        print('{} annotation is empty!'.format(img_path))
    time_elapsed = timer() - start
    # print('{}/{}，time{}: augnum:{}'.format(num,imgs_total_num,time_elapsed,img_num))