# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 08:04:59 2019
@author: zhangfeng
版本 10.31
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import argparse
import cv2
import numpy as np
import datetime
from lxml import etree
import xml.etree.ElementTree as ET
#from osgeo import gdal
#import gdal
from PIL import Image, ImageEnhance
from skimage import exposure,util
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose,ISONoise,ToGray,JpegCompression,Equalize#FancyPCA
)
# from albumentations.imgaug.transforms import (
#     HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
#     Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
#     IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
#     IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose,JpegCompression,
#     ISONoise,Equalize,FancyPCA,ToGray
# )

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('imagePath', help='the path of test image')
    parser.add_argument('resultPath', help='the path to save result')
    args = parser.parse_args()
    return args

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



#xml txt  读写区
def write_xml(xml_name,boxes,labels,CLASSES):
    #将检测结果表示为中科星图比赛格式的程序
    headstr = """\
    <?xml version="1.0" encoding="utf-8"?>
    <Research Direction="高分软件大赛" ImageName="{}">
      <Department>国防科技大学电子科学学院</Department>
      <Date>{}</Date>
      <PluginName>目标识别</PluginName>
      <PluginClass>检测</PluginClass>
      <Results Coordinate="Pixel">
    """
    objstr = """\
        <Object>{}</Object>
        <Pixel Coordinate="X and Y">
          <Pt index="1" LeftTopX="{:.4f}" LeftTopY="{:.4f}" RightBottomX="" RightBottomY="" />
          <Pt index="2" LeftTopX="{:.4f}" LeftTopY="{:.4f}" RightBottomX="" RightBottomY="" />
          <Pt index="3" LeftTopX="{:.4f}" LeftTopY="{:.4f}" RightBottomX="" RightBottomY="" />
          <Pt index="4" LeftTopX="{:.4f}" LeftTopY="{:.4f}" RightBottomX="" RightBottomY="" />
        </Pixel>
    """
    tailstr = '''\
      </Results>
    </Research>
    '''
    name=os.path.split(xml_name)[-1].replace('xml','tif')
    day2 = datetime.date.today()
    head=headstr.format(name,day2)
    tail = tailstr

    f = open(xml_name, "w",encoding='utf-8')
    f.write(head)
    for i, box in enumerate(boxes):
        obj=objstr.format(CLASSES[labels[i]],box[0][0],box[0][1],box[1][0],box[1][1],box[2][0],box[2][1],box[3][0],box[3][1])
        f.write(obj)
    f.write(tail)
    f.close()

def read_tiff(inpath):
  ds=gdal.Open(inpath)
  row=ds.RasterXSize
  col=ds.RasterYSize
  band=ds.RasterCount
  geoTransform=ds.GetGeoTransform()
  x_gsd,y_gsd=geoTransform[1],geoTransform[5]
  data=np.zeros([row,col,band])
  for i in range(band):
      dt=ds.GetRasterBand(1)
      data[:,:,i]=dt.ReadAsArray(0,0,col,row)
  img_size=[row,col,band]
  return img_size,x_gsd,y_gsd,data


def read_VOC_xml(xml_path,NAME_LABEL_MAP):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 5],
           and has [xmin, ymin, xmax, ymax, label] in a per row
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    for child_of_root in root:
        # if child_of_root.tag == 'filename':
        #     assert child_of_root.text == xml_path.split('/')[-1].split('.')[0] \
        #                                  + FLAGS.img_format, 'xml_name and img_name cannot match'
        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)
        if child_of_root.tag == 'object':
            label = None
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    label_name=child_item.text.replace('\ufeff','')
                    label =NAME_LABEL_MAP[label_name]#float(child_item.text) #训练VOC用NAME_LABEL_MAP[child_item.text]#因为用自己的这边的ID是编号  训练卫星数据用1
#                    if child_item.text=='plane' or child_item.text=='airplane'or child_item.text=='aircraft':
#                        label=0
#                    elif(child_item.text=='helicopter'):
#                        label=1
#                    else:
#                        print('label {} error!)'.format(child_item.text))
                if child_item.tag == 'bndbox':
                    tmp_box = [0, 0, 0, 0]
                    for node in child_item:
                        if node.tag == 'xmin':
                            tmp_box[0] = float(node.text)
                        if node.tag == 'ymin':
                            tmp_box[1] = float(node.text)
                        if node.tag == 'xmax':
                            tmp_box[2] = float(node.text)
                        if node.tag == 'ymax':
                            tmp_box[3] = float(node.text)
                    assert label is not None, 'label is none, error'
                    tmp_box.append(label)
                    box_list.append(tmp_box)
#    gtbox_label = np.array(box_list, dtype=np.int32) 
    return img_height, img_width, box_list

def read_VOC_xml2(xml_path,NAME_LABEL_MAP):
    tree = etree.parse(xml_path)
    # get bbox
    object_num=len(tree.xpath('//object'))
    object_array=[]
    for i in range(object_num):   
        object_element=tree.xpath('//object')[i]# 获取object素的内容
        name=(object_element.getchildren()[3].text)
#        label=NAME_LABEL_MAP[name]
        if name=='plane' or name=='airplane'or name=='aircraft':
            label=0
        elif name=='helicopter':
            label=1
        else:
            print('label {} error!)'.format(name))
        xmin=float(object_element.getchildren()[0].getchildren()[0].text)
        ymin=float(object_element.getchildren()[0].getchildren()[1].text)
        xmax=float(object_element.getchildren()[0].getchildren()[3].text)
        ymax=float(object_element.getchildren()[0].getchildren()[2].text)
        object_array.append([xmin,ymin,xmax,ymax,label])
    return object_array

def write_VOC_xml(output_floder,img_name,size,gsd,imagesource,gtbox_label,CLASSES):
    #将检测结果表示为VOC格式的xml文件
    headstr = """\
   <annotation>
	<folder>{}</folder>
	<filename>{}</filename>
	<path>{}</path>
	<source>
		<database>{}</database>
	</source>
	<size>
		<width>{}</width>
		<height>{}</height>
		<depth>{}</depth>
	</size>
	<segmented>0</segmented>
    """
    objstr = """\
      <object>
		<name>{}</name>
		<pose>0</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{}</xmin>
			<ymin>{}</ymin>
			<xmax>{}</xmax>
			<ymax>{}</ymax>
		</bndbox>
	</object>
    """
    tailstr = '''\
      </annotation>
    '''
    [floder,name]=os.path.split(img_name)
    filename=os.path.join(floder,os.path.splitext(name)[0]+'.xml')
    foldername=os.path.split(img_name)[0]
#    head=headstr.format(name,foldername,size[1],size[0])
    head=headstr.format(gsd,name,foldername,imagesource,size[1],size[0],size[2])
    rotate_xml_name=os.path.join(output_floder,os.path.split(filename)[1])
    f = open(rotate_xml_name, "w",encoding='utf-8')
    f.write(head)
    for i,box in enumerate (gtbox_label):
        obj=objstr.format(CLASSES[int(box[4]-1)],box[0],box[1],box[2],box[3])
#        obj=objstr.format(labelMat[i],difficultMat[i],center[0],center[1],NewWidth,NewHeight,Angle)
        f.write(obj)
    f.write(tailstr)
    f.close()

def read_rotate_xml(xml_path,NAME_LABEL_MAP):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 5],
           and has [xmin, ymin, xmax, ymax, label] in a per row
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    extra=[]
    for child_of_root in root:
        if child_of_root.tag == 'folder':#读取gsd之前把它赋予到了folder字段
            try:
                gsd = float(child_of_root.text)
            except:
                gsd =0
        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)
                if child_item.tag == 'depth':
                    img_depth = int(child_item.text)
        
        if child_of_root.tag == 'source':
            for child_item in child_of_root:
                if child_item.tag == 'database':
                    imagesource=child_item.text
        if child_of_root.tag == 'object':
            label = None
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    
#                    label_name=child_item.text.replace('plane','other').replace('\ufeffB-1B','B-1B').replace('F-31','F-35').replace('L-39','L-159')
                    label_name=child_item.text.replace('\ufeff','').replace('其它','其他').replace('尼米兹级','航母').replace('圣安东尼奥','圣安东尼奥级').replace('圣安东尼奥级级','圣安东尼奥级')#.replace('塔瓦拉级','黄蜂级')
                    
                    label =NAME_LABEL_MAP[label_name]#float(child_item.text) #训练VOC用NAME_LABEL_MAP[child_item.text]#因为用自己的这边的ID是编号  训练卫星数据用1
                if child_item.tag == 'difficult':
                    difficult=int(child_item.text)
                if child_item.tag == 'extra':
                    extra.append(child_item.text)
                if child_item.tag == 'robndbox':
                    tmp_box = [0, 0, 0, 0, 0,0,0]
                    for node in child_item:
                        if node.tag == 'cx':
                            tmp_box[0] = float(node.text)
                        if node.tag == 'cy':
                            tmp_box[1] = float(node.text)
                        if node.tag == 'w':
                            tmp_box[2] = float(node.text)
                        if node.tag == 'h':
                            tmp_box[3] = float(node.text)
                        if node.tag == 'angle':
                            tmp_box[4] = float(node.text)
                    assert label is not None, 'label is none, error'
                    tmp_box[5]=label
                    tmp_box[6]=difficult
                    box_list.append(tmp_box)

#    gtbox_label = np.array(box_list, dtype=np.int32) 
    img_size=[img_height,img_width,img_depth]
    return img_size,gsd,imagesource,box_list,extra

def write_rotate_xml(output_floder,img_name,size,gsd,imagesource,gtbox_label,CLASSES,extra=[]):#size,gsd,imagesource
    #将检测结果表示为中科星图比赛格式的程序,这里用folder字段记录gsd
    headstr = """\
   <annotation>
	<folder>{}</folder>
	<filename>{}</filename>
	<path>{}</path>
	<source>
		<database>{}</database>
	</source>
	<size>
		<width>{}</width>
		<height>{}</height>
		<depth>{}</depth>
	</size>
	<segmented>0</segmented>
    """
    objstr = """\
       <object>
		<name>{}</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>{}</difficult>
		<robndbox>
			<cx>{}</cx>
			<cy>{}</cy>
			<w>{}</w>
			<h>{}</h>
			<angle>{}</angle>
		</robndbox>
		<extra>{}</extra>
	</object>
    """
    tailstr = '''\
      </annotation>
    '''
    [floder,name]=os.path.split(img_name)
    filename=os.path.join(floder,os.path.splitext(name)[0]+'.xml')
    foldername=os.path.split(img_name)[0]
    head=headstr.format(gsd,name,foldername,imagesource,size[1],size[0],size[2])
    rotate_xml_name=os.path.join(output_floder,os.path.split(filename)[1])
    f = open(rotate_xml_name, "w",encoding='utf-8')
    f.write(head)
    for i,box in enumerate (gtbox_label):
        if len(extra)==0:
            obj=objstr.format(CLASSES[int(box[5])],int(box[6]),box[0],box[1],box[2],box[3],box[4],' ')
        else:
            obj=objstr.format(CLASSES[int(box[5])],int(box[6]),box[0],box[1],box[2],box[3],box[4],extra[i])
        f.write(obj)
    f.write(tailstr)
    f.close()

    
def read_txt_tianzhi(txt_dir,txt_path,NAME_LABEL_MAP_CHA):  

    txt_name=os.path.split(txt_path)[1]
    txt_name=os.path.splitext(txt_name)[0]+'.txt'
    txt_name=os.path.join(txt_dir,txt_name)
#    f = open(txt_path,"r",encoding='UTF-8')   #设置文件对象
#    txt_name='E:/Standard_data_set/Dota/train/labelrbb1.5/P0000.txt'
#    txt_name='E:/Dataset/2019tianzhi/russia_plane_result3/0.5-khurba-su24-20140919_111.txt'
    f = open(txt_name)
    data = f.readlines()  #直接将文件中按行读到list里，效果与方法2一样
    f.close()       
    box_list = []
    for i in data:
        str_num=i.strip().split(' ')
        #以下语句是替换掉无用的符号
        for j in range(len(str_num)):
            str_num[j]=str_num[j].replace('（', '').replace('）', '').replace('）', '').replace(',', '')
        label=NAME_LABEL_MAP_CHA[str_num[0]]
        
        
        ##到时候这里读真值的时候要改
        object_array=[float(str_num[2]),float(str_num[3]),float(str_num[4]),float(str_num[5]),label]
        box_list.append(object_array)
#    gtbox_label = np.array(box_list) 
    return box_list


def write_tianzhi_txt(input_dir,image_ID,boxes,label,CLASSES,scores=[]):
    #将检测结果表示为可以评价的txt文件的程序
#    txt_name=os.path.join(input_dir,image_ID.replace('jpg','txt').replace('tiff','txt'))
    txt_name=os.path.split(image_ID)[1]
    txt_name=os.path.splitext(txt_name)[0]+'.txt'
    txt_name=os.path.join(input_dir,txt_name)
    f = open(txt_name, "w",encoding='utf-8')
    if len(scores)>0:
        objstr = '{} {} {} {} {} {}\n'
        for i, box in enumerate(boxes):
#            ((x_center,y_center),(cv_w,cv_h),cv_angle) 
            cv_rotete_rect=rotate_rect2cv(box[0:5])
            rect_box = (cv2.boxPoints(cv_rotete_rect))
            xmin,ymin,xmax,ymax=np.min(rect_box[:,0]),np.min(rect_box[:,1]),np.max(rect_box[:,0]),np.max(rect_box[:,1])
            obj=objstr.format(CLASSES[int(label[i])],  '%.2f' % float(scores[i]) , '%.2f' % xmin, '%.2f' % ymin, '%.2f' % xmax, '%.2f' % ymax)
            f.write(obj)
    else:#这是模拟生成给定的文件
        objstr = '{} {} {} {} {}\n'
        for i, box in enumerate(boxes):
            cv_rotete_rect=rotate_rect2cv(box[0:5])
            rect_box = (cv2.boxPoints(cv_rotete_rect))
            xmin,ymin,xmax,ymax=np.min(rect_box[:,0]),np.min(rect_box[:,1]),np.max(rect_box[:,0]),np.max(rect_box[:,1])
            obj=objstr.format(CLASSES[int(label[i])], '%.2f' %xmin, '%.2f' %ymin, '%.2f' %xmax, '%.2f' %ymax)
            f.write(obj)
    f.close()
 


    
def read_ship_tianzhi_txt(txt_dir,txt_path,NAME_LABEL_MAP_CHA):  
    txt_name=os.path.split(txt_path)[1]
    txt_name=os.path.splitext(txt_name)[0]+'.txt'
    txt_name=os.path.join(txt_dir,txt_name)
#    f = open(txt_path,"r",encoding='UTF-8')   #设置文件对象
#    txt_name='E:/Standard_data_set/Dota/train/labelrbb1.5/P0000.txt'
#    txt_name='E:/Dataset/2019tianzhi/russia_plane_result3/0.5-khurba-su24-20140919_111.txt'
    f = open(txt_name)
    data = f.readlines()  #直接将文件中按行读到list里，效果与方法2一样
    f.close()       
    box_list = []
    for i in data:
        str_num=i.strip().split(' ')
        #以下语句是替换掉无用的符号
        for j in range(len(str_num)):
            str_num[j]=str_num[j].replace('（', '').replace('）', '').replace('）', '').replace(',', '')
        label=NAME_LABEL_MAP_CHA[str_num[0]]
        ##到时候这里读真值的时候要改
        object_array=[float(str_num[1]),float(str_num[2]),float(str_num[3]),float(str_num[4]),float(str_num[5]),float(str_num[6]),float(str_num[7]),float(str_num[8]),label]
        box_list.append(object_array)
#    gtbox_label = np.array(box_list) 
    return box_list

def tianzhi2roatate_xml(box_list,NAME_LABEL_MAP_CHA):  
    rotate_box_list=[]
    for i,box in enumerate (box_list):
        point1=np.array([box[0],box[1]])
        point2=np.array([box[2],box[3]])
        point3=np.array([box[4],box[5]])
        point4=np.array([box[6],box[7]])
        l12=np.linalg.norm(point1-point2) 
        l23=np.linalg.norm(point2-point3) 
        l34=np.linalg.norm(point3-point4)
        l41=np.linalg.norm(point4-point1)
        
        #  head
        head=(point1+point2)/2#头部坐标
        center=(point1+point2+point3+point4)/4#中心坐标
        Width=(l23+l41)/2
        Height=(l12+l34)/2
        det1=point2-point3
        det2=point1-point4
        if det1[0]==0:
            if det1[1]>0:
                Angle1=np.pi/2
            else:
                Angle1=-np.pi/2
        else:
             Angle1=np.arctan(det1[1]/det1[0])
        if det2[0]==0:
            if det2[1]>0:
                Angle2=np.pi/2
            else:
                Angle2=-np.pi/2
        else:
            Angle2=np.arctan(det2[1]/det2[0])
        #还会出现一种情况就是angle1 angle2 都比较大，但是由于在90度俯角，导致两个差异很大
        if np.abs(Angle1)>np.pi/2-np.pi/36:
            if Angle2<0:
                Angle1=-np.pi/2
            else:
                Angle1=np.pi/2
        if np.abs(Angle2)>np.pi/2-np.pi/36:
            if Angle1<0:
                Angle2=-np.pi/2
            else:
                Angle2=np.pi/2
        Angle=(Angle1+Angle2)/2
        #以上得到了HRSC格式的表示的各项数据，以下将其转为旋转xml格式的表示的数据
        #分别计算旋转矩形两个头部的坐标，和实际我们得出的头部坐标比较，距离小的我们就认为他是头部
        head_rect_right=[center[0]+Width/2*np.cos(Angle),center[1]+Width/2*np.sin(Angle)]
        head_rect_left=[center[0]-Width/2*np.cos(Angle),center[1]-Width/2*np.sin(Angle)]
        l_head_right=np.linalg.norm(head_rect_right-head) 
        l_head_left=np.linalg.norm(head_rect_left-head) 
        if l_head_right<l_head_left:#头部方向在第一四象限
            Angle=Angle+np.pi/2
        else:
            Angle=Angle+np.pi*3/2#头部方向在第二三象限，角度要在原来基础上加上PI
        NewWidth=Height
        NewHeight=Width
        if NewWidth>NewHeight:
            NewWidth=Width
            NewHeight=Height
            Angle=Angle-np.pi/2
            
        tmp_box=[center[0],center[1],NewWidth,NewHeight,Angle,box_list[i][8],0]
        rotate_box_list.append(tmp_box)
    return rotate_box_list



def write_ship_tianzhi_txt(input_dir,image_ID,boxes,label,CLASSES,scores=[]):
    #将检测结果表示为可以评价的txt文件的程序
#    txt_name=os.path.join(input_dir,image_ID.replace('jpg','txt').replace('tiff','txt'))
    txt_name=os.path.split(image_ID)[1]
    txt_name=os.path.splitext(txt_name)[0]+'.txt'
    txt_name=os.path.join(input_dir,txt_name)
    f = open(txt_name, "w",encoding='utf-8')
    if len(scores)>0:
        objstr = '{} {} {} {} {} {} {} {} {} {}\n'
#        objstr = '{} {} {} {} {} {} {} {} {}\n'
        for i, box in enumerate(boxes):
#            ((x_center,y_center),(cv_w,cv_h),cv_angle) 
            cv_rotete_rect=rotate_rect2cv(box[0:5])
            rect_box = (cv2.boxPoints(cv_rotete_rect))
            p0,p1,p2,p3=rect_box[0],rect_box[1],rect_box[2],rect_box[3]
#            xmin,ymin,xmax,ymax=np.min(rect_box[:,0]),np.min(rect_box[:,1]),np.max(rect_box[:,0]),np.max(rect_box[:,1])
            obj=objstr.format(CLASSES[int(label[i])],  '%.2f' % float(scores[i]) , '%.2f' % p0[0], '%.2f' % p0[1], '%.2f' % p1[0], '%.2f' % p1[1], '%.2f' % p2[0], '%.2f' % p2[1], '%.2f' % p3[0], '%.2f' % p3[1])
#            obj=objstr.format(CLASSES[int(label[i])], '%.2f' % p0[0], '%.2f' % p0[1], '%.2f' % p1[0], '%.2f' % p1[1], '%.2f' % p2[0], '%.2f' % p2[1], '%.2f' % p3[0], '%.2f' % p3[1])
            f.write(obj)
    
def GT_xml2txt(xmlname):
    #将HRSC的xml文件转为可以评价的txt的函数
    tree = ET.parse(xmlname)
    root = tree.getroot()
    txt_name=xmlname.replace('jpg','txt')
    txt_name=os.path.split(xmlname)[1]
    txt_name=os.path.splitext(txt_name)[0]+'.txt'
    bboxs=[]
    for i in root:  # 遍历一级节点
        if i.tag == 'HRSC_Objects':
            for j in i:
                if j.tag == 'HRSC_Object':
                    bbox = []
                    xmin = 0
                    ymin = 0
                    xmax = 0
                    ymax = 0
                    for r in j:
                        if r.tag == 'Class_ID':
                            Class_ID = 'ship'#eval(r.text)
                        if r.tag == 'box_xmin':
                            xmin = eval(r.text)
                        if r.tag == 'box_ymin':
                            ymin = eval(r.text)
                        if r.tag == 'box_xmax':
                            xmax = eval(r.text)
                        if r.tag == 'box_ymax':
                            ymax = eval(r.text)
                    bbox.append(Class_ID)   # 保存当前box对应的image_id
                    bbox.append(xmin)
                    bbox.append(ymin)
                    bbox.append(xmax)
                    bbox.append(ymax)
                    bboxs.append(bbox)
    f = open(txt_name, "w",encoding='utf-8')
    objstr = '{} {} {} {} {}\n'
    for i, box in enumerate(bboxs):
#        obj=objstr.format(CLASSES[label[i]],scores[i],box[0],box[1],box[2],box[3])
        obj=objstr.format(box[0],box[1],box[2],box[3],box[4])
        f.write(obj)
    f.close()

def write_evl_txt(input_dir,image_ID,boxes,label,scores,CLASSES):
    #将检测结果表示为可以评价的txt文件的程序
    txt_name=os.path.join(input_dir,image_ID.replace('jpg','txt'))
    txt_name=os.path.split(image_ID)[1]
    txt_name=os.path.splitext(txt_name)[0]+'.txt'
    txt_name=os.path.join(input_dir,txt_name)
    f = open(txt_name, "w",encoding='utf-8')
    objstr = '{} {} {} {} {} {}\n'
    for i, box in enumerate(boxes):
        obj=objstr.format('ship',scores[i],box[0],box[1],box[2],box[3])
        f.write(obj)
    f.close()


##xml 变换区  
def DOTA2Rotatexml(imgfolder,txtfolder,filename):
    name=os.path.split(filename)[1]
    txt_name=os.path.splitext(name)[0]+'.txt'
    txt_name=os.path.join(txtfolder,txt_name)
    file=open(txt_name) 
    #'imagesource':imagesource
    #'gsd':gsd
    #x1, y1, x2, y2, x3, y3, x4, y4, category, difficult
    #x1, y1, x2, y2, x3, y3, x4, y4, category, difficult
    #...	   
    dataMat=[]  
    labelMat=[]
    difficultMat=[]
    box_list=[]
    for i,line in enumerate (file.readlines()): 
        curLine=line.strip().split("\t")
        curLine=''.join(curLine)
        data=curLine.split( )
        if i==0:
            imagesource=curLine.split(':')[1]
        if i==1:
            data=curLine.split(':')
            if data[1] != 'null':
                gsd=float(data[1])
            else:
                gsd=0;
        elif i>1:
    #        floatLine=map(float,curLine)#这里使用的是map函数直接把数据转化成为float类型    
            points=[float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5]),float(data[6]),float(data[7])]
            label=data[8]
            difficult=int(data[9])
            dataMat.append(points) 
            labelMat.append(label)
            difficultMat.append(difficult)
    img=cv2.imread(filename)#（高，宽，（B,G,R））
    size = img.shape#高，宽 ，通道
    for i,box in enumerate (dataMat):
        point1=np.array([box[0],box[1]])
        point2=np.array([box[2],box[3]])
        point3=np.array([box[4],box[5]])
        point4=np.array([box[6],box[7]])
        l12=np.linalg.norm(point1-point2) 
        l23=np.linalg.norm(point2-point3) 
        l34=np.linalg.norm(point3-point4)
        l41=np.linalg.norm(point4-point1)
        head=(point1+point2)/2#头部坐标
        center=(point1+point2+point3+point4)/4#中心坐标
        Width=(l23+l41)/2
        Height=(l12+l34)/2
        det1=point2-point3
        det2=point1-point4
        if det1[0]==0:
            if det1[1]>0:
                Angle1=np.pi/2
            else:
                Angle1=-np.pi/2
        else:
             Angle1=np.arctan(det1[1]/det1[0])
        if det2[0]==0:
            if det2[1]>0:
                Angle2=np.pi/2
            else:
                Angle2=-np.pi/2
        else:
            Angle2=np.arctan(det2[1]/det2[0])
        #还会出现一种情况就是angle1 angle2 都比较大，但是由于在90度俯角，导致两个差异很大
        if np.abs(Angle1)>np.pi/2-np.pi/36:
            if Angle2<0:
                Angle1=-np.pi/2
            else:
                Angle1=np.pi/2
        if np.abs(Angle2)>np.pi/2-np.pi/36:
            if Angle1<0:
                Angle2=-np.pi/2
            else:
                Angle2=np.pi/2
        Angle=(Angle1+Angle2)/2
        #以上得到了HRSC格式的表示的各项数据，以下将其转为旋转xml格式的表示的数据
        #分别计算旋转矩形两个头部的坐标，和实际我们得出的头部坐标比较，距离小的我们就认为他是头部
        head_rect_right=[center[0]+Width/2*np.cos(Angle),center[1]+Width/2*np.sin(Angle)]
        head_rect_left=[center[0]-Width/2*np.cos(Angle),center[1]-Width/2*np.sin(Angle)]
        l_head_right=np.linalg.norm(head_rect_right-head) 
        l_head_left=np.linalg.norm(head_rect_left-head) 
        if l_head_right<l_head_left:#头部方向在第一四象限
            Angle=Angle+np.pi/2
        else:
            Angle=Angle+np.pi*3/2#头部方向在第二三象限，角度要在原来基础上加上PI
        NewWidth=Height
        NewHeight=Width
        tmp_box=[center[0],center[1],NewWidth,NewHeight,Angle,NAME_LABEL_MAP[labelMat[i]],difficultMat[i]]
        box_list.append(tmp_box)
    return size,gsd,imagesource,box_list


##xml 变换区  
#def Tianzhiship2Rotatexml():
#    points=[float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5]),float(data[6]),float(data[7])]
#            label=data[8]
#            difficult=int(data[9])
#            dataMat.append(points) 
#            labelMat.append(label)
#            difficultMat.append(difficult)
#    img=cv2.imread(filename)#（高，宽，（B,G,R））
#    size = img.shape#高，宽 ，通道
#    for i,box in enumerate (dataMat):
#        point1=np.array([box[0],box[1]])
#        point2=np.array([box[2],box[3]])
#        point3=np.array([box[4],box[5]])
#        point4=np.array([box[6],box[7]])
#        l12=np.linalg.norm(point1-point2) 
#        l23=np.linalg.norm(point2-point3) 
#        l34=np.linalg.norm(point3-point4)
#        l41=np.linalg.norm(point4-point1)
#        head=(point1+point2)/2#头部坐标
#        center=(point1+point2+point3+point4)/4#中心坐标
#        Width=(l23+l41)/2
#        Height=(l12+l34)/2
#        det1=point2-point3
#        det2=point1-point4
#        if det1[0]==0:
#            if det1[1]>0:
#                Angle1=np.pi/2
#            else:
#                Angle1=-np.pi/2
#        else:
#             Angle1=np.arctan(det1[1]/det1[0])
#        if det2[0]==0:
#            if det2[1]>0:
#                Angle2=np.pi/2
#            else:
#                Angle2=-np.pi/2
#        else:
#            Angle2=np.arctan(det2[1]/det2[0])
#        #还会出现一种情况就是angle1 angle2 都比较大，但是由于在90度俯角，导致两个差异很大
#        if np.abs(Angle1)>np.pi/2-np.pi/36:
#            if Angle2<0:
#                Angle1=-np.pi/2
#            else:
#                Angle1=np.pi/2
#        if np.abs(Angle2)>np.pi/2-np.pi/36:
#            if Angle1<0:
#                Angle2=-np.pi/2
#            else:
#                Angle2=np.pi/2
#        Angle=(Angle1+Angle2)/2
#        #以上得到了HRSC格式的表示的各项数据，以下将其转为旋转xml格式的表示的数据
#        #分别计算旋转矩形两个头部的坐标，和实际我们得出的头部坐标比较，距离小的我们就认为他是头部
#        head_rect_right=[center[0]+Width/2*np.cos(Angle),center[1]+Width/2*np.sin(Angle)]
#        head_rect_left=[center[0]-Width/2*np.cos(Angle),center[1]-Width/2*np.sin(Angle)]
#        l_head_right=np.linalg.norm(head_rect_right-head) 
#        l_head_left=np.linalg.norm(head_rect_left-head) 
#        if l_head_right<l_head_left:#头部方向在第一四象限
#            Angle=Angle+np.pi/2
#        else:
#            Angle=Angle+np.pi*3/2#头部方向在第二三象限，角度要在原来基础上加上PI
#        NewWidth=Height
#        NewHeight=Width
#        tmp_box=[center[0],center[1],NewWidth,NewHeight,Angle,NAME_LABEL_MAP[labelMat[i]],difficultMat[i]]
#        box_list.append(tmp_box)
#    return size,gsd,imagesource,box_list

def rotate_xml_filter(size,gsd,imagesource,box_list):
    #过滤xml的程序，只有当指定类别满足时才会保存当前的标注
    valid_gtbox_label=[]
    for i,box in enumerate (box_list):
       #[center[0],center[1],NewWidth,NewHeight,Angle,NAME_LABEL_MAP[labelMat[i]],difficultMat[i]]
        if box[5]==8 or box[5]==10:
            valid_gtbox_label.append(box)
    return size,gsd,imagesource,valid_gtbox_label

def rect_xml_filter(box_list):
    #过滤xml的程序，只有当指定类别满足时才会保存当前的标注
    valid_gtbox_label=[]
    for i,box in enumerate (box_list):
       #[center[0],center[1],NewWidth,NewHeight,Angle,NAME_LABEL_MAP[labelMat[i]],difficultMat[i]]
        if box[4]==8:
            valid_gtbox_label.append(box)
    return valid_gtbox_label

def rotate_xml_transform(t_y,t_x,scale,gsd,box_list):
    #对图像和平移和缩放后的对应的xml文件的变换,如果同时进行，则先缩放，再平移
    valid_gtbox_label=[]
    for i,box in enumerate (box_list):
        #[center[0],center[1],NewWidth,NewHeight,Angle,NAME_LABEL_MAP[labelMat[i]],difficultMat[i]]
        valid_gtbox_label.append([box[0]*scale-t_x,box[1]*scale-t_y,box[2]*scale,box[3]*scale,box[4],box[5],box[6]])
    return gsd/scale,valid_gtbox_label

def rotate_rect2cv(rotatebox):
    #此程序将rotatexml中旋转矩形的表示，转换为cv2的RotateRect
    
    [x_center,y_center,w,h,angle]=rotatebox
    angle_mod=angle*180/np.pi%180
    if angle_mod>=0 and angle_mod<90:
        [cv_w,cv_h,cv_angle]=[h,w,angle_mod-90]
    if angle_mod>=90 and angle_mod<180:
        [cv_w,cv_h,cv_angle]=[w,h,angle_mod-180]
    return ((x_center,y_center),(cv_w,cv_h),cv_angle) 
def rotate_rect2cv_np(rotatebox):
    #此程序将rotatexml中旋转矩形的表示，转换为cv2的RotateRect
    
    [x_center,y_center,w,h,angle]=rotatebox
    angle_mod=angle*180/np.pi%180
    if angle_mod>=0 and angle_mod<90:
        [cv_w,cv_h,cv_angle]=[h,w,angle_mod-90]
    if angle_mod>=90 and angle_mod<180:
        [cv_w,cv_h,cv_angle]=[w,h,angle_mod-180]
    return x_center,y_center,cv_w,cv_h,cv_angle 

def rotate_xml_valid(h_crop,w_crop,outrange_ratio,box_list,flag='normal'):
    #周处切割区域的坐标
    gtbox_label_valid=[]
    rect_boxs=[]
#    rect_box_draw=[]
    for i,box in enumerate (box_list):
        #[center[0],center[1],NewWidth,NewHeight,Angle,NAME_LABEL_MAP[labelMat[i]],difficultMat[i]]
        cv_rotete_rect=rotate_rect2cv(box[0:5])
        rect_box = np.int0(cv2.boxPoints(cv_rotete_rect))
        box_xmin , box_xmax  ,box_ymin, box_ymax=np.min(rect_box[:,0]) ,np.max(rect_box[:,0]) ,np.min(rect_box[:,1]) ,np.max(rect_box[:,1])
        box_w ,box_h=box_xmax-box_xmin ,box_ymax-box_ymin
        #找出位置合适的目标标签，当出界不超过20%的时候保留
        if box_xmin>-box_w*outrange_ratio and box_ymin>-box_h*outrange_ratio and box_xmax-w_crop<box_w*outrange_ratio and box_ymax-h_crop<box_h*outrange_ratio:
            gtbox_label_valid.append(box)

            #这里有两种斜框转为正框的格式，一种是正常的转换就是
            if flag=='plane':
                rect_box_new=[]
                rect_box_new.append(np.int32((rect_box[0]+rect_box[1])/2))
                rect_box_new.append(np.int32((rect_box[1]+rect_box[2])/2))
                rect_box_new.append(np.int32((rect_box[2]+rect_box[3])/2))
                rect_box_new.append(np.int32((rect_box[3]+rect_box[0])/2))
                rect_box_new=np.array(rect_box_new)
                xmin,ymin,xmax,ymax=np.min(rect_box_new[:,0]),np.min(rect_box_new[:,1]),np.max(rect_box_new[:,0]),np.max(rect_box_new[:,1])
                rect_boxs.append([xmin,ymin,xmax,ymax,box[5]])
            else:
                rect_boxs.append([box_xmin,box_ymin,box_xmax,box_ymax,box[5]])
    return gtbox_label_valid,rect_boxs

def rotate_xml_rotate(ang_rad,rot_center,box_list):
    #对图像和平移和缩放后的对应的xml文件的变换,如果同时进行，则先缩放，再平移
#    ang_rad=ang/180*(np.pi)
    gtbox_label_valid=[]
    for i,box in enumerate (box_list):
        #[center[0],center[1],NewWidth,NewHeight,Angle,NAME_LABEL_MAP[labelMat[i]],difficultMat[i]]
        [x_center,y_center,w,h,angle]=box[0:5]
        [x_relative_center,y_relative_center]=[x_center-rot_center[0] ,y_center-rot_center[1] ]
        angle_new=(angle+ang_rad)%(2*np.pi)
        RotateMatrix=np.array([
                              [np.cos(ang_rad),-np.sin(ang_rad)],
                              [np.sin(ang_rad),np.cos(ang_rad)]])
        a=np.transpose([x_relative_center,y_relative_center])
        new_center=np.transpose(np.dot(RotateMatrix, a))+rot_center
        gtbox_label_valid.append([new_center[0],new_center[1],w,h,angle_new,box[5],box[6]])
#        rect_box_draw.append(rect_box)
#    rect_box_draw=np.array(rect_box_draw, dtype=np.int32) 
#    cv2.polylines(img,rect_box_draw,True,(255,0,0))
##        cv2.drawContours(img, [rect_boxs], 0, (0, 0, 255), 2)
#    cv2.imshow('rotate',img.astype('uint8'))
#    cv2.waitKey(0)
    return gtbox_label_valid

def rotate_xml_flip(h_crop,w_crop,flag,box_list):
    #水平或垂直翻转xml文件 horizontal vertical
    gtbox_label_valid=[]
    if flag=='horizontal':
        for i,box in enumerate (box_list):
            [x_center,y_center,w,h,angle]=box[0:5]
            gtbox_label_valid.append([w_crop-x_center,y_center,w,h,2*np.pi-angle,box[5],box[6]])
    elif flag=='vertical':
        for i,box in enumerate (box_list):
            [x_center,y_center,w,h,angle]=box[0:5]
            angle_ver=(3*np.pi-angle)%(2*np.pi)
            gtbox_label_valid.append([x_center,h_crop-y_center,w,h,angle_ver,box[5],box[6]])
    else:
        print('flag error!')
    return gtbox_label_valid

def rect_xml_flip(h_crop,w_crop,flag,box_list):
    #水平或垂直翻转xml文件 horizontal vertical
    gtbox_label_valid=[]
    if flag=='horizontal':
        for i,box in enumerate (box_list):
            [xmin,ymin,xmax,ymax]=box[0:4]
            gtbox_label_valid.append([w_crop-xmax,ymin,w_crop-xmin,ymax,box[4]])
    elif flag=='vertical':
        for i,box in enumerate (box_list):
             [xmin,ymin,xmax,ymax]=box[0:4]
             gtbox_label_valid.append([xmin,h_crop-ymax,xmax,h_crop-ymin,box[4]])
    else:
        print('flag error!')
    return gtbox_label_valid





#以下是图像变换区
#图像旋转用，里面的angle是角度制的
def im_rotate(im,angle,center = None,scale = 1.0):
    #旋转角度逆时针为正值角度值
    angle=angle/np.pi*180
    h,w = im.shape[:2]
    if center is None:
        center = (w/2,h/2)
    M = cv2.getRotationMatrix2D(center,angle,scale)
    im_rot = cv2.warpAffine(im,M,(w,h))
    return im_rot

def randomColor(image):
    #    颜色抖动
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # cv2 转 PIL
    random_factor = np.random.randint(0, 23) / 10.  # 随机因子31
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 13) / 10.  # 随机因子21
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 13) / 10.  # 随机因1子21
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 23) / 10.  # 随机因子31
    img=ImageEnhance.Sharpness(contrast_image).enhance(random_factor)# 调整图像锐度
    # PIL 转 cv2
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return img.astype(np.uint8)

def PCAJitter(img):
    #    PCA抖动
    image = img
    img = image / 255.0
    img_size = img.size // 3
    img1 = img.reshape(img_size, 3)
    img1 = np.transpose(img1)
    img_cov = np.cov([img1[0], img1[1], img1[2]]) #计算协方差，默认情况下每一行代表一个变量（属性），每一列代表一个观测
    lamda, p = np.linalg.eig(img_cov)
    p = np.transpose(p)
    alpha0 = np.random.uniform(0, 0.3)
    alpha1 = np.random.uniform(0, 0.3)
    alpha2 = np.random.uniform(0, 0.3)
    v = np.transpose((alpha0*img1[0], alpha1*img1[1], alpha2*img1[2]))
    add_num = 2*np.dot(v, p).reshape(np.shape(img))
    return np.array(image + add_num).astype(np.uint8)

def adjustGamma(image,para):
    return  exposure.adjust_gamma(image, para)

def downUpSample(image,scale=1/2):
    #下采样，上采样
    img_downsampled = cv2.resize(image,(int(scale*image.shape[1]),int(scale*image.shape[0])))
    img_sam=cv2.resize(img_downsampled,(image.shape[1],image.shape[0]))
    return img_sam

def addNoise(image,sigma=0.04):
    #图像加燥 模式 gaussian localvar poisson salt pepper s&p speckle
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # cv2 转 PIL
    img_noise =util.random_noise(image, var=sigma**2)*255
    img_noise=img_noise.astype(np.uint8)
    return cv2.cvtColor(np.asarray(img_noise), cv2.COLOR_RGB2BGR)

def randFlip(image,flag='Image.FLIP_TOP_BOTTOM'):
    #随机翻转
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if flag=='Image.FLIP_TOP_BOTTOM':
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

def strong_aug(image,p=0.5):
    image2 =Compose([#加躁
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
            ISONoise(),
        ], p=0.2),
        OneOf([#模糊
            MotionBlur(p=0.1),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
            JpegCompression(p=.1),
        ], p=0.2),
        OneOf([#锐化
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
     OneOf([#直方图均衡，对比度，色度变化,pca
            HueSaturationValue(),
            RandomBrightnessContrast(),
            Equalize(),
            # FancyPCA(),
        ], p=0.3),
    ToGray(p=0.1),
    ], p=p)(image=image)['image']
    return image2


