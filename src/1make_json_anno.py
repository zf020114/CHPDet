# coding=utf-8
import xml.etree.ElementTree as ET
import os
import json
import numpy as np
import DataFunction
import cv2
from timeit import default_timer as timer
from All_Class_NAME_LABEL import NAME_LABEL_MAP_HRSC,NAME_LABEL_MAP_USnavy_20,NAME_LABEL_MAP_ICDAR,NAME_LABEL_MAP_HRSC

NAME_LABEL_MAP=NAME_LABEL_MAP_USnavy_20#NAME_LABEL_MAP_ICDAR
data_dir='/home/zf/CenterNet/data/coco_usnavy_val/'#'/media/zf/E/Dataset/HRSC2016_aug1024/'#r'E:\Dataset\US_Navy_train_aug'

Rotatexmls =os.path.join( data_dir,'rotatexml_val')
json_name =os.path.join( data_dir,'annotations/person_keypoints_val2017.json')

# Rotatexmls =os.path.join( data_dir,'rotatexml')
# json_name =os.path.join( data_dir,'annotations/person_keypoints_train2017.json')

def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict
LABEl_NAME_MAP = get_label_name_map()
voc_clses=[]
for name, label in NAME_LABEL_MAP.items():
    voc_clses.append(name)

categories = []
for iind, cat in enumerate(voc_clses):
    cate = {}
    cate['supercategory'] = cat
    cate['name'] = cat
    cate['id'] = iind + 1
    cate['keypoints'] = [
                "ship head"
            ]
    cate['skeleton'] =   [
                [
                    1,
                    1
                ]
            ]
    categories.append(cate)
    
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

def txt2list(txtfile): 
    f = open(txtfile)
    l = []
    for line in f:
        l.append(line[:-1])
    return l

def xml2bbox_seg(xmlname, rotate_box_list,i_index):
    #通过这个函数将旋转矩形框换正框转换为coco可以读取的格式。其中将旋转矩形框转换维恩带有头部的分割点集
    bbox=[]
    for ind, rectbox in enumerate(rotate_box_list):
        rotatebox=rotate_box_list[ind]
        #rect_box_list:xmin ymin xmax ymax label
        #rotate_box_list:cx cy w h angle label difficult
        
        img_id=i_index#int(os.path.split(xmlname)[1].replace('.xml','').split('/')[-1]) 
        [x_center,y_center,w,h,angle,label]=rotatebox[0:6]
        # cv_rotete_rect=DataFunction.rotate_rect2cv(rotatebox[0:5])
         #以下的代码是我想更精确地回归bbox，所以以机头，机尾，以及两个机翼的大概位置为四点，然后确定bbox的大小，但是这种方法证明是错误的。
        #因为实例分割他是在检测结果的基础上再进行分割，这样之后，检测的结果会限制分割的结果，导致在最后的分割模型进行最小矩形拟合的时候，都是正方形，得不到正确的方向。
        xmin,ymin,xmax,ymax = x_center-w/2, y_center-h/2 , x_center+w/2, y_center+h/2 
        RotateMatrix=np.array([
                              [np.cos(angle),-np.sin(angle)],
                              [np.sin(angle),np.cos(angle)]])
    #以下是舰船的liu点的mask制作方式
        r1,rhead1,rhead2,rhead, r2,r3,r4=np.transpose([-w/2,-h/2+w]),np.transpose([-w/12,-h/2]),np.transpose([w/12,-h/2]),np.transpose([0,-h/2]),np.transpose([w/2,-h/2+w]),np.transpose([w/2,h/2]),np.transpose([-w/2,h/2])
    #这是矩形的mask制作方式
#        r1,rhead,r2,r3,r4=np.transpose([-w/2,-h/2]),np.transpose([0,-h/2]),np.transpose([w/2,-h/2]),np.transpose([w/2,h/2]),np.transpose([-w/2,h/2])
    #这是飞机的制作mask的方式
#        r1,rhead,r2,r3,r4=np.transpose([-w/2,0]),np.transpose([0,-h/2]),np.transpose([w/2,0]),np.transpose([w/2,h/2]),np.transpose([-w/2,h/2])
        p1=np.transpose(np.dot(RotateMatrix, r1))+[x_center,y_center]
        head1=np.transpose(np.dot(RotateMatrix, rhead1))+[x_center,y_center]
        head2=np.transpose(np.dot(RotateMatrix, rhead2))+[x_center,y_center]
        head=np.transpose(np.dot(RotateMatrix, rhead))+[x_center,y_center]
        p2=np.transpose(np.dot(RotateMatrix, r2))+[x_center,y_center]
        p3=np.transpose(np.dot(RotateMatrix, r3))+[x_center,y_center]
        p4=np.transpose(np.dot(RotateMatrix, r4))+[x_center,y_center]
        area=h*w*3/4
        bbox.append([p1[0],p1[1],head1[0],head1[1],head2[0],head2[1],p2[0],p2[1],p3[0],p3[1],p4[0],p4[1],head[0],head[1],2,xmin,ymin,w,h,img_id,label,area])
#        bbox.append[p1,head,head,p2,p3,p4,xmin,ymin,xmax - xmin,ymax - ymin,img_id,label,h*w*3/4]
    return bbox

if not os.path.isdir(os.path.split(json_name)[0]):
    os.makedirs(os.path.split(json_name)[0])
#voc2007xmls = '/home/zf/dataset/HRSC/Annotitions/'
#json_name = '/home/zf/dataset/HRSC/annotations/instances_train2017.json'

xmls = get_file_paths_recursive(Rotatexmls,'xml')
bboxes = []
ann_js = {}
images = []
total_num=len(xmls)
start = timer()
for i_index, xml_file in enumerate(xmls):
#    img_height, img_width, rect_box_list=TestFunction.read_VOC_xml(xml_file)
#    rotate_xml_path=xml_file.replace(voc2007xmls,Rotatexmls)
    img_size,gsd,imagesource,rotate_box_list,extra=DataFunction.read_rotate_xml(xml_file,NAME_LABEL_MAP)
#    assert img_size[0] == img_height and img_size[1] == img_width, 'ann %s size dont match!' % (xml_file)
#    assert len(rect_box_list) == len(rotate_box_list), 'ann %s size dont match!' % (xml_file)
    image = {}
    image['file_name'] = os.path.split(xml_file)[-1].replace('xml','jpg')
    image['width'] = 1024#img_size[1]#width#
    image['height'] =1024#img_size[0]#600#
    image['id'] =i_index#int(os.path.split(xml_file)[1].replace('.xml','').split('/')[-1]) 
    #p1[0],p1[1],head[0],head[1],p2[0],p2[1],p3[0],p3[1],p4[0],p4[1],xmin,ymin,xmax-xmin,ymax-ymin,img_id,label,area
#    sig_xml_bbox_old=xml2bbox_seg_old(xml_file,rect_box_list, rotate_box_list)
    sig_xml_bbox=xml2bbox_seg(xml_file,rotate_box_list,i_index)
#    image, sig_xml_bbox = getimages(xml_file, i_index)
    images.append(image)
    bboxes.extend(sig_xml_bbox)
    if i_index%1000==0:
        print('{}/{}'.format(i_index,total_num))
time_elapsed = timer() - start
print('time:{}s'.format(time_elapsed))

ann_js['images'] = images
ann_js['categories'] = categories
annotations = []
total_box=len(bboxes)
for box_ind, box in enumerate(bboxes):
    anno = {}
    anno['image_id'] =  box[-3]
    anno['category_id'] = box[-2]  #1 HRSC
    anno['bbox'] = box[-7:-3]
    anno['id'] = box_ind
    anno['area'] = box[-1]
    anno['iscrowd'] = 0
    anno['segmentation']=[box[0:12]]
    anno['num_keypoints'] = 1
    z1=np.array(box[12:15]).reshape((1,-1))#np.ones([1,3])
    zeros_mat=np.zeros([1,48])
    # z2=np.squeeze(np.column_stack((z1,zeros_mat))).astype(np.int32)
    anno['keypoints']=z1.tolist()
    annotations.append(anno)
    if box_ind%5000==0:
        print('{}/{}'.format(box_ind,total_box))
ann_js['annotations'] = annotations
json.dump(ann_js, open(json_name, 'w'), indent=4)  # indent=4 更加美观显示
time_elapsed = timer() - start
print('time:{}s'.format(time_elapsed))
print('down!')
