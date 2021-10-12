# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 18:22:36 2019
版本10.31
@author: admin
"""

import os
import cv2
import numpy as np
import sys
sys.path.append('/home/zf/0tools/')
import DataFunction
from timeit import default_timer as timer
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool

#voc路径
# img_dir = '/home/zf/2020ZKXT/final_all/最终所有resize2.69图片/dota2.69rgb_1'
img_dir = '/home/zf/2020HJJ/train_worm/Test/images'
xml_dir = '/home/zf/2020HJJ/train_worm/Test/rotatexml_val'
outputfolder='/home/zf/2020HJJ/train_worm/Test/teseaug0'#r'E:\Dataset\US_Navy_test_aug'
file_ext=['.jpg','.png','.tif']
[w_crop,h_crop]=[1024,1024]  #这是要切割的图像尺寸  第一个是宽，第二个是高
overlap_ratio=1/5
outrange_ratio=2/3
ratios=[1]#设置在切割过程中缩放的倍数

skip_center=3 #表示如果机场有30以下的飞机，则正常都遍历，如果超过了30个，每增加一倍，取中心的步距增加一倍
angle_range = [np.pi/2,np.pi*19/20]#角度im_rotate用到的是角度制 旋转角度的范围

NAME_LABEL_MAP = {
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
}
def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict
LABEl_NAME_MAP = get_label_name_map()

##每一类别的飞机旋转的次数
angle_number_dict=  {
        'ship': 2
        } 

#新建输出文件夹
if not os.path.isdir(outputfolder):
    os.makedirs(outputfolder)
    
#读取原图全路径  
imgs_path = DataFunction.get_file_paths_recursive_list(img_dir, file_ext) 
#旋转角的大小，整数表示逆时针旋转
imgs_total_num=len(imgs_path)
def process(img_path):  #将这个看成一个循环
#for num,img_path in enumerate(imgs_path,0):
    num=np.random.randint(1,13)
    angle_range[0]=(angle_range[0]+num*np.pi/10)%2*np.pi
    angle_range[1]=(angle_range[1]+num*np.pi/10)%2*np.pi
    start = timer()
    #一、读取图像并获取基本参数
#    img_path=imgs_path[num]
#    img = cv2.imread(img_path)
    img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
    img_size=img.shape#高，宽 ，通道
    img_num=1
    #二、读取标注文件
    [floder,name]=os.path.split(img_path)
    xml_path=os.path.join(xml_dir,os.path.splitext(name)[0]+'.xml')
    
    img_size1,gsd,imagesource,gtbox,extra=DataFunction.read_rotate_xml(xml_path,NAME_LABEL_MAP)
   
    #三、把单通道图像转为三通道
    if img_size[2]==1:
        img_new=np.zeros((img_size[0],img_size[1],3))
        img_new[:,:,0]=img
        img_new[:,:,1]=img
        img_new[:,:,2]=img
        img=img_new
        img_size[2]=3
    if len(gtbox)>0:
        #图像切割
        # img_num=DataFunction.crop_img_rotatexml(ratios,overlap_ratio,outrange_ratio,h_crop,w_crop,img_path,outputfolder,'',img_num,img_size,img,gsd,imagesource,gtbox,LABEl_NAME_MAP)
        #  #image resize
        # if img_size[0]/img_size[1]<1.333 and img_size[0]/img_size[1]>0.75 and img_size[0]/h_crop>0.5 and img_size[0]/h_crop<5 and img_size[1]/w_crop>0.5 and img_size[1]/w_crop<2:
        #    image_resize=cv2.resize(img,(h_crop, w_crop))
        #    rboxs=[]
        #    x_scale=w_crop/img_size[1]
        #    y_scale=h_crop/img_size[0]
        #    for box in gtbox:
        #        cv_rotete_rect=DataFunction.rotate_rect2cv(box[0:5])
        #        cnt =cv2.boxPoints(cv_rotete_rect)
        #        cnt[:,0]=cnt[:,0]*x_scale
        #        cnt[:,1]=cnt[:,1]*y_scale
        #        rect = cv2.minAreaRect(cnt)
        #        rbox=DataFunction.cv2OriBox(rect)
        #        rboxs.append(rbox)
        #    rboxs=np.array(rboxs)
        #    gtbox_array_resize=np.array(gtbox)
        #    gtbox_array_resize[:,0:5]=rboxs
        #    rboxs=gtbox_array_resize.tolist()
        #    src_img=image_resize.astype(np.uint8)
        #    src_img=DataFunction.strong_aug(src_img,p_all=0.7)
        #    [floder,name]=os.path.split(img_path)
        #    img_new_name=os.path.splitext(name)[0]+('__{}__{}__{}__{}.{}').format(x_scale,y_scale,0,0,'jpg')
        #    img_new_name=os.path.join(outputfolder,img_new_name)
        #    cv2.imwrite(img_new_name, src_img)
        #    DataFunction.write_rotate_xml(outputfolder,img_new_name.replace('.jpg','.xml'),img_size,0.5,imagesource,rboxs,LABEl_NAME_MAP)

     #图像旋转切割
        gtbox_label_array=np.array(gtbox) 
        label_num=gtbox_label_array[:,5]
        centers=gtbox_label_array[:,0:2].tolist()
        labels=gtbox_label_array[:,5].tolist()
        center=[int(img_size[0]/2),int(img_size[1]/2)]
       
        angles=[4/18*np.pi]#转为弧度,7/18*np.pi
        #这里做一个判断，如果一幅图像目标数量过多，则减少围绕旋转的中心数量
        multiple=int(len(centers)/skip_center)+1
        centers_new=centers[0:len(centers)+1:multiple]
        labels_new=labels[0:len(centers)+1:multiple]
        for ang in angles:
            im_rot=DataFunction.im_rotate(img,-ang,center =(center[0],center[1]),scale=1.0)
            gtbox_label_rotate=DataFunction.rotate_xml_rotate(ang,center,gtbox)
            gtbox_label_rotate,rect_box=DataFunction.rotate_xml_valid(img_size[0],img_size[1],outrange_ratio,gtbox_label_rotate)
            img_num=DataFunction.crop_img_rotatexml(ratios,overlap_ratio,outrange_ratio,h_crop,w_crop,img_path,outputfolder,'',img_num,img_size,im_rot,gsd,imagesource,gtbox_label_rotate,LABEl_NAME_MAP,center,flag='plane')
#
#        for i, center in enumerate (centers_new):
#            angle_number=angle_number_dict[LABEl_NAME_MAP[labels_new[i]]]
#            # angle_number=angle_number_class[int(labels_new[i])-1]
#            angles=np.linspace(min(angle_range),max(angle_range),num=angle_number)+i*8*np.pi/180#转为弧度
#            for ang in angles:
#                im_rot=DataFunction.im_rotate(img,-ang,center =(center[0],center[1]),scale=1.0)
#                gtbox_label_rotate=DataFunction.rotate_xml_rotate(ang,center,gtbox)
#                gtbox_label_rotate,rect_box=DataFunction.rotate_xml_valid(img_size[0],img_size[1],outrange_ratio,gtbox_label_rotate)
#                img_num=DataFunction.crop_img_rotatexml(ratios,overlap_ratio,outrange_ratio,h_crop,w_crop,img_path,outputfolder,'',img_num,img_size,im_rot,gsd,imagesource,gtbox_label_rotate,LABEl_NAME_MAP,center,flag='plane')
    else :
        print('{} annotation is empty!'.format(img_path))
    time_elapsed = timer() - start
    print('{}/{}，time{}: augnum:{}'.format(num,imgs_total_num,time_elapsed,img_num))
    
    
    
start = timer()
#pool = ThreadPool()
pool = Pool(10)
pool.map(process, imgs_path)
pool.close()
pool.join()
time_elapsed = timer() - start
print(time_elapsed)