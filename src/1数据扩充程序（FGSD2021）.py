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


#voc路径
img_dir = '/media/zf/E/Dataset/US_Navy_test'#r'E:\Dataset\US_Navy_test'
xml_dir = img_dir
outputfolder='/media/zf/E/Dataset/US_Navy_test_aug'#r'E:\Dataset\US_Navy_test_aug'
output_VOC_folder=r'E:\Dataset\U.S.Navy_aug_1'#存储新的VOCanno位置
file_ext='.jpg'

#img_dir = 'E:/Dataset/2019tianzhi/plane_ori/YSplane'
#xml_dir = img_dir
#outputfolder='E:/Dataset/2019tianzhi/YSplane/'
#output_VOC_folder='E:/Dataset/2019tianzhi/YSplane_VOC/'#存储新的VOCanno位置
#output_Rotate_folder='E:/Dataset/2019tianzhi/YSplane_VOC/Rotate'
#file_ext='.jpg'

[w_crop,h_crop]=[1333,800]  #这是要切割的图像尺寸  第一个是宽，第二个是高
overlap_ratio=1/5
outrange_ratio=1/6
ratios=[1]#设置在切割过程中缩放的倍数

skip_center=50 #表示如果机场有30以下的飞机，则正常都遍历，如果超过了30个，每增加一倍，取中心的步距增加一倍
angle_range = [np.pi/20,np.pi*19/20]#角度im_rotate用到的是角度制 旋转角度的范围

NAME_LABEL_MAP =  {
        '航母': 1,
        '黄蜂级': 2,
        '塔瓦拉级': 3,
        '蓝岭级': 4,
        '奥斯汀级': 5,
        '惠特贝岛级': 6,
        '圣安东尼奥级': 7,
        '新港级': 8,
        '提康德罗加级': 9,
        '阿利·伯克级': 10,
        '朱姆沃尔特级': 11,
        '佩里级': 12,
        '刘易斯和克拉克级': 13,
        '供应级': 14,
        '凯泽级': 15,
        '霍普级': 16,
        '仁慈级': 17,
        '先锋级': 18,
        '自由级': 19,
        '独立级': 20,
        '复仇者级': 21,
        '胜利级': 22,
        '潜艇':23,
        '其他':24
        }
def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict
LABEl_NAME_MAP = get_label_name_map()

##每一类别的飞机旋转的次数
angle_number_dict=  {
        '航母': 3,
        '黄蜂级': 3,
        '塔瓦拉级': 6,
        '蓝岭级': 60,
        '奥斯汀级': 3,
        '惠特贝岛级': 3,
        '圣安东尼奥级': 4,
        '新港级': 10,
        '提康德罗加级': 1,
        '阿利·伯克级': 1,
        '朱姆沃尔特级': 60,
        '佩里级': 1,
        '刘易斯和克拉克级': 6,
        '供应级': 36,
        '凯泽级': 6,
        '霍普级': 10,
        '仁慈级': 24,
        '先锋级': 100,
        '自由级': 10,
        '独立级': 10,
        '复仇者级': 6,
        '胜利级': 100,
        '潜艇':1,
        '其他':0
        } 
angle_num_multiple=2 

for name, aug_num in angle_number_dict.items():
    angle_number_dict[name] = aug_num*angle_num_multiple

    

#新建输出文件夹
if not os.path.isdir(outputfolder):
    os.makedirs(outputfolder)
#if not os.path.isdir(output_VOC_folder):
#    os.makedirs(output_VOC_folder)
#if not os.path.isdir(output_Rotate_folder):
#    os.makedirs(output_Rotate_folder)
    
#读取原图全路径  
imgs_path = DataFunction.get_file_paths_recursive(img_dir, file_ext) 
#旋转角的大小，整数表示逆时针旋转
imgs_total_num=len(imgs_path)
for num,img_path in enumerate(imgs_path,0):
    start = timer()
    #一、读取图像并获取基本参数
    img_path=imgs_path[num]
    img = cv2.imread(img_path)
    #16 to 8
#    min_pix=img.min()
#    max_pix=img.max()
#    img = (img /max_pix * 255).astype(np.uint8)
    img_size=img.shape#高，宽 ，通道
    img_num=1
    #二、读取标注文件
    [floder,name]=os.path.split(img_path)
    xml_path=os.path.join(xml_dir,os.path.splitext(name)[0]+'.xml')
    img_size,gsd,imagesource,gtbox,extra=DataFunction.read_rotate_xml(xml_path,NAME_LABEL_MAP)
    
#    #过滤其他类别
#    gtbox_label=[]
#    for i,box in enumerate(gtbox):
#        if box[5]>13:
#            continue
#        gtbox_label.append(box)
    
    gsd=0.5
#    #三、把单通道图像转为三通道
#    if img_size[2]==1:
#        img_new=np.zeros((img_size[0],img_size[1],3))
#        img_new[:,:,0]=img
#        img_new[:,:,1]=img
#        img_new[:,:,2]=img
#        img=img_new
#        img_size[2]=3
    if len(gtbox)>0:
#        
#         #四、将图片变为横向的图片
        if img_size[0]>img_size[1]:#说明图像是高度大于宽度，是竖直的图像
            r_center=(img_size[1]/2,img_size[0]/2)
            M = cv2.getRotationMatrix2D(r_center, -90, 1) #
            # compute the new bounding dimensions of the image
            nW = img_size[0]
            nH = img_size[1]
            # adjust the rotation matrix to take into account translation
            t_x=(nW / 2) - r_center[0]
            t_y=-t_x
            M[0, 2] += (nW / 2) - r_center[0]
            M[1, 2] += (nH / 2) - r_center[1]
            img = cv2.warpAffine(img, M, (img_size[0], img_size[1])) #6
            gtbox=DataFunction.rotate_xml_rotate(np.pi/2,(img_size[0]/2,img_size[1]/2),gtbox)
            gsd_scale,gtbox=DataFunction.rotate_xml_transform(-t_x,t_y,1,gsd,gtbox)
            img_size=[img_size[1],img_size[0],img_size[2]]

#        DataFunction.write_rotate_xml(outputfolder,img_path,img_size,gsd,imagesource,gtbox_label,LABEl_NAME_MAP)
#        img_filename=os.path.join(outputfolder,os.path.splitext(name)[0]+'.jpg')
#        cv2.imwrite(img_filename, img)
#        #五、对于标签框占比较大的图像进行缩小  以下程序是找出图像中所有标签的最大和最小尺寸
#        gtbox_label_array = np.array(gtbox_label) 
#        objects_Width=gtbox_label_array[:,2]
#        objects_Height=gtbox_label_array[:,3]
#        max_len=np.max([np.max(objects_Width),np.max(objects_Height)])   
#        min_len=np.min([np.max(objects_Width),np.max(objects_Height)])  
            
        #六、将需要处理图像切小，切掉多余的区域，思路，因为在旋转的时候，最多转的半径为最大的的切割尺寸，所有在那个范围之外的图像都切掉
#        #得到最大的旋转半径
#        r_max=w_crop/2+100
#        #得到中心，最大值，最小值点，
#        gtbox_label_array = np.array(gtbox_label) 
#        c_x,c_y = gtbox_label_array[:,0], gtbox_label_array[:,1]
#        c_xmin ,c_ymin, c_xmax, c_ymax=int(np.min(c_x)), int(np.min(c_y)), int(np.max(c_x)), int(np.max(c_y))
#        #确定切割位置
#        c_xmin ,c_ymin, c_xmax, c_ymax= int(max([c_xmin-r_max,0])),int(max([ c_ymin-r_max,0])) ,int(min([c_xmax+r_max,img_size[1]])), int(min([c_ymax+r_max,img_size[0]]))
#        if (c_xmin !=0 or c_ymin !=0 or c_xmax !=img_size[1] or c_ymax !=img_size[0]):
#            img=img[int(c_ymin):int(c_ymax),int(c_xmin):int(c_xmax),:]
#            img_size=[c_ymax-c_ymin,c_xmax-c_xmin ,3]
#            gsd_scale,gtbox_label=DataFunction.rotate_xml_transform(c_ymin,c_xmin,1,gsd,gtbox_label)
#            DataFunction.write_rotate_xml(outputfolder,img_path,img_size,gsd_scale,imagesource,gtbox_label,LABEl_NAME_MAP)
#            img_filename=os.path.join(outputfolder,os.path.splitext(name)[0]+'.jpg')
#            cv2.imwrite(img_filename, img)
        #第1种方法将gsd缩放到0.5
        if gsd>0:#说明读取到了gsd数据
            zoom_scale=gsd/1
            if zoom_scale!=1:
                img_size=[int(img_size[0]*zoom_scale),int(img_size[1]*zoom_scale),img_size[2]]
                img=cv2.resize(img,(img_size[1],img_size[0]))
                gsd_scale,gtbox=DataFunction.rotate_xml_transform(0,0,zoom_scale,gsd,gtbox)
#            DataFunction.write_rotate_xml(outputfolder,img_path,img_size,gsd_scale,imagesource,gtbox_label,LABEl_NAME_MAP)
#            img_filename=os.path.join(outputfolder,os.path.splitext(name)[0]+'.jpg')
#            cv2.imwrite(img_filename, img)
#            img_num+=1
#        else:
#             #第2种方法标签筛选如果标签尺寸超过所需分辨率1/4，则将图片和标签缩小两倍
##            if max_len>h_crop/3:
##                img_size=[int(img_size[0]/2),int(img_size[1]/2),img_size[2]]
##                img=cv2.resize(img,(img_size[1],img_size[0]))
##                gsd,gtbox_label=DataFunction.rotate_xml_transform(0,0,1/2,gsd,gtbox_label)
#             
#             #如果没有GSD数据，则将所有的标签大小归一化到 max_plane_len,min_plane_len=142,18之间
#            shrink_ratio=max_plane_len/max_len
#            enlargement_ratio=min_plane_len/min_len
#            if shrink_ratio<1:#表示最大的飞机尺寸大于140，要缩小
#                zoom_scale=shrink_ratio*0.9
#            elif enlargement_ratio>1:#表示最小的飞机尺寸小于18，要放大
#                zoom_scale=enlargement_ratio
#            else:
#                zoom_scale=1
#            img_size=[int(img_size[0]*zoom_scale),int(img_size[1]*zoom_scale),img_size[2]]
#            img=cv2.resize(img,(img_size[1],img_size[0]))
#            gsd_scale,gtbox_label=DataFunction.rotate_xml_transform(0,0,zoom_scale,gsd,gtbox_label)
                
#                
    ##图像切割
        img_num=DataFunction.crop_img_rotatexml(ratios,overlap_ratio,outrange_ratio,h_crop,w_crop,img_path,outputfolder,output_VOC_folder,img_num,img_size,img,gsd,imagesource,gtbox,LABEl_NAME_MAP)
    
##     #图像旋转切割
#        gtbox_label_array=np.array(gtbox) 
#        label_num=gtbox_label_array[:,5]
##        if  29 in label_num: #or 35 in label_num or  37 in label_num:
#        centers=gtbox_label_array[:,0:2].tolist()
#        labels=gtbox_label_array[:,5].tolist()
#        #这里做一个判断，如果一幅图像目标数量过多，则减少围绕旋转的中心数量
#        multiple=int(len(centers)/skip_center)+1
#        centers_new=centers[0:len(centers)+1:multiple]
#        labels_new=labels[0:len(centers)+1:multiple]
#        for i, center in enumerate (centers_new):
#            angle_number=angle_number_dict[LABEl_NAME_MAP[labels_new[i]]]
#            # angle_number=angle_number_class[int(labels_new[i])-1]
#
#            angles=np.linspace(angle_range[0],angle_range[1],num=angle_number)+i*2*np.pi/180#转为弧度
#            for ang in angles:
#                im_rot=DataFunction.im_rotate(img,-ang,center =(center[0],center[1]),scale=1.0)
#                gtbox_label_rotate=DataFunction.rotate_xml_rotate(ang,center,gtbox)
#                gtbox_label_rotate,rect_box=DataFunction.rotate_xml_valid(img_size[0],img_size[1],outrange_ratio,gtbox_label_rotate)
#                img_num=DataFunction.crop_img_rotatexml(ratios,overlap_ratio,outrange_ratio,h_crop,w_crop,img_path,outputfolder,output_VOC_folder,img_num,img_size,im_rot,gsd,imagesource,gtbox_label_rotate,LABEl_NAME_MAP,center,flag='plane')
##                                                        (ratios,overlap_ratio,outrange_ratio,h_crop,w_crop,img_path,outputfolder,output_VOC_folder,img_num,img_size,img,gsd,imagesource,gtbox_label,LABEl_NAME_MAP,center_select = None)
#                DataFunction.write_rotate_xml(outputfolder,img_path,img_size,gsd_scale,imagesource,gtbox_label_rotate)
#                img_filename=os.path.join(outputfolder,os.path.splitext(name)[0]+'.jpg')
#                cv2.imwrite(img_filename, im_rot)
    else :
        print('{} annotation is empty!'.format(img_path))
    time_elapsed = timer() - start
    print('{}/{}，time{}: augnum:{}'.format(num,imgs_total_num,time_elapsed,img_num))