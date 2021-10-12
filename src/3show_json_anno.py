#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:34:41 2019

@author: zf
"""

# -*- coding:utf-8 -*-

from __future__ import print_function
from pycocotools.coco import COCO
import os, sys, zipfile
import urllib.request
import shutil
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import matplotlib
import json
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
matplotlib.use('TkAgg')

datadir='/home/zf/E/ship/all_10'
data_set='train'
ana_set='person_keypoints'#'instances'person_keypoints
index=500
annFile=os.path.join(datadir,'annotations/{}_{}2017.json'.format(ana_set,data_set))
folder=os.path.join(datadir,'{}2017'.format(data_set))
jpgdir=os.path.join(datadir,'{}')
one_json_file=os.path.join(datadir,'one_{}_{}2017.json'.format(ana_set,data_set))

coco=COCO(annFile)
json_file=annFile # # Object Instance 类型的标注
#json_file='D:/dataset/COCO/annotations/instances_train2017.json' # # Object Instance 类型的标注
#json_file='/home/zf/dataset/HRSC/HRSC_train_mask.json'
#json_file='/home/zf/dataset/ZKXT_traindata/annotations/instances_train2017.json' # # Object Instance 类型的标注
# json_file='./annotations/person_keypoints_val2017.json'  # Object Keypoint 类型的标注格式
# json_file='./annotations/captions_val2017.json' # Image Caption的标注格式
data=json.load(open(json_file,'r'))
data_2={}
#data_2['info']=data['info']
#data_2['licenses']=data['licenses']
data_2['images']=[data['images'][index]] # 只提取第一张图片
data_2['categories']=data['categories']  # Image Caption 没有该字段
annotation=[]
# 通过imgID 找到其所有对象
imgID=data_2['images'][0]['id']
for ann in data['annotations']:
    if ann['image_id']==imgID:
        annotation.append(ann)

data_2['annotations']=annotation

# 保存到新的JSON文件，便于查看数据特点
json.dump(data_2,open(one_json_file,'w'),indent=4) # indent=4 更加美观显示
# json.dump(data_2,open('./new_person_keypoints_val2017.json','w'),indent=4) # indent=4 更加美观显示
# json.dump(data_2,open('./new_captions_val2017.json','w'),indent=4) # indent=4 更加美观显示
print('json down!')
# display COCO categories and supercategories

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# imgIds = coco.getImgIds(imgIds = [324158])
imgIds = coco.getImgIds()
img = coco.loadImgs(imgIds[index])[0]
#folder='/home/zf/dataset/data_convert_example/coco/'
I = io.imread(os.path.join(folder,img['file_name'].replace('xml','jpg')))
#I = io.imread('%s/%s/%s'%(dataDir,dataType,img['file_name']))

plt.axis('off')
plt.imshow(I)
plt.show()

# load and display instance annotations
# 加载实例掩膜
# catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
# catIds=coco.getCatIds()
catIds=[]
for ann in coco.dataset['annotations']:
    if ann['image_id']==imgIds[index]:
        catIds.append(ann['category_id'])

plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)


plt.savefig(jpgdir.format(img['file_name'].replace('xml','')))
