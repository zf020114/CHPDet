from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math

class CTDetDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
        #TODO change
        # bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3],
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3],box[4]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
         #border 128  pic_len w or h
    i = 1
    while size - border // i <= border // i:
                # 如果图像宽高小于 boder*2，i增大，返回128 // i
      # 正常返回128，图像小于256，则返回64
        i *= 2
    return border // i

  def __getitem__(self, index):
       #函数为入口。这里我们可以得到我们输出参数，分别是\color{red}{inp, hm, reg\_mask, ind, wh}。
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)# 目标个数,这里为100

    img = cv2.imread(img_path) 
#接着我们获取图片的最长边以及输入尺寸(512,512)
    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)# 获取中心点
    if self.opt.keep_res:# False
      input_h = (height | self.opt.pad) + 1
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:# True
      s = max(img.shape[0], img.shape[1]) * 1.0# s最长的边长
      input_h, input_w = self.opt.input_h, self.opt.input_w# 512， 512
    
     #对数据进行一系列处理。最终输出结果即我们第一个所需要的输入图像\color{red}{inp}.
    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        s = s# * np.random.choice(np.arange(0.6, 1.4, 0.1))# 随机尺度
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      
      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1 # 随机裁剪
        

    trans_input = get_affine_transform(
      c, s, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, 
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)# 放射变换
    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

#接着我们需要完成我们的heatmap的生成。
    output_h = input_h // self.opt.down_ratio# 输出512//4=128
    output_w = input_w // self.opt.down_ratio
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)# heatmap(80,128,128)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)# 中心点宽高(100*2)
    angs = np.zeros((self.max_objs, 1), dtype=np.float32)# 角度(100*2)
    dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)# 返回2*128*128
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)# 记录下采样带来的误差,返回100*2的小数
    ind = np.zeros((self.max_objs), dtype=np.int64)# 返回100个ind
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)# 返回8个 回归mask
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32) # 100*80*2
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)# 100*80*2
    #这里mse_loss为False, 所以我们只需要关注draw_umich_gaussian函数即可
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])
      cls_id = int(self.cat_ids[ann['category_id']])
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        bbox[[4]] = 180 - bbox[[4]] 
      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:4] = affine_transform(bbox[2:4], trans_output)

      #这里是导致舰船检测过程中出现中心点偏移的关键
  
      #bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
      #bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      #TODO insert
      ang = bbox[4] 
      h = np.clip(h , 0,  output_h - 1)
      w = np.clip(w , 0, output_w - 1)
      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))#关键是如何确定高斯半径
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        if ct[0]<0 or ct[0]>output_w - 1 or ct[1]<0 or ct[1]>output_h - 1: #
                    continue
        # ct[0] = np.clip(ct[0], 0, output_w - 1)
        # ct[1] = np.clip(ct[1], 0, output_h - 1)
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[cls_id], ct_int, radius)
        #cv2.imwrite("/data/humaocheng/CenterNet-master/single_heatmap.jpg", hm[0]*255)
        wh[k] = 1. * w, 1. * h # 目标矩形框的宽高——目标尺寸损失
        angs[k]=1. * ang
        ind[k] = ct_int[1] * output_w + ct_int[0]# 目标中心点在128×128特征图中的索引
        reg[k] = ct - ct_int# off Loss, # ct 即 center point reg是偏置回归数组，存放每个中心店的偏置值 k是当前图中第k个目标
        # 实际例子为
        # [98.97667 2.3566666] - [98  2] = [0.97667, 0.3566666]
        reg_mask[k] = 1 #是记录我们前100个点，这里相当于记载一张图片存在哪些目标，
        #有的话对应索引设置为1，其余设置为0。
        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
        if self.opt.dense_wh:
          draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
        # gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
        #                ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
        #TODO insert
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                       ct[0] + w / 2, ct[1] + h / 2, ang, 1, cls_id])
     # cv2.imwrite("/data/humaocheng/CenterNet-master/heatmap.jpg",hm[0]*255)
    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'ang': angs}
    if self.opt.dense_wh:
      hm_a = hm.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif self.opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta
    return ret