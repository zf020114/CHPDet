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
from utils.debugger import Debugger
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from opts import opts
opt = opts().parse()
class MultiPoseDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def debug(self,debugger, img,  output, scale=1):
    pred = debugger.gen_colormap(output['hm'])
    debugger.add_blend_img(img, pred, 'pred_hm')
    pred = debugger.gen_colormap_hp(output['hm_hp'])
    debugger.add_blend_img(img, pred, 'pred_hp')
    debugger.show_all_imgs()
  
  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)
    img = cv2.imread(img_path)
    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    s = max(img.shape[0], img.shape[1]) * 1.0
    rot = 0

    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
            #TODO这里是更改多尺度训练的地方。
        s = s#* np.random.choice(np.arange(0.8, 1.5, 0.1))#change 0.6 1.4
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
      if np.random.random() < self.opt.aug_rot:# roate aug
        rf = self.opt.rotate
        rot = np.clip(np.random.randn()*rf, -rf*2, rf*2)

      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1
  
    trans_input = get_affine_transform(
      c, s, rot, [self.opt.input_res, self.opt.input_res])
    # inp1 = cv2.warpAffine(img, trans_input, 
    #                      (self.opt.input_res, self.opt.input_res),
    #                      flags=cv2.INTER_LINEAR)
    inp = cv2.warpAffine(img, trans_input, 
                         (self.opt.input_res, self.opt.input_res),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:
          color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

    output_res = self.opt.output_res
    input_res = self.opt.input_res
    num_joints = self.num_joints
    
    if self.opt.angle_norm and self.split == 'train': 
      #首先是读取标注文件，获得中心点和头部点获得所有角度的集合 
      angle_list=[]
      gt_masks=[]
      box_sizes=[]
      for k in range(num_objs):
        ann = anns[k]
        bbox = self._coco_box_to_bbox(ann['bbox'])
        pts = np.array(ann['keypoints'][0:3], np.float32).reshape(num_joints, 3)#tmjx
        bbox[:2] = affine_transform(bbox[:2], trans_input)
        bbox[2:] = affine_transform(bbox[2:], trans_input)
        pts[0, :2] = affine_transform(pts[0, :2], trans_input)
        if flipped:
          bbox[[0, 2]] = width - bbox[[2, 0]] - 1
          pts[:, 0] = width - pts[:, 0] - 1
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        # if ct[0]<0 or ct[0]>input_res - 1 or ct[1]<0 or ct[1]>input_res - 1:
        #       continue
        angle= math.atan2((pts[0, 0] - ct[0]), (pts[0, 1] - ct[1]))
        angle_list.append(angle)
        gt_masks.append([ct[0],ct[1],pts[0, 0],pts[0, 1]])
        box_sizes.append([w,h])
        #下面这段代码求旋转的角度
      # if len(angle_list)==0:
      #       return
      angle_list=np.array(angle_list)%np.pi #首先归一化到np.pi
      angle_int=(angle_list// (np.pi/9)).astype('int')
      angle_b=np.bincount(angle_int)
      index_rot=np.argmax(angle_b)
      ind_rot=(angle_list>(index_rot)*np.pi/9) *  (angle_list<=(index_rot+1)*np.pi/9)
      angle_rot=np.average(angle_list[ind_rot])
      angle_img_rot=angle_rot*(-180)/np.pi
      #这段代码是旋转图像，和中间点特征图，关键点特征图
      matrix = cv2.getRotationMatrix2D(((input_res)/2.0,(input_res)/2.0),angle_img_rot,1)
      inp = cv2.warpAffine(inp,matrix,(self.opt.input_res,self.opt.input_res))
      gt_masks=np.array(gt_masks)
      cx, cy, hx, hy = np.split(  gt_masks, gt_masks.shape[-1], axis=-1)
      coordinates = np.stack([[cx, cy], [hx, hy]])  # [4, 2, nb_bbox, 1]
      # pad 1 to convert from format [x, y] to homogeneous
      # coordinates format [x, y, 1]
      coordinates = np.concatenate( (coordinates,  np.ones((2, 1, coordinates.shape[2], 1), coordinates.dtype)),  axis=1)  # [4, 3, nb_bbox, 1]
      coordinates = coordinates.transpose( (2, 0, 1, 3))  # [nb_bbox, 4, 3, 1]
      rotated_coords = np.matmul(matrix,  coordinates)  # [nb_bbox, 4, 2, 1]
      # rotated_coords = np.matmul(rotate_matrix,  coordinates)  # [nb_bbox, 4, 2, 1]
      rotated_coords = rotated_coords[..., 0].astype(np.float32)
      #结束
    
    hm = np.zeros((self.num_classes, output_res, output_res), dtype=np.float32)
    hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
    dense_kps = np.zeros((num_joints, 2, output_res, output_res), 
                          dtype=np.float32)
    dense_kps_mask = np.zeros((num_joints, output_res, output_res), 
                               dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)
    hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)
    hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)
    hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)

    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian
    gt_det = []
    #缩放相应倍数
    rotated_coords/=input_res/output_res
    box_sizes=np.array(box_sizes)
    box_sizes/=input_res/output_res
    num_objs=len(box_sizes)
    for k in range(num_objs):
      ann = anns[k]
      cls_id = int(ann['category_id']) - 1
      w,h = box_sizes[k]
      h = np.clip(h , 0,  output_res - 1)
      w = np.clip(w , 0, output_res - 1)
      
      if (h > 0 and w > 0) or (rot != 0):
        radius = gaussian_radius((math.ceil(h), math.ceil(w))) *1.2
        sqrt_wh = np.sqrt(np.sqrt(h*w))
        radius_w = radius * np.sqrt(w) / sqrt_wh
        radius_h = radius * np.sqrt(h) / sqrt_wh
        radius_w = self.opt.hm_gauss if self.opt.mse_loss else max(0, np.ceil(radius_w)) 
        radius_h = self.opt.hm_gauss if self.opt.mse_loss else max(0, np.ceil(radius_h)) 
        ct=rotated_coords[k][0]
        if ct[0]<0 or ct[0]>output_res - 1 or ct[1]<0 or ct[1]>output_res - 1:
              continue
        ct_int = ct.astype(np.int32)
        wh[k] = 1. * w, 1. * h
        ind[k] = ct_int[1] * output_res + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        num_kpts = 1#pts[:, 2].sum()
        if num_kpts == 0:
          hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
          reg_mask[k] = 0

        hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        hp_radius = self.opt.hm_gauss \
                    if self.opt.mse_loss else max(0, int(hp_radius)) 
        for j in range(num_joints):
          pts[j, :2]=rotated_coords[k][1]
          if pts[j, 0] >= 0 and pts[j, 0] < output_res and \
              pts[j, 1] >= 0 and pts[j, 1] < output_res:
            kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
            kps_mask[k, j * 2: j * 2 + 2] = 1
            pt_int = pts[j, :2].astype(np.int32)
            hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
            hp_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]
            hp_mask[k * num_joints + j] = 1
            if self.opt.dense_hp:
              # must be before draw center hm gaussian
              draw_dense_reg(dense_kps[j], hm[cls_id], ct_int, 
                              pts[j, :2] - ct_int, radius, is_offset=True)
              draw_gaussian(dense_kps_mask[j], ct_int, radius)
            draw_gaussian(hm_hp[j], pt_int, hp_radius)
        #TODO change
        angle= math.atan2((pts[0, 0] - ct[0]), (pts[0, 1] - ct[1]))
        # angle_list.append(angle)
        draw_gaussian(hm[cls_id], ct_int, [radius_w,radius_h,angle])
        # draw_gaussian(hm[cls_id], ct_int, radiusw,radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                       ct[0] + w / 2, ct[1] + h / 2, 1] + 
                       pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])
    if rot != 0:
      hm = hm * 0 + 0.9999
      reg_mask *= 0
      kps_mask *= 0
      
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)
    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
           'hps': kps, 'hps_mask': kps_mask}
    
    if self.opt.dense_hp:
      dense_kps = dense_kps.reshape(num_joints * 2, output_res, output_res)
      dense_kps_mask = dense_kps_mask.reshape(
        num_joints, 1, output_res, output_res)
      dense_kps_mask = np.concatenate([dense_kps_mask, dense_kps_mask], axis=1)
      dense_kps_mask = dense_kps_mask.reshape(
        num_joints * 2, output_res, output_res)
      ret.update({'dense_hps': dense_kps, 'dense_hps_mask': dense_kps_mask})
      del ret['hps'], ret['hps_mask']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.hm_hp:
      ret.update({'hm_hp': hm_hp})
    if self.opt.reg_hp_offset:
      ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 40), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta
#这里是调试可视化生成的特征图的程序
    # debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
    #                     theme=self.opt.debugger_theme)
    # inp1 = inp.transpose(1,2,0)
    # inp1=(inp1*self.std + self.mean)*255.
    # self.debug(debugger, inp1,  ret)
    return ret
