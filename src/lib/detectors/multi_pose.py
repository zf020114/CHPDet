from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import os
try:
  from external.nms import soft_nms_39
except:
  print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import multi_pose_decode
from models.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform
from utils.post_process import multi_pose_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector
# NAME_LABEL_MAP ={
#         'plane': 1,
#         'baseball-diamond': 2,
#         'bridge': 3,
#         'ground-track-field': 4,
#         'small-vehicle': 5,
#         'large-vehicle': 6,
#         'ship': 7,
#         'tennis-court': 8,
#         'basketball-court': 9,
#         'storage-tank': 10,
#         'soccer-ball-field': 11,
#         'roundabout': 12,
#         'harbor': 13,
#         'swimming-pool': 14,
#         'helicopter': 15,
#         'container-crane': 16
#         }
NAME_LABEL_MAP = {
        'Aircraft carriers': 1,
        'Wasp ': 2,
        'Tarawa ': 3,
        'Austin ': 4,
        'Whidbey Island ': 5,
        'San Antonio ': 6,
        'Newport ': 7,
        'Ticonderoga  ': 8,
        ' Burke ': 9,
        'Perry ': 10,
        'Lewis and Clark ': 11,
        'Supply ': 12,
        'Henry J. Kaiser ': 13,
        ' Hope ': 14,
        'Mercy ': 15,
        'Freedom ': 16,
        'Independence ': 17,
        'Avenger ': 18,
        'Submarine':19,
        'Other':20
        }
def get_label_name_map(NAME_LABEL_MAP):
  reverse_dict = {}
  for name, label in NAME_LABEL_MAP.items():
    reverse_dict[label] = name
  return reverse_dict
LABEL_NAME_MAP=get_label_name_map(NAME_LABEL_MAP)
  
class MultiPoseDetector(BaseDetector):
  def __init__(self, opt):
    super(MultiPoseDetector, self).__init__(opt)
    self.flip_idx = opt.flip_idx

  def process(self, images, return_time=False):
    with torch.no_grad():
      torch.cuda.synchronize()
      output = self.model(images)[-1]
      output['hm'] = output['hm'].sigmoid_()
      if self.opt.hm_hp and not self.opt.mse_loss:
        output['hm_hp'] = output['hm_hp'].sigmoid_()

      reg = output['reg'] if self.opt.reg_offset else None
      hm_hp = output['hm_hp'] if self.opt.hm_hp else None
      hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
      torch.cuda.synchronize()
      forward_time = time.time()
      
      if self.opt.flip_test:
        output['hm'] = (output['hm'][0:1] + flip_tensor(output['hm'][1:2])) / 2
        output['wh'] = (output['wh'][0:1] + flip_tensor(output['wh'][1:2])) / 2
        output['hps'] = (output['hps'][0:1] + 
          flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
        hm_hp = (hm_hp[0:1] + flip_lr(hm_hp[1:2], self.flip_idx)) / 2 \
                if hm_hp is not None else None
        reg = reg[0:1] if reg is not None else None
        hp_offset = hp_offset[0:1] if hp_offset is not None else None
      
      dets = multi_pose_decode(
        output['hm'], output['wh'], output['hps'],
        reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)

    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets = multi_pose_post_process(
      dets.copy(), [meta['c']], [meta['s']],
      meta['out_height'], meta['out_width'])
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 7)
      dets[0][j][:, :4] /= scale
      dets[0][j][:, 5:] /= scale
#    for j in range(1,  self.num_classes+1):# +
#        if len(dets[0][j])>1:
#          dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 39)
#          # import pdb; pdb.set_trace()
#          dets[0][j][:, :4] /= scale
#          dets[0][j][:, 5:] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    for i in range(self.num_classes ):
        results[i+1] = np.concatenate(
            [detection[i+1] for detection in detections], axis=0).astype(np.float32)
    
        if self.opt.nms or len(self.opt.test_scales) > 1:
          soft_nms_39(results[i+1], Nt=0.8, method=2)
        results[i+1] = results[i+1].tolist()
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    dets = dets.detach().cpu().numpy().copy()
    dets[:, :, :4] *= self.opt.down_ratio
    dets[:, :, 5:7] *= self.opt.down_ratio#39
    img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(((
      img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
    pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
    debugger.add_blend_img(img, pred, 'pred_hm')
    if self.opt.hm_hp:
      pred = debugger.gen_colormap_hp(
        output['hm_hp'][0].detach().cpu().numpy()) 
      debugger.add_blend_img(img, pred, 'pred_hmhp')
      
  
  def show_results(self, debugger, image, results,image_path=None):
    debugger.add_img(image, img_id='multi_pose')
    for i in range(self.opt.num_classes):#change24
        for bbox in results[i+1]:
          if bbox[4] > self.opt.vis_thresh:
            # debugger.add_coco_bbox(bbox[:4], 0, bbox[4], img_id='multi_pose')
            
            debugger.add_coco_rbox(bbox,bbox[4], img_id='multi_pose',label_name=LABEL_NAME_MAP[i+1])
            debugger.add_coco_hp(bbox[5:7], img_id='multi_pose')#      
    debugger.show_all_imgs(pause=self.pause)
    # prefix=os.path.splitext(os.path.split(image_path)[1])[0]
    debugger.save_all_imgs(prefix='')