import mmcv
import numpy as np
from mmcv import Config
import cv2
from mmdet.datasets import build_dataloader, build_dataset
import numpy as np
import matplotlib.pyplot as plt
import os
from DOTA import DOTA
import dota_utils as util
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 10.0)
import math
from matplotlib.collections import PatchCollection
from dota_poly2rbox import rbox2poly_single
from matplotlib.patches import Polygon, Circle


def showAnns( img,rboxes):
        plt.imshow(img)
        plt.axis('off')
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        circles = []
        r = 5
        for rbox in rboxes:
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            pts = rbox2poly_single(rbox)
            poly = [(pts[0], pts[1]), (pts[2], pts[3]), (pts[4], pts[5]), (pts[6], pts[7])]
            
            polygons.append(Polygon(poly))
            color.append(c)
            point = poly[0]
            circle = Circle((point[0], point[1]), r)
            circles.append(circle)
        p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)
        p = PatchCollection(circles, facecolors='red')
        ax.add_collection(p)
        plt.imshow(img)
        plt.show()
    
    
def run_datatloader(cfg):
    """
    可视化数据增强后的效果，同时也可以确认训练样本是否正确
    Args:
        cfg: 配置
    Returns:
    """
    # Build dataset
    dataset = build_dataset(cfg.data.train)
    # prepare data loaders
    data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist= False,
            shuffle=False)
    # rootdir=os.path.dirname(os.path.dirname(dataset.img_prefix))
    # example = DOTA(rootdir)
    # imgids = example.getImgIds()
    # len(imgids)
    # imgid=imgids[0]
    for i, data_batch in enumerate(data_loader):
        img_batch =data_batch['img'].data
        gt_label = data_batch['gt_labels'].data
        gt_box = data_batch['gt_bboxes'].data
        for batch_i in range(len(img_batch)):
            img = img_batch[batch_i]
            mean_value = np.array(cfg.img_norm_cfg['mean'])
            std_value = np.array(cfg.img_norm_cfg['std'])
            img_hwc = np.transpose(np.squeeze(img.numpy()), [1, 2, 0])
            img = (img_hwc * std_value) + mean_value
            img = np.array(img, np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # img_numpy_float = mmcv.imdenormalize(img_hwc, mean_value, std_value)
            img_numpy_uint8 = np.array(img, np.uint8)
            
            label = gt_label[batch_i]
            boxes = gt_box[batch_i][0].cpu().numpy()
            showAnns( img,boxes)
            # mmcv.imshow(img_numpy_uint8, 'img', 0)
 
 
if __name__ == '__main__':
    cfg = Config.fromfile('/home/zf/2020HJJ/train_worm/s2anet_r101_fpn_3x_hrsc2016.py')
    run_datatloader(cfg)