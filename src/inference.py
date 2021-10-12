import sys
sys.path.insert(0, '/home/zf/CenterNet')
from lib.detectors.detector_factory import detector_factory
from opts import opts
import glob
import numpy as np
from utils.post_process import multi_pose_post_process
try:
  from external.nms import soft_nms_39
except:
  print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
  
  

MODEL_PATH = '../models/multi_pose_dla__52_1024*1024.pth'
TASK = 'multi_pose' # or 'ctdet' for human pose estimation
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
opt.debug = max(opt.debug, 1)
opt.vis_thresh=0.2
detector = detector_factory[opt.task](opt)
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
img_path='../images'
imgs = glob.glob(img_path+"/*.jpg")
for i, img in enumerate(imgs):
    ret = detector.run(img)
    time_str = ''
    for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
    print(time_str)
