from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from .ddd_utils import compute_box_3d, project_to_image, draw_box_3d

class Debugger(object):
  def __init__(self, ipynb=False, theme='black', 
               num_classes=-1, dataset=None, down_ratio=4):
    self.ipynb = ipynb
    if not self.ipynb:
      import matplotlib.pyplot as plt
      self.plt = plt
    self.imgs = {}
    self.theme = theme
    colors = [(color_list[_]).astype(np.uint8) \
            for _ in range(len(color_list))]
    self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
    if self.theme == 'white':
      self.colors = self.colors.reshape(-1)[::-1].reshape(len(colors), 1, 1, 3)
      self.colors = np.clip(self.colors, 0., 0.6 * 255).astype(np.uint8)
    self.dim_scale = 1
    if dataset == 'coco_hp':
      self.names = ['p']
      self.num_class =16
      self.num_joints = 1
#      self.num_class = 1
#      self.num_joints = 17
      self.edges = [[0, 1], [0, 2], [1, 3], [2, 4], 
                    [3, 5], [4, 6], [5, 6], 
                    [5, 7], [7, 9], [6, 8], [8, 10], 
                    [5, 11], [6, 12], [11, 12], 
                    [11, 13], [13, 15], [12, 14], [14, 16]]
      self.ec = [(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), 
                 (255, 0, 0), (0, 0, 255), (255, 0, 255),
                 (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255),
                 (255, 0, 0), (0, 0, 255), (255, 0, 255),
                 (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)]
      self.colors_hp = [(255, 0, 255), (255, 0, 0), (0, 0, 255), 
        (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
        (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
        (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
        (255, 0, 0), (0, 0, 255)]
    elif num_classes == 80 or dataset == 'coco':
      self.names = coco_class_name
    elif num_classes == 20 or dataset == 'pascal':
      self.names = pascal_class_name
    elif dataset == 'gta':
      self.names = gta_class_name
      self.focal_length = 935.3074360871937
      self.W = 1920
      self.H = 1080
      self.dim_scale = 3
    elif dataset == 'viper':
      self.names = gta_class_name
      self.focal_length = 1158
      self.W = 1920
      self.H = 1080
      self.dim_scale = 3
    elif num_classes == 3 or dataset == 'kitti':
      self.names = kitti_class_name
      self.focal_length = 721.5377
      self.W = 1242
      self.H = 375
    num_classes = len(self.names)
    self.down_ratio=down_ratio
    # for bird view
    self.world_size = 64
    self.out_size = 384

  def add_img(self, img, img_id='default', revert_color=False):
    if revert_color:
      img = 255 - img
    self.imgs[img_id] = img.copy()
  
  def add_mask(self, mask, bg, imgId = 'default', trans = 0.8):
    self.imgs[imgId] = (mask.reshape(
      mask.shape[0], mask.shape[1], 1) * 255 * trans + \
      bg * (1 - trans)).astype(np.uint8)
  
  def show_img(self, pause = False, imgId = 'default'):
    cv2.imshow('{}'.format(imgId), self.imgs[imgId])
    if pause:
      cv2.waitKey()
  
  def add_blend_img(self, back, fore, img_id='blend', trans=0.7):
    if self.theme == 'white':
      fore = 255 - fore
    if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
      fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
    if len(fore.shape) == 2:
      fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
    self.imgs[img_id] = (back * (1. - trans) + fore * trans)
    self.imgs[img_id][self.imgs[img_id] > 255] = 255
    self.imgs[img_id][self.imgs[img_id] < 0] = 0
    self.imgs[img_id] = self.imgs[img_id].astype(np.uint8).copy()

  '''
  # slow version
  def gen_colormap(self, img, output_res=None):
    # num_classes = len(self.colors)
    img[img < 0] = 0
    h, w = img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    color_map = np.zeros((output_res[0], output_res[1], 3), dtype=np.uint8)
    for i in range(img.shape[0]):
      resized = cv2.resize(img[i], (output_res[1], output_res[0]))
      resized = resized.reshape(output_res[0], output_res[1], 1)
      cl = self.colors[i] if not (self.theme == 'white') \
           else 255 - self.colors[i]
      color_map = np.maximum(color_map, (resized * cl).astype(np.uint8))
    return color_map
    '''

  
  def gen_colormap(self, img, output_res=None):
    img = img.copy()
    c, h, w = img.shape[0], img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
    colors = np.array(
      self.colors, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
    if self.theme == 'white':
      colors = 255 - colors
    color_map = (img * colors).max(axis=2).astype(np.uint8)
    color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
    return color_map
    
  '''
  # slow
  def gen_colormap_hp(self, img, output_res=None):
    # num_classes = len(self.colors)
    # img[img < 0] = 0
    h, w = img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    color_map = np.zeros((output_res[0], output_res[1], 3), dtype=np.uint8)
    for i in range(img.shape[0]):
      resized = cv2.resize(img[i], (output_res[1], output_res[0]))
      resized = resized.reshape(output_res[0], output_res[1], 1)
      cl =  self.colors_hp[i] if not (self.theme == 'white') else \
        (255 - np.array(self.colors_hp[i]))
      color_map = np.maximum(color_map, (resized * cl).astype(np.uint8))
    return color_map
  '''
  def gen_colormap_hp(self, img, output_res=None):
    c, h, w = img.shape[0], img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
    colors = np.array(
      self.colors_hp, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
    if self.theme == 'white':
      colors = 255 - colors
    color_map = (img * colors).max(axis=2).astype(np.uint8)
    color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
    return color_map


  def add_rect(self, rect1, rect2, c, conf=1, img_id='default'): 
    cv2.rectangle(
      self.imgs[img_id], (rect1[0], rect1[1]), (rect2[0], rect2[1]), c, 2)
    if conf < 1:
      cv2.circle(self.imgs[img_id], (rect1[0], rect1[1]), int(10 * conf), c, 1)
      cv2.circle(self.imgs[img_id], (rect2[0], rect2[1]), int(10 * conf), c, 1)
      cv2.circle(self.imgs[img_id], (rect1[0], rect2[1]), int(10 * conf), c, 1)
      cv2.circle(self.imgs[img_id], (rect2[0], rect1[1]), int(10 * conf), c, 1)

  def add_coco_bbox(self, bbox, cat, conf=1, show_txt=True, img_id='default'): 
    bbox = np.array(bbox, dtype=np.int32)
    # cat = (int(cat) + 1) % 80
    cat = int(cat)
    # print('cat', cat, self.names[cat])
    c = self.colors[cat][0][0].tolist()
    if self.theme == 'white':
      c = (255 - np.array(c)).tolist()
    txt = '{}{:.1f}'.format(self.names[cat], conf)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    cv2.rectangle(
      self.imgs[img_id], (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
    if show_txt:
      cv2.rectangle(self.imgs[img_id],
                    (bbox[0], bbox[1] - cat_size[1] - 2),
                    (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
      cv2.putText(self.imgs[img_id], txt, (bbox[0], bbox[1] - 2), 
                  font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
      
  def result2rotatebox_cvbox(self, result):
    cx, cy, w,h=(result[:,0]+result[:,2])/2 ,(result[:,1]+result[:,3])/2 ,(result[:,2]-result[:,0]) ,(result[:,3]-result[:,1])
    head=result[:,5:7]#头部坐标
    center=np.vstack((cx,cy)).T#中心坐标
    det=head-center
    Angle=np.zeros_like(cx)
    for i in range(det.shape[0]):
        if det[i,0]==0:
            if det[i,1]>0:
                Angle[i]=np.pi/2
            else:
                Angle[i]=-np.pi/2
        elif det[i,0]<0:
            Angle[i]=np.arctan(det[i,1]/det[i,0])+np.pi*3/2
        else:
            Angle[i]=np.arctan(det[i,1]/det[i,0])+np.pi/2
    rotateboxes=np.vstack((cx, cy, w,h,Angle,result[:,4])).T
    center_head=np.sqrt(det[:,0]*det[:,0]+det[:,1]*det[:,1])
    keep=center_head>h/3
    rotateboxes=rotateboxes[keep]
    # show_rotate_box(src_img,rotateboxes)
    #   在opencv中，坐标系原点在左上角，相对于x轴，逆时针旋转角度为负，顺时针旋转角度为正。所以，θ∈（-90度，0]。
    #rotatexml  start normal y  shunp_cv
    p_cv=[]
    cvboxes=np.zeros_like(rotateboxes)
    for i in range(rotateboxes.shape[0]):
        angle_cv=rotateboxes[i,4]/np.pi*180
        if angle_cv>0:
            angle_cv=angle_cv
            cv_h=rotateboxes[i,3]
            cv_w=rotateboxes[i,2]
        else:
            angle_cv=angle_cv+90
            cv_h=rotateboxes[i,2]
            cv_w=rotateboxes[i,3]
        cvboxes[i,:]=[rotateboxes[i,0],rotateboxes[i,1],cv_w,cv_h,angle_cv,result[i,4]]
    return cvboxes,rotateboxes
  
  def show_rotate_box(self,src_img,rotateboxes     ,c       ):
    def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20):
        dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
        pts= []
        for i in  np.arange(0,dist,gap):
            r=i/dist
            x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
            y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
            p = (x,y)
            pts.append(p)

        if style=='dotted':
            for p in pts:
                cv2.circle(img,p,thickness,color,-1)
        else:
            s=pts[0]
            e=pts[0]
            i=0
            for p in pts:
                s=e
                e=p
                if i%2==1:
                    cv2.line(img,s,e,color,thickness)
                i+=1
                
    cx, cy, w,h,Angle=rotateboxes[:,0], rotateboxes[:,1], rotateboxes[:,2], rotateboxes[:,3], rotateboxes[:,4]
    p_rotate=[]
    for i in range(rotateboxes.shape[0]):
        RotateMatrix=np.array([
                              [np.cos(Angle[i]),-np.sin(Angle[i])],
                              [np.sin(Angle[i]),np.cos(Angle[i])]])
        # rhead,r1,r2,r3,r4=np.transpose([0,-h/2]),np.transpose([-w[i]/2,-h[i]/2+w[i]]),np.transpose([w[i]/2,-h[i]/2+w[i]]),np.transpose([w[i]/2,h[i]/2]),np.transpose([-w[i]/2,h[i]/2])
        rhead,r1,r2,r3,r4=np.transpose([0,-h/2]),np.transpose([-w[i]/2,-h[i]/2]),np.transpose([w[i]/2,-h[i]/2]),np.transpose([w[i]/2,h[i]/2]),np.transpose([-w[i]/2,h[i]/2])
        # rhead_out=np.transpose([0,-h/2-w])
        # rhead_out=np.transpose(np.dot(RotateMatrix, rhead_out))+[cx[i],cy[i]]
        rhead=np.transpose(np.dot(RotateMatrix, rhead))+[cx[i],cy[i]]
        p1=np.transpose(np.dot(RotateMatrix, r1))+[cx[i],cy[i]]
        p2=np.transpose(np.dot(RotateMatrix, r2))+[cx[i],cy[i]]
        p3=np.transpose(np.dot(RotateMatrix, r3))+[cx[i],cy[i]]
        p4=np.transpose(np.dot(RotateMatrix, r4))+[cx[i],cy[i]]
        p_rotate_=np.int32(np.vstack((p1,rhead,p2,p3,p4)))
        p_rotate.append(p_rotate_)
        # drawline(src_img,(int(rhead_out[0]), int(rhead_out[1])), (int(p1[0]),int(p1[1])), c,thickness=2,style='dotted',gap=7)
        # drawline(src_img,(int(rhead_out[0]), int(rhead_out[1])), (int(p2[0]),int(p2[1])), c,thickness=2,style='dotted',gap=7)
    
    cv2.polylines( src_img, np.array(p_rotate), True, c, 2)
    # cv2.line(src_img,(int(rhead_out[0]), int(rhead_out[1])), (int(p1[0]),int(p1[1])), c, 2)
    # cv2.line(src_img,(int(rhead_out[0]), int(rhead_out[1])), (int(p2[0]),int(p2[1])), c, 2)
    
  def add_coco_rbox(self, bbox, conf=1, show_txt=True, img_id='default',label_name=''): 
    # bbox = np.array(bbox, dtype=np.int32)
    # cat = (int(cat) + 1) % 80
    cat =0# int(cat)
    # print('cat', cat, self.names[cat])
    c = self.colors[cat][0][0].tolist()
    if self.theme == 'white':
      c = (255 - np.array(c)).tolist()
    txt = '{}{:.1f}'.format(label_name, conf)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    _,rotateboxes=self.result2rotatebox_cvbox(np.array(bbox).reshape((1,-1)))

    # show_txt=True
    if show_txt:
      try:
        bboxint=[int(rotateboxes[0,0]),int(rotateboxes[0,1])]
      except:
          bboxint = np.array(bbox, dtype=np.int32)
      cv2.rectangle(self.imgs[img_id],
                    (bboxint[0], bboxint[1] - cat_size[1] - 2),
                    (bboxint[0] + cat_size[0], bboxint[1] - 2), c, -1)
      cv2.putText(self.imgs[img_id], txt, (bboxint[0], bboxint[1] - 2), 
                  font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    
    self.show_rotate_box( self.imgs[img_id],rotateboxes,c)
    if rotateboxes.size>0:
        cv2.circle(self.imgs[img_id],
                   (int(rotateboxes[ 0,0]), int(rotateboxes[0,1])), 3, (255,0,0), -1)
        cv2.circle(self.imgs[img_id],
                   (int(rotateboxes[ 0,0]), int(rotateboxes[0,1])), 5, (255,0,0), -1)
                #  (int(rotateboxes[ 0,0]), int(rotateboxes[0,1])), 3, self.colors_hp[2], -1)
    # cv2.rectangle(
    #   self.imgs[img_id], (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
    
  def add_coco_hp(self, points, img_id='default'): 
    points = np.array(points, dtype=np.int32).reshape(1, 2)#self.num_joints
    for j in range(self.num_joints):
      cv2.circle(self.imgs[img_id],
                 (points[j, 0], points[j, 1]), 3, (0 ,0 , 255), -1)
      cv2.circle(self.imgs[img_id],
                 (points[j, 0], points[j, 1]), 5, (0 ,0 , 255), -1)
                #  (points[j, 0], points[j, 1]), 3, self.colors_hp[j], -1)
#    for j, e in enumerate(self.edges):
#      if points[e].min() > 0:
#        cv2.line(self.imgs[img_id], (points[e[0], 0], points[e[0], 1]),
#                      (points[e[1], 0], points[e[1], 1]), self.ec[j], 2,
#                      lineType=cv2.LINE_AA)

  def add_points(self, points, img_id='default'):
    num_classes = len(points)
    # assert num_classes == len(self.colors)
    for i in range(num_classes):
      for j in range(len(points[i])):
        c = self.colors[i, 0, 0]
        cv2.circle(self.imgs[img_id], (points[i][j][0] * self.down_ratio, 
                                       points[i][j][1] * self.down_ratio),
                   5, (255, 255, 255), -1)
        cv2.circle(self.imgs[img_id], (points[i][j][0] * self.down_ratio,
                                       points[i][j][1] * self.down_ratio),
                   3, (int(c[0]), int(c[1]), int(c[2])), -1)

  def show_all_imgs(self, pause=False, time=0):
    if not self.ipynb:
      for i, v in self.imgs.items():
        cv2.imshow('{}'.format(i), v)
      if cv2.waitKey(0 if pause else 1) == 27:
        import sys
        sys.exit(0)
    else:
      self.ax = None
      nImgs = len(self.imgs)
      fig=self.plt.figure(figsize=(nImgs * 10,10))
      nCols = nImgs
      nRows = nImgs // nCols
      for i, (k, v) in enumerate(self.imgs.items()):
        fig.add_subplot(1, nImgs, i + 1)
        if len(v.shape) == 3: 
          self.plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
        else:
          self.plt.imshow(v)
      self.plt.show()

  def save_img(self, imgId='default', path='./cache/debug/'):
    cv2.imwrite(path + '{}.png'.format(imgId), self.imgs[imgId])
    
  def save_all_imgs(self, path='/home/zf/CenterNet/debug/', prefix='', genID=False):
    if genID:
      try:
        idx = int(np.loadtxt(path + '/id.txt'))
      except:
        idx = 0
      prefix=idx
      np.savetxt(path + '/id.txt', np.ones(1) * (idx + 1), fmt='%d')
    for i, v in self.imgs.items():
      cv2.imwrite(path + '/{}{}.png'.format(prefix, i), v)

  def remove_side(self, img_id, img):
    if not (img_id in self.imgs):
      return
    ws = img.sum(axis=2).sum(axis=0)
    l = 0
    while ws[l] == 0 and l < len(ws):
      l+= 1
    r = ws.shape[0] - 1
    while ws[r] == 0 and r > 0:
      r -= 1
    hs = img.sum(axis=2).sum(axis=1)
    t = 0
    while hs[t] == 0 and t < len(hs):
      t += 1
    b = hs.shape[0] - 1
    while hs[b] == 0 and b > 0:
      b -= 1
    self.imgs[img_id] = self.imgs[img_id][t:b+1, l:r+1].copy()

  def project_3d_to_bird(self, pt):
    pt[0] += self.world_size / 2
    pt[1] = self.world_size - pt[1]
    pt = pt * self.out_size / self.world_size
    return pt.astype(np.int32)

  def add_ct_detection(
    self, img, dets, show_box=False, show_txt=True, 
    center_thresh=0.5, img_id='det'):
    # dets: max_preds x 5
    self.imgs[img_id] = img.copy()
    if type(dets) == type({}):
      for cat in dets:
        for i in range(len(dets[cat])):
          if dets[cat][i, 2] > center_thresh:
            cl = (self.colors[cat, 0, 0]).tolist()
            ct = dets[cat][i, :2].astype(np.int32)
            if show_box:
              w, h = dets[cat][i, -2], dets[cat][i, -1]
              x, y = dets[cat][i, 0], dets[cat][i, 1]
              bbox = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2],
                              dtype=np.float32)
              self.add_coco_bbox(
                bbox, cat - 1, dets[cat][i, 2], 
                show_txt=show_txt, img_id=img_id)
    else:
      for i in range(len(dets)):
        if dets[i, 2] > center_thresh:
          # print('dets', dets[i])
          cat = int(dets[i, -1])
          cl = (self.colors[cat, 0, 0] if self.theme == 'black' else \
                                       255 - self.colors[cat, 0, 0]).tolist()
          ct = dets[i, :2].astype(np.int32) * self.down_ratio
          cv2.circle(self.imgs[img_id], (ct[0], ct[1]), 3, cl, -1)
          if show_box:
            w, h = dets[i, -3] * self.down_ratio, dets[i, -2] * self.down_ratio
            x, y = dets[i, 0] * self.down_ratio, dets[i, 1] * self.down_ratio
            bbox = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2],
                            dtype=np.float32)
            self.add_coco_bbox(bbox, dets[i, -1], dets[i, 2], img_id=img_id)


  def add_3d_detection(
    self, image_or_path, dets, calib, show_txt=False, 
    center_thresh=0.5, img_id='det'):
    if isinstance(image_or_path, np.ndarray):
      self.imgs[img_id] = image_or_path
    else: 
      self.imgs[img_id] = cv2.imread(image_or_path)
    for cat in dets:
      for i in range(len(dets[cat])):
        cl = (self.colors[cat - 1, 0, 0]).tolist()
        if dets[cat][i, -1] > center_thresh:
          dim = dets[cat][i, 5:8]
          loc  = dets[cat][i, 8:11]
          rot_y = dets[cat][i, 11]
          # loc[1] = loc[1] - dim[0] / 2 + dim[0] / 2 / self.dim_scale
          # dim = dim / self.dim_scale
          if loc[2] > 1:
            box_3d = compute_box_3d(dim, loc, rot_y)
            box_2d = project_to_image(box_3d, calib)
            self.imgs[img_id] = draw_box_3d(self.imgs[img_id], box_2d, cl)

  def compose_vis_add(
    self, img_path, dets, calib,
    center_thresh, pred, bev, img_id='out'):
    self.imgs[img_id] = cv2.imread(img_path)
    # h, w = self.imgs[img_id].shape[:2]
    # pred = cv2.resize(pred, (h, w))
    h, w = pred.shape[:2]
    hs, ws = self.imgs[img_id].shape[0] / h, self.imgs[img_id].shape[1] / w
    self.imgs[img_id] = cv2.resize(self.imgs[img_id], (w, h))
    self.add_blend_img(self.imgs[img_id], pred, img_id)
    for cat in dets:
      for i in range(len(dets[cat])):
        cl = (self.colors[cat - 1, 0, 0]).tolist()
        if dets[cat][i, -1] > center_thresh:
          dim = dets[cat][i, 5:8]
          loc  = dets[cat][i, 8:11]
          rot_y = dets[cat][i, 11]
          # loc[1] = loc[1] - dim[0] / 2 + dim[0] / 2 / self.dim_scale
          # dim = dim / self.dim_scale
          if loc[2] > 1:
            box_3d = compute_box_3d(dim, loc, rot_y)
            box_2d = project_to_image(box_3d, calib)
            box_2d[:, 0] /= hs
            box_2d[:, 1] /= ws
            self.imgs[img_id] = draw_box_3d(self.imgs[img_id], box_2d, cl)
    self.imgs[img_id] = np.concatenate(
      [self.imgs[img_id], self.imgs[bev]], axis=1)

  def add_2d_detection(
    self, img, dets, show_box=False, show_txt=True, 
    center_thresh=0.5, img_id='det'):
    self.imgs[img_id] = img
    for cat in dets:
      for i in range(len(dets[cat])):
        cl = (self.colors[cat - 1, 0, 0]).tolist()
        if dets[cat][i, -1] > center_thresh:
          bbox = dets[cat][i, 1:5]
          self.add_coco_bbox(
            bbox, cat - 1, dets[cat][i, -1], 
            show_txt=show_txt, img_id=img_id)

  def add_bird_view(self, dets, center_thresh=0.3, img_id='bird'):
    bird_view = np.ones((self.out_size, self.out_size, 3), dtype=np.uint8) * 230
    for cat in dets:
      cl = (self.colors[cat - 1, 0, 0]).tolist()
      lc = (250, 152, 12)
      for i in range(len(dets[cat])):
        if dets[cat][i, -1] > center_thresh:
          dim = dets[cat][i, 5:8]
          loc  = dets[cat][i, 8:11]
          rot_y = dets[cat][i, 11]
          rect = compute_box_3d(dim, loc, rot_y)[:4, [0, 2]]
          for k in range(4):
            rect[k] = self.project_3d_to_bird(rect[k])
            # cv2.circle(bird_view, (rect[k][0], rect[k][1]), 2, lc, -1)
          cv2.polylines(
              bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)],
              True,lc,2,lineType=cv2.LINE_AA)
          for e in [[0, 1]]:
            t = 4 if e == [0, 1] else 1
            cv2.line(bird_view, (rect[e[0]][0], rect[e[0]][1]),
                    (rect[e[1]][0], rect[e[1]][1]), lc, t,
                    lineType=cv2.LINE_AA)
    self.imgs[img_id] = bird_view

  def add_bird_views(self, dets_dt, dets_gt, center_thresh=0.3, img_id='bird'):
    alpha = 0.5
    bird_view = np.ones((self.out_size, self.out_size, 3), dtype=np.uint8) * 230
    for ii, (dets, lc, cc) in enumerate(
      [(dets_gt, (12, 49, 250), (0, 0, 255)), 
       (dets_dt, (250, 152, 12), (255, 0, 0))]):
      # cc = np.array(lc, dtype=np.uint8).reshape(1, 1, 3)
      for cat in dets:
        cl = (self.colors[cat - 1, 0, 0]).tolist()
        for i in range(len(dets[cat])):
          if dets[cat][i, -1] > center_thresh:
            dim = dets[cat][i, 5:8]
            loc  = dets[cat][i, 8:11]
            rot_y = dets[cat][i, 11]
            rect = compute_box_3d(dim, loc, rot_y)[:4, [0, 2]]
            for k in range(4):
              rect[k] = self.project_3d_to_bird(rect[k])
            if ii == 0:
              cv2.fillPoly(
                bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)],
                lc,lineType=cv2.LINE_AA)
            else:
              cv2.polylines(
                bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)],
                True,lc,2,lineType=cv2.LINE_AA)
            # for e in [[0, 1], [1, 2], [2, 3], [3, 0]]:
            for e in [[0, 1]]:
              t = 4 if e == [0, 1] else 1
              cv2.line(bird_view, (rect[e[0]][0], rect[e[0]][1]),
                      (rect[e[1]][0], rect[e[1]][1]), lc, t,
                      lineType=cv2.LINE_AA)
    self.imgs[img_id] = bird_view


kitti_class_name = [
  'p', 'v', 'b'
]

gta_class_name = [
  'p', 'v'
]

pascal_class_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
  "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
  "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

coco_class_name = [
     'person', 'bicycle', 'car', 'motorcycle', 'airplane',
     'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
     'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
     'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
     'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
     'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
     'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
     'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

color_list = np.array(
        [
            1.000, 1.000, 1.000,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255
