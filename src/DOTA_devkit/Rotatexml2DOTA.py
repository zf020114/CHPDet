import json
import os
import os.path as osp
import numpy as np
import xmltodict
import cv2
from dota_poly2rbox import rbox2poly_single


def parse_ann_info(objects):
    bboxes, labels, bboxes_ignore, labels_ignore = [], [], [], []
    # only one annotation
    if type(objects) != list:
        objects = [objects]
    for obj in objects:
        if obj['difficult'] == '0':
            bbox = float(obj['mbox_cx']), float(obj['mbox_cy']), float(
                obj['mbox_w']), float(obj['mbox_h']), float(obj['mbox_ang'])
            label = 'ship'
            bboxes.append(bbox)
            labels.append(label)
        elif obj['difficult'] == '1':
            bbox = float(obj['mbox_cx']), float(obj['mbox_cy']), float(
                obj['mbox_w']), float(obj['mbox_h']), float(obj['mbox_ang'])
            label = 'ship'
            bboxes_ignore.append(bbox)
            labels_ignore.append(label)
    return bboxes, labels, bboxes_ignore, labels_ignore

def parse_rotatexml_ann_info(objects):
    bboxes, labels, bboxes_ignore, labels_ignore = [], [], [], []
    # only one annotation
    if type(objects) != list:
        objects = [objects]
    for obj in objects:
        if obj['difficult'] == '0':
            bbox = float(obj['robndbox']['cx']), float(obj['robndbox']['cy']), float(
                obj['robndbox']['w']), float(obj['robndbox']['h']), float(obj['robndbox']['angle'])
            label =obj['name']
            bboxes.append(bbox)
            labels.append(label)
        elif obj['difficult'] == '1':
            bbox = float(obj['robndbox']['cx']), float(obj['robndbox']['cy']), float(
                obj['robndbox']['w']), float(obj['robndbox']['h']), float(obj['robndbox']['angle'])
            label =obj['name']
            bboxes.append(bbox)
            labels.append(label)
            bboxes_ignore.append(bbox)
            labels_ignore.append(label)
    return bboxes, labels, bboxes_ignore, labels_ignore

def ann_to_txt(ann):
    out_str = ''
    for bbox, label in zip(ann['bboxes'], ann['labels']):
        poly = rbox2poly_single(bbox)
        str_line = '{} {} {} {} {} {} {} {} {} {}\n'.format(
            poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7], label, '0')
        out_str += str_line
    for bbox, label in zip(ann['bboxes_ignore'], ann['labels_ignore']):
        poly = rbox2poly_single(bbox)
        str_line = '{} {} {} {} {} {} {} {} {} {}\n'.format(
            poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7], label, '1')
        out_str += str_line
    return out_str
    

def drow_box_on_image(img_name,bboxes):
    img= cv2.imdecode(np.fromfile(img_name,dtype=np.uint8),-1)
    for box in bboxes:
        box = rbox2poly_single(box)
        box=np.int32(np.array(box).reshape((-1,2)))
        cv2.polylines(img,[box],True,(0,255,255))
        pts=box
        cv2.circle(img, (pts[0][0],pts[0][1]), 2, (0, 0, 255), 0)
        cv2.circle(img, (pts[1][0],pts[1][1]), 4, (0, 0, 255), 0)
        cv2.circle(img, (pts[2][0],pts[2][1]), 6, (0, 0, 255), 0)
        cv2.circle(img, (pts[3][0],pts[3][1]), 8, (0, 0, 255), 0)
        cv2.imwrite('1.jpg', img)
        # cv2.imshow('rotate_box',img.astype('uint8'))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite(os.path.join(outputfolder,name), img)
    

def generate_txt_labels_train(root_path,):
    img_path = osp.join(root_path, 'train2017')
    label_path = osp.join(root_path, 'rotatexml')
    label_txt_path = osp.join(root_path, 'labelTxt')
    if not osp.exists(label_txt_path):
        os.mkdir(label_txt_path)

    img_names = [osp.splitext(img_name.strip())[0] for img_name in os.listdir(img_path)]
    for img_name in img_names:
        print(img_name)
        label = osp.join(label_path, img_name+'.xml')
        label_txt = osp.join(label_txt_path, img_name+'.txt')
        f_label = open(label)
        data_dict = xmltodict.parse(f_label.read())
        data_dict = data_dict['annotation']
        f_label.close()
        label_txt_str = ''
        # with annotations
        if len(data_dict['object'])>0:
            objects = data_dict['object']
            bboxes, labels, bboxes_ignore, labels_ignore = parse_rotatexml_ann_info(
                objects)
            ann = dict(
                bboxes=bboxes,
                labels=labels,
                bboxes_ignore=bboxes_ignore,
                labels_ignore=labels_ignore)
            label_txt_str = ann_to_txt(ann)
            imgfile_path=osp.join(img_path,img_name+'.jpg')
            # drow_box_on_image(imgfile_path,bboxes)
        with open(label_txt,'w') as f_txt:
            f_txt.write(label_txt_str)
            
def generate_txt_labels_val(root_path):
    img_path = osp.join(root_path, 'val2017')
    label_path = osp.join(root_path, 'rotatexml_val')
    label_txt_path = osp.join(root_path, 'labelTxt_val')
    if not osp.exists(label_txt_path):
        os.mkdir(label_txt_path)

    img_names = [osp.splitext(img_name.strip())[0] for img_name in os.listdir(img_path)]
    for img_name in img_names:
        label = osp.join(label_path, img_name+'.xml')
        label_txt = osp.join(label_txt_path, img_name+'.txt')
        f_label = open(label)
        data_dict = xmltodict.parse(f_label.read())
        data_dict = data_dict['annotation']
        f_label.close()
        label_txt_str = ''
        # with annotations
        if 'object' in data_dict.keys():
            if len(data_dict['object'])>0:
                objects = data_dict['object']
                bboxes, labels, bboxes_ignore, labels_ignore = parse_rotatexml_ann_info(
                    objects)
                ann = dict(
                    bboxes=bboxes,
                    labels=labels,
                    bboxes_ignore=bboxes_ignore,
                    labels_ignore=labels_ignore)
                label_txt_str = ann_to_txt(ann)
                imgfile_path=osp.join(img_path,img_name+'.jpg')
                # drow_box_on_image(imgfile_path,bboxes)
        with open(label_txt,'w') as f_txt:
            f_txt.write(label_txt_str)

def generate_txt_labels(img_path,label_path,label_txt_path):
    # img_path = osp.join(root_path, 'val2017')
    # label_path = osp.join(root_path, 'rotatexml_val')
    # label_txt_path = osp.join(root_path, 'labelTxt_val')
    if not osp.exists(label_txt_path):
        os.mkdir(label_txt_path)

    img_names = [osp.splitext(img_name.strip())[0] for img_name in os.listdir(img_path)]
    for img_name in img_names:
        label = osp.join(label_path, img_name+'.xml')
        label_txt = osp.join(label_txt_path, img_name+'.txt')
        f_label = open(label)
        data_dict = xmltodict.parse(f_label.read())
        data_dict = data_dict['annotation']
        f_label.close()
        label_txt_str = ''
        # with annotations
        if 'object' in data_dict.keys():
            if len(data_dict['object'])>0:
                objects = data_dict['object']
                bboxes, labels, bboxes_ignore, labels_ignore = parse_rotatexml_ann_info(
                    objects)
                ann = dict(
                    bboxes=bboxes,
                    labels=labels,
                    bboxes_ignore=bboxes_ignore,
                    labels_ignore=labels_ignore)
                label_txt_str = ann_to_txt(ann)
                # imgfile_path=osp.join(img_path,img_name+'.jpg')
                # drow_box_on_image(imgfile_path,bboxes)
        with open(label_txt,'w') as f_txt:
            f_txt.write(label_txt_str)
            
if __name__ == '__main__':
    # img_path = '/media/zf/E/Dataset/US_Navy_train_square/val2017'
    # label_path = '/home/zf/Dataset/USnavy_test_gt/rotatexml'
    # label_txt_path ='/home/zf/Dataset/USnavy_test_gt/labelTxt_val'
    img_path = '/media/zf/E/Dataset/US_Navy_train_square/val2017'
    label_path = '/media/zf/E/Dataset/US_Navy_train_square/rotatexml_val'
    label_txt_path ='/media/zf/E/Dataset/US_Navy_train_square/labelTxt_val'

    generate_txt_labels(img_path,label_path,label_txt_path)
    # generate_txt_labels('/project/jmhan/data/HRSC2016/Test')
    print('done!')
