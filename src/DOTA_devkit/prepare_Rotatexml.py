import os
import os.path as osp
from DOTA_devkit.Rotatexml2DOTA import generate_txt_labels_train,generate_txt_labels_val
# from DOTA_devkit.DOTA2JSON import generate_json_labels
import json
import os
import os.path as osp
import random
from PIL import Image
from dota_poly2rbox import poly2rbox_single_v2
import shutil

def parse_ann_info(img_base_path, label_base_path, img_name):
    lab_path = osp.join(label_base_path, img_name+'.txt')
    bboxes, labels, bboxes_ignore, labels_ignore = [], [], [], []
    with open(lab_path, 'r') as f:
        for ann_line in f.readlines():
            ann_line = ann_line.strip().split(' ')
            bbox = [float(ann_line[i]) for i in range(8)]
            # 8 point to 5 point xywha
            bbox = poly2rbox_single_v2(bbox)
            class_name = ann_line[8]
            difficult = int(ann_line[9])
            # ignore difficult =2
            if difficult == 0:
                bboxes.append(bbox)
                labels.append(class_name)
            elif difficult == 1:
                bboxes_ignore.append(bbox)
                labels_ignore.append(class_name)
    return bboxes, labels, bboxes_ignore, labels_ignore


def generate_txt_labels(src_path, out_path, trainval=True):
    """Generate .txt labels recording img_names
    Args:
        src_path: dataset path containing images and labelTxt folders.
        out_path: output txt file path
        trainval: trainval or test?
    """
    img_path = os.path.join(src_path, 'images')
    label_path = os.path.join(src_path, 'labelTxt')
    img_lists = os.listdir(img_path)
    with open(out_path, 'w') as f:
        for img in img_lists:
            img_name = osp.splitext(img)[0]
            label = os.path.join(label_path, img_name+'.txt')
            if(trainval == True):
                if(os.path.exists(label) == False):
                    print('Label:'+img_name+'.txt'+' Not Exist')
                else:
                    f.write(img_name+'\n')
            else:
                f.write(img_name+'\n')


def generate_json_labels(src_path, out_path, trainval=True):
    """Generate .json labels which is similar to coco format
    Args:
        src_path: dataset path containing images and labelTxt folders.
        out_path: output json file path
        trainval: trainval or test?
    """
    img_path = os.path.join(src_path, 'images')
    label_path = os.path.join(src_path, 'labelTxt')
    img_lists = os.listdir(img_path)

    data_dict = []

    with open(out_path, 'w') as f:
        for id, img in enumerate(img_lists):
            img_info = {}
            img_name = osp.splitext(img)[0]
            label = os.path.join(label_path, img_name+'.txt')
            img = Image.open(osp.join(img_path, img))
            #TODO change
            # img_info['filename'] = img_name+'.png'
            img_info['filename'] = img_name+'.tif'
            img_info['height'] = img.height
            img_info['width'] = img.width
            img_info['id'] = id
            if(trainval == True):
                if(os.path.exists(label) == False):
                    print('Label:'+img_name+'.txt'+' Not Exist')
                else:
                    bboxes, labels, bboxes_ignore, labels_ignore = parse_ann_info(
                        img_path, label_path, img_name)
                    ann = {}
                    ann['bboxes'] = bboxes
                    ann['labels'] = labels
                    ann['bboxes_ignore'] = bboxes_ignore
                    ann['labels_ignore'] = labels_ignore
                    img_info['annotations'] = ann
            data_dict.append(img_info)
        json.dump(data_dict, f)



def preprare_rotatexml(data_dir):
    train_dir = osp.join(data_dir,'Train')
    test_dir = osp.join(data_dir, 'Test')
    if  not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if  not os.path.isdir(test_dir):
        os.mkdir(test_dir)
        
    ## convert hrsc2016 to dota raw format
    # generate_txt_labels_train(data_dir)
    # generate_txt_labels_val(data_dir)
    #move folder as HRSC2016
    # shutil.move(osp.join(data_dir,'labelTxt'),train_dir)
    # shutil.move(osp.join(data_dir,'train2017'),train_dir)
    # shutil.move(osp.join(data_dir,'rotatexml'),train_dir)
    # shutil.move(osp.join(data_dir,'labelTxt_val'),test_dir)
    # shutil.move(osp.join(data_dir,'val2017'),test_dir)
    # shutil.move(osp.join(data_dir,'rotatexml_val'),test_dir)
    #rename folder
    # shutil.move(osp.join(train_dir,'train2017'),osp.join(train_dir,'images')) 
    # shutil.move(osp.join(test_dir,'labelTxt_val'),osp.join(test_dir,'labelTxt')) 
    # shutil.move(osp.join(test_dir,'val2017'),osp.join(test_dir,'images')) 
    # convert it to json format
    # generate_json_labels(train_dir,osp.join(train_dir,'trainval.json'))
    generate_json_labels(test_dir,osp.join(test_dir,'test.json'),trainval=False)

if __name__ == '__main__':
    rotatexml_dir = '/home/zf/2020HJJ/train_worm'
    preprare_rotatexml(rotatexml_dir)
    print('done')
