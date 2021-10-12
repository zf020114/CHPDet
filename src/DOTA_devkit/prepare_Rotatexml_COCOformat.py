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

def parse_ann_info(img_base_path, label_base_path, img_name,NAME_LABEL_MAP):
    lab_path = osp.join(label_base_path, img_name+'.txt')
    bboxes, labels, bboxes_ignore, labels_ignore = [], [], [], []
    with open(lab_path, 'r') as f:
        for ann_line in f.readlines():
            ann_line = ann_line.strip().split(' ')
            bbox = [float(ann_line[i]) for i in range(8)]
            # 8 point to 5 point xywha
            bbox = poly2rbox_single_v2(bbox)
            class_name = NAME_LABEL_MAP[ann_line[8]]
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


def generate_json_labels(img_path,label_path, out_path, NAME_LABEL_MAP,trainval=True,ext='.jpg'):
    """Generate .json labels which is similar to coco format
    Args:
        src_path: dataset path containing images and labelTxt folders.
        out_path: output json file path
        trainval: trainval or test?
    """
    # img_path = os.path.join(src_path, 'train2017')
    # label_path = osp.join(osp.dirname(img_path), 'labelTxt')
    # label_path = img_path.replace(src_path, 'labelTxt')
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
            img_info['filename'] = img_name+ext
            img_info['height'] = img.height
            img_info['width'] = img.width
            img_info['id'] = id
            if(trainval == True):
                if(os.path.exists(label) == False):
                    print('Label:'+img_name+'.txt'+' Not Exist')
                else:
                    bboxes, labels, bboxes_ignore, labels_ignore = parse_ann_info(
                        img_path, label_path, img_name,NAME_LABEL_MAP)
                    ann = {}
                    ann['bboxes'] = bboxes
                    ann['labels'] = labels
                    ann['bboxes_ignore'] = bboxes_ignore
                    ann['labels_ignore'] = labels_ignore
                    img_info['annotations'] = ann
            data_dict.append(img_info)
        json.dump(data_dict, f)



def preprare_rotatexml(data_dir,NAME_LABEL_MAP):
    ## convert hrsc2016 to dota raw format
    # generate_txt_labels_train(data_dir)
    # generate_txt_labels_val(data_dir)
  
    # convert it to json format
    generate_json_labels(osp.join(data_dir,'train2017'),
                         osp.join(data_dir,'labelTxt'),
                         osp.join(data_dir,'annotations','trainval.json'),
                         NAME_LABEL_MAP)
    generate_json_labels(osp.join(data_dir,'val2017'),
                         osp.join(data_dir,'labelTxt_val'),
                         osp.join(data_dir,'annotations', 'test.json'), 
                         NAME_LABEL_MAP)
    # generate_json_labels(osp.join(data_dir,'val2017'),osp.join(data_dir,'annotations','test.json'),trainval=False)
if __name__ == '__main__':
    rootdir = '/media/zf/E/Dataset/US_Navy_train_square'
    NAME_LABEL_MAP_USnavy =  {
        '航母': 1,
        '黄蜂级': 2,
        '塔瓦拉级': 3,
        '蓝岭级': 24,
        '奥斯汀级': 5,
        '惠特贝岛级': 6,
        '圣安东尼奥级': 7,
        '新港级': 8,
        '提康德罗加级': 9,
        '阿利·伯克级': 10,
        '朱姆沃尔特级': 24,
        '佩里级': 12,
        '刘易斯和克拉克级': 13,
        '供应级': 14,
        '凯泽级': 15,
        '霍普级': 16,
        '仁慈级': 17,
        '先锋级': 24,
        '自由级': 19,
        '独立级': 20,
        '复仇者级': 21,
        '胜利级': 24,
        '潜艇':23,
        '其他':24
        }
    data_dir='/home/zf/Dataset/USnavy_test_gt/usnavy_1024_s2anet/trainval_split'
    generate_json_labels(osp.join(data_dir,'images'),
                         osp.join(data_dir,'labelTxt'),
                         osp.join(data_dir,'trainval.json'),
                         NAME_LABEL_MAP_USnavy)
    # preprare_rotatexml(rootdir,NAME_LABEL_MAP_USnavy)
    # print('done')
