# --------------------------------------------------------
# dota_evaluation_task1
# Licensed under The MIT License [see LICENSE for details]
# Written by Jian Ding, based on code from Bharath Hariharan
# --------------------------------------------------------

"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import xml.etree.ElementTree as ET
import os
#import cPickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/zf/s2anet_rep/DOTA_devkit')
import polyiou
from functools import partial
from dota_utils import GetFileFromThisRootDir
from dota_poly2rbox import poly2rbox_single_v2,rbox2poly_single
# from 得到文件列表程序 import generate_file_list

def parse_gt(filename):
    """
    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    objects = []
    with  open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if (len(splitlines) < 9):
                    continue
                object_struct['name'] = splitlines[8]

                if (len(splitlines) == 9):
                    object_struct['difficult'] = 0
                elif (len(splitlines) == 10):
                    object_struct['difficult'] = int(splitlines[9])
                object_struct['bbox'] = [float(splitlines[0]),
                                         float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7])]
                objects.append(object_struct)
            else:
                break
    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
            # cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    #if not os.path.isdir(cachedir):
     #   os.mkdir(cachedir)
    #cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    
    recs = {}
    for i, imagename in enumerate(imagenames):
        #print('parse_files name: ', annopath.format(imagename))
        gttxt_path=os.path.join(annopath,imagename+'.txt')
        recs[imagename] = parse_gt(gttxt_path)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets from Task1* files
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])

    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    ## note the usage only in numpy not for list
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT
        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]

            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):

                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps
            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall

    print('check fp:', fp)
    print('check tp', tp)


    print('npos num:', npos)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def generate_file_list(img_dir,output_txt,file_ext='.txt'):
    #读取原图路径  
    # img_dir=os.path.split(img_dir)[0]
    imgs_path = GetFileFromThisRootDir(img_dir, file_ext) 
    f = open(output_txt, "w",encoding='utf-8')
    for num,img_path in enumerate(imgs_path,0):
        obj='{}\n'.format(os.path.splitext(os.path.split(img_path)[1])[0])
        f.write(obj)
    f.close()
    print('Generate {} down!'.format(output_txt))


def parse_ann_info( lab_path):
    # lab_path = osp.join(label_base_path, img_name+'.txt')
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


    ### GT
def DOTAtxt2GT_to_eval(GT_xml_path,txt_dir_h,file_ext='.txt'):
    if  not os.path.exists(txt_dir_h):
        os.mkdir(txt_dir_h)
    file_paths = GetFileFromThisRootDir(GT_xml_path,file_ext ) 
    for count, txt_path in enumerate(file_paths):   
        bboxes, labels, bboxes_ignore, labels_ignore = parse_ann_info(txt_path)
        # eval txt
        # Task1 #
        write_handle_h = open(os.path.join(txt_dir_h, 'Task1_gt_{}.txt'.format(os.path.splitext(os.path.split(txt_path)[1])[0])), 'w')#Task1_gt_
        for i, rbox in enumerate(bboxes):
            # rbox[4]=0
            # rbox_cv=rotate_rect2cv(rbox)
            # rect_box = cv2.boxPoints(rbox_cv)
            rect_box =rbox2poly_single(rbox)
            command = '%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %s 0\n' % (
                                                        rect_box[0], rect_box[1], rect_box[2], rect_box[3],
                                                        rect_box[4], rect_box[5], rect_box[6], rect_box[7],
                                                        labels[i]
                                                        )
            write_handle_h.write(command)
        write_handle_h.close()


def eval_dotaresult(detpath,annopath,classnames,ovthresh=0.5):
    
    detpath =detpath+'/Task1_{:s}.txt'
    # annopath = r'/home/zf/2020HJJ/train_worm/Test/labelTxt' # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    # annopath = r'/project/jmhan/data/dota/test/OrientlabelTxt-utf-8/{:s}.txt' # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    imagesetfile = annopath+'/testset.txt'
    generate_file_list(annopath,imagesetfile)
    # DOTAtxt2GT_to_eval(txt_path,gt_path)
    # For DOTA-v1.5
    # classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    #             'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']
    # For DOTA-v1.0
    # classnames = ['1', '2', '3', '4', '5']
   
    # classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    #             'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
    classaps = []
    map = 0
    for classname in classnames:
        print('classname:', classname)
        rec, prec, ap = voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=ovthresh,
             use_07_metric=True)
        map = map + ap
        #print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        print('ap: ', ap)
        classaps.append(ap)

        # # umcomment to show p-r curve of each category
        # plt.figure(figsize=(8,4))
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.xticks(fontsize=11)
        # plt.yticks(fontsize=11)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # ax = plt.gca()
        # ax.spines['top'].set_color('none')
        # ax.spines['right'].set_color('none')
        # plt.plot(rec, prec)
        # # plt.show()
        # plt.savefig('pr_curve/{}.png'.format(classname))
    map = map/len(classnames)
    print('map:', map)
    classaps = 100*np.array(classaps)
    print('classaps: ', classaps)
if __name__ == '__main__':
    detpath =r'/home/zf/2020HJJ/train_worm/s2anet_r101_fpn_3x/result_raw'
    annopath = r'/home/zf/2020HJJ/train_worm/Test/labelTxt'
    ovthresh=0.5
    classnames =['航母',
        '黄蜂级',
        '塔瓦拉级',
        '蓝岭级',
        '奥斯汀级',
        '惠特贝岛级',
        '圣安东尼奥级',
        '新港级',
        '提康德罗加级',
        '阿利·伯克级',
         '朱姆沃尔特级',
        '佩里级',
        '刘易斯和克拉克级',
        '供应级',
        '凯泽级',
        '霍普级',
        '仁慈级',
        '先锋级',
        '自由级',
        '独立级',
        '复仇者级',
        '胜利级',
        '潜艇',
        '其他']
    eval_dotaresult(detpath,annopath,classnames,ovthresh)