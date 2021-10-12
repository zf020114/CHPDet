# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 18:22:36 2019
版本10.31
@author: admin
"""

import os
import numpy as np
import DataFunction



#voc路径

def generate_file_list(img_dir,output_txt,file_ext='.txt'):
    #读取原图路径  
    imgs_path = DataFunction.get_file_paths_recursive(img_dir, file_ext) 
    f = open(output_txt, "w",encoding='utf-8')
    for num,img_path in enumerate(imgs_path,0):
        obj='{}\n'.format(os.path.splitext(os.path.split(img_path)[1])[0])
        f.write(obj)
    f.close()
    print('Down!')

if __name__ == '__main__':
    img_dir ='/media/zf/E/Dataset/USnavy_test_gt'
    output_txt='/media/zf/E/Dataset/US_Navy_train_square/eval/testlist.txt'#存储新的VOCanno位置
    file_ext='.txt'
    generate_file_list(img_dir,output_txt,file_ext)