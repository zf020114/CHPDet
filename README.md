# CHPDet
**PyTorch implementation of "*Arbitrary-Oriented Ship Detection through Center-Head Point Extraction*",  [<a href="https://raw.github.com/zf020114/DARDet/master/Figs/GRSL.pdf">pdf</a>].**<br><br>


## *Highlights:*
#### 1. *we propose a center-head point extraction based detector (named CHPDet) to achieve arbitrary-oriented ship detection in remote sensing images.*
  <p align="center"> <img src="https://raw.github.com/zf020114/CHPDet/master/Figs/framework.png" width="100%"></p>
 
#### 2. *Our CHPDet achieves state-of-the-art performance on the FGSD2021, HRSC2016 and  UCAS-AOD datasets with high efficiency.*
  <p align="center"> <img src="https://raw.github.com/zf020114/CHPDet/master/Figs/result3.png" width="100%"></p>
  <p align="center"> <img src="https://raw.github.com/zf020114/CHPDet/master/Figs/resulthrsc.png" width="100%"></p>
  <p align="center"> <img src="https://raw.github.com/zf020114/CHPDet/master/Figs/resultaod.png" width="100%"></p>

#### 2. *we proposed a new dataset named FGSD2021 for multi-class arbitrary-oriented ship detection in remote sensing images at a fixed GSD.*
  <p align="center"> <img src="https://raw.github.com/zf020114/CHPDet/master/Figs/dataset.png" width="75%"></p>

## Benchmark and model zoo (exact code 'nudt')

|Model          |    Backbone     |    Dataset  |  Rotate | img_size  | Inf time (fps) | box AP (ori./now) | Download|
|:-------------:| :-------------: | :-----:| :-----: | :-----:  | :------------: | :----: | :---------------------------------------------------------------------------------------: |
|CHPDet         |    DLA-34_OIM     |   FGSD2021     |  ✓     |   512x512     |      41.7      |  87.91 |    [model](https://pan.baidu.com/s/1RmYPbAmNhMfoS5AS9sjq7Q )    |
|CHPDet         |    Hourglass_104  |   HRSC2016     |  ✓    |  1024x1024     |      13.7      |  90.55 |    [model](https://pan.baidu.com/s/1JHu1BeTHOKLyATpE6nadlg)     |

The FGSD2021 dataset is available at  [<a href="https://pan.baidu.com/s/1q1YQsFAR6nvWIVoa6eo85w?pwd=nudt提取码：nudt">DataSet</a>] (exact code 'nudt')

The FGAD2022 dataset is available at  [<a href="https://pan.baidu.com/s/1PO0ZtHI5-4J9j5tcKSPp6Q?pwd=nudt提取码：nudt">DataSet</a>] (exact code 'nudt')

## Installation

The code was tested on Ubuntu 16.04, with [Anaconda](https://www.anaconda.com/download) Python 3.7 and [PyTorch]((http://pytorch.org/)) v1.4.0. NVIDIA GPUs are needed for both training and testing.
After installing Anaconda:

1. [Optional but recommended] create a new conda environment. 

    ~~~
    conda create --name CHPDet python=3.7
    ~~~
    And activate the environment.
    
    ~~~
    conda activate CHPDet
    ~~~

2. Install PyTorch 1.4.0:

     
3. Install [COCOAPI](https://github.com/cocodataset/cocoapi):

    ~~~
    # COCOAPI=/path/to/clone/cocoapi
    git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
    cd $COCOAPI/PythonAPI
    make
    python setup.py install --user
    ~~~

4. Clone this repo:

    ~~~
    CHPDet_ROOT=/path/to/clone/CHPDet
    git clone https://github.com/zf020114/CHPDet $CHPDet_ROOT
    ~~~


5. Install the requirements

    ~~~
    pip install -r requirements.txt
    ~~~
    
    
6. Compile deformable convolutions (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_0.4)).

    ~~~
    cd $CHPDet_ROOT/src/lib/models/networks/DCNv2
    ./make.sh
    ~~~

7. compile orn from s2anet
     cd $CHPDet_ROOT/src/lib/models/networks/orn
     bash make.sh


7.install DOTA_devkit to get the evel result
7.1. install swig
```
    sudo apt-get install swig
```
7.2. create the c++ extension for python
```cd  /CHPDet/src/DOTA_devkit
    swig -c++ -python polyiou.i
    python setup.py build_ext --inplace
```

8. [Optional, only required if you are using extremenet or multi-scale testing] Compile NMS if your want to use multi-scale testing or test ExtremeNet.

    ~~~
    cd $CenterNet_ROOT/src/lib/external
    make
    ~~~

9. Download pertained models for [detection]() or [pose estimation]() and move them to `$CHPDet_ROOT/models/`. 


## Prepare dataset.
    Transform dataset to COCO format

    1. prepare data as FGSD2021 using the tool labimg2 (https://github.com/chinakook/labelImg2),
    Here, we give an example for image and annotation in ./Figs/

    2. crop data to fixed size using ./src/0data_crop.py.
    
    3. make json file using ./src/1make_json_anno.py.
    
    4. check the json file by ./src/3show_json_anno.py

    5. It is recommended to symlink the dataset root to `data/coco`.


* Inference CHPDet on DOTA.
   1. download model to /exp/multi_pose/dla_USnavy512_RGuass1_12raduis_arf
   2. python test.py multi_pose --exp_id dla_USnavy512_RGuass1_12raduis_arf --dataset coco_hp --resume  --debug 2


*If you want to evaluate the result on DOTA test-dev, zip the files in ```work_dirs/dardet_r50_fpn_1x_dcn_val/result_after_nms``` and submit it to the  [evaluation server](https://captain-whu.github.io/DOTA/index.html).



## Train a model

1.1. Train centernet-Rbb:
python main.py ctdet --exp_id CenR_usnavy512_dla_2x --arch dla_34 --batch_size 26 --master_batch 14--lr 5e-4 --gpus 0,1 --num_workers 8 --num_epochs 230 --lr_step 180,210  --load_model ../exp/multi_pose_dla_3x.pth
1.2. test centernet-Rbb :
python test.py ctdet --exp_id CenR_usnavy512_dla_2x --arch dla_34 --keep_res --resume 

2.1 Train CHP_DLA_34:

python main.py multi_pose --exp_id dla_USnavy512_RGuass1_12raduis_arf --arch dlaarf_34  --dataset coco_hp --batch_size 28  --master_batch 16   --gpus 0,1 --lr 5e-4 --load_model ../exp/ctdet_coco_hg.pth  --num_workers 8 --num_epochs 320 --lr_step 270,300
2.2 test CHP_DLA_34:
python test.py multi_pose --exp_id dla_USnavy512_RGuass1_12raduis_arf --arch dlaarf_34  --dataset coco_hp --resume  

## eval a model
1. Test the model to get a result.json file.
2. Make DOTA format ground-truth file by ./src/4rotate_xml2DOTAtxt.py  
   Here, we give an example for DOTA format ground-truth file in ./Figs/GT20_TxT
3. run to eval on test set by run ./srv/5eval_json.py

## Citation

If you find this project useful for your research, please use the following BibTeX entry.

@article{CHPDet,
  title={Arbitrary-Oriented Ship Detection through Center-Head Point Extraction},
  author={Zhang, Feng and Wang, Xueying and Zhou, Shilin and Wang, Yingqian and Hou, Yi},
  journal={IEEE Transactions  on Geoscience and Remote Sensing},
  year={2021},
  publisher={IEEE}
}

## Contact
**Any question regarding this work can be addressed to [zhangfeng01@nudt.edu.cn](zhangfeng01@nudt.edu.cn).**
