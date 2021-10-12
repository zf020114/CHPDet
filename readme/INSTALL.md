# Installation


The code was tested on Ubuntu 16.04, with [Anaconda](https://www.anaconda.com/download) Python 3.7 and [PyTorch]((http://pytorch.org/)) v1.4.0. NVIDIA GPUs are needed for both training and testing.
After install Anaconda:

0. [Optional but recommended] create a new conda environment. 

    ~~~
    conda create --name CHPDet python=3.7
    ~~~
    And activate the environment.
    
    ~~~
    conda activate CHPDet
    ~~~

1. Install pytorch 1.4.0:

     
2. Install [COCOAPI](https://github.com/cocodataset/cocoapi):

    ~~~
    # COCOAPI=/path/to/clone/cocoapi
    git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
    cd $COCOAPI/PythonAPI
    make
    python setup.py install --user
    ~~~

3. Clone this repo:

    ~~~
    CHPDet_ROOT=/path/to/clone/CHPDet
    git clone https://github.com/zf020114/CHPDet $CHPDet_ROOT
    ~~~


4. Install the requirements

    ~~~
    pip install -r requirements.txt
    ~~~
    
    
5. Compile deformable convolutional (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_0.4)).

    ~~~
    cd $CHPDet_ROOT/src/lib/models/networks/DCNv2
    ./make.sh
    ~~~

6. compile orn from s2anet
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
