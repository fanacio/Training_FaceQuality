# 人脸识别及人脸质量模型项目

### **From 樊一超**
该网络在提取特征向量的同时给出明确的定量质量评分，在特征提取网络上加一个质量的网络分支，不需要训练带有质量标签的数据 

## 1. 准备工作
### 1.1 环境准备
1. 训练镜像：recognition，获取方式docker pull fanacio/recognition:v0 或者本地保存镜像recognition.tar
2. rec2img镜像：rec2img，获取方式docker pull fanacio/rec2img 或者本地保存镜像rec2img.tar
3. 执行环境：A10显卡服务器（2卡机）、ubuntu18.04系统、显卡驱动470.141.03 
4. 提示：训练镜像不使用yolo_yolox是因为其python版本太高，无法安装bcolz库，且为了做到环境隔离，故制作了recognition镜像。

### 1.2 数据集准备
- Download [MS1Mv2](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)
```
我下载且使用的是Face Recognition Datasets/CASIA-Webface (10K ids/0.5M images)->faces_webface_112x112训练集，文件夹名称为faces_webface_112x112。
下载后解压，存放在rec2image文件夹中备用。其目录结构如下：
    .
    |-- agedb_30.bin
    |-- calfw.bin
    |-- cfp_ff.bin
    |-- cfp_fp.bin
    |-- cplfw.bin
    |-- lfw.bin
    |-- property
    |-- train.idx
    |-- train.lst
    `-- train.rec
```
- Extract将数据集中的images提取出来
    ```
    由于数据集格式为bin及rec格式，需要转换成img格式，在这里本人使用的是mxnet库中的方法进行转换的。
    ```
    #### 方法一(不推荐，跳过)
    - step1: 使用pip检查mxnet库是否存在，如果不存在则按照[Install Instruction](https://mxnet.apache.org/get_started)进行安装。
    - step2: 为了安装快捷，则使用如下命令(安装哪个版本根据自身cuda版本决定，本人下载的是mxnet-cu110 1.9.1)：
    ```
    pip --default-timeout=100000 install mxnet-cu110 -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
    ```
    - step3: 安装好之后到rec2image目录下执行命令：python rec2image.py，不出意外会报错找不到libmxnet.so等库，此时需要find它们的位置并将其放在执行目录或全局目录中即可。
    - step4: 执行完step3操作后继续执行python rec2image.py依旧会报错如下内容：
    ```
    OSError: libnccl.so.2: cannot open shared object file: No such file or directory
    Notes: Starting from version 1.8.0, cuDNN and NCCL should be installed by users in advance.                       
    Please follow the instructions in https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html to install NCCL
    ```
    报错解决方法参考：
    [参考链接1](https://blog.csdn.net/fyfugoyfa/article/details/124203296)
    [参考链接1](https://blog.csdn.net/qq_38154295/article/details/121435876?spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-121435876-blog-124203296.pc_relevant_3mothn_strategy_and_data_recovery&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-121435876-blog-124203296.pc_relevant_3mothn_strategy_and_data_recovery&utm_relevant_index=7)
    
    ```
    此时需要下载安装NCCL，本人选择离线安装，并将下载的NCCL库放在了lib文件中。
    ```
    - styep5: 安装NCCL，参考[NVIDIA官网链接](https://docs.nvidia.com/deeplearning/nccl/install-guide/)
    ```
    执行dpkg命令：
    dpkg -i nccl-local-repo-ubuntu1804-2.15.5-cuda11.0_1.0-1_amd64.deb
    提示没有安装密钥，并给出安装密钥的方法，直接复制执行即可：
    cp /var/nccl-local-repo-ubuntu1804-2.15.5-cuda11.0/nccl-local-B5FFC818-keyring.gpg /usr/share/keyrings/
    更新源：apt update
    安装libnccl2等命令：apt install libnccl2=2.15.5-1+cuda11.0 libnccl-dev=2.15.5-1+cuda11.0 （这个版本和上面下载的deb版本须对应）
    安装完成...
    ```
    - step6: 继续执行python rec2image.py，报错OSError: libcusolver.so.10: cannot open shared object file: No such file or directory

    本次安装的mxnet库和nccl库都为cuda11对应版本，报错libcusolver.so.10属实不正常，具体解决办法参考[链接](https://blog.csdn.net/qq_42935201/article/details/124992636)

    ```
    执行如下命令：
    find / -name libcusolver.*
    将libcusolver.so.11.0.1.105（软链的根）移动到lib目录中，然后重命名：
    mv libcusolver.so.11.0.1.105 libcusolver.so.10
    ```
    ```
    此方法缺点：安装的mxnet是对应着cuda11.0版本的，但需要的libcusolver.so.10依赖是cuda10的，在使用step6重命名后，在使用时很容易导致错误，不推荐使用。
    ```
    #### 方法二（推荐使用）
    ```
    直接使用带有mxnet的docker进行操作，这个docker本人已经制作好了(使用的基础镜像为docker pull bitnami/mxnet)，按照如下操作即可
    ```
    - step1: 下载镜像：docker pull fanacio/rec2img（也可以在本地拉取，名字为rec2img.tar）
    - step2: 制作docker，执行如下命令(-u 表示用root权限进入)：
    ```
    docker run -u 0 -it -p 3285:22 --gpus all --privileged --net=bridge --ipc=host --pid=host -v /data/data/fanyichao/model_trainging/:/home bitnami/mxnet:latest /bin/bash
    ```
    - step3: 到rec2images目录下执行rec2image.py文件将rec格式的训练集转换成图片：python rec2image.py --include faces_webface_112x112/ --output images
    ```
    step3中的参数--include表示输入数据集路径，--output表示结果存放路径。
    保存结果的目录结构如下所示：
        |-- 0_493462
        |   |-- 0.jpg
        |   |-- 1.jpg
        |   |-- 10.jpg
        |   |-- 11.jpg
        |   |-- 15.jpg
        |-- 0_501187
        |-- 0_501188
        |-- 0_501189
        `-- 0_501195
    ```
- 生成训练文件列表Generate the training file list
    ```
    cd dataset
    python generate_file_list.py
    需要注意的是generate_file_list.py文件中的路径参数需要修改，即DATA_DIR和train_data_dirs，改成自己的参数即可，使得DATA_DIR和train_data_dirs组成完整的step3中的output路径。
    最终的输出结果是face_train_ms1mv2.txt文件，内容为：
    /home/FaceQuality-master/rec2image/images/0_495950/165.jpg;0
    /home/FaceQuality-master/rec2image/images/0_495950/196.jpg;0
    ......
    ```

### 1.3 预训练权重下载
[模型下载地址](https://drive.google.com/drive/folders/1YtSxo5-NuzDY1baV7wQkUxN3ysvwW6Wp?usp=sharing)
```
这个drive.google网盘一般情况进不去，本人已经下载好放在了pre_weights目录，直接使用即可。
```

## 2. 测试预训练权重

```bash
执行如下代码即可
python test_quality.py --backbone pre_weights/backbone.pth --quality pre_weights/quality.pth --file test_faces
```
```
结果保存在quality_result目录中，且文件名为质量分数。
```

## 3. 开始训练
### 3.0 set config.py
```
由于设备的不同使得其训练参数设置也不同，此工程的参数主要设置为config.py文件，为了防止内存不足等问题出现，需根据实际情况设置如下参数：
    BATCH_SIZE = 500   #batch大小，batch越大训练越快，占用显存也就越大，根据显存大小设定
    ...
    BACKBONE_LR = 0.05       #如果嫌训练过程收敛慢则可以调大学习率
    QUALITY_LR = 0.01        #如果嫌训练过程收敛慢则可以调大学习率
    HEAD_GPUS = [0]
    BACKBONE_GPUS = [0 , 1]  #训练feature（人脸识别）时使用的显卡ID，默认两张卡都使用
```
### 3.1 set config.py, then run **python train_feature.py**
config.py文件设置如下：
```json
    ...
    BACKBONE_RESUME_ROOT = ''
    HEAD_RESUME_ROOT = ''
    TRAIN_FILES = './dataset/face_train_ms1mv2.txt'
    BACKBONE_LR = 0.05
    PRETRAINED_BACKBONE = ''
    PRETRAINED_QUALITY = ''
    ...
```
```
执行python train_feature.py命令时报错ModuleNotFoundError: No module named 'bcolz'
意思是此环境中没有bcolz库，解决方案为：
conda install bcolz

注意，不能使用如下👇方法安装：
pip --default-timeout=100000 install bcolz -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
```
[报错参考链接](https://blog.csdn.net/weixin_41848012/article/details/121675751)


**这里需要注意的是**：bcolz不能使用pip安装，且其对python版本要求严格，本人使用python版本未3.8.5，cuda为11.1
```
这里训练出来的结果保存在backbone_resume.pth目录下，名称为Backbone_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth和Head_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth，损失log保存在head_resume.pth供可视化实时监控训练情况。
此步训练完后，将两个权重文件放在工程根目录下并重命名为backbone_resume_part1.pth和head_resume_part1.pth，以便开始如下第二步训练。
```

### 3.2 set config.py, then run **python train_quality.py**
```json
    ...
    BACKBONE_RESUME_ROOT = './backbone_resume_part1.pth'
    HEAD_RESUME_ROOT = './head_resume_part1.pth'
    TRAIN_FILES = './dataset/face_train_ms1mv2.txt'
    BACKBONE_LR = 0.05
    PRETRAINED_BACKBONE = ''
    PRETRAINED_QUALITY = ''
    ...
```

```
这里训练出来的结果保存在backbone_resume.pth目录下，名称为Quality_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth和Head_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth，损失log保存在head_resume.pth供可视化实时监控训练情况。
此步训练完后，将权重文件Quality_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth放在工程根目录下并重命名为pretrained_qulity_resume.pth，将第一步训练所得的Backbone_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth放在工程根目录下并重命名为pretrained_backbone_resume.pth，以便开始如下第三步训练。
```

### 3.3 set config.py, then run **python train_feature.py**
```json
    ...
    BACKBONE_RESUME_ROOT = ''
    HEAD_RESUME_ROOT = ''
    TRAIN_FILES = './dataset/face_train_ms1mv2.txt'
    BACKBONE_LR = 0.05
    PRETRAINED_BACKBONE = ''
    PRETRAINED_QUALITY = ''

    PRETRAINED_BACKBONE = 'pretrained_backbone_resume.pth'
    PRETRAINED_QUALITY = 'pretrained_qulity_resume.pth'
    ...
```
```
最终所有权重保存在backbone_resume.pth目录下，根据tensorboard观察结果选择损失和精度表现最佳的pth权重即可。
```
**注意**：NUM_EPOCH可以设置大一点，比如200等，然后根据可视化的损失情况来选择合适的pth权重作为下一阶段预训练的权重，NUM_EPOCH设置较大的时候不见得最后一次保存的pth为最好的结果（可能过拟合）；NUM_EPOCH设置较小的时候不见得最后一次保存的pth为最好的结果（可能欠拟合）...以我这次训练开源数据集为例，NUM_EPOCH=90比较小，没有收敛到最好。

## 4. 关于可能出现的报错
### 4.1 报错1
1. 报错内容
```bash
ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /opt/conda/lib/python3.9/site-packages/matplotlib/_path.cpython-39-x86_64-linux-gnu.so)
```
2. 错误解读
```
意思就是/usr/lib/x86_64-linux-gnu中的libstdc++.so.6没有GLIBCXX_3.4.29参数，使用命令去查看：
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4.*
发现确实没有GLIBCXX_3.4.29，只到GLIBCXX_3.4.28，这是由于我执行了conda update --prefix /opt/conda anaconda命令升级了一些python库所导致的
```
3. 解决方案
```
执行命令：find / -name libstdc++.so*，对此库进行搜索，搜索发现在/opt/conda/lib/目录下，然后执行：
strings /opt/conda/lib/libstdc++.so.6 | grep GLIBCXX_3.4.29，确定输出存在，则将其加到临时环境变量里即可，执行命令添加临时环境变量：
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/
```

### 4.2 报错2
1. 报错内容
```bash
SyntaxError: future feature annotations is not defined
```
2. 错误解读
此错误在[源码安装](https://blog.csdn.net/weixin_41848012/article/details/121675751)bcolz时候报错的，大体意思是python版本与bcolz版本不兼容，尝试了多个bcolz版本后发现都无法安装
3. 解决方案
```
升级python至3.7+版本
```

### 4.3 报错3
1. 报错内容
```bash
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory
ImportError: libSM.so.6: cannot open shared object file: No such file or directory
```
2. 错误解读
```
缺少此库
```
3. 解决方案
```
安装即可：apt install libgl1-mesa-glx
安装即可：apt install libglib2.0-dev，若执行失败，则执行命令：apt-get install libglib2.0-0
安装即可：apt-get install libsm6
```

### 4.4 报错4
1. 报错内容
```bash
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
2. 错误解读
```
可能cuda版本太低了，导致其在性能好的显卡上无法运行
```
3. 解决方案
```
升级cuda
```
### 4.5 报错5
1. 报错内容
```bash
(base) root@1d8a4317d206:/home# conda install bcolz
...                                                                                                                                                                                                  
UnsatisfiableError: The following specifications were found
to be incompatible with the existing python installation in your environment:

Specifications:

  - bcolz -> python[version='>=2.7,<2.8.0a0|>=3.5,<3.6.0a0|>=3.7,<3.8.0a0|>=3.8,<3.9.0a0|>=3.6,<3.7.0a0']

Your python: python=3.9

If python is on the left-most side of the chain, that's the version you've asked for.
When python appears to the right, that indicates that the thing on the left is somehow
not available for the python version you are constrained to. Note that conda will not
change your python version to a different minor version unless you explicitly specify
that.

The following specifications were found to be incompatible with your system:

  - feature:/linux-64::__glibc==2.31=0
  - feature:|@/linux-64::__glibc==2.31=0
  - bcolz -> libgcc-ng[version='>=7.5.0'] -> __glibc[version='>=2.17']

Your installed version is: 2.31
```
2. 错误解读
```
python版本太高了，没有与之适配的bcolz库
```

### 4.6 报错6
1. 报错内容
```bash
Requirement already satisfied: numpy>=1.14.5; python_version >= "3.7" in /root/anaconda3/lib/python3.7/site-packages (from opencv-python) (1.16.4)
Building wheels for collected packages: opencv-python
  Building wheel for opencv-python (PEP 517)
......  
```
2. 解决方案
```
安装opencv-python时,在这里卡死了，原因可能是opencv-python版本太高了，应该给定一个相对不那么高的版本，如：
pip --default-timeout=10000 install opencv-python==4.2.0.34 -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
或者pip版本太低了，应该升级pip，如下命令：
pip3 install --upgrade pip
```

### 4.7 报错7
1. 报错内容
```bash
...
File "/root/anaconda3/lib/python3.8/site-packages/torch/nn/functional.py", line 2149, in batch_norm
    return torch.batch_norm(
RuntimeError: CUDA out of memory. Tried to allocate 3.74 GiB (GPU 0; 22.20 GiB total capacity; 19.83 GiB already allocated; 693.38 MiB free; 19.88 GiB reserved in total by PyTorch)
```
```bash
...
    return F.batch_norm(
  File "/root/anaconda3/lib/python3.8/site-packages/torch/nn/functional.py", line 2149, in batch_norm
    return torch.batch_norm(
RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.
```
2. 解决方案
```
这是由于config.py文件没有设置好所导致的，导致内存溢出。根本原因是显存不足所导致，此时应减少batchsize，多开显卡等。具体需要修改的参数为：BATCH_SIZE、HEAD_GPUS和BACKBONE_GPUS。
```

### 4.8 报错8
1. 报错内容
```bash
(base) root@4022d88ca5c3:/home/FaceQuality-master# python train_feature.py
Number of Training Classes: 10572
...
  0%|▏                                                                                                                                                              | 1/981 [00:12<3:27:50, 12.72s/it]
Traceback (most recent call last):
  File "train_feature.py", line 191, in <module>
    train()
  File "train_feature.py", line 134, in train
    features = BACKBONE(inputs)
...
  File "/root/anaconda3/lib/python3.8/site-packages/torch/_utils.py", line 429, in reraise
    raise self.exc_type(msg)
RuntimeError: Caught RuntimeError in replica 1 on device 1.
Original Traceback (most recent call last):
...
    return F.conv2d(input, weight, bias, self.stride,
RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED
```
2. 解决方案
```
这是在执行第三步训练时候的报错，报错原因是显存占用太大了，适当的减小batchsize即可。
```


## 5. 一些安装命名
- images2docker  
sudo docker run -it -p 2912:22 --gpus all --privileged --net=bridge --ipc=host --pid=host -v /data/data/fanyichao/model_trainging/:/home danny99wong/cuda11.1_python3.8_torch1.9:v1 /bin/bash
- 安装cv2（opencv-python）   
pip --default-timeout=10000 install opencv-python -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
- 安装anaconda3 [链接](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)   
bash Anaconda3-2020.11-Linux-x86_64.sh
- 安装pytorch [链接](https://pytorch.org/get-started/previous-versions/)   
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
- 安装tensorboardX   
pip --default-timeout=10000 install tensorboardX -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
- 安装ptflops   
pip --default-timeout=10000 install ptflops -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
- 安装loguru   
pip --default-timeout=10000 install loguru -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

## 6. 可视化工具
1. 使用tensorboard可视化loss
```
首先进入./head_resume.pth目录，然后执行如下命令：
tensorboard --logdir . --bind_all
此时会在vscode终端出现如下内容：
```
```bash
(base) root@4022d88ca5c3:/home/FaceQuality-master/head_resume.pth# tensorboard --logdir .  --bind_all
TensorFlow installation not found - running with reduced feature set.

NOTE: Using experimental fast data loading logic. To disable, pass
    "--load_fast=false" and report issues on GitHub. More details:
    https://github.com/tensorflow/tensorboard/issues/4784

TensorBoard 2.11.0 at http://4022d88ca5c3:6006/ (Press CTRL+C to quit)
```
```
如上内容中：http://4022d88ca5c3:6006/表示容器4022d88ca5c3中的端口6006，然后在端口那栏添加端口在本地浏览器打开即可。
```
**注意**：可视化时此文件夹中不能保存上一次训练所得的文件，只能存在目前正在训练的一个events.out.tfevents...

## 7. 关于转模型（以onnx为例）及部署的实践
### 7.1 转onnx
```
在test.py文件中设置了pytorch2onnx开关，完成转onnx操作，在这里分为两部分：
第一部分：backbone部分，完成人脸识别，其结构为resnet100+FC+BN，降维至512维；
第二部分：quality部分，完成人脸质量，其结构为将识别的resnet100输出作为输入，进行一个简单的FC+BN+RELU+FC+sigmoid等操作。
故，转onnx完成时共转出两个onnx模型，一个可供识别使用，两个结合起来可供人脸质量使用。
```
### 7.2 关于推理部署
从test.py可以看出来：
```python
    _, fc = backbone(ccropped.to(device), True)
    s = quality(fc)[0]
```
其中，_表示的便是识别的512维度，fc便是降维前的tensor，s便是质量模型的输出（tensor shape = 1）。获得识别的512维度tensor后，可以通过求欧式余弦相似度的方法来判断是否为同一个人


## 8. 总结
### 8.1 严格的环境说明
```
由于本工程需要安装bcolz三方库，此库的安装导致对环境要求及其严格。
最初本人依据本人早些时候制作的yolo_yolox镜像进行尝试，是可以完成测试操作的，但在制作训练集rec2img以及训练安装bcolz库时会报错，这大概是由于此docker中python版本太高所导致，本人担心镜像越来越庞大所以没有使用虚拟环境，以至于又重新制作了镜像；
第二，尝试了cuda10.2+python3.7.3+A10卡的配置，虽然可以安装成功所有库，但依旧会报错CUDA类的错误，这大概是由于cuda版本太低在A10卡上无法运行所导致；
第三，之后本人又尝试了python3.6的版本也依旧行不通...
最终，本人制作了此完美的镜像，关于镜像的参数写在下方。
```
### 8.2 镜像参数信息
- 镜像获取方式：docker pull 
- 镜像名称：recognition
- 镜像中包含库信息：
    - Anaconda3-2020.11-Linux-x86_64.sh
    - opencv-python 4.6.0.66
    - pytorch 1.8.1+cu111
    - bcolz 1.2.1
    - python 3.8.5

### 8.3 环境制作流程
- step1: 首选寻找一个cuda的基础镜像，docker pull之后，docker run成容器，在容器中进行测试；
- step2: 执行conda env list等操作查看是否安装了conda，如果没有则安装适当版本的anaconda（安装完之后重启docker）；
- step3: 接着直接测试test_quality.py看是否可以run，如果不行，则安装库，一般需要安装pytorch及cv2（此测试与安装一定要在conda环境中安装，即base环境）；
- step4：安装bcolz准备训练，必须采用conda方式安装；
- step5：缺少什么库使用pip安装即可，直至可以正常训练 ；
- step6: 安装一些小插件，如python vscode调试，使用ssh进行vscode远程等等。
