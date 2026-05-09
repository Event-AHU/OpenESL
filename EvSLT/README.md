**Event Stream-based Sign Language Translation: A High-Definition Benchmark Dataset and A Novel Baseline**

<div align="center">

<img src="https://github.com/Event-AHU/OpenESL/blob/main/EvSLT/pictures/EventSLT_framework.jpg" width="800">

</div>

# :dvd: Download the Event-CSL dataset 

* **BaiduYun**: 
```
Raw bin file obtained from Prophesee EVK4-HD Event camera： 
链接：https://pan.baidu.com/s/11yGZOhF2IpJGi0D5aOHQSA?pwd=1234 
提取码：1234

Processed Event frames： 
链接：https://pan.baidu.com/s/1gOGoxHc_4SGpR5h2T2t9SQ?pwd=bdnn
提取码：bdnn 
```

The directory should have the following format:
```Shell
├── Event_CSL
    ├── SL_image
        ├── 0001
            ├── 0000.png
            ├── 0001.png
            ├── ...
        ├── 0002
        ├── ...
    ├── label
        ├── train.pkl
        ├── val.pkl
        ├── test.pkl
```


# :hammer: Environment 

Raw results (Including training logs, checkpoints, and test results) :
https://pan.baidu.com/s/1xCXYMR4qgxYHlhJqPlg0pg?pwd=AHUE

```
conda create -n eventslt python=3.10.13
conda activate eventslt
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```


## Train & Test

Download pretrain.zip from https://pan.baidu.com/s/1ktTTv5aNOFv_codBCzTXEA (code: paxx) and put it in the EvSLT directory. 

```
# train:
bash run.sh

# test:
bash test_run.sh
```
