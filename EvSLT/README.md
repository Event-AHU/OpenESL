**Event Stream-based Sign Language Translation: A High-Definition Benchmark Dataset and A Novel Baseline**

<div align="center">

<img src="https://github.com/Event-AHU/OpenESL/blob/main/EvSLT/pictures/EventSLT_framework.jpg" width="800">

</div>

## Download the Event-CSL dataset 

* **BaiduYun**: 
```
Raw bin file obtained from Prophesee EVK4-HD Event camera： 
链接：https://pan.baidu.com/s/11yGZOhF2IpJGi0D5aOHQSA?pwd=1234 
提取码：1234
```

The directory should have the following format:
```Shell
├── Event_CSL
    ├── SL_raw_image 14827 videos (training subset: 12,602 videos; valid subset: 741 videos; testing subset: 1,484 videos;))
        ├── 0001
            ├── 0000.png
            ├── 0001.png
            ├── ...
        ├── 0002
            ├── 0000.png
            ├── 0001.png
            ├── ...
        ├── 0003
        ├── ...
    ├── label
        ├── train.pkl
        ├── dev.pkl
        ├── test.pkl
```


## Environment Setting 
```
conda create -n eventslt python=3.10.13
conda activate eventslt
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Train & Test
```
Train:
cd Event_CSL
bash run.sh

Test：
cd Event_CSL
bash test_run.sh
```
