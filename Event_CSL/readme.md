
<p align="center">
  <strong style="font-size:20px;">Event-CSL</strong>
  A Large-scale High-Definition Benchmark Dataset for Event-based Sign Language Translation 
</p>

## Abstract 
Sign Language Translation (SLT) is a core task in the field of AI-assisted disability. Unlike traditional SLT based on visible light videos, which is easily affected by factors such as lighting, rapid hand movements, and privacy breaches, this paper proposes the use of high-definition Event streams for SLT, effectively mitigating the aforementioned issues. This is primarily because Event streams have a high dynamic range and dense temporal signals, which can withstand low illumination and motion blur well. Additionally, due to their sparsity in space, they effectively protect the privacy of the target person. More specifically, we propose a new high-resolution Event stream sign language dataset, termed Event-CSL, which effectively fills the data gap in this area of research. It contains 14,827 videos, 14,821 glosses, and 2,544 Chinese words in the text vocabulary. These samples are collected in a variety of indoor and outdoor scenes, encompassing multiple angles, light intensities, and camera movements. We have benchmarked existing mainstream SLT works to enable fair comparison for future efforts. Based on this dataset and several other large-scale datasets, we propose a novel baseline method that fully leverages the Mamba model’s ability to integrate temporal information of CNN features, resulting in improved sign language translation outcomes.


## Sign Language Demo

<p align="center">
  <a>
    <img src="https://github.com/Event-AHU/OpenESL/blob/main/Event_CSL/figures/EventSLT_demos.jpg" alt="DemoVideo" width="800"/>
  </a>
</p>



## A Hybrid CNN-Mamba Framework for Event-based Sign Language Translation
<div align="center">
<img src="https://github.com/Event-AHU/OpenESL/blob/main/Event_CSL/figures/EventSLT_framework.jpg" width="800">  
</div>


## Quality_Analysis
<div align="center">
<img src="https://github.com/Event-AHU/OpenESL/blob/main/Event_CSL/figures/quality_analysis.jpg" width="800">  
</div>


## Download the Event-CSL dataset 

* **BaiduYun**: 
```
raw bin file obtained from Prophesee EVK4-HD Event camera： 
Link：  Password：
```

The directory should have the below format:
```Shell
├── Event_CSL
    ├── SL_raw_image 14827 videos (training subset: 12,602 videos; valid subset: 741 videos; testing subset: 1,484 videos;))
        ├── 0001
            ├── 0000.png
            ├── 0001.png
            ├── 0002.png
            ├── ...
        ├── 0002
            ├── 0000.png
            ├── 0001.png
            ├── 0002.png
            ├── ...
        ├── 0003
        ├── ...
    ├── annotations
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




## Acknowledgement 
### Our code is implemented based on <a href="https://github.com/zhoubenjia/GFSLT-VLP">GFSLT-VLP</a>, <a href="https://github.com/hustvl/Vim">Vim</a>.


## Citation 

If you find this work helps your research, please cite the following paper and give us a star. 


Please leave an **issue** if you have any questions about this work. 



