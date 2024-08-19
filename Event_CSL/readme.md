#### Event-CSL 
A Large-scale High-Definition Benchmark Dataset for Event-based Sign Language Translation 

<div align="center">
<img src="https://github.com/Event-AHU/OpenESL/blob/main/Event_CSL/figures/videonumber_resolution.jpg" width="450"><img src= "https://github.com/Event-AHU/OpenESL/blob/main/Event_CSL/figures/EventSLT_demos.jpg" width="450">
</div>


## Abstract 
Sign Language Translation (SLT) is a core task in the field of AI-assisted disability. Unlike traditional SLT based on visible light videos, which is easily affected by factors such as lighting, rapid hand movements, and privacy breaches, this paper proposes the use of high-definition Event streams for SLT, effectively mitigating the aforementioned issues. This is primarily because Event streams have a high dynamic range and dense temporal signals, which can withstand low illumination and motion blur well. Additionally, due to their sparsity in space, they effectively protect the privacy of the target person. More specifically, we propose a new high-resolution Event stream sign language dataset, termed Event-CSL, which effectively fills the data gap in this area of research. It contains 14,827 videos, 14,821 glosses, and 2,544 Chinese words in the text vocabulary. These samples are collected in a variety of indoor and outdoor scenes, encompassing multiple angles, light intensities, and camera movements. We have benchmarked existing mainstream SLT works to enable fair comparison for future efforts. Based on this dataset and several other large-scale datasets, we propose a novel baseline method that fully leverages the Mamba model’s ability to integrate temporal information of CNN features, resulting in improved sign language translation outcomes.


## A Hybrid CNN-Mamba Framework for Event-based Sign Language Translation
<div align="center">
<img src="https://github.com/Event-AHU/OpenESL/blob/main/Event_CSL/figures/EventSLT_framework.jpg" width="800">  
</div>


## Environment Setting 


## Train & Test
```
Train:
cd Event_CSL
bash run.sh

Test：
cd Event_CSL
bash test_run.sh
```

## Experimental Results and Visualization


## Acknowledgement 
### Our code is implemented based on <a href="https://github.com/open-mmlab/mmaction2">MMAction2</a>.


## Citation 

If you find this work helps your research, please cite the following paper and give us a star. 


Please leave an **issue** if you have any questions about this work. 



