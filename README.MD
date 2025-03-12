<p align="center">
<img src="https://github.com/Event-AHU/OpenESL/blob/main/EventCSL_gif.gif" width="500">
</p>

## OpenESL: Sign Language Translation/Recognition using Event Camera 

## :dart: Update Log: 

* [Aug-19, 2024] Event-CSL Benchmark Dataset for Sign Language Translation is released. 



## :dart: [Event-CSL Benchmark Dataset]() 
[arXiv:2408.10488] **Event Stream based Sign Language Translation: A High-Definition Benchmark Dataset and A New Algorithm**, 
Xiao Wang, Yao Rong, Fuling Wang, Jianing Li, Lin Zhu, Bo Jiang, Yaowei Wang, arXiv 2024,   
[[Paper](https://arxiv.org/abs/2408.10488)] 

Sign Language Translation (SLT) is a core task in the field of AI-assisted disability. Unlike traditional SLT based on visible light videos, which is easily affected by factors such as lighting, rapid hand movements, and privacy breaches, this paper proposes the use of high-definition Event streams for SLT, effectively mitigating the aforementioned issues. This is primarily because Event streams have a high dynamic range and dense temporal signals, which can withstand low illumination and motion blur well. Additionally, due to their sparsity in space, they effectively protect the privacy of the target person. 
More specifically, we propose a new high-resolution Event stream sign language dataset, termed Event-CSL, which effectively fills the data gap in this area of research. It contains 14,827 videos, 14,821 glosses, and 2,544 Chinese words in the text vocabulary. These samples are collected in a variety of indoor and outdoor scenes, encompassing multiple angles, light intensities, and camera movements. We have benchmarked existing mainstream SLT works to enable fair comparison for future efforts. Based on this dataset and several other large-scale datasets, we propose a novel baseline method that fully leverages the Mamba model's ability to integrate temporal information of CNN features, resulting in improved sign language translation outcomes. 

![Baseline](https://github.com/Event-AHU/OpenESL/blob/main/Event_CSL/figures/EventSLT_framework.jpg)
![Baseline](https://github.com/Event-AHU/OpenESL/blob/main/Event_CSL/figures/EventSLT_demos.jpg)






#### Other Materials 
* **[Paper with code]**: https://paperswithcode.com/task/sign-language-recognition/codeless  
* **[Awesome-Sign-Language]**: https://github.com/ZechengLi19/Awesome-Sign-Language

<div align="center">
  <video src="https://github.com/Event-AHU/OpenESL/blob/main/sign_demo" width="100%" poster=""> </video>
</div>





## :newspaper: Citation 
If you find this work helps your research, please star this GitHub and cite the following papers: 
```bibtex
@misc{wang2024EventCSL,
      title={Event Stream based Sign Language Translation: A High-Definition Benchmark Dataset and A New Algorithm}, 
      author={Xiao Wang and Yao Rong and Fuling Wang and Jianing Li and Lin Zhu and Bo Jiang and Yaowei Wang},
      year={2024},
      eprint={2408.10488},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.10488}, 
}

```

If you have any questions about these works, please feel free to leave an issue. 


