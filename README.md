
<div align=center>

# DocAligner: Automating the Annotation of Photographed Documents Through Real-virtual Alignment

</div>

<p align="center">
  <img src="img/motivation.jpg" width='500'>
</p>

## News 
🔥 A comprehensive [Recommendation for Document Image Processing](https://github.com/ZZZHANG-jx/Recommendations-Document-Image-Processing) is available.


## Environment
```
conda env create -f environment.yml
```
The correlation layer is implemented in CUDA using CuPy, so CuPy is a required dependency. It can be installed using pip install cupy or alternatively using one of the provided binary packages as outlined in the CuPy repository. The code was developed using Python 3.7 & PyTorch 1.11 & CUDA 11.2, which is why I installed cupy for cuda112. For another CUDA version, change accordingly.
```
pip install cupy-cuda112 --no-cache-dir
``` 

## Training
Coming soon
<!-- ```
python train_efficient.py
``` -->


## Inference
1. Weight preparation:
    * Put non-rigid pre-alignment module weights [mbd.pkl](https://1drv.ms/f/s!Ak15mSdV3Wy4ibRvLXIMbJoIzkpSpQ?e=mkiGBp) to `data/MBD/checkpoint/`
    * Put hierarchical alignment module weights [checkpoint.pkl](https://1drv.ms/f/s!Ak15mSdV3Wy4ibRvLXIMbJoIzkpSpQ?e=mkiGBp) to `checkpoint/docaligner/`

2. Data preparation: 
    * Source and target images need to be put in `data/example`. 
    * The names of source captured images should end with `_origin`, while names of target clean images should end with `_target`. 
    * The names of paired source and target should be the same except for the different name endings.
    * Put some flows into `data/` for augmentation in the subsequent self-supervised steps.
    Put some [flow map](https://1drv.ms/f/s!Ak15mSdV3Wy4ibRvLXIMbJoIzkpSpQ?e=mkiGBp) to `data/augmentation_flow`, which can assist the following self-supervision steps (step 4).

3. Perform non-rigid pre-alignment
```
python ./data/preprocess/MBD/infer.py --im_folder ./data/example
```
4. Inference. Note that we offer 3 modes, which are 
    * 1: no self-supervision
    * 2: self-supervised optimization for each image separately
    * 3: self-supervised optimization for the whole dataset. For the vast majority of cases, Mode 2 works best but is very time consuming, mode 3 is a compromise between performance and efficiency. 
```
python infer.py --mode 3 --im_folder ./data/example
```
5. Obtain final grid based on grid from step 2 and 3. Such final grid correlates the source target clean image toward the source captured image.
```
python tools/sum_backwardmap.py --im_folder ./data/example
```  
6. Utilization of final grid .
    * Annotation transform (COCO format): Please refer to `tools/annotate_transform.py`. We also provide a script for COCO data visualization: `tools/coco_visualize.py`.
    * Dewarping: We provide a script for dewarping the source captured image based on the final grid: `tools/dewarp.py`



## DocAligner-acquired Dataset
The dataset mentioned in our paper, obtained using DocAligner, for document layout analysis, table structure recognition, illumination correction, binarization and geometric rectification tasks can be downloaded [**here**](https://arxiv.org/abs/2306.05749).





## Citation
If you are using our code and data, please consider citing our paper.
```
@article{zhang2023docaligner,
title={DocAligner: Annotating Real-world Photographic Document Images by Simply Taking Pictures},
author={Zhang, Jiaxin and Chen, Bangdong and Cheng, Hiuyi and Guo, Fengjun and Ding, Kai and Jin, Lianwen},
journal={arXiv preprint arXiv:2306.05749},
year={2023}}
```


## ⭐ Star Rising
[![Star Rising](https://api.star-history.com/svg?repos=ZZZHANG-jx/DocAligner&type=Timeline)](https://star-history.com/#ZZZHANG-jx/DocAligner&Timeline)