
<p align="center">
  <img src="img/architecture2.jpg">
</p>

# DocAligner 
This repository contains the codes for [**DocAligner**](xxx).


## environment
```
conda env create -f environment.yml
```
The correlation layer is implemented in CUDA using CuPy, so CuPy is a required dependency. It can be installed using pip install cupy or alternatively using one of the provided binary packages as outlined in the CuPy repository. The code was developed using Python 3.7 & PyTorch 1.11 & CUDA 11.2, which is why I installed cupy for cuda112. For another CUDA version, change accordingly.
```
pip install cupy-cuda112 --no-cache-dir
```


## Training
```
train_efficient_glu_gru.py
```

## Inference:
1. Data preparation: Source and target images need to be put in `./data/dataset1/all_data/`. The names of source captured images should end with `_origin`, while names of target clean images should end with `_target`. The names of paired source and target should be the same except for the different name endings.

2. Perform non-rigid pre-alignment
```
python ./data/preprocess/MBD/infer.py --im_folder ./data/dataset1/all_data/
```
3. Inference. Note that we offer 3 modes, which are 1: no self-supervision, 2: self-supervised optimization for each image separately, and 3: self-supervised optimization for the whole dataset. Mode 2 works best but is very time consuming, mode 3 is a compromise between performance and efficiency. 
```
python infer_test_time_optimize.py --mode 2 --im_folder ./data/dataset1/all_data/
```
4. Obtain final grid based on grid from step 2 and 3. Such final grid correlates the source target clean image toward the source captured image. We provide a script for dewarping the source captured image based on this final grid: `data/dewarp.py`
```
python data/sum_backwardmap.py --im_folder ./data/dataset1/all_data/
```  
5. Annotation transform (COCO format). Please refer to `data/annotate_transform.py`. We also provide a script for COCO data visualization: `data/coco_visualize.py`



