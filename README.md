FRLIE: Low-light Image Enhancement via Feature Restoration
=================================
This is the PyTorch implementation of paper FRLIE: Low-light Image Enhancement via Feature Restoration


<img src="examples/37.jpg" width = "366" height = "240" alt="37_low" align=center /> <img src="examples/37_high.png" width = "366" height = "240" alt="37_high" align=center />

Prerequisites
---------------------------------
* Python 3.7
* Pytorch 1.7
* NVIDIA GPU + CUDA cuDNN

Installation
---------------------------------
* Clone this repo:
```
git clone https://github.com/YaN9-Y/FRLIE
```
* Install Pytorch
* Install python requirements:
```
pip install -r requirements.txt
```

### 0.Quick Testing
To hold a quick-testing of our model, just run:
```
python3 test.py --model 2 --checkpoints ./checkpoints/quick_test
```
and check the results in `checkpoints/quick_test/results/feature_process`.

Citation
-------------------------------------
If you find our paper useful or used our code, please consider citing:
```
@INPROCEEDINGS{FRLIE,
  author={Yang, Yang and Zhang, Yonghua and Guo, Xiaojie},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Low-Light Image Enhancement via Feature Restoration}, 
  year={2022},
  pages={2440-2444},
  doi={10.1109/ICASSP43922.2022.9747174}
  }
```
