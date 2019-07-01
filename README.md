# 2019 ACM RecSys Challenge RosettaAI Solution


<a href="https://rosetta.ai/"><img src="https://www.rosetta.ai/images/logo.png" height="90" ></a>
<a href="https://www.ntu.edu.tw/"><img src="https://upload.wikimedia.org/wikipedia/zh/thumb/4/4c/National_Taiwan_University_logo.svg/1200px-National_Taiwan_University_logo.svg.png"  height="120"></a>
<a href="https://www.utdallas.edu/"><img src="https://yt3.ggpht.com/a/AGF-l7-x9pb2HmLWEJxTncC5EjzekRKX9I-qpX4nXg=s900-mo-c-c0xffffffff-rj-k-no"  height="130"></a>


## Team members 
_Kung-hsiang (Steeve), Huang_ __(Rosetta.ai)__; _Yi-fu, Fu_; _Yi-ting, Lee_; _Zong-han, Lee_; _Yao-chun, Jan_ __(National Taiwan University)__; _Yi-hui, Lee_ __(University of Texas at Dallas)__

Contact: steeve@rosetta.ai



## Introduction
This repository contains RosettaAI's approach to the 2019 ACM Recys Challenge. Instead of treating it as a ranking problem, we use Binary Cross Entropy as our loss function. Three different models were implemented:
1. Neural Networks (based on [DeepFM](https://arxiv.org/pdf/1804.04950.pdf))
2. LightGBM 
3. XGBoost



## Environment
* Nvidia Tesla V100
* CUDA 9.0 
* Python==3.6.8
* Numpy==1.16
* Pandas==0.24.2
* PyTorch==1.1.0  
* Sklearn==0.21.2
* Scipy==1.3.0
* LightGBM==2.2.4
* XGBoost == 0.9
