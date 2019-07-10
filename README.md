# The 5th Place Solution to the 2019 ACM RecSys Challenge


<a href="https://rosetta.ai/"><img src="https://www.rosetta.ai/images/logo.png" height="90" ></a>
<a href="https://www.ntu.edu.tw/"><img src="https://upload.wikimedia.org/wikipedia/zh/thumb/4/4c/National_Taiwan_University_logo.svg/1200px-National_Taiwan_University_logo.svg.png"  height="120"></a>
<a href="https://www.utdallas.edu/"><img src="https://yt3.ggpht.com/a/AGF-l7-x9pb2HmLWEJxTncC5EjzekRKX9I-qpX4nXg=s900-mo-c-c0xffffffff-rj-k-no"  height="130"></a> 


## Team Members 
_Kung-hsiang (Steeve), Huang_ __(Rosetta.ai)__; _Yi-fu, Fu_; _Yi-ting, Lee_; _Tzong-hann, Lee_; _Yao-chun, Chan_ __(National Taiwan University)__; _Yi-hui, Lee_ __(University of Texas at Dallas)__; _Shou-de, Lin_ __(National Taiwan University)__

Contact: steeve@rosetta.ai



## Introduction
This repository contains RosettaAI's approach to the 2019 ACM Recys Challenge ([accompanying writeup](https://medium.com/@huangkh19951228/the-5th-place-approach-to-the-2019-acm-recsys-challenge-by-team-rosettaai-eb3c4e6178c4)). Instead of treating it as a ranking problem, we use __Binary Cross Entropy__ as our loss function. Three different models were implemented:
1. Neural Networks (based on [DeepFM](https://arxiv.org/pdf/1804.04950.pdf) and this [Youtube paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf))
2. LightGBM 
3. XGBoost



## Environment
* Ubuntu 16.04
* CUDA 9.0 
* Python==3.6.8
* Numpy==1.16
* Pandas==0.24.2
* PyTorch==1.1.0  
* Sklearn==0.21.2
* Scipy==1.3.0
* LightGBM==2.2.4
* XGBoost==0.9
* timezonefinder==4.0.3
* geopy==1.20.0

## Project Structure

```
├── input
├── output
├── src
└── weights
```

## Setup
Run the following commands to create directories that conform to the structure of the project, then place the unzipped data into the ```input``` directory.:

```. setup.sh```



Run the two python scripts to picklize the input data and obtain the utc offsets from countries:
```
cd src
python picklization.py
python country2utc.py
```

To enable the model to train on the whole data, set ```debug``` and ```subsample``` to ```False``` in the ```config.py``` file.

```
class Configuration(object):

    def __init__(self):
        ...
        self.debug = False
        self.sub_sample = False
        ...
```


## Training & Submission

The models are all trained in an end-to-end fashion. To train and predict each of the three models, simply run the following commands:
```
python run_nn.py
python run_lgb.py
python run_xgb.py
```
The submission files are stored in the ```output``` directory. 

The results generated from LightGBM alone would place us at the 5th position in the public leaderboard. To ensemble these three models, change the output name of each model in ```Merge.ipynb``` and run it.


## Performance

| Model        | Local Validation MRR           | Public Leaderboard MRR  |
| ------------- |-------------:| -----:|
| LightGBM      | 0.685787 | N/A |
| XGBoost      | 0.684521      |   0.681128  |
| NN | 0.675206      |    0.672117  |
