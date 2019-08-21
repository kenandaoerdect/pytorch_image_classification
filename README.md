# Image classification in PyTorch
## Prerequisites
* PyTorch 0.4+
* Python 3.5+
## Datasets
Put your datasets in `./data/train/`. As follows:  
  
```
├─data
│  ├─abyssinian
│  │      Abyssinian_0.jpg
│  │      Abyssinian_1.jpg
│  │      ...
│  │      
│  ├─american_bulldog
│  │      american_bulldog_0.jpg
│  │      american_bulldog_1.jpg
│  │      ...
│  │      
│  ├─Birman
│  │      Birman_0.jpg
│  │      Birman_1.jpg
│  │      ...
│  │      
│  ├─Sphynx
│  │      Sphynx_0.jpg
│  │      Sphynx_1.jpg
│  │      ...
│  │      
│  └─yorkshire_terrier
│          yorkshire_terrier_0.jpg
│          yorkshire_terrier_1.jpg
│          ...
```  
  
## Usage
Train your classifier: `$ python train.py`.  
  
Test your classifier: `$ python test.py`.