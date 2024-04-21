This repo is to pratice the DLRM model with the following dataset:
the categorical feature encoding challenge from Kaggle: https://www.kaggle.com/competitions/cat-in-the-dat-ii/data
The DLRM model is from the following paper: https://arxiv.org/abs/1906.00091. 

The structure of the repo is as follows:
```
DL_genAI_practicing/
  |- reco_model/
  |  |- README.md
  |  |- dlrmnet.py: the DLRM model in pytorch
  |  |- reco_data.py: the data processing and loading
  |  |- train.py: the training script
```
When training the dlrm net, we are using the pytorch lightning library to facilitate the training process. Detailed information about the pytorch lightning library can be found here: https://lightning.ai/docs/pytorch/stable/ 
