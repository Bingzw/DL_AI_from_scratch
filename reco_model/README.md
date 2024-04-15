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

Below is the training result of the DLRM model on the categorical feature encoding challenge dataset:

Train accuracy:

<img width="352" alt="Screen Shot 2024-04-14 at 1 22 34 AM" src="https://github.com/Bingzw/DL_genAI_practicing/assets/7736032/a3e6db14-88a8-4224-9b3f-c7eb87449a8c">

Val accuracy:

<img width="351" alt="Screen Shot 2024-04-14 at 1 22 41 AM" src="https://github.com/Bingzw/DL_genAI_practicing/assets/7736032/9b10d1b1-b369-45d6-8575-b12f13b29c71">

Test accuracy:

<img width="349" alt="Screen Shot 2024-04-14 at 1 22 52 AM" src="https://github.com/Bingzw/DL_genAI_practicing/assets/7736032/8cdfd7d2-f00e-455f-9172-0c3d87978c42">

Train loss:

<img width="351" alt="Screen Shot 2024-04-14 at 1 21 50 AM" src="https://github.com/Bingzw/DL_genAI_practicing/assets/7736032/d86ff427-6308-4786-872a-cedf7b00d743">

Val loss:

<img width="352" alt="Screen Shot 2024-04-14 at 1 21 59 AM" src="https://github.com/Bingzw/DL_genAI_practicing/assets/7736032/57a56630-48ad-475a-9c2a-1c39d4a3f36f">

Test loss:

<img width="354" alt="Screen Shot 2024-04-14 at 1 22 22 AM" src="https://github.com/Bingzw/DL_genAI_practicing/assets/7736032/bd4e1c4d-1bc9-4e4f-93c1-b3a651544a99">


