import pandas as pd
import torch
import os
from sklearn.preprocessing import LabelEncoder
# lighting
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# model
from reco_model.criteo_data import CriteoDataModule
from reco_model.deepfm.dfmnet import DeepFMModule














if __name__ == "__main__":
    seed = 42
    checkpoint_path = "../../saved_models/reco_models"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # data cleaning
    user_path = "../../data/reco_data/ml-1m/users.dat"
    movie_path = "../../data/reco_data/ml-1m/movies.dat"
    rating_path = "../../data/reco_data/ml-1m/ratings.dat"


