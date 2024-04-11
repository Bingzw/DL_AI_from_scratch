import torch
import os
import pandas as pd
import torch.utils.data as data


class DLRMDataSet(data.Dataset):
    def __init__(self, data_path):
        super.__init__()
        self.data = pd.read_csv(data_path).fillna(0)







if __name__ == '__main__':
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    train_path = os.path.join(parent_directory, "data/reco_data/train.csv")
    train_data = pd.read_csv(train_path)

    # print
    shuffled_df = train_data.sample(frac=1, random_state=42)

    # Split the shuffled DataFrame into train and validation sets
    train_ratio = 0.8  # 80% of the data for training
    train_size = int(train_ratio * len(shuffled_df))
    train_df = shuffled_df[:train_size]
    val_df = shuffled_df[train_size:]


    print(train_df.shape)
    print(val_df.shape)
    train_df.to_csv(os.path.join(parent_directory, "data/reco_data/new_train.csv"), index=False)
    val_df.to_csv(os.path.join(parent_directory, "data/reco_data/val.csv"), index=False)
