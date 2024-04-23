import os
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data


class ContrastiveTransformations(object):
    def __init__(self, base_transformation, n_views=2):
        """
        :param base_transformation: the transformation that should be applied to the input data
        :param n_views: the number of transformations that should be applied to the input data
        """
        self.base_transformation = base_transformation
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transformation(x) for _ in range(self.n_views)]


@torch.no_grad()
def prepare_data_features(model, dataset):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # Prepare model
    network = deepcopy(model.convnet)
    network.fc = nn.Identity()  # Removing projection head g(.)
    network.eval()
    network.to(device)

    # Encode all images
    data_loader = data.DataLoader(dataset, batch_size=64, num_workers=os.cpu_count(), shuffle=False, drop_last=False)
    feats, labels = [], []
    for batch_imgs, batch_labels in tqdm(data_loader):
        batch_imgs = batch_imgs.to(device)
        batch_feats = network(batch_imgs)
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    # Sort images by labels
    labels, idxs = labels.sort()
    feats = feats[idxs]

    return data.TensorDataset(feats, labels)


def get_smaller_dataset(original_dataset, num_imgs_per_label):
    new_dataset = data.TensorDataset(
        *[t.unflatten(0, (10, -1))[:,:num_imgs_per_label].flatten(0, 1) for t in original_dataset.tensors]
    )
    return new_dataset