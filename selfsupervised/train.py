import os
from copy import deepcopy

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import matplotlib.pyplot as plt

## Torchvision
import torchvision
from torchvision.datasets import STL10
from torchvision import transforms

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# SimCLR Model
from simclr import SimCLR
from logisticregression import LogisticRegression
from selfsupervised.util import ContrastiveTransformations, prepare_data_features, get_smaller_dataset


def train_simclr(batch_size, max_epochs=500, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, 'SimCLR'),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc_top5'),
                                    LearningRateMonitor('epoch')])
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, 'SimCLR.ckpt')
    if os.path.isfile(pretrained_filename):
        print(f'Found pretrained model at {pretrained_filename}, loading...')
        model = SimCLR.load_from_checkpoint(
            pretrained_filename)  # Automatically loads the model with the saved hyperparameters
    else:
        train_loader = data.DataLoader(unlabeled_data, batch_size=batch_size, shuffle=True,
                                       drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
        val_loader = data.DataLoader(train_data_contrast, batch_size=batch_size, shuffle=False,
                                     drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
        pl.seed_everything(42)  # To be reproducable
        model = SimCLR(max_epochs=max_epochs, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = SimCLR.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training

    return model


def train_logreg(batch_size, train_feats_data, test_feats_data, model_suffix, max_epochs=100, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "LogisticRegression"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc'),
                                    LearningRateMonitor("epoch")],
                         enable_progress_bar=False,
                         check_val_every_n_epoch=10)
    trainer.logger._default_hp_metric = None

    # Data loaders
    train_loader = data.DataLoader(train_feats_data, batch_size=batch_size, shuffle=True,
                                   drop_last=False, pin_memory=True, num_workers=0)
    test_loader = data.DataLoader(test_feats_data, batch_size=batch_size, shuffle=False,
                                  drop_last=False, pin_memory=True, num_workers=0)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"LogisticRegression_{model_suffix}.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = LogisticRegression.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = LogisticRegression(**kwargs)
        trainer.fit(model, train_loader, test_loader)
        model = LogisticRegression.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on train and validation set
    train_result = trainer.test(model, train_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"train": train_result[0]["test_acc"], "test": test_result[0]["test_acc"]}

    return model, result


if __name__ == "__main__":
    SEED = 42
    # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
    DATASET_PATH = "../data"
    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = "../saved_models/self_supervised"
    NUM_WORKERS = os.cpu_count()
    # Setting the seed
    pl.seed_everything(SEED)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    # apply transforms to the data
    contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomResizedCrop(size=96),
                                              transforms.RandomApply([
                                                  transforms.ColorJitter(brightness=0.5,
                                                                         contrast=0.5,
                                                                         saturation=0.5,
                                                                         hue=0.1)
                                              ], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.GaussianBlur(kernel_size=9),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5,), (0.5,))
                                              ])

    # Load the dataset
    unlabeled_data = STL10(root=DATASET_PATH, split='unlabeled', download=True,
                           transform=ContrastiveTransformations(contrast_transforms, n_views=2))  # used for training,
    # returned as a tuple of images, labels. The structure looks like this:
    # ([transformed_image_1, transformed_image_2], label=-1)
    train_data_contrast = STL10(root=DATASET_PATH, split='train', download=True,
                                   transform=ContrastiveTransformations(contrast_transforms, n_views=2))  # used for validation

    simclr_model = train_simclr(batch_size=256, hidden_dim=128, lr=5e-4, temperature=0.07, weight_decay=1e-4,
                                max_epochs=1)

    # run a downstream classification task based on the learned representations
    img_transforms = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))])

    train_img_data = STL10(root=DATASET_PATH, split='train', download=True,
                           transform=img_transforms)
    test_img_data = STL10(root=DATASET_PATH, split='test', download=True,
                          transform=img_transforms)

    print("Number of training examples:", len(train_img_data))
    print("Number of test examples:", len(test_img_data))

    train_feats_simclr = prepare_data_features(simclr_model, train_img_data)
    test_feats_simclr = prepare_data_features(simclr_model, test_img_data)

    results = {}
    for num_imgs_per_label in [10, 20, 50, 100, 200, 500]:
        sub_train_set = get_smaller_dataset(train_feats_simclr, num_imgs_per_label)
        _, small_set_results = train_logreg(batch_size=64,
                                            train_feats_data=sub_train_set,
                                            test_feats_data=test_feats_simclr,
                                            model_suffix=num_imgs_per_label,
                                            feature_dim=train_feats_simclr.tensors[0].shape[1],
                                            num_classes=10,
                                            lr=1e-3,
                                            weight_decay=1e-3)
        results[num_imgs_per_label] = small_set_results
    print(results)

