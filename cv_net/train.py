import os
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
# torch vision
from torchvision.datasets import CIFAR10
from torchvision import transforms
from cv_net.util.util import set_seed
# lighting
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from cv_net.util.util import create_model
# model
from googlenet import GoogleNet
from resnet import ResNet
from densenet import DenseNet


class CIFARModule(pl.LightningModule):
    def __init__(self, model_dict, model_name, model_hparams, optimizer_name, optimizer_hparams):
        """
        :param model_name: Name of the model/CNN to run. Used for creating the model
        :param model_hparams: Hyperparameters for the model, as dictionary.
        :param optimizer_name: Name of the optimizer to use. Currently supported: Adam, SGD
        :param optimizer_hparams: Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = create_model(model_dict, model_name, model_hparams)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(
                self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f"Unknown optimizer: \"{self.hparams.optimizer_name}\""

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log('test_acc', acc)


def train_model(model_lookup_dict, model_name, save_name=None, **kwargs):
    """
    :param model_lookup_dict: Dictionary that maps model names to model classes
    :param model_name: Name of the model you want to run. Is used to look up the class in "model_dict"
    :param save_name: If specified, this name will be used for creating the checkpoint and logging directory.
    :param kwargs:
    :return: model, result
    """

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),  # Where to save models
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         # We run on a GPU (if possible)
                         devices=1,  # How many GPUs/CPUs we want to use (1 is enough for the notebooks)
                         max_epochs=3,  # How many epochs to train for if no patience is set
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                    LearningRateMonitor("epoch")],  # Log learning rate every epoch
                         enable_progress_bar=True)  # Set to False if you do not want a progress bar
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = CIFARModule.load_from_checkpoint(
            pretrained_filename)  # Automatically loads the model with the saved hyperparameters
    else:
        pl.seed_everything(42)  # To be reproducable
        model = CIFARModule(model_dict=model_lookup_dict, model_name=model_name, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = CIFARModule.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result


if __name__ == "__main__":
    SEED = 42
    CHECKPOINT_PATH = "../saved_models/cv_net"
    # set seed
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # download dataset
    DATASET_PATH = "../data/"
    train_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True)
    data_means = (train_dataset.data / 255).mean(axis=(0, 1, 2))
    data_stds = (train_dataset.data / 255).std(axis=(0, 1, 2))
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Data mean: {data_means}")
    print(f"Data std: {data_stds}")

    # data augumentation
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(data_means, data_stds)])
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(data_means, data_stds)])
    train_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True, transform=train_transform)
    val_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True, transform=test_transform)
    set_seed(SEED)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    set_seed(SEED)
    val_set, _ = torch.utils.data.random_split(val_dataset, [45000, 5000])
    test_set = CIFAR10(root=DATASET_PATH, train=False, download=True, transform=test_transform)

    train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True,
                                   num_workers=4, persistent_workers=True)
    val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4,
                                 persistent_workers=True)
    test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4,
                                  persistent_workers=True)
    # train a simple google net

    model_lookup = {"GoogleNet": GoogleNet}
    googlenet_model, googlenet_result = train_model(model_lookup_dict=model_lookup,
                                                        model_name="GoogleNet",
                                                    save_name="googlenet",
                                                    model_hparams={"num_classes": 10, "act_fn_name": "relu"},
                                                    optimizer_name="Adam",
                                                    optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4})
    print("GoogleNet result:", googlenet_result)
    # train a simple resnet
    model_lookup["ResNet"] = ResNet
    resnet_model, resnet_result = train_model(model_lookup_dict=model_lookup,
                                              model_name="ResNet",
                                               model_hparams={"num_classes": 10,
                                                              "c_hidden": [16, 32, 64],
                                                              "num_blocks": [3, 3, 3],
                                                              "act_fn_name": "relu"},
                                              save_name="resnet",
                                               optimizer_name="SGD",
                                               optimizer_hparams={"lr": 0.1,
                                                                  "momentum": 0.9,
                                                                  "weight_decay": 1e-4})
    print("ResNet result:", resnet_result)

    # train a simple densenet
    model_lookup["DenseNet"] = DenseNet
    print(model_lookup)
    densenet_model, densenet_results = train_model(model_lookup_dict=model_lookup,
                                                   model_name="DenseNet",
                                                   model_hparams={"num_classes": 10,
                                                                  "num_layers": [3, 3],
                                                                  "bn_size": 2,
                                                                  "growth_rate": 4,
                                                                  "act_fn_name": "relu"},
                                                   save_name="densenet",
                                                   optimizer_name="Adam",
                                                   optimizer_hparams={"lr": 1e-3,
                                                                      "weight_decay": 1e-4})
    print("DenseNet result:", densenet_results)








