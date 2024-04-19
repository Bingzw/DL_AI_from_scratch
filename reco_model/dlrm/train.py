import torch
import os
# lighting
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# model
from dlrm_data import DLRMDataModule
from dlrmnet import DLRMModule


if __name__ == "__main__":
    # Set seed
    SEED = 42
    CHECKPOINT_PATH = "../saved_models/reco_models"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    data_path = "../data/reco_data/sampled_criteo_data.csv"

    # set hyperparameters
    batch_size = 64
    hidden_dim = 10
    learning_rate = 0.001
    num_epochs = 30
    # create data module
    dlrm_data = DLRMDataModule(data_path, batch_size=batch_size, hidden_dim=hidden_dim)

    bottom_mlp_dims = [64, 32]
    top_mlp_dims = [64, 32]
    num_numerical_features = dlrm_data.data.dense_features.shape[1]
    embedding_sizes = dlrm_data.embedding_sizes
    save_name = "dlrmnet"
    # create data loader
    dlrm_data.setup()
    train_loader = dlrm_data.train_dataloader()
    val_loader = dlrm_data.val_dataloader()
    test_loader = dlrm_data.test_dataloader()
    # Initialize TensorBoard logger
    logger = pl.loggers.TensorBoardLogger("logs", name="dlrm_module")
    # define trainer
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),  # Where to save models
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         # We run on a GPU (if possible)
                         devices=1,  # How many GPUs/CPUs we want to use (1 is enough for the notebooks)
                         max_epochs=num_epochs,  # How many epochs to train for if no patience is set
                         #logger=logger,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                    LearningRateMonitor("epoch")],
                         enable_progress_bar=True)  # Set to False if you do not want a progress bar

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = DLRMModule.load_from_checkpoint(
            pretrained_filename,
            num_numerical_features=num_numerical_features,
            embedding_sizes=embedding_sizes,
            bottom_mlp_dims=bottom_mlp_dims,
            top_mlp_dims=top_mlp_dims
        )  # Automatically loads the model with the saved hyperparameters
    else:
        pl.seed_everything(SEED)  # To be reproducable
        model = DLRMModule(num_numerical_features=num_numerical_features, embedding_sizes=embedding_sizes,
                           bottom_mlp_dims=bottom_mlp_dims, top_mlp_dims=top_mlp_dims, lr=learning_rate)
        trainer.fit(model, train_loader, val_loader)
        model = DLRMModule.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            num_numerical_features=num_numerical_features,
            embedding_sizes=embedding_sizes,
            bottom_mlp_dims=bottom_mlp_dims,
            top_mlp_dims=top_mlp_dims
        )  # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    print("Best model checkpoint path:", trainer.checkpoint_callback.best_model_path)
    print(result)

    # run this in terminal to check logs: tensorboard --logdir saved_models/dlrm_net/lightning_logs



