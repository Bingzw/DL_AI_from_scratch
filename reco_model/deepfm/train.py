import torch
import os
# lighting
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# model
from reco_model.criteo_data import CriteoDataModule
from reco_model.deepfm.dfmnet import DeepFMModule


if __name__ == "__main__":
    # Set seed
    SEED = 42
    CHECKPOINT_PATH = "../../saved_models/reco_models"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    data_path = "../../data/reco_data/sampled_criteo_data.txt"

    # set hyperparameters
    batch_size = 128
    hidden_dim = 10
    learning_rate = 0.0001
    num_epochs = 100
    # create data module
    criteo_data = CriteoDataModule(data_path, batch_size=batch_size, hidden_dim=hidden_dim)
    print(criteo_data.dataset.features.head())

    mlp_dims = [64, 32]
    num_dense_features = criteo_data.dataset.dense_features.shape[1]
    embedding_sizes = criteo_data.embedding_sizes
    save_name = "dfmnet"
    # create data loader
    criteo_data.setup()
    train_loader = criteo_data.train_dataloader()
    val_loader = criteo_data.val_dataloader()
    test_loader = criteo_data.test_dataloader()
    # define trainer
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),  # Where to save models
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         # We run on a GPU (if possible)
                         devices=1,  # How many GPUs/CPUs we want to use (1 is enough for the notebooks)
                         max_epochs=num_epochs,  # How many epochs to train for if no patience is set
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                    LearningRateMonitor("epoch")],
                         enable_progress_bar=True)  # Set to False if you do not want a progress bar

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = DeepFMModule.load_from_checkpoint(
            pretrained_filename,
            num_dense_features=num_dense_features,
            embedding_sizes=embedding_sizes,
            mlp_dims=mlp_dims
        )  # Automatically loads the model with the saved hyperparameters
    else:
        pl.seed_everything(SEED)  # To be reproducable
        model = DeepFMModule(num_dense_features=num_dense_features, embedding_sizes=embedding_sizes,
                             mlp_dims=mlp_dims, lr=learning_rate)
        trainer.fit(model, train_loader, val_loader)
        model = DeepFMModule.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            num_dense_features=num_dense_features,
            embedding_sizes=embedding_sizes,
            mlp_dims=mlp_dims
        )  # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"]/1000, "val": val_result[0]["test_acc"]/1000}
    print("Best model checkpoint path:", trainer.checkpoint_callback.best_model_path)
    print(result)

    # run this in terminal to check logs: tensorboard --logdir saved_models/reco_models/dfmnet/lightning_logs