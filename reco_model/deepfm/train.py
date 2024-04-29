import torch
import os
# lighting
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# model
from reco_model.criteo_data import CriteoDataModule
from reco_model.deepfm.dfmnet import DeepFMModule


class EarlyStoppingOnAucDifference(pl.Callback):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if 'train_auc' in metrics and 'val_auc' in metrics:
            auc_diff = metrics['train_auc'] - metrics['val_auc']
            if auc_diff > self.threshold:
                trainer.should_stop = True


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
    hidden_dim = 4
    learning_rate = 0.00005
    num_epochs = 30
    dropout_rate = 0.6
    # create data module
    criteo_data = CriteoDataModule(data_path, batch_size=batch_size, hidden_dim=hidden_dim)
    print(criteo_data.dataset.features.head())

    mlp_dims = [16, 8]
    num_dense_features = criteo_data.dataset.dense_features.shape[1]
    embedding_sizes = criteo_data.embedding_sizes
    save_directory_name = "dfmnet"
    pretrained_model_name = ".ckpt"  # the correct path should be
    # pretrained_model_name = "/lightning_logs/version_0/checkpoints/epoch=77-step=36582.ckpt"
    # create data loader
    criteo_data.setup()
    train_loader = criteo_data.train_dataloader()
    val_loader = criteo_data.val_dataloader()
    test_loader = criteo_data.test_dataloader()
    # define trainer
    checkpoint_callback = ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_auc")
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_directory_name),  # Where to save models
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         # We run on a GPU (if possible)
                         devices=1,  # How many GPUs/CPUs we want to use (1 is enough for the notebooks)
                         max_epochs=num_epochs,  # How many epochs to train for if no patience is set
                         callbacks=[
                                    EarlyStoppingOnAucDifference(threshold=0.1),
                                    checkpoint_callback,
                                    # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                    LearningRateMonitor("epoch")],
                         enable_progress_bar=True)  # Set to False if you do not want a progress bar

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_directory_name + pretrained_model_name)
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = DeepFMModule.load_from_checkpoint(pretrained_filename)  # Automatically loads the model with the saved hyperparameters
    else:
        pl.seed_everything(SEED)  # To be reproducable
        model = DeepFMModule(num_dense_features=num_dense_features, embedding_sizes=embedding_sizes,
                             mlp_dims=mlp_dims, lr=learning_rate)
        trainer.fit(model, train_loader, val_loader)
        model = DeepFMModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training

    # Test best model on validation and test set
    test_result = trainer.test(model, test_loader, verbose=False)
    print("Best model checkpoint path:", trainer.checkpoint_callback.best_model_path)
    print("test_result: ", test_result[0])

    # run this in terminal to check logs: tensorboard --logdir saved_models/reco_models/dfmnet/lightning_logs