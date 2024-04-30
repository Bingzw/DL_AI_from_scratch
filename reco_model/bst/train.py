import torch
import os
# lighting
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# model
from reco_model.bst.bst import BSTModule
from reco_model.movie_rating_seq import MovieRatingSeqDataModule


class EarlyStoppingOnRMSEDifference(pl.Callback):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def on_validation_end(self, model_trainer, pl_module):
        metrics = model_trainer.callback_metrics
        if 'train_rmse' in metrics and 'val_rmse' in metrics:
            rmse_diff = metrics['val_rmse'] - metrics['train_rmse']
            if rmse_diff > self.threshold:
                model_trainer.should_stop = True


if __name__ == "__main__":
    seed = 42
    CHECKPOINT_PATH = "../../saved_models/reco_models"
    save_directory_name = "bstnet"
    pretrained_model_name = ".ckpt"  # the correct path should be
    # pretrained_model_name = "/lightning_logs/version_0/checkpoints/epoch=77-step=36582.ckpt"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # data cleaning
    user_path = "../../data/ml-1m/users.dat"
    movie_path = "../../data/ml-1m/movies.dat"
    rating_path = "../../data/ml-1m/ratings.dat"

    # data loading parameters
    sequence_length = 8
    step_size = 1
    batch_size = 1024

    movie_seq_data = MovieRatingSeqDataModule(user_path, movie_path, rating_path, sequence_length, step_size,
                                              batch_size)
    movie_seq_data.setup()
    train_loader = movie_seq_data.train_dataloader()
    val_loader = movie_seq_data.val_dataloader()
    test_loader = movie_seq_data.test_dataloader()

    # model hyperparameters
    num_epochs = 100
    sparse_cardinality = movie_seq_data.dataset.sparse_cardinality
    mlp_dims = [4]
    movie_hidden_dim_per_head = 2
    num_head = 3
    dropout_rate = 0.3
    lr = 0.00005

    # define trainer
    checkpoint_callback = ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_rmse")
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_directory_name),  # Where to save models
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         # We run on a GPU (if possible)
                         devices=1,  # How many GPUs/CPUs we want to use (1 is enough for the notebooks)
                         max_epochs=num_epochs,  # How many epochs to train for if no patience is set
                         callbacks=[
                             EarlyStoppingOnRMSEDifference(threshold=0.5),
                             checkpoint_callback,
                             # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                             LearningRateMonitor("epoch")],
                         enable_progress_bar=True)  # Set to False if you do not want a progress bar

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_directory_name + pretrained_model_name)
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = BSTModule.load_from_checkpoint(
            pretrained_filename)  # Automatically loads the model with the saved hyperparameters
    else:
        pl.seed_everything(seed)  # To be reproducable
        model = BSTModule(sparse_cardinality=sparse_cardinality, sequence_length=sequence_length, mlp_dims=mlp_dims,
                          movie_hidden_dim_per_head=movie_hidden_dim_per_head, num_head=num_head,
                          dropout_rate=dropout_rate, lr=lr)
        trainer.fit(model, train_loader, val_loader)
        model = BSTModule.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training

    # Test best model on validation and test set
    test_result = trainer.test(model, test_loader, verbose=False)
    print("Best model checkpoint path:", trainer.checkpoint_callback.best_model_path)
    print("test_result: ", test_result[0])

    # run this in terminal to check logs: tensorboard --logdir saved_models/reco_models/bstnet/lightning_logs
