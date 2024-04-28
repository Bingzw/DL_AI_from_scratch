import torch
import os
# lighting
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# model
from reco_model.census_income import CensusDataModule
from reco_model.multitask.mmoe import MMOEModule


if __name__ == "__main__":
    # Set seed
    SEED = 42
    CHECKPOINT_PATH = "../../saved_models/reco_models"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    raw_train_data_path = "../../data/reco_data/census-income.data.gz"
    raw_test_data_path = "../../data/reco_data/census-income.test.gz"
    save_directory_name = "mmoenet"
    pretrained_model_name = ".ckpt"  # the correct path should be
    # pretrained_model_name = "/lightning_logs/version_0/checkpoints/epoch=77-step=36582.ckpt"
    # set hyperparameters
    batch_size = 1024
    hidden_dim = 10
    learning_rate = 0.0001
    num_epochs = 10
    num_tasks = 2
    num_experts = 3
    dropout = 0.1
    expert_mlp_dims = [64, 32]
    tower_mlp_dims = [64, 32]

    # create data module
    mmoe_data = CensusDataModule(train_path=raw_train_data_path, test_path=raw_test_data_path, batch_size=batch_size)
    num_dense_features = mmoe_data.dataset.dense_features.shape[1]
    sparse_cardinality = mmoe_data.sparse_cardinality
    # create data loader
    mmoe_data.setup()
    train_loader = mmoe_data.train_dataloader()
    val_loader = mmoe_data.val_dataloader()
    test_loader = mmoe_data.test_dataloader()

    # define trainer
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_directory_name),  # Where to save models
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         # We run on a GPU (if possible)
                         devices=1,  # How many GPUs/CPUs we want to use (1 is enough for the notebooks)
                         max_epochs=num_epochs,  # How many epochs to train for if no patience is set
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_aggregated_auc"),
                                    # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                    LearningRateMonitor("epoch")],
                         enable_progress_bar=True)  # Set to False if you do not want a progress bar

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_directory_name + pretrained_model_name)
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = MMOEModule.load_from_checkpoint(pretrained_filename)  # Automatically loads the model with the saved hyperparameters
    else:
        pl.seed_everything(SEED)  # To be reproducable
        model = MMOEModule(num_dense_features=num_dense_features,
                           sparse_cardinality=sparse_cardinality,
                           hidden_dim=hidden_dim,
                           expert_mlp_dims=expert_mlp_dims,
                           tower_mlp_dims=tower_mlp_dims,
                           num_tasks=num_tasks,
                           num_experts=num_experts,
                           dropout=dropout,
                           lr=learning_rate)
        trainer.fit(model, train_loader, val_loader)
        model = MMOEModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        # Load best checkpoint after training

    # Test best model on validation and test set
    test_result = trainer.test(model, test_loader, verbose=False)
    print("Best model checkpoint path:", trainer.checkpoint_callback.best_model_path)
    print("test_result: ", test_result[0])
    # without much tuning, the auc for task income is 0.91 and for task education is 0.99
