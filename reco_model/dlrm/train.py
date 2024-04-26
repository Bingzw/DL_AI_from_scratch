import torch
import os
# lighting
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# ray
from ray import train, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# model
from reco_model.criteo_data import CriteoDataModule
from dlrmnet import DLRMModule


def train_dlrm(config):
    # create data module
    dlrm_data = CriteoDataModule(data_path, batch_size=config["batch_size"], hidden_dim=config["hidden_dim"])
    num_dense_features = dlrm_data.dataset.dense_features.shape[1]
    embedding_sizes = dlrm_data.embedding_sizes

    dlrm_data.setup()
    train_loader = dlrm_data.train_dataloader()
    val_loader = dlrm_data.val_dataloader()
    test_loader = dlrm_data.test_dataloader()

    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_directory_name),  # Where to save models
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         # We run on a GPU (if possible)
                         devices=1,  # How many GPUs/CPUs we want to use
                         max_epochs=config["num_epochs"],  # How many epochs to train for if no patience is set
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_auc"),
                                    # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                    LearningRateMonitor("epoch")],
                         enable_progress_bar=True)

    model = DLRMModule(config=config, embedding_sizes=embedding_sizes, num_dense_features=num_dense_features)
    trainer.fit(model, train_loader, val_loader)
    model = DLRMModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training
    # Test best model on validation and test set
    test_result = trainer.test(model, test_loader, verbose=False)
    print("Best model checkpoint path:", trainer.checkpoint_callback.best_model_path)
    train.report({"test_auc": test_result[0]['test_auc']})


if __name__ == "__main__":
    # Set seedtune
    SEED = 42
    CHECKPOINT_PATH = "/Users/bwang7/ebay/bing_github/DL_genAI_practicing/saved_models/reco_models"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    data_path = "/Users/bwang7/ebay/bing_github/DL_genAI_practicing/data/reco_data/sampled_criteo_data.txt"
    save_directory_name = "dlrmnet"
    pl.seed_everything(SEED)  # To be reproducable
    # set hyperparameters
    hyper_config = {
        "batch_size": tune.choice([64, 128, 256, 512]),
        "hidden_dim": tune.choice([4, 8, 16]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "num_epochs": tune.choice([100, 200, 300]),
        "bottom_mlp_dims": tune.choice([[64, 32], [32, 16], [64, 32, 16], [32, 16, 8]]),
        "top_mlp_dims": tune.choice([[64, 32], [32, 16], [64, 32, 16], [32, 16, 8]])
    }

    # Define scheduler and reporter
    """
    In each iteration of tune.run, a single set of hyperparameters is sampled and a trial is run with these 
    hyperparameters. The performance of this trial is then reported back to the scheduler.  
    
    The scheduler keeps track of all trials that have been run so far, along with their performances. Based on these 
    performances, the scheduler decides which trials to stop and which ones to continue. In the case of ASHAScheduler, 
    it implements the Asynchronous Successive Halving Algorithm, which is a bandit-based approach to hyperparameter 
    optimization.  
    
    In the first few iterations, all trials are allowed to run for a minimum number of epochs 
    (specified by grace_period). After this, in each successive round, the scheduler keeps only the top 
    1/reduction_factor fraction of trials and prunes the rest. The surviving trials are allowed to run for more epochs. 
    This process is repeated until only one trial remains or some stopping condition is met.  
    
    So, even though only one hyperparameter set is sampled in each iteration, the scheduler makes its decisions based 
    on the performance of all trials that have been run so far.
    """
    scheduler = ASHAScheduler(metric="test_auc", mode="max",
                              max_t=10,  # the maximum time units to be run (epochs in this case)
                              grace_period=1,  # the minimum time units to be run (epochs in this case)
                              reduction_factor=2)  # the halving rate, each round, only the top 1/reduction_factor runs
    # are considered
    reporter = CLIReporter(
        metric_columns=["test_auc", "training_iteration"])

    result = tune.run(
        train_dlrm,
        resources_per_trial={"cpu": 1, "gpu": 0},
        config=hyper_config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter)

    # Get the best trial
    best_trial = result.get_best_trial("test_auc", "max", "last")
    # Get the best hyperparameters
    best_hyperparameters = best_trial.config
    # Print the best hyperparameters
    print("Best trial hyperparameters:", best_hyperparameters)
    # Get the path to the checkpoint of the best model
    best_model_checkpoint = best_trial.checkpoint.value
    # Get the performance of the best model
    best_model_performance = best_trial.last_result["test_auc"]
    # Print the path to the checkpoint and the performance of the best model
    print("Best model checkpoint path:", best_model_checkpoint)
    print("Best model performance (test_auc):", best_model_performance)
    # run this in terminal to check logs: tensorboard --logdir saved_models/reco_models/dlrmnet/lightning_logs



