import torch
import os
# lighting
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
# model
from reco_model.criteo_data import CriteoDataModule
from dlrmnet import DLRMModule


class MetricsCallback(pl.Callback):
    """
    stores the trainer.callback_metrics dictionary at the end of each validation epoch,
    wait for the validation epoch to complete before reporting the metrics
    """
    def __init__(self):
        super().__init__()
        self.metrics = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        self.metrics = trainer.callback_metrics


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


def train_dlrm_per_worker(config):
    # create data module
    dlrm_data = CriteoDataModule(data_path, batch_size=config["batch_size"], hidden_dim=config["hidden_dim"])
    num_dense_features = dlrm_data.dataset.dense_features.shape[1]
    embedding_sizes = dlrm_data.embedding_sizes

    dlrm_data.setup()
    train_loader = dlrm_data.train_dataloader()
    val_loader = dlrm_data.val_dataloader()
    test_loader = dlrm_data.test_dataloader()

    metrics_callback = MetricsCallback()
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_directory_name),  # Where to save models
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         # We run on a GPU (if possible)
                         devices=1,  # How many GPUs/CPUs we want to use
                         max_epochs=config["max_epochs"],  # How many epochs to train for if no patience is set,
                         callbacks=[metrics_callback,
                                    EarlyStoppingOnAucDifference(threshold=0.05),
                                    ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_auc", save_top_k=1),
                                    # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                    LearningRateMonitor("epoch")],
                         enable_progress_bar=True)

    model = DLRMModule(config=config, embedding_sizes=embedding_sizes, num_dense_features=num_dense_features)
    trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    test_result = trainer.test(model, test_loader, verbose=False)
    print("Best model checkpoint path:", trainer.checkpoint_callback.best_model_path)
    best_model_directory = os.path.dirname(trainer.checkpoint_callback.best_model_path)
    checkpoint = Checkpoint(path=best_model_directory)  #
    train.report(metrics={"train_auc": metrics_callback.metrics['train_auc'].item(),
                          "val_auc": metrics_callback.metrics['val_auc'].item(),
                          "test_auc": test_result[0]['test_auc'],
                          "best_checkpoint": trainer.checkpoint_callback.best_model_path},
                 checkpoint=checkpoint)
    # note that the model save path is specified as default_root_dir in the pl.Trainer and I also set the best model
    # directory as the checkpoint and pass this to train.report. This would also save the models in the best model
    # directory to the checkpoint path specified by ray results. However, we would not able to know which step is
    # optimal since ray only knows the best hyperparameter but does not know which step version is the optimal for the
    # given hyperparameter. So I save the best model path in the metrics to fast load the best model later


if __name__ == "__main__":
    # Set seed
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
        "batch_size": tune.choice([64, 128]),
        "hidden_dim": tune.choice([4, 6]),
        "lr": tune.loguniform(1e-5, 1e-3),
        "max_epochs": tune.choice([100, 200]),
        "bottom_mlp_dims": tune.choice([[16, 8], [8, 8, 4]]),
        "top_mlp_dims": tune.choice([[16, 8], [8, 8, 4]]),
        "dropout_rate": tune.uniform(0.3, 0.8),
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
    scheduler = ASHAScheduler(max_t=10,  # the maximum time units to be run (epochs in this case)
                              grace_period=1,  # the minimum time units to be run (epochs in this case)
                              reduction_factor=2)  # the halving rate, each round, only the top 1/reduction_factor runs

    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=False, resources_per_worker={"CPU": 1, "GPU": 0}
    )

    ray_trainer = TorchTrainer(
        train_dlrm_per_worker,
        scaling_config=scaling_config
    )

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": hyper_config},
        tune_config=tune.TuneConfig(
            metric="val_auc",
            mode="max",
            num_samples=10,
            scheduler=scheduler,
        ),
    )

    # Save the mapping between the hyperparameters and the best checkpoint
    result = tuner.fit()
    # Get the best trial
    best_result = result.get_best_result("val_auc", "max", "all")
    print("best_trial: ", best_result)
    # Get the best hyperparameters
    best_hyperparameters = best_result.config
    # Print the best hyperparameters
    print("Best trial hyperparameters:", best_hyperparameters)
    # get the best performance
    best_performance = best_result.metrics
    print("best_performance: ", best_performance)

    # run this in terminal to check logs: tensorboard --logdir saved_models/reco_models/dlrmnet/lightning_logs
    # the best test AUC so far is 0.73



